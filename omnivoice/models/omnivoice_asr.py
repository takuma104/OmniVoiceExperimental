#!/usr/bin/env python3
"""Speech recognition wrapper for OmniVoice.

This module reuses OmniVoice's audio embeddings and Qwen-initialized LLM body,
then adds a Qwen text head for autoregressive transcript generation.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

logger = logging.getLogger(__name__)


@dataclass
class OmniVoiceASROutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


class OmniVoiceForSpeechRecognition(PreTrainedModel):
    """Autoregressive ASR wrapper around ``OmniVoice``.

    Input layout is a mixed text/audio sequence. Audio positions use
    OmniVoice's summed codebook embeddings, while text positions use the
    underlying Qwen embedding table. The loss is computed only on text labels.
    """

    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    config_class = OmniVoiceConfig

    def __init__(
        self,
        config: OmniVoiceConfig,
        omnivoice: Optional[OmniVoice] = None,
    ):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        self.omnivoice = omnivoice if omnivoice is not None else OmniVoice(config)
        hidden_size = self.config.llm_config.hidden_size
        vocab_size = self.config.llm_config.vocab_size
        self.text_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def llm(self):
        """Expose the LLM body for the existing trainer optimizer helpers."""
        return self.omnivoice.llm

    @property
    def audio_embeddings(self):
        return self.omnivoice.audio_embeddings

    @property
    def audio_heads(self):
        return self.omnivoice.audio_heads

    def get_input_embeddings(self):
        return self.omnivoice.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.omnivoice.set_input_embeddings(value)

    @classmethod
    def from_omnivoice_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        base = OmniVoice.from_pretrained(
            pretrained_model_name_or_path,
            train=True,
            **kwargs,
        )
        return cls(config=base.config, omnivoice=base)

    def resize_text_vocab(self, vocab_size: int):
        """Resize LLM input embeddings and the ASR text head together."""
        old_head = self.text_head
        old_size, hidden_size = old_head.weight.shape

        if vocab_size == old_size:
            self.config.llm_config.vocab_size = vocab_size
            return

        self.llm.resize_token_embeddings(vocab_size)
        self.config.llm_config.vocab_size = vocab_size

        new_head = nn.Linear(hidden_size, vocab_size, bias=False)
        new_head = new_head.to(
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )
        with torch.no_grad():
            rows = min(old_size, vocab_size)
            new_head.weight[:rows].copy_(old_head.weight[:rows])
            if vocab_size > rows:
                input_weight = self.get_input_embeddings().weight
                copy_rows = min(vocab_size, input_weight.size(0))
                if copy_rows > rows and input_weight.size(1) == hidden_size:
                    new_head.weight[rows:copy_rows].copy_(input_weight[rows:copy_rows])
        self.text_head = new_head

    def init_text_head_from_qwen(self, qwen_model_name_or_path: str):
        """Initialize text head from Qwen CausalLM head or tied embeddings.

        Extra tokenizer rows added by OmniVoice/ASR special tokens are initialized
        from the current LLM input embeddings before Qwen rows are copied.
        """
        logger.info("Initializing ASR text head from %s", qwen_model_name_or_path)
        qwen = AutoModelForCausalLM.from_pretrained(
            qwen_model_name_or_path,
            dtype=torch.float32,
        )
        output_embeddings = qwen.get_output_embeddings()
        source_weight = (
            output_embeddings.weight
            if output_embeddings is not None
            else qwen.get_input_embeddings().weight
        )

        target_weight = self.text_head.weight
        input_weight = self.get_input_embeddings().weight
        if source_weight.size(1) != target_weight.size(1):
            raise ValueError(
                "Qwen text head hidden size mismatch: "
                f"{source_weight.size(1)} vs {target_weight.size(1)}"
            )

        with torch.no_grad():
            copy_rows = min(target_weight.size(0), input_weight.size(0))
            if input_weight.size(1) == target_weight.size(1):
                target_weight[:copy_rows].copy_(
                    input_weight[:copy_rows].to(
                        device=target_weight.device,
                        dtype=target_weight.dtype,
                    )
                )

            rows = min(target_weight.size(0), source_weight.size(0))
            target_weight[:rows].copy_(
                source_weight[:rows].to(
                    device=target_weight.device,
                    dtype=target_weight.dtype,
                )
            )

        del qwen
        logger.info("ASR text head initialized.")

    def set_trainable_modules(
        self,
        train_llm_body: bool = True,
        freeze_text_embedding: bool = True,
        freeze_text_head: bool = True,
        freeze_audio_embeddings: bool = True,
    ):
        for p in self.parameters():
            p.requires_grad = False

        if train_llm_body:
            for p in self.llm.parameters():
                p.requires_grad = True

        if freeze_text_embedding:
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = False

        if not freeze_text_head:
            for p in self.text_head.parameters():
                p.requires_grad = True

        if not freeze_audio_embeddings:
            for p in self.audio_embeddings.parameters():
                p.requires_grad = True

        # The TTS audio heads are not used by ASR.
        for p in self.audio_heads.parameters():
            p.requires_grad = False

    def _build_prefix_lm_mask(
        self,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        text_causal_mask: torch.Tensor,
        device: torch.device,
    ):
        return create_block_mask(
            _get_asr_prefix_lm_mask(
                document_ids[0].to(device),
                text_causal_mask[0].to(device),
            ),
            B=None,
            H=None,
            Q_LEN=input_ids.size(-1),
            KV_LEN=input_ids.size(-1),
            _compile=True,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        text_causal_mask: Optional[torch.Tensor] = None,
    ):
        inputs_embeds = self.omnivoice._prepare_embed_inputs(input_ids, audio_mask)

        if (
            attention_mask is None
            and document_ids is not None
            and text_causal_mask is not None
        ):
            attention_mask = self._build_prefix_lm_mask(
                input_ids=input_ids,
                document_ids=document_ids,
                text_causal_mask=text_causal_mask,
                device=inputs_embeds.device,
            )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=position_ids,
        )
        hidden_states = llm_outputs[0]
        logits = self.text_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return OmniVoiceASROutput(loss=loss, logits=logits)

    @torch.inference_mode()
    def generate_text(
        self,
        audio_tokens: torch.Tensor,
        tokenizer: AutoTokenizer,
        language: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Simple non-KV cached greedy/sampling ASR generation helper."""
        device = next(self.parameters()).device
        if audio_tokens.dim() == 3:
            audio_tokens = audio_tokens.squeeze(0)
        audio_tokens = audio_tokens.to(device=device, dtype=torch.long)

        style = "<|asr|>"
        if language is not None:
            style += f"<|lang_start|>{language}<|lang_end|>"
        style_ids = tokenizer(style, return_tensors="pt").input_ids.to(device)
        text_start_ids = tokenizer("<|text_start|>", return_tensors="pt").input_ids.to(
            device
        )
        eos_ids = tokenizer("<|text_end|>", return_tensors="pt").input_ids.to(device)
        eos_id = eos_ids[0, -1].item()

        c = self.config.num_audio_codebook
        style_inputs = style_ids.repeat(c, 1)
        text_inputs = text_start_ids.repeat(c, 1)
        input_ids = torch.cat([style_inputs, audio_tokens, text_inputs], dim=1)
        audio_mask = torch.zeros(input_ids.size(1), dtype=torch.bool, device=device)
        audio_start = style_inputs.size(1)
        audio_end = audio_start + audio_tokens.size(1)
        audio_mask[audio_start:audio_end] = True
        text_causal_mask = torch.zeros_like(audio_mask)
        text_causal_mask[-text_inputs.size(1) :] = True

        generated = []
        for _ in range(max_new_tokens):
            batch = {
                "input_ids": input_ids.unsqueeze(0),
                "audio_mask": audio_mask.unsqueeze(0),
                "text_causal_mask": text_causal_mask.unsqueeze(0),
                "document_ids": torch.zeros(
                    1, input_ids.size(1), dtype=torch.int32, device=device
                ),
                "position_ids": torch.arange(
                    input_ids.size(1), device=device
                ).unsqueeze(0),
            }
            logits = self(**batch).logits[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_id.item()
            if token_id == eos_id:
                break
            generated.append(token_id)

            next_col = next_id.repeat(c, 1)
            input_ids = torch.cat([input_ids, next_col], dim=1)
            audio_mask = torch.cat([audio_mask, audio_mask.new_zeros(1)])
            text_causal_mask = torch.cat(
                [text_causal_mask, text_causal_mask.new_ones(1)]
            )

        return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _get_asr_prefix_lm_mask(document_ids, text_causal_mask):
    return partial(_mask_mod_asr_prefix_lm, document_ids, text_causal_mask)


def _mask_mod_asr_prefix_lm(document_ids, text_causal_mask, b, h, q_idx, kv_idx):
    same_doc = document_ids[q_idx] == document_ids[kv_idx]
    valid_doc = document_ids[q_idx] >= 0
    kv_is_prefix = ~text_causal_mask[kv_idx]
    q_is_text = text_causal_mask[q_idx]
    causal_text = q_is_text & (q_idx >= kv_idx)
    return valid_doc & same_doc & (kv_is_prefix | causal_text)

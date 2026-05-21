#!/usr/bin/env python3
"""Speech recognition wrapper for OmniVoice.

This module reuses OmniVoice's audio embeddings and Qwen-initialized LLM body,
then adds a Qwen text head for autoregressive transcript generation.
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Sequence, Union

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
    text_loss: Optional[torch.Tensor] = None
    timestamp_loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    past_key_values: Optional[object] = None
    attentions: Optional[object] = None


class OmniVoiceForSpeechRecognition(PreTrainedModel):
    """Autoregressive ASR wrapper around ``OmniVoice``.

    Input layout is a mixed text/audio sequence. Audio positions use
    OmniVoice's summed codebook embeddings, while text positions use the
    underlying Qwen embedding table. The loss is computed only on text labels.
    """

    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(
        self,
        config: OmniVoiceConfig,
        omnivoice: Optional[OmniVoice] = None,
        audio_embedding_mode: Optional[str] = None,
        audio_adapter_hidden_size: Optional[int] = None,
        attention_mode: Optional[str] = None,
    ):
        super().__init__(config)
        self.all_tied_weights_keys = {}
        self.omnivoice = omnivoice if omnivoice is not None else OmniVoice(config)
        hidden_size = self.config.llm_config.hidden_size
        vocab_size = self.config.llm_config.vocab_size
        self.text_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.audio_embedding_mode = (
            audio_embedding_mode
            or getattr(self.config, "asr_audio_embedding_mode", "all_sum")
        )
        self.config.asr_audio_embedding_mode = self.audio_embedding_mode
        if audio_adapter_hidden_size is None:
            audio_adapter_hidden_size = getattr(
                self.config, "asr_audio_adapter_hidden_size", None
            )

        if self.audio_embedding_mode not in {
            "all_sum",
            "all_sum_adapter",
            "weighted_sum",
        }:
            raise ValueError(
                "Unsupported ASR audio embedding mode: "
                f"{self.audio_embedding_mode!r}"
            )

        if self.audio_embedding_mode == "all_sum_adapter":
            adapter_hidden = audio_adapter_hidden_size or hidden_size
            self.config.asr_audio_adapter_hidden_size = adapter_hidden
            self.audio_embedding_adapter = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, adapter_hidden),
                nn.GELU(),
                nn.Linear(adapter_hidden, hidden_size),
            )
        else:
            self.config.asr_audio_adapter_hidden_size = audio_adapter_hidden_size
            self.audio_embedding_adapter = None

        if self.audio_embedding_mode == "weighted_sum":
            self.audio_codebook_weights = nn.Parameter(
                torch.ones(self.config.num_audio_codebook)
            )
        else:
            self.audio_codebook_weights = None

        timestamp_head_dim = getattr(self.config, "asr_timestamp_head_dim", 256)
        self.config.asr_timestamp_head_dim = timestamp_head_dim
        self.timestamp_query_proj = nn.Linear(hidden_size, timestamp_head_dim, bias=False)
        self.timestamp_audio_key_proj = nn.Linear(
            hidden_size,
            timestamp_head_dim,
            bias=False,
        )

        self.asr_attention_mode = attention_mode or getattr(
            self.config, "asr_attention_mode", "prefix_lm"
        )
        if self.asr_attention_mode not in {"prefix_lm", "causal"}:
            raise ValueError(
                "Unsupported ASR attention mode: "
                f"{self.asr_attention_mode!r}"
            )
        self.config.asr_attention_mode = self.asr_attention_mode

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
        train_audio_embedding_adapter: bool = True,
        train_timestamp_head: bool = True,
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

        if train_audio_embedding_adapter:
            if self.audio_embedding_adapter is not None:
                for p in self.audio_embedding_adapter.parameters():
                    p.requires_grad = True
            if self.audio_codebook_weights is not None:
                self.audio_codebook_weights.requires_grad = True

        # The TTS audio heads are not used by ASR.
        for p in self.audio_heads.parameters():
            p.requires_grad = False

        if train_timestamp_head:
            for p in self.timestamp_query_proj.parameters():
                p.requires_grad = True
            for p in self.timestamp_audio_key_proj.parameters():
                p.requires_grad = True

    @classmethod
    def from_omnivoice_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        audio_embedding_mode: Optional[str] = None,
        audio_adapter_hidden_size: Optional[int] = None,
        attention_mode: Optional[str] = None,
        timestamp_head_dim: Optional[int] = None,
        **kwargs,
    ):
        base = OmniVoice.from_pretrained(
            pretrained_model_name_or_path,
            train=True,
            **kwargs,
        )
        if timestamp_head_dim is not None:
            base.config.asr_timestamp_head_dim = timestamp_head_dim
        return cls(
            config=base.config,
            omnivoice=base,
            audio_embedding_mode=audio_embedding_mode,
            audio_adapter_hidden_size=audio_adapter_hidden_size,
            attention_mode=attention_mode,
        )

    def _prepare_embed_inputs(
        self, input_ids: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        """Prepare mixed text/audio embeddings for ASR.

        ``all_sum`` matches OmniVoice's native audio embedding path. The
        adapter and weighted modes keep the codec embedding table shared and
        only change how per-codebook embeddings are combined.
        """
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])
        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1)
        ) + self.omnivoice.codebook_layer_offsets.view(1, -1, 1)
        per_codebook_embeds = self.audio_embeddings(shifted_ids)

        if self.audio_embedding_mode == "weighted_sum":
            weights = self.audio_codebook_weights.to(dtype=per_codebook_embeds.dtype)
            audio_embeds = (per_codebook_embeds * weights.view(1, -1, 1, 1)).sum(dim=1)
        else:
            audio_embeds = per_codebook_embeds.sum(dim=1)
            if self.audio_embedding_adapter is not None:
                audio_embeds = audio_embeds + self.audio_embedding_adapter(audio_embeds)

        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def _build_prefix_lm_mask(
        self,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        text_causal_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ):
        attn_implementation = getattr(self.llm.config, "_attn_implementation", None)
        if attn_implementation != "flex_attention":
            return self._build_dense_prefix_lm_mask(
                input_ids=input_ids,
                document_ids=document_ids,
                text_causal_mask=text_causal_mask,
                device=device,
                dtype=dtype,
                as_attention_bias=attn_implementation == "eager",
            )

        if self.asr_attention_mode == "causal":
            mask_mod = _get_asr_causal_mask(document_ids.to(device))
        else:
            mask_mod = _get_asr_prefix_lm_mask(
                document_ids.to(device),
                text_causal_mask.to(device),
            )

        return create_block_mask(
            mask_mod,
            B=input_ids.size(0),
            H=None,
            Q_LEN=input_ids.size(-1),
            KV_LEN=input_ids.size(-1),
            _compile=True,
            device=device,
        )

    def _build_dense_prefix_lm_mask(
        self,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        text_causal_mask: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        as_attention_bias: bool = False,
    ):
        document_ids = document_ids.to(device)
        text_causal_mask = text_causal_mask.to(device=device, dtype=torch.bool)
        if document_ids.ndim == 1:
            document_ids = document_ids.unsqueeze(0)
        if text_causal_mask.ndim == 1:
            text_causal_mask = text_causal_mask.unsqueeze(0)

        seq_len = input_ids.size(-1)
        q_idx = torch.arange(seq_len, device=device).view(1, seq_len, 1)
        kv_idx = torch.arange(seq_len, device=device).view(1, 1, seq_len)

        same_doc = document_ids[:, :, None] == document_ids[:, None, :]
        valid_doc = document_ids[:, :, None] >= 0
        if self.asr_attention_mode == "causal":
            attention_mask = valid_doc & same_doc & (q_idx >= kv_idx)
        else:
            kv_is_prefix = ~text_causal_mask[:, None, :]
            q_is_text = text_causal_mask[:, :, None]
            causal_text = q_is_text & (q_idx >= kv_idx)
            attention_mask = valid_doc & same_doc & (kv_is_prefix | causal_text)

        # SDPA can produce NaNs for padding query rows whose whole key row is
        # masked. Those rows are never read by the loss/generation code, but
        # keeping a harmless self-edge prevents backend-specific NaN propagation.
        padding_self = (document_ids[:, :, None] < 0) & (q_idx == kv_idx)
        attention_mask = attention_mask | padding_self
        attention_mask = attention_mask.unsqueeze(1)

        if as_attention_bias:
            min_dtype = torch.finfo(dtype).min
            return torch.where(
                attention_mask,
                torch.tensor(0.0, dtype=dtype, device=device),
                torch.tensor(min_dtype, dtype=dtype, device=device),
            )

        return attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        audio_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        text_causal_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        timestamp_center_labels: Optional[torch.LongTensor] = None,
        return_hidden_states: bool = False,
    ):
        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)

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
                dtype=inputs_embeds.dtype,
            )

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = llm_outputs[0]
        logits = self.text_head(hidden_states)

        text_loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            text_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        timestamp_loss = None
        if timestamp_center_labels is not None:
            timestamp_loss = self._compute_timestamp_center_loss(
                hidden_states=hidden_states,
                timestamp_center_labels=timestamp_center_labels,
                audio_mask=audio_mask,
                document_ids=document_ids,
            )

        loss = text_loss
        if timestamp_loss is not None:
            weight = getattr(self.config, "asr_timestamp_loss_weight", 1.0)
            if loss is None:
                loss = timestamp_loss * weight
            else:
                loss = loss + timestamp_loss * weight

        return OmniVoiceASROutput(
            loss=loss,
            text_loss=text_loss,
            timestamp_loss=timestamp_loss,
            logits=logits,
            hidden_states=hidden_states if return_hidden_states else None,
            past_key_values=getattr(llm_outputs, "past_key_values", None),
            attentions=getattr(llm_outputs, "attentions", None),
        )

    def _compute_timestamp_center_loss(
        self,
        hidden_states: torch.Tensor,
        timestamp_center_labels: torch.Tensor,
        audio_mask: torch.Tensor,
        document_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if document_ids is None:
            raise ValueError("timestamp_center_labels requires document_ids.")

        labels = timestamp_center_labels.to(device=hidden_states.device)
        audio_mask = audio_mask.to(device=hidden_states.device, dtype=torch.bool)
        document_ids = document_ids.to(device=hidden_states.device)
        valid_query_mask = labels >= 0
        if not torch.any(valid_query_mask):
            return (
                self.timestamp_query_proj.weight.sum()
                + self.timestamp_audio_key_proj.weight.sum()
            ) * 0.0

        losses = []
        scale = self.timestamp_query_proj.out_features**-0.5
        for batch_idx in range(hidden_states.size(0)):
            batch_doc_ids = document_ids[batch_idx]
            batch_labels = labels[batch_idx]
            batch_valid = valid_query_mask[batch_idx]
            doc_values = torch.unique(batch_doc_ids[batch_valid])
            for doc_id in doc_values.tolist():
                if doc_id < 0:
                    continue
                doc_mask = batch_doc_ids == doc_id
                audio_positions = torch.nonzero(
                    doc_mask & audio_mask[batch_idx],
                    as_tuple=False,
                ).squeeze(1)
                if audio_positions.numel() == 0:
                    continue

                query_positions = torch.nonzero(
                    doc_mask & batch_valid,
                    as_tuple=False,
                ).squeeze(1)
                targets = batch_labels[query_positions].long()
                target_mask = (targets >= 0) & (targets < audio_positions.numel())
                if not torch.any(target_mask):
                    continue

                query_positions = query_positions[target_mask]
                targets = targets[target_mask]
                query_states = hidden_states[batch_idx, query_positions]
                audio_states = hidden_states[batch_idx, audio_positions]
                query = self.timestamp_query_proj(query_states)
                keys = self.timestamp_audio_key_proj(audio_states)
                pointer_logits = torch.matmul(query, keys.transpose(0, 1)) * scale
                losses.append(F.cross_entropy(pointer_logits, targets))

        if not losses:
            return (
                self.timestamp_query_proj.weight.sum()
                + self.timestamp_audio_key_proj.weight.sum()
            ) * 0.0
        return torch.stack(losses).mean()

    @staticmethod
    def _resolve_attention_indices(
        selection: Optional[Union[str, int, Sequence[int]]],
        total: int,
        name: str,
    ) -> List[int]:
        if selection is None:
            return list(range(total))
        if isinstance(selection, str):
            value = selection.strip().lower()
            if value in {"", "all"}:
                return list(range(total))
            if value == "last":
                return [total - 1]
            raw_indices = [item.strip() for item in value.split(",") if item.strip()]
            indices = [int(item) for item in raw_indices]
        elif isinstance(selection, int):
            indices = [selection]
        else:
            indices = [int(item) for item in selection]

        resolved = []
        for index in indices:
            if index < 0:
                index += total
            if index < 0 or index >= total:
                raise ValueError(
                    f"{name} index {index} is out of range for {total} {name}s."
                )
            resolved.append(index)
        return resolved

    def _aggregate_audio_attention(
        self,
        attentions: object,
        query_index: int,
        audio_start: int,
        audio_end: int,
        layers: Optional[Union[str, int, Sequence[int]]],
        heads: Optional[Union[str, int, Sequence[int]]],
        batch_index: int = 0,
    ) -> torch.Tensor:
        _, per_layer = self._audio_attention_per_layer(
            attentions=attentions,
            query_index=query_index,
            audio_start=audio_start,
            audio_end=audio_end,
            layers=layers,
            heads=heads,
            batch_index=batch_index,
        )
        return self._combine_layer_audio_attention(per_layer)

    def _audio_attention_per_layer(
        self,
        attentions: object,
        query_index: int,
        audio_start: int,
        audio_end: int,
        layers: Optional[Union[str, int, Sequence[int]]],
        heads: Optional[Union[str, int, Sequence[int]]],
        batch_index: int = 0,
    ) -> tuple[List[int], List[torch.Tensor]]:
        if attentions is None:
            raise RuntimeError(
                "The LLM did not return attentions. Load the model with "
                'attn_implementation="eager" and call with output_attentions=True.'
            )

        layer_attentions = list(attentions)
        layer_indices = self._resolve_attention_indices(
            layers if layers is not None else "last",
            len(layer_attentions),
            "layer",
        )

        per_layer = []
        for layer_index in layer_indices:
            layer_attention = layer_attentions[layer_index]
            if layer_attention is None:
                raise RuntimeError(
                    f"Layer {layer_index} did not return attention weights."
                )
            num_heads = layer_attention.size(1)
            head_indices = self._resolve_attention_indices(heads, num_heads, "head")
            query_attention = layer_attention[
                batch_index,
                head_indices,
                query_index,
                audio_start:audio_end,
            ]
            scores = query_attention.float().mean(dim=0)
            total = scores.sum()
            if torch.isfinite(total) and total > 0:
                scores = scores / total
            per_layer.append(scores)
        return layer_indices, per_layer

    @staticmethod
    def _combine_layer_audio_attention(per_layer: Sequence[torch.Tensor]) -> torch.Tensor:
        scores = torch.stack(per_layer, dim=0).mean(dim=0)
        total = scores.sum()
        if torch.isfinite(total) and total > 0:
            scores = scores / total
        return scores.detach().cpu()

    @torch.inference_mode()
    def generate_text_attention_trace(
        self,
        audio_tokens: torch.Tensor,
        tokenizer: AutoTokenizer,
        language: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        layers: Optional[Union[str, int, Sequence[int]]] = "last",
        heads: Optional[Union[str, int, Sequence[int]]] = None,
        include_eos: bool = False,
        return_layer_attentions: bool = False,
    ) -> dict[str, Any]:
        """Generate one transcript and collect audio-prefix attention per step.

        Each attention row is the self-attention of the query position whose
        hidden state predicts the recorded output token. For the first emitted
        text token, that query is the final ``<|text_start|>`` position.
        """
        samples = self._normalize_generate_audio_batch(
            audio_tokens=audio_tokens,
            audio_lengths=None,
            num_codebooks=self.config.num_audio_codebook,
        )
        if len(samples) != 1:
            raise ValueError("generate_text_attention_trace() expects one sample.")

        device = next(self.parameters()).device
        c = self.config.num_audio_codebook
        sample = samples[0].to(device=device, dtype=torch.long)

        text_start_ids = tokenizer("<|text_start|>", return_tensors="pt").input_ids.to(
            device
        )
        eos_ids = tokenizer("<|text_end|>", return_tensors="pt").input_ids.to(device)
        eos_id = eos_ids[0, -1].item()

        style = "<|asr|>"
        if language is not None:
            style += f"<|lang_start|>{language}<|lang_end|>"
        style_ids = tokenizer(style, return_tensors="pt").input_ids.to(device)

        style_inputs = style_ids.repeat(c, 1)
        text_inputs = text_start_ids.repeat(c, 1)
        sample_input_ids = torch.cat([style_inputs, sample, text_inputs], dim=1)
        seq_len = sample_input_ids.size(1)

        input_ids = sample_input_ids.unsqueeze(0)
        audio_mask = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
        audio_start = style_inputs.size(1)
        audio_end = audio_start + sample.size(1)
        audio_mask[:, audio_start:audio_end] = True

        text_causal_mask = torch.zeros_like(audio_mask)
        text_causal_mask[:, -text_inputs.size(1) :] = True
        document_ids = torch.zeros(1, seq_len, dtype=torch.int32, device=device)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        outputs = self(
            input_ids=input_ids,
            audio_mask=audio_mask,
            text_causal_mask=text_causal_mask,
            document_ids=document_ids,
            position_ids=position_ids,
            use_cache=True,
            output_attentions=True,
        )
        past_key_values = outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError("LLM did not return past_key_values.")

        logits = outputs.logits[:, -1, :]
        current_attentions = outputs.attentions
        current_query_index = -1
        current_query_id = int(input_ids[0, 0, -1].item())

        generated: List[int] = []
        token_ids: List[int] = []
        token_texts: List[str] = []
        query_token_ids: List[int] = []
        query_token_texts: List[str] = []
        audio_attention_rows: List[torch.Tensor] = []
        layer_audio_attention_rows: dict[int, List[torch.Tensor]] = {}
        prompt_valid_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
        lengths_tensor = torch.tensor([seq_len], dtype=torch.long, device=device)

        for step in range(max_new_tokens):
            next_id = self._sample_next_text_ids(logits, temperature)
            token_id = int(next_id.item())

            if token_id != eos_id or include_eos:
                if return_layer_attentions:
                    layer_indices, per_layer = self._audio_attention_per_layer(
                        attentions=current_attentions,
                        query_index=current_query_index,
                        audio_start=audio_start,
                        audio_end=audio_end,
                        layers=layers,
                        heads=heads,
                    )
                    audio_attention_rows.append(
                        self._combine_layer_audio_attention(per_layer)
                    )
                    for layer_index, layer_scores in zip(layer_indices, per_layer):
                        layer_audio_attention_rows.setdefault(layer_index, []).append(
                            layer_scores.detach().cpu()
                        )
                else:
                    audio_attention_rows.append(
                        self._aggregate_audio_attention(
                            attentions=current_attentions,
                            query_index=current_query_index,
                            audio_start=audio_start,
                            audio_end=audio_end,
                            layers=layers,
                            heads=heads,
                        )
                    )
                token_ids.append(token_id)
                token_texts.append(
                    tokenizer.decode([token_id], skip_special_tokens=False)
                )
                query_token_ids.append(current_query_id)
                query_token_texts.append(
                    tokenizer.decode([current_query_id], skip_special_tokens=False)
                )

            if token_id == eos_id:
                break

            generated.append(token_id)
            next_col = next_id.view(1, 1, 1).expand(-1, c, -1)
            next_audio_mask = audio_mask.new_zeros(1, 1)
            next_inputs_embeds = self._prepare_embed_inputs(next_col, next_audio_mask)
            generated_mask = torch.ones(
                1,
                step + 1,
                dtype=torch.bool,
                device=device,
            )
            decode_attention_mask = torch.cat(
                [prompt_valid_mask, generated_mask],
                dim=1,
            )
            next_position_ids = (lengths_tensor + step).unsqueeze(1)

            decode_outputs = self.llm(
                inputs_embeds=next_inputs_embeds,
                attention_mask=decode_attention_mask,
                return_dict=True,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            past_key_values = decode_outputs.past_key_values
            current_attentions = getattr(decode_outputs, "attentions", None)
            current_query_index = -1
            current_query_id = token_id
            logits = self.text_head(decode_outputs[0])[:, -1, :]

        if audio_attention_rows:
            audio_attention = torch.stack(audio_attention_rows, dim=0)
        else:
            audio_attention = torch.empty(0, sample.size(1), dtype=torch.float32)

        layer_audio_attentions = None
        if return_layer_attentions:
            layer_audio_attentions = {
                layer_index: torch.stack(rows, dim=0)
                if rows
                else torch.empty(0, sample.size(1), dtype=torch.float32)
                for layer_index, rows in layer_audio_attention_rows.items()
            }

        return {
            "text": tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "token_ids": token_ids,
            "token_texts": token_texts,
            "query_token_ids": query_token_ids,
            "query_token_texts": query_token_texts,
            "audio_attention": audio_attention,
            "layer_audio_attentions": layer_audio_attentions,
            "audio_start": audio_start,
            "audio_end": audio_end,
            "audio_num_tokens": sample.size(1),
            "layers": layers,
            "heads": heads,
        }

    @torch.inference_mode()
    def generate_text_attention_trace_batch(
        self,
        audio_tokens: Union[torch.Tensor, Sequence[torch.Tensor]],
        tokenizer: AutoTokenizer,
        languages: Optional[Sequence[Optional[str]]] = None,
        audio_lengths: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        layers: Optional[Union[str, int, Sequence[int]]] = "last",
        heads: Optional[Union[str, int, Sequence[int]]] = None,
        include_eos: bool = False,
        return_layer_attentions: bool = False,
    ) -> List[dict[str, Any]]:
        """Batched ASR generation with per-token audio-prefix attention traces."""
        device = next(self.parameters()).device
        c = self.config.num_audio_codebook
        samples = self._normalize_generate_audio_batch(
            audio_tokens=audio_tokens,
            audio_lengths=audio_lengths,
            num_codebooks=c,
        )
        if not samples:
            return []

        batch_size = len(samples)
        if languages is None:
            languages = [None] * batch_size
        elif isinstance(languages, str):
            languages = [languages] * batch_size
        elif len(languages) != batch_size:
            raise ValueError(
                f"Expected {batch_size} language entries, got {len(languages)}."
            )

        text_start_ids = tokenizer("<|text_start|>", return_tensors="pt").input_ids.to(
            device
        )
        eos_ids = tokenizer("<|text_end|>", return_tensors="pt").input_ids.to(device)
        eos_id = eos_ids[0, -1].item()

        sample_inputs = []
        sample_audio_masks = []
        sample_text_causal_masks = []
        lengths = []
        audio_starts = []
        audio_ends = []
        audio_lengths_out = []
        for sample, language in zip(samples, languages):
            sample = sample.to(device=device, dtype=torch.long)
            style = "<|asr|>"
            if language is not None:
                style += f"<|lang_start|>{language}<|lang_end|>"
            style_ids = tokenizer(style, return_tensors="pt").input_ids.to(device)

            style_inputs = style_ids.repeat(c, 1)
            text_inputs = text_start_ids.repeat(c, 1)
            sample_input_ids = torch.cat([style_inputs, sample, text_inputs], dim=1)

            audio_mask = torch.zeros(
                sample_input_ids.size(1), dtype=torch.bool, device=device
            )
            audio_start = style_inputs.size(1)
            audio_end = audio_start + sample.size(1)
            audio_mask[audio_start:audio_end] = True

            text_causal_mask = torch.zeros_like(audio_mask)
            text_causal_mask[-text_inputs.size(1) :] = True

            sample_inputs.append(sample_input_ids)
            sample_audio_masks.append(audio_mask)
            sample_text_causal_masks.append(text_causal_mask)
            lengths.append(sample_input_ids.size(1))
            audio_starts.append(audio_start)
            audio_ends.append(audio_end)
            audio_lengths_out.append(sample.size(1))

        max_len = max(lengths)
        input_ids = torch.zeros(
            batch_size,
            c,
            max_len,
            dtype=torch.long,
            device=device,
        )
        audio_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
            device=device,
        )
        text_causal_mask = torch.zeros_like(audio_mask)
        document_ids = torch.full(
            (batch_size, max_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        position_ids = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.long,
            device=device,
        )

        for idx, (
            sample_input_ids,
            sample_audio_mask,
            sample_text_causal_mask,
        ) in enumerate(zip(sample_inputs, sample_audio_masks, sample_text_causal_masks)):
            length = sample_input_ids.size(1)
            input_ids[idx, :, :length] = sample_input_ids
            audio_mask[idx, :length] = sample_audio_mask
            text_causal_mask[idx, :length] = sample_text_causal_mask
            document_ids[idx, :length] = 0
            position_ids[idx, :length] = torch.arange(length, device=device)

        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
        prompt_valid_mask = document_ids >= 0
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = lengths_tensor - 1
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        generated: List[List[int]] = [[] for _ in range(batch_size)]
        token_ids: List[List[int]] = [[] for _ in range(batch_size)]
        token_texts: List[List[str]] = [[] for _ in range(batch_size)]
        query_token_ids: List[List[int]] = [[] for _ in range(batch_size)]
        query_token_texts: List[List[str]] = [[] for _ in range(batch_size)]
        audio_attention_rows: List[List[torch.Tensor]] = [
            [] for _ in range(batch_size)
        ]
        layer_audio_attention_rows: List[dict[int, List[torch.Tensor]]] = [
            {} for _ in range(batch_size)
        ]

        outputs = self(
            input_ids=input_ids,
            audio_mask=audio_mask,
            text_causal_mask=text_causal_mask,
            document_ids=document_ids,
            position_ids=position_ids,
            use_cache=True,
            output_attentions=True,
        )
        past_key_values = outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError("LLM did not return past_key_values.")

        logits = outputs.logits[batch_indices, last_indices, :]
        current_attentions = outputs.attentions
        current_query_indices = last_indices.tolist()
        current_query_ids = [
            int(input_ids[idx, 0, current_query_indices[idx]].item())
            for idx in range(batch_size)
        ]

        for step in range(max_new_tokens):
            next_id = self._sample_next_text_ids(logits, temperature)
            next_id = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_id, eos_id),
                next_id,
            )
            next_ids = next_id.squeeze(1)
            newly_finished = next_ids == eos_id

            for idx, token_id in enumerate(next_ids.tolist()):
                if bool(finished[idx].item()):
                    continue

                if token_id != eos_id or include_eos:
                    if return_layer_attentions:
                        layer_indices, per_layer = self._audio_attention_per_layer(
                            attentions=current_attentions,
                            query_index=current_query_indices[idx],
                            audio_start=audio_starts[idx],
                            audio_end=audio_ends[idx],
                            layers=layers,
                            heads=heads,
                            batch_index=idx,
                        )
                        audio_attention_rows[idx].append(
                            self._combine_layer_audio_attention(per_layer)
                        )
                        for layer_index, layer_scores in zip(layer_indices, per_layer):
                            layer_audio_attention_rows[idx].setdefault(
                                layer_index, []
                            ).append(layer_scores.detach().cpu())
                    else:
                        audio_attention_rows[idx].append(
                            self._aggregate_audio_attention(
                                attentions=current_attentions,
                                query_index=current_query_indices[idx],
                                audio_start=audio_starts[idx],
                                audio_end=audio_ends[idx],
                                layers=layers,
                                heads=heads,
                                batch_index=idx,
                            )
                        )

                    token_ids[idx].append(token_id)
                    token_texts[idx].append(
                        tokenizer.decode([token_id], skip_special_tokens=False)
                    )
                    query_id = current_query_ids[idx]
                    query_token_ids[idx].append(query_id)
                    query_token_texts[idx].append(
                        tokenizer.decode([query_id], skip_special_tokens=False)
                    )

                if token_id != eos_id:
                    generated[idx].append(token_id)

            finished = finished | newly_finished
            if finished.all():
                break

            next_col = next_id.view(batch_size, 1, 1).expand(-1, c, -1)
            next_audio_mask = audio_mask.new_zeros(batch_size, 1)
            next_inputs_embeds = self._prepare_embed_inputs(
                next_col,
                next_audio_mask,
            )
            generated_mask = torch.ones(
                batch_size,
                step + 1,
                dtype=torch.bool,
                device=device,
            )
            decode_attention_mask = torch.cat(
                [prompt_valid_mask, generated_mask],
                dim=1,
            )
            next_position_ids = (lengths_tensor + step).unsqueeze(1)

            decode_outputs = self.llm(
                inputs_embeds=next_inputs_embeds,
                attention_mask=decode_attention_mask,
                return_dict=True,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            past_key_values = decode_outputs.past_key_values
            current_attentions = getattr(decode_outputs, "attentions", None)
            current_query_indices = [-1] * batch_size
            current_query_ids = [int(token_id) for token_id in next_ids.tolist()]
            logits = self.text_head(decode_outputs[0])[:, -1, :]

        traces = []
        for idx in range(batch_size):
            if audio_attention_rows[idx]:
                audio_attention = torch.stack(audio_attention_rows[idx], dim=0)
            else:
                audio_attention = torch.empty(
                    0,
                    audio_lengths_out[idx],
                    dtype=torch.float32,
                )

            layer_audio_attentions = None
            if return_layer_attentions:
                layer_audio_attentions = {
                    layer_index: torch.stack(rows, dim=0)
                    if rows
                    else torch.empty(
                        0,
                        audio_lengths_out[idx],
                        dtype=torch.float32,
                    )
                    for layer_index, rows in layer_audio_attention_rows[idx].items()
                }

            traces.append(
                {
                    "text": tokenizer.decode(
                        generated[idx],
                        skip_special_tokens=True,
                    ).strip(),
                    "token_ids": token_ids[idx],
                    "token_texts": token_texts[idx],
                    "query_token_ids": query_token_ids[idx],
                    "query_token_texts": query_token_texts[idx],
                    "audio_attention": audio_attention,
                    "layer_audio_attentions": layer_audio_attentions,
                    "audio_start": audio_starts[idx],
                    "audio_end": audio_ends[idx],
                    "audio_num_tokens": audio_lengths_out[idx],
                    "layers": layers,
                    "heads": heads,
                }
            )
        return traces

    @torch.inference_mode()
    def generate_text_pointer_trace_batch(
        self,
        audio_tokens: Union[torch.Tensor, Sequence[torch.Tensor]],
        tokenizer: AutoTokenizer,
        languages: Optional[Sequence[Optional[str]]] = None,
        audio_lengths: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        include_eos: bool = False,
    ) -> List[dict[str, Any]]:
        """Batched ASR generation with timestamp-pointer probabilities per token."""
        device = next(self.parameters()).device
        c = self.config.num_audio_codebook
        samples = self._normalize_generate_audio_batch(
            audio_tokens=audio_tokens,
            audio_lengths=audio_lengths,
            num_codebooks=c,
        )
        if not samples:
            return []

        batch_size = len(samples)
        if languages is None:
            languages = [None] * batch_size
        elif isinstance(languages, str):
            languages = [languages] * batch_size
        elif len(languages) != batch_size:
            raise ValueError(
                f"Expected {batch_size} language entries, got {len(languages)}."
            )

        text_start_ids = tokenizer("<|text_start|>", return_tensors="pt").input_ids.to(
            device
        )
        eos_ids = tokenizer("<|text_end|>", return_tensors="pt").input_ids.to(device)
        eos_id = eos_ids[0, -1].item()

        sample_inputs = []
        sample_audio_masks = []
        sample_text_causal_masks = []
        lengths = []
        audio_starts = []
        audio_ends = []
        audio_lengths_out = []
        for sample, language in zip(samples, languages):
            sample = sample.to(device=device, dtype=torch.long)
            style = "<|asr|>"
            if language is not None:
                style += f"<|lang_start|>{language}<|lang_end|>"
            style_ids = tokenizer(style, return_tensors="pt").input_ids.to(device)

            style_inputs = style_ids.repeat(c, 1)
            text_inputs = text_start_ids.repeat(c, 1)
            sample_input_ids = torch.cat([style_inputs, sample, text_inputs], dim=1)

            audio_mask = torch.zeros(
                sample_input_ids.size(1), dtype=torch.bool, device=device
            )
            audio_start = style_inputs.size(1)
            audio_end = audio_start + sample.size(1)
            audio_mask[audio_start:audio_end] = True

            text_causal_mask = torch.zeros_like(audio_mask)
            text_causal_mask[-text_inputs.size(1) :] = True

            sample_inputs.append(sample_input_ids)
            sample_audio_masks.append(audio_mask)
            sample_text_causal_masks.append(text_causal_mask)
            lengths.append(sample_input_ids.size(1))
            audio_starts.append(audio_start)
            audio_ends.append(audio_end)
            audio_lengths_out.append(sample.size(1))

        max_len = max(lengths)
        input_ids = torch.zeros(
            batch_size,
            c,
            max_len,
            dtype=torch.long,
            device=device,
        )
        audio_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
            device=device,
        )
        text_causal_mask = torch.zeros_like(audio_mask)
        document_ids = torch.full(
            (batch_size, max_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        position_ids = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.long,
            device=device,
        )

        for idx, (
            sample_input_ids,
            sample_audio_mask,
            sample_text_causal_mask,
        ) in enumerate(zip(sample_inputs, sample_audio_masks, sample_text_causal_masks)):
            length = sample_input_ids.size(1)
            input_ids[idx, :, :length] = sample_input_ids
            audio_mask[idx, :length] = sample_audio_mask
            text_causal_mask[idx, :length] = sample_text_causal_mask
            document_ids[idx, :length] = 0
            position_ids[idx, :length] = torch.arange(length, device=device)

        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
        prompt_valid_mask = document_ids >= 0
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = lengths_tensor - 1
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        generated: List[List[int]] = [[] for _ in range(batch_size)]
        token_ids: List[List[int]] = [[] for _ in range(batch_size)]
        token_texts: List[List[str]] = [[] for _ in range(batch_size)]
        query_token_ids: List[List[int]] = [[] for _ in range(batch_size)]
        query_token_texts: List[List[str]] = [[] for _ in range(batch_size)]
        pointer_rows: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        outputs = self(
            input_ids=input_ids,
            audio_mask=audio_mask,
            text_causal_mask=text_causal_mask,
            document_ids=document_ids,
            position_ids=position_ids,
            use_cache=True,
            return_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError("LLM did not return past_key_values.")
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("LLM did not return hidden states.")

        logits = outputs.logits[batch_indices, last_indices, :]
        audio_keys = [
            self.timestamp_audio_key_proj(hidden_states[idx, start:end])
            for idx, (start, end) in enumerate(zip(audio_starts, audio_ends))
        ]
        current_query_states = hidden_states[batch_indices, last_indices]
        current_query_ids = [
            int(input_ids[idx, 0, int(last_indices[idx].item())].item())
            for idx in range(batch_size)
        ]
        pointer_scale = self.timestamp_query_proj.out_features**-0.5

        for step in range(max_new_tokens):
            next_id = self._sample_next_text_ids(logits, temperature)
            next_id = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_id, eos_id),
                next_id,
            )
            next_ids = next_id.squeeze(1)
            newly_finished = next_ids == eos_id

            for idx, token_id in enumerate(next_ids.tolist()):
                if bool(finished[idx].item()):
                    continue

                if token_id != eos_id or include_eos:
                    query = self.timestamp_query_proj(
                        current_query_states[idx : idx + 1]
                    )
                    pointer_logits = (
                        torch.matmul(query, audio_keys[idx].transpose(0, 1))
                        * pointer_scale
                    )
                    pointer_probs = torch.softmax(pointer_logits.float(), dim=-1)[0]
                    pointer_rows[idx].append(pointer_probs.detach().cpu())

                    token_ids[idx].append(token_id)
                    token_texts[idx].append(
                        tokenizer.decode([token_id], skip_special_tokens=False)
                    )
                    query_id = current_query_ids[idx]
                    query_token_ids[idx].append(query_id)
                    query_token_texts[idx].append(
                        tokenizer.decode([query_id], skip_special_tokens=False)
                    )

                if token_id != eos_id:
                    generated[idx].append(token_id)

            finished = finished | newly_finished
            if finished.all():
                break

            next_col = next_id.view(batch_size, 1, 1).expand(-1, c, -1)
            next_audio_mask = audio_mask.new_zeros(batch_size, 1)
            next_inputs_embeds = self._prepare_embed_inputs(
                next_col,
                next_audio_mask,
            )
            generated_mask = torch.ones(
                batch_size,
                step + 1,
                dtype=torch.bool,
                device=device,
            )
            decode_attention_mask = torch.cat(
                [prompt_valid_mask, generated_mask],
                dim=1,
            )
            next_position_ids = (lengths_tensor + step).unsqueeze(1)

            decode_outputs = self.llm(
                inputs_embeds=next_inputs_embeds,
                attention_mask=decode_attention_mask,
                return_dict=True,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = decode_outputs.past_key_values
            current_query_states = decode_outputs[0][:, -1, :]
            current_query_ids = [int(token_id) for token_id in next_ids.tolist()]
            logits = self.text_head(decode_outputs[0])[:, -1, :]

        traces = []
        for idx in range(batch_size):
            if pointer_rows[idx]:
                audio_attention = torch.stack(pointer_rows[idx], dim=0)
            else:
                audio_attention = torch.empty(
                    0,
                    audio_lengths_out[idx],
                    dtype=torch.float32,
                )

            traces.append(
                {
                    "text": tokenizer.decode(
                        generated[idx],
                        skip_special_tokens=True,
                    ).strip(),
                    "token_ids": token_ids[idx],
                    "token_texts": token_texts[idx],
                    "query_token_ids": query_token_ids[idx],
                    "query_token_texts": query_token_texts[idx],
                    "audio_attention": audio_attention,
                    "audio_start": audio_starts[idx],
                    "audio_end": audio_ends[idx],
                    "audio_num_tokens": audio_lengths_out[idx],
                    "timestamp_source": "pointer",
                }
            )
        return traces

    @torch.inference_mode()
    def generate_text(
        self,
        audio_tokens: torch.Tensor,
        tokenizer: AutoTokenizer,
        language: Optional[str] = None,
        task_mode: str = "plain",
        source_text: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> str:
        """Simple non-KV cached greedy/sampling ASR generation helper."""
        if audio_tokens.dim() == 3 and audio_tokens.size(0) != 1:
            raise ValueError(
                "generate_text() expects a single sample. "
                "Use generate_text_batch() for batched input."
            )
        return self.generate_text_batch(
            audio_tokens=audio_tokens,
            tokenizer=tokenizer,
            languages=[language],
            task_mode=task_mode,
            source_texts=[source_text],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=use_cache,
        )[0]

    @torch.inference_mode()
    def generate_text_batch(
        self,
        audio_tokens: Union[torch.Tensor, Sequence[torch.Tensor]],
        tokenizer: AutoTokenizer,
        languages: Optional[Sequence[Optional[str]]] = None,
        task_mode: str = "plain",
        source_texts: Optional[Sequence[Optional[str]]] = None,
        audio_lengths: Optional[Union[torch.Tensor, Sequence[int]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> List[str]:
        """Batched non-KV cached greedy/sampling ASR generation helper.

        Args:
            audio_tokens: Either a padded ``[B, C, T]`` tensor, one ``[C, T]``
                tensor, or a sequence of variable-length ``[C, T]`` tensors.
            tokenizer: Text tokenizer containing ASR special tokens.
            languages: Optional per-sample language ids. ``None`` omits the
                language prompt for that sample.
            audio_lengths: Optional valid audio lengths for padded tensor input.
            max_new_tokens: Maximum number of text tokens to generate.
            temperature: ``0`` for greedy decoding, otherwise sampling temperature.
            use_cache: Whether to use the LLM KV cache after the audio prefill.
        """
        device = next(self.parameters()).device
        c = self.config.num_audio_codebook
        task_mode = self._normalize_generation_task_mode(task_mode)

        samples = self._normalize_generate_audio_batch(
            audio_tokens=audio_tokens,
            audio_lengths=audio_lengths,
            num_codebooks=c,
        )
        if not samples:
            return []

        batch_size = len(samples)
        if languages is None:
            languages = [None] * batch_size
        elif isinstance(languages, str):
            languages = [languages] * batch_size
        elif len(languages) != batch_size:
            raise ValueError(
                f"Expected {batch_size} language entries, got {len(languages)}."
            )

        if source_texts is None:
            source_texts = [None] * batch_size
        elif isinstance(source_texts, str):
            source_texts = [source_texts] * batch_size
        elif len(source_texts) != batch_size:
            raise ValueError(
                f"Expected {batch_size} source_text entries, got {len(source_texts)}."
            )

        if task_mode == "furigana_rewrite" and any(
            source_text is None for source_text in source_texts
        ):
            raise ValueError(
                "task_mode='furigana_rewrite' requires source_texts for every sample."
            )

        text_start_ids = tokenizer("<|text_start|>", return_tensors="pt").input_ids.to(
            device
        )
        eos_ids = tokenizer("<|text_end|>", return_tensors="pt").input_ids.to(device)
        eos_id = eos_ids[0, -1].item()

        sample_inputs = []
        sample_audio_masks = []
        sample_text_causal_masks = []
        lengths = []
        task_token = self._generation_task_token(task_mode)
        for sample, language, source_text in zip(samples, languages, source_texts):
            sample = sample.to(device=device, dtype=torch.long)
            style = task_token
            if language is not None:
                style += f"<|lang_start|>{language}<|lang_end|>"
            style_ids = tokenizer(style, return_tensors="pt").input_ids.to(device)

            style_inputs = style_ids.repeat(c, 1)
            text_inputs = text_start_ids.repeat(c, 1)
            sequence_parts = [style_inputs, sample]
            if task_mode == "furigana_rewrite":
                source_ids = tokenizer(
                    f"<|src_text_start|>{source_text}<|src_text_end|>",
                    return_tensors="pt",
                ).input_ids.to(device)
                sequence_parts.append(source_ids.repeat(c, 1))

            sequence_parts.append(text_inputs)
            sample_input_ids = torch.cat(sequence_parts, dim=1)

            audio_mask = torch.zeros(
                sample_input_ids.size(1), dtype=torch.bool, device=device
            )
            audio_start = style_inputs.size(1)
            audio_end = audio_start + sample.size(1)
            audio_mask[audio_start:audio_end] = True

            text_causal_mask = torch.zeros_like(audio_mask)
            text_causal_mask[-text_inputs.size(1) :] = True

            sample_inputs.append(sample_input_ids)
            sample_audio_masks.append(audio_mask)
            sample_text_causal_masks.append(text_causal_mask)
            lengths.append(sample_input_ids.size(1))

        max_len = max(lengths)
        input_ids = torch.zeros(
            batch_size,
            c,
            max_len,
            dtype=torch.long,
            device=device,
        )
        audio_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.bool,
            device=device,
        )
        text_causal_mask = torch.zeros_like(audio_mask)
        document_ids = torch.full(
            (batch_size, max_len),
            -1,
            dtype=torch.int32,
            device=device,
        )
        position_ids = torch.zeros(
            batch_size,
            max_len,
            dtype=torch.long,
            device=device,
        )

        for idx, (
            sample_input_ids,
            sample_audio_mask,
            sample_text_causal_mask,
        ) in enumerate(zip(sample_inputs, sample_audio_masks, sample_text_causal_masks)):
            length = sample_input_ids.size(1)
            input_ids[idx, :, :length] = sample_input_ids
            audio_mask[idx, :length] = sample_audio_mask
            text_causal_mask[idx, :length] = sample_text_causal_mask
            document_ids[idx, :length] = 0
            position_ids[idx, :length] = torch.arange(length, device=device)

        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
        prompt_valid_mask = document_ids >= 0

        if use_cache:
            try:
                return self._generate_text_batch_with_cache(
                    input_ids=input_ids,
                    audio_mask=audio_mask,
                    text_causal_mask=text_causal_mask,
                    document_ids=document_ids,
                    position_ids=position_ids,
                    prompt_valid_mask=prompt_valid_mask,
                    lengths_tensor=lengths_tensor,
                    tokenizer=tokenizer,
                    eos_id=eos_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as exc:
                if not getattr(self, "_warned_asr_kv_cache_failure", False):
                    logger.warning(
                        "KV-cache ASR generation failed; falling back to uncached "
                        "generation. Error: %s",
                        exc,
                    )
                    self._warned_asr_kv_cache_failure = True

        return self._generate_text_batch_uncached(
            input_ids=input_ids,
            audio_mask=audio_mask,
            text_causal_mask=text_causal_mask,
            document_ids=document_ids,
            position_ids=position_ids,
            lengths_tensor=lengths_tensor,
            tokenizer=tokenizer,
            eos_id=eos_id,
            max_len=max_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def _sample_next_text_ids(
        self,
        logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if temperature and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    def _decode_generated_texts(
        self,
        generated: List[List[int]],
        tokenizer: AutoTokenizer,
    ) -> List[str]:
        return [
            tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated
        ]

    def _generate_text_batch_uncached(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        text_causal_mask: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor,
        lengths_tensor: torch.Tensor,
        tokenizer: AutoTokenizer,
        eos_id: int,
        max_len: int,
        max_new_tokens: int,
        temperature: float,
    ) -> List[str]:
        batch_size = input_ids.size(0)
        c = input_ids.size(1)
        device = input_ids.device
        last_indices = lengths_tensor - 1
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated: List[List[int]] = [[] for _ in range(batch_size)]

        for _ in range(max_new_tokens):
            batch = {
                "input_ids": input_ids,
                "audio_mask": audio_mask,
                "text_causal_mask": text_causal_mask,
                "document_ids": document_ids,
                "position_ids": position_ids,
            }
            logits = self(**batch).logits
            batch_indices = torch.arange(batch_size, device=device)
            logits = logits[batch_indices, last_indices, :]
            next_id = self._sample_next_text_ids(logits, temperature)

            next_id = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_id, eos_id),
                next_id,
            )
            next_ids = next_id.squeeze(1)
            newly_finished = next_ids == eos_id

            for idx, token_id in enumerate(next_ids.tolist()):
                if not finished[idx] and token_id != eos_id:
                    generated[idx].append(token_id)

            finished = finished | newly_finished
            if finished.all():
                break

            next_col = next_id.view(batch_size, 1, 1).expand(-1, c, -1)
            input_ids = torch.cat([input_ids, next_col], dim=2)
            audio_mask = torch.cat(
                [audio_mask, audio_mask.new_zeros(batch_size, 1)], dim=1
            )
            text_causal_mask = torch.cat(
                [text_causal_mask, text_causal_mask.new_ones(batch_size, 1)],
                dim=1,
            )
            document_ids = torch.cat(
                [document_ids, document_ids.new_zeros(batch_size, 1)],
                dim=1,
            )
            generated_steps = input_ids.size(2) - max_len - 1
            next_positions = lengths_tensor + generated_steps
            position_ids = torch.cat(
                [position_ids, next_positions.unsqueeze(1)],
                dim=1,
            )
            last_indices = torch.full_like(last_indices, input_ids.size(2) - 1)

        return self._decode_generated_texts(generated, tokenizer)

    def _generate_text_batch_with_cache(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        text_causal_mask: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor,
        prompt_valid_mask: torch.Tensor,
        lengths_tensor: torch.Tensor,
        tokenizer: AutoTokenizer,
        eos_id: int,
        max_new_tokens: int,
        temperature: float,
    ) -> List[str]:
        batch_size = input_ids.size(0)
        c = input_ids.size(1)
        device = input_ids.device
        batch_indices = torch.arange(batch_size, device=device)
        last_indices = lengths_tensor - 1
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated: List[List[int]] = [[] for _ in range(batch_size)]

        prefill_outputs = self(
            input_ids=input_ids,
            audio_mask=audio_mask,
            text_causal_mask=text_causal_mask,
            document_ids=document_ids,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = prefill_outputs.past_key_values
        if past_key_values is None:
            raise RuntimeError("LLM did not return past_key_values.")

        logits = prefill_outputs.logits[batch_indices, last_indices, :]
        for step in range(max_new_tokens):
            next_id = self._sample_next_text_ids(logits, temperature)
            next_id = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_id, eos_id),
                next_id,
            )
            next_ids = next_id.squeeze(1)
            newly_finished = next_ids == eos_id

            for idx, token_id in enumerate(next_ids.tolist()):
                if not finished[idx] and token_id != eos_id:
                    generated[idx].append(token_id)

            finished = finished | newly_finished
            if finished.all():
                break

            next_col = next_id.view(batch_size, 1, 1).expand(-1, c, -1)
            next_audio_mask = audio_mask.new_zeros(batch_size, 1)
            next_inputs_embeds = self.omnivoice._prepare_embed_inputs(
                next_col,
                next_audio_mask,
            )
            generated_mask = torch.ones(
                batch_size,
                step + 1,
                dtype=torch.bool,
                device=device,
            )
            decode_attention_mask = torch.cat(
                [prompt_valid_mask, generated_mask],
                dim=1,
            )
            next_position_ids = (lengths_tensor + step).unsqueeze(1)

            decode_outputs = self.llm(
                inputs_embeds=next_inputs_embeds,
                attention_mask=decode_attention_mask,
                return_dict=True,
                position_ids=next_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = decode_outputs.past_key_values
            logits = self.text_head(decode_outputs[0])[:, -1, :]

        return self._decode_generated_texts(generated, tokenizer)

    @staticmethod
    def _normalize_generation_task_mode(task_mode: str) -> str:
        aliases = {
            "asr": "plain",
            "plain": "plain",
            "asr_plain": "plain",
            "furigana": "furigana_audio",
            "furigana_audio": "furigana_audio",
            "asr_furigana": "furigana_audio",
            "rewrite": "furigana_rewrite",
            "furigana_rewrite": "furigana_rewrite",
        }
        normalized = aliases.get(task_mode)
        if normalized is None:
            raise ValueError(
                f"Unsupported ASR generation task_mode: {task_mode!r}. "
                "Expected plain, furigana_audio, or furigana_rewrite."
            )
        return normalized

    @staticmethod
    def _generation_task_token(task_mode: str) -> str:
        if task_mode == "plain":
            return "<|asr|>"
        if task_mode == "furigana_audio":
            return "<|asr_furigana|>"
        if task_mode == "furigana_rewrite":
            return "<|furigana_rewrite|>"
        raise ValueError(f"Unsupported ASR generation task_mode: {task_mode!r}")

    @staticmethod
    def _normalize_generate_audio_batch(
        audio_tokens: Union[torch.Tensor, Sequence[torch.Tensor]],
        audio_lengths: Optional[Union[torch.Tensor, Sequence[int]]],
        num_codebooks: int,
    ) -> List[torch.Tensor]:
        if isinstance(audio_tokens, torch.Tensor):
            if audio_tokens.dim() == 2:
                samples = [audio_tokens]
            elif audio_tokens.dim() == 3:
                if audio_lengths is None:
                    lengths = [audio_tokens.size(2)] * audio_tokens.size(0)
                elif isinstance(audio_lengths, torch.Tensor):
                    lengths = [int(length) for length in audio_lengths.tolist()]
                else:
                    lengths = [int(length) for length in audio_lengths]
                if len(lengths) != audio_tokens.size(0):
                    raise ValueError(
                        f"Expected {audio_tokens.size(0)} audio lengths, "
                        f"got {len(lengths)}."
                    )
                samples = [
                    audio_tokens[idx, :, : lengths[idx]]
                    for idx in range(audio_tokens.size(0))
                ]
            else:
                raise ValueError(
                    "audio_tokens must have shape [C, T], [B, C, T], "
                    "or be a sequence of [C, T] tensors."
                )
        else:
            if audio_lengths is not None:
                raise ValueError(
                    "audio_lengths is only supported when audio_tokens is a tensor."
                )
            samples = []
            for sample in audio_tokens:
                if sample.dim() == 3 and sample.size(0) == 1:
                    sample = sample.squeeze(0)
                samples.append(sample)

        for idx, sample in enumerate(samples):
            if sample.dim() != 2:
                raise ValueError(
                    f"Expected sample {idx} to have shape [C, T], got "
                    f"{tuple(sample.shape)}."
                )
            if sample.size(0) != num_codebooks:
                raise ValueError(
                    f"Expected sample {idx} to have {num_codebooks} audio "
                    f"codebooks, got {sample.size(0)}."
                )

        return samples


def _get_asr_causal_mask(document_ids):
    return partial(_mask_mod_asr_causal, document_ids)


def _mask_mod_asr_causal(document_ids, b, h, q_idx, kv_idx):
    if document_ids.ndim == 1:
        q_doc = document_ids[q_idx]
        kv_doc = document_ids[kv_idx]
    else:
        q_doc = document_ids[b, q_idx]
        kv_doc = document_ids[b, kv_idx]

    same_doc = q_doc == kv_doc
    valid_doc = q_doc >= 0
    return valid_doc & same_doc & (q_idx >= kv_idx)


def _get_asr_prefix_lm_mask(document_ids, text_causal_mask):
    return partial(_mask_mod_asr_prefix_lm, document_ids, text_causal_mask)


def _mask_mod_asr_prefix_lm(document_ids, text_causal_mask, b, h, q_idx, kv_idx):
    if document_ids.ndim == 1:
        q_doc = document_ids[q_idx]
        kv_doc = document_ids[kv_idx]
        kv_is_prefix = ~text_causal_mask[kv_idx]
        q_is_text = text_causal_mask[q_idx]
    else:
        q_doc = document_ids[b, q_idx]
        kv_doc = document_ids[b, kv_idx]
        kv_is_prefix = ~text_causal_mask[b, kv_idx]
        q_is_text = text_causal_mask[b, q_idx]

    same_doc = q_doc == kv_doc
    valid_doc = q_doc >= 0
    causal_text = q_is_text & (q_idx >= kv_idx)
    return valid_doc & same_doc & (kv_is_prefix | causal_text)

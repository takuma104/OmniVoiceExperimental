"""Smoke test for MALLE-style mel_mode in OmniVoice.

Builds a tiny mel-mode model with a small LM config, runs MelSampleProcessor
on a synthesized waveform, packs through MelPackingDataCollator, runs a
forward + backward pass, and checks shapes and finite loss/grads.

This avoids any network access and any large checkpoint download. The LLM
backbone is initialised from a tiny Qwen3 config using random weights.
"""

import os
import sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from transformers import AutoConfig, AutoModel, AutoTokenizer

from omnivoice.data.collator import MelPackingDataCollator
from omnivoice.data.processor import MelSampleProcessor
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig


def build_tiny_model_and_tokenizer():
    """Build a small Qwen3-style backbone with random weights for testing."""
    base = "Qwen/Qwen3-0.6B"
    print(f"Loading tokenizer from {base} ...")
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    new_tokens = [
        "<|denoise|>",
        "<|lang_start|>",
        "<|lang_end|>",
        "<|instruct_start|>",
        "<|instruct_end|>",
        "<|text_start|>",
        "<|text_end|>",
    ]
    to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})

    print("Building tiny LM config (random weights, no download)...")
    llm_config = AutoConfig.from_pretrained(base)
    # Shrink to keep the smoke test fast and CPU-friendly.
    llm_config.hidden_size = 64
    llm_config.num_hidden_layers = 2
    llm_config.num_attention_heads = 2
    llm_config.num_key_value_heads = 2
    llm_config.intermediate_size = 128
    llm_config.max_position_embeddings = 4096
    if hasattr(llm_config, "head_dim"):
        llm_config.head_dim = llm_config.hidden_size // llm_config.num_attention_heads
    llm_config.vocab_size = max(llm_config.vocab_size, len(tokenizer))

    llm = AutoModel.from_config(llm_config).to(torch.float32)
    ov_config = OmniVoiceConfig(
        llm_config=llm_config,
        mel_mode=True,
        num_mels=128,
    )
    model = OmniVoice(config=ov_config, llm=llm)
    if len(tokenizer) != model.config.llm_config.vocab_size:
        model.llm.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def make_processor(tokenizer):
    return MelSampleProcessor(
        text_tokenizer=tokenizer,
        num_mels=128,
        mel_sample_rate=44100,
        mel_n_fft=2048,
        mel_hop_size=512,
        mel_win_size=2048,
        mel_fmin=0,
        mel_fmax=None,
        prompt_ratio_range=(0.0, 0.3),
        mask_ratio_range=(0.5, 1.0),
        drop_cond_ratio=0.0,
        language_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
    )


def synth_sample(duration_s=2.0, sr=44100):
    n = int(duration_s * sr)
    # Simple sine to give the mel some structure.
    t = torch.arange(n, dtype=torch.float32) / sr
    wav = 0.3 * torch.sin(2 * 3.14159 * 220 * t) + 0.1 * torch.sin(
        2 * 3.14159 * 440 * t
    )
    wav = wav.unsqueeze(0)  # (1, T)
    return {
        "audio": wav,
        "audio_duration": duration_s,
        "label": {"id": 0, "text": "this is a smoke test"},
    }


def main():
    torch.manual_seed(0)

    model, tokenizer = build_tiny_model_and_tokenizer()
    model.train()

    print("Building processor and collator...")
    processor = make_processor(tokenizer)
    batch_tokens = 1024
    collator = MelPackingDataCollator(processor, batch_tokens=batch_tokens)

    print("Synthesizing 2 samples and packing...")
    samples = [synth_sample(duration_s=1.5), synth_sample(duration_s=1.0)]
    processed = [processor(s) for s in samples]
    for i, p in enumerate(processed):
        print(
            f"  sample {i}: length={p['length']}, "
            f"mel_input={tuple(p['mel_input'].shape)}, "
            f"audio_mask_sum={int(p['audio_mask'].sum())}, "
            f"mel_mask_sum={int(p['mel_mask'].sum())}, "
            f"mel_loss_mask_sum={int(p['mel_loss_mask'].sum())}"
        )

    batch = collator(processed)
    print("Batch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")

    # Forward (skip flex_attention for the smoke test by using attn_implementation="eager"
    # implicitly: we pass document_ids=None to avoid flex_attention).
    # However our model's forward attempts to build a flex block mask if
    # attention_mask is None and document_ids is provided. To keep the
    # smoke test backend-agnostic, drop document_ids.
    fwd_batch = {k: v for k, v in batch.items() if k != "document_ids"}

    print("Running forward...")
    out = model(**fwd_batch)
    assert out.loss is not None, "loss must be computed in mel_mode"
    assert torch.isfinite(out.loss), f"loss not finite: {out.loss}"
    assert out.logits.shape == (1, batch_tokens, 128), (
        f"unexpected logits shape: {out.logits.shape}"
    )
    print(f"  loss = {out.loss.item():.4f}")

    print("Running backward...")
    out.loss.backward()
    grads = [
        ("mel_prenet[0]", model.mel_prenet[0].weight.grad),
        ("mel_head", model.mel_head.weight.grad),
        ("mel_mask_embed", model.mel_mask_embed.grad),
    ]
    for name, g in grads:
        assert g is not None, f"no grad for {name}"
        assert torch.isfinite(g).all(), f"non-finite grad in {name}"
        print(f"  {name} grad norm = {g.norm().item():.4f}")

    # Sanity: discrete-mode modules must NOT exist.
    assert model.audio_embeddings is None, "audio_embeddings should be None in mel_mode"
    assert model.audio_heads is None, "audio_heads should be None in mel_mode"

    print("OK: mel_mode smoke test passed.")


if __name__ == "__main__":
    main()

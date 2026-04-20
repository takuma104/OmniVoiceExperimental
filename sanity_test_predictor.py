"""Sanity test for Cascaded Predictor integration.

Runs a tiny synthetic forward + backward pass on CPU to verify:
- Predictor instantiation and pretrained weight load
- Shape correctness for cb0 logits and Predictor logits
- Loss is finite and backward succeeds
"""

import logging
import sys

import torch
from transformers import AutoConfig

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def main():
    llm_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    llm_config.num_hidden_layers = 2  # shrink for speed

    ov_config = OmniVoiceConfig(
        audio_vocab_size=2049,
        audio_mask_id=2048,
        num_audio_codebook=16,
        llm_config=llm_config,
        use_predictor=True,
        predictor_pretrained_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    import os
    target_dtype = torch.bfloat16 if os.environ.get("BF16") == "1" else torch.float32
    print(f"target_dtype = {target_dtype}")

    model = OmniVoice(config=ov_config)
    model._load_predictor_pretrained(dtype=target_dtype)
    model = model.to(target_dtype)
    model.train()

    print("audio_heads params:", sum(p.numel() for p in model.audio_heads.parameters()))
    print(
        "backbone_to_talker_proj params:",
        sum(p.numel() for p in model.backbone_to_talker_proj.parameters()),
    )
    print("predictor params:", sum(p.numel() for p in model.predictor.parameters()))

    # dtype audit
    dtypes = {p.dtype for p in model.parameters()}
    print("distinct param dtypes:", dtypes)

    # Fake batch: B=1, S=20, text prefix len=5, audio len=15
    B, C, S = 1, 16, 20
    text_len, audio_len = 5, 15
    V = ov_config.audio_vocab_size
    MASK = ov_config.audio_mask_id

    input_ids = torch.randint(0, 1000, (B, C, S))
    # Mark audio region
    audio_mask = torch.zeros(B, S, dtype=torch.bool)
    audio_mask[:, text_len:] = True

    # Fake clean codec tokens for cb1..15 in audio region.
    clean_audio = torch.randint(0, 2048, (B, C, audio_len))
    input_ids[:, :, text_len:] = clean_audio

    # Mask ~50% of cb0 in audio region; mask cb1..15 fully (predictor_mode).
    cb0_mask = torch.rand(B, audio_len) < 0.5
    masked_input = input_ids.clone()
    masked_input[:, 0, text_len:][cb0_mask] = MASK
    masked_input[:, 1:, text_len:] = MASK

    labels = torch.full_like(masked_input, -100)
    # cb0: compute loss on masked positions only.
    labels[:, 0, text_len:][cb0_mask] = clean_audio[:, 0][cb0_mask]
    # cb1..15: compute loss everywhere in audio region.
    labels[:, 1:, text_len:] = clean_audio[:, 1:]

    out = model(input_ids=masked_input, audio_mask=audio_mask, labels=labels)
    print("loss:", out.loss.item())
    print("logits shape:", out.logits.shape)

    out.loss.backward()
    print("backward OK")

    # Check that Predictor, backbone_to_talker_proj, and audio_heads all got grads.
    for name in ("audio_heads", "backbone_to_talker_proj", "predictor"):
        mod = getattr(model, name)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
        print(f"{name}: grad_nonzero={has_grad}")


if __name__ == "__main__":
    main()

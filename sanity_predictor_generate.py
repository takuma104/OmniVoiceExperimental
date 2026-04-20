"""Sanity test for Predictor.generate() API shape used in _fill_cb1_to_15_with_predictor."""

import logging
import sys

import torch
from transformers import AutoConfig

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)


def main():
    llm_config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    llm_config.num_hidden_layers = 2

    ov_config = OmniVoiceConfig(
        audio_vocab_size=2049,
        audio_mask_id=2048,
        num_audio_codebook=16,
        llm_config=llm_config,
        use_predictor=True,
        predictor_pretrained_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
    model = OmniVoice(config=ov_config)
    model._load_predictor_pretrained()
    model = model.to(torch.float32)
    model.eval()

    T = 10
    H_llm = llm_config.hidden_size

    with torch.no_grad():
        h = torch.randn(T, H_llm)
        talker = model.backbone_to_talker_proj(h)  # [T, H_talker]
        cb0 = torch.randint(0, 2048, (T,))
        cb0_emb = model.predictor.model.codec_embedding[0](cb0)  # [T, H_talker]

        pred_inputs = torch.stack([talker, cb0_emb], dim=1)  # [T, 2, H_talker]
        print("pred_inputs shape:", pred_inputs.shape)

        gen_out = model.predictor.generate(
            inputs_embeds=pred_inputs,
            max_new_tokens=15,
            do_sample=False,
        )
        print("gen_out type:", type(gen_out))
        cb_1_15 = gen_out if isinstance(gen_out, torch.Tensor) else gen_out.sequences
        print("cb_1_15 shape:", cb_1_15.shape)
        cb_1_15 = cb_1_15[:, -15:]
        print("trimmed cb_1_15 shape:", cb_1_15.shape, "expected [T=10, 15]")
        assert cb_1_15.shape == (T, 15), f"Unexpected shape: {cb_1_15.shape}"
        print("OK")


if __name__ == "__main__":
    main()

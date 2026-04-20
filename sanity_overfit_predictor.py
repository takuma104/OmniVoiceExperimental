"""1-sample overfit test for Cascaded Predictor.

Runs ~150 steps on a single synthetic batch. Expected:
- Total loss drops noticeably.
- Initial cb1..15 loss is lower than random (pretrained weights working).
"""

import logging
import sys

import torch
import torch.nn.functional as F
from transformers import AutoConfig

from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)


def compute_split_losses(model, input_ids, audio_mask, labels):
    """Recompute cb0 and cb1..15 losses separately for logging."""
    inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
    llm_outputs = model.llm(inputs_embeds=inputs_embeds, return_dict=True)
    hidden_states = llm_outputs[0]

    cb0_logits = model.audio_heads(hidden_states)
    cb0_labels = labels[:, 0, :]
    cb0_per_token = F.cross_entropy(
        cb0_logits.permute(0, 2, 1), cb0_labels, reduction="none", ignore_index=-100
    )
    cb0_valid = (cb0_labels != -100).float()
    cb0_loss = (cb0_per_token * cb0_valid).sum() / cb0_valid.sum().clamp(min=1.0)

    clean_tokens = torch.where(labels != -100, labels, input_ids)
    clean_bsc = clean_tokens.permute(0, 2, 1)
    labels_bsc = labels.permute(0, 2, 1)
    talker_hidden_flat = hidden_states[audio_mask]
    clean_flat = clean_bsc[audio_mask]
    pred_labels = labels_bsc[audio_mask][:, 1:]

    talker_projected = model.backbone_to_talker_proj(talker_hidden_flat)
    codec_embs = [
        model.predictor.model.codec_embedding[i](clean_flat[:, i])
        for i in range(model._num_predicted_codebooks)
    ]
    pred_inputs = torch.stack([talker_projected, *codec_embs], dim=1)
    pred_logits = model.predictor.forward_finetune(inputs_embeds=pred_inputs).logits
    pred_per_token = F.cross_entropy(
        pred_logits.permute(0, 2, 1), pred_labels, reduction="none", ignore_index=-100
    )
    pred_valid = (pred_labels != -100).float()
    layer_means = (pred_per_token * pred_valid).sum(dim=0) / pred_valid.sum(dim=0).clamp(min=1.0)
    return cb0_loss.item(), layer_means.mean().item()


def main():
    torch.manual_seed(0)

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
    model.train()

    # Single fixed batch.
    B, C, S = 1, 16, 40
    text_len, audio_len = 8, 32
    MASK = ov_config.audio_mask_id

    input_ids = torch.randint(0, 1000, (B, C, S))
    audio_mask = torch.zeros(B, S, dtype=torch.bool)
    audio_mask[:, text_len:] = True
    clean_audio = torch.randint(0, 2048, (B, C, audio_len))
    input_ids[:, :, text_len:] = clean_audio

    cb0_mask = torch.rand(B, audio_len) < 0.5
    masked_input = input_ids.clone()
    masked_input[:, 0, text_len:][cb0_mask] = MASK
    masked_input[:, 1:, text_len:] = MASK

    labels = torch.full_like(masked_input, -100)
    labels[:, 0, text_len:][cb0_mask] = clean_audio[:, 0][cb0_mask]
    labels[:, 1:, text_len:] = clean_audio[:, 1:]

    # Initial split losses.
    with torch.no_grad():
        init_cb0, init_cb_rest = compute_split_losses(model, masked_input, audio_mask, labels)
    print(f"initial  cb0_loss={init_cb0:.4f}  cb1..15_mean={init_cb_rest:.4f}  "
          f"(random baseline: ln(2049)={torch.log(torch.tensor(2049.0)).item():.4f})")

    optim = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for step in range(150):
        optim.zero_grad()
        out = model(input_ids=masked_input, audio_mask=audio_mask, labels=labels)
        out.loss.backward()
        optim.step()
        if step == 0 or (step + 1) % 25 == 0:
            print(f"step {step+1:4d}  loss={out.loss.item():.4f}")

    with torch.no_grad():
        final_cb0, final_cb_rest = compute_split_losses(model, masked_input, audio_mask, labels)
    print(f"final    cb0_loss={final_cb0:.4f}  cb1..15_mean={final_cb_rest:.4f}")


if __name__ == "__main__":
    main()

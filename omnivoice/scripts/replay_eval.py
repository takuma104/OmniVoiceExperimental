#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Teacher-forced reconstruction script for AR checkpoints.

Replays what the trainer's eval loop does: for each sample in a webdataset
manifest, run the AR sample processor (the same one used during training),
forward the model with ground-truth context, take ``argmax`` over the
codebook heads, and decode the resulting tokens back to audio.

If the checkpoint has truly memorized the data (eval loss << 1), the
predicted tokens should match the ground truth almost exactly and the
decoded audio should be nearly identical to the original. This isolates
the question of memorization from the question of free-running AR
generation, which can fail for unrelated reasons (sampling instability,
EOS placement, distribution mismatch).

Usage:

    python -m omnivoice.scripts.replay_eval \
        --checkpoint output/run0-checkpoint-1000 \
        --manifest  data/emilia/tokens/emilia_en_dev/data.lst \
        --output_dir replay_out \
        --max_samples 5

The ``--manifest`` file follows the same ``.lst`` format used by training
(``tar_path label_jsonl_path num_items num_seconds`` per line).
"""

import argparse
import logging
import os
import random
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice
from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
from omnivoice.data.processor import OmniVoiceARSampleProcessor

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument(
        "--manifest",
        required=True,
        type=str,
        help="Path to a .lst manifest (tar/jsonl/num_items/num_seconds per line).",
    )
    p.add_argument("--output_dir", required=True, type=str)
    p.add_argument("--max_samples", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    p.add_argument(
        "--prompt_ratio",
        type=float,
        default=0.0,
        help="Fixed prompt ratio for the AR processor. 0.0 means no prompt"
        " audio is observed; the model must predict the entire target.",
    )
    p.add_argument(
        "--language_ratio",
        type=float,
        default=0.0,
        help="Probability of including the language tag in the conditioning."
        " Set to match the training config (default: 0.0 for the AR Emilia"
        " setup).",
    )
    p.add_argument(
        "--instruct_ratio",
        type=float,
        default=0.0,
        help="Probability of including the instruct tag in the conditioning.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save_gt",
        action="store_true",
        default=True,
        help="Also save ground-truth audio (decoded from the original tokens)"
        " for side-by-side comparison.",
    )
    return p.parse_args()


def find_audio_span(audio_mask_1d: torch.Tensor) -> tuple[int, int]:
    """Return [start, end) of the contiguous audio region in the input."""
    positions = torch.where(audio_mask_1d.bool())[0]
    if len(positions) == 0:
        return -1, -1
    return int(positions[0].item()), int(positions[-1].item()) + 1


def _decode_tokens(model: OmniVoice, tokens: torch.Tensor) -> np.ndarray:
    """Decode [C, T] tokens to a 1-D waveform via the audio tokenizer."""
    tokenizer_device = model.audio_tokenizer.device
    wav = (
        model.audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0))
        .audio_values[0]
        .detach()
        .cpu()
        .numpy()
    )
    return np.squeeze(wav)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    logger.info("Loading checkpoint from %s ...", args.checkpoint)
    model = OmniVoice.from_pretrained(
        args.checkpoint, device_map=args.device, dtype=dtype
    )
    model.eval()

    if not getattr(model.config, "ar_mode", False):
        logger.warning(
            "WARNING: model.config.ar_mode is False. This script targets AR"
            " checkpoints; results on a NAR checkpoint will be meaningless."
        )

    audio_eos_id = getattr(model.config, "audio_eos_id", model.config.audio_mask_id)
    audio_vocab_size = model.config.audio_vocab_size
    num_codebooks = model.config.num_audio_codebook
    valid_token_max = audio_vocab_size - 2  # last valid token id (EOS occupies last slot)

    processor = OmniVoiceARSampleProcessor(
        text_tokenizer=model.text_tokenizer,
        num_channels=num_codebooks,
        audio_eos_id=audio_eos_id,
        prompt_ratio_range=(args.prompt_ratio, args.prompt_ratio),
        drop_cond_ratio=0.0,
        language_ratio=args.language_ratio,
        use_pinyin_ratio=0.0,
        instruct_ratio=args.instruct_ratio,
        only_instruct_ratio=0.0,
    )

    manifests = webdataset_manifest_reader(args.manifest)
    reader = WebDatasetReader(manifests=manifests, evaluation=True)

    sample_rate = model.sampling_rate or 24000
    device = torch.device(args.device)

    logger.info(
        "Replaying up to %d samples (prompt_ratio=%.2f, language_ratio=%.2f,"
        " instruct_ratio=%.2f)...",
        args.max_samples,
        args.prompt_ratio,
        args.language_ratio,
        args.instruct_ratio,
    )

    count = 0
    total_match = 0
    total_count = 0
    for raw_sample in reader:
        if count >= args.max_samples:
            break

        label = raw_sample.get("label", {})
        sample_id = label.get("id", f"sample_{count:04d}")
        text = label.get("text", "")
        logger.info("\n[%d] id=%s", count, sample_id)
        logger.info("    text: %s", text)

        try:
            processed = processor(raw_sample)
        except Exception as e:
            logger.warning("    skip: processor error: %s", e)
            continue

        input_ids = processed["input_ids"].unsqueeze(0).to(device)  # [1, C, L]
        labels = processed["labels"].unsqueeze(0).to(device)  # [1, C, L]
        audio_mask = processed["audio_mask"].unsqueeze(0).to(device)  # [1, L]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                audio_mask=audio_mask,
                labels=labels,
            )
        logits = outputs.logits  # [1, C, L, V]
        loss = float(outputs.loss) if outputs.loss is not None else float("nan")
        logger.info("    teacher-forced loss: %.4f", loss)

        # Predicted token at position p = argmax of logits at position p,
        # which is the model's guess for input_ids[p+1].
        predicted = logits.argmax(dim=-1)  # [1, C, L]

        # Find the audio span. audio_inputs covers [audio_start, audio_end);
        # the model's predictions for THIS span come from logits at
        # [audio_start - 1, audio_end - 1).
        audio_start, audio_end = find_audio_span(audio_mask.squeeze(0))
        if audio_start < 0:
            logger.warning("    skip: no audio positions in this sample")
            continue

        gt_audio_tokens = raw_sample["audio_tokens"].long()
        if gt_audio_tokens.dim() == 3:
            gt_audio_tokens = gt_audio_tokens.squeeze(0)
        gt_audio_tokens = gt_audio_tokens.to(device)
        T_audio = gt_audio_tokens.size(-1)

        # Predicted tokens for the audio range (T_audio frames + 1 EOS frame).
        # We only need the first T_audio of these for decoding (drop the EOS
        # prediction at the end).
        if audio_start - 1 < 0:
            logger.warning("    skip: audio_start is at the very beginning")
            continue
        pred_audio_range = predicted[
            0, :, audio_start - 1 : audio_end - 1
        ]  # [C, T_audio + 1]
        pred_audio_frames = pred_audio_range[:, :T_audio]  # [C, T_audio]

        # Token-level match rate vs ground truth, per codebook + total.
        match = (pred_audio_frames == gt_audio_tokens[:, :T_audio]).float()
        per_cb = match.mean(dim=-1).cpu().tolist()
        overall = float(match.mean().item())
        total_match += int(match.sum().item())
        total_count += int(match.numel())
        logger.info(
            "    token match: overall=%.1f%%, per-codebook=[%s]",
            overall * 100,
            ", ".join(f"{x*100:.1f}" for x in per_cb),
        )

        # Defensive: clamp any out-of-range tokens (EOS slot) so the decoder
        # doesn't IndexError. With a memorized checkpoint this shouldn't
        # happen, but warn if it does.
        oob_mask = pred_audio_frames > valid_token_max
        if oob_mask.any():
            n = int(oob_mask.sum().item())
            logger.warning(
                "    %d predicted tokens are >= %d (EOS slot); clamping to 0.",
                n,
                valid_token_max + 1,
            )
            pred_audio_frames = pred_audio_frames.clone()
            pred_audio_frames[oob_mask] = 0

        pred_wav = _decode_tokens(model, pred_audio_frames)
        out_path = os.path.join(args.output_dir, f"{sample_id}_pred.wav")
        sf.write(out_path, pred_wav, samplerate=sample_rate)
        logger.info("    wrote %s", out_path)

        if args.save_gt:
            gt_wav = _decode_tokens(model, gt_audio_tokens[:, :T_audio])
            gt_path = os.path.join(args.output_dir, f"{sample_id}_gt.wav")
            sf.write(gt_path, gt_wav, samplerate=sample_rate)
            logger.info("    wrote %s", gt_path)

        count += 1

    logger.info("\n=== Summary ===")
    logger.info("samples processed: %d", count)
    if total_count > 0:
        logger.info(
            "overall token match: %.2f%% (%d / %d)",
            total_match / total_count * 100,
            total_match,
            total_count,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate an OmniVoice ASR checkpoint on a WebDataset data.lst."""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
from omnivoice.models.omnivoice_asr import OmniVoiceForSpeechRecognition
from omnivoice.utils.flex_attention_patch import patch_flex_attention_for_sm12

logger = logging.getLogger(__name__)


def _resolve_dtype(dtype: str, device: str):
    if dtype == "auto":
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _strip_generation_suffix(text: str, extra_markers: Sequence[str] = ()) -> str:
    markers = [
        "<|text_end|>",
        "<|endoftext|>",
        "<|im_end|>",
        "</s>",
        "</think>",
        "<think>",
        "\ufffd",
    ]
    markers.extend(m for m in extra_markers if m)
    cut = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut].strip()


def _levenshtein(ref: Sequence[str], hyp: Sequence[str]) -> int:
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref

    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ref_item != hyp_item)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _char_units(text: str) -> list[str]:
    return [c for c in text if not c.isspace()]


def _word_units(text: str) -> list[str]:
    return text.split()


def _safe_rate(errors: int, total: int) -> float:
    return errors / total if total > 0 else 0.0


def _iter_samples(data_lst: str) -> Iterable[dict]:
    manifests = webdataset_manifest_reader(data_lst)
    reader = WebDatasetReader(manifests=manifests, evaluation=True)
    return iter(reader)


def evaluate(args):
    patch_flex_attention_for_sm12()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _resolve_dtype(args.dtype, device)

    if device == "cpu":
        logger.warning(
            "CPU evaluation may be very slow and may not support flex attention "
            "on all torch versions."
        )

    logger.info("Loading checkpoint: %s", args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = OmniVoiceForSpeechRecognition.from_pretrained(
        args.checkpoint,
        attn_implementation="flex_attention",
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else None
    fout = None
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        fout = output_jsonl.open("w", encoding="utf-8")

    total_char_errors = 0
    total_chars = 0
    total_word_errors = 0
    total_words = 0
    count = 0

    try:
        progress = tqdm(_iter_samples(args.data_lst), total=args.limit)
        for sample in progress:
            label = sample["label"]
            ref_text = label["text"]
            language = args.language or label.get("language_id")
            audio_tokens = sample["audio_tokens"]

            hyp_text = model.generate_text(
                audio_tokens=audio_tokens,
                tokenizer=tokenizer,
                language=language,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            if not args.no_strip_suffix:
                hyp_text = _strip_generation_suffix(hyp_text, args.stop_marker)

            if args.no_normalize:
                ref_norm = ref_text
                hyp_norm = hyp_text
            else:
                ref_norm = _normalize_text(ref_text)
                hyp_norm = _normalize_text(hyp_text)

            ref_chars = _char_units(ref_norm)
            hyp_chars = _char_units(hyp_norm)
            char_errors = _levenshtein(ref_chars, hyp_chars)

            ref_words = _word_units(ref_norm)
            hyp_words = _word_units(hyp_norm)
            word_errors = _levenshtein(ref_words, hyp_words)

            total_char_errors += char_errors
            total_chars += len(ref_chars)
            total_word_errors += word_errors
            total_words += len(ref_words)
            count += 1

            item = {
                "id": label.get("id"),
                "language_id": language,
                "reference": ref_text,
                "hypothesis": hyp_text,
                "reference_normalized": ref_norm,
                "hypothesis_normalized": hyp_norm,
                "cer": _safe_rate(char_errors, len(ref_chars)),
                "wer": _safe_rate(word_errors, len(ref_words)),
                "char_errors": char_errors,
                "chars": len(ref_chars),
                "word_errors": word_errors,
                "words": len(ref_words),
            }
            if fout is not None:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()

            progress.set_postfix(
                {
                    "cer": f"{_safe_rate(total_char_errors, total_chars) * 100:.2f}",
                    "wer": f"{_safe_rate(total_word_errors, total_words) * 100:.2f}",
                }
            )

            if args.print_samples and count <= args.print_samples:
                logger.info("REF[%s]: %s", label.get("id"), ref_text)
                logger.info("HYP[%s]: %s", label.get("id"), hyp_text)

            if args.limit is not None and count >= args.limit:
                break
    finally:
        if fout is not None:
            fout.close()

    summary = {
        "checkpoint": args.checkpoint,
        "data_lst": args.data_lst,
        "num_samples": count,
        "cer": _safe_rate(total_char_errors, total_chars),
        "wer": _safe_rate(total_word_errors, total_words),
        "char_errors": total_char_errors,
        "chars": total_chars,
        "word_errors": total_word_errors,
        "words": total_words,
        "output_jsonl": str(output_jsonl) if output_jsonl is not None else None,
    }

    logger.info("Samples: %d", count)
    logger.info(
        "CER: %.4f%% (%d / %d)",
        summary["cer"] * 100,
        total_char_errors,
        total_chars,
    )
    logger.info(
        "WER: %.4f%% (%d / %d)",
        summary["wer"] * 100,
        total_word_errors,
        total_words,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate an OmniVoice ASR checkpoint")
    parser.add_argument("--checkpoint", required=True, help="ASR checkpoint directory")
    parser.add_argument("--data_lst", required=True, help="WebDataset data.lst path")
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Optional path to write per-sample predictions as JSONL",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max samples to eval")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, cpu, etc.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Override language id. Defaults to label['language_id'].",
    )
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument(
        "--no_strip_suffix",
        action="store_true",
        help="Disable stripping after known generation suffix markers.",
    )
    parser.add_argument(
        "--stop_marker",
        action="append",
        default=[],
        help="Additional decoded string marker to strip at. Can be repeated.",
    )
    parser.add_argument(
        "--print_samples",
        type=int,
        default=0,
        help="Log the first N reference/hypothesis pairs.",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print batched OmniVoice ASR transcriptions for a WebDataset data.lst."""

import argparse
import json
import logging
from typing import Iterable, Iterator, Optional

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


def _get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_attn_implementation(attn_implementation: str) -> str:
    if attn_implementation != "auto":
        return attn_implementation
    # Batch generation changes sequence length every step. SDPA avoids repeated
    # flex block-mask compilation while still using the model's prefix-LM mask.
    return "sdpa"


def _strip_generation_suffix(text: str, extra_markers: Iterable[str] = ()) -> str:
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


def _iter_samples(data_lst: str) -> Iterator[dict]:
    manifests = webdataset_manifest_reader(data_lst)
    reader = WebDatasetReader(manifests=manifests, evaluation=True)
    return iter(reader)


def _iter_batches(
    samples: Iterable[dict],
    batch_size: int,
    limit: Optional[int],
) -> Iterator[list[dict]]:
    batch = []
    count = 0
    for sample in samples:
        if limit is not None and count >= limit:
            break
        batch.append(sample)
        count += 1
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def transcribe(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    device = args.device
    if device == "auto":
        device = _get_best_device()
    dtype = _resolve_dtype(args.dtype, device)
    attn_implementation = _resolve_attn_implementation(args.attn_implementation)
    if attn_implementation == "flex_attention":
        patch_flex_attention_for_sm12()

    if device == "cpu":
        logger.warning("CPU transcription may be very slow.")

    logger.info(
        "Loading checkpoint: %s (device=%s, dtype=%s, attn_implementation=%s)",
        args.checkpoint,
        device,
        dtype,
        attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = OmniVoiceForSpeechRecognition.from_pretrained(
        args.checkpoint,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    progress = tqdm(
        total=args.limit,
        disable=args.no_progress,
        unit="sample",
    )
    try:
        for batch in _iter_batches(
            _iter_samples(args.data_lst),
            args.batch_size,
            args.limit,
        ):
            labels = [sample["label"] for sample in batch]
            audio_tokens = [sample["audio_tokens"] for sample in batch]
            languages = [
                args.language
                if args.language is not None
                else label.get("language_id")
                for label in labels
            ]

            texts = model.generate_text_batch(
                audio_tokens=audio_tokens,
                tokenizer=tokenizer,
                languages=languages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            for label, language, text in zip(labels, languages, texts):
                if not args.no_strip_suffix:
                    text = _strip_generation_suffix(text, args.stop_marker)

                if args.plain:
                    print(text, flush=True)
                    continue

                item = {
                    "id": label.get("id"),
                    "language_id": language,
                    "text": text,
                }
                if args.include_reference:
                    item["reference"] = label.get("text")
                print(json.dumps(item, ensure_ascii=False), flush=True)

            progress.update(len(batch))
    finally:
        progress.close()


def main():
    parser = argparse.ArgumentParser(
        description="Print batched OmniVoice ASR transcriptions for a WebDataset data.lst"
    )
    parser.add_argument("--checkpoint", required=True, help="ASR checkpoint directory")
    parser.add_argument("--data_lst", required=True, help="WebDataset data.lst path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Max samples to transcribe")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, mps, cpu, etc.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="auto",
        choices=["auto", "flex_attention", "sdpa", "eager"],
        help="Attention implementation. auto uses sdpa for batch generation.",
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
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print only transcript text, one line per sample.",
    )
    parser.add_argument(
        "--include_reference",
        action="store_true",
        help="Include label['text'] in JSONL output for quick inspection.",
    )
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
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress on stderr.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    transcribe(args)


if __name__ == "__main__":
    main()

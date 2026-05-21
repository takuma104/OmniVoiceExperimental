#!/usr/bin/env python3
"""Print batched OmniVoice ASR transcriptions for a WebDataset data.lst."""

import argparse
import json
import logging
from typing import Any, Iterable, Iterator, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
from omnivoice.models.omnivoice_asr import OmniVoiceForSpeechRecognition
from omnivoice.utils.flex_attention_patch import patch_flex_attention_limited_smem
from accelerate.utils import set_seed

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


def _sample_audio_length(sample: dict) -> int:
    audio_tokens = sample["audio_tokens"]
    if audio_tokens.dim() == 3 and audio_tokens.size(0) == 1:
        audio_tokens = audio_tokens.squeeze(0)
    return int(audio_tokens.size(-1))


def _iter_windows(
    samples: Iterable[dict],
    window_size: int,
    limit: Optional[int],
) -> Iterator[list[dict[str, Any]]]:
    window = []
    count = 0
    for sample in samples:
        if limit is not None and count >= limit:
            break
        window.append(
            {
                "order": count,
                "sample": sample,
                "audio_length": _sample_audio_length(sample),
            }
        )
        count += 1
        if len(window) >= window_size:
            yield window
            window = []
    if window:
        yield window


def _iter_batch_chunks(
    window: list[dict[str, Any]],
    batch_size: int,
    use_bucketing: bool,
) -> Iterator[list[dict[str, Any]]]:
    items = window
    if use_bucketing:
        items = sorted(window, key=lambda item: item["audio_length"])
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def transcribe(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )

    set_seed(args.seed)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.bucket_size < 0:
        raise ValueError("--bucket_size must be >= 0")

    use_bucketing = args.bucket_size > args.batch_size
    window_size = args.bucket_size if use_bucketing else args.batch_size

    device = args.device
    if device == "auto":
        device = _get_best_device()
    dtype = _resolve_dtype(args.dtype, device)
    attn_implementation = _resolve_attn_implementation(args.attn_implementation)
    if attn_implementation == "flex_attention":
        patch_flex_attention_limited_smem()

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

    if args.output_jsonl is not None:
        output_file = open(args.output_jsonl, "w", encoding="utf-8")
        logger.info("Writing predictions to %s", args.output_jsonl)
    else:
        output_file = None

    progress = tqdm(
        total=args.limit,
        disable=args.no_progress,
        unit="sample",
    )
    try:
        for window in _iter_windows(
            _iter_samples(args.data_lst),
            window_size,
            args.limit,
        ):
            window_outputs: dict[int, tuple[dict, Optional[str], str]] = {}
            for batch_items in _iter_batch_chunks(
                window,
                args.batch_size,
                use_bucketing,
            ):
                samples = [item["sample"] for item in batch_items]
                labels = [sample["label"] for sample in samples]
                audio_tokens = [sample["audio_tokens"] for sample in samples]
                languages = [
                    args.language
                    if args.language is not None
                    else label.get("language_id")
                    for label in labels
                ]
                source_texts = None
                if args.task_mode == "furigana_rewrite":
                    source_texts = []
                    for label in labels:
                        if args.source_text_field not in label:
                            raise KeyError(
                                f"Sample {label.get('id', '?')!r} is missing "
                                f"source text field {args.source_text_field!r}."
                            )
                        source_texts.append(label[args.source_text_field])

                texts = model.generate_text_batch(
                    audio_tokens=audio_tokens,
                    tokenizer=tokenizer,
                    languages=languages,
                    task_mode=args.task_mode,
                    source_texts=source_texts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    use_cache=not args.no_kv_cache,
                )

                for item, label, language, text in zip(
                    batch_items,
                    labels,
                    languages,
                    texts,
                ):
                    if not args.no_strip_suffix:
                        text = _strip_generation_suffix(text, args.stop_marker)
                    window_outputs[item["order"]] = (label, language, text)

                progress.update(len(batch_items))

            for order in sorted(window_outputs):
                label, language, text = window_outputs[order]
                if args.plain:
                    if output_file is not None:
                        print(text, file=output_file, flush=True)
                    else:
                        print(text, flush=True)
                    continue

                item = {
                    "id": label.get("id"),
                    "language_id": language,
                    "task_mode": args.task_mode,
                    "text": text,
                }
                if args.include_reference:
                    reference_field = args.reference_field
                    if reference_field is None:
                        reference_field = (
                            "text"
                            if args.task_mode == "plain"
                            else args.furigana_text_field
                        )
                    item["reference"] = label.get(reference_field)
                    item["reference_field"] = reference_field
                if args.task_mode == "furigana_rewrite":
                    item["source_text"] = label.get(args.source_text_field)
                if output_file is not None:
                    print(json.dumps(item, ensure_ascii=False), file=output_file, flush=True)
                else:
                    print(json.dumps(item, ensure_ascii=False), flush=True)
    finally:
        progress.close()
        if output_file is not None:
            output_file.close()

def main():
    parser = argparse.ArgumentParser(
        description="Print batched OmniVoice ASR transcriptions for a WebDataset data.lst"
    )
    parser.add_argument("--checkpoint", required=True, help="ASR checkpoint directory")
    parser.add_argument("--data_lst", required=True, help="WebDataset data.lst path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=128,
        help=(
            "Sort samples by audio token length within this many input samples "
            "before batching. Set 0 to disable."
        ),
    )
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
        "--task_mode",
        default="plain",
        choices=["plain", "furigana_audio", "furigana_rewrite"],
        help="ASR generation task.",
    )
    parser.add_argument(
        "--source_text_field",
        default="text",
        help="Label field used as prefilled source transcript for furigana_rewrite.",
    )
    parser.add_argument(
        "--furigana_text_field",
        default="text_fugashi",
        help="Default reference field for furigana task outputs.",
    )
    parser.add_argument(
        "--reference_field",
        default=None,
        help="Explicit label field to emit when --include_reference is set.",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print only transcript text, one line per sample.",
    )
    parser.add_argument(
        "--include_reference",
        action="store_true",
        help="Include a reference label field in JSONL output for quick inspection.",
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV-cache generation and use full recomputation per token.",
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility. Affects generation sampling if --temperature > 0.",
    )
    parser.add_argument(
        "--output_jsonl",
        default=None,
        help="Optional path to write per-sample predictions as JSONL",
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    transcribe(args)


if __name__ == "__main__":
    main()

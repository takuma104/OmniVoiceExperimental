#!/usr/bin/env python3
"""Generate OmniVoice ASR transcripts with token-level audio-token timestamps."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import torch
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from omnivoice.cli.visualize_asr_attention import (
    _enhance_attention_for_alignment,
    _get_best_device,
    _resolve_dtype,
)
from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
from omnivoice.models.omnivoice_asr import OmniVoiceForSpeechRecognition

logger = logging.getLogger(__name__)


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


def _audio_duration_seconds(
    label: dict,
    audio_num_tokens: int,
    audio_duration: Optional[float],
    audio_frame_rate: float,
) -> float:
    if audio_duration is not None:
        return float(audio_duration)
    duration = label.get("audio_duration")
    if duration is not None:
        return float(duration)
    return float(audio_num_tokens) / float(audio_frame_rate)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def _repair_empty_attention_rows(
    enhanced_attention: np.ndarray,
    raw_attention: np.ndarray,
) -> np.ndarray:
    repaired = enhanced_attention.astype(np.float32, copy=True)
    empty_rows = repaired.sum(axis=1) <= 0
    if np.any(empty_rows):
        repaired[empty_rows] = _normalize_rows(raw_attention[empty_rows])
    return repaired


def _viterbi_monotonic_centers(
    attention: np.ndarray,
    max_step_ratio: float,
    transition_penalty: float,
) -> list[int]:
    num_tokens, num_audio_tokens = attention.shape
    if num_tokens == 0 or num_audio_tokens == 0:
        return []

    if num_tokens == 1:
        return [int(np.argmax(attention[0]))]

    log_scores = np.log(np.maximum(attention, 1e-12))
    avg_step = max(float(num_audio_tokens - 1) / float(max(num_tokens - 1, 1)), 1.0)
    if max_step_ratio > 0:
        max_step = max(1, int(np.ceil(avg_step * max_step_ratio)))
    else:
        max_step = num_audio_tokens

    backpointers = np.zeros((num_tokens, num_audio_tokens), dtype=np.int32)
    prev = log_scores[0].astype(np.float32)

    for token_idx in range(1, num_tokens):
        current = np.full(num_audio_tokens, -np.inf, dtype=np.float32)
        back = np.zeros(num_audio_tokens, dtype=np.int32)
        for audio_idx in range(num_audio_tokens):
            start = max(0, audio_idx - max_step)
            prev_indices = np.arange(start, audio_idx + 1)
            step = audio_idx - prev_indices
            if transition_penalty > 0:
                penalty = transition_penalty * np.abs(step - avg_step) / avg_step
                candidates = prev[start : audio_idx + 1] - penalty
            else:
                candidates = prev[start : audio_idx + 1]
            best_offset = int(np.argmax(candidates))
            best_prev = int(prev_indices[best_offset])
            current[audio_idx] = log_scores[token_idx, audio_idx] + candidates[
                best_offset
            ]
            back[audio_idx] = best_prev
        prev = current
        backpointers[token_idx] = back

    centers = [0] * num_tokens
    centers[-1] = int(np.argmax(prev))
    for token_idx in range(num_tokens - 1, 0, -1):
        centers[token_idx - 1] = int(backpointers[token_idx, centers[token_idx]])
    return centers


def _peak_centers(attention: np.ndarray) -> list[int]:
    if attention.size == 0:
        return []
    return np.maximum.accumulate(np.argmax(attention, axis=1)).astype(int).tolist()


def _centers_to_spans(
    centers: list[int],
    num_audio_tokens: int,
    min_span_tokens: int,
) -> list[tuple[int, int]]:
    if not centers:
        return []

    boundaries = [0]
    for prev_center, next_center in zip(centers[:-1], centers[1:]):
        boundaries.append(int((prev_center + next_center + 1) // 2))
    boundaries.append(num_audio_tokens)

    spans = []
    for idx in range(len(centers)):
        start = max(0, min(num_audio_tokens, boundaries[idx]))
        end = max(start, min(num_audio_tokens, boundaries[idx + 1]))
        if min_span_tokens > 0 and end < num_audio_tokens:
            end = min(num_audio_tokens, max(end, start + min_span_tokens))
        spans.append((start, end))
    return spans


def _token_confidence(row: np.ndarray, center: int) -> dict:
    if row.size == 0 or row.sum() <= 0:
        return {
            "confidence": 0.0,
            "peak_attention": 0.0,
            "peak_ratio": 0.0,
            "entropy": 1.0,
        }

    probs = row / row.sum()
    peak = float(probs[center])
    if probs.size > 1:
        top2 = np.partition(probs, -2)[-2:]
        peak_ratio = float(top2[-1] / max(top2[-2], 1e-12))
    else:
        peak_ratio = 1.0
    if probs.size > 1:
        entropy = float(
            -np.sum(probs * np.log(np.maximum(probs, 1e-12))) / np.log(probs.size)
        )
    else:
        entropy = 0.0
    confidence = peak * max(0.0, 1.0 - entropy) * min(peak_ratio / 3.0, 1.0)
    return {
        "confidence": float(confidence),
        "peak_attention": peak,
        "peak_ratio": peak_ratio,
        "entropy": entropy,
    }


def _build_timestamp_tokens(
    trace: dict,
    attention: np.ndarray,
    audio_seconds: float,
    args: argparse.Namespace,
) -> list[dict]:
    num_audio_tokens = int(trace["audio_num_tokens"])
    if args.timestamp_method == "peak":
        centers = _peak_centers(attention)
    else:
        centers = _viterbi_monotonic_centers(
            attention=attention,
            max_step_ratio=args.max_step_ratio,
            transition_penalty=args.transition_penalty,
        )
    spans = _centers_to_spans(
        centers=centers,
        num_audio_tokens=num_audio_tokens,
        min_span_tokens=args.min_span_tokens,
    )

    token_items = []
    for idx, (token_id, token_text, query_id, query_text, center, span) in enumerate(
        zip(
            trace["token_ids"],
            trace["token_texts"],
            trace["query_token_ids"],
            trace["query_token_texts"],
            centers,
            spans,
        )
    ):
        start, end = span
        score = _token_confidence(attention[idx], center)
        token_items.append(
            {
                "index": idx,
                "token_id": int(token_id),
                "token": token_text,
                "query_token_id": int(query_id),
                "query_token": query_text,
                "center_audio_token": int(center),
                "start_audio_token": int(start),
                "end_audio_token": int(end),
                "center_sec": float((center + 0.5) * audio_seconds / num_audio_tokens),
                "start_sec": float(start * audio_seconds / num_audio_tokens),
                "end_sec": float(end * audio_seconds / num_audio_tokens),
                **score,
            }
        )
    return token_items


def _build_output_item(
    trace: dict,
    label: dict,
    language: Optional[str],
    args: argparse.Namespace,
) -> dict:
    raw_attention = trace["audio_attention"].numpy()
    enhanced_attention = _enhance_attention_for_alignment(
        attention=raw_attention,
        mode=args.enhance,
        baseline_quantile=args.enhance_baseline_quantile,
        power=args.enhance_power,
        smooth_radius=args.enhance_smooth_radius,
    )
    timestamp_attention = _repair_empty_attention_rows(
        enhanced_attention=enhanced_attention,
        raw_attention=raw_attention,
    )
    audio_seconds = _audio_duration_seconds(
        label=label,
        audio_num_tokens=int(trace["audio_num_tokens"]),
        audio_duration=args.audio_duration,
        audio_frame_rate=args.audio_frame_rate,
    )
    tokens = _build_timestamp_tokens(
        trace=trace,
        attention=timestamp_attention,
        audio_seconds=audio_seconds,
        args=args,
    )

    item = {
        "id": label.get("id"),
        "language_id": language,
        "text": trace["text"],
        "audio_num_tokens": int(trace["audio_num_tokens"]),
        "audio_seconds": audio_seconds,
        "timestamp_span_semantics": "[start_audio_token, end_audio_token)",
        "timestamp_method": args.timestamp_method,
        "layers": args.layers,
        "heads": args.heads,
        "enhance": args.enhance,
        "tokens": tokens,
    }
    if args.include_reference:
        item["reference"] = label.get("text")
    return item


def timestamp_asr(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    set_seed(args.seed)

    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.bucket_size < 0:
        raise ValueError("--bucket_size must be >= 0")
    if args.max_step_ratio < 0:
        raise ValueError("--max_step_ratio must be >= 0")
    if args.transition_penalty < 0:
        raise ValueError("--transition_penalty must be >= 0")
    if args.min_span_tokens < 0:
        raise ValueError("--min_span_tokens must be >= 0")
    if not 0.0 <= args.enhance_baseline_quantile <= 1.0:
        raise ValueError("--enhance_baseline_quantile must be between 0 and 1")
    if args.enhance_smooth_radius < 0:
        raise ValueError("--enhance_smooth_radius must be >= 0")
    if args.enhance_power <= 0:
        raise ValueError("--enhance_power must be > 0")

    use_bucketing = args.bucket_size > args.batch_size
    window_size = args.bucket_size if use_bucketing else args.batch_size

    device = args.device
    if device == "auto":
        device = _get_best_device()
    dtype = _resolve_dtype(args.dtype, device)
    if args.attn_implementation != "eager":
        logger.warning(
            "Timestamp extraction needs attention weights; eager attention is safest."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = OmniVoiceForSpeechRecognition.from_pretrained(
        args.checkpoint,
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_file = output_jsonl.open("w", encoding="utf-8")
    count = 0
    errors = 0
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
            window_outputs: dict[int, dict] = {}
            for batch_items in _iter_batch_chunks(
                window,
                args.batch_size,
                use_bucketing,
            ):
                original_batch_len = len(batch_items)
                samples = [item["sample"] for item in batch_items]
                labels = [sample["label"] for sample in samples]

                valid_batch_items = []
                valid_samples = []
                valid_labels = []
                for item_info, label, sample in zip(batch_items, labels, samples):
                    if "audio_tokens" not in sample:
                        message = (
                            f"Sample {label.get('id')} does not contain "
                            "precomputed audio_tokens."
                        )
                        if not args.continue_on_error:
                            raise ValueError(message)
                        window_outputs[item_info["order"]] = {
                            "id": label.get("id"),
                            "error": message,
                        }
                        errors += 1
                        count += 1
                        continue
                    valid_batch_items.append(item_info)
                    valid_samples.append(sample)
                    valid_labels.append(label)

                batch_items = valid_batch_items
                samples = valid_samples
                labels = valid_labels
                if not batch_items:
                    progress.update(original_batch_len)
                    progress.set_postfix({"written": count, "errors": errors})
                    continue

                languages = [
                    args.language
                    if args.language is not None
                    else label.get("language_id")
                    for label in labels
                ]

                try:
                    traces = model.generate_text_attention_trace_batch(
                        audio_tokens=[sample["audio_tokens"] for sample in samples],
                        tokenizer=tokenizer,
                        languages=languages,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        layers=args.layers,
                        heads=args.heads,
                        include_eos=args.include_eos,
                    )
                except Exception:
                    if not args.continue_on_error:
                        raise
                    if device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    traces = []
                    for sample, label, language in zip(samples, labels, languages):
                        try:
                            trace = model.generate_text_attention_trace(
                                audio_tokens=sample["audio_tokens"],
                                tokenizer=tokenizer,
                                language=language,
                                max_new_tokens=args.max_new_tokens,
                                temperature=args.temperature,
                                layers=args.layers,
                                heads=args.heads,
                                include_eos=args.include_eos,
                            )
                        except Exception as exc:
                            traces.append(
                                {
                                    "error": str(exc),
                                    "id": label.get("id"),
                                }
                            )
                        else:
                            traces.append(trace)

                if len(traces) != len(samples):
                    message = (
                        f"Expected {len(samples)} traces from batch generation, "
                        f"got {len(traces)}."
                    )
                    if not args.continue_on_error:
                        raise RuntimeError(message)
                    traces = [
                        {
                            "id": label.get("id"),
                            "error": message,
                        }
                        for label in labels
                    ]

                for item_info, label, language, trace in zip(
                    batch_items,
                    labels,
                    languages,
                    traces,
                ):
                    try:
                        if "error" in trace:
                            raise RuntimeError(trace["error"])
                        output_item = _build_output_item(
                            trace=trace,
                            label=label,
                            language=language,
                            args=args,
                        )
                    except Exception as exc:
                        errors += 1
                        if not args.continue_on_error:
                            raise
                        output_item = {
                            "id": label.get("id"),
                            "error": str(exc),
                        }
                    window_outputs[item_info["order"]] = output_item
                    count += 1

                progress.update(original_batch_len)
                progress.set_postfix({"written": count, "errors": errors})

            for order in sorted(window_outputs):
                print(
                    json.dumps(
                        window_outputs[order],
                        ensure_ascii=False,
                    ),
                    file=output_file,
                    flush=True,
                )
    finally:
        progress.close()
        output_file.close()

    logger.info("Wrote %d samples to %s (%d errors)", count, args.output_jsonl, errors)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate OmniVoice ASR transcripts with token-level audio-token "
            "timestamps for a WebDataset data.lst"
        )
    )
    parser.add_argument("--checkpoint", required=True, help="ASR checkpoint directory")
    parser.add_argument("--data_lst", required=True, help="WebDataset data.lst path")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help=(
            "Number of samples per attention-trace generation batch. Attention "
            "weights are memory-heavy, so this default is smaller than plain ASR."
        ),
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=64,
        help=(
            "Sort samples by audio token length within this many input samples "
            "before batching. Set 0 to disable."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--layers",
        default="last",
        help="'last', 'all', or comma-separated layer indices such as 8,16,-1",
    )
    parser.add_argument(
        "--heads",
        default=None,
        help="'all' or comma-separated head indices. Defaults to all heads.",
    )
    parser.add_argument("--include_eos", action="store_true")
    parser.add_argument(
        "--timestamp_method",
        default="viterbi",
        choices=["viterbi", "peak"],
        help="Use monotonic Viterbi alignment or row-wise monotonic peaks.",
    )
    parser.add_argument(
        "--max_step_ratio",
        type=float,
        default=8.0,
        help="Max token-to-token audio jump as a multiple of average step. 0 disables.",
    )
    parser.add_argument(
        "--transition_penalty",
        type=float,
        default=0.03,
        help="Penalty for deviating from the average monotonic step.",
    )
    parser.add_argument(
        "--min_span_tokens",
        type=int,
        default=0,
        help="Minimum [start,end) span length unless the span reaches audio end.",
    )
    parser.add_argument(
        "--enhance",
        default="alignment",
        choices=["none", "col_center", "row_zscore", "tfidf", "alignment"],
        help="Postprocess attention before timestamp extraction.",
    )
    parser.add_argument(
        "--enhance_baseline_quantile",
        type=float,
        default=0.5,
        help="Column baseline quantile subtracted by col_center/tfidf/alignment.",
    )
    parser.add_argument(
        "--enhance_power",
        type=float,
        default=1.4,
        help="Power applied after baseline removal. Larger values sharpen peaks.",
    )
    parser.add_argument(
        "--enhance_smooth_radius",
        type=int,
        default=1,
        help="Moving-average radius along the audio-token axis after enhancement.",
    )
    parser.add_argument(
        "--audio_duration",
        type=float,
        default=None,
        help="Override audio duration in seconds for every sample.",
    )
    parser.add_argument(
        "--audio_frame_rate",
        type=float,
        default=25.0,
        help="Fallback audio token frame rate when label audio_duration is absent.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, mps, cpu, etc.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="eager",
        choices=["eager", "sdpa", "flex_attention"],
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
    parser.add_argument("--include_reference", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    timestamp_asr(args)


if __name__ == "__main__":
    main()

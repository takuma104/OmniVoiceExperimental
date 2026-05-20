#!/usr/bin/env python3
"""Evaluate OmniVoice ASR timestamp pointer head with teacher-forced text."""

import argparse
import json
import logging
from pathlib import Path
from statistics import median
from typing import Any, Iterator

import torch
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from omnivoice.cli.timestamp_asr import _audio_duration_seconds, _centers_to_spans
from omnivoice.cli.transcribe_asr import (
    _get_best_device,
    _iter_batch_chunks,
    _iter_windows,
    _resolve_attn_implementation,
    _resolve_dtype,
)
from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
from omnivoice.data.processor import OmniVoiceASRSampleProcessor
from omnivoice.models.omnivoice_asr import OmniVoiceForSpeechRecognition
from omnivoice.training.asr_builder import _load_timestamp_file
from omnivoice.utils.flex_attention_patch import patch_flex_attention_limited_smem

logger = logging.getLogger(__name__)


def _iter_samples(data_lst: str) -> Iterator[dict]:
    manifests = webdataset_manifest_reader(data_lst)
    reader = WebDatasetReader(manifests=manifests, evaluation=True)
    return iter(reader)


def _collate_padded(
    processed_samples: list[dict[str, Any]],
    pad_token_id: int,
    num_channels: int,
) -> dict[str, torch.Tensor]:
    batch_size = len(processed_samples)
    max_len = max(int(sample["length"]) for sample in processed_samples)

    input_ids = torch.full(
        (batch_size, num_channels, max_len),
        int(pad_token_id),
        dtype=torch.long,
    )
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    timestamp_center_labels = torch.full(
        (batch_size, max_len),
        -100,
        dtype=torch.long,
    )
    audio_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    text_causal_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    document_ids = torch.full((batch_size, max_len), -1, dtype=torch.int32)
    position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)

    for idx, sample in enumerate(processed_samples):
        length = int(sample["length"])
        input_ids[idx, :, :length] = sample["input_ids"]
        labels[idx, :length] = sample["labels"]
        audio_mask[idx, :length] = sample["audio_mask"]
        text_causal_mask[idx, :length] = sample["text_causal_mask"]
        document_ids[idx, :length] = 0
        position_ids[idx, :length] = torch.arange(length, dtype=torch.long)
        if "timestamp_center_labels" in sample:
            timestamp_center_labels[idx, :length] = sample["timestamp_center_labels"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "timestamp_center_labels": timestamp_center_labels,
        "audio_mask": audio_mask,
        "text_causal_mask": text_causal_mask,
        "document_ids": document_ids,
        "position_ids": position_ids,
    }


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _predict_batch_centers(
    model: OmniVoiceForSpeechRecognition,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(**batch, return_hidden_states=True)
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states.")

    labels = batch["timestamp_center_labels"]
    predictions = torch.full_like(labels, -100)
    confidences = torch.zeros(labels.shape, dtype=torch.float32, device=labels.device)
    scale = model.timestamp_query_proj.out_features**-0.5

    for batch_idx in range(hidden_states.size(0)):
        audio_positions = torch.nonzero(
            (batch["document_ids"][batch_idx] == 0) & batch["audio_mask"][batch_idx],
            as_tuple=False,
        ).squeeze(1)
        query_positions = torch.nonzero(labels[batch_idx] >= 0, as_tuple=False).squeeze(
            1
        )
        if audio_positions.numel() == 0 or query_positions.numel() == 0:
            continue

        query_states = hidden_states[batch_idx, query_positions]
        audio_states = hidden_states[batch_idx, audio_positions]
        query = model.timestamp_query_proj(query_states)
        keys = model.timestamp_audio_key_proj(audio_states)
        logits = torch.matmul(query, keys.transpose(0, 1)) * scale
        probs = torch.softmax(logits.float(), dim=-1)
        confidence, pred = probs.max(dim=-1)
        predictions[batch_idx, query_positions] = pred.to(predictions.dtype)
        confidences[batch_idx, query_positions] = confidence

    return predictions.detach().cpu(), confidences.detach().cpu()


def _build_output_item(
    sample: dict,
    processed: dict[str, Any],
    tokenizer: AutoTokenizer,
    pred_labels: torch.Tensor,
    confidence_labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict, list[int]]:
    label = sample["label"]
    gold_labels = processed["timestamp_center_labels"]
    query_positions = torch.nonzero(gold_labels >= 0, as_tuple=False).squeeze(1)
    audio_num_tokens = int(sample["audio_tokens"].squeeze(0).size(-1))
    audio_seconds = _audio_duration_seconds(
        label=label,
        audio_num_tokens=audio_num_tokens,
        audio_duration=args.audio_duration,
        audio_frame_rate=args.audio_frame_rate,
    )

    predicted_centers = []
    token_items = []
    errors = []
    for token_index, query_pos in enumerate(query_positions.tolist()):
        pred_center = int(pred_labels[query_pos].item())
        gold_center = int(gold_labels[query_pos].item())
        token_id = int(processed["input_ids"][0, query_pos + 1].item())
        predicted_centers.append(pred_center)
        abs_error = abs(pred_center - gold_center)
        errors.append(abs_error)
        token_items.append(
            {
                "index": token_index,
                "token_id": token_id,
                "token": tokenizer.decode([token_id], skip_special_tokens=False),
                "center_audio_token": pred_center,
                "reference_center_audio_token": gold_center,
                "center_error_audio_tokens": abs_error,
                "center_sec": float(
                    (pred_center + 0.5) * audio_seconds / audio_num_tokens
                ),
                "reference_center_sec": float(
                    (gold_center + 0.5) * audio_seconds / audio_num_tokens
                ),
                "pointer_confidence": float(confidence_labels[query_pos].item()),
            }
        )

    spans = _centers_to_spans(
        centers=predicted_centers,
        num_audio_tokens=audio_num_tokens,
        min_span_tokens=args.min_span_tokens,
    )
    for token_item, (start, end) in zip(token_items, spans):
        token_item["start_audio_token"] = int(start)
        token_item["end_audio_token"] = int(end)
        token_item["start_sec"] = float(start * audio_seconds / audio_num_tokens)
        token_item["end_sec"] = float(end * audio_seconds / audio_num_tokens)

    output_item = {
        "id": label.get("id"),
        "language_id": label.get("language_id"),
        "text": label.get("text"),
        "audio_num_tokens": audio_num_tokens,
        "audio_seconds": audio_seconds,
        "timestamp_span_semantics": "[start_audio_token, end_audio_token)",
        "timestamp_method": "center_pointer_teacher_forced",
        "tokens": token_items,
    }
    if args.include_reference:
        output_item["reference"] = label.get("text")
    return output_item, errors


def _summarize(errors: list[int], second_errors: list[float], violations: int, pairs: int):
    if not errors:
        return {
            "num_tokens": 0,
            "center_mae_tokens": None,
            "center_median_abs_error_tokens": None,
            "center_rmse_tokens": None,
            "within_1_token": None,
            "within_2_tokens": None,
            "within_5_tokens": None,
            "within_10_tokens": None,
            "center_mae_seconds": None,
            "monotonic_violation_rate": None,
        }

    n = len(errors)
    return {
        "num_tokens": n,
        "center_mae_tokens": float(sum(errors) / n),
        "center_median_abs_error_tokens": float(median(errors)),
        "center_rmse_tokens": float((sum(e * e for e in errors) / n) ** 0.5),
        "within_1_token": float(sum(e <= 1 for e in errors) / n),
        "within_2_tokens": float(sum(e <= 2 for e in errors) / n),
        "within_5_tokens": float(sum(e <= 5 for e in errors) / n),
        "within_10_tokens": float(sum(e <= 10 for e in errors) / n),
        "center_mae_seconds": float(sum(second_errors) / len(second_errors))
        if second_errors
        else None,
        "monotonic_violation_rate": float(violations / pairs) if pairs else None,
    }


def evaluate_timestamp_pointer(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    set_seed(args.seed)

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.bucket_size < 0:
        raise ValueError("--bucket_size must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")

    use_bucketing = args.bucket_size > args.batch_size
    window_size = args.bucket_size if use_bucketing else args.batch_size

    device = args.device
    if device == "auto":
        device = _get_best_device()
    dtype = _resolve_dtype(args.dtype, device)
    attn_implementation = _resolve_attn_implementation(args.attn_implementation)
    if attn_implementation == "flex_attention":
        patch_flex_attention_limited_smem()

    timestamps = _load_timestamp_file(args.timestamp_path)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = OmniVoiceForSpeechRecognition.from_pretrained(
        args.checkpoint,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    processor = OmniVoiceASRSampleProcessor(
        text_tokenizer=tokenizer,
        num_channels=model.config.num_audio_codebook,
        language_ratio=1.0,
        timestamp_enabled=True,
        timestamp_min_confidence=args.timestamp_min_confidence,
    )

    output_file = None
    if args.output_jsonl is not None:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", encoding="utf-8")

    all_errors: list[int] = []
    all_second_errors: list[float] = []
    monotonic_violations = 0
    monotonic_pairs = 0
    processed_samples = 0
    skipped_samples = 0
    progress = tqdm(total=args.limit, disable=args.no_progress, unit="sample")

    try:
        for window in _iter_windows(
            _iter_samples(args.data_lst),
            window_size,
            args.limit,
        ):
            window_outputs: dict[int, tuple[dict, list[int]]] = {}
            for batch_items in _iter_batch_chunks(
                window,
                args.batch_size,
                use_bucketing,
            ):
                valid_items = []
                processed = []
                raw_samples = []
                for item in batch_items:
                    sample = item["sample"]
                    sample_id = sample.get("label", {}).get("id")
                    if sample_id not in timestamps:
                        skipped_samples += 1
                        continue
                    sample = dict(sample)
                    sample["timestamp"] = timestamps[sample_id]
                    if args.language is not None:
                        label = dict(sample["label"])
                        label["language_id"] = args.language
                        sample["label"] = label
                    processed_sample = processor(sample)
                    if torch.any(processed_sample["timestamp_center_labels"] >= 0):
                        valid_items.append(item)
                        raw_samples.append(sample)
                        processed.append(processed_sample)
                    else:
                        skipped_samples += 1

                if not processed:
                    progress.update(len(batch_items))
                    continue

                batch = _collate_padded(
                    processed,
                    pad_token_id=tokenizer.pad_token_id,
                    num_channels=model.config.num_audio_codebook,
                )
                predictions, confidences = _predict_batch_centers(
                    model,
                    _to_device(batch, device),
                )

                for row_idx, (item, sample, processed_sample) in enumerate(
                    zip(valid_items, raw_samples, processed)
                ):
                    output_item, errors = _build_output_item(
                        sample=sample,
                        processed=processed_sample,
                        tokenizer=tokenizer,
                        pred_labels=predictions[row_idx],
                        confidence_labels=confidences[row_idx],
                        args=args,
                    )
                    centers = [
                        token["center_audio_token"] for token in output_item["tokens"]
                    ]
                    monotonic_violations += sum(
                        next_center < center
                        for center, next_center in zip(centers[:-1], centers[1:])
                    )
                    monotonic_pairs += max(0, len(centers) - 1)
                    audio_seconds = float(output_item["audio_seconds"])
                    audio_num_tokens = int(output_item["audio_num_tokens"])
                    all_errors.extend(errors)
                    all_second_errors.extend(
                        error * audio_seconds / audio_num_tokens for error in errors
                    )
                    window_outputs[item["order"]] = (output_item, errors)
                    processed_samples += 1

                progress.update(len(batch_items))
                progress.set_postfix(
                    {
                        "samples": processed_samples,
                        "tokens": len(all_errors),
                        "skipped": skipped_samples,
                    }
                )

            if output_file is not None:
                for order in sorted(window_outputs):
                    print(
                        json.dumps(window_outputs[order][0], ensure_ascii=False),
                        file=output_file,
                        flush=True,
                    )
    finally:
        progress.close()
        if output_file is not None:
            output_file.close()

    summary = _summarize(
        all_errors,
        all_second_errors,
        monotonic_violations,
        monotonic_pairs,
    )
    summary["num_samples"] = processed_samples
    summary["skipped_samples"] = skipped_samples

    if args.summary_json is not None:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OmniVoice ASR center timestamp pointer head with "
            "teacher-forced reference text."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="ASR checkpoint directory")
    parser.add_argument("--data_lst", required=True, help="WebDataset data.lst path")
    parser.add_argument(
        "--timestamp_path",
        required=True,
        help="Reference timestamp JSON/JSONL generated by timestamp_asr.py",
    )
    parser.add_argument("--output_jsonl", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--bucket_size",
        type=int,
        default=128,
        help="Sort samples by audio token length within this many input samples.",
    )
    parser.add_argument("--timestamp_min_confidence", type=float, default=0.0)
    parser.add_argument("--min_span_tokens", type=int, default=0)
    parser.add_argument("--audio_duration", type=float, default=None)
    parser.add_argument("--audio_frame_rate", type=float, default=25.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attn_implementation",
        default="auto",
        choices=["auto", "flex_attention", "sdpa", "eager"],
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
    )
    parser.add_argument("--language", default=None)
    parser.add_argument("--include_reference", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    evaluate_timestamp_pointer(args)


if __name__ == "__main__":
    main()

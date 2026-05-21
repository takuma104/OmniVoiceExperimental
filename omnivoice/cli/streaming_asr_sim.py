#!/usr/bin/env python3
"""Simulate growing-prefix OmniVoice ASR with timestamp-based commit."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from omnivoice.cli.timestamp_asr import (
    _build_timestamp_tokens,
    _repair_empty_attention_rows,
    _resolve_attn_implementation,
)
from omnivoice.cli.transcribe_asr import (
    _get_best_device,
    _resolve_dtype,
    _strip_generation_suffix,
)
from omnivoice.cli.visualize_asr_attention import _enhance_attention_for_alignment
from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
from omnivoice.models.omnivoice_asr import OmniVoiceForSpeechRecognition
from omnivoice.utils.flex_attention_patch import patch_flex_attention_limited_smem

logger = logging.getLogger(__name__)


def _iter_samples(data_lst: str) -> Iterator[dict]:
    manifests = webdataset_manifest_reader(data_lst)
    reader = WebDatasetReader(manifests=manifests, evaluation=True)
    return iter(reader)


def _read_timestamp_items(
    timestamp_json: str,
    limit: Optional[int],
    sample_id: Optional[str],
) -> list[dict[str, Any]]:
    path = Path(timestamp_json)
    if not path.is_file():
        raise FileNotFoundError(f"Timestamp file does not exist: {timestamp_json}")
    if limit == 0:
        return []

    def accept(item: Any) -> bool:
        return (
            isinstance(item, dict)
            and "id" in item
            and "tokens" in item
            and "error" not in item
            and (sample_id is None or str(item.get("id")) == sample_id)
        )

    items = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not accept(item):
                    continue
                items.append(item)
                if limit is not None and len(items) >= limit:
                    break
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            if "id" in data and "tokens" in data:
                candidates = [data]
            elif isinstance(data.get("items"), list):
                candidates = data["items"]
            else:
                candidates = list(data.values())
        else:
            raise ValueError(f"Unsupported timestamp JSON root in {timestamp_json}")

        for item in candidates:
            if not accept(item):
                continue
            items.append(item)
            if limit is not None and len(items) >= limit:
                break

    if sample_id is not None and not items:
        raise ValueError(f"Sample id not found in timestamp file: {sample_id}")
    return items


def _load_samples_by_id(
    data_lst: str,
    sample_ids: set[str],
    no_progress: bool,
) -> dict[str, dict]:
    samples = {}
    progress = tqdm(
        _iter_samples(data_lst),
        total=len(sample_ids),
        disable=no_progress,
        unit="sample",
        desc="Loading codec tokens",
    )
    try:
        for sample in progress:
            label = sample.get("label", {})
            sample_id = str(label.get("id"))
            if sample_id in sample_ids:
                samples[sample_id] = sample
                if len(samples) >= len(sample_ids):
                    break
    finally:
        progress.close()

    missing = sample_ids.difference(samples)
    if missing:
        raise ValueError(
            f"{len(missing)} timestamp ids were not found in data.lst. "
            f"First missing id: {sorted(missing)[0]}"
        )
    return samples


def _resolve_audio_tokens(sample: dict) -> torch.Tensor:
    audio_tokens = sample["audio_tokens"]
    if audio_tokens.dim() == 3 and audio_tokens.size(0) == 1:
        audio_tokens = audio_tokens.squeeze(0)
    if audio_tokens.dim() != 2:
        raise ValueError(f"Expected audio tokens [C,T], got {tuple(audio_tokens.shape)}")
    return audio_tokens.to(dtype=torch.long)


def _audio_seconds(
    label: dict[str, Any],
    audio_num_tokens: int,
    args: argparse.Namespace,
) -> float:
    if args.audio_duration is not None:
        return float(args.audio_duration)
    duration = label.get("audio_duration")
    if duration is not None:
        return float(duration)
    return float(audio_num_tokens) / float(args.audio_frame_rate)


def _prefix_audio_seconds(
    full_audio_seconds: float,
    full_audio_tokens: int,
    prefix_audio_tokens: int,
) -> float:
    if full_audio_tokens <= 0:
        return 0.0
    return full_audio_seconds * float(prefix_audio_tokens) / float(full_audio_tokens)


def _trace_timestamp_tokens(
    trace: dict[str, Any],
    audio_seconds: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    raw_attention = trace["audio_attention"].numpy()
    if args.timestamp_source == "pointer":
        timestamp_attention = raw_attention
    else:
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
    return _build_timestamp_tokens(
        trace=trace,
        attention=timestamp_attention,
        audio_seconds=audio_seconds,
        args=args,
    )


def _iter_prefix_ends(
    audio_num_tokens: int,
    initial_tokens: int,
    chunk_tokens: int,
    limit_steps: Optional[int],
) -> list[int]:
    if audio_num_tokens <= 0:
        return []
    first_end = min(audio_num_tokens, max(1, initial_tokens))
    ends = list(range(first_end, audio_num_tokens + 1, chunk_tokens))
    if not ends or ends[-1] != audio_num_tokens:
        ends.append(audio_num_tokens)
    if limit_steps is not None:
        ends = ends[:limit_steps]
    return ends


def _stable_prefix_length(hypotheses: list[list[int]]) -> int:
    if not hypotheses:
        return 0
    limit = min(len(item) for item in hypotheses)
    stable = 0
    for idx in range(limit):
        token_id = hypotheses[0][idx]
        if any(item[idx] != token_id for item in hypotheses[1:]):
            break
        stable += 1
    return stable


def _starts_with(values: list[int], prefix: list[int]) -> bool:
    return len(values) >= len(prefix) and values[: len(prefix)] == prefix


def _eligible_prefix_length(
    tokens: list[dict[str, Any]],
    commit_boundary_token: int,
    commit_field: str,
    min_confidence: float,
) -> int:
    count = 0
    for token in tokens:
        if int(token.get(commit_field, 0)) > commit_boundary_token:
            break
        if min_confidence > 0 and float(token.get("confidence", 0.0)) < min_confidence:
            break
        count += 1
    return count


def _decode_ids(tokenizer: AutoTokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _generate_prefix_traces(
    model: OmniVoiceForSpeechRecognition,
    tokenizer: AutoTokenizer,
    prefixes: list[torch.Tensor],
    language: Optional[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    languages = [language] * len(prefixes)
    if args.timestamp_source == "pointer":
        return model.generate_text_pointer_trace_batch(
            audio_tokens=prefixes,
            tokenizer=tokenizer,
            languages=languages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            include_eos=args.include_eos,
        )
    return model.generate_text_attention_trace_batch(
        audio_tokens=prefixes,
        tokenizer=tokenizer,
        languages=languages,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        layers=args.layers,
        heads=args.heads,
        include_eos=args.include_eos,
    )


def _simulate_sample(
    model: OmniVoiceForSpeechRecognition,
    tokenizer: AutoTokenizer,
    sample: dict,
    timestamp_item: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    label = sample.get("label", {})
    sample_id = label.get("id", timestamp_item.get("id"))
    language = args.language if args.language is not None else label.get("language_id")
    audio_tokens = _resolve_audio_tokens(sample)
    audio_num_tokens = int(audio_tokens.size(1))
    full_audio_seconds = _audio_seconds(label, audio_num_tokens, args)
    prefix_ends = _iter_prefix_ends(
        audio_num_tokens=audio_num_tokens,
        initial_tokens=args.initial_tokens,
        chunk_tokens=args.chunk_tokens,
        limit_steps=args.limit_steps,
    )

    committed_ids: list[int] = []
    hypothesis_history: list[list[int]] = []
    commit_events: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    prefix_mismatches = 0

    for batch_start in range(0, len(prefix_ends), args.prefix_batch_size):
        batch_ends = prefix_ends[batch_start : batch_start + args.prefix_batch_size]
        prefixes = [audio_tokens[:, :end] for end in batch_ends]
        traces = _generate_prefix_traces(
            model=model,
            tokenizer=tokenizer,
            prefixes=prefixes,
            language=language,
            args=args,
        )
        if len(traces) != len(prefixes):
            raise RuntimeError(
                f"Expected {len(prefixes)} prefix traces, got {len(traces)}."
            )

        for batch_offset, (audio_end_token, trace) in enumerate(zip(batch_ends, traces)):
            step_index = batch_start + batch_offset
            prefix_seconds = _prefix_audio_seconds(
                full_audio_seconds=full_audio_seconds,
                full_audio_tokens=audio_num_tokens,
                prefix_audio_tokens=audio_end_token,
            )
            tokens = _trace_timestamp_tokens(
                trace=trace,
                audio_seconds=prefix_seconds,
                args=args,
            )
            token_ids = [int(token["token_id"]) for token in tokens]
            hypothesis_history.append(token_ids)
            if len(hypothesis_history) > args.stable_steps:
                hypothesis_history = hypothesis_history[-args.stable_steps :]

            commit_boundary = max(0, audio_end_token - args.guard_tokens)
            eligible_count = _eligible_prefix_length(
                tokens=tokens,
                commit_boundary_token=commit_boundary,
                commit_field=args.commit_field,
                min_confidence=args.min_confidence,
            )
            stable_count = (
                _stable_prefix_length(hypothesis_history)
                if len(hypothesis_history) >= args.stable_steps
                else 0
            )
            target_count = min(eligible_count, stable_count)
            new_commit_ids: list[int] = []

            if not _starts_with(token_ids, committed_ids):
                prefix_mismatches += 1
            elif target_count > len(committed_ids):
                new_commit_ids = token_ids[len(committed_ids) : target_count]
                committed_ids.extend(new_commit_ids)
                committed_text = _decode_ids(tokenizer, committed_ids)
                commit_events.append(
                    {
                        "step": step_index,
                        "audio_end_token": int(audio_end_token),
                        "audio_end_sec": float(prefix_seconds),
                        "commit_boundary_token": int(commit_boundary),
                        "commit_token_range": [
                            int(target_count - len(new_commit_ids)),
                            int(target_count),
                        ],
                        "new_token_ids": [int(token_id) for token_id in new_commit_ids],
                        "new_text": _decode_ids(tokenizer, new_commit_ids),
                        "committed_text": committed_text,
                    }
                )

            step_item = {
                "step": step_index,
                "audio_end_token": int(audio_end_token),
                "audio_end_sec": float(prefix_seconds),
                "commit_boundary_token": int(commit_boundary),
                "generated_text": _strip_generation_suffix(trace.get("text", "")),
                "generated_token_count": len(token_ids),
                "eligible_token_count": int(eligible_count),
                "stable_token_count": int(stable_count),
                "committed_token_count": len(committed_ids),
                "new_commit_token_count": len(new_commit_ids),
            }
            if args.include_step_tokens:
                step_item["tokens"] = tokens
            if args.include_steps:
                steps.append(step_item)

    output = {
        "id": sample_id,
        "language_id": language,
        "audio_num_tokens": audio_num_tokens,
        "audio_seconds": full_audio_seconds,
        "chunk_tokens": args.chunk_tokens,
        "initial_tokens": args.initial_tokens,
        "guard_tokens": args.guard_tokens,
        "stable_steps": args.stable_steps,
        "commit_field": args.commit_field,
        "timestamp_source": args.timestamp_source,
        "committed_text": _decode_ids(tokenizer, committed_ids),
        "committed_token_ids": [int(token_id) for token_id in committed_ids],
        "committed_token_count": len(committed_ids),
        "commit_events": commit_events,
        "prefix_mismatches": prefix_mismatches,
        "reference": label.get("text"),
        "reference_timestamp_text": timestamp_item.get("text"),
        "reference_timestamp_token_count": len(timestamp_item.get("tokens") or []),
    }
    if args.include_steps:
        output["steps"] = steps
    return output


def streaming_asr_sim(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    set_seed(args.seed)

    if args.chunk_tokens < 1:
        raise ValueError("--chunk_tokens must be >= 1")
    if args.initial_tokens < 1:
        raise ValueError("--initial_tokens must be >= 1")
    if args.guard_tokens < 0:
        raise ValueError("--guard_tokens must be >= 0")
    if args.stable_steps < 1:
        raise ValueError("--stable_steps must be >= 1")
    if args.prefix_batch_size < 1:
        raise ValueError("--prefix_batch_size must be >= 1")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    if args.limit_steps is not None and args.limit_steps < 1:
        raise ValueError("--limit_steps must be >= 1")
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

    timestamp_items = _read_timestamp_items(
        timestamp_json=args.timestamp_json,
        limit=args.limit,
        sample_id=args.sample_id,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not timestamp_items:
        output_path.write_text("", encoding="utf-8")
        logger.info("No timestamp items selected; wrote empty file to %s", output_path)
        return

    sample_ids = {str(item["id"]) for item in timestamp_items}
    samples_by_id = _load_samples_by_id(
        data_lst=args.data_lst,
        sample_ids=sample_ids,
        no_progress=args.no_progress,
    )

    device = args.device
    if device == "auto":
        device = _get_best_device()
    dtype = _resolve_dtype(args.dtype, device)
    attn_implementation = _resolve_attn_implementation(
        args.attn_implementation,
        args.timestamp_source,
    )
    if attn_implementation == "flex_attention":
        patch_flex_attention_limited_smem()

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

    errors = 0
    progress = tqdm(
        timestamp_items,
        disable=args.no_progress,
        unit="sample",
        desc="Streaming simulation",
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for timestamp_item in progress:
            sample_id = str(timestamp_item["id"])
            try:
                output_item = _simulate_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample=samples_by_id[sample_id],
                    timestamp_item=timestamp_item,
                    args=args,
                )
            except Exception as exc:
                errors += 1
                if not args.continue_on_error:
                    raise
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                output_item = {"id": timestamp_item.get("id"), "error": str(exc)}

            print(
                json.dumps(output_item, ensure_ascii=False),
                file=output_file,
                flush=True,
            )
            progress.set_postfix({"errors": errors})

    logger.info("Wrote %d samples to %s (%d errors)", len(timestamp_items), output_path, errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate growing-prefix OmniVoice ASR and commit stable tokens using "
            "generated token timestamps."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="ASR checkpoint directory")
    parser.add_argument("--data_lst", required=True, help="WebDataset data.lst path")
    parser.add_argument(
        "--timestamp_json",
        "--timestamp_jsonl",
        dest="timestamp_json",
        required=True,
        help="Reference timestamp JSON/JSONL used to select samples.",
    )
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to process")
    parser.add_argument("--sample_id", default=None, help="Process one sample id")
    parser.add_argument(
        "--chunk_tokens",
        type=int,
        default=5,
        help="Streaming audio chunk size in codec tokens. 5 tokens is 200 ms at 25 Hz.",
    )
    parser.add_argument(
        "--initial_tokens",
        type=int,
        default=25,
        help="First prefix size in codec tokens before starting decode.",
    )
    parser.add_argument(
        "--guard_tokens",
        type=int,
        default=15,
        help="Do not commit tokens whose timestamp is within this many tokens of the prefix end.",
    )
    parser.add_argument(
        "--stable_steps",
        type=int,
        default=2,
        help="Require the same token prefix across this many consecutive hypotheses.",
    )
    parser.add_argument(
        "--commit_field",
        default="end_audio_token",
        choices=["start_audio_token", "center_audio_token", "end_audio_token"],
        help="Timestamp field used for the guard-based commit boundary.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="Minimum timestamp confidence for a token to be commit-eligible.",
    )
    parser.add_argument(
        "--prefix_batch_size",
        type=int,
        default=1,
        help="Offline batching for prefix hypotheses. Keep 1 for closest real-time simulation.",
    )
    parser.add_argument(
        "--limit_steps",
        type=int,
        default=None,
        help="Optional max number of streaming prefix updates per sample.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--timestamp_source",
        default="attention",
        choices=["attention", "pointer"],
        help="Use attention alignment or the trained timestamp pointer head.",
    )
    parser.add_argument(
        "--layers",
        default="19",
        help="'last', 'all', or comma-separated layer indices such as 19 or 8,16,-1.",
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
        help="Override full audio duration in seconds for every sample.",
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
        default="auto",
        choices=["auto", "eager", "sdpa", "flex_attention"],
        help="auto uses eager for attention timestamps and sdpa for pointer timestamps.",
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
        "--include_steps",
        action="store_true",
        help="Include every streaming hypothesis summary in the output JSONL.",
    )
    parser.add_argument(
        "--include_step_tokens",
        action="store_true",
        help="When --include_steps is set, include per-token timestamp rows for every step.",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    streaming_asr_sim(args)


if __name__ == "__main__":
    main()

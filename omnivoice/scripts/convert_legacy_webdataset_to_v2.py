#!/usr/bin/env python3
"""Convert legacy OmniVoice WebDataset shards to self-contained v2 shards.

Legacy OmniVoice data uses a tar shard for audio tokens plus a sidecar JSONL
file for text/metadata. This script writes pure WebDataset shards containing
audio tokens, tokenized text ids, metadata, and optional ASR timestamps in the
same tar sample.

Examples:
    python -m omnivoice.scripts.convert_legacy_webdataset_to_v2 \
        --input_data_config output_asr/run0/data.json \
        --output_dir data/asr_v2 \
        --text_tokenizer_path output_asr/run0/checkpoint-1000

    python -m omnivoice.scripts.convert_legacy_webdataset_to_v2 \
        --input_manifest data/train/data.lst \
        --output_dir data/train_v2 \
        --split train \
        --language_id ja \
        --text_tokenizer_path Qwen/Qwen3-0.6B
"""

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import webdataset as wds
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
from omnivoice.utils.common import str2bool


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _numpy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _tokenize_text(tokenizer: Any, text: str) -> np.ndarray:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return np.asarray(ids, dtype=np.int32)


def _load_timestamp_file(timestamp_path: str) -> dict[str, dict]:
    path = Path(timestamp_path)
    if not path.is_file():
        raise FileNotFoundError(f"Timestamp file does not exist: {timestamp_path}")

    timestamps: dict[str, dict] = {}
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if "id" in item and "tokens" in item:
                    timestamps[item["id"]] = item
        return timestamps

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "id" in data and "tokens" in data:
            items = [data]
        elif "items" in data and isinstance(data["items"], list):
            items = data["items"]
        else:
            items = list(data.values())
    else:
        raise ValueError(f"Unsupported timestamp JSON root in {timestamp_path}")

    for item in items:
        if isinstance(item, dict) and "id" in item and "tokens" in item:
            timestamps[item["id"]] = item
    return timestamps


def _resolve_path(path: str, base_dir: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)
    return str(base_dir / path)


def _url_for_path(path: Path, output_root: Path, url_prefix: Optional[str]) -> str:
    if url_prefix:
        rel = path.resolve().relative_to(output_root.resolve()).as_posix()
        return f"{url_prefix.rstrip('/')}/{rel}"
    return str(path.resolve())


def _normalise_audio_tokens(audio_tokens: Any) -> np.ndarray:
    tokens = _to_numpy(audio_tokens)
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)
    if tokens.ndim != 2:
        raise ValueError(f"Expected audio tokens [C,T], got shape {tokens.shape}")
    return tokens.astype(np.int16, copy=False)


def _metadata_for_v2(
    label: dict[str, Any],
    audio_tokens: np.ndarray,
    store_raw_text: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in label.items():
        if key in {"text_ids", "text_pinyin_ids"}:
            continue
        if not store_raw_text and key in {"text", "text_pinyin"}:
            continue
        if isinstance(value, torch.Tensor):
            value = value.item() if value.ndim == 0 else value.detach().cpu().tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        metadata[key] = value

    metadata.setdefault("id", str(label.get("id", "")))
    metadata["format_version"] = 2
    metadata["num_audio_tokens"] = int(audio_tokens.shape[1])
    metadata["num_audio_codebook"] = int(audio_tokens.shape[0])
    return metadata


def convert_manifests_to_v2(
    manifests: list[tuple[str, str, int, float]],
    output_dir: Path,
    output_root: Path,
    tokenizer: Any,
    samples_per_shard: int,
    store_raw_text: bool,
    url_prefix: Optional[str],
    language_id: Optional[str] = None,
    repeat: int = 1,
    timestamps: Optional[dict[str, dict]] = None,
    s3_url_mode: Optional[str] = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_pattern = output_dir / "shard-%06d.tar"
    timestamps = timestamps or {}

    reader = WebDatasetReader(manifests=manifests, evaluation=True)
    tar_writer = None
    shard_idx = 0
    shard_sample_count = 0
    total_samples = 0
    total_seconds = 0.0
    timestamp_count = 0
    shard_urls: list[str] = []

    def open_new_shard():
        nonlocal tar_writer, shard_idx, shard_sample_count
        if tar_writer is not None:
            tar_writer.close()
        tar_path = Path(str(shard_pattern) % shard_idx)
        tar_writer = wds.TarWriter(str(tar_path))
        shard_urls.append(_url_for_path(tar_path, output_root, url_prefix))
        shard_idx += 1
        shard_sample_count = 0

    try:
        total_hint = sum(item[2] for item in manifests)
        for sample in tqdm(reader, total=total_hint or None, desc=str(output_dir)):
            label = dict(sample["label"])
            sample_id = str(label.get("id") or "")
            if not sample_id:
                raise ValueError(f"Sample is missing label.id: {label!r}")

            if language_id is not None:
                label.setdefault("language_id", language_id)

            audio_tokens = _normalise_audio_tokens(sample["audio_tokens"])
            text = label.get("text")
            if text is None and "text_ids" not in label:
                raise ValueError(f"Sample {sample_id} is missing text")
            text_ids = (
                _to_numpy(label["text_ids"]).astype(np.int32, copy=False)
                if "text_ids" in label
                else _tokenize_text(tokenizer, str(text))
            )

            record = {
                "__key__": sample_id,
                "audio_tokens.npy": _numpy_bytes(audio_tokens),
                "text_ids.npy": _numpy_bytes(text_ids),
            }

            if "text_pinyin_ids" in label:
                text_pinyin_ids = _to_numpy(label["text_pinyin_ids"]).astype(
                    np.int32,
                    copy=False,
                )
                record["text_pinyin_ids.npy"] = _numpy_bytes(text_pinyin_ids)
            elif label.get("text_pinyin") is not None:
                text_pinyin_ids = _tokenize_text(tokenizer, str(label["text_pinyin"]))
                record["text_pinyin_ids.npy"] = _numpy_bytes(text_pinyin_ids)

            metadata = _metadata_for_v2(label, audio_tokens, store_raw_text)
            record["json"] = _json_bytes(metadata)

            timestamp = timestamps.get(sample_id) or sample.get("timestamp")
            if timestamp is not None:
                record["timestamp.json"] = _json_bytes(timestamp)
                timestamp_count += 1

            if tar_writer is None or shard_sample_count >= samples_per_shard:
                open_new_shard()
            tar_writer.write(record)
            shard_sample_count += 1
            total_samples += 1
            total_seconds += float(metadata.get("audio_duration", 0.0) or 0.0)
    finally:
        if tar_writer is not None:
            tar_writer.close()

    item: dict[str, Any] = {
        "format": "webdataset_v2",
        "urls": shard_urls,
        "num_items": total_samples,
        "num_seconds": total_seconds,
        "repeat": repeat,
    }
    if language_id is not None:
        item["language_id"] = language_id
    if s3_url_mode is not None and url_prefix and url_prefix.startswith("s3://"):
        item["s3_url_mode"] = s3_url_mode
    if timestamp_count > 0:
        item["timestamps_embedded"] = True
    return item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_manifest", help="Legacy data.lst to convert.")
    source.add_argument(
        "--input_data_config",
        help="Legacy data_config JSON containing train/dev manifest_path entries.",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument(
        "--output_data_config",
        default=None,
        help="Path for the generated v2 data_config JSON.",
    )
    parser.add_argument(
        "--text_tokenizer_path",
        required=True,
        help="Tokenizer/checkpoint used to convert raw text to token ids.",
    )
    parser.add_argument("--samples_per_shard", type=int, default=1000)
    parser.add_argument("--split", default="train", help="Split for --input_manifest.")
    parser.add_argument("--language_id", default=None)
    parser.add_argument(
        "--timestamp_path",
        default=None,
        help="Optional timestamp JSON/JSONL for --input_manifest.",
    )
    parser.add_argument(
        "--store_raw_text",
        type=str2bool,
        default=True,
        help="Keep raw text/text_pinyin in embedded metadata.",
    )
    parser.add_argument(
        "--url_prefix",
        default=None,
        help="URL prefix for generated data_config, e.g. s3://bucket/dataset.",
    )
    parser.add_argument(
        "--s3_url_mode",
        choices=["native", "awscli_pipe"],
        default="awscli_pipe",
        help="How training should open generated s3:// URLs in data_config.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    args = build_parser().parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_data_config = (
        Path(args.output_data_config)
        if args.output_data_config
        else output_root / "data_v2.json"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer_path)
    data_config: dict[str, list[dict[str, Any]]] = {}

    if args.input_manifest:
        timestamps = (
            _load_timestamp_file(args.timestamp_path) if args.timestamp_path else None
        )
        manifests = webdataset_manifest_reader(args.input_manifest)
        data_config[args.split] = [
            convert_manifests_to_v2(
                manifests=manifests,
                output_dir=output_root / args.split,
                output_root=output_root,
                tokenizer=tokenizer,
                samples_per_shard=args.samples_per_shard,
                store_raw_text=args.store_raw_text,
                url_prefix=args.url_prefix,
                language_id=args.language_id,
                timestamps=timestamps,
                s3_url_mode=args.s3_url_mode,
            )
        ]
    else:
        input_config_path = Path(args.input_data_config)
        with input_config_path.open("r", encoding="utf-8") as f:
            input_config = json.load(f)

        for split in ("train", "dev"):
            if split not in input_config:
                continue
            data_config[split] = []
            for item_index, item in enumerate(input_config[split]):
                manifest_paths = item.get("manifest_path")
                if manifest_paths is None:
                    raise ValueError(
                        "Only legacy manifest_path entries can be converted. "
                        f"Invalid item in {split}: {item!r}"
                    )
                if isinstance(manifest_paths, str):
                    manifest_paths = [manifest_paths]

                manifests = []
                for manifest_path in manifest_paths:
                    resolved = _resolve_path(manifest_path, input_config_path.parent)
                    manifests.extend(webdataset_manifest_reader(resolved))

                timestamp_path = (
                    item.get("timestamp_path")
                    or item.get("timestamp_jsonl")
                    or item.get("timestamp_json")
                )
                timestamps = None
                if timestamp_path:
                    resolved_timestamp_path = _resolve_path(
                        timestamp_path,
                        input_config_path.parent,
                    )
                    timestamps = _load_timestamp_file(resolved_timestamp_path)

                output_dir = output_root / split / f"item-{item_index:03d}"
                data_config[split].append(
                    convert_manifests_to_v2(
                        manifests=manifests,
                        output_dir=output_dir,
                        output_root=output_root,
                        tokenizer=tokenizer,
                        samples_per_shard=args.samples_per_shard,
                        store_raw_text=args.store_raw_text,
                        url_prefix=args.url_prefix,
                        language_id=item.get("language_id"),
                        repeat=int(item.get("repeat", 1)),
                        timestamps=timestamps,
                        s3_url_mode=args.s3_url_mode,
                    )
                )

    output_data_config.parent.mkdir(parents=True, exist_ok=True)
    with output_data_config.open("w", encoding="utf-8") as f:
        json.dump(data_config, f, ensure_ascii=False, indent=4)
    logging.info("Wrote v2 data_config to %s", output_data_config)


if __name__ == "__main__":
    main()

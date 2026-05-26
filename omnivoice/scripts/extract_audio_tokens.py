#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract audio tokens from audio data and pack them into WebDataset shards.

Supports three input modes:

1. WebDataset manifest (data.lst):
    python extract_audio_tokens.py \
        --input_manifest data.lst \
        --tar_output_pattern output/audios/shard-%06d.tar \
        --jsonl_output_pattern output/txts/shard-%06d.jsonl

2. Raw JSONL (each line: {"id": "...", "audio_path": "...", "text": "...", ...}):
    python extract_audio_tokens.py \
        --input_jsonl data.jsonl \
        --tar_output_pattern output/audios/shard-%06d.tar \
        --jsonl_output_pattern output/txts/shard-%06d.jsonl

3. HuggingFace Dataset:
    python extract_audio_tokens.py \
        --dataset_name amphion/Emilia-Dataset \
        --data_files "Emilia/EN/*.tar" \
        --split en \
        --streaming True \
        --tar_output_pattern output/audios/shard-%06d.tar \
        --jsonl_output_pattern output/txts/shard-%06d.jsonl

Output structure:
    output_dir/
    ├── audios/           # WebDataset tar shards (.npy audio tokens + .json metadata)
    │   ├── shard_000000.tar
    │   └── ...
    ├── txts/             # Per-shard JSONL metadata
    │   ├── shard_000000.jsonl
    │   └── ...
    ├── data.lst          # Manifest: <tar_path> <jsonl_path> <sample_count> <total_duration>
    └── errors.jsonl      # Failed samples with error details
"""

import argparse
import io
import json
import logging
import multiprocessing as mp
import os
import re
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import webdataset as wds
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer, HiggsAudioV2TokenizerModel

from omnivoice.data.dataset import (
    JsonlDatasetReader,
    WebDatasetReader,
    load_audio_webdataset,
)
from omnivoice.utils.common import str2bool

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.nn.utils.weight_norm"
)

HIGGS_INPUT_SAMPLE_RATE = 24_000


# Global variables: Store tokenizer and device for each worker process
worker_tokenizer = None
worker_feature_extractor = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_manifest",
        default=None,
        help="Path to input dataset manifest (data.lst).",
    )
    parser.add_argument(
        "--input_jsonl",
        default=None,
        help="Path to raw JSONL file (alternative to --input_manifest).",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="HuggingFace dataset name or path "
        '(e.g. "amphion/Emilia-Dataset").',
    )
    parser.add_argument(
        "--data_files",
        default=None,
        help="Data file pattern or JSON dict for load_dataset "
        '(e.g. "Emilia/EN/*.tar").',
    )
    parser.add_argument(
        "--split",
        default=None,
        help='Dataset split name for HuggingFace input (e.g. "en", "train").',
    )
    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=True,
        help="Use HuggingFace streaming mode.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="HuggingFace token for gated datasets.",
    )
    parser.add_argument(
        "--tar_output_pattern",
        required=True,
        help="Tar shard pattern passed to WebDataset",
    )
    parser.add_argument(
        "--jsonl_output_pattern",
        default=None,
        help="Jsonl shard pattern passed to WebDataset (legacy output only)",
    )
    parser.add_argument(
        "--output_format",
        choices=["legacy", "webdataset_v2"],
        default="legacy",
        help="legacy writes tar + sidecar JSONL; webdataset_v2 embeds metadata "
        "and tokenized text in the tar shard.",
    )
    parser.add_argument(
        "--text_tokenizer_path",
        type=str,
        default=None,
        help="Text tokenizer/checkpoint used for webdataset_v2 text_ids.",
    )
    parser.add_argument(
        "--store_raw_text",
        type=str2bool,
        default=True,
        help="Keep raw text/text_pinyin in webdataset_v2 metadata.",
    )
    parser.add_argument(
        "--data_config_split",
        choices=["train", "dev"],
        default="train",
        help="Split name used in generated webdataset_v2 data_v2.json.",
    )
    parser.add_argument(
        "--language_id",
        default=None,
        help="Optional language_id written to generated webdataset_v2 data_v2.json.",
    )
    parser.add_argument(
        "--samples_per_shard",
        type=int,
        default=1000,
        help="Maximum records per shard",
    )
    parser.add_argument(
        "--min_num_shards",
        type=int,
        default=32,
        help="Minimum number of output shards (use to ensure "
        "shard count >= num_gpu * num_workers)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="eustlb/higgs-audio-v2-tokenizer",
        help="Path to audio tokenizer.",
    )
    parser.add_argument(
        "--skip_errors", action="store_true", help="Skip items that fail to process"
    )
    parser.add_argument(
        "--min_length",
        type=float,
        default=0.0,
        help="Minimum audio duration in seconds (e.g. 2.0)",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=float("inf"),
        help="Maximum audio duration in seconds (e.g. 15.0)",
    )
    parser.add_argument(
        "--num_machines",
        type=int,
        default=1,
        help="Total number of machines for distributed runs",
    )
    parser.add_argument(
        "--machine_index",
        type=int,
        default=0,
        help="Zero-based machine index when distributing across multiple "
        "machines (e.g. 0, 1, ... num_machines-1)",
    )
    parser.add_argument(
        "--nj_per_gpu",
        type=int,
        default=3,
        help="Number of worker processes to spawn per GPU.",
    )
    parser.add_argument(
        "--loader_workers",
        type=int,
        default=24,
        help="Number of DataLoader workers for streaming IterableDataset.",
    )
    parser.add_argument(
        "--shuffle",
        type=str2bool,
        default=True,
        help="Shuffle data by default.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Random seed for shuffle (default: 42).",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=10_000,
        help="Buffer size for HuggingFace streaming shuffle (default: 10000).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing data.lst: skip already-processed sample IDs "
        "and continue appending new shards. Only supported for HF dataset input.",
    )
    return parser


def parse_resume_state(manifest_path: str) -> tuple[int, set[str]]:
    """Parse data.lst and return (next_shard_idx, set of already-processed IDs).

    The shard index is parsed from the tar filename (expects shard-NNNNNN.tar).
    Processed IDs are loaded from every referenced per-shard JSONL file.
    """
    next_idx = 0
    jsonl_paths: list[str] = []
    shard_pattern = re.compile(r"shard-(\d+)")
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                continue
            tar_path, jsonl_path = parts[0], parts[1]
            m = shard_pattern.search(os.path.basename(tar_path))
            if m:
                next_idx = max(next_idx, int(m.group(1)) + 1)
            jsonl_paths.append(jsonl_path)

    processed_ids: set[str] = set()
    for jp in tqdm(jsonl_paths, desc="Loading processed IDs"):
        if not os.path.exists(jp):
            logging.warning(f"Resume: jsonl listed in manifest is missing: {jp}")
            continue
        with open(jp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = rec.get("id")
                if sample_id is not None:
                    processed_ids.add(sample_id)
    return next_idx, processed_ids


def count_lines(path):
    with open(path, "rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b""))


def serialise_numpy(key: str, tokens: np.ndarray) -> dict:
    buffer = io.BytesIO()
    np.save(buffer, tokens)
    return {"__key__": key, "npy": buffer.getvalue()}


def serialise_numpy_bytes(tokens: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, tokens)
    return buffer.getvalue()


def tokenize_text_ids(text_tokenizer, text: str) -> np.ndarray:
    ids = text_tokenizer(text, add_special_tokens=False).input_ids
    return np.asarray(ids, dtype=np.int32)


def build_v2_record(
    key: str,
    audio_tokens_np: np.ndarray,
    metadata: dict[str, Any],
    text_tokenizer,
    store_raw_text: bool,
) -> dict[str, Any]:
    if "text" not in metadata:
        raise ValueError(f"Sample {key} is missing text for webdataset_v2 output.")

    record = {
        "__key__": key,
        "audio_tokens.npy": serialise_numpy_bytes(audio_tokens_np),
        "text_ids.npy": serialise_numpy_bytes(
            tokenize_text_ids(text_tokenizer, str(metadata["text"]))
        ),
    }
    if metadata.get("text_pinyin") is not None:
        record["text_pinyin_ids.npy"] = serialise_numpy_bytes(
            tokenize_text_ids(text_tokenizer, str(metadata["text_pinyin"]))
        )

    cleaned = {
        meta_key: _normalise_value(value)
        for meta_key, value in metadata.items()
        if value is not None
        and (store_raw_text or meta_key not in {"text", "text_pinyin"})
    }
    cleaned["format_version"] = 2
    cleaned["num_audio_tokens"] = int(audio_tokens_np.shape[1])
    cleaned["num_audio_codebook"] = int(audio_tokens_np.shape[0])
    record["json"] = json.dumps(cleaned, ensure_ascii=False).encode("utf-8")
    return record


def process_init(rank_queue, tokenizer_path):
    """
    Initialization function for each worker process.
    Assigns a specific GPU to the process and loads the tokenizer.
    """
    global worker_tokenizer, worker_feature_extractor

    # Configure worker process logging
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d]"
        " [Worker %(process)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    # Get assigned GPU rank
    rank = rank_queue.get()
    # Determine device
    if rank != -1 and torch.cuda.is_available():
        worker_device = torch.device(f"cuda:{rank}")
    else:
        worker_device = torch.device("cpu")

    logging.debug(f"Worker process initialized with device: {worker_device}")
    # Load tokenizer onto the specified device
    worker_feature_extractor = AutoFeatureExtractor.from_pretrained(tokenizer_path)
    worker_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        tokenizer_path, device_map=worker_device
    )
    logging.debug(f"Tokenizer loaded successfully on device {worker_device}")


def process_single_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Single-sample processing function executed in worker processes.
    Skips invalid samples during streaming processing.
    """
    try:
        audio_tensor = sample.get("audio", None)  # shape (1, T)
        if audio_tensor is None:
            raise ValueError("Sample missing 'audio' field")

        with torch.inference_mode():
            key = sample["label"]["id"]
            inputs = worker_feature_extractor(
                raw_audio=audio_tensor.squeeze(0).numpy(),
                sampling_rate=HIGGS_INPUT_SAMPLE_RATE,
                return_tensors="pt",
            ).to(worker_tokenizer.device)
            audio_tokens = worker_tokenizer.encode(
                inputs["input_values"],
            ).audio_codes.squeeze(0)

            assert len(audio_tokens.shape) == 2
            assert audio_tokens.size(0) == 8

            num_tokens = audio_tokens.size(1)
            metadata = sample["label"]
            metadata["num_tokens"] = num_tokens

            # Convert to numpy format for subsequent serialization (int16 to save space)
            audio_tokens_np = audio_tokens.to(torch.int16).cpu().numpy()

            return {
                "status": "success",
                "key": key,
                "audio_tokens": audio_tokens_np,
                "metadata": metadata,
                "error_msg": None,
            }
    except Exception as e:
        sample_id = sample.get("label", {}).get("id", "unknown")
        logging.error(f"Failed to process sample {sample_id}: {e}")
        return {
            "status": "error",
            "key": sample_id,
            "audio_tokens": None,
            "metadata": None,
            "error_msg": str(e),
        }


def _normalise_value(value: Any) -> Any:
    """Convert tensors and NumPy scalars to serialisable Python objects."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _encode_metadata(metadata: dict[str, Any]) -> bytes:
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        cleaned[key] = _normalise_value(value)
    return json.dumps(cleaned, ensure_ascii=False).encode("utf-8")


class HFDatasetAdapter(IterableDataset):
    """Convert HuggingFace samples to the internal audio tokenization format."""

    def __init__(
        self,
        hf_dataset,
        sample_rate: int = HIGGS_INPUT_SAMPLE_RATE,
        normalize_audio: bool = True,
        skip_ids: set[str] | None = None,
    ):
        self.hf_dataset = hf_dataset
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        self.skip_ids = skip_ids or set()

    def __iter__(self):
        # Worker- and node-level sharding is handled by the HF dataset itself
        # (DataLoader workers via auto-sharding of `_ex_iterable`, and nodes via
        # `.shard(...)` applied before wrapping). Adding modulo skipping here
        # would compound with HF's auto-sharding and drop most samples.
        for sample in self.hf_dataset:
            if self.skip_ids:
                sample_id = self._peek_id(sample)
                if sample_id is not None and sample_id in self.skip_ids:
                    continue
            converted = self._convert_sample(sample)
            if converted is not None:
                yield converted

    def _peek_id(self, sample: dict[str, Any]) -> str | None:
        """Extract sample id without decoding audio (for resume-time skipping)."""
        json_meta = sample.get("json")
        if isinstance(json_meta, bytes):
            try:
                json_meta = json.loads(json_meta.decode("utf-8"))
            except Exception:
                json_meta = {}
        elif isinstance(json_meta, str):
            try:
                json_meta = json.loads(json_meta)
            except Exception:
                json_meta = {}
        elif not isinstance(json_meta, dict):
            json_meta = {}
        for field in ("id", "__key__"):
            value = sample.get(field)
            if value is not None:
                return value
            value = json_meta.get(field)
            if value is not None:
                return value
        return None

    def _convert_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Convert an HF dataset sample to {"audio": Tensor, "label": dict}."""
        try:
            waveform = self._decode_audio(sample)
            if waveform is None:
                return None

            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if self.normalize_audio:
                waveform = (waveform / (waveform.abs().max() + 1e-7)) * 0.9

            json_meta = sample.get("json")
            if isinstance(json_meta, bytes):
                json_meta = json.loads(json_meta.decode("utf-8"))
            elif isinstance(json_meta, str):
                json_meta = json.loads(json_meta)
            elif json_meta is None:
                json_meta = {}

            def get_field(key: str, *fallbacks: str) -> Any:
                for field in (key, *fallbacks):
                    value = sample.get(field)
                    if value is not None:
                        return value
                    value = json_meta.get(field)
                    if value is not None:
                        return value
                return None

            sample_id = get_field("id", "__key__") or "unknown"
            audio_duration = waveform.shape[1] / self.sample_rate
            label = {
                "id": sample_id,
                "text": get_field("text"),
                "language_id": get_field("language_id", "language"),
                "audio_duration": audio_duration,
                "speaker": get_field("speaker"),
                "dnsmos": get_field("dnsmos"),
            }

            ignored_keys = {"wav", "mp3", "flac", "ogg", "audio", "__key__", "__url__"}
            for key, value in json_meta.items():
                if key not in label and key not in ignored_keys:
                    label[key] = value

            return {"audio": waveform, "label": label}

        except Exception as exc:
            sample_id = sample.get("id") or sample.get("__key__", "unknown")
            logging.warning(f"Failed to convert HF sample {sample_id}: {exc}")
            return None

    def _decode_audio(self, sample: dict[str, Any]) -> torch.Tensor | None:
        """Decode audio from common HF Dataset and WebDataset-style records."""
        for audio_key in ("mp3", "flac", "wav", "ogg"):
            audio_value = sample.get(audio_key)
            if audio_value is None:
                continue
            if isinstance(audio_value, bytes):
                return load_audio_webdataset(audio_value, sample_rate=self.sample_rate)
            if hasattr(audio_value, "get_all_samples"):
                return self._audio_decoder_to_tensor(audio_value)

        audio_info = sample.get("audio")
        if isinstance(audio_info, dict):
            if "array" in audio_info and audio_info["array"] is not None:
                waveform = torch.tensor(audio_info["array"], dtype=torch.float32)
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                sr = audio_info.get("sampling_rate", self.sample_rate)
                return self._maybe_resample(waveform, sr)
            if "bytes" in audio_info and audio_info["bytes"] is not None:
                return load_audio_webdataset(
                    audio_info["bytes"], sample_rate=self.sample_rate
                )
            if "path" in audio_info and audio_info["path"] is not None:
                waveform, sr = torchaudio.load(audio_info["path"])
                return self._maybe_resample(waveform, sr)

        if hasattr(audio_info, "get_all_samples"):
            return self._audio_decoder_to_tensor(audio_info)

        logging.warning(f"No decodable audio found in sample {sample.get('id', '?')}")
        return None

    def _audio_decoder_to_tensor(self, audio_decoder) -> torch.Tensor:
        audio_samples = audio_decoder.get_all_samples()
        waveform = audio_samples.data
        return self._maybe_resample(waveform, audio_samples.sample_rate)

    def _maybe_resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform


class StreamingLengthFilteredDataset(IterableDataset):
    def __init__(
        self,
        base_iterable,
        min_len: float,
        max_len: float,
        sr: int,
    ):
        self.base_iterable = base_iterable
        self.min_len = min_len
        self.max_len = max_len
        self.sr = sr
        self.filtered_count = 0

    def __iter__(self):
        """Stream samples one by one and filter on the fly."""
        for sample in self.base_iterable:
            try:
                duration = sample["audio"].size(-1) / self.sr
                if self.min_len <= duration <= self.max_len:
                    yield sample
                else:
                    self.filtered_count += 1
                    logging.warning(
                        f"Filtered sample (duration out of range): "
                        f"{sample['label']['id']} ({duration:.2f}s)"
                    )
            except Exception as e:
                logging.warning(f"Skipped invalid sample during streaming: {e}")
                continue


def main() -> None:
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)
    parser = build_parser()
    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)

    # Validate input arguments
    input_modes = [
        bool(args.input_manifest),
        bool(args.input_jsonl),
        bool(args.dataset_name),
    ]
    assert (
        sum(input_modes) == 1
    ), "Exactly one of --input_manifest, --input_jsonl, or --dataset_name must be provided."

    if args.output_format == "legacy" and not args.jsonl_output_pattern:
        raise ValueError("--jsonl_output_pattern is required for legacy output.")
    if args.output_format == "webdataset_v2":
        if args.jsonl_output_pattern:
            logging.warning(
                "--jsonl_output_pattern is ignored when --output_format=webdataset_v2"
            )
        if args.resume:
            raise ValueError("--resume is not supported for webdataset_v2 output.")
        if args.text_tokenizer_path is None:
            raise ValueError(
                "--text_tokenizer_path is required for webdataset_v2 output."
            )

    if args.num_machines > 1:
        assert (
            0 <= args.machine_index < args.num_machines
        ), f"machine_index {args.machine_index} must be in [0, {args.num_machines})"

    # Resolve output paths early so we can detect / parse resume state.
    tar_output_pattern = str(Path(args.tar_output_pattern).expanduser())
    jsonl_output_pattern = (
        str(Path(args.jsonl_output_pattern).expanduser())
        if args.jsonl_output_pattern
        else None
    )
    Path(tar_output_pattern).parent.mkdir(parents=True, exist_ok=True)
    if jsonl_output_pattern:
        Path(jsonl_output_pattern).parent.mkdir(parents=True, exist_ok=True)
    if args.output_format == "legacy" or Path(tar_output_pattern).parent.name in {
        "audios",
        "shards",
    }:
        output_dir = Path(tar_output_pattern).parent.parent
    else:
        output_dir = Path(tar_output_pattern).parent
    error_log_path = str(output_dir / "errors.jsonl")
    manifest_path = (
        str(output_dir / "data.lst")
        if args.output_format == "legacy"
        else str(output_dir / "data_v2.json")
    )

    # Resume handling: parse existing data.lst before opening anything for write.
    resume_skip_ids: set[str] = set()
    resume_next_shard_idx = 0
    manifest_exists = os.path.exists(manifest_path)
    if args.resume:
        if not args.dataset_name:
            raise RuntimeError(
                "--resume is only supported for HuggingFace dataset input "
                "(--dataset_name)."
            )
        if manifest_exists:
            logging.info(f"Resume: parsing existing manifest {manifest_path}")
            resume_next_shard_idx, resume_skip_ids = parse_resume_state(manifest_path)
            logging.info(
                f"Resume: next shard idx = {resume_next_shard_idx}, "
                f"already-processed samples = {len(resume_skip_ids)}"
            )
        else:
            logging.info(
                f"Resume: no existing manifest at {manifest_path}; starting fresh."
            )
    elif manifest_exists:
        raise RuntimeError(
            f"{manifest_path} already exists. Pass --resume to continue from it, "
            f"or remove the file to start over."
        )

    # Build base dataset and count total samples based on input mode
    if args.input_jsonl:
        logging.info(f"Input mode: raw JSONL ({args.input_jsonl})")
        total_samples = count_lines(args.input_jsonl)
        base_dataset = JsonlDatasetReader(
            args.input_jsonl,
            sample_rate=HIGGS_INPUT_SAMPLE_RATE,
            shuffle=args.shuffle,
            shuffle_seed=args.shuffle_seed,
        )
        loader_workers = args.loader_workers
    elif args.input_manifest:
        logging.info(f"Input mode: WebDataset manifest ({args.input_manifest})")
        manifest_num_lines = count_lines(args.input_manifest)
        loader_workers = min(args.loader_workers, manifest_num_lines)
        total_samples = 0
        manifests = []
        with open(args.input_manifest, "r", encoding="utf-8") as f:
            for line_id, line in tqdm(
                enumerate(f),
                total=manifest_num_lines,
                desc="Calculating dataset length",
            ):
                items = line.strip().split(" ")
                tar_path, jsonl_path, num_items, duration = (
                    items[0],
                    items[1],
                    int(items[2]),
                    float(items[3]),
                )
                assert os.path.exists(tar_path), f"File {tar_path} does not exist."
                assert os.path.exists(jsonl_path), f"File {jsonl_path} does not exist."
                assert jsonl_path.endswith(
                    ".jsonl"
                ), f"File {jsonl_path} is not a .jsonl file."
                if (
                    args.num_machines > 1
                    and line_id % args.num_machines != args.machine_index
                ):
                    continue
                total_samples += num_items
                manifests.append((tar_path, jsonl_path, num_items, duration))
        logging.info(
            f"Total shards: {manifest_num_lines}, "
            f"Shards for current index: {len(manifests)}"
        )
        base_dataset = WebDatasetReader(
            manifests=manifests,
            sample_rate=HIGGS_INPUT_SAMPLE_RATE,
            evaluation=True,
        )
    else:
        data_files = args.data_files
        if data_files:
            try:
                data_files = json.loads(data_files)
            except (json.JSONDecodeError, TypeError):
                pass

        logging.info(
            f"Input mode: HuggingFace Dataset ({args.dataset_name}), "
            f"data_files={data_files}, split={args.split}, streaming={args.streaming}"
        )
        load_kwargs: dict[str, Any] = {
            "path": args.dataset_name,
            "streaming": args.streaming,
        }
        if args.split:
            load_kwargs["split"] = args.split
        if data_files is not None:
            if isinstance(data_files, str) and args.split:
                load_kwargs["data_files"] = {args.split: data_files}
            else:
                load_kwargs["data_files"] = data_files
        if args.hf_cache_dir:
            load_kwargs["cache_dir"] = args.hf_cache_dir
        if args.hf_token:
            load_kwargs["token"] = args.hf_token

        hf_dataset = load_dataset(**load_kwargs)

        # When --split is omitted, load_dataset returns a (Iterable)DatasetDict;
        # collapse it to a single split so downstream iteration works.
        chosen_split = args.split
        if args.split is None and hasattr(hf_dataset, "keys"):
            splits = list(hf_dataset.keys())
            assert splits, "Loaded dataset has no splits."
            chosen_split = splits[0]
            if len(splits) > 1:
                logging.warning(
                    f"No --split specified; dataset has multiple splits "
                    f"({splits}). Using '{chosen_split}'."
                )
            else:
                logging.info(
                    f"No --split specified; using only available split: "
                    f"'{chosen_split}'."
                )
            hf_dataset = hf_dataset[chosen_split]

        # Multi-node sharding: do this on the HF dataset itself so we don't
        # compound with HF's auto-sharding across DataLoader workers.
        if args.num_machines > 1:
            hf_dataset = hf_dataset.shard(
                num_shards=args.num_machines,
                index=args.machine_index,
            )
        if args.shuffle:
            if args.streaming:
                hf_dataset = hf_dataset.shuffle(
                    seed=args.shuffle_seed,
                    buffer_size=args.shuffle_buffer_size,
                )
            else:
                hf_dataset = hf_dataset.shuffle(seed=args.shuffle_seed)

        total_samples = None
        if not args.streaming:
            total_samples = len(hf_dataset)
        else:
            try:
                info = hf_dataset.info
                if info and info.splits and chosen_split in info.splits:
                    total_samples = info.splits[chosen_split].num_examples
            except Exception:
                pass
        if total_samples is not None and args.num_machines > 1:
            total_samples = (
                total_samples + args.num_machines - 1 - args.machine_index
            ) // args.num_machines

        if total_samples is None:
            logging.info(
                "Total samples unknown for HuggingFace input; using indeterminate "
                "progress bar"
            )
        else:
            logging.info(f"Estimated total samples: {total_samples}")

        base_dataset = HFDatasetAdapter(
            hf_dataset,
            sample_rate=HIGGS_INPUT_SAMPLE_RATE,
            normalize_audio=True,
            skip_ids=resume_skip_ids,
        )
        loader_workers = args.loader_workers
        # Account for already-processed samples in the progress total so the
        # bar reflects only the remaining work.
        if total_samples is not None and resume_skip_ids:
            total_samples = max(0, total_samples - len(resume_skip_ids))

    # Adjust samples_per_shard if min_num_shards would be violated
    samples_per_shard = args.samples_per_shard
    if total_samples is not None and total_samples > 0:
        estimated_shards = max(
            1, (total_samples + samples_per_shard - 1) // samples_per_shard
        )
        if estimated_shards < args.min_num_shards:
            samples_per_shard = max(1, total_samples // args.min_num_shards)
            logging.info(
                f"Adjusted samples_per_shard from {args.samples_per_shard} to "
                f"{samples_per_shard} to meet min_num_shards={args.min_num_shards} "
                f"(total_samples={total_samples})"
            )

    # Apply length filter and create DataLoader
    filtered_dataset = StreamingLengthFilteredDataset(
        base_iterable=base_dataset,
        min_len=args.min_length,
        max_len=args.max_length,
        sr=HIGGS_INPUT_SAMPLE_RATE,
    )
    dataloader = DataLoader(
        dataset=filtered_dataset,
        batch_size=None,
        num_workers=loader_workers,
        persistent_workers=loader_workers > 0,
        pin_memory=False,
    )

    # Configure multi-GPU multi-process setup
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        logging.warning("No GPUs detected - using CPU for processing")
        num_processes = args.nj_per_gpu
    else:
        num_processes = num_devices * args.nj_per_gpu
    logging.info(
        f"GPU count: {num_devices}, Processes per GPU: {args.nj_per_gpu}, "
        f"Total processes: {num_processes}"
    )

    # Shared GPU rank queue for process assignment
    manager = mp.Manager()
    rank_queue = manager.Queue()
    for rank in list(range(num_devices)) * args.nj_per_gpu:
        rank_queue.put(rank)
    if num_devices == 0:
        for _ in range(num_processes):
            rank_queue.put(-1)

    # Output paths were resolved above (before resume parsing).
    file_open_mode = "a" if args.resume and manifest_exists else "w"
    text_tokenizer = (
        AutoTokenizer.from_pretrained(args.text_tokenizer_path)
        if args.output_format == "webdataset_v2"
        else None
    )

    # Setup error logger (writes to errors.jsonl)
    error_logger = logging.getLogger("error_log")
    error_logger.setLevel(logging.ERROR)
    error_logger.handlers.clear()
    error_fh = logging.FileHandler(
        error_log_path, mode=file_open_mode, encoding="utf-8"
    )
    error_fh.setFormatter(logging.Formatter("%(message)s"))
    error_logger.addHandler(error_fh)

    # Progress and error tracking
    processed_count = 0
    error_count = 0
    write_error_count = 0
    failed_ids = []
    shard_idx = resume_next_shard_idx
    shard_sample_count = 0
    shard_duration = 0.0
    shard_manifest_count = 0
    total_written_duration = 0.0

    tar_writer = None
    jsonl_file = None
    manifest_file = (
        open(manifest_path, file_open_mode, encoding="utf-8")
        if args.output_format == "legacy"
        else None
    )
    v2_urls = []

    def append_shard_to_manifest(idx: int, sample_count: int, duration: float) -> None:
        nonlocal shard_manifest_count, total_written_duration
        tar_path = os.path.abspath(tar_output_pattern % idx)
        if args.output_format == "legacy":
            assert manifest_file is not None and jsonl_output_pattern is not None
            jsonl_path = os.path.abspath(jsonl_output_pattern % idx)
            manifest_file.write(
                f"{tar_path} {jsonl_path} {sample_count} {duration:.3f}\n"
            )
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        else:
            v2_urls.append(tar_path)
        total_written_duration += duration
        shard_manifest_count += 1

    def open_new_shard():
        nonlocal tar_writer, jsonl_file, shard_idx, shard_sample_count, shard_duration
        had_previous = tar_writer is not None
        if tar_writer is not None:
            tar_writer.close()
        if jsonl_file is not None:
            jsonl_file.close()
        # Record manifest for the previous shard
        if had_previous and shard_sample_count > 0:
            prev_idx = shard_idx - 1
            append_shard_to_manifest(prev_idx, shard_sample_count, shard_duration)
        tar_fname = tar_output_pattern % shard_idx
        tar_writer = wds.TarWriter(tar_fname)
        if args.output_format == "legacy":
            assert jsonl_output_pattern is not None
            jsonl_fname = jsonl_output_pattern % shard_idx
            jsonl_file = open(jsonl_fname, "w", encoding="utf-8")
        shard_idx += 1
        shard_sample_count = 0
        shard_duration = 0.0

    def write_sample(key, audio_tokens_np, metadata):
        nonlocal shard_sample_count, write_error_count, shard_duration
        assert tar_writer is not None
        try:
            if args.output_format == "legacy":
                assert jsonl_file is not None
                token_record = serialise_numpy(key, audio_tokens_np)
                json_record = _encode_metadata(metadata)
                tar_writer.write(token_record)
                jsonl_file.write(json_record.decode("utf-8") + "\n")
            else:
                assert text_tokenizer is not None
                tar_writer.write(
                    build_v2_record(
                        key=key,
                        audio_tokens_np=audio_tokens_np,
                        metadata=metadata,
                        text_tokenizer=text_tokenizer,
                        store_raw_text=args.store_raw_text,
                    )
                )
            shard_sample_count += 1
            shard_duration += float(metadata.get("audio_duration", 0.0) or 0.0)
        except Exception as exc:
            write_error_count += 1
            failed_ids.append(key)
            error_logger.error(
                json.dumps({"id": key, "reason": str(exc)}, ensure_ascii=False)
            )
            logging.error(f"Write failed for sample {key}: {exc}")

    def handle_result(result):
        nonlocal processed_count, error_count
        if result["status"] == "success":
            # Rotate shard if needed
            if tar_writer is None or shard_sample_count >= samples_per_shard:
                open_new_shard()
            write_sample(result["key"], result["audio_tokens"], result["metadata"])
            processed_count += 1
        else:
            error_count += 1
            failed_ids.append(result["key"])
            error_logger.error(
                json.dumps(
                    {"id": result["key"], "reason": result["error_msg"]},
                    ensure_ascii=False,
                )
            )
            if not args.skip_errors:
                raise RuntimeError(
                    f"Sample {result['key']} processing failed due "
                    f"to {result['error_msg']} - terminating"
                )
            logging.warning(
                f"Skipping failed sample {result['key']}: {result['error_msg']}"
            )

    main_progress = tqdm(total=total_samples, desc="Extracting Audio Tokens")

    try:
        with ProcessPoolExecutor(
            max_workers=num_processes,
            initializer=process_init,
            initargs=(rank_queue, args.tokenizer_path),
        ) as executor:
            logging.info(f"Submitting tasks... ({num_processes} workers)")
            futures = set()
            max_pending = num_processes * 10

            def drain_completed():
                """Wait for at least one future to complete, process all done."""
                nonlocal futures
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for f in done:
                    futures.discard(f)
                    result = f.result()
                    main_progress.update(1)
                    handle_result(result)
                    main_progress.set_postfix(
                        Samples=processed_count,
                        Errors=error_count,
                    )

            # Stream samples from DataLoader
            for sample in dataloader:
                if len(futures) >= max_pending:
                    drain_completed()

                future = executor.submit(process_single_sample, sample)
                futures.add(future)

            # Process remaining futures
            logging.info("Processing remaining pending samples...")
            while futures:
                drain_completed()

    except Exception:
        logging.error("Critical error during processing", exc_info=True)
        raise
    finally:
        main_progress.close()
        if tar_writer is not None:
            tar_writer.close()
        if jsonl_file is not None:
            jsonl_file.close()
        # Record the last shard in the manifest
        if shard_idx > 0 and shard_sample_count > 0:
            last_idx = shard_idx - 1
            append_shard_to_manifest(last_idx, shard_sample_count, shard_duration)
        if manifest_file is not None:
            manifest_file.close()

        if args.output_format == "webdataset_v2":
            data_item = {
                "format": "webdataset_v2",
                "urls": v2_urls,
                "num_items": max(0, processed_count - write_error_count),
                "num_seconds": total_written_duration,
                "repeat": 1,
            }
            if args.language_id is not None:
                data_item["language_id"] = args.language_id
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({args.data_config_split: [data_item]}, f, indent=4)

    # Output final statistics
    total_failed = error_count + write_error_count
    filtered_and_skipped = (
        total_samples - processed_count - total_failed
        if total_samples is not None
        else "unknown"
    )
    logging.info(
        f"Processing Complete - Successful: {processed_count}, Failed: {total_failed}, "
        f"Filtered/Skipped: {filtered_and_skipped}, Shards written: {shard_idx}"
    )
    logging.info(f"Manifest written to: {manifest_path} ({shard_manifest_count} shards)")
    if total_failed > 0:
        logging.info(f"Error details: {error_log_path}")
    if failed_ids and args.skip_errors:
        logging.warning(
            f"Failed sample IDs (count: {len(failed_ids)}): {failed_ids[:100]}..."
        )
    if write_error_count > 0 and not args.skip_errors:
        raise RuntimeError(
            f"{write_error_count} samples failed to write - check logs for details"
        )


if __name__ == "__main__":
    main()

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import torch
import io
import sys
from tqdm import tqdm
from loguru import logger
import soundfile as sf
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import torchaudio
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from dataclasses import dataclass
from transformers import AutoFeatureExtractor
from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel
import numpy as np
import librosa
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.nn.utils.weight_norm"
)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

HIGGS_INPUT_SAMPLE_RATE = 24_000
MAX_DURATION = 40.0  # seconds

@dataclass
class AudioData:
    row_id: int
    transcribe: str
    audio: np.ndarray
    original_length: int = 0
    duration: float = 0.0


@dataclass
class PreparedBatch:
    """GPU処理用に準備されたバッチ"""

    audio_data_list: list[AudioData]
    batch_audios: list[np.ndarray]


def load_audio(bytes: bytes, target_sr=HIGGS_INPUT_SAMPLE_RATE) -> np.ndarray:
    audio, sr = librosa.load(io.BytesIO(bytes), sr=None, mono=True)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32)


def load_audio_data(row) -> AudioData | None:
    """Load audio from row in a thread-safe manner."""
    try:
        audio = load_audio(
            row["audio_data"], target_sr=HIGGS_INPUT_SAMPLE_RATE
        )
        original_length = audio.shape[0]
        duration = original_length / HIGGS_INPUT_SAMPLE_RATE
        if duration > MAX_DURATION:
            return None
        return AudioData(
            row_id=row["row_id"],
            transcribe=row["transcribe"],
            audio=audio,
            original_length=original_length,
            duration=duration,
        )
    except Exception as e:
        logger.error(f"Error loading audio for row {row['row_id']}: {e}")
        return None


def parquet_reader(
    file_path: Path,
    chunk_size: int,
    output_queue: Queue,
):
    """Stage 1: Parquetファイルを読み込み、チャンクをQueueに投入"""
    with pq.ParquetFile(file_path) as pf:
        for batch in pf.iter_batches(batch_size=chunk_size):
            df = batch.to_pandas()
            rows = [row for _, row in df.iterrows()]
            output_queue.put(rows)
    output_queue.put(None)  # Signal completion


def cpu_processor(
    input_queue: Queue,
    output_queue: Queue,
    total_duration: float,
):
    """Stage 2: オーディオを読み込み、バッチを準備してQueueに投入"""
    while True:
        rows = input_queue.get()
        if rows is None:
            break

        # ThreadPoolExecutorでオーディオを読み込み
        with ThreadPoolExecutor() as executor:
            audio_data_list = list(executor.map(load_audio_data, rows))

        # Filter out failed loads
        audio_data_list = [ad for ad in audio_data_list if ad is not None]

        if not audio_data_list:
            continue

        # Sort by length for efficient batching
        audio_data_list.sort(key=lambda x: x.original_length)

        # Create prepared batches
        # for batch_start in range(0, len(audio_data_list), batch_size):
        #     batch = audio_data_list[batch_start : batch_start + batch_size]
        #     batch_audios = [ad.audio for ad in batch]
        #     prepared = PreparedBatch(
        #         audio_data_list=batch,
        #         batch_audios=batch_audios,
        #     )
        #     output_queue.put(prepared)

        # total_durationを超えないようにバッチを作成
        current_batch = []
        current_duration = 0.0
        for ad in audio_data_list:
            if current_duration + ad.duration > total_duration and current_batch:
                batch_audios = [item.audio for item in current_batch]
                prepared = PreparedBatch(
                    audio_data_list=current_batch,
                    batch_audios=batch_audios,
                )
                output_queue.put(prepared)
                current_batch = []
                current_duration = 0.0
            current_batch.append(ad)
            current_duration += ad.duration

        # Put the last batch if it exists
        if current_batch:
            batch_audios = [item.audio for item in current_batch]
            prepared = PreparedBatch(
                audio_data_list=current_batch,
                batch_audios=batch_audios,
            )
            output_queue.put(prepared)

    output_queue.put(None)  # Signal completion


def process_parquet_file(
    file_path: Path,
    chunk_size: int = 256,
    total_duration: float = 80.0,
) -> pd.DataFrame:
    """3ステージパイプラインでParquetファイルを処理"""
    # Get total rows for progress bar
    with pq.ParquetFile(file_path) as pf:
        total_rows = pf.metadata.num_rows

    # Create queues for pipeline
    parquet_queue: Queue = Queue(maxsize=2)  # Parquet -> CPU
    batch_queue: Queue = Queue(maxsize=16)  # CPU -> GPU

    # Start Stage 1: Parquet reader thread
    reader_thread = Thread(
        target=parquet_reader,
        args=(file_path, chunk_size, parquet_queue),
        daemon=True,
    )
    reader_thread.start()

    # Start Stage 2: CPU processor thread
    processor_thread = Thread(
        target=cpu_processor,
        args=(parquet_queue, batch_queue, total_duration),
        daemon=True,
    )
    processor_thread.start()

    # Stage 3: GPU processing (main thread)
    results = []
    with tqdm(total=total_rows, desc="Processing", unit="rows") as pbar:
        while True:
            prepared = batch_queue.get()

            if prepared is None:
                break

            for raw_audio, ad in zip(prepared.batch_audios, prepared.audio_data_list):
                with torch.inference_mode():
                    inputs = worker_feature_extractor(
                        raw_audio=raw_audio,
                        sampling_rate=HIGGS_INPUT_SAMPLE_RATE,
                        return_tensors="pt",
                    ).to(worker_tokenizer.device)
                    audio_tokens = worker_tokenizer.encode(
                        inputs["input_values"],
                    ).audio_codes.squeeze(0)
                    audio_tokens_np = audio_tokens.to(torch.int16).cpu().numpy()
                    codes = audio_tokens_np.flatten().tolist()
                    results.append([ad.row_id, ad.transcribe, codes])

            pbar.update(len(prepared.audio_data_list))

    reader_thread.join()
    processor_thread.join()

    df = pd.DataFrame(results, columns=["row_id", "transcribe", "audio_codes"])
    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python encode_audio_tokens_to_parquet.py [source_directory] [dst_directory]"
        )
        sys.exit(1)
    else:
        tokenizer_path = "eustlb/higgs-audio-v2-tokenizer"
        worker_feature_extractor = AutoFeatureExtractor.from_pretrained(tokenizer_path)
        worker_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
            tokenizer_path, device_map=device
        )

        schema = pa.schema(
            [
                ("row_id", pa.int32()),
                ("transcribe", pa.string()),
                ("audio_codes", pa.list_(pa.uint16())),
            ]
        )

        if False:
            source_directory = Path(sys.argv[1])
            dst_directory = Path(sys.argv[2])
            dst_directory.mkdir(parents=True, exist_ok=True)
            source_files = sorted(list(source_directory.rglob("*.parquet")))
            for source_file in tqdm(source_files):
                dst_file = dst_directory / source_file.name
                if dst_file.exists():
                    continue
                logger.info(f"Processing {source_file}")
                try:
                    df = process_parquet_file(source_file)
                    df.to_parquet(dst_file, index=False, schema=schema)
                except Exception as e:
                    logger.error(f"Failed to process {source_file}: {e}")
        else:
            file_path = Path(sys.argv[1])
            save_path = Path(sys.argv[2])
            df = process_parquet_file(file_path)
            df.to_parquet(save_path, index=False, schema=schema)

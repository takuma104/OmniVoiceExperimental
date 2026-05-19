#!/usr/bin/env python3
"""Visualize OmniVoice ASR token timestamps on decoded mel spectrograms."""

import argparse
import html
import json
import logging
import re
from pathlib import Path
from typing import Iterator, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm
from transformers import HiggsAudioV2TokenizerModel

from omnivoice.cli.visualize_asr_attention import _get_best_device
from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader

logger = logging.getLogger(__name__)


def _iter_samples(data_lst: str) -> Iterator[dict]:
    manifests = webdataset_manifest_reader(data_lst)
    reader = WebDatasetReader(manifests=manifests, evaluation=True)
    return iter(reader)


def _read_timestamp_items(
    timestamp_jsonl: str,
    limit: Optional[int],
    sample_id: Optional[str],
) -> list[dict]:
    items = []
    with open(timestamp_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if sample_id is not None and item.get("id") != sample_id:
                continue
            if "error" in item:
                continue
            items.append(item)
            if limit is not None and len(items) >= limit:
                break
    if sample_id is not None and not items:
        raise ValueError(f"Sample id not found in timestamp JSONL: {sample_id}")
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
            label = sample["label"]
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


def _sanitize_filename(value: object) -> str:
    text = str(value or "sample")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "sample"


def _resolve_audio_tokens(sample: dict) -> torch.Tensor:
    audio_tokens = sample["audio_tokens"]
    if audio_tokens.dim() == 3 and audio_tokens.size(0) == 1:
        audio_tokens = audio_tokens.squeeze(0)
    if audio_tokens.dim() != 2:
        raise ValueError(f"Expected audio tokens [C,T], got {tuple(audio_tokens.shape)}")
    return audio_tokens.to(dtype=torch.long)


@torch.inference_mode()
def _decode_audio_tokens(
    tokenizer: HiggsAudioV2TokenizerModel,
    audio_tokens: torch.Tensor,
) -> np.ndarray:
    tokenizer_device = tokenizer.device
    decoded = tokenizer.decode(audio_tokens.to(tokenizer_device).unsqueeze(0))
    waveform = decoded.audio_values[0].detach().cpu().numpy()
    return np.asarray(waveform).squeeze().astype(np.float32)


def _mel_spectrogram_db(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)
    return mel_db.astype(np.float32)


def _downsample_time(matrix: np.ndarray, max_frames: int) -> np.ndarray:
    if matrix.shape[1] <= max_frames:
        return matrix
    edges = np.linspace(0, matrix.shape[1], max_frames + 1)
    starts = np.floor(edges[:-1]).astype(np.int64)
    ends = np.floor(edges[1:]).astype(np.int64)
    ends = np.maximum(ends, starts + 1)
    ends[-1] = matrix.shape[1]
    return np.stack(
        [matrix[:, start:end].mean(axis=1) for start, end in zip(starts, ends)],
        axis=1,
    )


def _token_rows(tokens: list[dict], audio_seconds: float) -> list[dict]:
    rows = []
    for token in tokens:
        rows.append(
            {
                "index": int(token["index"]),
                "token": str(token["token"]),
                "startSec": float(token.get("start_sec", 0.0)),
                "endSec": float(token.get("end_sec", token.get("start_sec", 0.0))),
                "centerSec": float(token.get("center_sec", token.get("start_sec", 0.0))),
                "startAudioToken": int(token.get("start_audio_token", 0)),
                "endAudioToken": int(token.get("end_audio_token", 0)),
                "centerAudioToken": int(token.get("center_audio_token", 0)),
                "confidence": float(token.get("confidence", 0.0)),
            }
        )
    rows.sort(key=lambda item: (item["startSec"], item["index"]))
    for row in rows:
        row["startSec"] = max(0.0, min(audio_seconds, row["startSec"]))
        row["endSec"] = max(row["startSec"], min(audio_seconds, row["endSec"]))
        row["centerSec"] = max(0.0, min(audio_seconds, row["centerSec"]))
    return rows


def _write_html(
    path: Path,
    item: dict,
    label: dict,
    mel: np.ndarray,
    audio_seconds: float,
    sample_rate: int,
):
    tokens = _token_rows(item.get("tokens", []), audio_seconds)
    payload = {
        "mel": mel.tolist(),
        "tokens": tokens,
        "audioSeconds": audio_seconds,
        "sampleRate": sample_rate,
        "text": item.get("text", ""),
        "reference": item.get("reference") or label.get("text") or "",
    }
    token_table_rows = "\n".join(
        "<tr>"
        f"<td>{token['index']}</td>"
        f"<td>{html.escape(token['token'])}</td>"
        f"<td>{token['startAudioToken']}:{token['endAudioToken']}</td>"
        f"<td>{token['startSec']:.3f}-{token['endSec']:.3f}</td>"
        f"<td>{token['confidence']:.4f}</td>"
        "</tr>"
        for token in tokens
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OmniVoice ASR Timestamps</title>
<style>
body {{
  margin: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #172026;
  background: #f7f8fa;
}}
h1 {{ font-size: 20px; margin: 0 0 8px; }}
.meta {{ margin: 0 0 16px; color: #4b5963; line-height: 1.45; }}
.viewer {{
  overflow-x: auto;
  background: #ffffff;
  border: 1px solid #d7dde2;
  padding: 12px;
}}
canvas {{
  display: block;
}}
.controls {{
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin: 0 0 12px;
  font-size: 13px;
}}
label {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
table {{
  margin-top: 18px;
  border-collapse: collapse;
  width: 100%;
  background: #ffffff;
  border: 1px solid #d7dde2;
}}
th, td {{
  border-bottom: 1px solid #e8ecef;
  padding: 6px 8px;
  font-size: 12px;
  text-align: left;
}}
th {{ background: #eef2f5; }}
</style>
</head>
<body>
<h1>OmniVoice ASR Timestamps</h1>
<p class="meta">
sample: {html.escape(str(item.get("id")))}<br>
language: {html.escape(str(item.get("language_id")))}<br>
audio: {audio_seconds:.3f}s, sample rate: {sample_rate}, audio tokens: {item.get("audio_num_tokens")}<br>
hypothesis: {html.escape(str(item.get("text", "")))}<br>
reference: {html.escape(str(payload["reference"]))}
</p>
<div class="controls">
  <label><input id="showSpans" type="checkbox" checked>Token spans</label>
  <label><input id="showCenters" type="checkbox" checked>Centers</label>
  <label><input id="showLabels" type="checkbox" checked>Labels</label>
</div>
<div class="viewer">
  <canvas id="spectrogram"></canvas>
</div>
<table>
  <thead><tr><th>#</th><th>Token</th><th>Audio tokens</th><th>Seconds</th><th>Confidence</th></tr></thead>
  <tbody>{token_table_rows}</tbody>
</table>
<script>
const payload = {json.dumps(payload, ensure_ascii=False)};
const mel = payload.mel;
const tokens = payload.tokens;
const cols = mel[0] ? mel[0].length : 0;
const rows = mel.length;
const specHeight = Math.max(120, rows * 3);
const tokenBandHeight = 78;
const axisHeight = 24;
const width = Math.max(900, cols * 2);
const height = specHeight + tokenBandHeight + axisHeight;
const canvas = document.getElementById("spectrogram");
canvas.width = width;
canvas.height = height;
canvas.style.width = `${{width}}px`;
canvas.style.height = `${{height}}px`;
const ctx = canvas.getContext("2d");

function xFromSec(sec) {{
  return Math.max(0, Math.min(width, sec / payload.audioSeconds * width));
}}

function melColor(value) {{
  const x = Math.max(0, Math.min(1, value));
  const r = Math.round(8 + 235 * Math.pow(x, 1.7));
  const g = Math.round(20 + 170 * x);
  const b = Math.round(32 + 90 * (1 - x));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function render() {{
  ctx.clearRect(0, 0, width, height);
  const cellW = width / Math.max(cols, 1);
  const cellH = specHeight / Math.max(rows, 1);
  for (let r = 0; r < rows; r++) {{
    const y = specHeight - (r + 1) * cellH;
    for (let c = 0; c < cols; c++) {{
      ctx.fillStyle = melColor(mel[r][c]);
      ctx.fillRect(c * cellW, y, Math.ceil(cellW), Math.ceil(cellH));
    }}
  }}

  const showSpans = document.getElementById("showSpans").checked;
  const showCenters = document.getElementById("showCenters").checked;
  const showLabels = document.getElementById("showLabels").checked;
  if (showSpans) {{
    for (const token of tokens) {{
      const x0 = xFromSec(token.startSec);
      const x1 = xFromSec(token.endSec);
      const alpha = Math.max(0.16, Math.min(0.62, 0.18 + token.confidence * 3.5));
      ctx.fillStyle = `rgba(22, 119, 190, ${{alpha}})`;
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), specHeight);
    }}
  }}
  if (showCenters) {{
    ctx.strokeStyle = "rgba(210, 40, 40, 0.9)";
    ctx.lineWidth = 1.2;
    for (const token of tokens) {{
      const x = xFromSec(token.centerSec);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, specHeight);
      ctx.stroke();
    }}
  }}

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, specHeight, width, tokenBandHeight + axisHeight);
  ctx.strokeStyle = "#cad2da";
  ctx.beginPath();
  ctx.moveTo(0, specHeight);
  ctx.lineTo(width, specHeight);
  ctx.stroke();

  if (showLabels) {{
    ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.textBaseline = "top";
    let lastLabelX = -1000;
    for (const token of tokens) {{
      const x = xFromSec(token.centerSec);
      const y = specHeight + (token.index % 3) * 22 + 4;
      ctx.strokeStyle = "rgba(210, 40, 40, 0.35)";
      ctx.beginPath();
      ctx.moveTo(x, specHeight);
      ctx.lineTo(x, y + 14);
      ctx.stroke();
      if (x - lastLabelX > 10) {{
        ctx.fillStyle = "#1c2b33";
        ctx.fillText(token.token, x + 2, y);
        lastLabelX = x;
      }}
    }}
  }}

  ctx.fillStyle = "#53616b";
  ctx.font = "11px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
  ctx.textBaseline = "top";
  const tickStep = Math.max(0.5, Math.ceil(payload.audioSeconds / 10));
  for (let sec = 0; sec <= payload.audioSeconds + 1e-6; sec += tickStep) {{
    const x = xFromSec(sec);
    ctx.strokeStyle = "#d8dde2";
    ctx.beginPath();
    ctx.moveTo(x, specHeight + tokenBandHeight);
    ctx.lineTo(x, specHeight + tokenBandHeight + 6);
    ctx.stroke();
    ctx.fillText(`${{sec.toFixed(1)}}s`, x + 2, specHeight + tokenBandHeight + 7);
  }}
}}

for (const id of ["showSpans", "showCenters", "showLabels"]) {{
  document.getElementById(id).addEventListener("change", render);
}}
render();
</script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def visualize(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = _read_timestamp_items(
        timestamp_jsonl=args.timestamp_jsonl,
        limit=args.limit,
        sample_id=args.sample_id,
    )
    if not items:
        raise ValueError("No timestamp items to visualize.")

    samples = _load_samples_by_id(
        data_lst=args.data_lst,
        sample_ids={str(item["id"]) for item in items},
        no_progress=args.no_progress,
    )

    device = args.device
    if device == "auto":
        device = _get_best_device()
    logger.info("Loading audio tokenizer: %s", args.audio_tokenizer_path)
    tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
        args.audio_tokenizer_path,
        device_map=device,
    )

    index_entries = []
    progress = tqdm(items, disable=args.no_progress, unit="sample", desc="Rendering")
    for item in progress:
        sample = samples[str(item["id"])]
        label = sample["label"]
        audio_tokens = _resolve_audio_tokens(sample)
        waveform = _decode_audio_tokens(tokenizer, audio_tokens)
        mel = _mel_spectrogram_db(
            waveform=waveform,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        )
        mel = _downsample_time(mel, args.max_mel_frames)
        audio_seconds = float(item.get("audio_seconds") or len(waveform) / args.sample_rate)

        stem = _sanitize_filename(item.get("id"))
        html_path = output_dir / f"{stem}.timestamps.html"
        wav_name = None
        if args.save_wav:
            wav_path = output_dir / f"{stem}.wav"
            sf.write(wav_path, waveform, args.sample_rate)
            wav_name = wav_path.name
        _write_html(
            path=html_path,
            item=item,
            label=label,
            mel=mel,
            audio_seconds=audio_seconds,
            sample_rate=args.sample_rate,
        )
        index_entries.append(
            (item.get("id"), html_path.name, wav_name, item.get("text", ""))
        )

    index_rows = "\n".join(
        (
            "<li>"
            f'<a href="{html.escape(filename)}">{html.escape(str(sample_id))}</a> '
            + (
            f'<a href="{html.escape(wav_name)}">wav</a> '
            if wav_name is not None
            else ""
            )
            + f"{html.escape(text)}"
            + "</li>"
        )
        for sample_id, filename, wav_name, text in index_entries
    )
    (output_dir / "index.html").write_text(
        f"<!doctype html><meta charset='utf-8'><title>ASR timestamps</title>"
        f"<h1>ASR timestamp visualizations</h1><ul>{index_rows}</ul>",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "count": len(index_entries)}))


def main():
    parser = argparse.ArgumentParser(
        description="Visualize timestamp_asr JSONL over decoded mel spectrograms"
    )
    parser.add_argument("--timestamp_jsonl", required=True)
    parser.add_argument("--data_lst", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--audio_tokenizer_path",
        default="eustlb/higgs-audio-v2-tokenizer",
        help="Higgs audio tokenizer path used to decode codec tokens.",
    )
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--n_mels", type=int, default=96)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--max_mel_frames", type=int, default=1800)
    parser.add_argument(
        "--save_wav",
        action="store_true",
        help="Save decoded codec audio as WAV next to each timestamp HTML.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:0, mps, cpu, etc.",
    )
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()

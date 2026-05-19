#!/usr/bin/env python3
"""Visualize OmniVoice ASR audio-prefix attention for one sample."""

import argparse
import html
import json
import logging
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from accelerate.utils import set_seed
from transformers import AutoTokenizer

from omnivoice.data.dataset import WebDatasetReader, webdataset_manifest_reader
from omnivoice.models.omnivoice_asr import OmniVoiceForSpeechRecognition

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


def _iter_samples(data_lst: str) -> Iterable[dict]:
    manifests = webdataset_manifest_reader(data_lst)
    reader = WebDatasetReader(manifests=manifests, evaluation=True)
    return iter(reader)


def _select_sample(
    data_lst: str,
    sample_index: int,
    sample_id: Optional[str],
) -> tuple[int, dict]:
    for idx, sample in enumerate(_iter_samples(data_lst)):
        label = sample["label"]
        if sample_id is not None:
            if label.get("id") == sample_id:
                return idx, sample
        elif idx == sample_index:
            return idx, sample
    if sample_id is not None:
        raise ValueError(f"Sample id not found: {sample_id}")
    raise ValueError(f"Sample index out of range: {sample_index}")


def _sanitize_filename(value: object) -> str:
    text = str(value or "sample")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "sample"


def _audio_duration_seconds(
    label: dict,
    audio_num_tokens: int,
    args: argparse.Namespace,
) -> float:
    if args.audio_duration is not None:
        return float(args.audio_duration)
    duration = label.get("audio_duration")
    if duration is not None:
        return float(duration)
    return float(audio_num_tokens) / float(args.audio_frame_rate)


def _bin_attention(
    attention: np.ndarray,
    max_audio_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if attention.size == 0:
        return attention, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    num_audio_tokens = attention.shape[1]
    if num_audio_tokens <= max_audio_bins:
        starts = np.arange(num_audio_tokens, dtype=np.int64)
        ends = starts + 1
        return attention, starts, ends

    edges = np.linspace(0, num_audio_tokens, max_audio_bins + 1)
    starts = np.floor(edges[:-1]).astype(np.int64)
    ends = np.floor(edges[1:]).astype(np.int64)
    ends = np.maximum(ends, starts + 1)
    ends[-1] = num_audio_tokens

    binned = np.stack(
        [attention[:, start:end].sum(axis=1) for start, end in zip(starts, ends)],
        axis=1,
    )
    return binned, starts, ends


def _moving_average_axis1(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size == 0:
        return values
    window = radius * 2 + 1
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    cumsum = np.cumsum(padded, axis=1)
    cumsum = np.pad(cumsum, ((0, 0), (1, 0)), mode="constant")
    return (cumsum[:, window:] - cumsum[:, :-window]) / float(window)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)


def _enhance_attention_for_alignment(
    attention: np.ndarray,
    mode: str,
    baseline_quantile: float,
    power: float,
    smooth_radius: int,
) -> np.ndarray:
    if attention.size == 0 or mode == "none":
        return attention.copy()

    values = attention.astype(np.float32, copy=True)
    if mode in {"col_center", "alignment"}:
        baseline = np.quantile(values, baseline_quantile, axis=0, keepdims=True)
        values = np.maximum(values - baseline, 0.0)
    elif mode == "row_zscore":
        mean = values.mean(axis=1, keepdims=True)
        std = values.std(axis=1, keepdims=True)
        values = np.maximum((values - mean) / np.maximum(std, 1e-12), 0.0)
    elif mode == "tfidf":
        baseline = np.quantile(values, baseline_quantile, axis=0, keepdims=True)
        centered = np.maximum(values - baseline, 0.0)
        active = centered > 0
        doc_freq = active.sum(axis=0, keepdims=True)
        idf = np.log((values.shape[0] + 1.0) / (doc_freq + 1.0)) + 1.0
        values = centered * idf
    else:
        raise ValueError(f"Unsupported --enhance mode: {mode}")

    if mode == "alignment":
        row_peak = values.max(axis=1, keepdims=True)
        row_gate = values >= (row_peak * 0.15)
        values = np.where(row_gate, values, 0.0)

    if smooth_radius > 0:
        values = _moving_average_axis1(values, smooth_radius)

    if power > 0 and power != 1.0:
        values = np.power(np.maximum(values, 0.0), power)

    return _normalize_rows(values)


def _top_audio_tokens(
    attention: np.ndarray,
    token_texts: list[str],
    query_token_texts: list[str],
    audio_seconds: float,
    top_k: int,
) -> list[dict]:
    if attention.size == 0 or top_k <= 0:
        return []
    k = min(top_k, attention.shape[1])
    rows = []
    for row_idx, scores in enumerate(attention):
        top_indices = np.argpartition(-scores, kth=k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        rows.append(
            {
                "index": row_idx,
                "token": token_texts[row_idx],
                "query_token": query_token_texts[row_idx],
                "top_audio_tokens": [
                    {
                        "audio_token": int(idx),
                        "time_sec": (float(idx) + 0.5)
                        * audio_seconds
                        / attention.shape[1],
                        "attention": float(scores[idx]),
                    }
                    for idx in top_indices
                ],
            }
        )
    return rows


def _write_html(
    path: Path,
    metadata: dict,
    display_attention: np.ndarray,
    bin_starts: np.ndarray,
    bin_ends: np.ndarray,
    layer_display_attentions: Optional[dict[str, np.ndarray]] = None,
):
    tokens = metadata["tokens"]
    rows = display_attention.shape[0]
    cols = display_attention.shape[1] if display_attention.ndim == 2 else 0
    display_percentile = metadata.get("display_percentile", 100.0)
    if display_attention.size:
        max_value = float(np.percentile(display_attention, display_percentile))
        if max_value <= 0:
            max_value = float(display_attention.max())
    else:
        max_value = 0.0
    payload = {
        "attention": display_attention.tolist(),
        "layerAttentions": {
            key: value.tolist()
            for key, value in (layer_display_attentions or {}).items()
        },
        "layerLabels": list((layer_display_attentions or {}).keys()),
        "tokens": [item["token"] for item in tokens],
        "queryTokens": [item["query_token"] for item in tokens],
        "binStarts": bin_starts.tolist(),
        "binEnds": bin_ends.tolist(),
        "audioSeconds": metadata["audio_seconds"],
        "maxValue": max_value,
        "displayPercentile": display_percentile,
        "rows": rows,
        "cols": cols,
        "overlayRidge": bool(metadata.get("overlay_ridge")),
    }

    token_labels = "\n".join(
        f'<div class="token-row" title="query: {html.escape(item["query_token"])}">'
        f'<span class="idx">{item["index"]}</span>'
        f'<span class="tok">{html.escape(item["token"])}</span>'
        "</div>"
        for item in tokens
    )
    top_table_rows = []
    for item in metadata["top_audio_tokens"]:
        top_summary = ", ".join(
            f"{top['time_sec']:.2f}s ({top['attention']:.4f})"
            for top in item["top_audio_tokens"][:3]
        )
        top_table_rows.append(
            "<tr>"
            f"<td>{item['index']}</td>"
            f"<td>{html.escape(item['token'])}</td>"
            f"<td>{html.escape(item['query_token'])}</td>"
            f"<td>{html.escape(top_summary)}</td>"
            "</tr>"
        )
    top_rows = "\n".join(top_table_rows)
    if layer_display_attentions:
        layer_controls = "\n".join(
            '<label class="layer-toggle">'
            f'<input type="checkbox" class="layer-check" data-layer="{html.escape(layer)}" checked>'
            f"Layer {html.escape(layer)}"
            "</label>"
            for layer in layer_display_attentions
        )
        layer_controls_html = f'<div class="layer-controls">{layer_controls}</div>'
    else:
        layer_controls_html = ""

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>OmniVoice ASR Attention</title>
<style>
body {{
  margin: 24px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #172026;
  background: #f7f8fa;
}}
h1 {{ font-size: 20px; margin: 0 0 8px; }}
.meta {{ margin: 0 0 18px; color: #4b5963; line-height: 1.45; }}
.viewer {{
  display: grid;
  grid-template-columns: minmax(160px, 260px) 1fr;
  gap: 12px;
  align-items: start;
}}
.labels {{
  background: #ffffff;
  border: 1px solid #d7dde2;
  overflow: hidden;
}}
.token-row {{
  height: 18px;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 8px;
  border-bottom: 1px solid #eef1f4;
  font-size: 12px;
  white-space: nowrap;
}}
.idx {{ color: #7b8790; width: 30px; text-align: right; }}
.tok {{ overflow: hidden; text-overflow: ellipsis; }}
.canvas-wrap {{
  overflow: auto;
  background: #ffffff;
  border: 1px solid #d7dde2;
  padding: 0;
}}
.layer-controls {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  margin: 0 0 10px;
  padding: 8px 10px;
  background: #ffffff;
  border: 1px solid #d7dde2;
  font-size: 12px;
}}
.layer-toggle {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: #31404a;
}}
canvas {{
  display: block;
  image-rendering: pixelated;
}}
table {{
  margin-top: 22px;
  border-collapse: collapse;
  width: 100%;
  background: #ffffff;
  border: 1px solid #d7dde2;
}}
th, td {{
  border-bottom: 1px solid #e8ecef;
  padding: 7px 9px;
  font-size: 13px;
  text-align: left;
}}
th {{ background: #eef2f5; }}
</style>
</head>
<body>
<h1>OmniVoice ASR Attention</h1>
<p class="meta">
sample: {html.escape(str(metadata.get("id")))}<br>
language: {html.escape(str(metadata.get("language_id")))}<br>
reference: {html.escape(str(metadata.get("reference", "")))}<br>
hypothesis: {html.escape(str(metadata.get("hypothesis", "")))}<br>
audio tokens: {metadata["audio_num_tokens"]}, audio seconds: {metadata["audio_seconds"]:.3f}<br>
display: {html.escape(str(metadata.get("enhance")))} / percentile {metadata.get("display_percentile")}<br>
</p>
{layer_controls_html}
<div class="viewer">
  <div class="labels">{token_labels}</div>
  <div class="canvas-wrap"><canvas id="heatmap"></canvas></div>
</div>
<table>
  <thead><tr><th>#</th><th>Predicted token</th><th>Query token</th><th>Top attention times</th></tr></thead>
  <tbody>{top_rows}</tbody>
</table>
<script>
const payload = {json.dumps(payload, ensure_ascii=False)};
const canvas = document.getElementById("heatmap");
const rowHeight = 18;
const cellWidth = 2;
canvas.width = Math.max(1, payload.cols * cellWidth);
canvas.height = Math.max(1, payload.rows * rowHeight);
canvas.style.width = `${{canvas.width}}px`;
canvas.style.height = `${{canvas.height}}px`;
const ctx = canvas.getContext("2d");
function percentile(values, pct) {{
  if (!values.length) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  const pos = Math.max(0, Math.min(sorted.length - 1, (pct / 100) * (sorted.length - 1)));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}}
function activeMatrix() {{
  const checks = Array.from(document.querySelectorAll(".layer-check"));
  const active = checks.filter((check) => check.checked).map((check) => check.dataset.layer);
  if (!active.length || !payload.layerLabels.length) return payload.attention;
  const matrix = Array.from({{length: payload.rows}}, () => Array(payload.cols).fill(0));
  for (const layer of active) {{
    const layerMatrix = payload.layerAttentions[layer];
    if (!layerMatrix) continue;
    for (let r = 0; r < payload.rows; r++) {{
      for (let c = 0; c < payload.cols; c++) {{
        matrix[r][c] += layerMatrix[r][c] / active.length;
      }}
    }}
  }}
  return matrix;
}}
function color(value, maxValue) {{
  const x = Math.max(0, Math.min(1, Math.sqrt(value / maxValue)));
  const r = Math.round(250 - 210 * x);
  const g = Math.round(252 - 118 * x);
  const b = Math.round(255 - 40 * x);
  return `rgb(${{r}},${{g}},${{b}})`;
}}
function render() {{
  const matrix = activeMatrix();
  const flat = matrix.flat();
  let maxValue = percentile(flat, payload.displayPercentile || 100);
  if (!maxValue || maxValue <= 0) maxValue = Math.max(...flat, payload.maxValue, 1e-12);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < payload.rows; r++) {{
    for (let c = 0; c < payload.cols; c++) {{
      ctx.fillStyle = color(matrix[r][c], maxValue);
      ctx.fillRect(c * cellWidth, r * rowHeight, cellWidth, rowHeight);
    }}
  }}
  if (payload.overlayRidge && payload.rows > 0 && payload.cols > 0) {{
    ctx.beginPath();
    ctx.strokeStyle = "rgba(206, 42, 42, 0.85)";
    ctx.lineWidth = 1.5;
    let lastCol = 0;
    for (let r = 0; r < payload.rows; r++) {{
      let bestCol = 0;
      let bestValue = -1;
      for (let c = 0; c < payload.cols; c++) {{
        const value = matrix[r][c];
        if (value > bestValue) {{
          bestValue = value;
          bestCol = c;
        }}
      }}
      if (bestCol < lastCol) bestCol = lastCol;
      lastCol = bestCol;
      const x = bestCol * cellWidth + cellWidth / 2;
      const y = r * rowHeight + rowHeight / 2;
      if (r === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}
}}
for (const check of document.querySelectorAll(".layer-check")) {{
  check.addEventListener("change", render);
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
    set_seed(args.seed)
    if args.max_audio_bins < 1:
        raise ValueError("--max_audio_bins must be >= 1")
    if not 0.0 <= args.enhance_baseline_quantile <= 1.0:
        raise ValueError("--enhance_baseline_quantile must be between 0 and 1")
    if args.enhance_smooth_radius < 0:
        raise ValueError("--enhance_smooth_radius must be >= 0")
    if args.enhance_power <= 0:
        raise ValueError("--enhance_power must be > 0")
    if args.display_percentile <= 0 or args.display_percentile > 100:
        raise ValueError("--display_percentile must be in (0, 100]")

    device = args.device
    if device == "auto":
        device = _get_best_device()
    dtype = _resolve_dtype(args.dtype, device)
    if args.attn_implementation != "eager":
        logger.warning(
            "Attention weights are most reliable with attn_implementation='eager'."
        )

    sample_index, sample = _select_sample(
        args.data_lst,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
    )
    label = sample["label"]
    language = (
        args.language if args.language is not None else label.get("language_id")
    )
    if "audio_tokens" not in sample:
        raise ValueError("Selected sample does not contain precomputed audio_tokens.")
    audio_tokens = sample["audio_tokens"]

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = OmniVoiceForSpeechRecognition.from_pretrained(
        args.checkpoint,
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    )
    model.to(device)
    model.eval()

    trace = model.generate_text_attention_trace(
        audio_tokens=audio_tokens,
        tokenizer=tokenizer,
        language=language,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        layers=args.layers,
        heads=args.heads,
        include_eos=args.include_eos,
        return_layer_attentions=not args.no_html_layer_controls,
    )

    attention = trace["audio_attention"].numpy()
    enhanced_attention = _enhance_attention_for_alignment(
        attention=attention,
        mode=args.enhance,
        baseline_quantile=args.enhance_baseline_quantile,
        power=args.enhance_power,
        smooth_radius=args.enhance_smooth_radius,
    )
    layer_labels = []
    layer_attention_stack = None
    layer_enhanced_attention_stack = None
    layer_display_attentions = None
    layer_audio_attentions = trace.get("layer_audio_attentions")
    if layer_audio_attentions:
        layer_display_attentions = {}
        layer_attention_arrays = []
        layer_enhanced_arrays = []
        for layer_index, layer_attention in sorted(layer_audio_attentions.items()):
            layer_labels.append(int(layer_index))
            layer_attention_array = layer_attention.numpy()
            layer_enhanced = _enhance_attention_for_alignment(
                attention=layer_attention_array,
                mode=args.enhance,
                baseline_quantile=args.enhance_baseline_quantile,
                power=args.enhance_power,
                smooth_radius=args.enhance_smooth_radius,
            )
            layer_binned, _, _ = _bin_attention(
                layer_enhanced,
                max_audio_bins=args.max_audio_bins,
            )
            layer_display_attentions[str(layer_index)] = layer_binned
            layer_attention_arrays.append(layer_attention_array)
            layer_enhanced_arrays.append(layer_enhanced)
        layer_attention_stack = np.stack(layer_attention_arrays, axis=0)
        layer_enhanced_attention_stack = np.stack(layer_enhanced_arrays, axis=0)
    audio_seconds = _audio_duration_seconds(label, trace["audio_num_tokens"], args)
    top_audio_tokens = _top_audio_tokens(
        attention=enhanced_attention,
        token_texts=trace["token_texts"],
        query_token_texts=trace["query_token_texts"],
        audio_seconds=audio_seconds,
        top_k=args.top_k,
    )

    tokens = [
        {
            "index": idx,
            "token_id": int(token_id),
            "token": token_text,
            "query_token_id": int(query_token_id),
            "query_token": query_token_text,
        }
        for idx, (
            token_id,
            token_text,
            query_token_id,
            query_token_text,
        ) in enumerate(
            zip(
                trace["token_ids"],
                trace["token_texts"],
                trace["query_token_ids"],
                trace["query_token_texts"],
            )
        )
    ]

    metadata = {
        "id": label.get("id"),
        "sample_index": sample_index,
        "language_id": language,
        "reference": label.get("text"),
        "hypothesis": trace["text"],
        "audio_num_tokens": trace["audio_num_tokens"],
        "audio_seconds": audio_seconds,
        "layers": args.layers,
        "heads": args.heads,
        "enhance": args.enhance,
        "enhance_baseline_quantile": args.enhance_baseline_quantile,
        "enhance_power": args.enhance_power,
        "enhance_smooth_radius": args.enhance_smooth_radius,
        "display_percentile": args.display_percentile,
        "overlay_ridge": not args.no_overlay_ridge,
        "html_layer_controls": not args.no_html_layer_controls,
        "html_layer_labels": layer_labels,
        "attention_semantics": (
            "Each row is the attention of the causal-LM query position whose "
            "hidden state predicted the listed output token."
        ),
        "tokens": tokens,
        "top_audio_tokens": top_audio_tokens,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_filename(label.get("id") or sample_index)
    json_path = output_dir / f"{stem}.attention.json"
    npz_path = output_dir / f"{stem}.attention.npz"
    html_path = output_dir / f"{stem}.attention.html"

    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audio_token_times = (
        (np.arange(trace["audio_num_tokens"], dtype=np.float32) + 0.5)
        * float(audio_seconds)
        / float(trace["audio_num_tokens"])
    )
    npz_payload = {
        "attention": attention,
        "enhanced_attention": enhanced_attention,
        "token_ids": np.array(trace["token_ids"], dtype=np.int64),
        "token_texts": np.array(trace["token_texts"]),
        "query_token_ids": np.array(trace["query_token_ids"], dtype=np.int64),
        "query_token_texts": np.array(trace["query_token_texts"]),
        "audio_token_times_sec": audio_token_times,
        "audio_seconds": np.array(audio_seconds, dtype=np.float32),
    }
    if layer_attention_stack is not None:
        npz_payload["layer_labels"] = np.array(layer_labels, dtype=np.int64)
        npz_payload["layer_attention"] = layer_attention_stack
        npz_payload["layer_enhanced_attention"] = layer_enhanced_attention_stack
    np.savez_compressed(npz_path, **npz_payload)

    display_attention, bin_starts, bin_ends = _bin_attention(
        enhanced_attention,
        max_audio_bins=args.max_audio_bins,
    )
    _write_html(
        html_path,
        metadata,
        display_attention,
        bin_starts,
        bin_ends,
        layer_display_attentions=layer_display_attentions,
    )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "npz": str(npz_path),
                "html": str(html_path),
                "hypothesis": trace["text"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize OmniVoice ASR audio-prefix attention for one sample"
    )
    parser.add_argument("--checkpoint", required=True, help="ASR checkpoint directory")
    parser.add_argument("--data_lst", required=True, help="WebDataset data.lst path")
    parser.add_argument("--output_dir", required=True, help="Directory for outputs")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sample_id", default=None)
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
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_audio_bins", type=int, default=1200)
    parser.add_argument(
        "--enhance",
        default="alignment",
        choices=["none", "col_center", "row_zscore", "tfidf", "alignment"],
        help=(
            "Display/postprocess attention for clearer alignment. Raw attention "
            "is still saved in the NPZ."
        ),
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
        "--display_percentile",
        type=float,
        default=99.5,
        help="Color-scale upper percentile for the HTML heatmap.",
    )
    parser.add_argument(
        "--no_overlay_ridge",
        action="store_true",
        help="Disable the monotonic row-peak ridge overlay in the HTML heatmap.",
    )
    parser.add_argument(
        "--no_html_layer_controls",
        action="store_true",
        help="Disable per-layer On/Off controls in the HTML heatmap.",
    )
    parser.add_argument(
        "--audio_duration",
        type=float,
        default=None,
        help="Override audio duration in seconds for time-axis conversion.",
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()

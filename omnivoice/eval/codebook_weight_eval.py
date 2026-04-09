#!/usr/bin/env python3
"""
Codebook weight importance evaluation for Qwen3 TTS Tokenizer.

Evaluates the relative importance of each codebook layer (0-15) using:
  Method 1: Single-codebook ablation (UTMOS drop per codebook)
  Method 2: Cumulative codebook inclusion (marginal contribution)
  Method 3: Information-theoretic analysis (entropy, mutual information)
  Method 4: Spectral analysis (per-codebook mel-spectrogram impact)
  Method 5: CER ablation (intelligibility impact via Whisper ASR)

Usage:
  python -m omnivoice.eval.codebook_weight_eval \
    --tar-path /path/to/shard-000000.tar \
    --utmos-model-path tts_eval_models/mos/utmos22_strong_step7459_v1.pt \
    --output-dir results/codebook_eval \
    --num-samples 30 \
    --num-trials 3 \
    --whisper-model openai/whisper-large-v3-turbo \
    --language en
"""

import argparse
import io
import json
import logging
import os
import random
import tarfile
import tempfile
from collections import Counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from jiwer import cer as compute_cer
from tqdm import tqdm

from omnivoice.eval.models.utmos import UTMOS22Strong

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

NUM_CODEBOOKS = 16
VOCAB_SIZE = 2048  # valid tokens: 0-2047
TOKENIZER_SR = 24000


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def load_tokens_from_tar(
    tar_path: str, num_samples: int, seed: int = 42
) -> tuple[list[np.ndarray], list[str]]:
    """Load token arrays from a WebDataset tar shard.

    Returns:
        tokens_list: list of int16 arrays with shape (16, T)
        sample_ids: list of sample ID strings (filename without .npy)
    """
    all_entries = []  # (sample_id, tokens)
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".npy"):
                f = tf.extractfile(member)
                if f is not None:
                    buf = io.BytesIO(f.read())
                    tokens = np.load(buf)
                    sample_id = os.path.basename(member.name).replace(".npy", "")
                    all_entries.append((sample_id, tokens))

    rng = random.Random(seed)
    if len(all_entries) > num_samples:
        all_entries = rng.sample(all_entries, num_samples)

    sample_ids = [e[0] for e in all_entries]
    tokens_list = [e[1] for e in all_entries]
    logger.info(f"Loaded {len(tokens_list)} token arrays from {tar_path}")
    return tokens_list, sample_ids


def load_metadata(jsonl_path: str) -> dict[str, dict]:
    """Load metadata from JSONL file. Returns dict keyed by sample ID."""
    metadata = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            metadata[d["id"]] = d
    return metadata


def decode_tokens(tokenizer, tokens: np.ndarray, device: torch.device) -> np.ndarray:
    """Decode (16, T) token array to waveform (numpy, 24kHz)."""
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2EncoderOutput,
    )

    t = torch.from_numpy(tokens.astype(np.int64)).to(device)
    # tokens: (C, T) -> decoder expects (B, T, C)
    wavs, _ = tokenizer.decode(Qwen3TTSTokenizerV2EncoderOutput(t.T.unsqueeze(0)))
    return wavs[0]  # numpy 1-D


def compute_utmos(utmos_model, waveform: np.ndarray, device: torch.device) -> float:
    """Compute UTMOS score for a waveform at 24kHz."""
    # Resample 24kHz -> 16kHz
    wav_16k = librosa.resample(waveform.astype(np.float32), orig_sr=TOKENIZER_SR, target_sr=16000)
    tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(device)
    with torch.no_grad():
        score = utmos_model(tensor, 16000)
    return score.item()


def replace_codebook(
    tokens: np.ndarray, k: int, strategy: str = "random", empirical_dist: np.ndarray | None = None
) -> np.ndarray:
    """Replace codebook k with ablated tokens.

    Args:
        tokens: (16, T) int16 array
        k: codebook index to replace
        strategy: "random" (uniform) or "empirical" (sample from distribution)
        empirical_dist: (2048,) probability distribution for empirical strategy
    """
    out = tokens.copy()
    T = tokens.shape[1]
    if strategy == "empirical" and empirical_dist is not None:
        out[k] = np.random.choice(VOCAB_SIZE, size=T, p=empirical_dist).astype(np.int16)
    else:
        out[k] = np.random.randint(0, VOCAB_SIZE, size=T, dtype=np.int16)
    return out


def compute_empirical_distributions(tokens_list: list[np.ndarray]) -> np.ndarray:
    """Compute per-codebook empirical token distributions.

    Returns: (16, 2048) array of probabilities.
    """
    counts = np.zeros((NUM_CODEBOOKS, VOCAB_SIZE), dtype=np.int64)
    for tokens in tokens_list:
        for k in range(NUM_CODEBOOKS):
            for v in tokens[k]:
                counts[k, v] += 1
    # Normalize
    sums = counts.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    return counts / sums


# ---------------------------------------------------------------------------
# Method 1: Single-codebook ablation
# ---------------------------------------------------------------------------


def run_single_ablation(
    tokenizer,
    utmos_model,
    tokens_list: list[np.ndarray],
    device: torch.device,
    num_trials: int = 3,
    empirical_dists: np.ndarray | None = None,
) -> dict:
    """Ablate each codebook individually and measure UTMOS drop."""
    results = {
        "baseline_scores": [],
        "ablation_random": np.zeros((NUM_CODEBOOKS, len(tokens_list))),
        "ablation_empirical": np.zeros((NUM_CODEBOOKS, len(tokens_list))),
    }

    for i, tokens in enumerate(tqdm(tokens_list, desc="Method 1: Single ablation")):
        # Baseline
        wav = decode_tokens(tokenizer, tokens, device)
        baseline = compute_utmos(utmos_model, wav, device)
        results["baseline_scores"].append(baseline)

        for k in range(NUM_CODEBOOKS):
            # Random replacement (average over trials)
            scores_rand = []
            scores_emp = []
            for t in range(num_trials):
                np.random.seed(42 + i * 1000 + k * 100 + t)
                # Random uniform
                ablated = replace_codebook(tokens, k, strategy="random")
                wav_abl = decode_tokens(tokenizer, ablated, device)
                scores_rand.append(compute_utmos(utmos_model, wav_abl, device))

                # Empirical distribution
                if empirical_dists is not None:
                    ablated_emp = replace_codebook(
                        tokens, k, strategy="empirical", empirical_dist=empirical_dists[k]
                    )
                    wav_emp = decode_tokens(tokenizer, ablated_emp, device)
                    scores_emp.append(compute_utmos(utmos_model, wav_emp, device))

            results["ablation_random"][k, i] = baseline - np.mean(scores_rand)
            if scores_emp:
                results["ablation_empirical"][k, i] = baseline - np.mean(scores_emp)

    return results


# ---------------------------------------------------------------------------
# Method 2: Cumulative codebook inclusion
# ---------------------------------------------------------------------------


def run_cumulative(
    tokenizer,
    utmos_model,
    tokens_list: list[np.ndarray],
    device: torch.device,
) -> dict:
    """Keep codebooks 0..k, randomize the rest. Measure UTMOS at each k."""
    results = {
        "utmos_at_k": np.zeros((NUM_CODEBOOKS, len(tokens_list))),
        "all_random": [],
    }

    for i, tokens in enumerate(tqdm(tokens_list, desc="Method 2: Cumulative inclusion")):
        np.random.seed(42 + i)

        # All random baseline
        all_rand = tokens.copy()
        for k in range(NUM_CODEBOOKS):
            all_rand[k] = np.random.randint(0, VOCAB_SIZE, size=tokens.shape[1], dtype=np.int16)
        wav_rand = decode_tokens(tokenizer, all_rand, device)
        results["all_random"].append(compute_utmos(utmos_model, wav_rand, device))

        # Cumulative: keep 0..k, randomize k+1..15
        for k in range(NUM_CODEBOOKS):
            cumul = tokens.copy()
            for j in range(k + 1, NUM_CODEBOOKS):
                cumul[j] = np.random.randint(0, VOCAB_SIZE, size=tokens.shape[1], dtype=np.int16)
            wav_cumul = decode_tokens(tokenizer, cumul, device)
            results["utmos_at_k"][k, i] = compute_utmos(utmos_model, wav_cumul, device)

    return results


# ---------------------------------------------------------------------------
# Method 3: Information-theoretic analysis
# ---------------------------------------------------------------------------


def run_entropy_analysis(tokens_list: list[np.ndarray]) -> dict:
    """Compute per-codebook entropy and pairwise mutual information."""
    # Concatenate all tokens across samples
    all_tokens = np.concatenate(tokens_list, axis=1)  # (16, total_T)

    entropies = []
    for k in range(NUM_CODEBOOKS):
        counts = Counter(all_tokens[k].tolist())
        total = sum(counts.values())
        probs = np.array([c / total for c in counts.values()])
        H = -np.sum(probs * np.log2(probs + 1e-12))
        entropies.append(H)

    # Pairwise MI between adjacent codebooks
    mutual_info = []
    for k in range(NUM_CODEBOOKS - 1):
        a = all_tokens[k]
        b = all_tokens[k + 1]

        # Bin to reduce sparsity (64 bins)
        num_bins = 64
        a_binned = (a.astype(np.int32) * num_bins // VOCAB_SIZE).clip(0, num_bins - 1)
        b_binned = (b.astype(np.int32) * num_bins // VOCAB_SIZE).clip(0, num_bins - 1)

        # Joint distribution
        joint_counts = np.zeros((num_bins, num_bins), dtype=np.int64)
        for ai, bi in zip(a_binned, b_binned):
            joint_counts[ai, bi] += 1
        joint_total = joint_counts.sum()
        joint_probs = joint_counts / (joint_total + 1e-12)

        # Marginals
        p_a = joint_probs.sum(axis=1)
        p_b = joint_probs.sum(axis=0)

        # MI = sum p(a,b) * log2(p(a,b) / (p(a)*p(b)))
        mi = 0.0
        for ai in range(num_bins):
            for bi in range(num_bins):
                if joint_probs[ai, bi] > 0 and p_a[ai] > 0 and p_b[bi] > 0:
                    mi += joint_probs[ai, bi] * np.log2(
                        joint_probs[ai, bi] / (p_a[ai] * p_b[bi] + 1e-12)
                    )
        mutual_info.append(mi)

    return {
        "entropies": entropies,
        "mutual_info": mutual_info,
        "max_entropy": np.log2(VOCAB_SIZE),
    }


# ---------------------------------------------------------------------------
# Method 4: Spectral analysis
# ---------------------------------------------------------------------------


def compute_mel(waveform: np.ndarray, sr: int = TOKENIZER_SR, n_mels: int = 80) -> np.ndarray:
    """Compute log-mel spectrogram."""
    mel = librosa.feature.melspectrogram(y=waveform.astype(np.float32), sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def run_spectral_analysis(
    tokenizer,
    tokens_list: list[np.ndarray],
    device: torch.device,
    n_mels: int = 80,
) -> dict:
    """Compute per-codebook spectral impact."""
    impact_matrix = np.zeros((NUM_CODEBOOKS, n_mels))
    count = 0

    for i, tokens in enumerate(tqdm(tokens_list, desc="Method 4: Spectral analysis")):
        np.random.seed(42 + i)
        wav_ref = decode_tokens(tokenizer, tokens, device)
        mel_ref = compute_mel(wav_ref, n_mels=n_mels)

        for k in range(NUM_CODEBOOKS):
            ablated = replace_codebook(tokens, k, strategy="random")
            wav_abl = decode_tokens(tokenizer, ablated, device)
            mel_abl = compute_mel(wav_abl, n_mels=n_mels)

            # Align lengths
            min_t = min(mel_ref.shape[1], mel_abl.shape[1])
            diff = np.abs(mel_ref[:, :min_t] - mel_abl[:, :min_t])
            impact_matrix[k] += diff.mean(axis=1)  # average across time

        count += 1

    if count > 0:
        impact_matrix /= count

    return {"impact_matrix": impact_matrix, "n_mels": n_mels}


# ---------------------------------------------------------------------------
# Method 5: CER ablation (intelligibility via Whisper)
# ---------------------------------------------------------------------------


def load_whisper_pipeline(whisper_model: str, device: torch.device):
    """Load Whisper ASR pipeline."""
    import transformers

    transformers.logging.set_verbosity_error()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = transformers.pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        dtype=dtype,
        device=device,
    )
    return pipe


def transcribe_waveform(
    pipe, waveform: np.ndarray, sr: int = TOKENIZER_SR, language: str = "ja"
) -> str:
    """Transcribe a waveform using Whisper pipeline."""
    # Whisper expects 16kHz
    if sr != 16000:
        wav_16k = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=16000)
    else:
        wav_16k = waveform.astype(np.float32)

    result = pipe(
        wav_16k,
        generate_kwargs={"language": language, "task": "transcribe"},
        return_timestamps=False,
    )
    return result["text"].strip()


def compute_cer_score(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate between reference and hypothesis."""
    # Character-level: insert spaces between each character
    ref_chars = " ".join(list(reference.replace(" ", "")))
    hyp_chars = " ".join(list(hypothesis.replace(" ", "")))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return compute_cer(ref_chars, hyp_chars)


def run_cer_ablation(
    tokenizer,
    whisper_pipe,
    tokens_list: list[np.ndarray],
    sample_ids: list[str],
    metadata: dict[str, dict],
    device: torch.device,
    language: str = "ja",
) -> dict:
    """Ablate each codebook and measure CER increase."""
    results = {
        "baseline_cer": [],
        "ablation_cer": np.zeros((NUM_CODEBOOKS, len(tokens_list))),
        "cer_increase": np.zeros((NUM_CODEBOOKS, len(tokens_list))),
    }

    valid_count = 0
    for i, (tokens, sid) in enumerate(
        tqdm(zip(tokens_list, sample_ids), total=len(tokens_list), desc="Method 5: CER ablation")
    ):
        meta = metadata.get(sid)
        if meta is None or "text" not in meta:
            logger.warning(f"No metadata for {sid}, skipping CER")
            results["baseline_cer"].append(float("nan"))
            continue

        ref_text = meta["text"]

        # Baseline
        wav = decode_tokens(tokenizer, tokens, device)
        baseline_hyp = transcribe_waveform(whisper_pipe, wav, language=language)
        baseline_cer = compute_cer_score(ref_text, baseline_hyp)
        results["baseline_cer"].append(baseline_cer)
        valid_count += 1

        for k in range(NUM_CODEBOOKS):
            np.random.seed(42 + i * 1000 + k)
            ablated = replace_codebook(tokens, k, strategy="random")
            wav_abl = decode_tokens(tokenizer, ablated, device)
            abl_hyp = transcribe_waveform(whisper_pipe, wav_abl, language=language)
            abl_cer = compute_cer_score(ref_text, abl_hyp)
            results["ablation_cer"][k, i] = abl_cer
            results["cer_increase"][k, i] = abl_cer - baseline_cer

    logger.info(f"CER ablation: {valid_count}/{len(tokens_list)} samples evaluated")
    return results


# ---------------------------------------------------------------------------
# Weight derivation
# ---------------------------------------------------------------------------


def derive_weights(
    ablation_results: dict,
    cumulative_results: dict,
    entropy_results: dict,
    spectral_results: dict,
    cer_results: dict | None = None,
    target_sum: float = 136.0,
) -> dict:
    """Combine signals to derive suggested codebook weights."""
    # 1. Ablation importance (UTMOS drop, higher = more important)
    ablation_importance = ablation_results["ablation_random"].mean(axis=1)
    ablation_importance = np.clip(ablation_importance, 0, None)

    # 2. Marginal contribution from cumulative
    utmos_means = cumulative_results["utmos_at_k"].mean(axis=1)
    all_rand_mean = np.mean(cumulative_results["all_random"])
    marginal = np.zeros(NUM_CODEBOOKS)
    marginal[0] = utmos_means[0] - all_rand_mean
    for k in range(1, NUM_CODEBOOKS):
        marginal[k] = utmos_means[k] - utmos_means[k - 1]
    marginal = np.clip(marginal, 0, None)

    # 3. Entropy (normalized)
    entropies = np.array(entropy_results["entropies"])

    # 4. Spectral impact (sum across mel bins)
    spectral_importance = spectral_results["impact_matrix"].sum(axis=1)

    # 5. CER importance (CER increase, higher = more important for intelligibility)
    cer_importance = None
    if cer_results is not None:
        valid_mask = ~np.isnan(cer_results["baseline_cer"])
        if valid_mask.any():
            cer_importance = cer_results["cer_increase"][:, valid_mask].mean(axis=1)
            cer_importance = np.clip(cer_importance, 0, None)

    def normalize(x):
        s = x.sum()
        return x / s if s > 0 else np.ones_like(x) / len(x)

    # Weighted combination
    if cer_importance is not None and cer_importance.sum() > 0:
        # With CER: rebalance to include intelligibility signal
        combined = (
            0.40 * normalize(ablation_importance)
            + 0.20 * normalize(marginal)
            + 0.10 * normalize(entropies)
            + 0.10 * normalize(spectral_importance)
            + 0.20 * normalize(cer_importance)
        )
    else:
        combined = (
            0.50 * normalize(ablation_importance)
            + 0.25 * normalize(marginal)
            + 0.15 * normalize(entropies)
            + 0.10 * normalize(spectral_importance)
        )
    combined = normalize(combined)

    # Scale to target sum and round to even integers
    raw_weights = combined * target_sum
    suggested = np.round(raw_weights / 2) * 2  # round to nearest even
    suggested = np.clip(suggested, 2, None)  # minimum weight of 2

    result = {
        "ablation_importance": ablation_importance,
        "marginal_contribution": marginal,
        "entropy_importance": entropies,
        "spectral_importance": spectral_importance,
        "combined_normalized": combined,
        "suggested_weights": suggested.astype(int).tolist(),
    }
    if cer_importance is not None:
        result["cer_importance"] = cer_importance
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def generate_plots(
    ablation_results: dict,
    cumulative_results: dict,
    entropy_results: dict,
    spectral_results: dict,
    weight_results: dict,
    output_dir: str,
    cer_results: dict | None = None,
):
    """Generate all evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    codebooks = list(range(NUM_CODEBOOKS))
    current_weights = [24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2]

    # --- Plot 1: Single-codebook ablation ---
    fig, ax = plt.subplots(figsize=(12, 5))
    drops_rand = ablation_results["ablation_random"].mean(axis=1)
    drops_emp = ablation_results["ablation_empirical"].mean(axis=1)
    x = np.arange(NUM_CODEBOOKS)
    width = 0.35
    ax.bar(x - width / 2, drops_rand, width, label="Random replacement", color="steelblue")
    if drops_emp.sum() > 0:
        ax.bar(x + width / 2, drops_emp, width, label="Empirical replacement", color="coral")
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("UTMOS drop (higher = more important)")
    ax.set_title("Method 1: Single-Codebook Ablation")
    ax.set_xticks(codebooks)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "01_single_ablation.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: Cumulative inclusion ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    utmos_means = cumulative_results["utmos_at_k"].mean(axis=1)
    all_rand_mean = np.mean(cumulative_results["all_random"])

    axes[0].plot(codebooks, utmos_means, "o-", color="steelblue", linewidth=2)
    axes[0].axhline(all_rand_mean, color="red", linestyle="--", label="All random", alpha=0.7)
    baseline_mean = np.mean(ablation_results["baseline_scores"])
    axes[0].axhline(baseline_mean, color="green", linestyle="--", label="Baseline (all intact)", alpha=0.7)
    axes[0].set_xlabel("Keep codebooks 0..k")
    axes[0].set_ylabel("UTMOS")
    axes[0].set_title("Cumulative Quality")
    axes[0].set_xticks(codebooks)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    marginal = weight_results["marginal_contribution"]
    axes[1].bar(codebooks, marginal, color="steelblue")
    axes[1].set_xlabel("Codebook index")
    axes[1].set_ylabel("Marginal UTMOS gain")
    axes[1].set_title("Marginal Contribution")
    axes[1].set_xticks(codebooks)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Method 2: Cumulative Codebook Inclusion", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "02_cumulative.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: Entropy + MI ---
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(codebooks, entropy_results["entropies"], color="steelblue", alpha=0.8, label="Entropy")
    ax1.axhline(entropy_results["max_entropy"], color="red", linestyle="--", alpha=0.5, label=f"Max ({entropy_results['max_entropy']:.1f} bits)")
    ax1.set_xlabel("Codebook index")
    ax1.set_ylabel("Shannon Entropy (bits)")
    ax1.set_xticks(codebooks)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    mi = entropy_results["mutual_info"]
    ax2.plot([k + 0.5 for k in range(len(mi))], mi, "s-", color="coral", linewidth=2, label="MI(k, k+1)")
    ax2.set_ylabel("Mutual Information (bits)")
    ax2.legend(loc="upper right")

    fig.suptitle("Method 3: Information-Theoretic Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "03_entropy_mi.png"), dpi=150)
    plt.close(fig)

    # --- Plot 4: Spectral heatmap ---
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        spectral_results["impact_matrix"],
        aspect="auto",
        cmap="hot",
        origin="lower",
    )
    ax.set_xlabel("Mel bin (low → high frequency)")
    ax.set_ylabel("Codebook index")
    ax.set_title("Method 4: Spectral Impact per Codebook")
    ax.set_yticks(codebooks)
    fig.colorbar(im, ax=ax, label="Avg |ΔMel| (dB)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "04_spectral_heatmap.png"), dpi=150)
    plt.close(fig)

    # --- Plot 5: CER ablation ---
    if cer_results is not None and "cer_importance" in weight_results:
        fig, ax = plt.subplots(figsize=(12, 5))
        cer_imp = weight_results["cer_importance"]
        ax.bar(codebooks, cer_imp, color="steelblue")
        ax.set_xlabel("Codebook index")
        ax.set_ylabel("CER increase (higher = more important for intelligibility)")
        ax.set_title("Method 5: CER Ablation (Whisper ASR)")
        ax.set_xticks(codebooks)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "05_cer_ablation.png"), dpi=150)
        plt.close(fig)

    # --- Plot 6: Current vs Suggested weights ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(NUM_CODEBOOKS)
    width = 0.35
    ax.bar(x - width / 2, current_weights, width, label="Current weights", color="steelblue")
    ax.bar(x + width / 2, weight_results["suggested_weights"], width, label="Suggested weights", color="coral")
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Weight")
    ax.set_title("Current vs. Suggested Codebook Weights")
    ax.set_xticks(codebooks)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "06_weight_comparison.png"), dpi=150)
    plt.close(fig)

    logger.info(f"Plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(
    ablation_results: dict,
    cumulative_results: dict,
    entropy_results: dict,
    spectral_results: dict,
    weight_results: dict,
    cer_results: dict | None = None,
):
    """Print a summary table of all results."""
    current_weights = [24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2]
    has_cer = "cer_importance" in weight_results

    width = 104 if has_cer else 90
    print("\n" + "=" * width)
    print("CODEBOOK WEIGHT EVALUATION SUMMARY")
    print("=" * width)

    # Header
    header = (
        f"{'CB':>3} | {'UTMOS Drop':>11} | {'Marginal':>9} | {'Entropy':>8} | "
        f"{'Spectral':>9}"
    )
    if has_cer:
        header += f" | {'CER Incr':>9}"
    header += f" | {'Current':>8} | {'Suggested':>9}"
    print(header)
    print("-" * width)

    ablation_imp = weight_results["ablation_importance"]
    marginal = weight_results["marginal_contribution"]
    entropies = weight_results["entropy_importance"]
    spectral = weight_results["spectral_importance"]
    suggested = weight_results["suggested_weights"]
    cer_imp = weight_results.get("cer_importance")

    for k in range(NUM_CODEBOOKS):
        row = (
            f"{k:>3} | {ablation_imp[k]:>11.4f} | {marginal[k]:>9.4f} | "
            f"{entropies[k]:>8.3f} | {spectral[k]:>9.2f}"
        )
        if has_cer and cer_imp is not None:
            row += f" | {cer_imp[k]:>9.4f}"
        row += f" | {current_weights[k]:>8} | {suggested[k]:>9}"
        print(row)

    print("-" * width)
    footer = f"{'Sum':>3} | {'':>11} | {'':>9} | {'':>8} | {'':>9}"
    if has_cer:
        footer += f" | {'':>9}"
    footer += f" | {sum(current_weights):>8} | {sum(suggested):>9}"
    print(footer)

    baseline_mean = np.mean(ablation_results["baseline_scores"])
    print(f"\nBaseline UTMOS (all codebooks intact): {baseline_mean:.4f}")
    if cer_results is not None:
        valid_cer = [c for c in cer_results["baseline_cer"] if not np.isnan(c)]
        if valid_cer:
            print(f"Baseline CER  (all codebooks intact): {np.mean(valid_cer):.4f}")
    print(f"Current weights:   {current_weights}")
    print(f"Suggested weights: {suggested}")
    print("=" * width + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate codebook weight importance")
    parser.add_argument(
        "--tar-path",
        type=str,
        required=True,
        help="Path to a WebDataset tar shard containing encoded tokens (.npy)",
    )
    parser.add_argument(
        "--utmos-model-path",
        type=str,
        required=True,
        help="Path to UTMOS22Strong checkpoint (.pt)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Path or HF model ID for Qwen3TTSTokenizer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/codebook_eval",
        help="Directory to save plots and results",
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=str,
        default=None,
        help="Path to metadata JSONL (for CER eval). Auto-detected from tar path if not set.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Whisper model for CER evaluation",
    )
    parser.add_argument("--language", type=str, default="ja", help="Language for Whisper ASR")
    parser.add_argument("--skip-cer", action="store_true", help="Skip CER evaluation (Method 5)")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of samples to evaluate")
    parser.add_argument("--num-trials", type=int, default=3, help="Random trials per ablation")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not set)")
    return parser


def main():
    args = get_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    from qwen_tts import Qwen3TTSTokenizer

    logger.info(f"Loading tokenizer from {args.tokenizer_path}...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(args.tokenizer_path, device_map=device)

    # Load UTMOS model
    logger.info(f"Loading UTMOS model from {args.utmos_model_path}...")
    utmos_model = UTMOS22Strong()
    state_dict = torch.load(args.utmos_model_path, map_location="cpu")
    utmos_model.load_state_dict(state_dict)
    utmos_model.to(device)
    utmos_model.eval()

    # Load tokens
    tokens_list, sample_ids = load_tokens_from_tar(args.tar_path, args.num_samples)
    if not tokens_list:
        logger.error("No tokens loaded. Exiting.")
        return

    # Compute empirical distributions for Method 1 extension
    logger.info("Computing empirical token distributions...")
    empirical_dists = compute_empirical_distributions(tokens_list)

    # Method 3: Entropy (no decoding, run first)
    logger.info("Running Method 3: Entropy analysis...")
    entropy_results = run_entropy_analysis(tokens_list)

    # Method 1: Single-codebook ablation
    logger.info("Running Method 1: Single-codebook ablation...")
    ablation_results = run_single_ablation(
        tokenizer, utmos_model, tokens_list, device,
        num_trials=args.num_trials, empirical_dists=empirical_dists,
    )

    # Method 2: Cumulative inclusion
    logger.info("Running Method 2: Cumulative codebook inclusion...")
    cumulative_results = run_cumulative(tokenizer, utmos_model, tokens_list, device)

    # Method 4: Spectral analysis
    logger.info("Running Method 4: Spectral analysis...")
    spectral_results = run_spectral_analysis(tokenizer, tokens_list, device)

    # Method 5: CER ablation
    cer_results = None
    if not args.skip_cer:
        # Resolve metadata JSONL path
        jsonl_path = args.metadata_jsonl
        if jsonl_path is None:
            # Auto-detect: audios/shard-XXXXXX.tar -> txts/shard-XXXXXX.jsonl
            tar_dir = os.path.dirname(args.tar_path)
            tar_basename = os.path.basename(args.tar_path).replace(".tar", ".jsonl")
            jsonl_path = os.path.join(os.path.dirname(tar_dir), "txts", tar_basename)

        if os.path.exists(jsonl_path):
            logger.info(f"Loading metadata from {jsonl_path}...")
            metadata = load_metadata(jsonl_path)

            logger.info(f"Loading Whisper model ({args.whisper_model})...")
            whisper_pipe = load_whisper_pipeline(args.whisper_model, device)

            logger.info("Running Method 5: CER ablation...")
            cer_results = run_cer_ablation(
                tokenizer, whisper_pipe, tokens_list, sample_ids, metadata, device,
                language=args.language,
            )

            # Free Whisper memory
            del whisper_pipe
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            logger.warning(f"Metadata JSONL not found at {jsonl_path}, skipping CER evaluation")

    # Derive weights
    weight_results = derive_weights(
        ablation_results, cumulative_results, entropy_results, spectral_results,
        cer_results=cer_results,
    )

    # Output
    print_summary(
        ablation_results, cumulative_results, entropy_results, spectral_results,
        weight_results, cer_results=cer_results,
    )
    generate_plots(
        ablation_results, cumulative_results, entropy_results, spectral_results,
        weight_results, args.output_dir, cer_results=cer_results,
    )

    # Save raw results
    results_path = os.path.join(args.output_dir, "results.npz")
    save_dict = dict(
        baseline_scores=ablation_results["baseline_scores"],
        ablation_random=ablation_results["ablation_random"],
        ablation_empirical=ablation_results["ablation_empirical"],
        cumulative_utmos=cumulative_results["utmos_at_k"],
        cumulative_all_random=cumulative_results["all_random"],
        entropies=entropy_results["entropies"],
        mutual_info=entropy_results["mutual_info"],
        spectral_impact=spectral_results["impact_matrix"],
        suggested_weights=weight_results["suggested_weights"],
        current_weights=[24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2],
    )
    if cer_results is not None:
        save_dict["baseline_cer"] = cer_results["baseline_cer"]
        save_dict["cer_increase"] = cer_results["cer_increase"]
        save_dict["ablation_cer"] = cer_results["ablation_cer"]
    np.savez(results_path, **save_dict)
    logger.info(f"Raw results saved to {results_path}")


if __name__ == "__main__":
    main()

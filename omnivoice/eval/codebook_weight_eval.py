#!/usr/bin/env python3
"""
Codebook weight importance evaluation for Qwen3 TTS Tokenizer.

Evaluates the relative importance of each codebook layer (0-15) using:
  Method 1: Single-codebook ablation (UTMOS drop per codebook)
  Method 2: Cumulative codebook inclusion (marginal contribution)
  Method 3: Information-theoretic analysis (entropy, mutual information)
  Method 4: Spectral analysis (per-codebook mel-spectrogram impact)

Usage:
  python -m omnivoice.eval.codebook_weight_eval \
    --tar-path /path/to/shard-000000.tar \
    --utmos-model-path tts_eval_models/mos/utmos22_strong_step7459_v1.pt \
    --output-dir results/codebook_eval \
    --num-samples 30 \
    --num-trials 3
"""

import argparse
import io
import logging
import os
import random
import tarfile
from collections import Counter

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
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


def load_tokens_from_tar(tar_path: str, num_samples: int, seed: int = 42) -> list[np.ndarray]:
    """Load token arrays from a WebDataset tar shard.

    Returns list of int16 arrays with shape (16, T).
    """
    all_tokens = []
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if member.name.endswith(".npy"):
                f = tf.extractfile(member)
                if f is not None:
                    buf = io.BytesIO(f.read())
                    tokens = np.load(buf)
                    all_tokens.append(tokens)

    rng = random.Random(seed)
    if len(all_tokens) > num_samples:
        all_tokens = rng.sample(all_tokens, num_samples)
    logger.info(f"Loaded {len(all_tokens)} token arrays from {tar_path}")
    return all_tokens


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
# Weight derivation
# ---------------------------------------------------------------------------


def derive_weights(
    ablation_results: dict,
    cumulative_results: dict,
    entropy_results: dict,
    spectral_results: dict,
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

    def normalize(x):
        s = x.sum()
        return x / s if s > 0 else np.ones_like(x) / len(x)

    # Weighted combination
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

    return {
        "ablation_importance": ablation_importance,
        "marginal_contribution": marginal,
        "entropy_importance": entropies,
        "spectral_importance": spectral_importance,
        "combined_normalized": combined,
        "suggested_weights": suggested.astype(int).tolist(),
    }


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

    # --- Plot 5: Current vs Suggested weights ---
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
    fig.savefig(os.path.join(output_dir, "05_weight_comparison.png"), dpi=150)
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
):
    """Print a summary table of all results."""
    current_weights = [24, 20, 16, 12, 8, 8, 6, 6, 4, 4, 4, 4, 2, 2, 2, 2]

    print("\n" + "=" * 90)
    print("CODEBOOK WEIGHT EVALUATION SUMMARY")
    print("=" * 90)

    # Header
    print(
        f"{'CB':>3} | {'UTMOS Drop':>11} | {'Marginal':>9} | {'Entropy':>8} | "
        f"{'Spectral':>9} | {'Current':>8} | {'Suggested':>9}"
    )
    print("-" * 90)

    ablation_imp = weight_results["ablation_importance"]
    marginal = weight_results["marginal_contribution"]
    entropies = weight_results["entropy_importance"]
    spectral = weight_results["spectral_importance"]
    suggested = weight_results["suggested_weights"]

    for k in range(NUM_CODEBOOKS):
        print(
            f"{k:>3} | {ablation_imp[k]:>11.4f} | {marginal[k]:>9.4f} | "
            f"{entropies[k]:>8.3f} | {spectral[k]:>9.2f} | "
            f"{current_weights[k]:>8} | {suggested[k]:>9}"
        )

    print("-" * 90)
    print(f"{'Sum':>3} | {'':>11} | {'':>9} | {'':>8} | {'':>9} | "
          f"{sum(current_weights):>8} | {sum(suggested):>9}")

    baseline_mean = np.mean(ablation_results["baseline_scores"])
    print(f"\nBaseline UTMOS (all codebooks intact): {baseline_mean:.4f}")
    print(f"Current weights:   {current_weights}")
    print(f"Suggested weights: {suggested}")
    print("=" * 90 + "\n")


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
    tokens_list = load_tokens_from_tar(args.tar_path, args.num_samples)
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

    # Derive weights
    weight_results = derive_weights(
        ablation_results, cumulative_results, entropy_results, spectral_results
    )

    # Output
    print_summary(ablation_results, cumulative_results, entropy_results, spectral_results, weight_results)
    generate_plots(ablation_results, cumulative_results, entropy_results, spectral_results, weight_results, args.output_dir)

    # Save raw results
    results_path = os.path.join(args.output_dir, "results.npz")
    np.savez(
        results_path,
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
    logger.info(f"Raw results saved to {results_path}")


if __name__ == "__main__":
    main()

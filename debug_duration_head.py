"""Debug script for duration head training diagnostics.

Loads a checkpoint and a training batch, then inspects:
1. Boundary extraction correctness (positions, valid_mask)
2. Duration bin targets (histogram of bucketized values)
3. Duration head output distribution (logits, predicted vs actual)
4. Comparison of training-mode vs inference-mode predictions
5. Hidden state comparison between training-path and inference-path
"""

import argparse
import math
import sys

import torch
from transformers import AutoTokenizer

from omnivoice.data.collator import PackingDataCollator
from omnivoice.data.processor import OmniVoiceSampleProcessor
from omnivoice.models.omnivoice import OmniVoice


def load_model(checkpoint_path: str, device: str = "cpu"):
    model = OmniVoice.from_pretrained(
        checkpoint_path,
        device_map=device,
        dtype=torch.float32,
    )
    model.eval()
    return model


def build_synthetic_batch(model, tokenizer, num_samples: int = 4):
    """Build a synthetic packed batch to test boundary extraction."""
    processor = OmniVoiceSampleProcessor(
        text_tokenizer=tokenizer,
        num_channels=model.config.num_audio_codebook,
        audio_mask_id=model.config.audio_mask_id,
        prompt_ratio_range=(0.1, 0.2),
        mask_ratio_range=(0.5, 1.0),
        drop_cond_ratio=0.0,  # no drop_cond so all docs have text boundary
        language_ratio=0.8,
        use_pinyin_ratio=0.0,
        instruct_ratio=1.0,
        only_instruct_ratio=0.5,
    )

    samples = []
    texts = [
        "こんにちは、世界。",
        "Hello, how are you?",
        "今日はいい天気ですね。散歩に行きましょう。",
        "これはテスト文です。",
    ]
    audio_lengths = [25, 38, 62, 20]

    for i in range(num_samples):
        audio_tokens = torch.randint(
            0, model.config.audio_vocab_size - 1,
            (model.config.num_audio_codebook, audio_lengths[i]),
        )
        sample = {
            "audio_tokens": audio_tokens,
            "label": {
                "text": texts[i],
                "language_id": "ja" if i % 2 == 0 else "en",
                "instruct": "None",
            },
        }
        samples.append(processor(sample))

    batch_tokens = sum(s["length"] for s in samples) + 64  # padding
    collator = PackingDataCollator(processor=processor, batch_tokens=batch_tokens)
    batch = collator(samples)
    return batch, texts, audio_lengths


def inspect_boundary_extraction(model, batch):
    """Run _extract_text_boundary_hidden and print diagnostics."""
    print("=" * 60)
    print("1. Boundary Extraction Diagnostics")
    print("=" * 60)

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    audio_mask = batch["audio_mask"].to(device)
    document_ids = batch["document_ids"].to(device)

    # Forward pass to get hidden states
    inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
    llm_outputs = model.llm(
        inputs_embeds=inputs_embeds,
        return_dict=True,
    )
    hidden_states = llm_outputs[0]

    # Extract boundaries
    boundary_hidden, valid_mask = model._extract_text_boundary_hidden(
        hidden_states, audio_mask, document_ids,
    )

    doc_ids = document_ids[0]
    a_mask = audio_mask[0]
    unique_docs = doc_ids[doc_ids >= 0].unique()

    print(f"  Sequence length: {doc_ids.shape[0]}")
    print(f"  Num documents:   {unique_docs.numel()}")
    print(f"  Valid docs:      {valid_mask.sum().item()} / {valid_mask.numel()}")
    print()

    for d in unique_docs:
        doc_pos = (doc_ids == d)
        text_pos = doc_pos & ~a_mask
        audio_pos = doc_pos & a_mask
        text_indices = text_pos.nonzero(as_tuple=False).squeeze(-1)
        audio_indices = audio_pos.nonzero(as_tuple=False).squeeze(-1)

        if text_indices.numel() > 0:
            boundary_idx = text_indices[-1].item()
        else:
            boundary_idx = None

        print(f"  Doc {d.item()}:")
        print(f"    Text positions:  {text_indices.numel()} "
              f"(range {text_indices[0].item()}-{text_indices[-1].item()})"
              if text_indices.numel() > 0 else "    Text positions: 0")
        print(f"    Audio positions: {audio_indices.numel()} "
              f"(range {audio_indices[0].item()}-{audio_indices[-1].item()})"
              if audio_indices.numel() > 0 else "    Audio positions: 0")
        print(f"    Boundary index:  {boundary_idx}")
        print(f"    Valid:           {valid_mask[d.item()].item()}")
    print()

    return hidden_states, boundary_hidden, valid_mask


def inspect_duration_targets(model, batch):
    """Visualize bucketized target distribution."""
    print("=" * 60)
    print("2. Duration Target Binning")
    print("=" * 60)

    num_audio_tokens = batch["num_audio_tokens"].squeeze(0)
    duration_bins = model.duration_bins

    print(f"  Duration bins range: [{duration_bins[0].item():.1f}, "
          f"{duration_bins[-1].item():.1f}]")
    print(f"  Num bins: {duration_bins.numel()}")
    print()

    targets = torch.bucketize(
        num_audio_tokens.float().to(duration_bins.device),
        duration_bins,
    ).clamp(0, model.config.num_duration_bins - 1)

    for i, (n, t) in enumerate(zip(num_audio_tokens.tolist(), targets.tolist())):
        bin_val = duration_bins[t].item()
        print(f"  Doc {i}: num_audio_tokens={n:4d} -> bin_idx={t:3d} "
              f"(bin_center={bin_val:.1f})")
    print()

    return targets


def inspect_duration_head_output(model, boundary_hidden, valid_mask, targets):
    """Check what the duration head actually predicts."""
    print("=" * 60)
    print("3. Duration Head Predictions (training-mode hidden states)")
    print("=" * 60)

    if not valid_mask.any():
        print("  No valid documents - skipping")
        return

    with torch.no_grad():
        dur_logits = model.duration_head(boundary_hidden[valid_mask])

    probs = torch.softmax(dur_logits, dim=-1)
    valid_targets = targets[valid_mask.to(targets.device)]

    for i in range(dur_logits.shape[0]):
        pred_bin = dur_logits[i].argmax().item()
        pred_val = model.duration_bins[pred_bin].item()
        true_bin = valid_targets[i].item()
        true_val = model.duration_bins[true_bin].item()
        target_prob = probs[i, true_bin].item()
        pred_prob = probs[i, pred_bin].item()

        print(f"  Doc {i}:")
        print(f"    Predicted: bin={pred_bin:3d} (value={pred_val:.1f}, "
              f"prob={pred_prob:.4f})")
        print(f"    Actual:    bin={true_bin:3d} (value={true_val:.1f}, "
              f"prob={target_prob:.4f})")
        print(f"    Match:     {'YES' if pred_bin == true_bin else 'NO'}")

        # Top-5 bins
        top5 = torch.topk(probs[i], 5)
        top5_str = ", ".join(
            f"bin {idx.item()}({model.duration_bins[idx].item():.0f})="
            f"{p.item():.3f}"
            for p, idx in zip(top5.values, top5.indices)
        )
        print(f"    Top-5:     {top5_str}")
    print()

    # Cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        dur_logits, valid_targets.to(dur_logits.device)
    )
    print(f"  Cross-entropy loss: {loss.item():.6f}")
    print()


def inspect_inference_mode(model, texts):
    """Run _predict_duration (inference path) and compare."""
    print("=" * 60)
    print("4. Inference-mode Predictions (_predict_duration)")
    print("=" * 60)

    langs = ["ja", "en", "ja", "ja"]

    for i, text in enumerate(texts):
        predicted = model._predict_duration(
            text=text,
            lang=langs[i],
            instruct="None",
        )
        print(f"  Text: {text[:40]:<40s} -> predicted={predicted} tokens")
    print()


def inspect_hidden_state_comparison(model, batch, texts):
    """Compare hidden states from training-path vs inference-path.

    For the first document in the packed batch, extracts the text-only
    tokens and runs a separate text-only forward pass (mimicking the
    inference _predict_duration path). Then compares the hidden states
    at corresponding positions.
    """
    print("=" * 60)
    print("5. Hidden State Comparison (training-path vs inference-path)")
    print("=" * 60)

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    audio_mask = batch["audio_mask"].to(device)
    document_ids = batch["document_ids"].to(device)

    # --- Training path: forward with packed batch (no block mask for simplicity) ---
    inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
    with torch.no_grad():
        train_hidden = model.llm(
            inputs_embeds=inputs_embeds, return_dict=True
        )[0]

    doc_ids = document_ids[0]
    a_mask = audio_mask[0]

    # We'll check document 0 which starts at position 0 (no cross-doc leakage)
    doc0_mask = (doc_ids == 0)
    doc0_text_mask = doc0_mask & ~a_mask
    doc0_text_indices = doc0_text_mask.nonzero(as_tuple=False).squeeze(-1)

    if doc0_text_indices.numel() == 0:
        print("  Doc 0 has no text positions - skipping")
        return

    boundary_idx = doc0_text_indices[-1].item()
    train_boundary_hs = train_hidden[0, boundary_idx, :]

    # --- Inference path: text-only forward (same as _predict_duration) ---
    # Reconstruct the text-only input for doc 0
    # Extract the text token IDs from the packed batch (layer 0)
    text_token_ids = input_ids[0, 0, doc0_text_indices]  # [num_text_tokens]
    text_token_ids = text_token_ids.unsqueeze(0)  # [1, N]
    seq_len = text_token_ids.size(1)

    # Expand to [1, C, N]
    text_ids_expanded = text_token_ids.unsqueeze(0).repeat(
        1, model.config.num_audio_codebook, 1
    )
    text_audio_mask = torch.zeros(
        1, seq_len, dtype=torch.bool, device=device
    )

    text_embeds = model._prepare_embed_inputs(text_ids_expanded, text_audio_mask)
    with torch.no_grad():
        infer_hidden = model.llm(
            inputs_embeds=text_embeds, return_dict=True
        )[0]

    infer_boundary_hs = infer_hidden[0, -1, :]  # last position

    # --- Comparison ---
    cos_sim = torch.nn.functional.cosine_similarity(
        train_boundary_hs.unsqueeze(0),
        infer_boundary_hs.unsqueeze(0),
    ).item()
    l2_dist = (train_boundary_hs - infer_boundary_hs).norm().item()
    train_norm = train_boundary_hs.norm().item()
    infer_norm = infer_boundary_hs.norm().item()

    print(f"  Doc 0 boundary (pos {boundary_idx}):")
    print(f"    Training-path hidden norm:  {train_norm:.4f}")
    print(f"    Inference-path hidden norm: {infer_norm:.4f}")
    print(f"    Cosine similarity:          {cos_sim:.6f}")
    print(f"    L2 distance:                {l2_dist:.4f}")
    print()

    # Also compare duration head outputs for both
    with torch.no_grad():
        train_logits = model.duration_head(train_boundary_hs)
        infer_logits = model.duration_head(infer_boundary_hs)

    train_pred = train_logits.argmax().item()
    infer_pred = infer_logits.argmax().item()
    train_probs = torch.softmax(train_logits, dim=-1)
    infer_probs = torch.softmax(infer_logits, dim=-1)

    print(f"  Duration head on training-path hs:")
    print(f"    Predicted bin: {train_pred} "
          f"(value={model.duration_bins[train_pred].item():.1f}, "
          f"prob={train_probs[train_pred].item():.4f})")
    print(f"  Duration head on inference-path hs:")
    print(f"    Predicted bin: {infer_pred} "
          f"(value={model.duration_bins[infer_pred].item():.1f}, "
          f"prob={infer_probs[infer_pred].item():.4f})")

    logits_cos = torch.nn.functional.cosine_similarity(
        train_logits.unsqueeze(0), infer_logits.unsqueeze(0)
    ).item()
    print(f"    Logits cosine similarity:   {logits_cos:.6f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Debug duration head")
    parser.add_argument(
        "checkpoint",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (cpu, cuda, mps)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device=args.device)

    # Print duration head architecture info
    print(f"  Duration head architecture:")
    for name, param in model.duration_head.named_parameters():
        print(f"    {name}: shape={param.shape}, norm={param.norm().item():.4f}")
    print()

    # Build batch
    tokenizer = model.text_tokenizer
    batch, texts, audio_lengths = build_synthetic_batch(model, tokenizer)
    print(f"Built synthetic batch with {len(texts)} samples")
    print(f"  Audio lengths: {audio_lengths}")
    print()

    # 1. Boundary extraction
    hidden_states, boundary_hidden, valid_mask = inspect_boundary_extraction(
        model, batch,
    )

    # 2. Target binning
    targets = inspect_duration_targets(model, batch)

    # 3. Duration head predictions from training hidden states
    inspect_duration_head_output(model, boundary_hidden, valid_mask, targets)

    # 4. Inference-mode predictions
    inspect_inference_mode(model, texts)

    # 5. Hidden state comparison
    inspect_hidden_state_comparison(model, batch, texts)


if __name__ == "__main__":
    main()

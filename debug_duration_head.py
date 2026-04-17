"""Debug script for duration head training diagnostics.

Loads a checkpoint and a training batch, then inspects:
1. Boundary extraction correctness (positions, valid_mask, <|text_end|> marker)
2. Duration regression targets (linear interpolation in log1p scale)
3. Duration head output distribution (predicted vs actual token counts)
4. Comparison of training-mode vs inference-mode predictions
5. Hidden state comparison between training-path and inference-path
"""

import argparse

import torch

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
        prompt_ratio_range=(0.0, 0.3),
        mask_ratio_range=(0.0, 1.0),
        drop_cond_ratio=0.0,  # no drop_cond so all docs have text boundary
        language_ratio=0.0,
        use_pinyin_ratio=0.0,
        instruct_ratio=0.0,
        only_instruct_ratio=0.0,
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
    """Run forward pass and print text position diagnostics."""
    print("=" * 60)
    print("1. Text Position Diagnostics")
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

    doc_ids = document_ids[0]
    a_mask = audio_mask[0]
    unique_docs = doc_ids[doc_ids >= 0].unique()

    # Check if <|text_end|> token is present at each document boundary
    text_end_token_id = model.text_tokenizer.convert_tokens_to_ids("<|text_end|>")
    layer0_ids = input_ids[0, 0, :]  # text token IDs (layer 0)

    print(f"  Sequence length: {doc_ids.shape[0]}")
    print(f"  Num documents:   {unique_docs.numel()}")
    marker_positions = (layer0_ids == text_end_token_id).nonzero(
        as_tuple=False
    ).squeeze(-1)
    print(f"  <|text_end|> token ID: {text_end_token_id}")
    print(f"  <|text_end|> positions: {marker_positions.tolist()}")
    print()

    for d in unique_docs:
        doc_pos = (doc_ids == d)
        text_pos = doc_pos & ~a_mask
        audio_pos = doc_pos & a_mask
        text_indices = text_pos.nonzero(as_tuple=False).squeeze(-1)
        audio_indices = audio_pos.nonzero(as_tuple=False).squeeze(-1)

        if text_indices.numel() > 0:
            boundary_idx = text_indices[-1].item()
            boundary_token_id = layer0_ids[boundary_idx].item()
            is_marker = boundary_token_id == text_end_token_id
        else:
            boundary_idx = None
            is_marker = False

        print(f"  Doc {d.item()}:")
        print(f"    Text positions:  {text_indices.numel()} "
              f"(range {text_indices[0].item()}-{text_indices[-1].item()})"
              if text_indices.numel() > 0 else "    Text positions: 0")
        print(f"    Audio positions: {audio_indices.numel()} "
              f"(range {audio_indices[0].item()}-{audio_indices[-1].item()})"
              if audio_indices.numel() > 0 else "    Audio positions: 0")
        print(f"    Boundary index:  {boundary_idx}")
        print(f"    Boundary is <|text_end|>: {is_marker}")
    print()

    return hidden_states


def inspect_duration_targets(batch, audio_lengths):
    """Visualize linear interpolation targets."""
    print("=" * 60)
    print("2. Duration Regression Targets (linear interpolation, log1p scale)")
    print("=" * 60)

    doc_ids = batch["document_ids"][0]
    a_mask = batch["audio_mask"][0]
    unique_docs = doc_ids[doc_ids >= 0].unique()

    for d_idx, d in enumerate(unique_docs):
        text_positions = (doc_ids == d) & ~a_mask
        k = text_positions.sum().item()
        n_audio = audio_lengths[d_idx]

        targets_raw = torch.arange(1, k + 1).float() / k * n_audio
        targets_log = targets_raw.log1p()

        print(f"  Doc {d_idx}: num_text={k}, num_audio={n_audio}")
        print(f"    First target:    {targets_raw[0].item():.1f} tokens "
              f"(log1p={targets_log[0].item():.4f})")
        print(f"    Boundary target: {targets_raw[-1].item():.1f} tokens "
              f"(log1p={targets_log[-1].item():.4f})")
    print()


def inspect_duration_head_output(model, hidden_states, batch, audio_lengths):
    """Check what the duration head predicts at each document's boundary."""
    print("=" * 60)
    print("3. Duration Head Predictions (training-mode hidden states)")
    print("=" * 60)

    doc_ids = batch["document_ids"][0]
    a_mask = batch["audio_mask"][0]
    hs = hidden_states[0]
    unique_docs = doc_ids[doc_ids >= 0].unique()

    with torch.no_grad():
        for d_idx, d in enumerate(unique_docs):
            text_positions = (doc_ids == d) & ~a_mask
            text_indices = text_positions.nonzero(as_tuple=False).squeeze(-1)
            if text_indices.numel() == 0:
                continue

            boundary_idx = text_indices[-1].item()
            boundary_hs = hs[boundary_idx]
            log_pred = model.duration_head(boundary_hs).squeeze(-1)
            pred_tokens = torch.expm1(log_pred).item()
            actual_tokens = audio_lengths[d_idx]

            print(f"  Doc {d_idx}:")
            print(f"    Predicted: {pred_tokens:.1f} tokens "
                  f"(log1p={log_pred.item():.4f})")
            print(f"    Actual:    {actual_tokens} tokens "
                  f"(log1p={torch.tensor(float(actual_tokens)).log1p().item():.4f})")
            print(f"    Error:     {abs(pred_tokens - actual_tokens):.1f} tokens")
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
    tokens and runs a separate forward pass mimicking the inference
    _predict_duration path. Then compares the hidden states at the final
    text position (<|text_end|>).
    """
    print("=" * 60)
    print("5. Hidden State Comparison (training-path vs inference-path)")
    print("=" * 60)

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    audio_mask = batch["audio_mask"].to(device)
    document_ids = batch["document_ids"].to(device)

    # --- Training path: forward with packed batch ---
    inputs_embeds = model._prepare_embed_inputs(input_ids, audio_mask)
    with torch.no_grad():
        train_hidden = model.llm(
            inputs_embeds=inputs_embeds, return_dict=True
        )[0]

    doc_ids = document_ids[0]
    a_mask = audio_mask[0]

    # Document 0 starts at position 0 (no cross-doc attention leakage)
    doc0_mask = (doc_ids == 0)
    doc0_text_mask = doc0_mask & ~a_mask
    doc0_text_indices = doc0_text_mask.nonzero(as_tuple=False).squeeze(-1)

    if doc0_text_indices.numel() == 0:
        print("  Doc 0 has no text positions - skipping")
        return

    boundary_idx = doc0_text_indices[-1].item()
    train_boundary_hs = train_hidden[0, boundary_idx, :]

    # --- Inference path: text-only forward with same tokens ---
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

    infer_boundary_hs = infer_hidden[0, -1, :]  # last position = <|text_end|>

    # --- Comparison ---
    cos_sim = torch.nn.functional.cosine_similarity(
        train_boundary_hs.unsqueeze(0),
        infer_boundary_hs.unsqueeze(0),
    ).item()
    l2_dist = (train_boundary_hs - infer_boundary_hs).norm().item()
    train_norm = train_boundary_hs.norm().item()
    infer_norm = infer_boundary_hs.norm().item()

    # Check what token is at the boundary
    text_end_token_id = model.text_tokenizer.convert_tokens_to_ids("<|text_end|>")
    boundary_token = input_ids[0, 0, boundary_idx].item()
    is_marker = boundary_token == text_end_token_id

    print(f"  Doc 0 boundary (pos {boundary_idx}, "
          f"is <|text_end|>: {is_marker}):")
    print(f"    Training-path hidden norm:  {train_norm:.4f}")
    print(f"    Inference-path hidden norm: {infer_norm:.4f}")
    print(f"    Cosine similarity:          {cos_sim:.6f}")
    print(f"    L2 distance:                {l2_dist:.4f}")
    print()

    # Compare duration head outputs for both
    with torch.no_grad():
        train_log_pred = model.duration_head(train_boundary_hs).squeeze(-1)
        infer_log_pred = model.duration_head(infer_boundary_hs).squeeze(-1)

    train_tokens = torch.expm1(train_log_pred).item()
    infer_tokens = torch.expm1(infer_log_pred).item()

    print(f"  Duration head on training-path hs:")
    print(f"    Predicted: {train_tokens:.1f} tokens (log1p={train_log_pred.item():.4f})")
    print(f"  Duration head on inference-path hs:")
    print(f"    Predicted: {infer_tokens:.1f} tokens (log1p={infer_log_pred.item():.4f})")
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

    # 1. Text position diagnostics
    hidden_states = inspect_boundary_extraction(model, batch)

    # 2. Target visualization
    inspect_duration_targets(batch, audio_lengths)

    # 3. Duration head predictions from training hidden states
    inspect_duration_head_output(model, hidden_states, batch, audio_lengths)

    # 4. Inference-mode predictions
    inspect_inference_mode(model, texts)

    # 5. Hidden state comparison
    inspect_hidden_state_comparison(model, batch, texts)


if __name__ == "__main__":
    main()

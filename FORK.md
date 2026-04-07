# Fork: Qwen3 TTS Tokenizer (12Hz) Integration

This branch (`qwen3_tts_tokenizer_model`) replaces the audio tokenizer used in OmniVoice from **Higgs Audio V2** (25Hz, 8 codebooks, vocab 1024) to **Qwen3 TTS Tokenizer 12Hz** (12.5Hz, 16 codebooks, vocab 2048).

## Key Changes

### Audio Tokenizer Swap

- Replaced `HiggsAudioV2TokenizerModel` (from `transformers`) with `Qwen3TTSTokenizer` (from `qwen-tts` package).
- Removed `AutoFeatureExtractor` usage in inference — the Qwen3TTSTokenizer wrapper handles feature extraction internally.
- Encode API: numpy array list + sample rate (`encode([wav_numpy], sr=24000)`) instead of preprocessed tensor input.
- Decode API: wraps output in `Qwen3TTSTokenizerV2EncoderOutput` with shape `(B, T, C)` for the decoder.
- Frame rate changed from 25Hz (integer) to 12.5Hz (float), exposed via `OmniVoice.audio_frame_rate` property.

### Model Configuration

| Parameter | Master (Higgs Audio V2) | This Branch (Qwen3 12Hz) |
|---|---|---|
| `audio_vocab_size` | 1025 | 2049 |
| `audio_mask_id` | 1024 | 2048 |
| `num_audio_codebook` | 8 | 16 |
| `audio_codebook_weights` | `[8,8,6,6,4,4,2,2]` | `[24,20,16,12,8,8,6,6,4,4,4,4,2,2,2,2]` |

The codebook weights are designed so that the upper codebooks (coarse/semantic information) receive a similar normalized weight to the master branch (~18% for codebook 0), compensating for the doubled number of codebooks.

### Training Infrastructure

- Added W&B logging support (`use_wandb`, `wandb_project`, `wandb_run_name`, `wandb_entity` in `TrainingConfig`).
- Added `train/text_tokens` and `train/audio_tokens` metrics to the training loop.
- Dataloader is now prepared via `accelerator.prepare()` (previously only model/optimizer/scheduler were prepared).
- `batch_tokens` reduced from 8192 to 4096 with `gradient_accumulation_steps` increased from 1 to 2 (same effective batch size).

### Token Extraction Scripts

- `extract_audio_tokens.py` / `extract_audio_tokens_add_noise.py`: updated to use `Qwen3TTSTokenizer.encode()` API with numpy inputs.
- `extract_audio_tokens_hf.py`: new script for batch token extraction using HuggingFace `datasets`, with checkpointing and shard-skipping support.

### Dependencies

- Added: `qwen-tts>=0.1.1`, `datasets>=4.8.4`, `torchcodec`, `wandb>=0.25.1`, `flash-attn` (Linux only).
- Relaxed version pins for `torch`, `torchaudio`, `transformers`.

### Other

- Added `fix_mistral_regex=True` to `AutoTokenizer.from_pretrained()` calls.
- Added `prepare.sh` and `train.sh` convenience scripts.
- Duration estimation fallback: `num_ref_audio_tokens` adjusted from 25 (1s at 25Hz) to 13 (1s at 12.5Hz).

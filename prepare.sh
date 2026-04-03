#!/bin/bash
uv run omnivoice/scripts/extract_audio_tokens_hf.py \
    --dataset_name amphion/Emilia-Dataset \
    --data_files "Emilia/EN/EN-*.tar" \
    --split en \
    --streaming True \
    --tar_output_pattern /mnt/artifacts/omnivoice-qwen3tts-tokenizer/emilia/audios/shard-%06d.tar \
    --jsonl_output_pattern /mnt/artifacts/omnivoice-qwen3tts-tokenizer/emilia/txts/shard-%06d.jsonl \
    --tokenizer_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --loader_workers 4 \
    --nj_per_gpu 1 \
    --batch_size 8 \
    --samples_per_shard 10000


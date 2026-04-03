#!/bin/bash
uv run omnivoice/scripts/extract_audio_tokens_hf.py \
    --dataset_name amphion/Emilia-Dataset \
    --data_files "Emilia/EN/EN-B0000??.tar" \
    --split en \
    --streaming True \
    --tar_output_pattern output/audios/shard-%06d.tar \
    --jsonl_output_pattern output/txts/shard-%06d.jsonl \
    --tokenizer_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --loader_workers 1 \
    --nj_per_gpu 1 \
    --batch_size 8

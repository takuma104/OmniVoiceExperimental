uv run omnivoice/scripts/extract_audio_tokens.py \
    --dataset_name amphion/Emilia-Dataset \
    --data_files "Emilia/JA/JA-*.tar" \
    --split ja \
    --streaming True \
    --tar_output_pattern /mnt/artifacts/omnivoice/emilia-ja/audios/shard-%06d.tar \
    --jsonl_output_pattern /mnt/artifacts/omnivoice/emilia-ja/txts/shard-%06d.jsonl \
    --samples_per_shard 10000 \
    --loader_workers 4 \
    --shuffle False



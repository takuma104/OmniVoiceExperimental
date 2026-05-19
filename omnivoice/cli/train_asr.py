#!/usr/bin/env python3
"""Training CLI for OmniVoice ASR."""

import argparse

from omnivoice.training.asr_builder import (
    build_asr_dataloaders,
    build_asr_model_and_tokenizer,
)
from omnivoice.training.config import TrainingConfig
from omnivoice.training.trainer import OmniTrainer
from omnivoice.utils.flex_attention_patch import patch_flex_attention_limited_smem


def main():
    parser = argparse.ArgumentParser(description="OmniVoice ASR Training Entry Point")
    parser.add_argument(
        "--train_config", type=str, required=True, help="Path to config JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save checkpoints"
    )
    parser.add_argument(
        "--data_config", type=str, required=True, help="Path to data config JSON"
    )
    args = parser.parse_args()

    patch_flex_attention_limited_smem()

    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    model, tokenizer = build_asr_model_and_tokenizer(config)
    train_loader, eval_loader = build_asr_dataloaders(config, tokenizer)

    trainer = OmniTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()

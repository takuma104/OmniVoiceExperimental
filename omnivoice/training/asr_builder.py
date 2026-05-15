#!/usr/bin/env python3
"""Builders for OmniVoice ASR training."""

import logging
from functools import partial
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import logging as hf_logging
from transformers.trainer_utils import seed_worker

from omnivoice.data.batching import PackingIterableDataset
from omnivoice.data.collator import ASRPackingDataCollator
from omnivoice.data.dataset import WebDatasetReader, prepare_data_manifests_from_json
from omnivoice.data.processor import OmniVoiceASRSampleProcessor
from omnivoice.models.omnivoice import OmniVoice, OmniVoiceConfig, _resolve_model_path
from omnivoice.models.omnivoice_asr import OmniVoiceForSpeechRecognition
from omnivoice.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def _add_asr_special_tokens(tokenizer):
    new_tokens = [
        "<|asr|>",
        "<|denoise|>",
        "<|lang_start|>",
        "<|lang_end|>",
        "<|instruct_start|>",
        "<|instruct_end|>",
        "<|text_start|>",
        "<|text_end|>",
        "<|src_lang_start|>",
        "<|src_lang_end|>",
        "<|tgt_lang_start|>",
        "<|tgt_lang_end|>",
    ]
    tokens_to_add = [t for t in new_tokens if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})


def build_asr_model_and_tokenizer(
    config: TrainingConfig,
) -> Tuple[OmniVoiceForSpeechRecognition, AutoTokenizer]:
    logger.info("Initializing OmniVoice ASR model & tokenizer...")

    tokenizer_path = (
        config.asr_init_from_asr_checkpoint
        or config.init_from_checkpoint
        or config.llm_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(_resolve_model_path(tokenizer_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _add_asr_special_tokens(tokenizer)

    if config.asr_codebook_mode != "all_sum":
        raise ValueError(
            "Only all-codebook embedding sum is supported for OmniVoice ASR. "
            f"Got asr_codebook_mode={config.asr_codebook_mode!r}."
        )
    if config.asr_audio_embedding_mode not in {
        "all_sum",
        "all_sum_adapter",
        "weighted_sum",
    }:
        raise ValueError(
            "Unsupported ASR audio embedding mode: "
            f"{config.asr_audio_embedding_mode!r}. Expected one of "
            "'all_sum', 'all_sum_adapter', or 'weighted_sum'."
        )
    if config.asr_attention_mode != "prefix_lm":
        raise ValueError(
            "Only prefix_lm attention is supported for OmniVoice ASR. "
            f"Got asr_attention_mode={config.asr_attention_mode!r}."
        )

    if config.asr_init_from_asr_checkpoint:
        logger.info(
            "Loading ASR weights from %s", config.asr_init_from_asr_checkpoint
        )
        model = OmniVoiceForSpeechRecognition.from_pretrained(
            _resolve_model_path(config.asr_init_from_asr_checkpoint),
            attn_implementation="flex_attention",
        )
    elif config.init_from_checkpoint:
        logger.info("Loading OmniVoice base weights from %s", config.init_from_checkpoint)
        model = OmniVoiceForSpeechRecognition.from_omnivoice_pretrained(
            config.init_from_checkpoint,
            audio_embedding_mode=config.asr_audio_embedding_mode,
            audio_adapter_hidden_size=config.asr_audio_adapter_hidden_size,
            attn_implementation="flex_attention",
            dtype=torch.float32,
        )
    else:
        logger.info("Building OmniVoice base from LLM %s", config.llm_name_or_path)
        resolved_llm = _resolve_model_path(config.llm_name_or_path)
        llm_config = AutoConfig.from_pretrained(resolved_llm)
        ov_config = OmniVoiceConfig(
            audio_vocab_size=config.audio_vocab_size,
            audio_mask_id=config.audio_mask_id,
            num_audio_codebook=config.num_audio_codebook,
            audio_codebook_weights=config.audio_codebook_weights,
            llm_config=llm_config,
        )

        original_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        llm = AutoModel.from_pretrained(
            resolved_llm,
            attn_implementation="flex_attention",
            dtype=torch.float32,
        )
        hf_logging.set_verbosity(original_level)
        model = OmniVoiceForSpeechRecognition(
            config=ov_config,
            omnivoice=OmniVoice(config=ov_config, llm=llm),
            audio_embedding_mode=config.asr_audio_embedding_mode,
            audio_adapter_hidden_size=config.asr_audio_adapter_hidden_size,
        )

    model.resize_text_vocab(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if not config.asr_init_from_asr_checkpoint:
        model.init_text_head_from_qwen(config.asr_qwen3_model_path)

    model.set_trainable_modules(
        train_llm_body=config.asr_train_llm_body,
        freeze_text_embedding=config.asr_freeze_text_embedding,
        freeze_text_head=config.asr_freeze_text_head,
        freeze_audio_embeddings=config.asr_freeze_audio_embeddings,
        train_audio_embedding_adapter=config.asr_train_audio_embedding_adapter,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("ASR trainable params: %d / %d", trainable, total)
    return model, tokenizer


def build_asr_dataloaders(
    config: TrainingConfig, tokenizer: AutoTokenizer
) -> Tuple[DataLoader, DataLoader]:
    logger.info("Initializing ASR data readers...")

    processor = OmniVoiceASRSampleProcessor(
        text_tokenizer=tokenizer,
        num_channels=config.num_audio_codebook,
        language_ratio=config.language_ratio,
    )

    train_manifests, dev_manifests = prepare_data_manifests_from_json(
        config.data_config
    )
    raw_train_ds = WebDatasetReader(manifests=train_manifests, evaluation=False)
    train_dataset = PackingIterableDataset(raw_train_ds, processor, config.batch_tokens)
    collate_fn = ASRPackingDataCollator(processor, config.batch_tokens)

    init_fn = partial(
        seed_worker,
        num_workers=config.num_workers,
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        pin_memory=True,
        prefetch_factor=4,
    )

    eval_loader = None
    if dev_manifests:
        raw_dev_ds = WebDatasetReader(manifests=dev_manifests, evaluation=True)
        dev_dataset = PackingIterableDataset(raw_dev_ds, processor, config.batch_tokens)
        eval_loader = DataLoader(
            dev_dataset,
            batch_size=None,
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=2,
        )

    return train_loader, eval_loader

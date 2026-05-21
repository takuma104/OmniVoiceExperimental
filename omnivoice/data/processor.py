#!/usr/bin/env python3
# Copyright    2026  Xiaomi Corp.        (authors:  Han Zhu)
#
# See ../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training sample processor for OmniVoice.

Converts raw audio/text samples into model-ready tensors: applies prompt/mask
tokenization, randomly drops conditioning, and injects language/instruct tokens.
Used by ``omnivoice.training.builder`` to build the data pipeline.

Contains two processor classes:
- ``OmniVoiceSampleProcessor``: Full processor used for training.
- ``OmniVoiceSimpleSampleProcessor``: Simplified processor (not used for training).
"""

import random
from difflib import SequenceMatcher
from typing import Any, Dict

import torch

ASR_TASK_PLAIN = "plain"
ASR_TASK_FURIGANA_AUDIO = "furigana_audio"
ASR_TASK_FURIGANA_REWRITE = "furigana_rewrite"
ASR_TASK_SAMPLE = "sample"
ASR_TASK_MODES = {
    ASR_TASK_PLAIN,
    ASR_TASK_FURIGANA_AUDIO,
    ASR_TASK_FURIGANA_REWRITE,
}


class OmniVoiceSampleProcessor:
    """
    Handles the logic of processing a raw sample into tensors
    (masking, tokenization, etc.).
    """

    def __init__(
        self,
        text_tokenizer: Any,
        num_channels: int,
        audio_mask_id: int,
        prompt_ratio_range: tuple,
        mask_ratio_range: tuple,
        drop_cond_ratio: float,
        language_ratio: float,
        use_pinyin_ratio: float,
        instruct_ratio: float,
        only_instruct_ratio: float,
    ):
        self.text_tokenizer = text_tokenizer
        self.num_channels = num_channels
        self.audio_mask_id = audio_mask_id
        self.prompt_ratio_range = prompt_ratio_range
        self.mask_ratio_range = mask_ratio_range
        self.drop_cond_ratio = drop_cond_ratio

        self.language_ratio = language_ratio
        self.use_pinyin_ratio = use_pinyin_ratio
        self.instruct_ratio = instruct_ratio
        self.only_instruct_ratio = only_instruct_ratio

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        # clean_start_token_idx is only used for prompt denoising training,
        # where the prompt region is augmented with noises and the model
        # needs to learn to recover the clean prompt.
        # clean_start_token_idx indicates the start index of the clean generated token.
        if "clean_start_token_idx" in sample["label"]:
            drop_cond = False
        else:
            drop_cond = random.uniform(0, 1) < self.drop_cond_ratio

        if drop_cond:
            prompt_ratio = 0.0
            drop_text = True
            use_language = False
            use_instruct = False
        else:
            prompt_ratio = random.uniform(*self.prompt_ratio_range)
            drop_text = False
            use_language = random.uniform(0, 1) < self.language_ratio
            use_instruct = random.uniform(0, 1) < self.instruct_ratio
            if use_instruct and random.uniform(0, 1) < self.only_instruct_ratio:
                prompt_ratio = 0.0

        mask_ratio = random.uniform(*self.mask_ratio_range)

        # --- Style ---
        style = ""
        if use_language:
            language = sample["label"].get("language_id", "None")
        else:
            language = "None"
        if use_instruct:
            instruct = sample["label"].get("instruct", "None")
        else:
            instruct = "None"

        if "clean_start_token_idx" in sample["label"]:
            style += "<|denoise|>"

        style += f"<|lang_start|>{language}<|lang_end|>"
        style += f"<|instruct_start|>{instruct}<|instruct_end|>"

        style_inputs = self.text_tokenizer(style, return_tensors="pt").input_ids.repeat(
            self.num_channels, 1
        )
        style_labels = torch.full(
            style_inputs.shape, -100
        )  # Style prompt does not compute loss

        # --- Text ---
        if (
            "text_pinyin" in sample["label"]
            and random.uniform(0, 1) < self.use_pinyin_ratio
        ):
            text = sample["label"]["text_pinyin"]
        else:
            text = sample["label"]["text"]
        text_inputs = self.text_tokenizer(
            f"<|text_start|>{text}<|text_end|>", return_tensors="pt"
        ).input_ids.repeat(self.num_channels, 1)
        text_labels = torch.full(text_inputs.shape, -100)  # Text does not compute loss

        # --- Audio ---
        audio_tokens = sample["audio_tokens"].long()
        if audio_tokens.dim() == 3:
            audio_tokens = audio_tokens.squeeze(0)

        # Masking Logic
        if "clean_start_token_idx" in sample["label"]:
            prompt_length = sample["label"]["clean_start_token_idx"]
        else:
            prompt_length = int(audio_tokens.shape[1] * prompt_ratio)

        audio_inputs = audio_tokens.clone()
        audio_labels = audio_tokens.clone()

        # Apply masking
        maskable_region = audio_tokens[:, prompt_length:]
        token_mask = torch.rand(maskable_region.shape) < mask_ratio
        audio_inputs[:, prompt_length:][token_mask] = self.audio_mask_id
        audio_labels[:, prompt_length:][
            ~token_mask
        ] = -100  # Only compute loss on masked tokens
        if not drop_cond:
            audio_labels[:, :prompt_length] = -100  # No loss on prompt region

        # --- Concatenation ---
        if drop_text:
            input_ids = audio_inputs
            labels = audio_labels
            total_length = input_ids.shape[1]
            audio_mask = torch.ones(total_length, dtype=torch.bool)
        else:
            input_ids = torch.cat([style_inputs, text_inputs, audio_inputs], dim=1)
            labels = torch.cat([style_labels, text_labels, audio_labels], dim=1)
            total_length = input_ids.shape[1]
            audio_start_idx = style_inputs.shape[1] + text_inputs.shape[1]
            audio_mask = torch.zeros(total_length, dtype=torch.bool)
            audio_mask[audio_start_idx:] = True

        return_dict = {
            "input_ids": input_ids,  # [C, L]
            "labels": labels,  # [C, L]
            "audio_mask": audio_mask,  # [L]
            "length": total_length,
        }

        return return_dict


class OmniVoiceSimpleSampleProcessor:
    """
    Handles the logic of processing a raw sample into tensors
    (masking, tokenization, etc.).
    This is a simpler version that does not include language, instructions,
        or denoising prompts.
    We do not use it for training as OmniVoiceSampleProcessor can cover this case.
    We keep it as a reference implementation for users to understand the basic logics.
    """

    def __init__(
        self,
        text_tokenizer: Any,
        num_channels: int,
        audio_mask_id: int,
        prompt_ratio_range: tuple,
        mask_ratio_range: tuple,
        drop_cond_ratio: float,
    ):
        self.text_tokenizer = text_tokenizer
        self.num_channels = num_channels
        self.audio_mask_id = audio_mask_id
        self.prompt_ratio_range = prompt_ratio_range
        self.mask_ratio_range = mask_ratio_range
        self.drop_cond_ratio = drop_cond_ratio

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        drop_cond = random.uniform(0, 1) < self.drop_cond_ratio
        mask_ratio = random.uniform(*self.mask_ratio_range)

        if drop_cond:
            prompt_ratio = 0.0
        else:
            prompt_ratio = random.uniform(*self.prompt_ratio_range)

        # --- Text ---
        text = sample["label"]["text"]
        text_inputs = self.text_tokenizer(
            f"<|text_start|>{text}<|text_end|>", return_tensors="pt"
        ).input_ids.repeat(self.num_channels, 1)
        text_labels = torch.full(text_inputs.shape, -100)  # Text does not compute loss

        # --- Audio ---
        audio_tokens = sample["audio_tokens"].long()

        # Masking Logic
        prompt_length = int(audio_tokens.shape[1] * prompt_ratio)
        audio_inputs = audio_tokens.clone()
        audio_labels = audio_tokens.clone()

        # Apply masking
        maskable_region = audio_tokens[:, prompt_length:]
        token_mask = torch.rand(maskable_region.shape) < mask_ratio
        audio_inputs[:, prompt_length:][token_mask] = self.audio_mask_id
        audio_labels[:, prompt_length:][
            ~token_mask
        ] = -100  # Only compute loss on masked tokens

        if not drop_cond:
            # No loss on prompt region
            audio_labels[:, :prompt_length] = -100

        # --- Concatenation ---
        if drop_cond:
            input_ids = audio_inputs
            labels = audio_labels
            total_length = input_ids.shape[1]
            audio_mask = torch.ones(total_length, dtype=torch.bool)
        else:
            input_ids = torch.cat([text_inputs, audio_inputs], dim=1)
            labels = torch.cat([text_labels, audio_labels], dim=1)
            total_length = input_ids.shape[1]
            audio_start_idx = text_inputs.shape[1]
            audio_mask = torch.zeros(total_length, dtype=torch.bool)
            audio_mask[audio_start_idx:] = True

        return_dict = {
            "input_ids": input_ids,  # [C, L]
            "labels": labels,  # [C, L]
            "audio_mask": audio_mask,  # [L]
            "length": total_length,
        }

        return return_dict


class OmniVoiceASRSampleProcessor:
    """Prepare OmniVoice codec tokens for autoregressive ASR training.

    Supported task layouts:

    - ``plain``: ``<|asr|> audio -> text``
    - ``furigana_audio``: ``<|asr_furigana|> audio -> text_fugashi``
    - ``furigana_rewrite``:
      ``<|furigana_rewrite|> audio + source text -> text_fugashi``
    """

    def __init__(
        self,
        text_tokenizer: Any,
        num_channels: int,
        language_ratio: float = 1.0,
        timestamp_enabled: bool = False,
        timestamp_min_confidence: float = 0.0,
        task_mode: str = ASR_TASK_SAMPLE,
        plain_ratio: float = 1.0,
        furigana_audio_ratio: float = 0.0,
        furigana_rewrite_ratio: float = 0.0,
        text_field: str = "text",
        furigana_text_field: str = "text_fugashi",
        source_text_field: str = "text",
    ):
        self.text_tokenizer = text_tokenizer
        self.num_channels = num_channels
        self.language_ratio = language_ratio
        self.timestamp_enabled = timestamp_enabled
        self.timestamp_min_confidence = timestamp_min_confidence
        self.task_mode = task_mode
        self.plain_ratio = plain_ratio
        self.furigana_audio_ratio = furigana_audio_ratio
        self.furigana_rewrite_ratio = furigana_rewrite_ratio
        self.text_field = text_field
        self.furigana_text_field = furigana_text_field
        self.source_text_field = source_text_field

        if self.task_mode != ASR_TASK_SAMPLE and self.task_mode not in ASR_TASK_MODES:
            raise ValueError(
                f"Unsupported ASR task mode: {self.task_mode!r}. "
                f"Expected one of {sorted(ASR_TASK_MODES | {ASR_TASK_SAMPLE})}."
            )
        self._task_weights = [
            (ASR_TASK_PLAIN, float(self.plain_ratio)),
            (ASR_TASK_FURIGANA_AUDIO, float(self.furigana_audio_ratio)),
            (ASR_TASK_FURIGANA_REWRITE, float(self.furigana_rewrite_ratio)),
        ]
        if self.task_mode == ASR_TASK_SAMPLE and not any(
            weight > 0 for _, weight in self._task_weights
        ):
            raise ValueError(
                "At least one ASR task ratio must be > 0 when task_mode='sample'."
            )

    def _choose_task_mode(self) -> str:
        if self.task_mode != ASR_TASK_SAMPLE:
            return self.task_mode

        total = sum(max(weight, 0.0) for _, weight in self._task_weights)
        draw = random.uniform(0.0, total)
        cumulative = 0.0
        for mode, weight in self._task_weights:
            if weight <= 0:
                continue
            cumulative += weight
            if draw <= cumulative:
                return mode
        return ASR_TASK_PLAIN

    @staticmethod
    def _task_token(task_mode: str) -> str:
        if task_mode == ASR_TASK_PLAIN:
            return "<|asr|>"
        if task_mode == ASR_TASK_FURIGANA_AUDIO:
            return "<|asr_furigana|>"
        if task_mode == ASR_TASK_FURIGANA_REWRITE:
            return "<|furigana_rewrite|>"
        raise ValueError(f"Unsupported ASR task mode: {task_mode!r}")

    @staticmethod
    def _require_text(label: Dict[str, Any], field: str, task_mode: str) -> str:
        if field not in label:
            sample_id = label.get("id", "?")
            raise KeyError(
                f"Sample {sample_id!r} is missing required field {field!r} "
                f"for ASR task mode {task_mode!r}."
            )
        return str(label[field])

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        label = sample["label"]
        task_mode = self._choose_task_mode()
        use_language = random.uniform(0, 1) < self.language_ratio

        style = self._task_token(task_mode)
        if use_language:
            language = label.get("language_id", "None")
            style += f"<|lang_start|>{language}<|lang_end|>"

        style_ids = self.text_tokenizer(style, return_tensors="pt").input_ids
        style_inputs = style_ids.repeat(self.num_channels, 1)

        audio_tokens = sample["audio_tokens"].long()
        if audio_tokens.dim() == 3:
            audio_tokens = audio_tokens.squeeze(0)
        if audio_tokens.size(0) != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} audio codebooks, "
                f"got {audio_tokens.size(0)}"
            )

        if task_mode == ASR_TASK_PLAIN:
            text = self._require_text(label, self.text_field, task_mode)
        else:
            text = self._require_text(label, self.furigana_text_field, task_mode)
        text_ids = self.text_tokenizer(
            f"<|text_start|>{text}<|text_end|>", return_tensors="pt"
        ).input_ids
        text_inputs = text_ids.repeat(self.num_channels, 1)

        sequence_parts = [style_inputs, audio_tokens]
        if task_mode == ASR_TASK_FURIGANA_REWRITE:
            source_text = self._require_text(label, self.source_text_field, task_mode)
            source_ids = self.text_tokenizer(
                f"<|src_text_start|>{source_text}<|src_text_end|>",
                return_tensors="pt",
            ).input_ids
            sequence_parts.append(source_ids.repeat(self.num_channels, 1))

        text_start_idx = sum(part.shape[1] for part in sequence_parts)
        sequence_parts.append(text_inputs)
        input_ids = torch.cat(sequence_parts, dim=1)
        total_length = input_ids.shape[1]
        audio_start_idx = style_inputs.shape[1]
        audio_end_idx = audio_start_idx + audio_tokens.shape[1]

        audio_mask = torch.zeros(total_length, dtype=torch.bool)
        audio_mask[audio_start_idx:audio_end_idx] = True

        text_causal_mask = torch.zeros(total_length, dtype=torch.bool)
        text_causal_mask[text_start_idx:] = True

        labels = torch.full((total_length,), -100, dtype=torch.long)
        # The first text token is <|text_start|>. It is an input prompt token;
        # following text positions are predicted autoregressively.
        labels[text_start_idx + 1 :] = text_ids.squeeze(0)[1:]

        return_dict = {
            "input_ids": input_ids,  # [C, L]
            "labels": labels,  # [L]
            "audio_mask": audio_mask,  # [L]
            "text_causal_mask": text_causal_mask,  # [L]
            "length": total_length,
            "task_mode": task_mode,
        }

        timestamp = sample.get("timestamp")
        if timestamp is not None and task_mode == ASR_TASK_PLAIN:
            return_dict["timestamp_center_labels"] = (
                self._build_timestamp_center_labels(
                    timestamp=timestamp,
                    text_token_ids=text_ids.squeeze(0)[1:-1].tolist(),
                    text_start_idx=text_start_idx,
                    total_length=total_length,
                    audio_num_tokens=audio_tokens.shape[1],
                )
            )
        elif self.timestamp_enabled:
            return_dict["timestamp_center_labels"] = torch.full(
                (total_length,),
                -100,
                dtype=torch.long,
            )

        return return_dict

    def _build_timestamp_center_labels(
        self,
        timestamp: Dict[str, Any],
        text_token_ids: list[int],
        text_start_idx: int,
        total_length: int,
        audio_num_tokens: int,
    ) -> torch.Tensor:
        labels = torch.full((total_length,), -100, dtype=torch.long)
        timestamp_tokens = timestamp.get("tokens") or []
        timestamp_token_pairs = [
            (idx, int(token_item["token_id"]))
            for idx, token_item in enumerate(timestamp_tokens)
            if token_item.get("token_id") is not None
        ]
        timestamp_token_ids = [token_id for _, token_id in timestamp_token_pairs]
        matcher = SequenceMatcher(
            a=text_token_ids,
            b=timestamp_token_ids,
            autojunk=False,
        )

        for block in matcher.get_matching_blocks():
            for offset in range(block.size):
                text_idx = block.a + offset
                timestamp_idx = timestamp_token_pairs[block.b + offset][0]
                token_item = timestamp_tokens[timestamp_idx]

                confidence = token_item.get("confidence")
                if (
                    confidence is not None
                    and float(confidence) < self.timestamp_min_confidence
                ):
                    continue

                center = token_item.get("center_audio_token")
                if center is None:
                    continue
                center = int(center)
                if center < 0 or center >= audio_num_tokens:
                    continue

                labels[text_start_idx + text_idx] = center

        return labels

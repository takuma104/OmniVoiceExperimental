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

"""Training loop for OmniVoice.

Wraps the HuggingFace Accelerate training loop with checkpoint saving/resuming,
evaluation, gradient accumulation, and learning rate scheduling.
Launched via ``omnivoice.cli.train``.
"""

import logging
import math
import os
import sys
import time
from datetime import timedelta
from typing import Any, Optional

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin, InitProcessGroupKwargs, set_seed
from torch.utils.data import DataLoader
from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from omnivoice.training.checkpoint import TrainLogger, load_checkpoint
from omnivoice.training.checkpoint import save_checkpoint as engine_save_checkpoint

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    """Move all tensors in a batch dict to the target device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class OmniTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,  # TrainingConfig
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # 1. Initialize Accelerator
        self.accelerator = self._init_accelerator()

        # 2. Setup Optimizer & Scheduler if not provided
        if optimizer is None:
            self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler()
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        # 3. DeepSpeed Hack (Batch Size fix)
        if self.accelerator.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = 1

        # 4. Prepare with Accelerator
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
        )

        if self.eval_dataloader is not None:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

        self.global_step = 0
        self.epoch = 0

        # Optional vocoder used for eval-time mel-sample logging.
        self.vocoder = None
        if (
            getattr(self.config, "mel_mode", False)
            and getattr(self.config, "eval_log_samples", 0) > 0
            and self.accelerator.is_main_process
        ):
            self._load_vocoder_for_logging()

    def _load_vocoder_for_logging(self):
        weights_dir = getattr(self.config, "bigvgan_weights_dir", None)
        if not weights_dir or not os.path.isdir(weights_dir):
            logger.warning(
                f"eval_log_samples > 0 but bigvgan_weights_dir is not a valid "
                f"directory ({weights_dir!r}); mel-sample logging disabled."
            )
            return
        try:
            from bigvgan import BigVGAN

            vocoder = BigVGAN.from_pretrained(weights_dir, use_cuda_kernel=False)
            vocoder = vocoder.to(self.accelerator.device).eval()
            vocoder.remove_weight_norm()
            self.vocoder = vocoder
            logger.info(
                f"Loaded BigVGAN vocoder from {weights_dir} for eval logging."
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Failed to load BigVGAN vocoder ({e}); mel-sample logging disabled."
            )

    def _init_accelerator(self) -> Accelerator:
        """Initialize Accelerator, DeepSpeed, and Logging."""
        # TF32 setup
        if getattr(self.config, "allow_tf32", False):
            torch.set_float32_matmul_precision("high")

        # Init handlers
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=60))

        # DeepSpeed setup
        deepspeed_plugin = None
        if self.config.use_deepspeed and self.config.deepspeed_config:
            if not os.path.exists(self.config.deepspeed_config):
                raise FileNotFoundError(
                    f"DeepSpeed config not found: {self.config.deepspeed_config}"
                )
            deepspeed_plugin = DeepSpeedPlugin(
                hf_ds_config=self.config.deepspeed_config,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                gradient_clipping=self.config.max_grad_norm,
            )

        log_with = "tensorboard"
        if getattr(self.config, "use_wandb", False):
            log_with = "wandb"

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with=log_with,
            project_dir=self.config.output_dir,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs, init_kwargs],
            deepspeed_plugin=deepspeed_plugin,
            split_batches=False,
        )

        # Logging setup
        if accelerator.is_main_process:
            os.makedirs(self.config.output_dir, exist_ok=True)
            # Try to save config if it has the method
            if hasattr(self.config, "save_to_json"):
                self.config.save_to_json(
                    os.path.join(self.config.output_dir, "initial_config.json")
                )

            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                datefmt="%m/%d/%Y %H:%M:%S",
                level=logging.INFO,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(
                        os.path.join(self.config.output_dir, "train.log")
                    ),
                ],
            )
        else:
            logging.basicConfig(level=logging.ERROR)

        logger.info(f"Loaded Config: {self.config}")
        set_seed(self.config.seed)

        if log_with == "wandb":
            accelerator.init_trackers(
                project_name=self.config.wandb_project,
                config=vars(self.config) if hasattr(self.config, "__dict__") else None,
                init_kwargs={
                    "wandb": {
                        "name": self.config.wandb_run_name,
                        "entity": self.config.wandb_entity,
                        "dir": self.config.output_dir,
                    }
                },
            )
        else:
            accelerator.init_trackers(
                "omnivoice",
                config=vars(self.config) if hasattr(self.config, "__dict__") else None,
            )
        return accelerator

    def _build_muon_optimizer(self):
        """Muon for LLM hidden weights, AdamW for everything else.

        Partitioning follows the Muon README: 2D+ hidden weights inside the
        LLM transformer body (excluding embeddings and the LM head) are
        optimized by Muon; all other parameters (LLM embeddings, any LLM
        biases/norms, and non-LLM modules like audio_embeddings and
        audio_heads) are optimized by AdamW.
        """
        try:
            from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
        except ImportError as e:
            raise ImportError(
                "use_muon_optimizer requires the muon package. "
                "Install it with: pip install git+https://github.com/KellerJordan/Muon"
            ) from e

        muon_params = []
        aux_params = []

        llm_param_ids = set()
        for name, p in self.model.llm.named_parameters():
            llm_param_ids.add(id(p))
            if not p.requires_grad:
                continue
            lname = name.lower()
            is_embed_or_head = "embed" in lname or "lm_head" in lname
            if p.ndim >= 2 and not is_embed_or_head:
                muon_params.append(p)
            else:
                aux_params.append(p)

        for _, p in self.model.named_parameters():
            if id(p) in llm_param_ids:
                continue
            if not p.requires_grad:
                continue
            aux_params.append(p)

        param_groups = [
            dict(
                params=aux_params,
                use_muon=False,
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                eps=1e-10,
                weight_decay=self.config.weight_decay,
            ),
            dict(
                params=muon_params,
                use_muon=True,
                lr=self.config.muon_lr,
                momentum=self.config.muon_momentum,
                weight_decay=self.config.muon_weight_decay,
            ),
        ]

        is_distributed = self.accelerator.num_processes > 1
        cls = MuonWithAuxAdam if is_distributed else SingleDeviceMuonWithAuxAdam
        optimizer = cls(param_groups)
        logger.info(
            "Using Muon optimizer (%s): %d hidden weight params (Muon), "
            "%d aux params (AdamW).",
            cls.__name__,
            len(muon_params),
            len(aux_params),
        )
        return optimizer

    def create_optimizer_and_scheduler(self):
        """Default AdamW + configurable LR Scheduler."""
        if getattr(self.config, "use_muon_optimizer", False):
            optimizer = self._build_muon_optimizer()
        elif getattr(self.config, "use_8bit_optimizer", False):
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "use_8bit_optimizer requires bitsandbytes. "
                    "Install it with: pip install bitsandbytes"
                )
            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            logger.info("Using 8-bit AdamW optimizer (bitsandbytes).")
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        if self.config.warmup_type == "ratio":
            final_warmup_steps = math.ceil(self.config.steps * self.config.warmup_ratio)
        else:
            final_warmup_steps = self.config.warmup_steps

        if self.config.lr_scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
            )
        else:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=final_warmup_steps,
                num_training_steps=self.config.steps,
            )
        return optimizer, lr_scheduler

    def save_checkpoint(self, step):
        """Wrapper for engine save_checkpoint."""
        engine_save_checkpoint(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.config.output_dir,
            step,
            self.config.keep_last_n_checkpoints,
        )
        # Save config copy for convenience
        if self.accelerator.is_main_process and hasattr(self.config, "save_to_json"):
            checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
            self.config.save_to_json(os.path.join(checkpoint_dir, "train_config.json"))

    def load_checkpoint(self, checkpoint_path):
        """Wrapper for loading."""
        step = load_checkpoint(self.accelerator, checkpoint_path)
        self.global_step = step
        logger.info(f"Resumed from step {self.global_step}")
        return step

    def evaluate(self):
        """Evaluation loop."""
        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        logger.info(f"Running evaluation at step {self.global_step}...")

        local_loss_sum = torch.tensor(0.0, device=self.accelerator.device)
        eval_count = 0

        log_samples = (
            self.vocoder is not None
            and self.accelerator.is_main_process
            and getattr(self.config, "eval_log_samples", 0) > 0
        )
        first_batch = None
        first_outputs = None

        with torch.no_grad():
            for eval_batch in self.eval_dataloader:
                eval_batch = _to_device(eval_batch, self.accelerator.device)
                outputs = self.model(**eval_batch)
                local_loss_sum += outputs.loss.detach()
                eval_count += 1
                if log_samples and first_batch is None:
                    first_batch = eval_batch
                    first_outputs = outputs

        if eval_count > 0:
            local_mean = local_loss_sum / eval_count
        else:
            local_mean = torch.tensor(0.0, device=self.accelerator.device)

        all_means = self.accelerator.gather(local_mean)
        final_eval_loss = all_means.mean().item()

        eval_metrics = {"eval/loss": final_eval_loss}
        self.accelerator.log(eval_metrics, step=self.global_step)
        logger.info(f"Eval Loss: {final_eval_loss:.4f}")

        if log_samples and first_batch is not None:
            try:
                self._log_mel_eval_samples(first_batch, first_outputs.logits)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to log mel eval samples: {e}")

        self.accelerator.wait_for_everyone()
        self.model.train()
        return eval_metrics

    def _log_mel_eval_samples(self, batch, mel_pred):
        """Log mel-spectrogram images and vocoded audio for a few eval samples.

        Splits the packed eval batch by ``document_ids`` and logs the first
        ``eval_log_samples`` documents. For each document we log:
          - target mel image
          - "filled" mel image (target with masked positions replaced by pred)
          - target audio (vocoded by BigVGAN)
          - filled audio (vocoded by BigVGAN)
        """
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image

        document_ids = batch["document_ids"][0]  # [L]
        audio_mask = batch["audio_mask"][0].bool()  # [L]
        mel_target = batch["mel_target"][0]  # [L, M]
        mel_mask = batch["mel_mask"][0].bool()  # [L]
        pred = mel_pred[0]  # [L, M]

        unique_docs = [d.item() for d in document_ids.unique() if d.item() != -1]
        n = min(getattr(self.config, "eval_log_samples", 0), len(unique_docs))
        if n == 0:
            return

        sr = int(getattr(self.config, "mel_sample_rate", 44100))
        images = {}
        audios = {}

        for idx, doc_id in enumerate(unique_docs[:n]):
            doc_sel = (document_ids == doc_id) & audio_mask
            if doc_sel.sum().item() < 2:
                continue
            mel_t = mel_target[doc_sel].float()  # [T, M]
            mel_p = pred[doc_sel].float()  # [T, M]
            m_mask = mel_mask[doc_sel]  # [T]

            mel_filled = mel_t.clone()
            if m_mask.any():
                mel_filled[m_mask] = mel_p[m_mask]

            # --- Mel image: 2 stacked panels (target / filled-pred). ---
            fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            t_np = mel_t.transpose(0, 1).cpu().numpy()
            f_np = mel_filled.transpose(0, 1).cpu().numpy()
            vmin = float(min(t_np.min(), f_np.min()))
            vmax = float(max(t_np.max(), f_np.max()))
            axes[0].imshow(
                t_np, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
            )
            axes[0].set_title(f"sample {idx} target (T={mel_t.shape[0]})")
            axes[0].set_ylabel("mel bin")
            axes[1].imshow(
                f_np, aspect="auto", origin="lower", vmin=vmin, vmax=vmax
            )
            mask_pct = 100.0 * m_mask.float().mean().item()
            axes[1].set_title(f"sample {idx} filled-pred (mask={mask_pct:.1f}%)")
            axes[1].set_ylabel("mel bin")
            axes[1].set_xlabel("frame")
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            images[f"eval/mel_sample_{idx}"] = np.array(Image.open(buf))

            # --- Vocode target and filled-pred. ---
            with torch.no_grad():
                voc_dtype = next(self.vocoder.parameters()).dtype
                voc_device = next(self.vocoder.parameters()).device
                wav_t = self.vocoder(
                    mel_t.transpose(0, 1).unsqueeze(0).to(voc_device, voc_dtype)
                )
                wav_f = self.vocoder(
                    mel_filled.transpose(0, 1).unsqueeze(0).to(voc_device, voc_dtype)
                )
            wav_t = wav_t.squeeze().float().cpu().numpy()
            wav_f = wav_f.squeeze().float().cpu().numpy()
            audios[f"eval/audio_sample_{idx}_target"] = wav_t
            audios[f"eval/audio_sample_{idx}_pred"] = wav_f

        if not images and not audios:
            return

        # Send to wandb tracker if available, otherwise fall back to disk dump.
        tracker = None
        try:
            tracker = self.accelerator.get_tracker("wandb", unwrap=True)
        except Exception:  # noqa: BLE001
            tracker = None

        if tracker is not None:
            import wandb

            payload = {}
            for k, v in images.items():
                payload[k] = wandb.Image(v)
            for k, v in audios.items():
                payload[k] = wandb.Audio(v, sample_rate=sr)
            tracker.log(payload, step=self.global_step)
        else:
            # Fallback: dump under output_dir/eval_samples/step_{N}/
            out_root = os.path.join(
                self.config.output_dir,
                "eval_samples",
                f"step_{self.global_step}",
            )
            os.makedirs(out_root, exist_ok=True)
            for k, v in images.items():
                Image.fromarray(v).save(
                    os.path.join(out_root, k.replace("/", "_") + ".png")
                )
            try:
                import soundfile as sf

                for k, v in audios.items():
                    sf.write(
                        os.path.join(out_root, k.replace("/", "_") + ".wav"),
                        v,
                        sr,
                    )
            except ImportError:
                logger.warning("soundfile not available; skipping wav dump.")

    def train(self):
        """Main training loop."""
        logger.info("Starting Training Loop...")

        # Resume if configured
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)

        # Handle IterableDataset Epochs
        if hasattr(self.train_dataloader.dataset, "set_epoch"):
            self.train_dataloader.dataset.set_epoch(self.epoch)

        # Logger
        train_logger = TrainLogger(
            self.accelerator, self.config.steps, self.config.logging_steps
        )
        train_logger.start(self.global_step)

        self.model.train()
        train_iterator = iter(self.train_dataloader)

        logging_start_time = time.time()
        logging_start_step = self.global_step
        tr_loss = torch.tensor(0.0).to(self.accelerator.device)
        logging_loss_scalar = 0.0
        logging_text_tokens = 0
        logging_audio_tokens = 0

        while self.global_step < self.config.steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                logger.info(f"Epoch {self.epoch} starting. Resetting dataloader...")
                if hasattr(self.train_dataloader.dataset, "set_epoch"):
                    self.train_dataloader.dataset.set_epoch(self.epoch)

                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)

            batch = _to_device(batch, self.accelerator.device)

            non_padding = (batch["document_ids"] != -1).squeeze(0)  # [L]
            audio_mask = batch["audio_mask"].squeeze(0).bool()  # [L]
            logging_audio_tokens += (non_padding & audio_mask).sum().item()
            logging_text_tokens += (non_padding & ~audio_mask).sum().item()

            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch)
                loss = outputs.loss
                tr_loss += loss.detach()
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # Clipping
                    grad_norm = 0.0
                    if self.config.max_grad_norm > 0:
                        grad_norm = self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )
                        grad_norm = (
                            grad_norm.item() if grad_norm is not None else 0.0
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    train_logger.update(
                        step=self.global_step, loss=loss.item(), lr=current_lr
                    )

                    if self.global_step % self.config.logging_steps == 0:
                        elapsed = time.time() - logging_start_time
                        steps_per_sec = (
                            (self.global_step - logging_start_step) / elapsed
                            if elapsed > 0
                            else 0
                        )

                        tr_loss_scalar = self.accelerator.gather(tr_loss).mean().item()
                        current_interval_loss = tr_loss_scalar - logging_loss_scalar
                        avg_loss = current_interval_loss / (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        logging_loss_scalar = tr_loss_scalar

                        logs = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/grad_norm": grad_norm,
                            "train/epoch": self.epoch,
                            "train/steps_per_sec": steps_per_sec,
                            "train/text_tokens": logging_text_tokens,
                            "train/audio_tokens": logging_audio_tokens,
                        }
                        train_logger.log_metrics(step=self.global_step, metrics=logs)

                        logging_start_time = time.time()
                        logging_start_step = self.global_step

                    # Evaluate
                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.config.eval_steps == 0
                    ):
                        self.evaluate()

                    # Save
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(self.global_step)

        # Final Save
        self.save_checkpoint(self.global_step)
        train_logger.close()
        self.accelerator.end_training()

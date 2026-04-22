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

"""Wrapper that adapts xcodec2 to OmniVoice's audio-tokenizer interface.

xcodec2 emits a single flat index per frame in ``[0, L_1*...*L_d)`` via FSQ with
levels ``[4, 4, 4, 4, 4, 4, 4, 4]`` (vocab = 65536, 50 Hz, 16 kHz). OmniVoice's
model operates on shape ``(B, C, T)`` per-codebook indices with a small vocab
per codebook. This wrapper unravels/reravels flat indices so xcodec2 exposes
``d`` FSQ dimensions as ``d`` virtual codebooks, each with vocab ``L_i``.

License notice: xcodec2 weights are released under CC-BY-NC-4.0 (non-commercial).
"""

from types import SimpleNamespace
from typing import List, Optional, Union

import torch
import torch.nn as nn

# xcodec2 is an optional dependency — import is deferred so that the main
# package can be installed without it.
_XCODEC2_IMPORT_ERROR: Optional[Exception] = None
try:
    from xcodec2.modeling_xcodec2 import XCodec2Model  # type: ignore
except Exception as e:  # pragma: no cover
    XCodec2Model = None
    _XCODEC2_IMPORT_ERROR = e


# FSQ levels hardcoded in xcodec2's CodecDecoderVocos (vq/codec_decoder_vocos.py:391):
#   levels = [4, 4, 4, 4, 4, 4, 4, 4]  → codebook_size = 4^8 = 65536
XCODEC2_FSQ_LEVELS: List[int] = [4, 4, 4, 4, 4, 4, 4, 4]
XCODEC2_FRAME_RATE: int = 50
XCODEC2_HOP_LENGTH: int = 320
XCODEC2_SAMPLING_RATE: int = 16000


def _require_xcodec2():
    if XCodec2Model is None:
        raise ImportError(
            "The `xcodec2` package is required to use XCodec2TokenizerModel. "
            "Install it with `pip install xcodec2==0.1.5`."
        ) from _XCODEC2_IMPORT_ERROR


class XCodec2TokenizerModel(nn.Module):
    """HiggsAudioV2TokenizerModel-compatible wrapper around xcodec2.

    The wrapper maps xcodec2's single flat index per frame to ``len(levels)``
    per-FSQ-dimension indices so OmniVoice's multi-codebook model can consume
    them directly.
    """

    def __init__(
        self,
        inner_model: nn.Module,
        levels: List[int] = XCODEC2_FSQ_LEVELS,
        frame_rate: int = XCODEC2_FRAME_RATE,
        hop_length: int = XCODEC2_HOP_LENGTH,
        sampling_rate: int = XCODEC2_SAMPLING_RATE,
    ):
        super().__init__()
        self.inner = inner_model
        self.levels = list(levels)

        # FSQ flat <-> per-dim conversion follows vector_quantize_pytorch's
        # convention (little-endian base-L):
        #     flat = sum_i (d_i * basis[i]),  basis = [1, L_0, L_0*L_1, ...]
        basis = [1]
        for L in self.levels[:-1]:
            basis.append(basis[-1] * L)
        self.register_buffer(
            "_basis", torch.tensor(basis, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_levels_t",
            torch.tensor(self.levels, dtype=torch.long),
            persistent=False,
        )

        self.config = SimpleNamespace(
            frame_rate=frame_rate,
            hop_length=hop_length,
            sampling_rate=sampling_rate,
            levels=list(levels),
            num_codebooks=len(levels),
            codebook_size=int(torch.prod(torch.tensor(levels)).item()),
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device_map: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "XCodec2TokenizerModel":
        _require_xcodec2()
        inner = XCodec2Model.from_pretrained(pretrained_model_name_or_path, **kwargs)
        inner.eval()
        wrapper = cls(inner_model=inner)
        if device_map is not None:
            wrapper = wrapper.to(device_map)
        return wrapper

    @property
    def device(self) -> torch.device:
        return next(self.inner.parameters()).device

    def _flat_to_per_dim(self, flat: torch.Tensor) -> torch.Tensor:
        """(B, 1, T) or (B, T) flat indices → (B, d, T) per-dim indices."""
        if flat.dim() == 3 and flat.size(1) == 1:
            flat = flat.squeeze(1)
        elif flat.dim() != 2:
            raise ValueError(
                f"Expected flat indices of shape (B, T) or (B, 1, T), got {flat.shape}"
            )
        basis = self._basis.to(flat.device)
        levels = self._levels_t.to(flat.device)
        # (B, 1, T) // (1, d, 1) % (1, d, 1) → (B, d, T)
        per_dim = (flat.unsqueeze(1) // basis.view(1, -1, 1)) % levels.view(1, -1, 1)
        return per_dim

    def _per_dim_to_flat(self, per_dim: torch.Tensor) -> torch.Tensor:
        """(B, d, T) per-dim indices → (B, 1, T) flat indices."""
        if per_dim.dim() != 3:
            raise ValueError(
                f"Expected per-dim indices of shape (B, d, T), got {per_dim.shape}"
            )
        if per_dim.size(1) != len(self.levels):
            raise ValueError(
                f"Expected {len(self.levels)} codebooks, got {per_dim.size(1)}"
            )
        basis = self._basis.to(per_dim.device)
        flat = (per_dim.long() * basis.view(1, -1, 1)).sum(dim=1, keepdim=True)
        return flat

    @torch.inference_mode()
    def encode(self, input_values: torch.Tensor, **_unused) -> SimpleNamespace:
        """Encode 16 kHz waveform to per-dim FSQ indices.

        Args:
            input_values: ``(B, T)`` or ``(B, 1, T)`` raw waveform at 16 kHz.

        Returns:
            SimpleNamespace with ``audio_codes`` of shape ``(B, d, T_frames)``.
        """
        if input_values.dim() == 3:
            input_values = input_values.squeeze(1)
        elif input_values.dim() == 1:
            input_values = input_values.unsqueeze(0)

        # xcodec2 expects raw 16 kHz float waveform in (B, T); it pads + extracts
        # W2V-BERT features internally.
        flat_codes = self.inner.encode_code(input_waveform=input_values)
        # flat_codes: (B, 1, T_frames) with int values in [0, prod(levels))
        per_dim = self._flat_to_per_dim(flat_codes)
        return SimpleNamespace(audio_codes=per_dim)

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor, **_unused) -> SimpleNamespace:
        """Decode per-dim FSQ indices to waveform.

        Args:
            codes: ``(B, d, T_frames)`` per-FSQ-dim indices.

        Returns:
            SimpleNamespace with ``audio_values`` of shape ``(B, 1, T_samples)``.
        """
        flat_codes = self._per_dim_to_flat(codes)
        wav = self.inner.decode_code(flat_codes)  # (B, 1, T_samples)
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        return SimpleNamespace(audio_values=wav)

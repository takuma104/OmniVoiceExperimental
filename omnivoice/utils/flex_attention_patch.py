"""Workaround for flex_attention OOM on SM 12.x (RTX 5090 / Blackwell consumer).

PyTorch 2.8 treats SM >= 9.0 as H100 (228KB shared memory) for autotune config
selection. SM 12.0 (GB202) is Blackwell consumer with only 100KB, so H100 configs
like FlexConfig(128, 64, 3, 8) require ~156KB and fail at Triton compile time.

This patch overrides the heuristic to use smaller block sizes for SM 12.x.
Remove once PyTorch adds native SM 12.x flex_attention configs.
"""

import logging

import torch

logger = logging.getLogger(__name__)


def patch_flex_attention_for_sm12() -> None:
    if not torch.cuda.is_available():
        return

    capability = torch.cuda.get_device_capability()
    if capability[0] < 12:
        return

    from torch._inductor import config as inductor_config
    from torch._inductor.template_heuristics import CUDAConfigHeuristic, FlexConfig

    # Conservative configs sized to fit within 100KB shared memory.
    # FlexConfig(block_m, block_n, num_stages, num_warps)
    # For bfloat16, head_dim=128: 64×128×2 + 64×128×2×2 + 64×64×4 = 81KB
    _sm12_fwd_configs = {
        (torch.float32, 64): FlexConfig(64, 32, 1, 4),
        (torch.float32, 128): FlexConfig(64, 32, 1, 4),
        (torch.float32, 256): FlexConfig(32, 16, 1, 4),
        (torch.bfloat16, 64): FlexConfig(64, 64, 1, 4),
        (torch.bfloat16, 128): FlexConfig(64, 64, 1, 4),
        (torch.bfloat16, 256): FlexConfig(32, 32, 1, 4),
        (torch.float16, 64): FlexConfig(64, 64, 1, 4),
        (torch.float16, 128): FlexConfig(64, 64, 1, 4),
        (torch.float16, 256): FlexConfig(32, 32, 1, 4),
    }

    def _patched_fwd(self, head_dim: int, dtype) -> list:
        configs = []
        if inductor_config.max_autotune:
            if inductor_config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_fwd_configs
            configs += self.flex_attn_fwd_autotune_configs
        default = _sm12_fwd_configs.get((dtype, head_dim), FlexConfig(64, 32, 1, 4))
        if default not in configs:
            configs.append(default)
        return configs

    def _patched_bwd(self, head_dim: int, dtype) -> list:
        configs = []
        if inductor_config.max_autotune:
            if inductor_config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_attn_bwd_configs
            configs += self.flex_attn_bwd_autotune_configs
        default = FlexConfig(16, 16, 1, 4)
        if default not in configs:
            configs.append(default)
        return configs

    CUDAConfigHeuristic.get_flex_attn_fwd_configs = _patched_fwd
    CUDAConfigHeuristic.get_flex_attn_bwd_configs = _patched_bwd

    device_name = torch.cuda.get_device_name(0)
    logger.info(
        f"Applied flex_attention SM 12.x patch for {device_name} "
        f"(SM {capability[0]}.{capability[1]}, 100KB shared mem limit)."
    )

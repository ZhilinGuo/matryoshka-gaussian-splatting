import math
from typing import Any, Dict, List, Optional

import torch


def compute_mrl_nesting_sizes(cap_max: int, min_splats: int = 100_000) -> List[int]:
    """MRL-style consistent halving: cap_max, cap_max//2, cap_max//4, ... until >= min_splats.

    Returns sorted ascending (smallest first). Only sizes >= min_splats are included.
    """
    if cap_max <= 0:
        return []
    sizes: List[int] = []
    m = cap_max
    while m >= min_splats:
        sizes.append(m)
        m = m // 2
    sizes = sorted(set(sizes))
    return sizes


def compute_mrl_nesting_sizes_paper(
    cap_max: int, base_max: int = 2048
) -> List[int]:
    """MRL paper representation lengths 2^3..2^11 (8..2048) scaled so base_max -> cap_max.

    So nesting sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048] * (cap_max / base_max),
    rounded to int, clamped to [1, cap_max]. Returns sorted ascending.
    """
    if cap_max <= 0 or base_max <= 0:
        return []
    scale = cap_max / base_max
    sizes = [
        max(1, min(cap_max, round((2 ** k) * scale)))
        for k in range(3, 12)
    ]
    return sorted(set(sizes))


class MRLSubsetScheduler:
    """MRL-exact: fixed nesting sizes (consistent halving from cap_max), all optimized each step.

    No stochastic sampling; each step returns one subset per nesting size (prefix of
    the sorted splat order). All losses weighted equally (1.0).

    If nesting_base_max is set (e.g. 2048), uses paper scaling: sizes 2^3..2^11
    scaled so nesting_base_max -> cap_max. Otherwise uses consistent halving from
    cap_max down to min_splats.
    """

    def __init__(
        self,
        cap_max: int,
        min_splats: int = 100_000,
        nesting_base_max: Optional[int] = None,
    ) -> None:
        assert cap_max >= 1
        self.cap_max = cap_max
        self.min_splats = min_splats
        if nesting_base_max is not None and nesting_base_max >= 1:
            self.nesting_splat_counts = compute_mrl_nesting_sizes_paper(
                cap_max, nesting_base_max
            )
        else:
            assert min_splats >= 1
            self.nesting_splat_counts = compute_mrl_nesting_sizes(
                cap_max, min_splats
            )

    def sample_subsets(
        self,
        num_splats: int,
        sort_indices: torch.Tensor,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        """Return one subset per MRL nesting size (prefix of sort_indices), clamped to num_splats."""
        if num_splats <= 0:
            return []
        entries: List[Dict[str, Any]] = []
        seen: Dict[int, bool] = {}
        for n in self.nesting_splat_counts:
            n_keep = max(1, min(n, num_splats))
            if seen.get(n_keep):
                continue
            seen[n_keep] = True
            subset_idx = sort_indices[:n_keep]
            is_full = n_keep == num_splats
            entries.append(
                {
                    "indices": subset_idx,
                    "keep_ratio": n_keep / num_splats,
                    "timestep": 0,
                    "is_full": is_full,
                    "weight": 1.0,
                }
            )
        # Sort by n_keep ascending so order is consistent
        entries.sort(key=lambda e: e["indices"].shape[0])
        return entries


class DiffusionSubsetScheduler:
    """Diffusion-style stochastic Matryoshka subset scheduler.

    At each training step, this scheduler samples multiple timesteps and converts
    them into keep-ratios. The splats are assumed to be pre-sorted (coarse-to-fine),
    and each subset is a prefix of that ordering up to a stochastic keep-ratio.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_subsets: int = 4,
        schedule: str = "cosine",
        min_keep_ratio: float = 0.0,
        max_keep_ratio: float = 0.9,
        include_full_subset: bool = True,
    ) -> None:
        assert num_timesteps > 0
        assert num_subsets >= 1
        assert 0.0 <= min_keep_ratio <= max_keep_ratio <= 1.0
        self.num_timesteps = num_timesteps
        self.num_subsets = num_subsets
        self.schedule = schedule
        self.min_keep_ratio = float(min_keep_ratio)
        self.max_keep_ratio = float(max_keep_ratio)
        self.include_full_subset = include_full_subset

        # Precompute a diffusion-like alpha_cumprod schedule on CPU.
        self.alphas_cumprod = self._build_alphas_cumprod(schedule)

    def _build_alphas_cumprod(self, schedule: str) -> torch.Tensor:
        if schedule == "uniform":
            # Linearly decreasing alpha_cumprod from 1 to 0.
            # When timesteps are sampled uniformly, this yields a uniform
            # distribution over keep ratios — the simplest baseline schedule.
            return torch.linspace(1.0, 0.0, self.num_timesteps)

        if schedule == "linear":
            # DDPM-style linear beta schedule.
            betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        elif schedule == "cosine":
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models".
            steps = self.num_timesteps + 1
            t = torch.linspace(0, self.num_timesteps, steps)
            s = 0.008
            alphas_cumprod = torch.cos(
                ((t / self.num_timesteps) + s) / (1 + s) * math.pi / 2
            ) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 1e-8, 0.999)
        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule}")

        alphas = 1.0 - betas
        return torch.cumprod(alphas, dim=0)  # [T]

    def sample_subsets(
        self,
        num_splats: int,
        sort_indices: torch.Tensor,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        """Sample a list of Matryoshka subsets for the current training step.

        Each subset is defined as a dict containing:
          - indices: Tensor of splat indices (into the global splat arrays).
          - keep_ratio: Fraction of splats kept in this subset.
          - timestep: Chosen diffusion timestep.
          - is_full: Whether this subset contains all splats.
          - weight: Scalar weight for this subset in the total loss.
        """
        if num_splats <= 0:
            return []

        entries: List[Dict[str, Any]] = []

        # Number of noisy subsets; optionally add a full subset later.
        n_random = (
            self.num_subsets - 1 if self.include_full_subset else self.num_subsets
        )
        if n_random > 0:
            # Avoid sampling timestep==0 if we explicitly add a clean full subset later.
            low_t = 1 if (self.include_full_subset and self.num_timesteps > 1) else 0
            timesteps = torch.randint(
                low=low_t,
                high=self.num_timesteps,
                size=(n_random,),
                device=device,
            )
        else:
            timesteps = torch.empty(0, dtype=torch.long, device=device)

        keep_ratios: List[float] = []
        for t in timesteps.tolist():
            alpha_bar = float(self.alphas_cumprod[t].item())
            # Smaller alpha_bar -> smaller keep_ratio -> coarser subset.
            keep = self.min_keep_ratio + alpha_bar * (
                self.max_keep_ratio - self.min_keep_ratio
            )
            keep = float(max(self.min_keep_ratio, min(self.max_keep_ratio, keep)))
            keep_ratios.append(keep)

        if self.include_full_subset:
            # Always add a clean full subset; use timestep 0 for logging.
            keep_ratios.append(1.0)
            timesteps = torch.cat(
                [timesteps, torch.zeros(1, dtype=torch.long, device=device)], dim=0
            )

        # Sort subsets by keep-ratio so they are nested (smallest -> largest).
        keep_tensor = torch.tensor(keep_ratios, device=device)
        order = torch.argsort(keep_tensor)
        keep_sorted = keep_tensor[order]
        timesteps_sorted = timesteps[order]

        for i in range(keep_sorted.shape[0]):
            r = float(keep_sorted[i].item())
            t = int(timesteps_sorted[i].item())
            n_keep = max(1, min(num_splats, int(round(r * num_splats))))
            subset_idx = sort_indices[:n_keep]
            is_full = n_keep == num_splats
            entries.append(
                {
                    "indices": subset_idx,
                    "keep_ratio": r,
                    "timestep": t,
                    "is_full": is_full,
                    "weight": 1.0,
                }
            )

        return entries

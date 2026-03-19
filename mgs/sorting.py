import torch
from typing import Optional

FIXED_ORDER_POLICIES = {
    "fixed_append": "append",
    "fixed_prepend": "prepend",
    "fixed_random": "random",
}


def resolve_fixed_order_policy(strategy: str) -> Optional[str]:
    return FIXED_ORDER_POLICIES.get(strategy)


class SplatSorter:
    def __init__(self, strategy: str = "by_volume_descending"):
        self.strategy = strategy

    def _parse_strategy(self) -> tuple[str, bool]:
        strategy = self.strategy
        if strategy.endswith("_ascending"):
            base = strategy[: -len("_ascending")]
            return base, False
        if strategy.endswith("_descending"):
            base = strategy[: -len("_descending")]
            return base, True
        if strategy in {"size", "volume", "by_volume"}:
            return "by_volume", True
        if strategy in {"by_opacity", "by_sh_energy", "by_color_variance"}:
            return strategy, True
        return strategy, True

    def _fixed_order_policy(self) -> Optional[str]:
        return resolve_fixed_order_policy(self.strategy)

    def _sh0_to_rgb(self, sh0: torch.Tensor) -> torch.Tensor:
        # sh0 is DC SH coefficient; approximate RGB in [0,1].
        C0 = 0.28209479177387814
        return sh0 * C0 + 0.5

    def _get_rgb(self, splats: torch.nn.ParameterDict) -> torch.Tensor:
        if "colors" in splats:
            return torch.sigmoid(splats["colors"])
        if "sh0" in splats:
            sh0 = splats["sh0"].squeeze(1)
            return self._sh0_to_rgb(sh0)
        raise ValueError("Color variance sorting requires 'colors' or 'sh0'.")

    def argsort(self, splats: torch.nn.ParameterDict) -> torch.Tensor:
        """
        Returns indices that sort the splats according to the strategy.
        Default 'by_volume_descending' strategy sorts by volume (product of scales)
        descending.
        """
        fixed_policy = self._fixed_order_policy()
        if fixed_policy is not None:
            if "order_key" not in splats:
                raise ValueError(
                    "Fixed-order sorting requires 'order_key' in splats."
                )
            order_key = splats["order_key"]
            return torch.argsort(order_key, descending=False)
        strategy, descending = self._parse_strategy()
        if strategy in {"size", "by_volume", "volume"}:
            # splats["scales"] are log-scales
            scales = torch.exp(splats["scales"])
            # Calculate volume: product of x,y,z scales
            volume = scales.prod(dim=1)
            # Sort descending: Largest (Coarse) -> Smallest (Fine)
            return torch.argsort(volume, descending=descending)
        if strategy == "by_opacity":
            opacities = torch.sigmoid(splats["opacities"])
            return torch.argsort(opacities, descending=descending)
        if strategy == "by_sh_energy":
            if "shN" in splats and "sh0" in splats:
                sh0 = splats["sh0"].squeeze(1)
                shN = splats["shN"]
                high_energy = (shN ** 2).sum(dim=(1, 2))
                dc_energy = (sh0 ** 2).sum(dim=1)
                score = high_energy / (dc_energy + 1e-6)
            elif "shN" in splats:
                score = (splats["shN"] ** 2).sum(dim=(1, 2))
            elif "colors" in splats:
                colors = torch.sigmoid(splats["colors"])
                score = (colors ** 2).sum(dim=1)
            else:
                raise ValueError("SH energy sorting requires SH coeffs or colors.")
            return torch.argsort(score, descending=descending)
        if strategy == "by_color_variance":
            rgb = self._get_rgb(splats)
            mean_rgb = rgb.mean(dim=1, keepdim=True)
            variance = ((rgb - mean_rgb) ** 2).mean(dim=1)
            return torch.argsort(variance, descending=descending)
        if strategy == "random":
            return torch.randperm(
                splats["means"].shape[0], device=splats["means"].device
            )
        raise NotImplementedError(f"Unknown sort strategy: {self.strategy}")

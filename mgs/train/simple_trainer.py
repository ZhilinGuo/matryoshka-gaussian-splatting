import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from datasets.auto import build_parser_and_datasets
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import reset_opa
from mgs import (
    DiffusionSubsetScheduler,
    MRLSubsetScheduler,
    SplatSorter,
    resolve_fixed_order_policy,
)
from mgs.utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed


@dataclass
class Config:
    # Path to the .pt files. If provided and resume is false, run evaluation only.
    ckpt: Optional[List[str]] = None
    # Resume training from a checkpoint instead of eval-only.
    resume: bool = False
    # Render trajectory path (kept for config compatibility)
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # When True, skip T3 (z-flip) in COLMAP parser so world matches CLOD-3DGS (T1+T2 only)
    normalize_skip_t3: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [30_000])
    # Save prediction-only images (no stitched GT+pred canvases)
    save_pred_images: bool = False
    # Splat sorting strategy for Matryoshka prefixes
    sort_strategy: Literal[
        "by_volume_descending",
        "by_volume_ascending",
        "by_opacity_descending",
        "by_opacity_ascending",
        "by_sh_energy_descending",
        "by_sh_energy_ascending",
        "by_color_variance_descending",
        "by_color_variance_ascending",
        "by_volume",
        "by_opacity",
        "by_sh_energy",
        "by_color_variance",
        "random",
        "fixed_append",
        "fixed_prepend",
        "fixed_random",
        "size",
    ] = "by_volume_descending"

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Weight for LPIPS perceptual loss
    lpips_lambda: float = 0.0

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # Diffusion-style stochastic Matryoshka parameters (training-time)
    diffusion_num_timesteps: int = 1000
    # Number of Matryoshka subsets sampled per training step (including the full set)
    diffusion_num_subsets: int = 4
    # Noise schedule used to generate keep ratios
    diffusion_schedule: Literal["linear", "cosine", "uniform"] = "uniform"
    # Minimum fraction of splats kept in any noisy subset
    diffusion_min_keep_ratio: float = 0.0
    # Maximum fraction of splats kept in noisy subsets.
    diffusion_max_keep_ratio: float = 1.0
    # Whether to always include a full (all-splats) subset each step
    diffusion_include_full_subset: bool = True
    # Loss weight for non-full (prefix) subsets when using diffusion sampling.
    diffusion_prefix_weight: float = 1.0
    # Loss weight for the explicit full subset when included.
    diffusion_full_weight: float = 1.0
    # Optional preset for single-prefix objectives.
    diffusion_objective: Optional[
        Literal[
            "single_prefix",
            "single_prefix_full",
            "multi_prefix_only",
            "mrl_fixed",
        ]
    ] = None
    # For mrl_fixed: minimum splat count when building M (consistent halving from cap_max).
    diffusion_mrl_nesting_min_splats: int = 100_000
    # For mrl_fixed: if set (e.g. 2048), use paper MRL sizes 2^3..2^11 scaled so this -> cap_max.
    diffusion_mrl_nesting_base_max: Optional[int] = None

    # Matryoshka-style *inference* parameters (for side-by-side comparisons)
    mgs_splits: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0]
    )
    # Optional: also evaluate at MRL nesting sizes (absolute splat counts) for ablation.
    mgs_splits_mrl: Optional[List[int]] = None
    mgs_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0]
    )
    # Run subset-wise inference for Matryoshka splits during eval.
    run_mgs_inference: bool = False
    # Save rendered images for each subset when running inference.
    mgs_inference_save_images: bool = True

    # Disable viewer (always True in this release; kept for CLI compatibility)
    disable_viewer: bool = True

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

    def apply_diffusion_objective(self) -> None:
        if self.diffusion_objective is None:
            return
        if self.diffusion_objective == "single_prefix":
            self.diffusion_num_subsets = 1
            self.diffusion_include_full_subset = False
        elif self.diffusion_objective == "single_prefix_full":
            self.diffusion_num_subsets = 2
            self.diffusion_include_full_subset = True
        elif self.diffusion_objective == "multi_prefix_only":
            self.diffusion_include_full_subset = False
        elif self.diffusion_objective == "mrl_fixed":
            pass
        else:
            raise ValueError(
                f"Unknown diffusion objective: {self.diffusion_objective}"
            )


def _select_resume_ckpt(
    ckpt_paths: List[str], world_rank: int, world_size: int
) -> str:
    if len(ckpt_paths) == 1:
        return ckpt_paths[0]
    if len(ckpt_paths) == world_size:
        return ckpt_paths[world_rank]
    raise ValueError(
        "Resume expects 1 checkpoint (single GPU) or one per rank."
    )


def _load_checkpoint(path: str, device: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        print(
            f"[WARN] weights_only load failed for {path}; retrying with "
            f"weights_only=False. Error: {exc}"
        )
        return torch.load(path, map_location=device, weights_only=False)


def create_splats_with_optimizers(
    parser: Any,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    order_key_policy: Optional[str] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    if order_key_policy is not None:
        if order_key_policy == "random":
            order_key = torch.rand((N,), device=device, dtype=torch.float32)
        else:
            order_key = torch.arange(N, device=device, dtype=torch.float32)
        splats["order_key"] = torch.nn.Parameter(order_key, requires_grad=False)

    BS = batch_size * world_size
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing with stochastic Matryoshka subsets."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.cfg.apply_diffusion_objective()
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.start_step = 0
        self._resume_scheduler_states: Optional[List[Dict[str, Any]]] = None

        os.makedirs(cfg.result_dir, exist_ok=True)

        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        self.parser, self.trainset, self.valset = build_parser_and_datasets(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            skip_t3=cfg.normalize_skip_t3,
        )
        if getattr(self.parser, "dataset_type", "") == "blender" and cfg.init_type == "sfm":
            raise ValueError(
                "Blender/NeRF-Synthetic datasets do not provide COLMAP point clouds; "
                "use --init_type random (or run COLMAP first)."
            )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.fixed_order_policy = resolve_fixed_order_policy(cfg.sort_strategy)
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            order_key_policy=self.fixed_order_policy,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.sorter = SplatSorter(strategy=cfg.sort_strategy)

        if cfg.diffusion_objective == "mrl_fixed":
            cap_max = getattr(cfg.strategy, "cap_max", None)
            if cap_max is None or cap_max < 1:
                raise ValueError(
                    "mrl_fixed objective requires strategy.cap_max "
                    "(e.g. MCMCStrategy with --strategy.cap-max 5000000)."
                )
            self.subset_scheduler = MRLSubsetScheduler(
                cap_max=int(cap_max),
                min_splats=cfg.diffusion_mrl_nesting_min_splats,
                nesting_base_max=cfg.diffusion_mrl_nesting_base_max,
            )
        else:
            self.subset_scheduler = DiffusionSubsetScheduler(
                num_timesteps=cfg.diffusion_num_timesteps,
                num_subsets=cfg.diffusion_num_subsets,
                schedule=cfg.diffusion_schedule,
                min_keep_ratio=cfg.diffusion_min_keep_ratio,
                max_keep_ratio=cfg.diffusion_max_keep_ratio,
                include_full_subset=cfg.diffusion_include_full_subset,
            )

        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)
        if self.fixed_order_policy is not None:
            self.strategy_state["order_policy"] = self.fixed_order_policy

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

    def load_checkpoint(self, ckpt_paths: List[str]) -> int:
        ckpt_path = _select_resume_ckpt(
            ckpt_paths, world_rank=self.world_rank, world_size=self.world_size
        )
        ckpt = _load_checkpoint(ckpt_path, self.device)
        if "splats" not in ckpt:
            raise ValueError("Checkpoint missing 'splats' for resume.")
        for k in self.splats.keys():
            if k not in ckpt["splats"]:
                raise ValueError(f"Checkpoint splats missing '{k}'.")
            self.splats[k].data = ckpt["splats"][k].to(self.device)

        if "pose_adjust" in ckpt:
            if not self.cfg.pose_opt:
                raise ValueError(
                    "Checkpoint has pose_adjust but pose_opt is disabled."
                )
            if self.world_size > 1:
                self.pose_adjust.module.load_state_dict(ckpt["pose_adjust"])
            else:
                self.pose_adjust.load_state_dict(ckpt["pose_adjust"])
        if "app_module" in ckpt:
            if not self.cfg.app_opt:
                raise ValueError("Checkpoint has app_module but app_opt is disabled.")
            if self.world_size > 1:
                self.app_module.module.load_state_dict(ckpt["app_module"])
            else:
                self.app_module.load_state_dict(ckpt["app_module"])

        if "strategy_state" in ckpt:
            self.strategy_state = ckpt["strategy_state"]
        if self.fixed_order_policy is not None:
            self.strategy_state["order_policy"] = self.fixed_order_policy

        if "optimizers" in ckpt:
            for name, state in ckpt["optimizers"].items():
                if name in self.optimizers:
                    self.optimizers[name].load_state_dict(state)
        if "pose_optimizers" in ckpt and self.pose_optimizers:
            for opt, state in zip(self.pose_optimizers, ckpt["pose_optimizers"]):
                opt.load_state_dict(state)
        if "app_optimizers" in ckpt and self.app_optimizers:
            for opt, state in zip(self.app_optimizers, ckpt["app_optimizers"]):
                opt.load_state_dict(state)

        self._resume_scheduler_states = ckpt.get("schedulers")
        step = int(ckpt.get("step", 0))
        self.start_step = max(step + 1, 0)
        return step

    def _build_subset_overrides(self, subset_indices: torch.Tensor) -> Dict[str, Tensor]:
        """Slice the current splats into a subset used for inference."""
        subset_indices = subset_indices.to(self.device).long()
        overrides: Dict[str, Tensor] = {
            "means": torch.index_select(self.splats["means"], 0, subset_indices),
            "quats": torch.index_select(self.splats["quats"], 0, subset_indices),
            "scales": torch.index_select(self.splats["scales"], 0, subset_indices),
            "opacities": torch.index_select(
                self.splats["opacities"], 0, subset_indices
            ),
        }
        if self.cfg.app_opt:
            overrides["features"] = torch.index_select(
                self.splats["features"], 0, subset_indices
            )
            overrides["colors"] = torch.index_select(
                self.splats["colors"], 0, subset_indices
            )
        else:
            overrides["sh0"] = torch.index_select(
                self.splats["sh0"], 0, subset_indices
            )
            overrides["shN"] = torch.index_select(
                self.splats["shN"], 0, subset_indices
            )
        return overrides

    @staticmethod
    def _format_split_label(split: float) -> str:
        """Convert a split ratio into a filesystem-friendly string."""
        split_decimal = Decimal(str(split)).normalize()
        split_str = format(split_decimal, "f").rstrip("0").rstrip(".")
        if split_str == "":
            split_str = "0"
        split_str = split_str.replace("-", "m").replace(".", "p")
        return split_str

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        splat_overrides: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        if splat_overrides is None:
            splat_overrides = {}

        means = splat_overrides.get("means", self.splats["means"])
        quats = splat_overrides.get("quats", self.splats["quats"])

        if "scales" in splat_overrides:
            scales = torch.exp(splat_overrides["scales"])
        else:
            scales = torch.exp(self.splats["scales"])

        if "opacities" in splat_overrides:
            opacities = torch.sigmoid(splat_overrides["opacities"])
        else:
            opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            features = splat_overrides.get("features", self.splats["features"])
            base_colors = splat_overrides.get("colors", self.splats["colors"])

            colors = self.app_module(
                features=features,
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + base_colors
            colors = torch.sigmoid(colors)
        else:
            if "sh0" in splat_overrides and "shN" in splat_overrides:
                colors = torch.cat([splat_overrides["sh0"], splat_overrides["shN"]], 1)
            else:
                colors = torch.cat(
                    [self.splats["sh0"], self.splats["shN"]], 1
                )  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = self.start_step

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        if self._resume_scheduler_states is not None:
            if len(self._resume_scheduler_states) != len(schedulers):
                print(
                    "[WARN] Scheduler state count mismatch; skipping restore."
                )
            else:
                for scheduler, state in zip(
                    schedulers, self._resume_scheduler_states
                ):
                    scheduler.load_state_dict(state)
        elif init_step > 0:
            for scheduler in schedulers:
                scheduler.last_epoch = init_step - 1
                scheduler.step()

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Matryoshka: sort and sample stochastic subsets
            sort_indices = self.sorter.argsort(self.splats)
            total_loss = 0.0

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)

            full_loss: Optional[Tensor] = None
            full_l1loss: Optional[Tensor] = None
            full_ssimloss: Optional[Tensor] = None
            full_lpipsloss: Optional[Tensor] = None
            full_depthloss: Optional[Tensor] = None

            subsets_info: List[Tuple[Dict[str, Tensor], torch.Tensor]] = []
            info: Dict[str, Tensor] = {}

            subsets = self.subset_scheduler.sample_subsets(
                num_splats=len(sort_indices),
                sort_indices=sort_indices,
                device=device,
            )
            for subset in subsets:
                if cfg.diffusion_include_full_subset and subset.get("timestep") == 0:
                    subset["weight"] = float(cfg.diffusion_full_weight)
                else:
                    subset["weight"] = float(cfg.diffusion_prefix_weight)

            for subset in subsets:
                subset_indices: torch.Tensor = subset["indices"]
                is_full = bool(subset["is_full"])
                weight = float(subset["weight"])

                subset_overrides: Dict[str, Tensor] = {
                    "means": self.splats["means"][subset_indices],
                    "quats": self.splats["quats"][subset_indices],
                    "scales": self.splats["scales"][subset_indices],
                    "opacities": self.splats["opacities"][subset_indices],
                }
                if cfg.app_opt:
                    subset_overrides["features"] = self.splats["features"][subset_indices]
                    subset_overrides["colors"] = self.splats["colors"][subset_indices]
                else:
                    subset_overrides["sh0"] = self.splats["sh0"][subset_indices]
                    subset_overrides["shN"] = self.splats["shN"][subset_indices]

                renders, alphas, info_k = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    masks=masks,
                    splat_overrides=subset_overrides,
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                if cfg.random_bkgd:
                    colors = colors + bkgd * (1.0 - alphas)

                target_pixels = torch.clamp(pixels, 0.0, 1.0)

                l1loss_k = F.l1_loss(colors, target_pixels)
                ssimloss_k = 1.0 - fused_ssim(
                    colors.permute(0, 3, 1, 2),
                    target_pixels.permute(0, 3, 1, 2),
                    padding="valid",
                )
                loss_k = l1loss_k * (1.0 - cfg.ssim_lambda) + ssimloss_k * cfg.ssim_lambda
                lpipsloss_k: Optional[Tensor] = None
                if cfg.lpips_lambda > 0.0:
                    colors_lpips = torch.clamp(colors, 0.0, 1.0)
                    lpipsloss_k = self.lpips(
                        colors_lpips.permute(0, 3, 1, 2),
                        target_pixels.permute(0, 3, 1, 2),
                    )
                    loss_k = loss_k + lpipsloss_k * cfg.lpips_lambda

                depthloss_k: Optional[Tensor] = None
                if cfg.depth_loss and depths is not None:
                    points_norm = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )
                    grid = points_norm.unsqueeze(2)  # [1, M, 1, 2]
                    depths_sampled = F.grid_sample(
                        depths.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    depths_sampled = depths_sampled.squeeze(3).squeeze(1)  # [1, M]
                    disp = torch.where(
                        depths_sampled > 0.0,
                        1.0 / depths_sampled,
                        torch.zeros_like(depths_sampled),
                    )
                    disp_gt = 1.0 / depths_gt  # [1, M]
                    depthloss_k = F.l1_loss(disp, disp_gt) * self.scene_scale
                    loss_k = loss_k + depthloss_k * cfg.depth_lambda

                total_loss = total_loss + loss_k * weight

                if isinstance(self.cfg.strategy, DefaultStrategy):
                    key = self.cfg.strategy.key_for_gradient
                    if key in info_k:
                        info_k[key].retain_grad()
                        subsets_info.append((info_k, subset_indices))

                if is_full:
                    full_loss = loss_k
                    full_l1loss = l1loss_k
                    full_ssimloss = ssimloss_k
                    full_lpipsloss = lpipsloss_k
                    full_depthloss = depthloss_k

                    info = {}
                    for k, v in info_k.items():
                        if isinstance(v, torch.Tensor) and v.shape[0] == len(sort_indices):
                            remapped = torch.empty_like(v)
                            remapped[sort_indices] = v
                            info[k] = remapped
                        else:
                            info[k] = v

                    if not isinstance(self.cfg.strategy, DefaultStrategy):
                        self.cfg.strategy.step_pre_backward(
                            params=self.splats,
                            optimizers=self.optimizers,
                            state=self.strategy_state,
                            step=step,
                            info=info,
                        )

            # regularizations
            if cfg.opacity_reg > 0.0:
                total_loss = total_loss + cfg.opacity_reg * torch.sigmoid(
                    self.splats["opacities"]
                ).mean()
            if cfg.scale_reg > 0.0:
                total_loss = total_loss + cfg.scale_reg * torch.exp(
                    self.splats["scales"]
                ).mean()

            loss = total_loss
            loss.backward()

            if full_loss is not None:
                loss = full_loss
            if full_l1loss is not None:
                l1loss = full_l1loss
            else:
                l1loss = l1loss_k
            if full_ssimloss is not None:
                ssimloss = full_ssimloss
            else:
                ssimloss = ssimloss_k

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss and full_depthloss is not None:
                desc += f"depth loss={full_depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                if cfg.lpips_lambda > 0.0 and full_lpipsloss is not None:
                    self.writer.add_scalar("train/lpipsloss", full_lpipsloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss and full_depthloss is not None:
                    self.writer.add_scalar("train/depthloss", full_depthloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {
                    "step": step,
                    "splats": self.splats.state_dict(),
                    "optimizers": {
                        name: opt.state_dict()
                        for name, opt in self.optimizers.items()
                    },
                    "schedulers": [sched.state_dict() for sched in schedulers],
                    "strategy_state": self.strategy_state,
                }
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                    data["pose_optimizers"] = [
                        opt.state_dict() for opt in self.pose_optimizers
                    ]
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                    data["app_optimizers"] = [
                        opt.state_dict() for opt in self.app_optimizers
                    ]
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Densification strategy post-backward
            if isinstance(self.cfg.strategy, DefaultStrategy):
                strategy = self.cfg.strategy
                params = self.splats
                optimizers = self.optimizers
                state = self.strategy_state
                packed = cfg.packed

                if step < strategy.refine_stop_iter:
                    for (info_k, subset_indices) in subsets_info:
                        key = strategy.key_for_gradient
                        if key not in info_k:
                            continue

                        if strategy.absgrad:
                            grads = info_k[key].absgrad.clone()
                        else:
                            grads = info_k[key].grad.clone()
                        grads[..., 0] *= info_k["width"] / 2.0 * info_k["n_cameras"]
                        grads[..., 1] *= info_k["height"] / 2.0 * info_k["n_cameras"]

                        n_gaussian = len(list(params.values())[0])
                        if state["grad2d"] is None:
                            state["grad2d"] = torch.zeros(
                                n_gaussian, device=grads.device
                            )
                        if state["count"] is None:
                            state["count"] = torch.zeros(
                                n_gaussian, device=grads.device
                            )
                        if (
                            strategy.refine_scale2d_stop_iter > 0
                            and state["radii"] is None
                        ):
                            state["radii"] = torch.zeros(
                                n_gaussian, device=grads.device
                            )

                        if packed:
                            gs_ids_subset = info_k["gaussian_ids"]
                            gs_ids_global = subset_indices[gs_ids_subset]
                            radii = info_k["radii"].max(dim=-1).values
                        else:
                            sel = (info_k["radii"] > 0.0).all(dim=-1)
                            gs_ids_subset = torch.where(sel)[1]
                            gs_ids_global = subset_indices[gs_ids_subset]
                            grads = grads[sel]
                            radii = info_k["radii"][sel].max(dim=-1).values

                        state["grad2d"].index_add_(0, gs_ids_global, grads.norm(dim=-1))
                        state["count"].index_add_(
                            0,
                            gs_ids_global,
                            torch.ones_like(gs_ids_global, dtype=torch.float32),
                        )
                        if strategy.refine_scale2d_stop_iter > 0:
                            state["radii"][gs_ids_global] = torch.maximum(
                                state["radii"][gs_ids_global],
                                radii
                                / float(
                                    max(
                                        info_k["width"],
                                        info_k["height"],
                                    )
                                ),
                            )

                    if (
                        step > strategy.refine_start_iter
                        and step % strategy.refine_every == 0
                        and step % strategy.reset_every
                        >= strategy.pause_refine_after_reset
                    ):
                        n_dupli, n_split = strategy._grow_gs(
                            params, optimizers, state, step
                        )
                        if strategy.verbose:
                            print(
                                f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                                f"Now having {len(params['means'])} GSs."
                            )

                        n_prune = strategy._prune_gs(
                            params, optimizers, state, step
                        )
                        if strategy.verbose:
                            print(
                                f"Step {step}: {n_prune} GSs pruned. "
                                f"Now having {len(params['means'])} GSs."
                            )

                        state["grad2d"].zero_()
                        state["count"].zero_()
                        if strategy.refine_scale2d_stop_iter > 0:
                            state["radii"].zero_()
                        torch.cuda.empty_cache()

                    if step % strategy.reset_every == 0 and step > 0:
                        reset_opa(
                            params=params,
                            optimizers=optimizers,
                            state=state,
                            value=strategy.prune_opa * 2.0,
                        )

            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                if cfg.run_mgs_inference:
                    self.eval_matryoshka_subsets(step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        times = []
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            elapsed = max(time.time() - tic, 1e-10)
            times.append(elapsed)

            colors = torch.clamp(colors, 0.0, 1.0)

            if world_rank == 0:
                pred_img = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}_pred.png",
                    pred_img,
                )
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if world_rank == 0:
            num_images = len(valloader)
            if len(times) > 3:
                avg_time = sum(times[3:]) / (len(times) - 3)
            elif times:
                avg_time = sum(times) / len(times)
            else:
                avg_time = 0.0
            time_per_image = avg_time
            fps = 1.0 / max(time_per_image, 1e-10)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": time_per_image,
                    "time_per_image": time_per_image,
                    "fps": fps,
                    "num_GS": len(self.splats["means"]),
                    "num_images": num_images,
                }
            )
            print(
                f"PSNR: {stats.get('psnr', 0.0):.3f}, "
                f"SSIM: {stats.get('ssim', 0.0):.4f}, "
                f"LPIPS: {stats.get('lpips', 0.0):.3f} "
                f"Time: {stats['time_per_image']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def eval_matryoshka_subsets(self, step: int, stage: str = "val"):
        """Evaluate each Matryoshka subset for quantitative metrics and FPS."""
        cfg = self.cfg
        if not cfg.mgs_splits:
            print("No Matryoshka splits configured; skipping subset inference.")
            return

        print("Running Matryoshka subset evaluation...")
        device = self.device
        world_rank = self.world_rank

        if stage == "val":
            dataset = self.valset
        elif stage == "train":
            dataset = self.trainset
        else:
            raise ValueError(f"Unsupported evaluation stage: {stage}")

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1
        )

        sort_indices = self.sorter.argsort(self.splats)
        total_splats = len(sort_indices)
        subset_entries = []
        for idx, split in enumerate(cfg.mgs_splits):
            n_keep = int(total_splats * split)
            if n_keep < 1:
                n_keep = 1
            n_keep = min(n_keep, total_splats)
            split_str = self._format_split_label(split)
            label = f"{idx:02d}_{split_str}"
            subset_entries.append(
                {
                    "split": split,
                    "label": label,
                    "num_splats": n_keep,
                    "mrl_position": False,
                    "overrides": self._build_subset_overrides(
                        sort_indices[:n_keep]
                    ),
                    "metrics": defaultdict(list),
                    "times": [],
                }
            )
        if cfg.mgs_splits_mrl:
            for n_requested in cfg.mgs_splits_mrl:
                n_keep = max(1, min(int(n_requested), total_splats))
                label = f"mrl_{n_keep}"
                subset_entries.append(
                    {
                        "split": n_keep / total_splats,
                        "label": label,
                        "num_splats": n_keep,
                        "mrl_position": True,
                        "overrides": self._build_subset_overrides(
                            sort_indices[:n_keep]
                        ),
                        "metrics": defaultdict(list),
                        "times": [],
                    }
                )

        for i, data in enumerate(dataloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            for subset in subset_entries:
                target_pixels = torch.clamp(pixels, 0.0, 1.0)

                torch.cuda.synchronize()
                tic = time.time()
                colors, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    masks=masks,
                    splat_overrides=subset["overrides"],
                )
                torch.cuda.synchronize()
                elapsed = max(time.time() - tic, 1e-10)

                colors = torch.clamp(colors, 0.0, 1.0)
                if world_rank != 0:
                    continue

                subset["times"].append(elapsed)
                if cfg.mgs_inference_save_images:
                    os.makedirs(self.render_dir, exist_ok=True)
                    pred_img = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_split{subset['label']}_step{step}_{i:04d}_pred.png",
                        pred_img,
                    )
                    if not cfg.save_pred_images:
                        canvas = (
                            torch.cat([target_pixels, colors], dim=2).squeeze(0).cpu().numpy()
                        )
                        canvas = (canvas * 255).astype(np.uint8)
                        imageio.imwrite(
                            f"{self.render_dir}/{stage}_split{subset['label']}_step{step}_{i:04d}.png",
                            canvas,
                        )

                pixels_p = target_pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                gt_p = pixels.permute(0, 3, 1, 2)
                subset["metrics"]["psnr_gt"].append(
                    torch.clamp(self.psnr(colors_p, gt_p), max=60.0)
                )
                subset["metrics"]["ssim_gt"].append(self.ssim(colors_p, gt_p))
                subset["metrics"]["lpips_gt"].append(self.lpips(colors_p, gt_p))

        if world_rank != 0:
            return

        for subset in subset_entries:
            metrics = subset["metrics"]
            if len(metrics["psnr_gt"]) == 0:
                continue
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            if len(subset["times"]) > 3:
                avg_time = float(np.mean(subset["times"][3:]))
            elif subset["times"]:
                avg_time = float(np.mean(subset["times"]))
            else:
                avg_time = 0.0
            fps = 1.0 / avg_time if avg_time > 0 else 0.0
            stats.update(
                {
                    "split": subset["split"],
                    "num_splats": subset["num_splats"],
                    "mrl_position": subset.get("mrl_position", False),
                    "ellipse_time": avg_time,
                    "time_per_image": avg_time,
                    "fps": fps,
                    "num_images": len(dataloader),
                }
            )
            subset_stage = f"{stage}_split{subset['label']}"
            print(
                f"[{subset_stage}] PSNR(GT): {stats.get('psnr_gt', 0):.3f}, "
                f"SSIM(GT): {stats.get('ssim_gt', 0):.4f}, "
                f"LPIPS(GT): {stats.get('lpips_gt', 0):.3f}, "
                f"FPS: {stats['fps']:.2f}, Num GS: {stats['num_splats']}"
            )
            with open(f"{self.stats_dir}/{subset_stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"{subset_stage}/{k}", v, step)
            self.writer.flush()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None and cfg.resume:
        runner.load_checkpoint(cfg.ckpt)
        runner.train()
    elif cfg.ckpt is not None:
        ckpts = [
            _load_checkpoint(file, runner.device)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        if cfg.run_mgs_inference:
            runner.eval_matryoshka_subsets(step=step)
    else:
        if cfg.resume:
            raise ValueError("Resume requested but no checkpoint provided.")
        runner.train()

import tyro

from gsplat.distributed import cli
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from mgs.train.simple_trainer import Config, main


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training with MCMC densification (paper config)
    CUDA_VISIBLE_DEVICES=0 python train.py mcmc \
        --data_dir ../benchmark/MipNeRF360/360_v2/bicycle \
        --data_factor 4 \
        --result_dir ../checkpoint/bicycle \
        --max_steps 50000 \
        --strategy.refine_stop_iter 50000 \
        --strategy.cap-max 5000000 \
        --sort_strategy by_opacity_descending \
        --diffusion_objective single_prefix_full \
        --diffusion_schedule uniform \
        --diffusion_min_keep_ratio 0.0 \
        --diffusion_max_keep_ratio 1.0 \
        --diffusion_include_full_subset \
        --eval_steps 49999 \
        --save_steps 49999 \
        --disable_viewer

    # Distributed training on 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py mcmc --steps_scaler 0.25 ...
    ```
    """

    configs = {
        "default": (
            "MGS with DefaultStrategy densification.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "MGS with MCMC densification (paper config).",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)

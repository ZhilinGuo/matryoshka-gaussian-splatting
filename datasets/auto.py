from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


def _is_blender_dataset(data_dir: str) -> bool:
    p = Path(data_dir)
    return (p / "transforms_train.json").exists()


def build_parser_and_datasets(
    *,
    data_dir: str,
    factor: int,
    normalize: bool,
    test_every: int,
    patch_size: Optional[int],
    load_depths: bool,
    skip_t3: bool = False,
) -> Tuple[object, object, object]:
    """
    Construct (parser, trainset, valset) for either:
    - COLMAP datasets: expects {images/, sparse[/0]/}
    - Blender/NeRF-Synthetic datasets: expects transforms_{train,val,test}.json
    """
    if _is_blender_dataset(data_dir):
        from .blender import Dataset, Parser  # local import to keep deps optional

        parser = Parser(
            data_dir=data_dir,
            factor=factor,
            normalize=normalize,
            test_every=test_every,
        )
        setattr(parser, "dataset_type", "blender")
        trainset = Dataset(
            parser,
            split="train",
            patch_size=patch_size,
            load_depths=load_depths,
        )
        valset = Dataset(parser, split="val")
        return parser, trainset, valset

    from .colmap import Dataset, Parser

    parser = Parser(
        data_dir=data_dir,
        factor=factor,
        normalize=normalize,
        test_every=test_every,
        skip_t3=skip_t3,
    )
    setattr(parser, "dataset_type", "colmap")
    trainset = Dataset(
        parser,
        split="train",
        patch_size=patch_size,
        load_depths=load_depths,
    )
    valset = Dataset(parser, split="val")
    return parser, trainset, valset


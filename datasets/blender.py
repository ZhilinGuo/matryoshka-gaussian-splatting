from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from .normalize import normalize as normalize_cameras_only


def _resolve_blender_image_path(data_dir: str, file_path: str) -> str:
    # NeRF-Synthetic uses paths like "./train/r_0" (no extension).
    rel = file_path.lstrip("./")
    p = Path(data_dir) / rel
    if p.suffix == "":
        # Default to png (NeRF Synthetic convention).
        p = p.with_suffix(".png")
    return str(p)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _downscale_image(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    h, w = image.shape[:2]
    new_w = int(round(w / factor))
    new_h = int(round(h / factor))
    pil = Image.fromarray(image)
    pil = pil.resize((new_w, new_h), Image.BICUBIC)
    return np.asarray(pil)


@dataclass
class _SplitData:
    image_paths: List[str]
    camtoworlds: np.ndarray  # [N, 4, 4]


class Parser:
    """
    Minimal Blender/NeRF-Synthetic parser.

    This intentionally mirrors the attribute surface used by the trainers that
    were originally written for COLMAP datasets.
    """

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,  # kept for API compatibility; unused
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        train_json = os.path.join(data_dir, "transforms_train.json")
        if not os.path.exists(train_json):
            raise ValueError(
                f"Expected Blender dataset at {data_dir}, missing transforms_train.json"
            )

        # Prefer explicit val split if present, otherwise fall back to test.
        val_json = os.path.join(data_dir, "transforms_val.json")
        test_json = os.path.join(data_dir, "transforms_test.json")

        train_meta = _load_json(train_json)
        val_meta = _load_json(val_json) if os.path.exists(val_json) else None
        test_meta = _load_json(test_json) if os.path.exists(test_json) else None

        angle_x = float(train_meta.get("camera_angle_x", 0.0))
        if angle_x <= 0.0:
            raise ValueError("Blender transforms JSON missing camera_angle_x.")

        train_split = self._parse_split(train_meta)
        if val_meta is not None:
            val_split = self._parse_split(val_meta)
        elif test_meta is not None:
            val_split = self._parse_split(test_meta)
        else:
            raise ValueError("Blender dataset missing transforms_val.json and transforms_test.json.")

        self.splits: Dict[str, _SplitData] = {
            "train": train_split,
            "val": val_split,
        }

        # For trajectory rendering, expose a combined camtoworld list (train then val).
        self.camtoworlds = np.concatenate(
            [self.splits["train"].camtoworlds, self.splits["val"].camtoworlds], axis=0
        )

        # Normalize cameras (camera-only similarity transform).
        if normalize:
            self.camtoworlds, self.transform = normalize_cameras_only(self.camtoworlds)
            # Re-slice back into splits.
            n_train = len(self.splits["train"].camtoworlds)
            self.splits["train"].camtoworlds = self.camtoworlds[:n_train]
            self.splits["val"].camtoworlds = self.camtoworlds[n_train:]
        else:
            self.transform = np.eye(4, dtype=np.float32)

        # Intrinsics: assume single shared pinhole intrinsics.
        first_image = imageio.imread(self.splits["train"].image_paths[0])[..., :3]
        first_image = _downscale_image(first_image, factor)
        height, width = first_image.shape[:2]

        focal = 0.5 * float(width) / math.tan(0.5 * angle_x)
        K = np.array([[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)

        self.image_names = [
            os.path.relpath(p, data_dir) for p in self.splits["train"].image_paths
        ] + [os.path.relpath(p, data_dir) for p in self.splits["val"].image_paths]
        self.image_paths = self.splits["train"].image_paths + self.splits["val"].image_paths

        self.camera_ids = [0 for _ in self.image_paths]
        self.Ks_dict = {0: K}
        self.params_dict = {0: np.empty(0, dtype=np.float32)}
        self.imsize_dict = {0: (width, height)}
        self.mask_dict = {0: None}

        # Scene scale (used by several hyperparameters) from camera extent.
        cam_locs = self.camtoworlds[:, :3, 3]
        scene_center = np.mean(cam_locs, axis=0)
        dists = np.linalg.norm(cam_locs - scene_center, axis=1)
        self.scene_scale = float(np.max(dists)) if len(dists) else 1.0

        # Keep COLMAP-like fields for interface compatibility, but Blender datasets
        # do not provide SfM point clouds.
        self.points = np.zeros((0, 3), dtype=np.float32)
        self.points_err = np.zeros((0,), dtype=np.float32)
        self.points_rgb = np.zeros((0, 3), dtype=np.uint8)
        self.point_indices: Dict[str, np.ndarray] = {}

        # Trajectory rendering expects these fields to exist.
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": True}
        self.bounds = np.array([0.01, max(self.scene_scale * 2.0, 1.0)], dtype=np.float32)

        print(
            f"[BlenderParser] train={len(self.splits['train'].image_paths)} "
            f"val={len(self.splits['val'].image_paths)} "
            f"factor={factor} size=({width}x{height})"
        )

    def _parse_split(self, meta: Dict[str, Any]) -> _SplitData:
        frames = meta.get("frames", [])
        if not frames:
            raise ValueError("Blender transforms JSON has no frames.")

        image_paths: List[str] = []
        c2ws: List[np.ndarray] = []
        for fr in frames:
            file_path = fr.get("file_path", "")
            if not file_path:
                raise ValueError("Frame missing file_path in Blender transforms JSON.")
            img_path = _resolve_blender_image_path(self.data_dir, file_path)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Missing image file referenced by transforms JSON: {img_path}")
            image_paths.append(img_path)

            T = np.array(fr["transform_matrix"], dtype=np.float32)
            if T.shape != (4, 4):
                raise ValueError("transform_matrix must be 4x4.")
            # Convert NeRF-Synthetic (OpenGL/Blender) camera convention to the OpenCV-like
            # convention used throughout this codebase.
            # Common conversion: flip Y and Z axes.
            T[:3, 1:3] *= -1.0
            c2ws.append(T)

        camtoworlds = np.stack(c2ws, axis=0)
        return _SplitData(image_paths=image_paths, camtoworlds=camtoworlds)


class Dataset:
    """A Blender dataset class matching the COLMAP Dataset interface."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        if load_depths:
            raise ValueError(
                "Depth supervision is not supported for Blender/NeRF-Synthetic datasets."
            )
        if split not in parser.splits:
            raise ValueError(f"Unknown split '{split}'. Expected one of: {list(parser.splits.keys())}")

        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths

        split_data = parser.splits[split]
        self.image_paths = split_data.image_paths
        self.camtoworlds = split_data.camtoworlds

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        image = imageio.imread(self.image_paths[item])[..., :3]
        image = _downscale_image(image, self.parser.factor)
        camtoworld = self.camtoworlds[item]

        K = self.parser.Ks_dict[0].copy()
        mask = None

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data: Dict[str, Any] = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()
        return data


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--factor", type=int, default=1)
    args = ap.parse_args()

    parser = Parser(data_dir=args.data_dir, factor=args.factor, normalize=False)
    train = Dataset(parser, split="train")
    val = Dataset(parser, split="val")
    print("train", len(train), "val", len(val))


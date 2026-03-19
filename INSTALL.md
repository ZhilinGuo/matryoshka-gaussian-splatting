# Installation

Tested on Ubuntu 24.04 with an NVIDIA A100 (CUDA 12.8, PyTorch 2.8.0).

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- CUDA 12.8 toolkit and compatible NVIDIA driver
- GCC 12

## Steps

### 1. Set up the workspace

```bash
mkdir MatryoshkaGaussianSplatting
cd MatryoshkaGaussianSplatting
git clone https://github.com/ZhilinGuo/matryoshka-gaussian-splatting.git
```

All subsequent commands assume you are inside `MatryoshkaGaussianSplatting/`.

### 2. Create the conda environment

```bash
conda create -n mgs python=3.11 --no-default-packages -y
conda activate mgs
```

### 3. Install PyTorch

```bash
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install Python dependencies

```bash
pip install "numpy<2" "opencv-python>=4.8,<4.12" \
    ninja scipy rich jaxtyping imageio tqdm torchmetrics tyro PyYAML \
    tensorboard scikit-learn
```

> **Why `numpy<2` and `opencv<4.12`?** pycolmap calls `np.uint64(-1)` which
> breaks on NumPy 2, and OpenCV 4.12+ requires NumPy 2.

### 5. Install gsplat (from source)

```bash
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
git checkout 982ce4b2e87c4cd556b7a6ce84ebaa75498b8fc2
cd ..

CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    pip install --no-build-isolation ./gsplat
```

> **First import:** gsplat compiles CUDA kernels on first use; this is normal
> and takes 2--10 minutes.

### 6. Install pycolmap

```bash
pip install git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e
```

### 7. Install fused-ssim

```bash
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 \
    pip install --no-build-isolation \
    "git+https://github.com/rahul-goel/fused-ssim.git@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
```

## Verification

```bash
conda activate mgs
python -c "import torch; import gsplat; print('PyTorch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

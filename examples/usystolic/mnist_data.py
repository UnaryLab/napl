"""
Minimal torchvision-free MNIST loader.

napl's conda env does not ship torchvision, and this example only needs MNIST
tensors (not the whole torchvision stack), so we download and parse the raw IDX
files directly from the same S3 mirror torchvision uses. Images are normalized
with the standard MNIST mean/std (0.1307 / 0.3081), matching the upstream
uSystolic transform, so the FP checkpoint and the HUB/FXP eval see identical
preprocessing.
"""
import gzip
import struct
import urllib.request
from pathlib import Path

import torch

_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081


def _download(data_dir: Path, fname: str) -> Path:
    dest = data_dir / fname
    if dest.exists():
        return dest
    data_dir.mkdir(parents=True, exist_ok=True)
    url = _MIRROR + fname
    print(f"downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _read_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad image magic {magic} in {path}"
        buf = f.read(n * rows * cols)
    data = torch.frombuffer(bytearray(buf), dtype=torch.uint8).view(n, 1, rows, cols)
    return data.float().div_(255.0).sub_(_MNIST_MEAN).div_(_MNIST_STD)


def _read_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad label magic {magic} in {path}"
        buf = f.read(n)
    return torch.frombuffer(bytearray(buf), dtype=torch.uint8).long()


def load_mnist(data_dir, train: bool):
    """Return (images, labels) for the train or test split, downloading if needed."""
    data_dir = Path(data_dir)
    split = "train" if train else "test"
    images = _read_images(_download(data_dir, _FILES[f"{split}_images"]))
    labels = _read_labels(_download(data_dir, _FILES[f"{split}_labels"]))
    return images, labels

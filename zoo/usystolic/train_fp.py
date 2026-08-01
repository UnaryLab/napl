"""
Train the FP32 convnet on MNIST and save its state_dict.

Mirrors the upstream uSystolic convnet_mnist training recipe (Adadelta lr=1.0,
StepLR gamma=0.7, batch 64, NLL loss on log_softmax) but for a small number of
epochs (default 3), which already reaches ~99% test accuracy. The checkpoint is
the input to eval_sweep.py.

Run (from the repo root, napl env):
    conda run -n napl python zoo/usystolic/train_fp.py
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from mnist_data import load_mnist
from model import ConvNetFP

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CKPT = HERE / "checkpoints" / "convnet_mnist.pt"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_batches(images, labels, batch_size, shuffle, device):
    n = images.size(0)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        yield images[idx].to(device), labels[idx].to(device)


def evaluate(model, images, labels, batch_size, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in iter_batches(images, labels, batch_size, False, device):
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return 100.0 * correct / images.size(0)


def main():
    parser = argparse.ArgumentParser(description="Train FP convnet on MNIST")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    train_x, train_y = load_mnist(DATA_DIR, train=True)
    test_x, test_y = load_mnist(DATA_DIR, train=False)
    print(f"train: {tuple(train_x.shape)}  test: {tuple(test_x.shape)}")

    model = ConvNetFP().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for bi, (x, y) in enumerate(iter_batches(train_x, train_y, args.batch_size, True, device)):
            optimizer.zero_grad()
            loss = F.nll_loss(model(x), y)
            loss.backward()
            optimizer.step()
            if bi % 200 == 0:
                print(f"epoch {epoch} batch {bi}  loss {loss.item():.4f}")
        acc = evaluate(model, test_x, test_y, 1000, device)
        print(f"epoch {epoch} test accuracy: {acc:.2f}%")
        scheduler.step()

    CKPT.parent.mkdir(parents=True, exist_ok=True)
    # CPU tensors keep the checkpoint device-independent.
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, CKPT)
    print(f"saved checkpoint -> {CKPT}")


if __name__ == "__main__":
    main()

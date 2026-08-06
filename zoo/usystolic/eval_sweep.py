"""
Reproduce the uSystolic convnet_mnist accuracy sweep.

Loads the FP checkpoint, reports FP32 test accuracy, then evaluates the HUB
(unary-MAC) model across MAC cycle counts and the FXP (fixed-point) model across
the matching bitwidths, where cycle = 2**(bitwidth-1). No retraining: every
precision point reuses the same FP weights. Results are written to
results/convnet_mnist_result.csv with columns: cycle, bitwidth, top1_uSys(HUB),
top1_Fxp.

Run (from the repo root, napl env), after train_fp.py:
    conda run -n napl python zoo/usystolic/eval_sweep.py
    conda run -n napl python zoo/usystolic/eval_sweep.py --test-size 2000 --device cpu
"""
import argparse
import csv
from pathlib import Path

import torch

from mnist_data import load_mnist
from model import ConvNetFP, ConvNetHUB, ConvNetFXP

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CKPT = HERE / "checkpoints" / "convnet_mnist.pt"
RESULT_CSV = HERE / "results" / "convnet_mnist_result.csv"

# Each pair satisfies cycle == 2**(bitwidth - 1).
SWEEP = [(32, 6), (64, 7), (128, 8), (256, 9), (512, 10), (1024, 11)]


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, images, labels, batch_size, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, images.size(0), batch_size):
            x = images[i:i + batch_size].to(device)
            y = labels[i:i + batch_size].to(device)
            pred = model(x).argmax(dim=1)
            if device.type == "mps":
                torch.mps.synchronize()
            correct += (pred == y).sum().item()
    return 100.0 * correct / images.size(0)


def main():
    parser = argparse.ArgumentParser(description="uSystolic convnet_mnist accuracy sweep")
    parser.add_argument("--test-size", type=int, default=2000,
                        help="number of test images to evaluate (default 2000; 10000 = full set)")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", type=str, default=None, help="cpu | mps (default: mps if available)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")

    if not CKPT.exists():
        raise FileNotFoundError(f"checkpoint {CKPT} not found; run train_fp.py first")
    state = torch.load(CKPT, map_location="cpu")

    test_x, test_y = load_mnist(DATA_DIR, train=False)
    n = min(args.test_size, test_x.size(0))
    test_x, test_y = test_x[:n], test_y[:n]
    print(f"evaluating on {n} test images, batch {args.batch_size}")

    fp = ConvNetFP().to(device)
    fp.load_state_dict(state)
    fp_acc = evaluate(fp, test_x, test_y, args.batch_size, device)
    print(f"\nFP32 baseline top-1: {fp_acc:.2f}%\n")

    rows = []
    print(f"{'cycle':>6} {'bw':>4} {'HUB top-1':>10} {'FXP top-1':>10}")
    for cycle, bw in SWEEP:
        hub = ConvNetHUB(state, cycle=cycle, width=bw).to(device)
        hub_acc = evaluate(hub, test_x, test_y, args.batch_size, device)
        del hub

        fxp = ConvNetFXP(state, bitwidth=bw).to(device)
        fxp_acc = evaluate(fxp, test_x, test_y, args.batch_size, device)
        del fxp

        print(f"{cycle:>6} {bw:>4} {hub_acc:>9.2f}% {fxp_acc:>9.2f}%")
        rows.append((cycle, bw, hub_acc, fxp_acc))

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "bitwidth", "top1_uSys(HUB)", "top1_Fxp"])
        for cycle, bw, hub_acc, fxp_acc in rows:
            w.writerow([cycle, bw, f"{hub_acc:.2f}", f"{fxp_acc:.2f}"])
    print(f"\nFP32 baseline top-1: {fp_acc:.2f}%  (device {device}, {n} images)")
    print(f"wrote {RESULT_CSV}")


if __name__ == "__main__":
    main()

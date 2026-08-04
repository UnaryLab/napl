"""
Evaluate the trained MLP3 in napl's streaming spike domain and reproduce
the UnarySim per-cycle (progressive-precision) accuracy curve.

Pipeline (a batch of flattened 32x32 MNIST images, all bipolar in [-1, 1]):

    encode input -> linear_pc(fc1) -> relu+clamp -> linear_pc(fc2) -> relu+clamp -> linear_pc(fc3)

Each linear layer is streamed for T = 2**bitwidth cycles. The kernel used is `linear_pc`
(the parallel-counter streaming linear, UnarySim's FSULinearPC): every cycle it returns the binary
inner-product count of the input spikes against freshly Sobol-encoded weight spikes, on a
distinct RNG dimension from the input so the operand streams are decorrelated. Accumulating
the count over k cycles and forming 2*(count/k) - entry recovers the bipolar inner product
W x + b with progressively higher precision as k grows. The activation (relu after a [-1, 1]
clamp) matches the trained clamp-eval model.

The layers are streamed sequentially: layer L is run to convergence (all T cycles), its
activation is read out, then layer L+1 is streamed. This keeps every layer's input stream
stationary (a re-encoded fixed value), which is what gives the high per-cycle fidelity; the
napl per-timestep `linear` (scaled saturating adder) instead crushes a fan-in-1024 inner
product into a near-zero range that the downstream ReLU cannot resolve (see the README note).
The reported per-cycle curve is the final layer's running prediction accuracy at each cycle
count, which is the progressive-precision readout: it rises from chance toward the
floating-point baseline as the streamed bit count grows.

Decorrelation: operands that must be independent are on distinct Sobol dimensions. Each layer
uses one dimension for its input and one for its weight/bias encoder; the three layers use
disjoint dimension pairs (1/2, 3/4, and 5/6).

Run (after train_fp.py):
    conda run -n napl python zoo/mlp/eval_unary.py
    conda run -n napl python zoo/mlp/eval_unary.py --device cpu --samples 64
    conda run -n napl python zoo/mlp/eval_unary.py --sanity
"""

import argparse
import csv
import os
import time

import torch
import torchvision
import torchvision.transforms as transforms

from napl import encode, linear_pc, relu_hub

from model import MLP3_clamp_eval

BITWIDTH = 8
TIMESTEP = 2 ** BITWIDTH
POLARITY = 'bipolar'
GENERATOR = 'sobol'

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
CKPT_PATH = os.path.join(HERE, 'checkpoints', 'mlp3_mnist.pt')
RESULT_PATH = os.path.join(HERE, 'results', 'cycle_accuracy_mlp.csv')


def get_device(name):
    if name == 'mps' and not torch.backends.mps.is_available():
        raise SystemExit('MPS requested but not available.')
    return torch.device(name)


def load_model(device):
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    model = MLP3_clamp_eval(in_size=ckpt['in_size'], width=ckpt['width'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(device)
    return model, ckpt


def get_test_data(samples, device):
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    testset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=samples, shuffle=False)
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)


def fp_accuracy(model, images, labels):
    with torch.no_grad():
        pred = model(images).argmax(dim=1)
    return (pred == labels).float().mean().item()


def stream_layer(value_in, weight, bias, in_dim, w_dim, timestep, device, record_cycles=False):
    """
    Stream one linear layer for `timestep` cycles. Returns the converged bipolar inner
    product (W x + b). If record_cycles, also returns the list of per-cycle running values
    (the progressive-precision estimates), used to build the final-layer accuracy curve.
    """
    out_features, in_features = weight.shape
    entry = in_features + 1  # Include the bias term.
    enc = encode({'polarity': POLARITY, 'timestep': timestep, 'generator': GENERATOR, 'dim': in_dim}).to(device)
    fc = linear_pc(weight.clone(), bias.clone(),
                       {'polarity': POLARITY, 'timestep': timestep, 'generator': GENERATOR, 'dim': w_dim}).to(device)

    acc = torch.zeros(value_in.size(0), out_features, device=device)
    cycle_vals = [] if record_cycles else None
    for k in range(1, timestep + 1):
        acc = acc + fc(enc(value_in))
        # Decode the bipolar inner product from the running count.
        v = 2.0 * (acc / k) - entry
        if record_cycles:
            cycle_vals.append(v.clone())
    final = v if not record_cycles else cycle_vals[-1]
    return (final, cycle_vals) if record_cycles else final


def run_unary(model, images, labels, timestep, device, sync):
    """
    Layer-sequential streaming of MLP3. Returns the per-cycle accuracy array read from
    the final (fc3) layer's progressive-precision prediction.
    """
    image_value = images.view(-1, model.in_size).clamp(-1, 1).to(device)
    W1, b1 = model.fc1.weight.detach(), model.fc1.bias.detach()
    W2, b2 = model.fc2.weight.detach(), model.fc2.bias.detach()
    W3, b3 = model.fc3.weight.detach(), model.fc3.bias.detach()
    activation = relu_hub({'scale': 1.0}).to(device)

    with torch.no_grad():
        v1 = stream_layer(image_value, W1, b1, 1, 2, timestep, device)
        a1 = activation(v1)
        v2 = stream_layer(a1, W2, b2, 3, 4, timestep, device)
        a2 = activation(v2)
        _, c3 = stream_layer(a2, W3, b3, 5, 6, timestep, device, record_cycles=True)

    if sync:
        torch.mps.synchronize()

    cycle_acc = torch.zeros(timestep)
    for k in range(timestep):
        pred = c3[k].argmax(dim=1)
        cycle_acc[k] = (pred == labels).float().mean().item()
    return cycle_acc.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default=None, help='cpu or mps (default: mps if available else cpu)')
    parser.add_argument('--samples', type=int, default=256, help='number of test images to stream')
    parser.add_argument('--timestep', type=int, default=TIMESTEP, help='cycles to stream (default 256)')
    parser.add_argument('--sanity', action='store_true',
                        help='tiny both-device sanity check (few samples, few cycles) and exit')
    args = parser.parse_args()

    if args.sanity:
        devices = ['cpu'] + (['mps'] if torch.backends.mps.is_available() else [])
        for dname in devices:
            device = get_device(dname)
            model, _ = load_model(device)
            images, labels = get_test_data(32, device)
            t0 = time.time()
            ca = run_unary(model, images, labels, 64, device, sync=(dname == 'mps'))
            print(f'[sanity {dname}] 32 samples x 64 cycles in {time.time() - t0:.2f}s, '
                  f'final-cycle acc={ca[-1].item():.3f}')
        return

    dname = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    device = get_device(dname)
    print(f'eval device: {device}, samples: {args.samples}, timestep: {args.timestep}')

    model, _ = load_model(device)
    images, labels = get_test_data(args.samples, device)

    fp_acc = fp_accuracy(model, images, labels)
    print(f'FP (clamp-eval) accuracy on {args.samples} test images: {fp_acc:.4f}')

    t0 = time.time()
    cycle_acc = run_unary(model, images, labels, args.timestep, device, sync=(dname == 'mps'))
    dt = time.time() - t0
    print(f'unary streaming: {args.timestep} cycles x {args.samples} samples in {dt:.1f}s ({device})')

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    with open(RESULT_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cycle', 'accuracy'])
        for k in range(args.timestep):
            w.writerow([k + 1, f'{cycle_acc[k].item():.6f}'])
    print(f'wrote per-cycle accuracy curve to {RESULT_PATH}')

    marks = [c for c in (1, 16, 64, 256) if c <= args.timestep]
    print('per-cycle accuracy:', {c: round(cycle_acc[c - 1].item(), 4) for c in marks})
    print(f'final-cycle ({args.timestep}) unary accuracy: {cycle_acc[-1].item():.4f}')
    print(f'FP baseline: {fp_acc:.4f}')


if __name__ == '__main__':
    main()

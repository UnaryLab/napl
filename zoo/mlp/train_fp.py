"""
Train MLP3 on MNIST (resized to 32x32) in floating point, then save the state_dict for the
unary-computing evaluation in eval_unary.py.

This is intentionally small: a narrow hidden width and a few epochs, just enough to give a
usable accuracy for the unary demo (the point of the example is the per-cycle unary curve,
not a state-of-the-art classifier). After every epoch the weights and biases are clamped and
quantized to the stochastic-computing range with NN_SC_Weight_Clipper(bitwidth=8), matching
the bipolar 8-bit streams used at eval time.

Run:
    conda run -n napl python zoo/mlp/train_fp.py
"""

import os
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from napl.utils import NN_SC_Weight_Clipper

from model import MLP3_clamp_train

WIDTH = 128
EPOCHS = 3
BATCH = 128
LR = 1e-3
BITWIDTH = 8
IN_SIZE = 32 * 32

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
CKPT_DIR = os.path.join(HERE, 'checkpoints')
CKPT_PATH = os.path.join(CKPT_DIR, 'mlp3_mnist.pt')


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_loaders():
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
    return trainloader, testloader


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main():
    device = get_device()
    print(f'training on device: {device}')
    os.makedirs(CKPT_DIR, exist_ok=True)

    trainloader, testloader = get_loaders()

    model = MLP3_clamp_train(in_size=IN_SIZE, width=WIDTH).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # Keep trained weights in the bitwidth-8 bipolar range.
    clipper = NN_SC_Weight_Clipper(bitwidth=BITWIDTH, mode='bipolar')

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        model.apply(clipper)
        acc = evaluate(model, testloader, device)
        print(f'epoch {epoch + 1}/{EPOCHS}: test acc = {acc:.4f}  ({time.time() - t0:.1f}s)')

    final_acc = evaluate(model, testloader, device)
    print(f'final FP test accuracy (clamped/quantized weights): {final_acc:.4f}')

    # CPU tensors keep the checkpoint device-independent.
    torch.save({'state_dict': model.cpu().state_dict(), 'width': WIDTH, 'in_size': IN_SIZE,
                'fp_test_acc': final_acc}, CKPT_PATH)
    print(f'saved checkpoint to {CKPT_PATH}')


if __name__ == '__main__':
    main()

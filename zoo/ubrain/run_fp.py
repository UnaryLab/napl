"""
Run the FP uBrain Cascade_CNN_RNN end-to-end on EEG-shaped RANDOM input.

This confirms the model runs (forward + a tiny synthetic train loop) on CPU and
MPS and prints the per-head output shapes. There is no EEG data and no trained
checkpoint here, so NOTHING in this script is a classification result: the input
is random and the train loop uses RANDOM labels purely as a smoke test that the
training path executes.

Run:
    conda run -n napl python zoo/ubrain/run_fp.py
"""

import time
import torch

from model import Cascade_CNN_RNN_FP


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def eeg_input(batch, win, input_sz, device):
    # EEG-shaped input uses (batch, win, height, width) and values in [-1, 1].
    return (torch.rand(batch, win, input_sz[0], input_sz[1], device=device) * 2 - 1)


def devices():
    available = ['cpu']
    if torch.cuda.is_available():
        available.append('cuda')
    if torch.backends.mps.is_available():
        available.append('mps')
    return available


def main():
    torch.manual_seed(0)
    input_sz = (10, 11)          # 10-10 MI grid
    win = 10
    num_class = (5, 2)           # MI 5-class + SP 2-class heads
    batch = 4

    print('=== uBrain FP Cascade_CNN_RNN smoke run (SYNTHETIC random input) ===')
    print(f'input_sz={input_sz}, win={win}, num_class={num_class}, batch={batch}')
    print(f'EEG input shape per batch: (batch={batch}, win={win}, h={input_sz[0]}, w={input_sz[1]})')

    for device in devices():
        print(f'\n--- device: {device} ---')
        model = Cascade_CNN_RNN_FP(input_sz=input_sz, rnn_win_sz=win, num_class=num_class).to(device)
        model.eval()
        x = eeg_input(batch, win, input_sz, device)

        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        sync(device)
        dt = time.perf_counter() - t0
        heads = model.split_heads(out)
        print(f'forward OK in {dt*1e3:.1f} ms; flat output shape {tuple(out.shape)}')
        for k, h in enumerate(heads):
            print(f'  head[{k}] (num_class={num_class[k]}) shape {tuple(h.shape)}, '
                  f'range [{h.min().item():.3f}, {h.max().item():.3f}]')

        # Random labels exercise the synthetic training path; they are not results.
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        lossfn = torch.nn.CrossEntropyLoss()
        labels = [torch.randint(0, c, (batch,), device=device) for c in num_class]
        losses = []
        for step in range(3):
            opt.zero_grad()
            out = model(x)
            heads = model.split_heads(out)
            loss = sum(lossfn(h, lab) for h, lab in zip(heads, labels))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f'synthetic train smoke (random labels, 3 steps) loss: '
              f'{[round(loss, 4) for loss in losses]}  [NOT a real result]')

    print('\nAll devices ran. (Random input/labels -> no classification meaning.)')


if __name__ == '__main__':
    main()

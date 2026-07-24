"""
FP-vs-HUB fidelity of the uBrain Cascade_CNN_RNN, swept over the unary bitwidth.

Builds one FP model and a HUB model that SHARES its weights (build_hub_from_fp),
runs BOTH on the same EEG-shaped random input, and reports the error of the HUB
(unary) output against the FP output. Sweeping `width` shows the error shrinking
as cycles grow (more cycles -> finer unary precision): the spirit of UnarySim
app/uBrain/layer_eval (RMSE vs bitwidth) and model_hub's ProgError reporting,
applied to the whole network output.

This is a FIDELITY check, not EEG classification: with no EEG data and no trained
checkpoint, the input is random and the FP output is the reference. The number
that matters is how closely the unary HUB output tracks the FP output.

The HUB MGU streams 2**width cycles internally and is looped over `win` steps, so
batch and width are kept small. Sweep runs on CPU; MPS is sanity-checked at one
width.

Run:
    /Users/diwu/anaconda3/envs/napl/bin/python examples/ubrain/eval_hub_fidelity.py
(equivalently `conda run -n napl python ...`; see README for the env note.)
"""

import os
import csv
import time
import torch

from model import Cascade_CNN_RNN_FP, build_hub_from_fp

RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'hub_fidelity.csv')


def sync(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def fidelity(fp_model, x, width, rng, device):
    """Build a weight-sharing HUB model at `width`, return (rmse, max_abs_err, seconds)."""
    with torch.no_grad():
        ref = fp_model(x)
    hub = build_hub_from_fp(fp_model, width=width, rng=rng).to(device).eval()
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = hub(x)
    sync(device)
    dt = time.perf_counter() - t0
    err = (out - ref)
    return err.pow(2).mean().sqrt().item(), err.abs().max().item(), dt


def main():
    torch.manual_seed(0)
    input_sz = (10, 11)        # 10-10 MI grid
    win = 10
    num_class = (5, 2)
    batch = 2                  # tiny: the HUB MGU runs 2**width cycles per step
    rng = 'sobol'
    widths = [6, 8, 10]

    print('=== uBrain FP-vs-HUB fidelity sweep (SYNTHETIC random input) ===')
    print(f'input_sz={input_sz}, win={win}, num_class={num_class}, batch={batch}, rng={rng}')
    print(f'EEG input shape: (batch={batch}, win={win}, h={input_sz[0]}, w={input_sz[1]})')
    print('FP output is the reference; HUB output is the unary approximation.\n')

    # CPU sweep
    device = 'cpu'
    fp = Cascade_CNN_RNN_FP(input_sz=input_sz, rnn_win_sz=win, num_class=num_class).to(device).eval()
    x = (torch.rand(batch, win, input_sz[0], input_sz[1], device=device) * 2 - 1)

    rows = []
    print(f'--- device: {device} ---')
    print(f'{"width":>6} {"cycles":>8} {"rmse":>10} {"max_abs_err":>12} {"seconds":>9}')
    for width in widths:
        cycles = 2 ** width
        rmse, maxerr, dt = fidelity(fp, x, width, rng, device)
        print(f'{width:>6} {cycles:>8} {rmse:>10.5f} {maxerr:>12.5f} {dt:>9.2f}')
        rows.append({'width': width, 'cycles': cycles, 'rmse': round(rmse, 6),
                     'max_abs_err': round(maxerr, 6)})

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['width', 'cycles', 'rmse', 'max_abs_err'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nsaved sweep to {RESULTS}')

    # MPS sanity check at one width: confirms the full HUB path runs on MPS.
    if torch.backends.mps.is_available():
        device = 'mps'
        w_chk = 8
        print(f'\n--- device: {device} (sanity check at width={w_chk}) ---')
        fp_m = fp.to(device)
        x_m = x.to(device)
        try:
            rmse, maxerr, dt = fidelity(fp_m, x_m, w_chk, rng, device)
            print(f'width={w_chk} cycles={2**w_chk}: rmse={rmse:.5f} max_abs_err={maxerr:.5f} '
                  f'in {dt:.2f}s  [MPS sanity OK]')
        except RuntimeError as e:
            print(f'MPS sanity check FAILED: {e}')
            print('  -> the CPU sweep above is still valid; debug the MPS path separately.')

    print('\nFidelity check done. Error should fall as width/cycles grow.')


if __name__ == '__main__':
    main()

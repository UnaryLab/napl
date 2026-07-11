import torch

from napl.utils import *
from napl.module import encoder
from napl.metric import stability


def test_stability():
    """
    Stability is a per-element fraction in [0, 1]; sobol-coded streams settle within
    threshold well before the end of the run, so the mean stability is high.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    timestep = 256
    cfg = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8).to(device)

    enc = encoder(cfg).to(device)
    stab = stability(val, {'polarity': 'bipolar', 'threshold': 0.05}).to(device)

    for _ in range(timestep):
        stab(enc(val))

    result = stab.report_stab()
    print(f'stability mean={result.mean().item():.4f}, min={result.min().item():.4f}, max={result.max().item():.4f}')

    assert result.min() >= 0.0 and result.max() <= 1.0, (result.min(), result.max())
    assert result.mean() > 0.3, result.mean()

    stab.reset()
    print('Test passed.')


if __name__ == '__main__':
    test_stability()

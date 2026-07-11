import torch

from napl.utils import *
from napl.module import encoder
from napl.metric import correlation


def test_correlation():
    """
    SCC of a stream with an identical copy is +1, with its complement is -1, and with
    an independent (different-rng-dim) stream is near 0.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    timestep = 256
    cfg = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    cfg_indep = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2}

    val = gen_rand_tensor('bipolar', shape=(1000,), width=8).to(device)

    enc_a = encoder(cfg).to(device)
    enc_b = encoder(cfg).to(device)          # same value, same dim -> identical stream
    enc_indep = encoder(cfg_indep).to(device)  # different dim -> independent stream

    corr_self = correlation().to(device)
    corr_inv = correlation().to(device)
    corr_indep = correlation().to(device)

    for _ in range(timestep):
        s_a = enc_a(val)
        s_b = enc_b(val)
        s_indep = enc_indep(val)
        corr_self(s_a, s_b)
        corr_inv(s_a, 1 - s_a)
        corr_indep(s_a, s_indep)

    scc_self = corr_self.report_corr()
    scc_inv = corr_inv.report_corr()
    scc_indep = corr_indep.report_corr()
    print(f'SCC self={scc_self.mean().item():.4f}, inv={scc_inv.mean().item():.4f}, indep={scc_indep.mean().item():.4f}')

    # identical/complement streams give SCC +1/-1; a few extreme-valued elements code to
    # constant streams whose SCC is 0 by definition, so check the mean rather than each element.
    assert scc_self.mean() > 0.99, scc_self.mean()
    assert scc_inv.mean() < -0.99, scc_inv.mean()
    assert scc_indep.abs().mean() < 0.2, scc_indep.abs().mean()

    print('Test passed.')


if __name__ == '__main__':
    test_correlation()

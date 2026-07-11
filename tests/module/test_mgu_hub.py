import torch

from napl.base import global_config
from napl.utils import gen_rand_tensor
from napl.module import mgu_hard, mgu_hub


def test_mgu_hub():
    """
    The hybrid mgu_hub (which runs the streaming mgu_fsu inner cell over 2**width cycles and
    decodes with the accuracy metric) reproduces the float mgu_hard within a stochastic-
    computing bound, validating both mgu_fsu and mgu_hub.
    """
    ntype = global_config.ntype
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 3

    ref = mgu_hard(isz, hsz, bias=True)
    Wf, bf, Wn, bn = ref.weight_f.data, ref.bias_f.data, ref.weight_n.data, ref.bias_n.data
    x = gen_rand_tensor('bipolar', (b, isz), 8).type(ntype)
    hx = gen_rand_tensor('bipolar', (b, hsz), 8).type(ntype)

    y_ref = ref(x, hx)
    hub = mgu_hub(isz, hsz, bias=True, weight_f=Wf, bias_f=bf, weight_n=Wn, bias_n=bn,
                  config={'polarity': 'bipolar', 'width': 8, 'generator': 'sobol'})
    y_hub = hub(x, hx)
    rmse = (y_hub - y_ref).pow(2).mean().sqrt().item()
    print(f'mgu_hub vs mgu_hard rmse={rmse:.4f}')

    assert y_hub.shape == y_ref.shape
    # the streaming MGU compounds ~7 SC ops (2 linears, sigmoid, 2 muls, add) so its error
    # envelope is wider than a single kernel: worst-case per-seed rmse ~0.11 at width 8,
    # tightening with width. Bound set above that envelope, not the seed-0 value.
    assert rmse < 0.15, rmse

    # hx=None path runs
    assert hub(x).shape == (b, hsz)

    print('Test passed.')


if __name__ == '__main__':
    test_mgu_hub()

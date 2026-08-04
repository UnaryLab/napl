import sys

import torch

sys.path.insert(0, '/Users/diwu/Projects')

from napl.sim.base import global_config
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, timer
from napl.sim.module import mgu, mgu_hard, mgu_hub
from napl.sim.operation import encode
from napl.sim.operation import mul_ugemm_sr
from UnarySim.kernel.rnn import FSUMGUCell


def test_mgu_hub():
    """Verify mgu_hub decodes close to mgu_hard within the stochastic error bound."""
    ntype = global_config.ntype
    torch.manual_seed(0)
    isz, hsz, b = 6, 4, 3

    ref = mgu_hard(isz, hsz, bias=True)
    x_cpu = gen_rand_tensor('bipolar', (b, isz), 8).type(ntype)
    hx_cpu = gen_rand_tensor('bipolar', (b, hsz), 8).type(ntype)

    for device in devices():
        ref = ref.to(device)
        x = x_cpu.to(device)
        hx = hx_cpu.to(device)
        with timer(device) as ref_elapsed:
            y_ref = ref(x, hx)
        hub = mgu_hub(
            isz,
            hsz,
            bias=True,
            weight_f=ref.weight_f.data,
            bias_f=ref.bias_f.data,
            weight_n=ref.weight_n.data,
            bias_n=ref.bias_n.data,
            config={'polarity': 'bipolar', 'width': 8, 'generator': 'sobol'},
        ).to(device)
        with timer(device) as elapsed:
            y_hub = hub(x, hx)
        rmse = (y_hub - y_ref).pow(2).mean().sqrt().item()
        print(
            f'[{device}] mgu_hub rmse={rmse:.4f}, '
            f'hard/hub ratio={ref_elapsed.seconds / max(elapsed.seconds, 1e-12):.2f}x'
        )

        assert y_hub.shape == y_ref.shape
        assert rmse < 0.15, (device, rmse)
        assert hub(x).shape == (b, hsz)

    print('Test passed.')

    test_mgu_unarysim_recurrent_path()


def test_mgu_unarysim_recurrent_path():
    """Compare the streaming MGU recurrent path with UnarySim FSUMGUCell."""
    torch.manual_seed(7)
    input_size, hidden_size, batch, timesteps = 2, 2, 2, 64
    base = mgu_hard(input_size, hidden_size, bias=True)
    input_values = torch.tensor([[0.25, -0.5], [-0.75, 0.5]], dtype=global_config.ntype)
    hidden_values = torch.tensor([[-0.5, 0.75], [0.25, -0.25]], dtype=global_config.ntype)

    for device in devices():
        weights = {
            'weight_f': base.weight_f.detach().to(device),
            'bias_f': base.bias_f.detach().to(device),
            'weight_n': base.weight_n.detach().to(device),
            'bias_n': base.bias_n.detach().to(device),
        }
        nap = mgu(
            weights['weight_f'], weights['bias_f'],
            weights['weight_n'], weights['bias_n'], hidden_values.to(device),
            {
                'polarity': 'bipolar',
                'timestep': timesteps,
                'generator': 'sobol',
                'width': 12,
                'depth_ismul': 6,
            },
        ).to(device)
        assert isinstance(nap.fg_ng_mul, mul_ugemm_sr)
        assert nap.fg_ng_mul.width == 6
        ref = FSUMGUCell(
            input_size,
            hidden_size,
            bias=True,
            binary_weight_f=weights['weight_f'],
            binary_bias_f=weights['bias_f'],
            binary_weight_n=weights['weight_n'],
            binary_bias_n=weights['bias_n'],
            hx_buffer=hidden_values.to(device),
            bitwidth=6,
            mode='bipolar',
            depth=timesteps,
            depth_ismul=6,
        ).to(device)
        i_enc = encode({
            'polarity': 'bipolar', 'timestep': timesteps,
            'generator': 'sobol', 'dim': 1,
        }).to(device)
        h_enc = encode({
            'polarity': 'bipolar', 'timestep': timesteps,
            'generator': 'sobol', 'dim': 2,
        }).to(device)
        n_acc = torch.zeros(batch, hidden_size, dtype=torch.float32, device=device)
        r_acc = torch.zeros_like(n_acc)
        input_device = input_values.to(device)
        hidden_device = hidden_values.to(device)
        for _ in range(timesteps):
            input_spike = i_enc(input_device)
            hidden_spike = h_enc(hidden_device)
            n_acc.add_(nap(input_spike, hidden_spike).float())
            r_acc.add_(ref(input_spike.float(), hidden_spike.float()).float())
        nap_value = 2 * n_acc / timesteps - 1
        ref_value = 2 * r_acc / timesteps - 1
        rmse = (nap_value - ref_value).pow(2).mean().sqrt().item()
        print(f'[{device}] mgu vs FSUMGUCell rmse={rmse:.4f}')
        assert rmse < 0.25, f'[{device}] recurrent MGU RMSE {rmse:.4f} exceeds bound'


if __name__ == '__main__':
    test_mgu_hub()

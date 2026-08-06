import torch

from napl.sim.base import global_config
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, timer
from napl.sim.module import mgu_hard, mgu_hub
from napl.sim.operation import mul_ugemm_sr


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

        # The wrapper owns its children, so a repeated call must restart them.
        assert torch.equal(hub(x, hx), y_hub), device
        # The internal cell composes the shift-register multiplier on the gate product.
        assert isinstance(hub.cell.fg_ng_mul, mul_ugemm_sr)
        assert hub.cell.fg_ng_mul.width == 6

    print('Test passed.')


def test_mgu_hub_polarity():
    """Verify mgu_hub rejects unipolar configuration."""
    isz, hsz = 6, 4
    try:
        mgu_hub(isz, hsz, bias=True, weight_f=torch.zeros(hsz, hsz + isz), bias_f=torch.zeros(hsz),
                weight_n=torch.zeros(hsz, hsz + isz), bias_n=torch.zeros(hsz),
                config={'polarity': 'unipolar', 'width': 6, 'generator': 'sobol'})
    except AssertionError:
        print('Test passed.')
        return
    raise AssertionError('mgu_hub accepted a unipolar configuration')


def test_mgu_hub_run_length():
    """Verify mgu_hub rejects a run that does not outlast the multiplier shift register."""
    isz, hsz = 6, 4
    weight = torch.zeros(hsz, hsz + isz)
    bias = torch.zeros(hsz)

    def build(width, depth_ismul):
        return mgu_hub(isz, hsz, bias=True, weight_f=weight, bias_f=bias,
                       weight_n=weight, bias_n=bias,
                       config={'polarity': 'bipolar', 'width': width,
                               'generator': 'sobol', 'depth_ismul': depth_ismul})

    for width in (4, 6):
        try:
            build(width, 6)
        except AssertionError as error:
            print(f'expected AssertionError at width={width}: {error}')
        else:
            raise AssertionError('a run that does not outlast the shift register must raise')

    # A run one step longer than the shift register is legal and still produces a hidden state.
    for device in devices():
        cell = build(7, 6).to(device)
        assert cell(torch.zeros(2, isz, device=device)).shape == (2, hsz)
        assert cell.lin_width == 10
        # The config key sets the floor of the internal accumulator width.
        assert mgu_hub(isz, hsz, bias=True, weight_f=weight, bias_f=bias,
                       weight_n=weight, bias_n=bias,
                       config={'polarity': 'bipolar', 'width': 7, 'generator': 'sobol',
                               'width_acc': 14}).lin_width == 14

    print('Test passed.')


if __name__ == '__main__':
    test_mgu_hub()
    test_mgu_hub_polarity()
    test_mgu_hub_run_length()

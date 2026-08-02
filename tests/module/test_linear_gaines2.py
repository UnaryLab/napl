import time
import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder
from napl.sim.module.linear_gaines2 import linear_gaines2
from napl.sim.module.linear import linear


class napl_linear_gaines2(napl_base):
    """Wire encoder -> linear_gaines2 and count the output spikes per-timestep."""


    def __init__(self, codec_config, lin_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.lin = linear_gaines2(weight, bias, lin_config)
        self.spike_cnt = None


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        o_spike = self.lin(self.encoder(input_x)).type(self.ntype)
        self.spike_cnt = o_spike if self.spike_cnt is None else self.spike_cnt + o_spike


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.encoder.reset()
        self.lin.reset()
        self.spike_cnt = None


def _decode(mean_spike, polarity):
    return 2 * mean_spike - 1 if polarity == 'bipolar' else mean_spike


def _kernel_specific_checks():
    """
    Gaines linear (gMUL + uADD) vs the analytic reference within the SC bound.
    scaled=True decodes to (Wx+b)/entry; scaled=False tracks clamp(Wx+b, lo, 1)
    (inputs scaled by 1/in_features so the sum stays in unary range).
    """
    torch.manual_seed(0)
    timestep = 1024
    in_features, out_features = 16, 8
    bound = 1.0 / (timestep ** 0.5)  # SC error scales as 1/sqrt(N).

    for device in devices():
        for polarity in ['unipolar', 'bipolar']:
            lo = 0.0 if polarity == 'unipolar' else -1.0
            for scaled in [True, False]:
                for has_bias in [True, False]:
                    weight = gen_rand_tensor(polarity, shape=(out_features, in_features), width=8).type(global_config.ntype).to(device)
                    bias = gen_rand_tensor(polarity, shape=(out_features,), width=8).type(global_config.ntype).to(device) if has_bias else None
                    input_x = gen_rand_tensor(polarity, shape=(in_features,), width=8).type(global_config.ntype).to(device)
                    if not scaled:
                        # Keep Wx+b within the non-scaled unary output range.
                        input_x = input_x / in_features

                    # Use an input Sobol dimension beyond all weight and bias dimensions.
                    inst = napl_linear_gaines2(
                        {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': in_features + 2},
                        {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'scaled': scaled},
                        weight, bias).to(device)
                    inst(input_x, timesteps=timestep)
                    val = _decode(inst.spike_cnt / timestep, polarity)

                    ref = torch.nn.functional.linear(input_x, weight, bias)
                    ref = ref / inst.lin.entry if scaled else ref.clamp(lo, 1.0)
                    err = (val - ref).abs()
                    rmse = torch.sqrt(err.pow(2).mean()).item()
                    tol = bound * 2 if scaled else bound * 4  # Non-scaled Gaines converges slower.
                    assert err.max().item() < tol, \
                        f'{device}/{polarity}/scaled={scaled}/bias={has_bias}: max_err={err.max().item()} >= {tol}'
                    assert inst.lin.timestep_cur == timestep
                    print(f'[{device}] {polarity} scaled={scaled} bias={has_bias}: '
                          f'rmse={rmse:.5f} max_err={err.max().item():.5f}')
                    inst.reset()

    # All-ones unipolar operands make the scaled output emit 1 every timestep.
    for device in devices():
        w1 = torch.ones(out_features, in_features).type(global_config.ntype).to(device)
        x1 = torch.ones(in_features).type(global_config.ntype).to(device)
        inst = napl_linear_gaines2(
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': in_features + 2},
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'scaled': True},
            w1, None).to(device)
        inst(x1, timesteps=64)
        assert torch.equal(
            inst.spike_cnt, torch.full((out_features,), 64.0, device=device)
        ), 'all-ones unipolar scaled must emit a spike every timestep'
        inst.reset()
        print(f'[{device}] known-answer corner passed.')


def _kernel_specific_perf():
    """Time linear_gaines2 against linear (same fan-in) on identical input spikes."""
    timestep = 256
    in_features, out_features = 16, 8
    torch.manual_seed(0)
    input_cpu = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype)
    weight_cpu = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    bias_cpu = gen_rand_tensor('bipolar', shape=(out_features,), width=8).type(global_config.ntype)

    for device in devices():
        input_x = input_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': in_features + 2}).to(device)
        gaines = linear_gaines2(weight, bias, {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'scaled': True}).to(device)
        lin = linear(weight, bias, {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'scale': None, 'width': 12}).to(device)

        spikes = [enc(input_x) for _ in range(timestep)]
        enc.reset()

        sync(device)
        t0 = time.time()
        for spike in spikes:
            gaines(spike)
        sync(device)
        t_gaines = time.time() - t0
        gaines.reset()

        sync(device)
        t0 = time.time()
        for spike in spikes:
            lin(spike)
        sync(device)
        t_lin = time.time() - t0
        lin.reset()

        print(f'[{device}] linear_gaines2 {t_gaines*1e3:.2f} ms vs linear {t_lin*1e3:.2f} ms '
              f'({t_lin/max(t_gaines, 1e-9):.2f}x)')


def _suite_weight(polarity):
    if polarity == 'unipolar':
        return torch.tensor([
            [0.25, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.25],
        ])
    return torch.tensor([
        [-0.75, -0.25, 0.25, 0.75],
        [0.75, 0.25, -0.25, -0.75],
    ])


def make_operation(polarity, timestep, _device):
    return linear_gaines2(
        _suite_weight(polarity), None,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'scaled': True,
        },
    )


def make_values(polarity):
    if polarity == 'unipolar':
        return (torch.tensor([0.1, 0.3, 0.5, 0.7]),)
    return (torch.tensor([-0.75, -0.25, 0.25, 0.75]),)


def analytic_reference(values, polarity):
    return _suite_weight(polarity) @ values[0] / 4


def known_answer_case(polarity):
    values = torch.ones(4)
    return (
        (values,),
        analytic_reference((values,), polarity),
        2.0 / (256 ** 0.5),
    )


def _all_kernel_specific_checks():
    _kernel_specific_checks()
    _kernel_specific_perf()


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 2.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [6],
    'timesteps': 256,
    'extra_checks': _all_kernel_specific_checks,
}


def test_linear_gaines2():
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear_gaines2()

import torch
import torch.nn.functional as F

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import encoder, decoder
from napl.sim.module.conv_ugemm import conv_ugemm
from napl.sim.module.conv import conv


class napl_conv_ugemm(napl_base):
    def __init__(self, codec_config, weight, bias, stride, padding, conv_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.conv = conv_ugemm(weight, bias, stride=stride, padding=padding, config=conv_config)


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        self.decoder(self.conv(self.encoder(input_x)))


    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.encoder.reset()
        self.conv.reset()
        self.decoder.reset()


def _kernel_specific_checks():
    """
    The streaming uGEMM-CSG conv reproduces (conv2d(x, W) + b) / (in*kh*kw + has_bias)
    within a stochastic-computing bound, for both polarities, on every device.
    padding=1 exercises the alternating 0/1 bipolar zero-pad stream.
    UnarySim reference: FSUConv2duGEMM (scaled=True).
    """
    ntype = global_config.ntype
    timestep = 256
    b, ic, oc, hw, k = 2, 3, 4, 8, 3

    for device in devices():
        for polarity in ['unipolar', 'bipolar']:
            for has_bias in [True, False]:
                for pad in [0, 1]:
                    x = gen_rand_tensor(polarity, (b, ic, hw, hw), 8).type(ntype).to(device)
                    weight = gen_rand_tensor(polarity, (oc, ic, k, k), 8).type(ntype).to(device)
                    bias = gen_rand_tensor(polarity, (oc,), 8).type(ntype).to(device) if has_bias else None

                    codec = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 2}
                    conv_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'width': 12}

                    inst = napl_conv_ugemm(codec, weight, bias, 1, pad, conv_config).to(device)
                    inst(x, timesteps=timestep)

                    entry = ic * k * k + (1 if has_bias else 0)
                    ref = F.conv2d(x, weight, bias, stride=1, padding=pad) / entry
                    err = (inst.decoder.spike_value - ref).abs()
                    rmse = err.pow(2).mean().sqrt().item()
                    assert inst.decoder.spike_value.shape == ref.shape, (device, polarity, has_bias, pad)
                    assert inst.conv.timestep_cur == timestep
                    assert rmse < 0.05, \
                        f'{device}/{polarity}/bias={has_bias}/pad={pad}: rmse {rmse} too large'
                    print(f'[{device}] {polarity} bias={has_bias} pad={pad}: rmse={rmse:.4f} '
                          f'max_err={err.max().item():.4f}')
                    inst.reset()

    # All-ones unipolar operands make the scaled adder emit 1 every timestep.
    for device in devices():
        w1 = torch.ones(oc, ic, k, k).type(ntype).to(device)
        x1 = torch.ones(1, ic, k, k).type(ntype).to(device)
        ugemm = conv_ugemm(
            w1,
            None,
            stride=1,
            padding=0,
            config={
                'polarity': 'unipolar',
                'timestep': 64,
                'generator': 'sobol',
            },
        ).to(device)
        for _ in range(64):
            spike = ugemm(torch.ones_like(x1))
            assert torch.equal(spike, torch.ones_like(spike)), \
                'all-ones unipolar uGEMM conv should spike every timestep'
        ugemm.reset()
        print(f'[{device}] known-answer corner passed.')

    # Compare against the free-running encoder variant on identical spikes.
    perf_x = gen_rand_tensor('bipolar', (b, ic, hw, hw), 8).type(ntype)
    perf_weight = gen_rand_tensor('bipolar', (oc, ic, k, k), 8).type(ntype)
    perf_bias = gen_rand_tensor('bipolar', (oc,), 8).type(ntype)
    for device in devices():
        x = perf_x.to(device)
        weight = perf_weight.to(device)
        bias = perf_bias.to(device)
        enc = encoder({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2}).to(device)
        ugemm = conv_ugemm(weight, bias, stride=1, padding=0,
                               config={'polarity': 'bipolar', 'timestep': timestep,
                                       'generator': 'sobol', 'width': 12}).to(device)
        full = conv(weight, bias, stride=1, padding=0,
                        config={'polarity': 'bipolar', 'timestep': timestep,
                                'generator': 'sobol', 'dim': 3, 'width': 12}).to(device)

        spikes = [enc(x).clone() for _ in range(timestep)]
        enc.reset()
        with timer(device) as t_ugemm:
            for spike in spikes:
                ugemm(spike)
        ugemm.reset()
        with timer(device) as t_full:
            for spike in spikes:
                full(spike)
        full.reset()
        print(f'[{device}] perf: conv_ugemm {t_ugemm.seconds*1e3:.1f}ms vs conv {t_full.seconds*1e3:.1f}ms '
              f'(ratio {t_full.seconds/max(t_ugemm.seconds,1e-9):.2f}x)')

    print('Test passed.')


def make_operation(polarity, timestep, _device):
    weight = torch.tensor([[[[0.5]]]], dtype=global_config.ntype)
    return conv_ugemm(
        weight, None, stride=1, padding=0,
        config={
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'width': 12,
        },
    )


def make_values(polarity):
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(low, high, 16).reshape(1, 1, 4, 4),)


def make_performance_values(polarity):
    values = make_values(polarity)[0]
    return (values.repeat(8192, 1, 1, 1),)


def analytic_reference(values, _polarity):
    weight = torch.tensor([[[[0.5]]]], dtype=values[0].dtype)
    return F.conv2d(values[0], weight)


def known_answer_case(_polarity):
    values = torch.ones(1, 1, 2, 2)
    return (values,), torch.full_like(values, 0.5), 3.0 / (256 ** 0.5)


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_conv_ugemm():
    """Verify conv_ugemm for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_conv_ugemm()

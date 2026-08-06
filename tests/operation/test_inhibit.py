import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, inhibit
from napl.sim.metric import accuracy


def _inhibit_reference(data, inhibiting, polarity):
    # An inhibited stream never falls; its all-ones output decodes to the
    # maximum representable value (race-logic infinity under napl's
    # larger-value-falls-later temporal code).
    ceiling = 1.0
    return torch.where(data <= inhibiting, data, torch.full_like(data, ceiling))


class napl_inhibit(napl_base):
    def __init__(self, codec_config1, codec_config2, inhibit_config):
        super().__init__()
        self.encoder0 = encode(codec_config1)
        self.encoder1 = encode(codec_config2)
        self.decoder = decode(codec_config1)
        self.inhibit = inhibit(inhibit_config)
        self.accuracy = accuracy({'polarity': codec_config1['polarity']})


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike = self.inhibit(i_spike0, i_spike1)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def _kernel_specific_checks():
    """
    Test inhibit with a simple configuration.
    """

    codec_config1={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'temporal',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'temporal',
        'dim': 2,
    }
    inhibit_config={'polarity': codec_config1['polarity']}

    input_0_cpu = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(codec_config1['timestep'])).type(global_config.ntype)
    input_1_cpu = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(codec_config2['timestep'])).type(global_config.ntype)

    for device in devices():
        input_0 = input_0_cpu.to(device)
        input_1 = input_1_cpu.to(device)
        inhibit_inst = napl_inhibit(codec_config1, codec_config2, inhibit_config).to(device)
        with timer(device) as elapsed:
            inhibit_inst(input_0, input_1, timesteps=codec_config1['timestep'])

        r_value = _inhibit_reference(input_0, input_1, codec_config1['polarity'])
        error, _ = inhibit_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt()
        assert rmse <= CONFIG['tolerance_scale'] / math.sqrt(codec_config1['timestep']), rmse
        assert inhibit_inst.inhibit.timestep_cur == codec_config1['timestep']
        inhibit_inst.reset()
        assert inhibit_inst.inhibit.timestep_cur == 0
        print(f'[{device}] time: {elapsed.seconds * 1000:.1f} ms')

    print('Test passed.')


def make_operation(polarity, timestep, _device):
    return inhibit({
        'polarity': polarity,
    })


def make_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 128)
    return left, left.roll(31)


def make_performance_values(polarity):
    lo, hi = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    left = torch.linspace(lo, hi, 131072)
    return left, left.roll(31)


def analytic_reference(values, polarity):
    return _inhibit_reference(values[0], values[1], polarity)


def known_answer_case(polarity):
    if polarity == 'unipolar':
        # Data passes when its edge arrives no later than the inhibitor's
        # (0.25 <= 0.75 and the 0.5 tie); a strictly earlier inhibitor
        # (0.25 < 0.75) forces the output to never fall.
        values = (torch.tensor([0.25, 0.75, 0.5]), torch.tensor([0.75, 0.25, 0.5]))
        expected = torch.tensor([0.25, 1.0, 0.5])
    else:
        values = (torch.tensor([-0.5, 0.5, 0.0]), torch.tensor([0.5, -0.5, 0.0]))
        expected = torch.tensor([-0.5, 1.0, 0.0])
    return values, expected, 0.0


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 3.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_generators': ['temporal', 'temporal'],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_inhibit():
    """Verify inhibit for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_inhibit()

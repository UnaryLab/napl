import math
import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import signabs
from napl.sim.metric import accuracy


class napl_signabs(napl_base):
    def __init__(self, codec_config, signabs_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder_sign = decoder(codec_config)
        self.decoder_abs = decoder(codec_config)
        self.signabs = signabs(signabs_config)
        self.accuracy_sign = accuracy(codec_config)
        self.accuracy_abs = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike_sign, o_spike_abs = self.signabs(i_spike)
        self.decoder_sign(o_spike_sign)
        self.decoder_abs(o_spike_abs)
        self.accuracy_sign(o_spike_sign)
        self.accuracy_abs(o_spike_abs)


def _kernel_specific_checks():
    """
    Test signabs with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    signabs_config={
        'width': 3
    }

    input_cpu = gen_rand_tensor(codec_config['polarity'], shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        signabs_inst = napl_signabs(codec_config, signabs_config).to(device)
        sync(device)
        start = time.perf_counter()
        signabs_inst(input, timesteps=codec_config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        r_value_sign = -torch.sign(input)
        r_value_abs = torch.abs(input)
        signabs_inst.accuracy_sign.analyze(r_value_sign, verbose=True)
        signabs_inst.accuracy_abs.analyze(r_value_abs, verbose=True)
        assert signabs_inst.signabs.timestep_cur == codec_config['timestep']
        signabs_inst.reset()
        assert signabs_inst.signabs.timestep_cur == 0
        print(f'[{device}] time: {elapsed * 1000:.1f} ms')

    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return signabs({'width': 3})


def make_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 128),)


def analytic_reference(values, _polarity):
    return values[0].abs()


def known_answer_case(_polarity):
    values = torch.tensor([-1.0, -0.5, 0.5, 1.0])
    return (values,), values.abs(), 0.2


CONFIG = {
    'polarities': ['bipolar'],
    'tolerance_scale': 4.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'apply_operation': lambda operation, spikes: operation(*spikes)[1],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_signabs():
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_signabs()

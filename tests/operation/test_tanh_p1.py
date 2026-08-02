import torch
import math
import time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
# tanh_p1 is not exported from operation/__init__.py.
from napl.sim.operation.tanh_p1 import tanh_p1
from napl.sim.metric import accuracy


class napl_tanh_p1(napl_base):
    def __init__(self, codec_config, tanh_p1_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.tanh_p1 = tanh_p1(tanh_p1_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.tanh_p1(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def _kernel_specific_checks():
    """
    Test tanh_p1 (combinational series-expansion tanh(x), unipolar) on every
    available device, checking accuracy against torch.tanh and runtime.
    """
    codec_config = {
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 5,
    }
    # Dimension 5 decorrelates input from constants on dimensions 1..4.
    tanh_p1_config = {
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    timestep = codec_config['timestep']

    # Generate once on CPU for identical inputs across devices.
    input_cpu = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(timestep)).type(global_config.ntype)
    r_value_cpu = torch.tanh(input_cpu)

    for device in devices():
        input = input_cpu.to(device)
        r_value = r_value_cpu.to(device)

        tanh_p1_inst = napl_tanh_p1(codec_config, tanh_p1_config).to(device)

        sync(device)
        start = time.perf_counter()
        tanh_p1_inst(input, timesteps=timestep)
        sync(device)
        elapsed = time.perf_counter() - start

        tanh_p1_inst.accuracy.analyze(r_value, verbose=True)
        rmse = torch.sqrt(torch.mean((tanh_p1_inst.decoder.spike_value - r_value)**2)).item()
        # The bound includes series truncation and DFF-decorrelated SC noise.
        assert rmse < 0.1, f'[{device}] rmse={rmse:.4f} exceeds bound 0.1'

        # Include the exact tanh(0)=0 case.
        zero_inst = napl_tanh_p1(codec_config, tanh_p1_config).to(device)
        zero_inst(torch.zeros(16, device=device), timesteps=timestep)
        assert zero_inst.decoder.spike_value.abs().max().item() < 0.1, f'[{device}] tanh(0) != 0'

        assert tanh_p1_inst.tanh_p1.timestep_cur == timestep
        tanh_p1_inst.reset()
        assert tanh_p1_inst.tanh_p1.timestep_cur == 0

        print(f'[{device}] rmse={rmse:.4f}, {timestep} timesteps x 10000 elems in {elapsed*1000:.1f} ms')

    print('Test passed.')


def make_operation(polarity, timestep, _device):
    return tanh_p1({
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 1,
    })


def make_values(_polarity):
    return (torch.linspace(0.0, 1.0, 128),)


def analytic_reference(values, _polarity):
    return torch.tanh(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([0.0, 0.5, 1.0])
    return (values,), torch.tanh(values), 0.1


CONFIG = {
    'polarities': ['unipolar'],
    'tolerance_scale': 1.6,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [5],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_tanh_p1():
    """Verify tanh_p1 against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_tanh_p1()

import torch
import math

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import encode, decode
from napl.sim.operation import exp_n2g
from napl.sim.metric import accuracy


class napl_exp_n2g(napl_base):
    def __init__(self, codec_config_in, codec_config_out, exp_n2g_config):
        super().__init__()
        self.encoder = encode(codec_config_in)
        self.decoder = decode(codec_config_out)
        self.accuracy = accuracy({'polarity': codec_config_out['polarity']})
        self.exp_n2g = exp_n2g(exp_n2g_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.exp_n2g(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def _kernel_specific_checks():
    """
    Test exp_n2g (FSM exp(-2*gain*x), bipolar in / unipolar out) on every device.
    """

    codec_config_in={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config_out={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    exp_n2g_config={
        'depth': 5,
        'gain': 1,
    }

    # The FSM accepts non-negative x for exp(-2x); CPU generation reuses inputs across devices.
    input_cpu = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config_in['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)

        exp_n2g_inst = napl_exp_n2g(codec_config_in, codec_config_out, exp_n2g_config).to(device)

        with timer(device) as elapsed:
            exp_n2g_inst(input, timesteps=codec_config_in['timestep'])

        r_value = torch.exp(input * (-2 * exp_n2g_config['gain']))

        exp_n2g_inst.accuracy.analyze(r_value, verbose=True)

        # The fidelity bound includes FSM approximation error and SC noise.
        err = (exp_n2g_inst.decoder.spike_value - r_value).abs()
        assert err.mean() < 0.05, f'[{device}] mean abs error {err.mean():.4f} exceeds bound'

        assert exp_n2g_inst.exp_n2g.timestep_cur == codec_config_in['timestep']
        exp_n2g_inst.reset()
        assert exp_n2g_inst.exp_n2g.timestep_cur == 0

        print(f'[{device}] Test passed in {elapsed.seconds:.3f} s.')


def make_operation(_polarity, _timestep, _device):
    return exp_n2g({'depth': 5, 'gain': 1})


def make_values(_polarity):
    return (torch.linspace(0.0, 1.0, 128),)


def make_performance_values(_polarity):
    return (torch.linspace(0.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return torch.exp(-2 * values[0])


def known_answer_case(_polarity):
    values = torch.tensor([0.0, 0.5, 1.0])
    return (values,), torch.exp(-2 * values), 0.1


CONFIG = {
    'polarities': ['bipolar'],
    'tolerance_scale': 1.6,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'input_polarities': ['bipolar'],
    'output_polarity': 'unipolar',
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_exp_n2g():
    """Verify exp_n2g against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_exp_n2g()

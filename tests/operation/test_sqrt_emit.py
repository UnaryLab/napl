import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, sqrt_emit
from napl.sim.metric import accuracy


class napl_sqrt_emit(napl_base):
    def __init__(self, codec_config, sqrt_emit_config):
        super().__init__()
        self.encoder = encode(codec_config)
        self.decoder = decode(codec_config)
        self.sqrt_emit = sqrt_emit(sqrt_emit_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.sqrt_emit(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test sqrt_emit with a simple configuration.
    """

    codec_config={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 4,
    }
    sqrt_emit_config={
        'polarity': 'unipolar',
    }
    
    input_cpu = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        sqrt_emit_inst = napl_sqrt_emit(codec_config, sqrt_emit_config).to(device)
        with timer(device) as elapsed:
            sqrt_emit_inst(input, timesteps=codec_config['timestep'])

        r_value = torch.sqrt(input)
        sqrt_emit_inst.accuracy.analyze(r_value, verbose=True)
        assert sqrt_emit_inst.sqrt_emit.timestep_cur == codec_config['timestep']
        sqrt_emit_inst.reset()
        assert sqrt_emit_inst.sqrt_emit.timestep_cur == 0
        print(f'[{device}] time: {elapsed.seconds * 1000:.1f} ms')

    print('Test passed.')


# Values the 0-dim check drives and the output spike counts they place over a
# RANK0_TIMESTEPS run. A 0-dim input carries no lane to place a value grid on, so
# the check reads one value at a time.
RANK0_TIMESTEPS = 256
RANK0_VALUES = (0.25, 0.5625)
RANK0_COUNT = {'unipolar': (137, 188), 'bipolar': (185, 227)}


def _rank0_checks():
    """Assert a 0-dim input is carried at rank 1 and places the pinned output spike count."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            for index, value in enumerate(RANK0_VALUES):
                operation = sqrt_emit({'polarity': polarity}).to(device)
                encoder = encode({
                    'polarity': polarity, 'timestep': RANK0_TIMESTEPS,
                    'generator': 'sobol', 'dim': CONFIG['encoder_dims'][0],
                }).to(device)
                input = torch.tensor(value, device=device)
                outputs = torch.stack([
                    operation(encoder(input)) for _ in range(RANK0_TIMESTEPS)
                ])
                assert outputs.shape == (RANK0_TIMESTEPS, 1), (
                    f'{polarity} sqrt_emit emitted per-timestep shape '
                    f'{tuple(outputs.shape[1:])} on a 0-dim input on {device}, not the '
                    f'rank-1 shape the emission feedback path leaves it at'
                )
                count = int(outputs.sum())
                assert count == RANK0_COUNT[polarity][index], (
                    f'{polarity} sqrt_emit counted {count} spikes on a 0-dim input '
                    f'{value} on {device}, against the pinned '
                    f'{RANK0_COUNT[polarity][index]}'
                )
    print('a 0-dim input is carried at rank 1 and keeps the pinned spike count.')


def make_operation(polarity, _timestep, _device):
    return sqrt_emit({'polarity': polarity})


def make_values(_polarity):
    return (torch.linspace(0.0, 1.0, 128),)


def make_random_perf_values(_polarity):
    return (torch.linspace(0.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return torch.sqrt(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([0.0, 0.25, 1.0])
    return (values,), torch.sqrt(values)


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [4],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_sqrt_emit():
    """Verify sqrt_emit for both polarities on the non-negative square-root domain and pin the 0-dim input rank."""
    streaming_suite(CONFIG)
    _rank0_checks()


if __name__ == '__main__':
    test_sqrt_emit()

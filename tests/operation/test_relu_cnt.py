import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, relu_cnt
from napl.sim.metric import accuracy


class napl_relu_cnt(napl_base):
    def __init__(self, codec_config, relu_cnt_config):
        super().__init__()
        self.encoder = encode(codec_config)
        self.decoder = decode(codec_config)
        self.relu_cnt = relu_cnt(relu_cnt_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.relu_cnt(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

    
def _kernel_specific_checks():
    """
    Test relu_cnt with a simple configuration.
    """

    codec_config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    relu_cnt_config={
        'width': 3,
    }
    
    input_cpu = gen_rand_tensor('bipolar', shape=(10000,), width=math.log2(codec_config['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)
        relu_cnt_inst = napl_relu_cnt(codec_config, relu_cnt_config).to(device)
        with timer(device) as elapsed:
            relu_cnt_inst(input, timesteps=codec_config['timestep'])

        r_value = torch.nn.ReLU()(input)
        error, _ = relu_cnt_inst.accuracy.analyze(r_value, verbose=True)
        rmse = error.pow(2).mean().sqrt()
        assert rmse <= CONFIG['tolerance_scale'] / math.sqrt(codec_config['timestep']), rmse
        assert relu_cnt_inst.relu_cnt.timestep_cur == codec_config['timestep']
        relu_cnt_inst.reset()
        assert relu_cnt_inst.relu_cnt.timestep_cur == 0
        print(f'[{device}] time: {elapsed.seconds * 1000:.1f} ms')

    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return relu_cnt({'width': 3})


def make_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 128),)


def make_performance_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 131072),)


def analytic_reference(values, _polarity):
    return torch.relu(values[0])


def known_answer_case(_polarity):
    values = torch.tensor([-1.0, 0.0, 1.0])
    return (values,), torch.tensor([0.0, 0.0, 1.0]), 0.35


CONFIG = {
    'polarities': ['bipolar'],
    'tolerance_scale': 5.5,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_relu_cnt():
    """Verify relu_cnt against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_relu_cnt()

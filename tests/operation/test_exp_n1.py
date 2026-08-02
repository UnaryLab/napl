import torch
import math
import time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation.exp_n1 import exp_n1
from napl.sim.metric import accuracy


class napl_exp_n1(napl_base):
    def __init__(self, codec_config, exp_n1_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})
        self.exp_n1 = exp_n1(exp_n1_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.exp_n1(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def _kernel_specific_checks():
    """
    Test exp_n1 (exp(-x), unipolar) on every available device: correctness
    against torch.exp(-x) within the SC bound, plus per-device runtime.
    """
    timestep = 256

    codec_config={
        'polarity': 'unipolar',
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 5,  # Distinct from the operation's internal dimensions 1..4.
    }
    exp_n1_config={
        'polarity': 'unipolar',
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 1,
    }

    # Generate once on CPU for identical inputs across devices.
    input_cpu = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(timestep)).type(global_config.ntype)
    # Include exp(0)=1 and exp(-1) known-answer cases.
    input_cpu[0] = 0.0
    input_cpu[1] = 1.0

    for device in devices():
        input = input_cpu.to(device)

        exp_n1_inst = napl_exp_n1(codec_config, exp_n1_config).to(device)
        sync(device)
        start = time.perf_counter()
        exp_n1_inst(input, timesteps=timestep)
        sync(device)
        elapsed = time.perf_counter() - start

        r_value = torch.exp(-input)

        exp_n1_inst.accuracy.analyze(r_value, verbose=True)

        out = exp_n1_inst.decoder.spike_value.cpu()
        ref = r_value.cpu()
        rmse = (out - ref).pow(2).mean().sqrt().item()
        bound = 1.5 / math.sqrt(timestep)  # Series truncation stays below 0.002.
        assert rmse < bound, f'[{device}] RMSE {rmse:.4f} exceeds bound {bound:.4f}'
        # Known-answer cases use the same SC bound.
        assert abs(out[0].item() - 1.0) < bound
        assert abs(out[1].item() - math.exp(-1)) < bound

        assert exp_n1_inst.exp_n1.timestep_cur == timestep
        exp_n1_inst.reset()
        assert exp_n1_inst.exp_n1.timestep_cur == 0

        # Compare the streaming kernel with the single-shot float reference.
        sync(device)
        start_ref = time.perf_counter()
        torch.exp(-input)
        sync(device)
        elapsed_ref = time.perf_counter() - start_ref
        print(f'[{device}] rmse={rmse:.4f} (bound {bound:.4f}), '
              f'kernel {elapsed*1e3:.1f} ms for {timestep} timesteps, '
              f'torch.exp {elapsed_ref*1e3:.3f} ms (ratio {elapsed/max(elapsed_ref, 1e-9):.0f}x)')

    print('Test passed.')


def make_operation(polarity, timestep, _device):
    return exp_n1({
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 1,
    })


def make_values(_polarity):
    return (torch.linspace(0.0, 1.0, 128),)


def analytic_reference(values, _polarity):
    return torch.exp(-values[0])


def known_answer_case(_polarity):
    values = torch.tensor([0.0, 1.0])
    return (values,), torch.exp(-values), 1.5 / math.sqrt(256)


CONFIG = {
    'polarities': ['unipolar'],
    'tolerance_scale': 1.5,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [5],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_exp_n1():
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_exp_n1()

"""
Per-device (cpu/cuda/mps) test of the accuracy metric on the canonical mul_csg
round-trip: the progressive error vs the analytic product stays within the SC
bound, the metric agrees with the decoder and the free analyze_error, and a
reset re-run reproduces the identical result. Timing is absolute (no baseline).
"""
import math
import time

import torch

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.metric import accuracy, analyze_error
from napl.module import decoder, encoder
from napl.operation import mul_csg
from napl.utils import devices, gen_rand_tensor, sync


class napl_mul_csg(napl_base):
    def __init__(self, codec_config, mul_csg_config, acc_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.mul_csg = mul_csg(mul_csg_config)
        self.accuracy = accuracy(acc_config)

    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike = self.encoder(input_0)
        o_spike = self.mul_csg(i_spike, input_1)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def run_accuracy(input_0, input_1, device, timestep, model=None):
    codec_config = {
        'polarity': 'bipolar',
        'timestep': timestep,
        'generator': 'sobol',
    }
    acc_config = {
        'polarity': 'bipolar',
        'name': 'mul_csg_inst',
    }
    i_0 = input_0.to(device)
    i_1 = input_1.to(device)
    if model is None:
        model = napl_mul_csg(codec_config, codec_config, acc_config).to(device)
    reference = i_0 * i_1

    sync(device)
    start = time.time()
    model(i_0, i_1, timesteps=timestep)
    spike_error, _ = model.accuracy.analyze(reference)
    sync(device)
    elapsed = time.time() - start

    return (
        model,
        reference,
        spike_error.detach().cpu().clone(),
        model.accuracy.spike_value.detach().cpu().clone(),
        elapsed,
    )


def make_inputs(timestep):
    input_0 = gen_rand_tensor(
        'bipolar', shape=(10000,), width=math.log2(timestep)
    ).type(global_config.ntype)
    input_1 = gen_rand_tensor(
        'bipolar', shape=(10000,), width=math.log2(timestep)
    ).type(global_config.ntype)
    return input_0, input_1


def test_fidelity():
    timestep = 256
    input_0, input_1 = make_inputs(timestep)

    for device in devices():
        model, reference, error, _, _ = run_accuracy(
            input_0, input_1, device, timestep
        )
        print(f'[{device}]')
        model.accuracy.analyze(reference, verbose=True)
        analyze_error(model.decoder.spike_value, reference)
        assert torch.equal(
            model.accuracy.spike_value, model.decoder.spike_value
        ), 'accuracy.spike_value != decoder.spike_value'
        assert error.abs().max() <= 1.0 / math.sqrt(timestep), error.abs().max()
        print(
            f'[{device}] accuracy mae={error.abs().mean().item():.4f}, '
            f'max={error.abs().max().item():.4f}'
        )


def test_reset():
    timestep = 256
    input_0, input_1 = make_inputs(timestep)

    for device in devices():
        model, _, _, first_value, _ = run_accuracy(
            input_0, input_1, device, timestep
        )
        model.reset()
        assert not model.accuracy.valid
        _, _, _, second_value, _ = run_accuracy(
            input_0, input_1, device, timestep, model
        )
        assert torch.equal(first_value, second_value), 'reset re-run diverged'


def test_performance():
    timestep = 256
    input_0, input_1 = make_inputs(timestep)

    for device in devices():
        model, _, _, _, _ = run_accuracy(input_0, input_1, device, timestep)
        model.reset()
        _, _, _, _, elapsed = run_accuracy(
            input_0, input_1, device, timestep, model
        )
        print(f'[{device}] time={elapsed * 1000:.1f}ms')


if __name__ == '__main__':
    test_fidelity()
    test_reset()
    test_performance()
    print('Test passed.')

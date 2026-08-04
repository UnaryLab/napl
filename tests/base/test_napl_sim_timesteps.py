import math

import torch
import napl

from napl.sim.base import napl_base, napl_sim_timesteps
from napl.sim.operation import encode, decode
from napl.sim.metric import accuracy
from napl.utils._shared_test import devices, timer


class codec(napl_base):
    def __init__(self, config):
        super().__init__()
        self.encoder = encode(config)
        self.decoder = decode(config)
        self.accuracy = accuracy(config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        spike = self.encoder(input)
        self.decoder(spike)
        self.accuracy(spike)
        # The composite ticks once per call; inner modules tick once per iteration.
        assert self.encoder.timestep_cur == self.decoder.timestep_cur, \
            f'Timestep mismatch: {self.encoder.timestep_cur}, {self.decoder.timestep_cur}.'


class reset_leaf(napl_base):
    def __init__(self):
        super().__init__()
        self.reset_count = 0


    def _reset(self):
        self.reset_count += 1


    def forward(self, input):
        return input


class reset_parent(napl_base):
    def __init__(self):
        super().__init__()
        self.child = reset_leaf()
        self.reset_count = 0


    def _reset(self):
        self.reset_count += 1


    def forward(self, input):
        return self.child(input)


def test_napl_sim_timesteps_class():
    """Verify the decorator repeats a streaming module for the requested timesteps."""
    config = {
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }

    input_cpu = torch.tensor([0.1, 0.5, 0.9])

    for device in devices():
        input = input_cpu.to(device)
        codec_inst = codec(config).to(device)
        with timer(device) as elapsed:
            codec_inst(input, timesteps=config['timestep'])

        error, _ = codec_inst.accuracy.analyze(input, verbose=True)
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(config['timestep'])
        assert codec_inst.encoder.timestep_cur == config['timestep']
        assert codec_inst.decoder.timestep_cur == config['timestep']
        print(f'[{device}] time={elapsed.seconds * 1000:.1f}ms')

        codec_inst.reset()
        assert codec_inst.timestep_cur == 0
        assert codec_inst.encoder.timestep_cur == 0
        assert codec_inst.decoder.timestep_cur == 0
        assert codec_inst.accuracy.timestep_cur == 0

    print('Test passed.')


def test_napl_sim_timesteps_class_rank2():
    """Verify decorated streaming modules preserve rank-two shapes and decoding accuracy."""
    config = {
        'polarity': 'bipolar',
        'timestep': 16,
        'generator': 'sobol',
    }
    input_cpu = torch.tensor([[0.1, 0.5], [0.9, -0.3]])

    for device in devices():
        input = input_cpu.to(device)
        codec_inst = codec(config).to(device)
        codec_inst(input, timesteps=config['timestep'])

        error, _ = codec_inst.accuracy.analyze(input)
        assert error.shape == input.shape
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(
            config['timestep']
        )


def test_napl_sim_timesteps_free_function_positional():
    """Verify the decorator repeats free functions with positional arguments."""
    calls = []

    @napl_sim_timesteps
    def run(value):
        calls.append(value)
        return len(calls)

    assert run('spike', timesteps=3) == 3
    assert calls == ['spike'] * 3


def test_napl_sim_timesteps_free_function_keyword_only():
    """Verify the decorator repeats free functions with keyword-only arguments."""
    calls = []

    @napl_sim_timesteps
    def run(*, value):
        calls.append(value)
        return len(calls)

    assert run(value='spike', timesteps=2) == 2
    assert calls == ['spike'] * 2


def test_reset_lifecycle():
    """Verify reset clears parent and child timestep state and invokes each reset hook."""
    for shape in ((1,), (2, 3)):
        module = reset_parent()
        input = torch.ones(shape)
        assert torch.equal(module(input), input)
        assert module.timestep_cur == 1
        assert module.child.timestep_cur == 1

        module.reset(verbose=True)
        assert module.timestep_cur == 0
        assert module.child.timestep_cur == 0
        assert module.reset_count == 1
        assert module.child.reset_count == 1


def test_reset_hook_format():
    """Verify every concrete NAPL simulation class defines its own reset hook."""
    classes = {
        value
        for value in vars(napl).values()
        if isinstance(value, type)
        and issubclass(value, napl_base)
        and value is not napl_base
        and value.__module__.startswith('napl.sim.')
    }
    missing = sorted(cls.__name__ for cls in classes if '_reset' not in cls.__dict__)
    assert missing == []


if __name__ == '__main__':
    test_napl_sim_timesteps_class()
    test_napl_sim_timesteps_class_rank2()
    test_napl_sim_timesteps_free_function_positional()
    test_napl_sim_timesteps_free_function_keyword_only()
    test_reset_lifecycle()
    test_reset_hook_format()

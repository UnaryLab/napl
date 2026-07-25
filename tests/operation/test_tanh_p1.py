import torch
import math
import time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
# direct import: not wired into operation/__init__.py yet
from napl.sim.operation.tanh_p1 import tanh_p1
from napl.sim.metric import accuracy


class napl_tanh_p1(napl_base):
    def __init__(self, codec_config, tanh_p1_config):
        super().__init__()
        # set up encoder, decoder, op, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.tanh_p1 = tanh_p1(tanh_p1_config)
        self.accuracy = accuracy(codec_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.tanh_p1(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def test_tanh_p1():
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
    # constants occupy sobol dims 1..4; input stream uses dim 5 to decorrelate
    tanh_p1_config = {
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    timestep = codec_config['timestep']

    # identical inputs on every device
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
        # series truncation + DFF-decorrelated SC noise at 256 timesteps
        assert rmse < 0.1, f'[{device}] rmse={rmse:.4f} exceeds bound 0.1'

        # known-answer corner: x=0 -> tanh(0)=0
        zero_inst = napl_tanh_p1(codec_config, tanh_p1_config).to(device)
        zero_inst(torch.zeros(16, device=device), timesteps=timestep)
        assert zero_inst.decoder.spike_value.abs().max().item() < 0.1, f'[{device}] tanh(0) != 0'

        assert tanh_p1_inst.tanh_p1.timestep_cur == timestep
        tanh_p1_inst.reset()
        assert tanh_p1_inst.tanh_p1.timestep_cur == 0

        print(f'[{device}] rmse={rmse:.4f}, {timestep} timesteps x 10000 elems in {elapsed*1000:.1f} ms')

    print('Test passed.')


if __name__ == '__main__':
    test_tanh_p1()

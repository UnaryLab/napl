import time

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder, linear
from napl.sim.metric import accuracy


class napl_linear(napl_base):
    def __init__(self, codec_config, lin_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.accuracy = accuracy(codec_config)
        self.linear = linear(weight, bias, lin_config)

    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        o_spike = self.linear(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def test_linear():
    """
    Streaming unary linear layer reproduces (W x + b) / (in_features + bias) within a
    stochastic-computing error bound.
    """
    timestep = 256
    in_features, out_features = 16, 8

    # input on rng dim 1; weight/bias on dims 2/3 (decorrelated operands)
    codec_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    lin_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'scale': None, 'width': 12}

    input_cpu = gen_rand_tensor(
        'bipolar', shape=(in_features,), width=8
    ).type(global_config.ntype)
    weight_cpu = gen_rand_tensor(
        'bipolar', shape=(out_features, in_features), width=8
    ).type(global_config.ntype)
    bias_cpu = gen_rand_tensor(
        'bipolar', shape=(out_features,), width=8
    ).type(global_config.ntype)

    for device in devices():
        input_x = input_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        inst = napl_linear(
            codec_config, lin_config, weight, bias
        ).to(device)
        sync(device)
        start = time.perf_counter()
        inst(input_x, timesteps=timestep)
        sync(device)
        elapsed = time.perf_counter() - start

        entry = in_features + 1
        r_value = (weight @ input_x + bias) / entry
        err, _ = inst.accuracy.analyze(r_value, verbose=True)
        rmse = torch.sqrt(err.abs().pow(2).mean())
        print(
            f'[{device}] linear rmse={rmse.item():.4f}, '
            f'max={err.abs().max().item():.4f}, time={elapsed * 1000:.1f}ms'
        )
        assert rmse < 0.05, (device, rmse)
        assert inst.linear.timestep_cur == timestep
        inst.reset()
        assert inst.linear.timestep_cur == 0
        assert inst.accuracy.timestep_cur == 0

    print('Test passed.')


if __name__ == '__main__':
    test_linear()

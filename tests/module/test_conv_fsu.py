import time

import torch.nn.functional as F

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import devices, gen_rand_tensor, sync
from napl.module import encoder, decoder, conv_fsu


class napl_conv_fsu(napl_base):
    def __init__(self, codec_config, weight, bias, stride, padding, fsu_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.conv = conv_fsu(weight, bias, stride=stride, padding=padding, config=fsu_config)

    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        self.decoder(self.conv(self.encoder(input_x)))


def test_conv_fsu():
    """
    Streaming unary conv reproduces (conv2d(x, W) + b) / (in*kh*kw + bias) within a
    stochastic-computing bound. padding=1 exercises the decorrelated rate-0.5 pad stream,
    which must not degrade the border (a plain 0-pad would inject bipolar -1).
    """
    ntype = global_config.ntype
    timestep = 256
    b, ic, oc, hw, k = 2, 3, 4, 8, 3
    x_cpu = gen_rand_tensor('bipolar', (b, ic, hw, hw), 8).type(ntype)
    weight_cpu = gen_rand_tensor('bipolar', (oc, ic, k, k), 8).type(ntype)
    bias_cpu = gen_rand_tensor('bipolar', (oc,), 8).type(ntype)
    entry = ic * k * k + 1

    codec = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    fsu = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'width': 12}

    for device in devices():
        x = x_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        for pad in [0, 1]:
            inst = napl_conv_fsu(codec, weight, bias, 1, pad, fsu).to(device)
            sync(device)
            start = time.perf_counter()
            inst(x, timesteps=timestep)
            sync(device)
            elapsed = time.perf_counter() - start
            ref = F.conv2d(x, weight, bias, stride=1, padding=pad) / entry
            rmse = (inst.decoder.spike_value - ref).pow(2).mean().sqrt().item()
            print(
                f'[{device}] pad={pad} conv_fsu rmse={rmse:.4f}, '
                f'time={elapsed * 1000:.1f}ms'
            )
            assert inst.decoder.spike_value.shape == ref.shape
            assert rmse < 0.03, (device, pad, rmse)
            assert inst.conv.timestep_cur == timestep
            inst.reset()
            assert inst.conv.timestep_cur == 0

    print('Test passed.')


if __name__ == '__main__':
    test_conv_fsu()

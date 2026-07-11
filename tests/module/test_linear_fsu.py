import torch

from napl.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder, linear_fsu
from napl.metric import report_error


class napl_linear_fsu(napl_base):
    def __init__(self, codec_config, lin_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.linear = linear_fsu(weight, bias, lin_config)

    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        o_spike = self.linear(i_spike)
        self.decoder(o_spike)


def test_linear_fsu():
    """
    Streaming unary linear layer reproduces (W x + b) / (in_features + bias) within a
    stochastic-computing error bound.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    timestep = 256
    in_features, out_features = 16, 8

    # input on rng dim 1; weight/bias on dims 2/3 (decorrelated operands)
    codec_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    lin_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'scale': None, 'width': 12}

    input_x = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype).to(device)
    weight = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype).to(device)
    bias = gen_rand_tensor('bipolar', shape=(out_features,), width=8).type(global_config.ntype).to(device)

    inst = napl_linear_fsu(codec_config, lin_config, weight, bias).to(device)
    inst(input_x, timesteps=timestep)

    entry = in_features + 1   # +1 for bias
    r_value = (weight @ input_x + bias) / entry

    err, _ = report_error(inst.decoder.spike_value, r_value)
    rmse = torch.sqrt(err.abs().pow(2).mean())
    print(f'linear_fsu RMSE={rmse.item():.4f}, max abs err={err.abs().max().item():.4f}')

    assert rmse < 0.05, rmse
    assert inst.linear.timestep_cur == timestep
    inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_linear_fsu()

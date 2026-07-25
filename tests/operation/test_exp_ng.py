import torch
import math
import time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
# imported from the module directly: not wired into napl.sim.operation yet
from napl.sim.operation.exp_ng import exp_ng
from napl.sim.metric import accuracy


class napl_exp_ng(napl_base):
    def __init__(self, codec_config_in, codec_config_out, exp_ng_config):
        super().__init__()
        # set up encoder, decoder, op, and accuracy
        self.encoder = encoder(codec_config_in)
        self.decoder = decoder(codec_config_out)
        self.accuracy = accuracy({'polarity': codec_config_out['polarity']})
        self.exp_ng = exp_ng(exp_ng_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.exp_ng(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def test_exp_ng():
    """
    Test exp_ng (FSM exp(-2*gain*x), bipolar in / unipolar out) on every device.
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
    exp_ng_config={
        'depth': 5,
        'gain': 1,
    }

    # Generate random inputs; the FSM approximates exp(-2x) for non-negative inputs,
    # so draw from [0, 1) (encoded on the bipolar codec). Same inputs on every device.
    input_cpu = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(codec_config_in['timestep'])).type(global_config.ntype)

    for device in devices():
        input = input_cpu.to(device)

        # generate the napl_exp_ng instance
        exp_ng_inst = napl_exp_ng(codec_config_in, codec_config_out, exp_ng_config).to(device)

        sync(device)
        start = time.perf_counter()
        exp_ng_inst(input, timesteps=codec_config_in['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        # calculate the reference output
        r_value = torch.exp(input * (-2 * exp_ng_config['gain']))

        # report the error
        exp_ng_inst.accuracy.analyze(r_value, verbose=True)

        # FSM approximation + SC noise: assert a loose fidelity bound
        err = (exp_ng_inst.decoder.spike_value - r_value).abs()
        assert err.mean() < 0.05, f'[{device}] mean abs error {err.mean():.4f} exceeds bound'

        assert exp_ng_inst.exp_ng.timestep_cur == codec_config_in['timestep']
        exp_ng_inst.reset()
        assert exp_ng_inst.exp_ng.timestep_cur == 0

        print(f'[{device}] Test passed in {elapsed:.3f} s.')


if __name__ == '__main__':
    test_exp_ng()

import math
import time

import torch

from napl.sim.base import napl_sim_timesteps_func
from napl.sim.module import encoder, decoder
from napl.sim.metric import accuracy
from napl.utils._shared_test import devices, sync


def test_napl_sim_timesteps_func():
    """
    Test the napl_sime_timesteps decorator with a simple configuration.
    """
    config={
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }

    input_cpu = torch.tensor([0.1, 0.5, 0.9])

    for device in devices():
        encoder_inst = encoder(config).to(device)
        decoder_inst = decoder(config).to(device)
        accuracy_inst = accuracy(config).to(device)

        @napl_sim_timesteps_func
        def this_run(input, timesteps=256):
            spike = encoder_inst(input)
            decoder_inst(spike)
            accuracy_inst(spike)
            assert (
                encoder_inst.timestep_cur
                == decoder_inst.timestep_cur
                == accuracy_inst.timestep_cur
            )

        input = input_cpu.to(device)
        sync(device)
        start = time.perf_counter()
        this_run(input, timesteps=config['timestep'])
        sync(device)
        elapsed = time.perf_counter() - start

        error, _ = accuracy_inst.analyze(input, verbose=True)
        assert error.pow(2).mean().sqrt() <= 1.0 / math.sqrt(config['timestep'])
        assert encoder_inst.timestep_cur == config['timestep']
        assert decoder_inst.timestep_cur == config['timestep']
        print(f'[{device}] time={elapsed * 1000:.1f}ms')

        encoder_inst.reset()
        decoder_inst.reset()
        accuracy_inst.reset()
        assert encoder_inst.timestep_cur == 0
        assert decoder_inst.timestep_cur == 0
        assert accuracy_inst.timestep_cur == 0

    print('Test passed.')


if __name__ == '__main__':
    test_napl_sim_timesteps_func()

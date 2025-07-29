import torch

from napl.base import napl_sim_timesteps_func
from napl.module import encoder, decoder
from napl.metric import report_error


def test_napl_sim_timesteps_func():
    """
    Test the napl_sime_timesteps decorator with a simple configuration.
    """
    config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }

    encoder_inst = encoder(config)
    decoder_inst = decoder(config)

    @napl_sim_timesteps_func
    def this_run(input, timesteps=256):
        spike = encoder_inst(input)
        decoder_inst(spike)
        # print(f'Timestep {encoder_inst.timestep_cur} processed.')
        assert encoder_inst.timestep_cur == decoder_inst.timestep_cur, \
            f'Timestep mismatch: {encoder_inst.timestep_cur}, {decoder_inst.timestep_cur}.'
    
    input = torch.tensor([0.1, 0.5, 0.9])
    
    this_run(input, timesteps=config['timestep'])

    report_error(decoder_inst.spike_value, input)

    print('Test passed.')


if __name__ == '__main__':
    test_napl_sim_timesteps_func()


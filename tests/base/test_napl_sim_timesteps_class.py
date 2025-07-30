import torch

from napl.base import napl_base, napl_sim_timesteps
from napl.module import encoder, decoder
from napl.metric import report_error


class codec(napl_base):
    def __init__(self, config):
        super().__init__()
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.timestep_cur = 0


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        spike = self.encoder(input)
        self.decoder(spike)
        self.timestep_cur += 1
        # print(f'Timestep {self.timestep_cur} processed.')
        assert self.timestep_cur == self.encoder.timestep_cur == self.decoder.timestep_cur, \
            f'Timestep mismatch: {self.timestep_cur}, {self.encoder.timestep_cur}, {self.decoder.timestep_cur}.'

    
def test_napl_sim_timesteps_class():
    """
    Test the napl_sime_timesteps decorator with a simple configuration.
    """
    config={
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }

    input = torch.tensor([0.1, 0.5, 0.9])

    codec_inst = codec(config)
    codec_inst(input, timesteps=config['timestep'])

    report_error(codec_inst.decoder.spike_value, input)

    codec_inst.reset()
    
    print('Test passed.')


if __name__ == '__main__':
    test_napl_sim_timesteps_class()


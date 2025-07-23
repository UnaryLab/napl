import torch

from napl.base import global_config, napl_base, napl_sim_timesteps_class
from napl.module import encoder, decoder
from napl.utils import check_config


class codec(napl_base):
    def __init__(self, config):
        super().__init__()
        self.encoder = encoder(config)
        self.decoder = decoder(config)
        self.timestep_cur = 0


    @napl_sim_timesteps_class
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

    codec_inst = codec(config)

    input = torch.tensor([0.1, 0.5, 0.9])
    
    codec_inst(input, timesteps=config['timestep'])

    print("Napl sim with multiple timesteps test passed.")


if __name__ == "__main__":
    test_napl_sim_timesteps_class()


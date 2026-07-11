import torch

from napl.base import napl_base, napl_sim_timesteps
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import mul_csg, add_any


class butterfly_spike(napl_base):
    def __init__(
            self,
            codec_config,
            mul_config,
            add_config,
            acc_config  # reserved for per-node accuracy instrumentation (not wired up)
        ):
        super().__init__()

        # butterfly equation
        # y0r = x0r + (wr * x1r - wi * x1i)
        # y0i = x0i + (wr * x1i + wi * x1r)
        # y1r = x0r - (wr * x1r - wi * x1i)
        # y1i = x0i - (wr * x1i + wi * x1r)

        # set up encoder, decoder, mul, and add
        self.encoder_x0r = encoder(codec_config)
        self.encoder_x0i = encoder(codec_config)
        self.encoder_x1r = encoder(codec_config)
        self.encoder_x1i = encoder(codec_config)

        self.decoder_y0r = decoder(codec_config)
        self.decoder_y0i = decoder(codec_config)
        self.decoder_y1r = decoder(codec_config)
        self.decoder_y1i = decoder(codec_config)

        # multiplication operations
        self.mul_wr_x1r = mul_csg(mul_config)
        self.mul_wr_x1i = mul_csg(mul_config)
        self.mul_wi_x1r = mul_csg(mul_config)
        self.mul_wi_x1i = mul_csg(mul_config)

        # addition operations
        self.add_y0r = add_any(add_config)
        self.add_y0i = add_any(add_config)
        self.add_y1r = add_any(add_config)
        self.add_y1i = add_any(add_config)


    @napl_sim_timesteps
    def forward(self, x0r, x0i, x1r, x1i, wr, wi):
        # all inputs and outputs are binary tensors

        # encode x
        x0r_spike = self.encoder_x0r(x0r)
        x0i_spike = self.encoder_x0i(x0i)
        x1r_spike = self.encoder_x1r(x1r)
        x1i_spike = self.encoder_x1i(x1i)

        # w * x
        wr_x1r_spike = self.mul_wr_x1r(x1r_spike, wr)
        wr_x1i_spike = self.mul_wr_x1i(x1i_spike, wr)
        wi_x1r_spike = self.mul_wi_x1r(x1r_spike, wi)
        wi_x1i_spike = self.mul_wi_x1i(x1i_spike, wi)

        # three input add
        # sum the 3 spikes (each in {0,1}) in stype first, then feed via dim=None
        # (entry=3). This avoids materializing a stacked tensor and casts once;
        # the int sum of {0,1,2,3} fits stype and is exact in ntype.
        y0r_spike = self.add_y0r((x0r_spike +     wr_x1r_spike + (1 - wi_x1i_spike)), entry=3, dim=None)
        y0i_spike = self.add_y0i((x0i_spike +     wr_x1i_spike +     wi_x1r_spike),  entry=3, dim=None)
        y1r_spike = self.add_y1r((x0r_spike + (1 - wr_x1r_spike) +     wi_x1i_spike), entry=3, dim=None)
        y1i_spike = self.add_y1i((x0i_spike + (1 - wr_x1i_spike) + (1 - wi_x1r_spike)), entry=3, dim=None)

        # decode y
        y0r = self.decoder_y0r(y0r_spike)
        y0i = self.decoder_y0i(y0i_spike)
        y1r = self.decoder_y1r(y1r_spike)
        y1i = self.decoder_y1i(y1i_spike)

        return y0r, y0i, y1r, y1i


class butterfly_binary(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0r, x0i, x1r, x1i, wr, wi):
        # butterfly equation
        wr_x1r = wr * x1r
        wr_x1i = wr * x1i
        wi_x1r = wi * x1r
        wi_x1i = wi * x1i

        y0r = x0r + (wr_x1r - wi_x1i)
        y0i = x0i + (wr_x1i + wi_x1r)
        y1r = x0r - (wr_x1r - wi_x1i)
        y1i = x0i - (wr_x1i + wi_x1r)

        return y0r, y0i, y1r, y1i


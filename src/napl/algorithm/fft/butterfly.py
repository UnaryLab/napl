import torch, math

from napl.base import napl_base, napl_sim_timesteps_class
from napl.utils import *
from napl.module import encoder, decoder
from napl.operation import mul_csg, add_any
from napl.metric import accuracy


class butterfly_spike(napl_base):
    def __init__(
            self,
            codec_config, 
            mul_config, 
            add_config,
            acc_config
        ):
        super().__init__()

        # butterfly equation
        # y0r = x0r + (wr * x1r - wi * x1i)
        # y0i = x0i + (wr * x1i + wi * x1r)
        # y1r = x0r - (wr * x1r - wi * x1i)
        # y1i = x0i - (wr * x1i + wi * x1r)

        # set up encoder, decoder, mul, add, and accuracy
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

        # # mul accuracy
        # self.acc_wr_x1r = accuracy(acc_config)
        # self.acc_wr_x1i = accuracy(acc_config)
        # self.acc_wi_x1r = accuracy(acc_config)
        # self.acc_wi_x1i = accuracy(acc_config)

        # # output accuracy
        # self.acc_y0r = accuracy(acc_config)
        # self.acc_y0i = accuracy(acc_config)
        # self.acc_y1r = accuracy(acc_config)
        # self.acc_y1i = accuracy(acc_config)


    @napl_sim_timesteps_class
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
        y0r_spike = self.add_y0r(torch.stack([x0r_spike,     wr_x1r_spike, 1 - wi_x1i_spike], dim=-1), dim=-1)
        y0i_spike = self.add_y0i(torch.stack([x0i_spike,     wr_x1i_spike,     wi_x1r_spike], dim=-1), dim=-1)
        y1r_spike = self.add_y1r(torch.stack([x0r_spike, 1 - wr_x1r_spike,     wi_x1i_spike], dim=-1), dim=-1)
        y1i_spike = self.add_y1i(torch.stack([x0i_spike, 1 - wr_x1i_spike, 1 - wi_x1r_spike], dim=-1), dim=-1)

        # decode y
        y0r = self.decoder_y0r(y0r_spike)
        y0i = self.decoder_y0i(y0i_spike)
        y1r = self.decoder_y1r(y1r_spike)
        y1i = self.decoder_y1i(y1i_spike)

        # # measure y accuracy
        # self.acc_y0r(y0r_spike)
        # self.acc_y0i(y0i_spike)
        # self.acc_y1r(y1r_spike)
        # self.acc_y1i(y1i_spike)

        # # measure mul accuracy
        # self.acc_wr_x1r(wr_x1r_spike)
        # self.acc_wr_x1i(wr_x1i_spike)
        # self.acc_wi_x1r(wi_x1r_spike)
        # self.acc_wi_x1i(wi_x1i_spike)

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

        # return y0r, y0i, y1r, y1i, wr_x1r, wr_x1i, wi_x1r, wi_x1i
        return y0r, y0i, y1r, y1i
    

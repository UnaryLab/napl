import torch, math

from napl.sim.base import napl_base, hw_params
from napl.sim.module.encoder import gen_num_seq
from napl.sim.operation.dff import dff
from loguru import logger


class tanh_p1(napl_base):
    """
    Combinational tanh(x) for unipolar spike streams via series expansion, from
    "K. Parhi and Y. Liu. Computing Arithmetic Functions Using Stochastic Logic
    by Series Expansion. IEEE TETC 2017", fig. 10: a NAND/AND cascade against
    four internal constant spike streams (62/153, 17/42, 2/5, 1/3), with DFF
    delay lines decorrelating the reused input and n_1 streams.
    """
    def __init__(
            self,
            config={
                'polarity': 'unipolar',
                'timestep': 256,
                'generator': 'sobol',
            }
    ):
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        assert self.polarity == 'unipolar', \
            logger.error(f'Invalid polarity: <{self.polarity}>; combinational tanh_p1 needs unipolar mode.')

        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        self.width = math.ceil(math.log2(self.timestep))
        self.generator = config['generator'].lower()
        self.len = 2**self.width

        # combinational input->output path (AND/NAND cascade); the internal DFF
        # delay lines are decorrelators holding state, not pipeline stages.
        self.hw = hw_params(pp_delay=0)

        # four constant spike streams on consecutive decorrelated dims, quantized
        # to self.width bits (round(c*len) vs floor(rng*len)) so the bits match
        # the hardware bit-stream generator exactly.
        dim = config.get('dim', 1)
        coef_seq = []
        for i, coef in enumerate([62/153, 17/42, 2/5, 1/3]):
            num_seq = gen_num_seq(config={'width': self.width,
                                          'generator': self.generator,
                                          'dim': dim + i})
            coef_bin = torch.tensor(coef, dtype=self.ntype).mul(self.len).round()
            coef_seq.append(torch.gt(coef_bin, num_seq.mul(self.len).floor()).type(torch.int8))
        self.coef_seq = torch.nn.Parameter(torch.stack(coef_seq), requires_grad=False)
        # per-timestep coefficient bits as Python ints: avoids a 4-way tensor
        # index (and its GPU launches) in the hot path, and lets forward()
        # constant-fold the NAND stages whose bit is 0.
        self.coef_bits = [tuple(bits) for bits in zip(*(seq.tolist() for seq in coef_seq))]

        # DFF delay lines: input tapped at depth 4 and 8, n_1 at depth 1, 2, 3
        self.input_dff_4 = dff({'depth': 4})
        self.input_dff_8 = dff({'depth': 4})
        self.n_1_dff_1 = dff({'depth': 1})
        self.n_1_dff_2 = dff({'depth': 1})
        self.n_1_dff_3 = dff({'depth': 1})


    def forward(self, input: torch.tensor):
        # input is a spike tensor
        in_i8 = input.type(torch.int8)
        c2, c3, c4, c5 = self.coef_bits[(self.timestep_cur - 1) % self.len]

        in_d4 = self.input_dff_4(in_i8)
        in_d8 = self.input_dff_8(in_d4)

        # delay lines always advance, independent of the coefficient bits
        n_1 = in_i8 & in_d4
        n_1_d1 = self.n_1_dff_1(n_1)
        n_1_d2 = self.n_1_dff_2(n_1_d1)
        n_1_d3 = self.n_1_dff_3(n_1_d2)

        # NAND cascade with the coefficient bits folded as Python 0/1
        # constants: a stage whose bit is 0 outputs the constant-1 stream,
        # represented as None so it costs no tensor kernel.
        n_2 = (1 - n_1) if c2 else None
        n_3 = (1 - (n_1_d1 if n_2 is None else n_2 & n_1_d1)) if c3 else None
        n_4 = (1 - (n_1_d2 if n_3 is None else n_3 & n_1_d2)) if c4 else None
        n_5 = (1 - (n_1_d3 if n_4 is None else n_4 & n_1_d3)) if c5 else None
        out = in_d8 if n_5 is None else (n_5 & in_d8)
        return out.type(self.stype)

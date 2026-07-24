import torch, math

from napl.sim.base import napl_base, hw_params
from napl.sim.module.encoder import gen_num_seq
from loguru import logger


class exp_n1(napl_base):
    """
    Unary exp(-x) for unipolar input, via the truncated Maclaurin series
    exp(-x) = 1 - x(1 - x/2(1 - x/3(1 - x/4(1 - x/5)))): a chain of NAND-style
    stages, each ANDing a delayed tap of the input stream (delays decorrelate
    the reused stream) with a constant stream (1/5, 1/4, 1/3, 1/2).
    Reference:
    K. Parhi and Y. Liu. "Computing Arithmetic Functions Using Stochastic
    Logic by Series Expansion." IEEE Transactions on Emerging Topics in
    Computing, 2017, Fig. 12.
    """
    def __init__(
        self,
        config={
            'polarity': 'unipolar',
            'timestep': 256,
            'generator': 'sobol',
            'dim': 1,
        }
    ):
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        assert self.polarity == 'unipolar', \
            logger.error(f'Invalid polarity: <{self.polarity}>; exp_n1 supports unipolar only.')

        self.timestep = config['timestep']
        assert self.timestep > 0, \
            logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        self.width = math.ceil(math.log2(self.timestep))
        self.len = 2**self.width
        dim = config.get('dim', 1)

        # combinational NAND chain input->output; the internal DFF taps only
        # decorrelate the reused input stream, they are not pipeline stages
        self.hw = hw_params(pp_delay=0)

        # series constants quantized to width-bit sources (round), matching the
        # RTL constant register and UnarySim's SourceGen bit-for-bit
        const_q = torch.tensor([0.2000, 0.2500, 0.3333, 0.5000]).mul(self.len).round().div(self.len)
        # one decorrelated sequence per constant (dims dim..dim+3); the streams
        # are periodic in self.len, so precompute the whole (len, 4) spike table
        seqs = torch.stack(
            [gen_num_seq({'width': self.width, 'generator': config['generator'], 'dim': dim + i}).data
             for i in range(4)], dim=1)
        self.const_spike = torch.nn.Parameter(
            torch.gt(const_q.unsqueeze(0).type(self.ntype), seqs).type(torch.int8), requires_grad=False)
        # host-side copy of the constant bits: spikes are {0,1}, so a 0 bit
        # collapses its whole NAND stage to the scalar 1 and a 1 bit makes the
        # AND an identity, skipping the per-timestep tensor & with a constant
        self._const_bits = self.const_spike.tolist()

        # input delay taps d1..d4; scalar zeros broadcast to the input shape on
        # the first forward()
        self.input_d1 = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)
        self.input_d2 = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)
        self.input_d3 = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)
        self.input_d4 = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)


    def _reset(self):
        for tap in (self.input_d1, self.input_d2, self.input_d3, self.input_d4):
            tap.data = torch.zeros(1, dtype=self.stype, device=tap.device)


    def forward(self, input: torch.tensor):
        # input is a spike tensor
        # n_k is a tensor iff its constant bit is 1, the scalar 1 otherwise
        c0, c1, c2, c3 = self._const_bits[(self.timestep_cur - 1) % self.len]
        n_1 = 1 - input.type(torch.int8) if c0 else 1
        if c1:
            d1 = self.input_d1.type(torch.int8)
            n_2 = 1 - (n_1 & d1) if c0 else 1 - d1
        else:
            n_2 = 1
        if c2:
            d2 = self.input_d2.type(torch.int8)
            n_3 = 1 - (n_2 & d2) if c1 else 1 - d2
        else:
            n_3 = 1
        if c3:
            d3 = self.input_d3.type(torch.int8)
            n_4 = 1 - (n_3 & d3) if c2 else 1 - d3
        else:
            n_4 = 1
        d4 = self.input_d4.type(torch.int8)
        output = 1 - (n_4 & d4) if c3 else 1 - d4
        if output.shape != input.shape:
            # d4 is still the scalar init tap: keep the input-shaped contract
            output = output.expand(input.shape)
        # shift the delay line oldest-first; taps hold references (the producer
        # emits a fresh tensor each timestep, same assumption as dff)
        self.input_d4.data = self.input_d3.data
        self.input_d3.data = self.input_d2.data
        self.input_d2.data = self.input_d1.data
        self.input_d1.data = input
        return output.type(self.stype)

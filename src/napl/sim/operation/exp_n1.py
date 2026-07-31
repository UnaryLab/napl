import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.module.encoder import gen_num_seq
from loguru import logger


class exp_n1(napl_base):
    """
    Approximate ``exp(-x)`` from a unipolar rate-coded spike stream.

    This streaming kernel uses a truncated Maclaurin-series circuit. Use it
    when the input represents values in ``[0, 1]`` and a stochastic
    approximation of the negative exponential is required.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import exp_n1

        operation = exp_n1()
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        K. Parhi and Y. Liu, *Computing Arithmetic Functions Using Stochastic Logic by Series Expansion*, IEEE Transactions on Emerging Topics in Computing, 2017, Fig. 12.
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
        """
        Configure the unipolar series-expansion kernel and its constant streams.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding. The only supported value is ``"unipolar"``; the default is ``"unipolar"``.
              - **timestep**: Positive target stream length used to select the sequence width; the default is ``256``.
              - **generator**: Number-sequence generator accepted by :func:`napl.sim.module.encoder.gen_num_seq`; the default is ``"sobol"``.
              - **dim**: First Sobol dimension used for the four constant streams; the default is ``1``.
              - **name**: Optional module name.
        """
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
            [gen_num_seq({'width': self.width, 'generator': config['generator'], 'dim': dim + i})
             for i in range(4)], dim=1)
        self.register_buffer(
            'const_spike',
            torch.gt(const_q.unsqueeze(0).type(self.ntype), seqs).type(torch.int8),
        )
        # host-side copy of the constant bits: spikes are {0,1}, so a 0 bit
        # collapses its whole NAND stage to the scalar 1 and a 1 bit makes the
        # AND an identity, skipping the per-timestep tensor & with a constant
        self._const_bits = self.const_spike.tolist()

        # input delay taps d1..d4; scalar zeros broadcast to the input shape on
        # the first forward()
        self.register_buffer('input_d1', torch.zeros(1).type(self.stype))
        self.register_buffer('input_d2', torch.zeros(1).type(self.stype))
        self.register_buffer('input_d3', torch.zeros(1).type(self.stype))
        self.register_buffer('input_d4', torch.zeros(1).type(self.stype))


    def _reset(self):
        """
        Clear the four input-delay taps to their scalar zero state.
        """
        self.input_d1.resize_(1).zero_()
        self.input_d2.resize_(1).zero_()
        self.input_d3.resize_(1).zero_()
        self.input_d4.resize_(1).zero_()


    def forward(self, input: torch.tensor):
        """
        Process one timestep of a unipolar input stream.

        The call advances the internal input-delay line and returns one
        unipolar output spike per input element.

        Args:
            input: Tensor of current 0/1 input spikes.

        Returns:
            Output spike tensor with the same shape and spike dtype as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
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
        # shift the delay line oldest-first
        self.input_d4.resize_as_(self.input_d3).copy_(self.input_d3.detach())
        self.input_d3.resize_as_(self.input_d2).copy_(self.input_d2.detach())
        self.input_d2.resize_as_(self.input_d1).copy_(self.input_d1.detach())
        self.input_d1.resize_as_(input).copy_(input.detach())
        return output.type(self.stype)

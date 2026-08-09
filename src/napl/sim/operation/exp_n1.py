import torch
import math

from napl.sim.base import napl_base
from .encode import encode
from loguru import logger


class exp_n1(napl_base):
    """
    Approximate ``exp(-x)`` from a unipolar rate-coded spike stream.

    This streaming kernel uses a truncated Maclaurin-series circuit. Use it
    when the input represents values in ``[0, 1]`` and a stochastic
    approximation of the negative exponential is required.

    The target operation is

    .. math::

       f(x) = \\exp(-x).

    The series is truncated after five terms and its coefficients are
    quantized, following the circuit in Fig. 12 of the reference below, so the
    approximated rate-domain operation is

    .. math::

       \\mathbb{E}[y] = 1 - x + c_4 x^2 - c_4 c_3 x^3
       + c_4 c_3 c_2 x^4 - c_4 c_3 c_2 c_1 x^5,

    where ``c_1``, ``c_2``, ``c_3``, and ``c_4`` are ``0.2``, ``0.25``,
    ``0.3333``, and ``0.5`` rounded to ``2**ceil(log2(T))`` levels for stream
    length ``T``. With the default ``T = 256`` they are
    ``(c_1, c_2, c_3, c_4) = (51/256, 1/4, 85/256, 1/2)``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import exp_n1

        operation = exp_n1()
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *Computing Arithmetic Functions Using Stochastic Logic by Series Expansion*, IEEE Transactions on Emerging Topics in Computing, 2019.
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
              - **generator**: Number-sequence generator accepted by :func:`napl.sim.operation.encode.gen_num_seq`; the default is ``"sobol"``.
              - **dim**: First Sobol dimension used for the four constant streams; the default is ``1``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim'], polarity_required=True)
        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; exp_n1 supports unipolar only.'
            logger.error(message)
            raise AssertionError(message)

        #: Requested stream length used to size the periodic coefficient sequences.
        self.timestep = config['timestep']
        if self.timestep <= 0:
            message = f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.'
            logger.error(message)
            raise AssertionError(message)
        #: Bit width of the power-of-two coefficient sequences.
        self.width = math.ceil(math.log2(self.timestep))
        #: Period of each coefficient spike sequence.
        self.len = 2**self.width
        dim = config.get('dim', 1)

        # Rounded width-bit constants match the RTL registers and UnarySim SourceGen.
        const_q = torch.tensor([0.2000, 0.2500, 0.3333, 0.5000]).mul(self.len).round().div(self.len)
        # Consecutive dimensions provide four decorrelated sequences of period self.len.
        # One encoder per coefficient generates its full period once, here, so the
        # hot path keeps reading Python ints.
        const_q = const_q.type(self.ntype)
        reference_encode = [encode({'polarity': 'unipolar', 'timestep': self.len,
                                    'generator': config['generator'], 'dim': dim + i})
                            for i in range(4)]
        #: Four periodic constant spike streams for the series coefficients.
        self.const_spike: torch.Tensor
        self.register_buffer(
            'const_spike',
            torch.stack([torch.cat([reference_encode_i(const_q[i:i + 1]) for _ in range(self.len)])
                         for i, reference_encode_i in enumerate(reference_encode)], dim=1).type(torch.int8),
        )
        self._const_bits = self.const_spike.tolist()

        # Scalar delay taps broadcast to the input shape on first use.
        #: Most recent input spike tensor in the four-stage delay line.
        self.input_d1: torch.Tensor
        self.register_buffer('input_d1', torch.zeros(1).type(self.stype))
        #: Input spike tensor delayed by two timesteps.
        self.input_d2: torch.Tensor
        self.register_buffer('input_d2', torch.zeros(1).type(self.stype))
        #: Input spike tensor delayed by three timesteps.
        self.input_d3: torch.Tensor
        self.register_buffer('input_d3', torch.zeros(1).type(self.stype))
        #: Input spike tensor delayed by four timesteps.
        self.input_d4: torch.Tensor
        self.register_buffer('input_d4', torch.zeros(1).type(self.stype))
        # DFF taps decorrelate the combinational NAND path without adding output latency.
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'unipolar', 'output': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


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
        # A zero coefficient bit represents the corresponding NAND output as scalar 1.
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
            # The scalar initial tap expands to the input shape.
            output = output.expand(input.shape)
        # Shift the delay line oldest first.
        self.input_d4.resize_as_(self.input_d3).copy_(self.input_d3.detach())
        self.input_d3.resize_as_(self.input_d2).copy_(self.input_d2.detach())
        self.input_d2.resize_as_(self.input_d1).copy_(self.input_d1.detach())
        self.input_d1.resize_as_(input).copy_(input.detach())
        return output.type(self.stype)

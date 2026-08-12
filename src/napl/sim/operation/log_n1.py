import torch
import math

from napl.sim.base import napl_base
from .encode import encode
from loguru import logger


class log_n1(napl_base):
    """
    Approximate ``log(1 + x)`` from a unipolar rate-coded spike stream.

    This streaming kernel uses a truncated Maclaurin-series circuit. Use it
    when the input represents values in ``[0, 1]`` and a stochastic
    approximation of the natural logarithm is required. Because the argument
    is ``1 + x``, it stays in ``[1, 2]`` and is positive over the whole legal
    input domain, so no input value is rejected; only the unipolar polarity is
    supported.

    The target operation is

    .. math::

       f(x) = \\log(1 + x), \\qquad x \\in [0, 1].

    The Maclaurin series is truncated after five terms and its coefficients are
    quantized, so the approximated rate-domain operation is

    .. math::

       \\mathbb{E}[y] = x - \\tfrac{1}{2} x^2 + c_3 x^3 - c_2 x^4 + c_1 x^5,

    where ``c_1``, ``c_2``, and ``c_3`` are ``0.2``, ``0.25``, and ``0.3333``
    when unquantized (the reciprocals ``1/5``, ``1/4``, ``1/3``). The output
    rate stays in ``[0, log(2)] \\subset [0, 1]`` over the legal domain.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import log_n1

        operation = log_n1()
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        *Computing Arithmetic Functions Using Stochastic Logic by Series Expansion*, IEEE Transactions on Emerging Topics in Computing, 2019.
    """
    # derived: the log circuit is the inverse-function counterpart of exp_n1
    # (same series-expansion method as the reference above, which derives the
    # exp(-x) circuit in its Fig. 12). The Maclaurin series is
    #     log(1 + x) = x - x^2/2 + x^3/3 - x^4/4 + x^5/5 - ...
    # Truncated to five terms and Horner-factored, this is
    #     log(1 + x) ~= x * (1 - (1/2) x (1 - (2/3) x (1 - (3/4) x (1 - (4/5) x)))).
    # A NAND stage computes 1 - a*b in the unipolar rate domain and an AND stage
    # computes a*b, so with rate p = rate(input) and independent stream copies
    # (the delay taps decorrelate them):
    #     m_1 = 1 - (4/5) p
    #     m_2 = 1 - (3/4) p * m_1
    #     m_3 = 1 - (2/3) p * m_2
    #     m_4 = 1 - (1/2) p * m_3
    #     out =           p * m_4
    # expands to p - p^2/2 + p^3/3 - p^4/4 + p^5/5, the truncated series above.
    # The only structural difference from exp_n1 is the constants (4/5, 3/4,
    # 2/3, 1/2 here vs 1/5, 1/4, 1/3, 1/2) and the final stage: an AND
    # (out = m_4 & input_d4, multiply by x) instead of exp_n1's output NAND.

    #: The coefficient streams are encoded from held coefficient codes, so the
    #: RTL counterpart holds its own encoder instead of sharing an external one.
    internal_encode = True


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
              - **generator**: Number-sequence generator accepted by ``gen_num_seq``; the default is ``"sobol"``.
              - **dim**: First Sobol dimension used for the four constant streams; the default is ``1``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim'], polarity_required=True)
        # log(1 + x) is well defined only for a positive argument; unipolar
        # input keeps x in [0, 1] so the argument 1 + x stays in [1, 2] > 0.
        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; log_n1 supports unipolar only.'
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

        # Round the Horner coefficients (4/5, 3/4, 2/3, 1/2) to self.len levels.
        const_q = torch.tensor([0.8000, 0.7500, 0.6667, 0.5000]).mul(self.len).round().div(self.len)
        const_q = const_q.type(self.ntype)
        # One encoder per coefficient on consecutive dimensions gives four decorrelated
        # sequences under a sobol generator; a dim-insensitive generator such as lfsr
        # yields four identical sequences.
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
        # Cache one full coefficient period as Python ints so the hot path skips tensor indexing.
        self._const_bits = self.const_spike.tolist()

        # Delay taps start scalar and resize to the input shape on the first call.
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
        # DFF taps decorrelate the combinational cascade without adding output latency.
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': 'unipolar', 'output': 'unipolar'}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear the four input-delay taps to their scalar zero state.
        """
        self.input_d1.resize_(1).zero_()
        self.input_d2.resize_(1).zero_()
        self.input_d3.resize_(1).zero_()
        self.input_d4.resize_(1).zero_()


    def forward(self, input: torch.Tensor):
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
        d4 = self.input_d4.clone().type(torch.int8)
        # Final stage is an AND (multiply by x), not exp_n1's output NAND.
        output = (n_4 & d4) if c3 else d4
        if output.shape != input.shape:
            output = output.expand(input.shape)
        # Shift the delay line oldest first.
        self.input_d4.resize_as_(self.input_d3).copy_(self.input_d3.detach())
        self.input_d3.resize_as_(self.input_d2).copy_(self.input_d2.detach())
        self.input_d2.resize_as_(self.input_d1).copy_(self.input_d1.detach())
        self.input_d1.resize_as_(input).copy_(input.detach())
        return output.type(self.stype)

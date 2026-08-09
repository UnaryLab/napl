import torch
import math

from napl.sim.base import napl_base
from .encode import encode
from .dff import dff
from loguru import logger


class tanh_p1(napl_base):
    r"""
    Approximate tanh with a code-quantized odd series on a unipolar spike stream.

    The target rate-domain operation is

    .. math::

       f(x) = \tanh(x).

    The kernel evaluates the truncated odd-series approximation, built from the
    circuit in Fig. 10 of the reference below,

    .. math::

       \mathbb{E}[y] =
       x-c_5x^3+c_5c_4x^5-c_5c_4c_3x^7+c_5c_4c_3c_2x^9,

    with :math:`c_2 = 62/153`, :math:`c_3 = 17/42`, :math:`c_4 = 2/5`, and
    :math:`c_5 = 1/3`, each quantized to the coefficient-sequence period
    :math:`L = 2^{\lceil \log_2 timestep \rceil}`.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import tanh_p1

        operation = tanh_p1()
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
            }
    ):
        """
        Configure the unipolar series-expansion kernel and coefficient streams.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding. The only supported value is ``"unipolar"``; the default is ``"unipolar"``.
              - **timestep**: Positive target stream length used to select the sequence width; the default is ``256``.
              - **generator**: Number-sequence generator accepted by :func:`napl.sim.operation.encode.gen_num_seq`; the default is ``"sobol"``.
              - **dim**: First Sobol dimension used for the four coefficient streams; the default is ``1``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['dim'], polarity_required=True)
        if self.polarity != 'unipolar':
            message = f'Invalid polarity: <{self.polarity}>; combinational tanh_p1 needs unipolar mode.'
            logger.error(message)
            raise AssertionError(message)

        #: Requested stream length used to size the coefficient sequences.
        self.timestep = config['timestep']
        if self.timestep <= 0:
            message = f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.'
            logger.error(message)
            raise AssertionError(message)
        #: Bit width of the power-of-two coefficient sequences.
        self.width = math.ceil(math.log2(self.timestep))
        #: Lowercase name of the configured number-sequence generator.
        self.generator = config['generator'].lower()
        #: Period of each coefficient spike sequence.
        self.len = 2**self.width


        # Width-bit quantization on consecutive dimensions matches the hardware generator.
        dim = config.get('dim', 1)
        # Quantizing to width bits turns the integer threshold comparison into the
        # encoder's own probability comparison, since self.len is a power of two.
        # Each encoder generates its full period once, here, so the hot path keeps
        # reading Python ints.
        coef_q = torch.tensor([62/153, 17/42, 2/5, 1/3],
                              dtype=self.ntype).mul(self.len).round().div(self.len)
        coef_seq = []
        for i in range(4):
            reference_encode = encode({'polarity': 'unipolar', 'timestep': self.len,
                                       'generator': self.generator, 'dim': dim + i})
            coef_seq.append(torch.cat([reference_encode(coef_q[i:i + 1])
                                       for _ in range(self.len)]).type(torch.int8))
        #: Four periodic coefficient spike streams used by the polynomial stages.
        self.coef_seq: torch.Tensor
        self.register_buffer('coef_seq', torch.stack(coef_seq))
        #: Timestep-major Python view of :attr:`coef_seq` used by the hot path.
        self.coef_bits = [tuple(bits) for bits in zip(*(seq.tolist() for seq in coef_seq))]

        # Input taps are at depths 4 and 8; n_1 taps are at depths 1, 2, and 3.
        #: First four-timestep delay segment for the input spike path.
        self.input_dff_4 = dff({'depth': 4})
        #: Second four-timestep delay segment, producing an eight-timestep input delay.
        self.input_dff_8 = dff({'depth': 4})
        #: One-timestep delay segment for the first polynomial intermediate.
        self.n_1_dff_1 = dff({'depth': 1})
        #: Second one-timestep delay segment for the first polynomial intermediate.
        self.n_1_dff_2 = dff({'depth': 1})
        #: Third one-timestep delay segment for the first polynomial intermediate.
        self.n_1_dff_3 = dff({'depth': 1})
        # DFF delay lines decorrelate the combinational path without adding output latency.
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'out': 'rc'}
        self.polarity_io = {'input': 'unipolar', 'out': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Reset no class-owned state; :meth:`reset` resets the child delay lines.
        """
        pass


    def forward(self, input: torch.tensor):
        """
        Process one timestep of a unipolar input stream.

        The call advances all decorrelating delay lines and evaluates the
        coefficient-selected NAND/AND cascade.

        Args:
            input: Tensor of current 0/1 unipolar input spikes.

        Returns:
            Unipolar 0/1 output spike tensor with the same shape as ``input``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.0, 1.0]))
        """
        in_i8 = input.type(torch.int8)
        c2, c3, c4, c5 = self.coef_bits[(self.timestep_cur - 1) % self.len]

        in_d4 = self.input_dff_4(in_i8)
        in_d8 = self.input_dff_8(in_d4)

        # Delay lines advance independently of the coefficient bits.
        n_1 = in_i8 & in_d4
        n_1_d1 = self.n_1_dff_1(n_1)
        n_1_d2 = self.n_1_dff_2(n_1_d1)
        n_1_d3 = self.n_1_dff_3(n_1_d2)

        # A zero coefficient bit represents the corresponding NAND stage as constant 1 (None).
        n_2 = (1 - n_1) if c2 else None
        n_3 = (1 - (n_1_d1 if n_2 is None else n_2 & n_1_d1)) if c3 else None
        n_4 = (1 - (n_1_d2 if n_3 is None else n_3 & n_1_d2)) if c4 else None
        n_5 = (1 - (n_1_d3 if n_4 is None else n_4 & n_1_d3)) if c5 else None
        out = in_d8 if n_5 is None else (n_5 & in_d8)
        return out.type(self.stype)

import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.module.encoder import gen_num_seq
from napl.sim.operation.dff import dff
from loguru import logger


class tanh_p1(napl_base):
    """
    Approximate ``tanh(x)`` from a unipolar spike stream by series expansion.

    Use this streaming kernel when the input represents values in ``[0, 1]``
    and a stochastic NAND/AND-cascade implementation of hyperbolic tangent is
    required.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import tanh_p1

        operation = tanh_p1()
        output = operation(torch.tensor([0.0, 1.0]))

    .. container:: api-references

        .. rubric:: References

        K. Parhi and Y. Liu, *Computing Arithmetic Functions Using Stochastic Logic by Series Expansion*, IEEE Transactions on Emerging Topics in Computing, 2017, Fig. 10.
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
              - **generator**: Number-sequence generator accepted by :func:`napl.sim.module.encoder.gen_num_seq`; the default is ``"sobol"``.
              - **dim**: First Sobol dimension used for the four coefficient streams; the default is ``1``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        assert self.polarity == 'unipolar', \
            logger.error(f'Invalid polarity: <{self.polarity}>; combinational tanh_p1 needs unipolar mode.')

        #: Requested stream length used to size the coefficient sequences.
        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        #: Bit width of the power-of-two coefficient sequences.
        self.width = math.ceil(math.log2(self.timestep))
        #: Lowercase name of the configured number-sequence generator.
        self.generator = config['generator'].lower()
        #: Period of each coefficient spike sequence.
        self.len = 2**self.width

        # DFF delay lines decorrelate the combinational path without adding output latency.
        #: Hardware latency and timing metadata for the combinational output path.
        self.hw = hw_params(pp_delay=0)

        # Width-bit quantization on consecutive dimensions matches the hardware generator.
        dim = config.get('dim', 1)
        coef_seq = []
        for i, coef in enumerate([62/153, 17/42, 2/5, 1/3]):
            num_seq = gen_num_seq(config={'width': self.width,
                                          'generator': self.generator,
                                          'dim': dim + i})
            coef_bin = torch.tensor(coef, dtype=self.ntype).mul(self.len).round()
            coef_seq.append(torch.gt(coef_bin, num_seq.mul(self.len).floor()).type(torch.int8))
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

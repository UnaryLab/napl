import torch

from napl.sim.base import napl_base
from .encode_cond import encode_cond
from loguru import logger


class mul_ugemm(napl_base):
    r"""
    Multiply a spike stream by a binary-domain operand.

    Use conditional spike generation when ``input_0`` arrives one timestep at a
    time and ``input_1`` is a numeric tensor. Each call reads ``input_1``, so an
    externally updated operand takes effect on the next call within a run.

    Sequence indices advance without wrapping, so a run has a per-polarity
    budget over the sequence period ``len = 2 ** ceil(log2(timestep))``:

    - **unipolar**: ``seq_idx`` advances only on enabling timesteps, so a run
      allows at most ``len`` timesteps with ``input_0 == 1``. Timesteps with
      ``input_0 == 0`` are unbounded.
    - **bipolar**: ``seq_idx`` advances on enabling timesteps and ``seq_idx_inv``
      advances on non-enabling timesteps, so a run allows at most ``len``
      timesteps with ``input_0 == 1`` **and** at most ``len`` timesteps with
      ``input_0 == 0``, up to ``2 * len`` timesteps in total.

    Exceeding either budget raises an ``IndexError`` from the number-sequence
    lookup; the index does not wrap silently. ``reset()`` clears both indices and
    starts a new run.

    The target rate-domain operation is

    .. math::

       p_y = p_0p_1 \quad (\text{unipolar}),\qquad
       v_y = v_0v_1 \quad (\text{bipolar}).

    The product is sampled over a finite number sequence, so the output rate
    approximates the target rather than matching it exactly.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import mul_ugemm

        multiply = mul_ugemm({'polarity': 'unipolar', 'timestep': 256,
                            'generator': 'sobol'})
        output = multiply(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([0.5]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.

        *uGEMM: Unary Computing for GEMM Applications*, IEEE Micro, 2021.
    """
    #: Encoder the hardware counterpart carries. Encoding advances
    #: conditionally on the data, so the RTL holds its own input-gated Sobol
    #: generator, which cannot ride a shared sequencer.
    internal_encode = 'private'
    #: Dominant hardware mechanism of this class.
    mechanism = 'conditional-generation'


    def __init__(
            self,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
            }
        ):
        """
        Configure the conditional number sequence.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Positive target stream length used to size the sequence; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)

        #: Conditional spike generator that owns the number sequence and its indices.
        self.gen = encode_cond(config)

        #: Requested stream length used to size the conditional number sequence.
        self.timestep = self.gen.timestep
        #: Bit width of the power-of-two conditional number sequence.
        self.width = self.gen.width
        #: Lowercase name of the configured number-sequence generator.
        self.generator = self.gen.generator
        #: Period of the conditional number sequence.
        self.len = self.gen.len

        # The product spike is combinational, so the pipeline delay is zero.
        #: Hardware latency and timing metadata for the combinational multiplier.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': self.polarity, 'input_1': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Hold no local state: ``reset()`` restarts the sequence indices through the registered generator.
        """
        pass


    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor):
        """
        Generate one product-spike timestep.

        Args:
            input_0: Current 0/1 spike tensor that enables sequence progress.
            input_1: Numeric multiplier tensor, read on every call, so an update
                between calls takes effect on the next call within a run.

        Returns:
            A product spike tensor with the broadcast input shape. The enabled
            sequence index, and both indices in bipolar mode, are updated.

        **Example:**

        .. code-block:: python

            output = multiply(torch.tensor([1], dtype=torch.int8),
                              torch.tensor([0.5]))
        """
        if input_1 is None:
            message = 'Invalid input_1: <None>; legal values: a numeric tensor.'
            logger.error(message)
            raise AssertionError(message)
        return self.gen(input_0, input_1)


    @property
    def num_seq(self) -> torch.Tensor:
        """
        Periodic number sequence the generator thresholds to produce operand spikes.
        """
        return self.gen.num_seq


    @property
    def seq_idx(self) -> torch.Tensor:
        """
        Per-element index of the next number-sequence value for input-one events.
        """
        return self.gen.seq_idx


    @property
    def seq_idx_inv(self) -> torch.Tensor:
        """
        Per-element index for the complementary bipolar input-zero path.
        """
        return self.gen.seq_idx_inv

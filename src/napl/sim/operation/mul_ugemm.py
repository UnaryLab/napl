import torch
import math

from napl.sim.base import napl_base
from .encode import encode
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
    #: Encoding advances conditionally on data, so the RTL counterpart holds
    #: its own encoder instead of sharing an external one.
    internal_encode = True


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

        #: Requested stream length used to size the conditional number sequence.
        self.timestep = config['timestep']
        if self.timestep <= 0:
            message = f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.'
            logger.error(message)
            raise AssertionError(message)
        #: Bit width of the power-of-two conditional number sequence.
        self.width = math.ceil(math.log2(self.timestep))
        #: Lowercase name of the configured number-sequence generator.
        self.generator = config['generator'].lower()
        #: Period of the conditional number sequence.
        self.len = 2**self.width


        #: Periodic number sequence used to generate conditional operand spikes.
        self.num_seq: torch.Tensor
        self.register_buffer(
            'num_seq',
            encode({'polarity': self.polarity,
                    'timestep': self.len,
                    'generator': self.generator}).num_seq,
        )

        #: Per-element index of the next number-sequence value for input-one events.
        self.seq_idx: torch.Tensor
        self.register_buffer('seq_idx', torch.zeros(1, dtype=torch.long))
        if self.polarity == 'bipolar':
            #: Per-element index for the complementary bipolar input-zero path.
            self.seq_idx_inv: torch.Tensor
            self.register_buffer('seq_idx_inv', torch.zeros(1, dtype=torch.long))

        # The product spike is combinational, so the pipeline delay is zero.
        #: Hardware latency and timing metadata for the combinational multiplier.
        self.hw.pp_delay = 0

        self.encoding_io = {'input_0': 'rc', 'input_1': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_0': self.polarity, 'input_1': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Restart the enabled sequence indices.
        """
        self.seq_idx.resize_(1).zero_()
        if self.polarity == 'bipolar':
            self.seq_idx_inv.resize_(1).zero_()


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
        in_1_prob = ((input_1 + 1) / 2 if self.polarity == 'bipolar' else input_1).type(self.ntype)

        # int8 inputs promote to long in the sequence-index update.
        in_0_i8 = input_0.type(torch.int8)
        spike_csg = torch.gt(in_1_prob, self.num_seq[self.seq_idx])
        path = in_0_i8 & spike_csg
        if self.seq_idx.shape == in_0_i8.shape:
            self.seq_idx.add_(in_0_i8)
        else:
            updated = self.seq_idx.add(in_0_i8)
            self.seq_idx.resize_as_(updated).copy_(updated.detach())

        if self.polarity == 'unipolar':
            return path.type(self.stype)
        else:
            spike_csg = torch.gt(in_1_prob, self.num_seq[self.seq_idx_inv])
            inv_in_0_i8 = in_0_i8 ^ 1
            path_inv = inv_in_0_i8 & ~spike_csg
            if self.seq_idx_inv.shape == inv_in_0_i8.shape:
                self.seq_idx_inv.add_(inv_in_0_i8)
            else:
                updated = self.seq_idx_inv.add(inv_in_0_i8)
                self.seq_idx_inv.resize_as_(updated).copy_(updated.detach())
            return (path | path_inv).type(self.stype)

import torch
import math

from napl.sim.base import napl_base
from .encode import encode
from loguru import logger


class encode_cond(napl_base):
    r"""
    Generate a spike stream for a numeric value, conditioned on an input stream.

    Use conditional spike generation when the generated stream must ride the
    timing of an incoming spike stream instead of a free-running clock, as
    :class:`encode` does. The number sequence advances only on enabling
    timesteps of ``input``, and the emitted spike is gated by the same
    condition. Each call reads ``value``, so an externally updated operand takes
    effect on the next call within a run.

    Sequence indices advance without wrapping, so a run has a per-polarity
    budget over the sequence period ``len = 2 ** ceil(log2(timestep))``:

    - **unipolar**: ``seq_idx`` advances only on enabling timesteps, so a run
      allows at most ``len`` timesteps with ``input == 1``. Timesteps with
      ``input == 0`` are unbounded.
    - **bipolar**: ``seq_idx`` advances on enabling timesteps and ``seq_idx_inv``
      advances on non-enabling timesteps, so a run allows at most ``len``
      timesteps with ``input == 1`` **and** at most ``len`` timesteps with
      ``input == 0``, up to ``2 * len`` timesteps in total.

    Exceeding either budget raises an ``IndexError`` from the number-sequence
    lookup; the index does not wrap silently. ``reset()`` clears both indices and
    starts a new run.

    The target rate-domain operation is

    .. math::

       p_y = p_ip_v \quad (\text{unipolar}),\qquad
       v_y = v_iv_v \quad (\text{bipolar}),

    where the subscripts denote the input stream and the generated value. The
    conditional stream is sampled over a finite number sequence, so the output
    rate approximates the target rather than matching it exactly.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import encode_cond

        generate = encode_cond({'polarity': 'unipolar', 'timestep': 256,
                                  'generator': 'sobol'})
        output = generate(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([0.5]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.

        *uGEMM: Unary Computing for GEMM Applications*, IEEE Micro, 2021.
    """
    #: Encoder the hardware counterpart carries. Encoding advances
    #: conditionally on the data, so the encoder belongs to this operation
    #: alone and cannot ride a shared sequencer.
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


        #: Periodic number sequence thresholded to generate the conditional spikes.
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

        # The generated spike is combinational, so the pipeline delay is zero.
        #: Hardware latency and timing metadata for the combinational generator.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'value': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'value': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Restart the enabled sequence indices.
        """
        self.seq_idx.resize_(1).zero_()
        if self.polarity == 'bipolar':
            self.seq_idx_inv.resize_(1).zero_()


    def forward(self, input: torch.Tensor, value: torch.Tensor):
        """
        Generate one conditional-spike timestep.

        Args:
            input: Current 0/1 spike tensor that enables sequence progress.
            value: Numeric tensor to generate, read on every call, so an update
                between calls takes effect on the next call within a run.

        Returns:
            A spike tensor with the broadcast input shape. The enabled sequence
            index, and both indices in bipolar mode, are updated.

        **Example:**

        .. code-block:: python

            output = generate(torch.tensor([1], dtype=torch.int8),
                              torch.tensor([0.5]))
        """
        if value is None:
            message = 'Invalid value: <None>; legal values: a numeric tensor.'
            logger.error(message)
            raise AssertionError(message)
        value_prob = ((value + 1) / 2 if self.polarity == 'bipolar' else value).type(self.ntype)

        # int8 inputs promote to long in the sequence-index update.
        in_i8 = input.type(torch.int8)
        spike_csg = torch.gt(value_prob, self.num_seq[self.seq_idx])
        path = in_i8 & spike_csg
        if self.seq_idx.shape == in_i8.shape:
            self.seq_idx.add_(in_i8)
        else:
            updated = self.seq_idx.add(in_i8)
            self.seq_idx.resize_as_(updated).copy_(updated.detach())

        if self.polarity == 'unipolar':
            return path.type(self.stype)
        else:
            spike_csg = torch.gt(value_prob, self.num_seq[self.seq_idx_inv])
            inv_in_i8 = in_i8 ^ 1
            path_inv = inv_in_i8 & ~spike_csg
            if self.seq_idx_inv.shape == inv_in_i8.shape:
                self.seq_idx_inv.add_(inv_in_i8)
            else:
                updated = self.seq_idx_inv.add(inv_in_i8)
                self.seq_idx_inv.resize_as_(updated).copy_(updated.detach())
            return (path | path_inv).type(self.stype)

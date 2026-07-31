import torch
import math

from napl.sim.base import napl_base, hw_params
from napl.sim.module.encoder import gen_num_seq
from loguru import logger


class mul_csg(napl_base):
    """
    Multiply a spike stream by a fixed binary-domain operand.

    Use conditional spike generation when ``input_0`` arrives one timestep at a
    time and ``input_1`` is a numeric tensor held constant for the run. Separate
    enabled sequence indices implement the bipolar positive and inverse paths.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mul_csg

        multiply = mul_csg({'polarity': 'unipolar', 'timestep': 256,
                            'generator': 'sobol'})
        output = multiply(torch.tensor([1], dtype=torch.int8),
                          torch.tensor([0.5]))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*.

        *uGEMM: Unary Computing for GEMM Applications*.
    """
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

        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        self.width = math.ceil(math.log2(self.timestep))
        self.generator = config['generator'].lower()
        self.len = 2**self.width

        # combinational output; internal seq-index counter(s) are self.width-bit
        # registers that hold no output latency (pp_delay=0). RTL must reset them
        # to 0 to match reset() and instance decorrelated RNGs per polarity path.
        self.hw = hw_params(pp_delay=0)

        # generate the number sequence
        # the sequence is used to compare with the input data
        # this is ntype tensor
        self.register_buffer(
            'num_seq',
            gen_num_seq(config={'width': self.width,
                                'generator': self.generator}),
        )

        # seq_idx is used later as an enable signal, get update every cycled
        self.register_buffer('seq_idx', torch.zeros(1, dtype=torch.long))
        # Generate two seperate spike generators and two enable signals for bipolar polarity
        if self.polarity == 'bipolar':
            self.register_buffer('seq_idx_inv', torch.zeros(1, dtype=torch.long))

        # only compute the prob of input_1 once in the first call
        self.in_1_prob = None
        self.is_first_call = True


    def _reset(self):
        """
        Restart the enabled sequence indices and forget the cached operand.
        """
        self.seq_idx.resize_(1).zero_()
        if self.polarity == 'bipolar':
            self.seq_idx_inv.resize_(1).zero_()
        self.in_1_prob = None
        self.is_first_call = True


    def forward(self, input_0: torch.tensor, input_1: torch.tensor):
        """
        Generate one product-spike timestep.

        Args:
            input_0: Current 0/1 spike tensor that enables sequence progress.
            input_1: Fixed numeric multiplier tensor. Supply it on the first call;
                its cached value is reused until :meth:`reset`.

        Returns:
            A product spike tensor with the broadcast input shape. The enabled
            sequence index, and both indices in bipolar mode, are updated.

        **Example:**

        .. code-block:: python

            output = multiply(torch.tensor([1], dtype=torch.int8),
                              torch.tensor([0.5]))
        """
        # input_0 is a spike tensor
        # input_1 is a binary tensor
        if self.is_first_call is True:
            assert input_1 is not None, logger.error('input_1 is None, please provide a valid input_1 tensor.')
            # cache once as ntype so the per-timestep compare below skips the recast
            self.in_1_prob = ((input_1 + 1) / 2 if self.polarity == 'bipolar' else input_1).type(self.ntype)
            self.is_first_call = False

        # reuse the same int8 view of input_0 across both polarity paths;
        # long.add(int8) promotes to long, so the seq-index counters stay long
        # indices without a separate per-timestep int8->long cast.
        in_0_i8 = input_0.type(torch.int8)
        # generate the conditional spike; kept as bool (int8 & bool promotes to
        # int8), skipping a per-timestep bool->int8 cast on each polarity path
        spike_csg = torch.gt(self.in_1_prob, self.num_seq[self.seq_idx])
        path = in_0_i8 & spike_csg
        # conditional update for seq index when input_0 is 1, which simulates the enable signal.
        if self.seq_idx.shape == in_0_i8.shape:
            self.seq_idx.add_(in_0_i8)
        else:
            updated = self.seq_idx.add(in_0_i8)
            self.seq_idx.resize_as_(updated).copy_(updated.detach())

        if self.polarity == 'unipolar':
            return path.type(self.stype)
        else:
            # generate the conditional spike (bool, as above)
            spike_csg = torch.gt(self.in_1_prob, self.num_seq[self.seq_idx_inv])
            inv_in_0_i8 = in_0_i8 ^ 1
            path_inv = inv_in_0_i8 & ~spike_csg
            # conditional update for seq_idx_inv
            if self.seq_idx_inv.shape == inv_in_0_i8.shape:
                self.seq_idx_inv.add_(inv_in_0_i8)
            else:
                updated = self.seq_idx_inv.add(inv_in_0_i8)
                self.seq_idx_inv.resize_as_(updated).copy_(updated.detach())
            return (path | path_inv).type(self.stype)

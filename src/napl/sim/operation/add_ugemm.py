import torch

from napl.sim.base import napl_base
from napl.utils import grow_state_to
from loguru import logger


class add_ugemm(napl_base):
    """
    Add rate-coded streams with the uGEMM accumulator.

    Use scaled mode for a stream representing the mean across the reduced
    inputs. Use non-scaled mode for a clipped sum. Both modes accept unipolar
    and bipolar streams and keep accumulation state across timesteps.

    The target reductions are

    .. math::

       y_{\\mathrm{scaled}} = \\frac{1}{n}\\sum_i x_i,\\qquad
       y_{\\mathrm{non-scaled}} = \\min\\left(1,\\sum_i x_i\\right).

    The number of reduced streams is taken from the first input and stays fixed
    until ``reset()``.

    The accumulator width bounds the running accumulation: to the signed range of
    that width in scaled mode, and to that range around the emitted-spike count in
    non-scaled mode. The realized output rate departs from the target once the
    accumulation reaches the bound.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.operation import add_ugemm

        adder = add_ugemm({'polarity': 'unipolar', 'scaled': True, 'width': 10})
        output = adder(torch.tensor([1, 1], dtype=torch.int8), dim=0)

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.
    """
    #: Dominant hardware mechanism of this class.
    mechanism = 'integrate-and-fire'


    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scaled' : True,
                'width' : 10,
            }
        ):
        """
        Configure the input encoding, scaling mode, and accumulator width.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **scaled**: Emit a mean-like scaled stream when ``True`` or a clipped sum when ``False``; the default is ``True``.
              - **width**: Signed accumulator width in bits; the default is ``10``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'scaled', 'width'], polarity_required=True)

        #: Whether the output represents the mean rather than the clipped sum.
        self.scaled = config['scaled']
        #: Signed accumulator width in bits.
        self.width = config['width']
        #: Largest value retained by the signed accumulator.
        self.acc_max = 2**(self.width-1) - 1
        #: Smallest value retained by the signed accumulator.
        self.acc_min = -2**(self.width-1)
        #: Number of streams reduced per call, inferred from the first input.
        self.acc_bound = 0
        #: Bipolar centering offset inferred from :attr:`acc_bound`.
        self.offset = 0
        #: Running input sum used by both scaled and non-scaled modes.
        self.accumulator: torch.Tensor
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        if not self.scaled:
            #: Running number of spikes emitted in non-scaled mode.
            self.out_accumulator: torch.Tensor
            self.register_buffer('out_accumulator', torch.zeros(1, dtype=self.ntype))
        #: Whether the next call must infer the reduction size and offset.
        self.is_first_call = True
        #: Hardware latency and timing metadata for the combinational uGEMM adder.
        self.hw.pp_delay = 0

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """
        Clear local accumulation and first-call shape state.
        """
        self.accumulator.resize_(1).zero_()
        if not self.scaled:
            self.out_accumulator.resize_(1).zero_()
        self.is_first_call = True


    def forward(self, input, dim=-1):
        """
        Accumulate one timestep and reduce the selected dimension.

        Args:
            input: Current spike tensor containing the streams to add.
            dim: Dimension to reduce; the default is ``-1``.

        Returns:
            A spike tensor with ``dim`` removed. The call updates the running
            accumulator and, in non-scaled mode, the emitted-spike count. The
            width bounds the accumulation to its signed range.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1, 0], dtype=torch.int8), dim=0)
        """
        if self.is_first_call:
            self.acc_bound = input.size()[dim]
            if self.polarity == 'bipolar':
                self.offset = (self.acc_bound - 1) / 2
            if self.scaled and self.acc_bound > self.acc_max:
                message = (
                    f'add_ugemm reduction size <{self.acc_bound}> exceeds accumulator maximum '
                    f'<{self.acc_max}> for width <{self.width}>.'
                )
                logger.error(message)
                raise AssertionError(message)
            self.is_first_call = False

        acc_delta = torch.sum(input, dim, dtype=self.ntype)
        grow_state_to(self.accumulator, acc_delta)
        if not self.scaled:
            grow_state_to(self.out_accumulator, acc_delta)
        self.accumulator.add_(acc_delta)
        if self.scaled:
            self.accumulator.clamp_(self.acc_min, self.acc_max)

        if self.scaled:
            output = torch.ge(self.accumulator, self.acc_bound).type(self.stype)
            self.accumulator.sub_(output, alpha=self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            # The width bounds the accumulator gap that drives the comparison.
            bounded = torch.minimum(
                torch.maximum(self.accumulator, self.out_accumulator + self.acc_min),
                self.out_accumulator + self.acc_max,
            )
            self.accumulator.copy_(bounded)
            output = torch.gt(self.accumulator, self.out_accumulator).type(self.stype)
            self.out_accumulator.add_(output)

        return output

import torch
from loguru import logger

from napl.sim.base import napl_base, hw_params


class sync_skewed_int(napl_base):
    r"""
    Synchronize streams with an integral stochastic output.

    Let x_1 be the current first-stream digit, x_2 the release spike,
    c_{-1}=0, and M = 2**width-1. The exact update is

    .. math::

       \begin{aligned}
       \tilde c_t &= c_{t-1}+x_{1,t},\\
       y_{1,t} &= \mathbf{1}\{x_{2,t}=1\}
       \operatorname{clip}(\tilde c_t,0,M),\\
       c_t &= \operatorname{clip}(\tilde c_t-y_{1,t},0,M),\qquad
       y_{2,t}=x_{2,t}.
       \end{aligned}

    The first stream accumulates until a release spike arrives, and the
    second stream passes through unchanged.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import sync_skewed_int

        sync = sync_skewed_int({'width': 4})
        first, second = sync(torch.tensor([1], dtype=torch.int8),
                             torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *VLSI Implementation of Deep Neural Network Using Integral Stochastic Computing*, TVLSI, 2017.
    """


    def __init__(
            self,
            config={
                'width' : 4,
            }
        ):
        """
        Configure the integral accumulation counter.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **width**: Counter width in bits, giving a maximum output digit of ``2**width - 1`` that must fit the configured integer spike dtype; the default is ``4``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['width'], polarity_required=False)

        #: Width of the stored integer stream-skew counter in bits.
        self.width = config['width']
        #: Largest accumulated first-stream count retained by the synchronizer.
        self.cnt_max = 2**self.width - 1
        if not self.stype.is_floating_point and not self.stype.is_complex and self.stype != torch.bool:
            assert self.cnt_max <= torch.iinfo(self.stype).max, logger.error(
                f'sync_skewed_int width <{self.width}> requires digit maximum '
                f'<{self.cnt_max}>, which exceeds spike dtype <{self.stype}> '
                f'maximum <{torch.iinfo(self.stype).max}>.'
            )
        #: Per-element first-stream count awaiting release by the second stream.
        self.cnt: torch.Tensor
        self.register_buffer('cnt', torch.zeros(1, dtype=self.ntype))
        #: Hardware latency and timing metadata for the combinational synchronizer.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_2': 'rc'}
        self.polarity_io = {'input_1': 'unipolar', 'input_2': 'unipolar'}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """
        Clear the local accumulated first-stream count.
        """
        self.cnt.resize_(1).zero_()


    def forward(self, input_1, input_2):
        """
        Accumulate and conditionally release one integer-coded timestep.

        Args:
            input_1: Current first-stream spike or integer digit tensor.
            input_2: Current 0/1 release-control spike tensor.

        Returns:
            A pair ``(output_1, input_2)``. ``output_1`` contains the released
            bounded integer digits, and the second tensor is unchanged. Unreleased
            counts remain in the local counter.

        **Example:**

        .. code-block:: python

            first, second = sync(torch.tensor([1], dtype=torch.int8),
                                 torch.tensor([1], dtype=torch.int8))
        """
        # input_2 passes through unchanged. Excess input_1 spikes accumulate and are released
        # as integer digits when input_2 spikes.
        input_2_eq_1 = torch.eq(input_2, 1)
        # ntype addition broadcasts the scalar counter on first use.
        temp_sum = self.cnt + input_1
        output_1 = input_2_eq_1 * temp_sum.clamp(0, self.cnt_max)
        if temp_sum.shape == output_1.shape:
            updated = temp_sum.sub_(output_1).clamp_(0, self.cnt_max)
        else:
            updated = (temp_sum - output_1).clamp_(0, self.cnt_max)
        if self.cnt.shape == updated.shape:
            self.cnt.copy_(updated.detach())
        else:
            self.cnt.resize_as_(updated).copy_(updated.detach())
        return output_1.type(self.stype), input_2

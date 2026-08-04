import torch

from napl.sim.base import napl_base, hw_params


class add_ugemm(napl_base):
    """
    Add rate-coded streams with the uGEMM accumulator.

    Use scaled mode for a stream representing the mean across the reduced
    inputs. Use non-scaled mode for a clipped sum. Both modes accept unipolar
    and bipolar streams and keep accumulation state across timesteps.

    The precise target reductions are

    .. math::

       y_{\\mathrm{scaled}} = \\frac{1}{n}\\sum_i x_i,\\qquad
       y_{\\mathrm{non-scaled}} = \\min\\left(1,\\sum_i x_i\\right).

    Let ``r_t = sum_i x_{i,t}``, ``n`` be the number of reduced streams, and
    ``o = (n - 1) / 2`` for bipolar input or ``0`` for unipolar input. The
    accumulator and output are exactly

    .. math::

       \\tilde a_t = a_{t-1} + r_t,\\qquad
       \\begin{aligned}
       \\text{scaled:}\\quad &y_t = \\mathbf{1}\\{\\tilde a_t \\geq n\\},\\quad
       a_t = \\tilde a_t - n y_t,\\\\
       \\text{non-scaled:}\\quad &a_t = \\tilde a_t - o,\\quad
       y_t = \\mathbf{1}\\{a_t > b_{t-1}\\},\\quad b_t = b_{t-1} + y_t.
       \\end{aligned}

    ``b_t`` is the running count of non-scaled output spikes.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import add_ugemm

        adder = add_ugemm({'polarity': 'unipolar', 'scaled': True})
        output = adder(torch.tensor([1, 1], dtype=torch.int8), dim=0)
    """


    def __init__(
            self,
            config={
                'polarity' : 'bipolar',
                'scaled' : True,
            }
        ):
        """
        Configure the input encoding and scaling mode.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Input encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **scaled**: Emit a mean-like scaled stream when ``True`` or a clipped sum when ``False``; the default is ``True``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'scaled'], polarity_required=True)

        #: Whether the output represents the mean rather than the clipped sum.
        self.scaled = config['scaled']
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
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input': 'rc', 'output': 'rc'}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


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
            accumulator and, in non-scaled mode, the emitted-spike count.

        **Example:**

        .. code-block:: python

            output = adder(torch.tensor([1, 0], dtype=torch.int8), dim=0)
        """
        if self.is_first_call:
            self.acc_bound = input.size()[dim]
            if self.polarity == 'bipolar':
                self.offset = (self.acc_bound - 1) / 2
            self.is_first_call = False

        acc_delta = torch.sum(input, dim, dtype=self.ntype)
        # The scalar initial state broadcasts out of place; matching shapes update in place.
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta)
        else:
            updated = self.accumulator.add(acc_delta)
            self.accumulator.resize_as_(updated).copy_(updated.detach())

        # Integer spikes promote exactly into the ntype accumulators.
        if self.scaled:
            output = torch.ge(self.accumulator, self.acc_bound).type(self.stype)
            self.accumulator.sub_(output, alpha=self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(self.stype)
            if self.out_accumulator.shape == output.shape:
                self.out_accumulator.add_(output)
            else:
                updated = self.out_accumulator.add(output)
                self.out_accumulator.resize_as_(updated).copy_(updated.detach())

        return output

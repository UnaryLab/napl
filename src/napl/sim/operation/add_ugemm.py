import torch

from napl.sim.base import napl_base, hw_params


class add_ugemm(napl_base):
    """
    Add rate-coded streams with the uGEMM accumulator.

    Use scaled mode for a stream representing the mean across the reduced
    inputs. Use non-scaled mode for a clipped sum. Both modes accept unipolar
    and bipolar streams and keep accumulation state across timesteps.

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
        self.hw = hw_params(pp_delay=0)

        # whether the addition is scaled (carry-out per acc_bound spikes)
        self.scaled = config['scaled']
        # upper bound of the accumulation counter (= entry count along dim)
        self.acc_bound = 0
        # per-timestep accumulation offset (non-scaled bipolar only)
        self.offset = 0
        # accumulator of the per-timestep partial counts
        self.register_buffer('accumulator', torch.zeros(1, dtype=self.ntype))
        # count of already-emitted output spikes (non-scaled mode)
        if not self.scaled:
            self.register_buffer('out_accumulator', torch.zeros(1, dtype=self.ntype))
        self.is_first_call = True


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
        # out-of-place add broadcasts the (1,) init up to the stream shape on the
        # first timestep (napl accumulator idiom); in-place afterwards (same ntype)
        if self.accumulator.shape == acc_delta.shape:
            self.accumulator.add_(acc_delta)
        else:
            updated = self.accumulator.add(acc_delta)
            self.accumulator.resize_as_(updated).copy_(updated.detach())

        # compare -> stype directly (one cast); sub_/add_ promote the int8 spike
        # to the float32 destination, so results are unchanged
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

import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import uni2bi, bi2uni, signabs, sync_skewed, div_cordiv


class div_iscb(napl_base):
    """
    Divide rate-coded streams with in-stream correlation-based division.

    Use this composed divider when its internal synchronizer, correlated divider,
    and bipolar conversion stages should handle the complete division path. It
    supports both unipolar and bipolar input streams.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import div_iscb

        divider = div_iscb({'polarity': 'unipolar'})
        quotient = divider(torch.tensor([1], dtype=torch.int8),
                           torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Stochastic Division and Square Root via Correlation*.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*.
    """
    def __init__(
        self,
        config={
            'polarity' : 'bipolar',
        }
    ):
        """
        Select the stream encoding for the composed divider.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **name**: Optional instance label.

        The internal synchronizer and conversion widths are fixed by the
        implementation.
        """
        super().__init__(config, ['polarity'], polarity_required=True)
        self.hw = hw_params(pp_delay=0)

        # fix width to optimal 3
        self.sync = sync_skewed({'width': 3})

        # for cordiv kernel, the config is fixed to optimal directly
        # this actually leads to 01 sequence
        self.cordiv_kernel = div_cordiv({'depth': 2, 'generator': 'sobol'})

        if self.polarity == 'bipolar':
            # fix width to optimal 3
            self.signabs_dividend = signabs({'width': 3})
            self.signabs_divisor  = signabs({'width': 3})
            # fix width to optimal 2
            self.bi2uni_dividend = bi2uni({'width': 2})
            self.bi2uni_divisor  = bi2uni({'width': 2})
            # fix width to optimal 3
            self.uni2bi_quotient = uni2bi({'width': 3})


    def _reset(self):
        """
        Reset no additional local state beyond the registered child modules.
        """
        pass


    def forward(self, dividend, divisor):
        """
        Dispatch one timestep through the configured division path.

        Args:
            dividend: Current 0/1 dividend spike tensor.
            divisor: Current 0/1 divisor spike tensor with a compatible shape.

        Returns:
            A quotient spike tensor in the configured polarity. Child converter,
            synchronizer, and divider state is updated by the call.

        **Example:**

        .. code-block:: python

            quotient = divider(torch.tensor([1], dtype=torch.int8),
                               torch.tensor([1], dtype=torch.int8))
        """
        if self.polarity == 'bipolar':
            output = self.bipolar_forward(dividend, divisor)
        else:
            output = self.unipolar_forward(dividend, divisor)
        return output.type(self.stype)


    def bipolar_forward(self, dividend: torch.tensor, divisor: torch.tensor):
        """
        Process one timestep through the bipolar division path.

        Args:
            dividend: Current bipolar-encoded dividend spike tensor.
            divisor: Current bipolar-encoded divisor spike tensor.

        Returns:
            A bipolar-encoded quotient spike tensor. The sign, conversion,
            synchronization, and correlated-division child states are updated.

        **Example:**

        .. code-block:: python

            divider = div_iscb({'polarity': 'bipolar'})
            quotient = divider.bipolar_forward(torch.tensor([1], dtype=torch.int8),
                                                torch.tensor([1], dtype=torch.int8))
        """
        # dividend and divisor are both spike tensors
        sign_dividend, abs_dividend = self.signabs_dividend(dividend)
        sign_divisor, abs_divisor = self.signabs_divisor(divisor)
        uni_abs_dividend = self.bi2uni_dividend(abs_dividend)
        uni_abs_divisor = self.bi2uni_divisor(abs_divisor)
        uni_abs_quotient = self.unipolar_forward(uni_abs_dividend, uni_abs_divisor)
        bi_abs_quotient = self.uni2bi_quotient(uni_abs_quotient)
        bi_quotient = sign_dividend.type(torch.int8) ^ sign_divisor.type(torch.int8) ^ bi_abs_quotient.type(torch.int8)
        return bi_quotient


    def unipolar_forward(self, dividend: torch.tensor, divisor: torch.tensor):
        """
        Process one timestep through the unipolar division path.

        Args:
            dividend: Current unipolar dividend spike tensor.
            divisor: Current unipolar divisor spike tensor.

        Returns:
            A unipolar quotient spike tensor. The synchronizer and correlated
            divider child states are updated.

        **Example:**

        .. code-block:: python

            divider = div_iscb({'polarity': 'unipolar'})
            quotient = divider.unipolar_forward(torch.tensor([1], dtype=torch.int8),
                                                 torch.tensor([1], dtype=torch.int8))
        """
        # dividend and divisor are both spike tensors
        dividend_sync, divisor_sync = self.sync(dividend, divisor)
        quotient = self.cordiv_kernel(dividend_sync, divisor_sync)
        return quotient

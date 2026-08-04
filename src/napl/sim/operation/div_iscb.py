import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import uni2bi, bi2uni, signabs, sync_skewed, div_cordiv


class div_iscb(napl_base):
    r"""
    Divide rate-coded streams with in-stream correlation-based division.

    Use this composed divider when its internal synchronizer, correlated divider,
    and bipolar conversion stages should handle the complete division path. It
    supports both unipolar and bipolar input streams.

    The precise target rate-domain operation is

    .. math::

       y = \frac{x}{d}.

    Let S = sync_skewed, C = div_cordiv, A = signabs, B = bi2uni, and
    U = uni2bi. The composed paths are

    .. math::

       \begin{aligned}
       (x'_t,d'_t) &= S(x_t,d_t), &
       y_t &= C(x'_t,d'_t) && (\text{unipolar}),\\
       (s_x,m_x) &= A(x_t), &
       (s_d,m_d) &= A(d_t),\\
       y_t &= s_x \oplus s_d \oplus
       U\!\left(C\!\left(S(B(m_x),B(m_d))\right)\right)
       && (\text{bipolar}).
       \end{aligned}

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import div_iscb

        divider = div_iscb({'polarity': 'unipolar'})
        quotient = divider(torch.tensor([1], dtype=torch.int8),
                           torch.tensor([1], dtype=torch.int8))

    .. container:: api-references

        .. rubric:: References

        *In-Stream Stochastic Division and Square Root via Correlation*, DAC, 2019.

        *In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing*, IEEE Design and Test, 2021.
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

        #: Skew synchronizer that correlates unipolar dividend and divisor streams.
        self.sync = sync_skewed({'width': 3})

        #: Correlated-divider stage applied after stream synchronization.
        self.cordiv_kernel = div_cordiv({'depth': 2, 'generator': 'sobol'})

        if self.polarity == 'bipolar':
            #: Sign-and-magnitude converter for the bipolar dividend stream.
            self.signabs_dividend = signabs({'width': 3})
            #: Sign-and-magnitude converter for the bipolar divisor stream.
            self.signabs_divisor  = signabs({'width': 3})
            #: Converter from bipolar dividend magnitude to a unipolar stream.
            self.bi2uni_dividend = bi2uni({'width': 3})
            #: Converter from bipolar divisor magnitude to a unipolar stream.
            self.bi2uni_divisor  = bi2uni({'width': 3})
            #: Converter from the unipolar magnitude quotient back to bipolar form.
            self.uni2bi_quotient = uni2bi({'width': 3})
        #: Hardware latency and timing metadata for the composed divider.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'dividend': 'rc', 'divisor': 'rc', 'output': 'rc'}
        self.polarity_io = {'dividend': self.polarity, 'divisor': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


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
            output = self._bipolar_forward(dividend, divisor)
        else:
            output = self._unipolar_forward(dividend, divisor)
        return output.type(self.stype)


    def _bipolar_forward(self, dividend: torch.tensor, divisor: torch.tensor):
        """Process one timestep through the bipolar division path."""
        sign_dividend, abs_dividend = self.signabs_dividend(dividend)
        sign_divisor, abs_divisor = self.signabs_divisor(divisor)
        uni_abs_dividend = self.bi2uni_dividend(abs_dividend)
        uni_abs_divisor = self.bi2uni_divisor(abs_divisor)
        uni_abs_quotient = self._unipolar_forward(uni_abs_dividend, uni_abs_divisor)
        bi_abs_quotient = self.uni2bi_quotient(uni_abs_quotient)
        bi_quotient = sign_dividend.type(torch.int8) ^ sign_divisor.type(torch.int8) ^ bi_abs_quotient.type(torch.int8)
        return bi_quotient


    def _unipolar_forward(self, dividend: torch.tensor, divisor: torch.tensor):
        """Process one timestep through the unipolar division path."""
        dividend_sync, divisor_sync = self.sync(dividend, divisor)
        quotient = self.cordiv_kernel(dividend_sync, divisor_sync)
        return quotient

import torch

from napl.sim.base import napl_base, hw_params
from napl.sim.operation import sigmoid_hard, mul_ugemm, mul_ugemm_sr, add_any
from napl.sim.module.linear import linear
from napl.sim.module._shared import _mgu_run_outlasts_ismul
from loguru import logger


class mgu(napl_base):
    r"""Evaluate a bipolar rate-coded MGU cell one timestep at a time.

    Use this class as the streaming inner cell for :class:`mgu_hub`, or directly
    when input and hidden spike streams are already available. The numeric hidden
    value bound to ``hx_value`` stays fixed for the whole run.

    The precise target is the Minimal Gated Unit recurrence

    .. math::

       f = \sigma\!\left(W_f [h, x] + b_f\right),\qquad
       n = \tanh\!\left(W_n [f \odot h, x] + b_n\right),

    .. math::

       h' = (1 - f) \odot n + f \odot h.

    Each timestep the cell evaluates that recurrence on spike streams with
    saturating unary adders and the hard sigmoid
    :math:`\sigma_h(v) = (v + 1)/2`, so the emitted stream tracks

    .. math::

       h' = \mathrm{clamp}\!\left(n - f \odot n + f \odot h,\, -1,\, 1\right)

    within the stochastic-computing error of the streams.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu

        cell = mgu(torch.zeros(3, 5), torch.zeros(3),
                   torch.zeros(3, 5), torch.zeros(3), torch.zeros(1, 3),
                   {"polarity": "bipolar", "timestep": 128,
                    "generator": "sobol", "width": 10})
        output_spike = cell(torch.ones(1, 2), torch.ones(1, 3))

    .. container:: api-references

        .. rubric:: References

        *uBrain: A Unary Brain Computer Interface*, ISCA, 2022.
    """


    def __init__(self, weight_f, bias_f, weight_n, bias_n, hx_value,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol', 'width': 10, 'depth_ismul': 6}):
        """Construct a streaming MGU from external gate parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight_f** – Forget-gate weight tensor shaped ``(hidden_size, hidden_size + input_size)``.
            - **bias_f** – Forget-gate bias tensor shaped ``(hidden_size,)`` or ``None``.
            - **weight_n** – New-gate weight tensor with the same shape as **weight_f**.
            - **bias_n** – New-gate bias tensor shaped ``(hidden_size,)`` or ``None``.
            - **hx_value** – Fixed numeric hidden value used by conditional spike generation, or ``None`` to bind ``hx_value`` before the run.
            - **config** – Configuration mapping.

              - **polarity**: Stream encoding, which must be ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Encoder stream length, which must be greater than ``2 ** depth_ismul`` so the run outlasts the multiplier's shift-register flush; construction fails otherwise. The default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **width**: Unary-adder accumulator width; the default is ``10``.
              - **depth_ismul**: Register-address width for the non-static forget and new multiplier; the default is ``6``.
              - **name**: Optional instance label.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], optional_key_list=['width', 'depth_ismul'], polarity_required=True)
        if self.polarity != 'bipolar':
            message = f'Invalid polarity: <{self.polarity}>; legal values: <[\'bipolar\']>.'
            logger.error(message)
            raise AssertionError(message)

        ts, gen = config['timestep'], config['generator']
        width = config.get('width', 10)
        self.depth_ismul = config.get('depth_ismul', 6)
        if not _mgu_run_outlasts_ismul(ts, self.depth_ismul):
            message = (f'Invalid timestep: <{ts}>; legal values: an integer greater than '
                       f'<{2 ** self.depth_ismul}>, the number of timesteps that flushing the '
                       f'depth_ismul <{self.depth_ismul}> multiplier shift register consumes, '
                       f'since a run of exactly that length leaves nothing behind.')
            logger.error(message)
            raise AssertionError(message)
        # Registering the run input as a buffer keeps it on the module's device.
        #: Fixed numeric hidden value used by conditional-spike multiplication.
        self.hx_value: torch.Tensor
        self.register_buffer('hx_value', hx_value, persistent=False)
        # Distinct RNG dimensions decorrelate the gates; scale 1 implements hard tanh.
        def lin(w, b, d):
            return linear(w, b, {'polarity': 'bipolar', 'timestep': ts, 'generator': gen,
                                     'dim': d, 'scale': 1, 'width': width})
        #: Streaming linear and hard-tanh block for the forget gate.
        self.fg_ug_tanh = lin(weight_f, bias_f, 3)
        #: Streaming linear and hard-tanh block for the candidate hidden state.
        self.ng_ug_tanh = lin(weight_n, bias_n, 5)
        #: Hard-sigmoid block applied to the forget-gate stream.
        self.fg_sigmoid = sigmoid_hard({'polarity': 'bipolar', 'width': width})
        #: Conditional-spike multiplier for the forget gate and fixed hidden value.
        self.fg_hx_mul = mul_ugemm({'polarity': 'bipolar', 'timestep': ts, 'generator': gen})
        #: Shift-register multiplier for the forget-gate and candidate streams.
        self.fg_ng_mul = mul_ugemm_sr({
            'polarity': 'bipolar',
            'width': self.depth_ismul,
            'generator': gen,
        })
        #: Saturating unary adder that forms the next hidden-state stream.
        self.hy_add = add_any({'polarity': 'bipolar', 'scale': 1, 'width': width})

        # The composed gates, multipliers, and adders are combinational within one timestep.
        #: Hardware latency and timing metadata for the streaming cell.
        self.hw = hw_params(pp_delay=0)

        self.encoding_io = {'input_spike': 'rc', 'hx_spike': 'rc', 'output': 'rc'}
        self.polarity_io = {'input_spike': self.polarity, 'hx_spike': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by this cell.

        This class has no additional local mutable state. The inherited
        ``reset()`` method resets its registered child modules.
        """
        pass


    def forward(self, input_spike, hx_spike):
        """Process one input and hidden-state spike timestep.

        Args:
            input_spike: Current input spike tensor shaped
                ``(batch, input_size)``.
            hx_spike: Current hidden spike tensor shaped
                ``(batch, hidden_size)``.

        Returns:
            Next-hidden-state spike tensor shaped ``(batch, hidden_size)``.

        The call updates registered streaming children and advances this cell's
        ``timestep_cur`` once. ``hx_value`` remains unchanged.
        """
        fg_in = self.fg_ug_tanh(torch.cat((hx_spike, input_spike), dim=1))
        fg = self.fg_sigmoid(fg_in)
        fg_hx = self.fg_hx_mul(fg, self.hx_value)
        ng = self.ng_ug_tanh(torch.cat((fg_hx, input_spike), dim=1))
        fg_ng = self.fg_ng_mul(fg, ng)
        fg_ng_inv = 1 - fg_ng.type(self.stype)
        # Three 0/1 operands sum exactly in ntype, and entry=3 matches their fan-in.
        return self.hy_add(ng + fg_ng_inv + fg_hx, dim=None, entry=3)

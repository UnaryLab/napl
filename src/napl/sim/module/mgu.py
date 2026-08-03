import torch
import math
import torch.nn.functional as F

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger
# Operation imports stay inside __init__ to avoid the module-operation import cycle.


class mgu(napl_base):
    """Evaluate a bipolar rate-coded MGU cell one timestep at a time.

    Use this class as the streaming inner cell for :class:`mgu_hub`, or directly
    when input and hidden spike streams are already available. The two gate linears
    use a saturating ``scale=1`` unary adder, which realizes ``linear + hard tanh``
    in the unary domain. The forget-gate hard sigmoid computes ``(x + 1) / 2``;
    ``fg * hx`` uses conditional-spike generation with fixed ``hx``; ``fg * ng``
    uses XNOR multiplication; and the output applies the same adder to
    ``[ng, 1 - fg * ng, fg * hx]``. The ``hx`` value remains fixed for the run.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu

        cell = mgu(torch.zeros(3, 5), torch.zeros(3),
                   torch.zeros(3, 5), torch.zeros(3), torch.zeros(1, 3),
                   {"polarity": "bipolar", "timestep": 4,
                    "generator": "sobol", "width": 12})
        output_spike = cell(torch.ones(1, 2), torch.ones(1, 3))
    """


    def __init__(self, weight_f, bias_f, weight_n, bias_n, hx_value,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol', 'width': 12, 'depth_ismul': 6}):
        """Construct a streaming MGU from external gate parameters.

        Args:
            weight_f: Forget-gate weight tensor shaped
                ``(hidden_size, hidden_size + input_size)``.
            bias_f: Forget-gate bias tensor shaped ``(hidden_size,)`` or ``None``.
            weight_n: New-gate weight tensor with the same shape as ``weight_f``.
            bias_n: New-gate bias tensor shaped ``(hidden_size,)`` or ``None``.
            hx_value: Fixed numeric hidden value used by conditional spike
                generation.
            config: Configuration mapping with these keys:

                * **polarity** - Must be ``"bipolar"``. Defaults to
                  ``"bipolar"``.
                * **timestep** - Encoder stream length. Defaults to ``256``.
                * **generator** - Number-sequence generator. Defaults to
                  ``"sobol"``.
                * **width** - Unary-adder accumulator width. Defaults to ``12``.
                * **depth_ismul** - Register-address width for the non-static
                  forget/new multiplier. Defaults to ``6``.
                * **name** - Optional instance label. Defaults to ``None``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        from napl.sim.operation import sigmoid_hard, mul_csg, mul_shiftreg, add_any
        from napl.sim.module.linear import linear
        assert self.polarity == 'bipolar', logger.error('mgu requires bipolar.')

        ts, gen = config['timestep'], config['generator']
        width = config.get('width', 12)
        self.depth_ismul = config.get('depth_ismul', 6)
        #: Fixed numeric hidden value used by conditional-spike multiplication.
        self.hx_value = hx_value
        # Distinct RNG dimensions decorrelate the gates; scale 1 implements hard tanh.
        def lin(w, b, d):
            return linear(w, b, {'polarity': 'bipolar', 'timestep': ts, 'generator': gen,
                                     'dim': d, 'scale': 1, 'width': width})
        #: Streaming linear and hard-tanh block for the forget gate.
        self.fg_ug_tanh = lin(weight_f, bias_f, 3)
        #: Streaming linear and hard-tanh block for the candidate hidden state.
        self.ng_ug_tanh = lin(weight_n, bias_n, 5)
        #: Hard-sigmoid block applied to the forget-gate stream.
        self.fg_sigmoid = sigmoid_hard({'polarity': 'bipolar'})
        #: Conditional-spike multiplier for the forget gate and fixed hidden value.
        self.fg_hx_mul = mul_csg({'polarity': 'bipolar', 'timestep': ts, 'generator': gen})
        #: Shift-register multiplier for the forget-gate and candidate streams.
        self.fg_ng_mul = mul_shiftreg({
            'polarity': 'bipolar',
            'width': self.depth_ismul,
            'generator': gen,
        })
        #: Saturating unary adder that forms the next hidden-state stream.
        self.hy_add = add_any({'polarity': 'bipolar', 'scale': 1, 'width': width})


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
        fg_ng_inv = 1 - fg_ng.type(torch.int8)
        # Three 0/1 operands sum exactly in ntype, and entry=3 matches their fan-in.
        return self.hy_add(ng + fg_ng_inv + fg_hx, dim=None, entry=3)

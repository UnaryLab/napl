import torch
import math
import torch.nn.functional as F

from napl.utils import *
from napl.sim.base import napl_base
from loguru import logger

# NB: operation primitives are imported lazily inside __init__ (not at module top),
# because napl.sim.operation.mul_csg imports napl.sim.module.encoder, so a top-level import here
# creates a circular import when napl.sim.operation is loaded before napl.sim.module.


def _init_mgu_params(module, input_size, hidden_size, bias):
    """MGU forget/new-gate weights/biases, truncated-normal init."""
    module.weight_f = torch.nn.Parameter(torch.empty(hidden_size, hidden_size + input_size))
    module.weight_n = torch.nn.Parameter(torch.empty(hidden_size, hidden_size + input_size))
    if bias:
        module.bias_f = torch.nn.Parameter(torch.empty(hidden_size))
        module.bias_n = torch.nn.Parameter(torch.empty(hidden_size))
    else:
        module.register_parameter('bias_f', None)
        module.register_parameter('bias_n', None)
    stdv = 1.0 / math.sqrt(hidden_size)
    for w in [module.weight_f, module.weight_n, module.bias_f, module.bias_n]:
        if w is not None:
            w.data = truncated_normal(w, 0.0, stdv)


class mgu_hard(napl_base):
    """Apply a trainable single-shot MGU cell with bounded hard activations.

    Use this binary-domain cell when intermediate and output values must remain in
    the legal unary range. It uses hard sigmoid and hard tanh by default and does
    not advance the streaming timestep.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hard

        cell = mgu_hard(2, 3)
        hidden = cell(torch.zeros(1, 2))

    References
    ----------
    *Simplified Minimal Gated Unit Variations for RNNs*.
    """
    streaming = False
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
        """Construct the MGU cell and initialize its trainable parameters.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden features.
            bias: Create trainable forget- and new-gate biases when ``True``.
                Defaults to ``True``.
            config: Configuration mapping with **hard**. ``True`` selects hard
                sigmoid and hard tanh; ``False`` selects ``Sigmoid`` and ``Tanh``
                for the gates while retaining the explicit bounding clamps.
                Defaults to ``True``. **name** is an optional instance label and
                defaults to ``None``.
        """
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.hard = config.get('hard', True)
        from napl.sim.operation import sigmoid_hub, tanh_hub
        self.htanh = tanh_hub()        # the explicit hard-tanh clamps (always hard)
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)

    def _reset(self):
        """Reset local recurrent state.

        The cell stores no hidden state between calls, so this hook returns
        ``None`` without changing its trainable parameters.
        """
        pass

    def forward(self, input, hx=None):
        """Compute one MGU recurrence in the binary domain.

        Args:
            input: Tensor of shape ``(batch, input_size)``.
            hx: Optional previous hidden tensor of shape
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            The next hidden tensor of shape ``(batch, hidden_size)``, bounded to
            ``[-1, 1]``.

        The call does not store ``hx`` or change ``timestep_cur``. Gradients flow
        to the input and trainable parameters.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        # forget gate
        fg_ug_in = torch.cat((hx, input), 1)
        fg_in = self.htanh(F.linear(fg_ug_in, self.weight_f, self.bias_f))
        fg = self.fg_sigmoid(fg_in)
        # new gate
        fg_hx = fg * hx
        ng_ug_in = torch.cat((fg_hx, input), 1)
        ng = self.ng_tanh(F.linear(ng_ug_in, self.weight_n, self.bias_n))
        # output: hy = hardtanh(ng*(1-fg) + fg*hx)
        fg_ng = fg * ng
        return self.htanh(ng - fg_ng + fg_hx)


class mgu_hardfxp(napl_base):
    """Apply a quantization-aware MGU cell with hard range bounds.

    Use this single-shot cell to train or evaluate an MGU while rounding operands
    to the configured fixed-point format. ``round_fxp`` supplies the
    straight-through gradient.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hardfxp

        cell = mgu_hardfxp(2, 3, config={"hard": True,
                                        "intwidth": 3, "fracwidth": 4})
        hidden = cell(torch.zeros(1, 2))
    """
    streaming = False
    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True, 'intwidth': 3, 'fracwidth': 4}):
        """Construct the fixed-point MGU and initialize trainable parameters.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden features.
            bias: Create trainable gate biases when ``True``. Defaults to ``True``.
            config: Configuration mapping with these keys:

                * **hard** - Use hard gate activations when ``True``; otherwise
                  use ``Sigmoid`` and ``Tanh``. Defaults to ``True``.
                * **intwidth** - Integer-bit count passed to ``round_fxp``.
                  Defaults to ``3``.
                * **fracwidth** - Fractional-bit count passed to ``round_fxp``.
                  Defaults to ``4``.
                * **name** - Optional instance label. Defaults to ``None``.
        """
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.hard = config.get('hard', True)
        from napl.sim.operation import sigmoid_hub, tanh_hub, round_fxp
        self.htanh = tanh_hub()
        self.trunc = round_fxp({'intwidth': config.get('intwidth', 3), 'fracwidth': config.get('fracwidth', 4)})
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)

    def _reset(self):
        """Reset local recurrent state.

        The cell stores no hidden state between calls, so this hook returns
        ``None`` without changing parameters or quantization settings.
        """
        pass

    def forward(self, input, hx=None):
        """Compute one quantized MGU recurrence.

        Args:
            input: Tensor of shape ``(batch, input_size)``.
            hx: Optional previous hidden tensor of shape
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            The next hidden tensor of shape ``(batch, hidden_size)``, bounded to
            ``[-1, 1]``.

        Operands are quantized during the call. The cell does not store ``hx`` or
        advance ``timestep_cur``.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        t = self.trunc
        # round_fxp is a pure (deterministic STE) function of its input, so each operand is
        # truncated once and the result reused across consumers (halves the redundant
        # quantizations vs. recomputing t(hx)/t(input)/t(fg)/t(fg_hx)/t(ng) at every use).
        t_hx, t_input = t(hx), t(input)
        fg_ug_in = torch.cat((t_hx, t_input), 1)
        fg_in = self.htanh(F.linear(t(fg_ug_in), t(self.weight_f), t(self.bias_f)))
        fg = self.fg_sigmoid(t(fg_in))
        t_fg = t(fg)
        fg_hx = t_fg * t_hx
        t_fg_hx = t(fg_hx)
        ng_ug_in = torch.cat((t_fg_hx, t_input), 1)
        ng = self.ng_tanh(t(F.linear(t(ng_ug_in), t(self.weight_n), t(self.bias_n))))
        t_ng = t(ng)
        fg_ng = t_fg * t_ng
        return self.htanh(t_ng - t(fg_ng) + t_fg_hx)


class mgu(napl_base):
    """Evaluate a bipolar rate-coded MGU cell one timestep at a time.

    Use this class as the streaming inner cell for :class:`mgu_hub`, or directly
    when input and hidden spike streams are already available. The two gate linears use a
    saturating scale-1 unary adder (which realizes linear + hard tanh in the unary domain);
    the forget-gate hard sigmoid is the scaled add (x+1)/2; fg*hx uses conditional-spike-
    generation multiply (hx is a fixed value), fg*ng uses XNOR multiply, and the output is a
    scale-1 unary add of [ng, 1-fg*ng, fg*hx] (= hard tanh of ng*(1-fg)+fg*hx). hx is the
    fixed hidden value for this streaming run.

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
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol', 'width': 12}):
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
                * **name** - Optional instance label. Defaults to ``None``.
        """
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        from napl.sim.operation import sigmoid_hard, mul_csg, mul_and, add_any
        from napl.sim.module.linear import linear
        assert self.polarity == 'bipolar', logger.error('mgu requires bipolar.')

        ts, gen = config['timestep'], config['generator']
        width = config.get('width', 12)
        self.hx_value = hx_value
        # gate linears on distinct weight rng dims; scale=1 makes the adder saturate (hard tanh)
        def lin(w, b, d):
            return linear(w, b, {'polarity': 'bipolar', 'timestep': ts, 'generator': gen,
                                     'dim': d, 'scale': 1, 'width': width})
        self.fg_ug_tanh = lin(weight_f, bias_f, 3)
        self.ng_ug_tanh = lin(weight_n, bias_n, 5)
        self.fg_sigmoid = sigmoid_hard({'polarity': 'bipolar'})
        self.fg_hx_mul = mul_csg({'polarity': 'bipolar', 'timestep': ts, 'generator': gen})  # fg (spike) * hx (value)
        self.fg_ng_mul = mul_and({'polarity': 'bipolar'})                                     # fg (spike) * ng (spike)
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
        # feed the pre-reduced 3-operand sum directly (entry=3 matches the stack size along
        # dim=0), avoiding materializing the stacked tensor; the 0/1 spikes sum to <=3 so the
        # integer add is exact and equals torch.sum(stack, 0, dtype=ntype).
        return self.hy_add(ng + fg_ng_inv + fg_hx, dim=None, entry=3)


class mgu_hub(napl_base):
    """Evaluate an MGU through an internal unary simulation in one call.

    Use this hybrid unary-binary cell when callers provide numeric tensors but
    the MGU computation should run through spike encoders, a streaming
    :class:`mgu`, and progressive decoding. It internally encodes input and hx into spike
    streams, runs mgu over 2**width cycles, and decodes the output with the accuracy
    (progressive-error) metric. Corresponds to mgu_hard with hard activations. Weights are
    external.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hub

        cell = mgu_hub(2, 3, weight_f=torch.zeros(3, 5), bias_f=torch.zeros(3),
                       weight_n=torch.zeros(3, 5), bias_n=torch.zeros(3),
                       config={"polarity": "bipolar", "width": 2,
                               "generator": "sobol"})
        hidden = cell(torch.zeros(1, 2))
    """
    streaming = False
    def __init__(self, input_size, hidden_size, bias=True,
                 weight_f=None, bias_f=None, weight_n=None, bias_n=None,
                 config={'polarity': 'bipolar', 'width': 8, 'generator': 'sobol'}):
        """Configure the hybrid run and attach external gate parameters.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden features.
            bias: Include gate bias in the fan-in calculation. Defaults to
                ``True``.
            weight_f: Forget-gate weight tensor. Defaults to ``None``.
            bias_f: Forget-gate bias tensor. Defaults to ``None``.
            weight_n: New-gate weight tensor. Defaults to ``None``.
            bias_n: New-gate bias tensor. Defaults to ``None``.
            config: Configuration mapping with **polarity** (passed through as
                ``"bipolar"`` internally), **width** (stream exponent, default
                ``8``), and **generator** (default ``"sobol"``). **name** is an
                optional instance label and defaults to ``None``.

        A complete forward call requires compatible weight tensors; construction
        does not create trainable parameters.
        """
        super().__init__(config, [])
        self.input_size, self.hidden_size, self.bias = input_size, hidden_size, bias
        self.width = config.get('width', 8)
        self.generator = config.get('generator', 'sobol')
        self.weight_f, self.bias_f = weight_f, bias_f
        self.weight_n, self.bias_n = weight_n, bias_n
        # accumulator width for the inner linears must hold the fan-in (hidden+input+bias)
        entry = hidden_size + input_size + (1 if bias else 0)
        self.lin_width = max(12, math.ceil(math.log2(entry)) + 2)

    def _reset(self):
        """Reset state owned directly by the hybrid wrapper.

        The wrapper creates its streaming components inside each call and has no
        persistent local run state, so this hook returns ``None``.
        """
        pass

    def forward(self, input, hx=None):
        """Run a complete ``2 ** width``-cycle unary MGU simulation.

        Args:
            input: Numeric tensor shaped ``(batch, input_size)``.
            hx: Optional numeric hidden tensor shaped
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            Decoded next-hidden tensor shaped ``(batch, hidden_size)``.

        The method creates temporary encoders, cell, and accuracy metric. It does
        not store ``hx`` and, as a single-shot module, does not advance
        ``timestep_cur``.
        """
        from napl.sim.module.encoder import encoder
        from napl.sim.metric import accuracy
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        ts = 2 ** self.width

        def enc(d):
            return encoder({'polarity': 'bipolar', 'timestep': ts, 'generator': self.generator, 'dim': d})
        i_enc, h_enc = enc(1), enc(2)
        cell = mgu(self.weight_f, self.bias_f, self.weight_n, self.bias_n, hx,
                       {'polarity': 'bipolar', 'timestep': ts, 'generator': self.generator, 'width': self.lin_width}
                       ).to(input.device)
        acc = accuracy({'polarity': 'bipolar'}).to(input.device)
        for _ in range(ts):
            acc(cell(i_enc(input), h_enc(hx)))
        return acc.spike_value

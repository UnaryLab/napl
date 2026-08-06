import torch
import torch.nn.functional as F

from napl.sim.base import napl_base
from napl.sim.module._shared import _init_mgu_params
from napl.sim.operation import sigmoid_hub, tanh_hub


# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'hard': True,
}


class mgu_hard(napl_base):
    r"""Apply a trainable single-shot MGU cell with bounded hard activations.

    Use this binary-domain cell when intermediate and output values must remain in
    the legal unary range. It uses hard sigmoid and hard tanh by default and does
    not advance the streaming timestep.

    The precise target is the Minimal Gated Unit recurrence

    .. math::

       f = \sigma\!\left(W_f [h, x] + b_f\right),\qquad
       n = \tanh\!\left(W_n [f \odot h, x] + b_n\right),

    .. math::

       h' = (1 - f) \odot n + f \odot h.

    The cell evaluates that recurrence with the hard activations

    .. math::

       \sigma_h(v) = \mathrm{clip}\!\left(\frac{v}{2} + \frac{1}{2},\, 0,\, 1
       \right),\qquad
       \tanh_h(v) = \mathrm{clamp}(v, -1, 1)

    when **hard** is ``True``, and with ``Sigmoid`` and ``Tanh`` otherwise. The
    forget-gate linear and the returned hidden state are clamped to ``[-1, 1]``
    in both cases.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hard

        cell = mgu_hard(2, 3)
        hidden = cell(torch.zeros(1, 2))

    .. container:: api-references

        .. rubric:: References

        *Simplified minimal gated unit variations for recurrent neural networks*, MWSCAS, 2017.
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True, config=_DEFAULT_CONFIG):
        """Construct the MGU cell and initialize its trainable parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **input_size** – Number of input features.
            - **hidden_size** – Number of hidden features.
            - **bias** – Create trainable forget-gate and new-gate biases when ``True``; the default is ``True``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **hard**: Use hard sigmoid and hard tanh when ``True``, or ``Sigmoid`` and ``Tanh`` while keeping the bounding clamps when ``False``; the default is ``True``.
              - **name**: Optional instance label.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether the forget and new gates include trainable biases.
        self.bias = bias
        #: Whether the forget and new gates use hard activations.
        self.hard = cfg['hard']
        #: Hard-tanh operator that bounds intermediate and output values.
        self.htanh = tanh_hub()
        #: Activation applied to the forget gate.
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        #: Activation applied to the candidate hidden state.
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


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
        fg_ug_in = torch.cat((hx, input), 1)
        fg_in = self.htanh(F.linear(fg_ug_in, self.weight_f, self.bias_f))
        fg = self.fg_sigmoid(fg_in)
        fg_hx = fg * hx
        ng_ug_in = torch.cat((fg_hx, input), 1)
        ng = self.ng_tanh(F.linear(ng_ug_in, self.weight_n, self.bias_n))
        fg_ng = fg * ng
        return self.htanh(ng - fg_ng + fg_hx)

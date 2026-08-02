import torch
import torch.nn.functional as F

from napl.sim.base import napl_base
from napl.sim.module.rnn import _init_mgu_params
# Operation imports stay inside __init__ to avoid the module-operation import cycle.


class mgu_hardnua(napl_base):
    """Apply a trainable MGU without unary-range clamps around linear stages.

    Use this single-shot cell to match the non-unary-aware UnarySim variant or to
    study the effect of removing the range clamps from :class:`mgu_hard`. The
    forget-gate linear input and output omit the hard-tanh clamps, so
    intermediate values and ``hy`` may leave the legal unary range. The cell is
    single-shot and trainable.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hardnua

        cell = mgu_hardnua(2, 3)
        hidden = cell(torch.zeros(1, 2))
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True, config={'hard': True}):
        """Construct the non-unary-aware cell and initialize its parameters.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden features.
            bias: Create trainable gate biases when ``True``. Defaults to ``True``.
            config: Configuration mapping with **hard**. ``True`` uses hard
                sigmoid and hard tanh; ``False`` uses ``Sigmoid`` and ``Tanh``.
                Defaults to ``True``. **name** is an optional instance label and
                defaults to ``None``.
        """
        super().__init__(config, [])
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether the forget and new gates include trainable biases.
        self.bias = bias
        #: Whether the forget and new gates use hard activations.
        self.hard = config.get('hard', True)
        from napl.sim.operation import sigmoid_hub, tanh_hub
        #: Activation applied to the forget gate.
        self.fg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        #: Activation applied to the candidate hidden state.
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()
        _init_mgu_params(self, input_size, hidden_size, bias)


    def _reset(self):
        """Reset local recurrent state.

        The cell stores no hidden state between calls, so this hook returns
        ``None`` without changing trainable parameters.
        """
        pass


    def forward(self, input, hx=None):
        """Compute one unclamped MGU recurrence.

        Args:
            input: Tensor shaped ``(batch, input_size)``.
            hx: Optional previous hidden tensor shaped
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            Next hidden tensor shaped ``(batch, hidden_size)``. Values may leave
            ``[-1, 1]``.

        The call does not store ``hx`` or advance ``timestep_cur``.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        fg = self.fg_sigmoid(F.linear(torch.cat((hx, input), 1), self.weight_f, self.bias_f))
        fg_hx = fg * hx
        ng = self.ng_tanh(F.linear(torch.cat((fg_hx, input), 1), self.weight_n, self.bias_n))
        return ng - fg * ng + fg_hx

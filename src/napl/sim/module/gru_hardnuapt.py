import torch
import torch.nn.functional as F

from napl.sim.base import napl_base
from napl.sim.operation import sigmoid_hub, tanh_hub
from napl.sim.module._shared import _init_mgu_pt_params


# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'hard': True,
}


class gru_hardnuapt(napl_base):
    r"""Apply a PyTorch-layout GRU cell with optional hard activations.

    Use this single-shot cell to match UnarySim ``HardGRUCellNUAPT`` while keeping
    the standard three-chunk ``GRUCell`` parameter layout. In the binary domain,
    it replaces sigmoid with scaled hard sigmoid and tanh with hard tanh.
    Intermediate values are not bounded to the legal unary range. The cell is
    single-shot and trainable.

    The precise target is the GRU recurrence in the PyTorch three-chunk layout,
    with :math:`g^i = W_{ih} x + b_{ih}` and :math:`g^h = W_{hh} h + b_{hh}` each
    split into reset, update, and new thirds,

    .. math::

       r = \sigma\!\left(g^i_r + g^h_r\right),\qquad
       z = \sigma\!\left(g^i_z + g^h_z\right),\qquad
       n = \tanh\!\left(g^i_n + r \odot g^h_n\right),

    .. math::

       h' = (1 - z) \odot n + z \odot h.

    The cell evaluates that recurrence exactly, substituting the hard
    activations,

    .. math::

       \sigma_h(v) = \mathrm{clip}\!\left(\frac{v}{2} + \frac{1}{2},\, 0,\, 1
       \right),\qquad
       \tanh_h(v) = \mathrm{clamp}(v, -1, 1),

    when **hard** is ``True``, and the exact ``Sigmoid`` and ``Tanh`` otherwise.
    Apart from the bound that :math:`\tanh_h` places on :math:`n`, no stage
    carries a range clamp, so :math:`h'` may leave ``[-1, 1]``.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import gru_hardnuapt

        cell = gru_hardnuapt(2, 3)
        hidden = cell(torch.zeros(1, 2))
    """
    #: Whether calls process one stream timestep; this cell is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True, config=_DEFAULT_CONFIG):
        """Construct the GRU cell and initialize trainable parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **input_size** – Number of input features.
            - **hidden_size** – Number of hidden features.
            - **bias** – Create input-hidden and hidden-hidden biases when ``True``; the default is ``True``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **hard**: Use scaled hard sigmoid and hard tanh when ``True``, or ``Sigmoid`` and ``Tanh`` when ``False``; the default is ``True``.
              - **name**: Optional instance label.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether the cell includes trainable input and hidden biases.
        self.bias = bias
        #: Whether the reset, update, and new gates use hard activations.
        self.hard = cfg['hard']
        #: Activation applied to the reset gate.
        self.rg_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        #: Activation applied to the update gate.
        self.ug_sigmoid = sigmoid_hub() if self.hard else torch.nn.Sigmoid()
        #: Activation applied to the candidate hidden state.
        self.ng_tanh = tanh_hub() if self.hard else torch.nn.Tanh()

        # PyTorch's three-chunk parameter layout stores reset, update, then new rows.
        _init_mgu_pt_params(self, input_size, hidden_size, bias, 3)
        #: Trainable input-hidden weight shaped ``(3 * hidden_size, input_size)``, in reset, update, new row order.
        self.weight_ih: torch.nn.Parameter
        #: Trainable hidden-hidden weight shaped ``(3 * hidden_size, hidden_size)``, in reset, update, new row order.
        self.weight_hh: torch.nn.Parameter
        #: Trainable input-hidden bias shaped ``(3 * hidden_size,)``, or ``None`` when **bias** is ``False``.
        self.bias_ih: torch.nn.Parameter
        #: Trainable hidden-hidden bias shaped ``(3 * hidden_size,)``, or ``None`` when **bias** is ``False``.
        self.bias_hh: torch.nn.Parameter

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset local recurrent state.

        The cell stores no hidden state between calls, so this hook returns
        ``None`` without changing trainable parameters.
        """
        pass


    def forward(self, input, hx=None):
        """Compute one GRU recurrence.

        Args:
            input: Tensor shaped ``(batch, input_size)``.
            hx: Optional previous hidden tensor shaped
                ``(batch, hidden_size)``. Defaults to zeros.

        Returns:
            Next hidden tensor shaped ``(batch, hidden_size)``. Because this
            variant is not unary-aware, values may leave ``[-1, 1]``.

        The call does not store ``hx`` or advance ``timestep_cur``.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        gate_i = F.linear(input, self.weight_ih, self.bias_ih)
        gate_h = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gate_i.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)

        rg = self.rg_sigmoid(i_r + h_r)
        ug = self.ug_sigmoid(i_z + h_z)
        ng = self.ng_tanh(i_n + rg * h_n)
        return (1 - ug) * ng + ug * hx

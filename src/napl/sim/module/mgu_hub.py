import torch
import math
import torch.nn.functional as F

from napl.sim.base import napl_base
from loguru import logger
# Operation imports stay inside __init__ to avoid the module-operation import cycle.


class mgu_hub(napl_base):
    """Evaluate an MGU through an internal unary simulation in one call.

    Use this hybrid unary-binary cell when callers provide numeric tensors but
    the MGU computation should run through spike encoders, a streaming
    :class:`mgu`, and progressive decoding. It internally encodes ``input`` and
    ``hx`` into spike streams, runs :class:`mgu` for ``2 ** width`` cycles, and
    decodes the output with the ``accuracy`` metric. Its gate equations match
    :class:`mgu_hard` with hard activations, using caller-provided weights.

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
    #: Whether calls process one stream timestep; this wrapper is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True,
                 weight_f=None, bias_f=None, weight_n=None, bias_n=None,
                 config={'polarity': 'bipolar', 'width': 8, 'generator': 'sobol', 'depth_ismul': 6}):
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
                ``8``), **generator** (default ``"sobol"``), and
                **depth_ismul** (non-static multiplier register-address width,
                default ``6``). **name** is an optional instance label and
                defaults to ``None``.

        A complete forward call requires compatible weight tensors; construction
        does not create trainable parameters.
        """
        super().__init__(config, [])
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether gate fan-in includes a bias term.
        self.bias = bias
        #: Base-two exponent of the internal unary stream length.
        self.width = config.get('width', 8)
        #: Number-sequence generator used by the internal encoders.
        self.generator = config.get('generator', 'sobol')
        #: Register-address width for the internal non-static multiplier.
        self.depth_ismul = config.get('depth_ismul', 6)
        #: Caller-provided forget-gate weight tensor.
        self.weight_f = weight_f
        #: Caller-provided forget-gate bias tensor, or ``None``.
        self.bias_f = bias_f
        #: Caller-provided candidate-gate weight tensor.
        self.weight_n = weight_n
        #: Caller-provided candidate-gate bias tensor, or ``None``.
        self.bias_n = bias_n
        # The inner accumulator must hold the hidden, input, and optional bias fan-in.
        entry = hidden_size + input_size + (1 if bias else 0)
        #: Accumulator width used by each internal streaming linear layer.
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
        from napl.sim.module.mgu import mgu
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        ts = 2 ** self.width

        def enc(d):
            return encoder({'polarity': 'bipolar', 'timestep': ts, 'generator': self.generator, 'dim': d})
        i_enc, h_enc = enc(1), enc(2)
        cell = mgu(self.weight_f, self.bias_f, self.weight_n, self.bias_n, hx,
                       {'polarity': 'bipolar', 'timestep': ts, 'generator': self.generator,
                        'width': self.lin_width, 'depth_ismul': self.depth_ismul}
                       ).to(input.device)
        acc = accuracy({'polarity': 'bipolar'}).to(input.device)
        for _ in range(ts):
            acc(cell(i_enc(input), h_enc(hx)))
        return acc.spike_value

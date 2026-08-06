import torch
import math

from napl.sim.base import napl_base
from napl.sim.operation import decode
from napl.sim.operation.encode import encode
from napl.sim.module.mgu import mgu


# Single source for every optional key: the signature default and the per-key fallback.
_DEFAULT_CONFIG = {
    'polarity': 'bipolar',
    'width': 8,
    'generator': 'sobol',
    'depth_ismul': 6,
}


class mgu_hub(napl_base):
    r"""Evaluate an MGU through an internal unary simulation in one call.

    Use this hybrid unary-binary cell when callers provide numeric tensors but
    the MGU computation should run through a full bipolar unary simulation. Gate
    weights and biases come from the caller, and the internal cell wraps each
    supplied tensor as a trainable parameter.

    The precise target is the Minimal Gated Unit recurrence

    .. math::

       f = \sigma\!\left(W_f [h, x] + b_f\right),\qquad
       n = \tanh\!\left(W_n [f \odot h, x] + b_n\right),

    .. math::

       h' = (1 - f) \odot n + f \odot h.

    The returned value is the decoded rate of a :math:`2^{\text{width}}`-cycle
    unary run of :class:`mgu`, so it carries both the saturating-adder behavior
    of that cell and the stochastic-computing error of the run length.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import mgu_hub

        cell = mgu_hub(2, 3, weight_f=torch.zeros(3, 5), bias_f=torch.zeros(3),
                       weight_n=torch.zeros(3, 5), bias_n=torch.zeros(3),
                       config={"polarity": "bipolar", "width": 2,
                               "generator": "sobol"})
        hidden = cell(torch.zeros(1, 2))

    .. container:: api-references

        .. rubric:: References

        *uBrain: A Unary Brain Computer Interface*, ISCA, 2022.
    """
    #: Whether calls process one stream timestep; this wrapper is single-shot.
    streaming = False


    def __init__(self, input_size, hidden_size, bias=True,
                 weight_f=None, bias_f=None, weight_n=None, bias_n=None,
                 config=_DEFAULT_CONFIG):
        """Configure the hybrid run and attach external gate parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **input_size** – Number of input features.
            - **hidden_size** – Number of hidden features.
            - **bias** – Include gate bias in the fan-in calculation when ``True``; the default is ``True``.
            - **weight_f** – Forget-gate weight tensor shaped ``(hidden_size, hidden_size + input_size)``. Required: construction fails when it is ``None``.
            - **bias_f** – Forget-gate bias tensor shaped ``(hidden_size,)``, or ``None`` for no bias; the default is ``None``.
            - **weight_n** – New-gate weight tensor with the same shape as **weight_f**. Required: construction fails when it is ``None``.
            - **bias_n** – New-gate bias tensor shaped ``(hidden_size,)``, or ``None`` for no bias; the default is ``None``.
            - **config** – Configuration mapping. Omitted keys fall back to the same defaults.

              - **polarity**: Recorded stream encoding; the internal run is always bipolar. The attribute takes the value present in **config**, and stays ``None`` when the mapping omits the key.
              - **width**: Base-two exponent of the internal stream length; the default is ``8``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **depth_ismul**: Register-address width of the non-static multiplier; the default is ``6``.
              - **name**: Optional instance label.

        Both weight tensors are required. The internal cell registers each
        supplied weight, and each supplied bias, as a trainable parameter.
        """
        super().__init__(config, [], optional_key_list=list(_DEFAULT_CONFIG))
        cfg = {**_DEFAULT_CONFIG, **config}
        #: Number of features in each input vector.
        self.input_size = input_size
        #: Number of features in each hidden-state vector.
        self.hidden_size = hidden_size
        #: Whether gate fan-in includes a bias term.
        self.bias = bias
        #: Base-two exponent of the internal unary stream length.
        self.width = cfg['width']
        #: Number-sequence generator used by the internal encoders.
        self.generator = cfg['generator']
        #: Register-address width for the internal non-static multiplier.
        self.depth_ismul = cfg['depth_ismul']
        #: Forget-gate weight tensor given at construction, shared with the internal cell; rebinding it later leaves the run unchanged.
        self.weight_f = weight_f
        #: Forget-gate bias tensor given at construction, or ``None``, shared with the internal cell; rebinding it later leaves the run unchanged.
        self.bias_f = bias_f
        #: Candidate-gate weight tensor given at construction, shared with the internal cell; rebinding it later leaves the run unchanged.
        self.weight_n = weight_n
        #: Candidate-gate bias tensor given at construction, or ``None``, shared with the internal cell; rebinding it later leaves the run unchanged.
        self.bias_n = bias_n
        # The inner accumulator must hold the hidden, input, and optional bias fan-in.
        entry = hidden_size + input_size + (1 if bias else 0)
        #: Accumulator width used by each internal streaming linear layer.
        self.lin_width = max(12, math.ceil(math.log2(entry)) + 2)

        ts = 2 ** self.width
        #: Encoder that converts the numeric input to spikes.
        self.i_encoder = encode({'polarity': 'bipolar', 'timestep': ts,
                                 'generator': self.generator, 'dim': 1})
        #: Encoder that converts the numeric hidden state to spikes.
        self.h_encoder = encode({'polarity': 'bipolar', 'timestep': ts,
                                 'generator': self.generator, 'dim': 2})
        # The cell binds its hidden value at construction; forward rebinds it per call.
        #: Streaming MGU cell driven by the two encoders.
        self.cell = mgu(self.weight_f, self.bias_f, self.weight_n, self.bias_n, None,
                        {'polarity': 'bipolar', 'timestep': ts, 'generator': self.generator,
                         'width': self.lin_width, 'depth_ismul': self.depth_ismul})
        #: Decoder that averages the emitted hidden-state spikes.
        self.decoder = decode({'polarity': 'bipolar', 'timestep': ts})

        self.encoding_io = {}
        self.polarity_io = {}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by the hybrid wrapper.

        This class has no extra local state. The inherited ``reset()`` method
        restarts the two encoders, the streaming cell, and the decoder.
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

        The method restarts its encoders, cell, and decoder, and binds ``hx`` into
        the cell for the run. It does not store ``hx`` and, as a single-shot
        module, does not advance ``timestep_cur``.
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        ts = 2 ** self.width

        self.to(input.device)
        # The cell multiplies by a hidden value held fixed for the whole run.
        self.cell.hx_value = hx
        self.reset()
        for _ in range(ts):
            self.decoder(self.cell(self.i_encoder(input), self.h_encoder(hx)))
        return self.decoder.spike_value

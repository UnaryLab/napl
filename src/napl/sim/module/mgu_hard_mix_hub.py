from napl.sim.base import napl_base, napl_sim_timesteps
from napl.sim.module._shared import _CORE_DIMS, _hub_core_config
from napl.sim.operation import decode, encode
from .mgu_hard_mix import mgu_hard_mix
from loguru import logger


class mgu_hard_mix_hub(napl_base):
    r"""Evaluate a hybrid unary-binary MGU cell with numeric input and output ports.

    Use this cell when the caller works in the binary domain and wants the
    streaming MGU cell to handle the whole stream. Encoding happens only at the
    input and hidden ports and decoding only at the output port, so
    :class:`~napl.sim.module.mgu_hard_mix` runs purely on spikes in between and
    the hidden state never leaves the spike domain during a run. The input port
    and the hidden port draw on distinct Sobol dimensions, so their streams
    decorrelate.

    Each call runs a complete **timestep**-cycle run on a fixed numeric input
    and returns the decoded next hidden state of the hard-activation Minimal
    Gated Unit,

    .. math::

       h' = \mathrm{clamp}\!\left(n - f \odot n + f \odot h,\, -1,\, 1\right),

    within the stochastic-computing error of the streams. The numeric hidden
    value bound at construction is held fixed for the whole run and is
    re-encoded every timestep, so it never leaves the spike domain. Stepping the
    run one timestep at a time through ``forward_timestep()`` returns the same
    value progressively refined.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.module import mgu_hard_mix_hub

        cell = mgu_hard_mix_hub(torch.zeros(3, 5), torch.zeros(3),
                                torch.zeros(3, 5), torch.zeros(3),
                                torch.zeros(1, 3))
        output_value = cell(torch.ones(1, 2))

    .. container:: api-references

        .. rubric:: References

        *uBrain: A Unary Brain Computer Interface*, ISCA, 2022.
    """
    #: Whether each call is one streaming timestep; one hub call is a whole run.
    streaming = False


    def __init__(
            self,
            weight_f,
            bias_f,
            weight_n,
            bias_n,
            hx_value,
            codec_config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'dim': 1,
            },
            core_config={
                'width': 10,
                'depth_ismul': 6,
            }
        ):
        """Construct the port codecs and the streaming cell from external gate parameters.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight_f** – Forget-gate weight tensor shaped ``(hidden_size, hidden_size + input_size)``.
            - **bias_f** – Forget-gate bias tensor shaped ``(hidden_size,)`` or ``None``.
            - **weight_n** – New-gate weight tensor with the same shape as **weight_f**.
            - **bias_n** – New-gate bias tensor shaped ``(hidden_size,)`` or ``None``.
            - **hx_value** – Numeric hidden value shaped ``(batch, hidden_size)``, encoded once per timestep and held fixed for the run.
            - **codec_config** – Configuration mapping for the numeric ports.

              - **polarity**: Stream encoding, which must be ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Stream length, which must be greater than ``2 ** depth_ismul``; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **dim**: Input-encoder number-sequence dimension, with the hidden encoder on the next dimension. Both must stay clear of the gate dimensions ``3`` to ``6``; the default is ``1``.
              - **seed**: Optional integer seed for the ``lfsr`` and ``sys`` generators.
              - **taps**: Optional LFSR feedback-tap list.
              - **name**: Optional instance label.

            - **core_config** – Configuration mapping passed to :class:`~napl.sim.module.mgu_hard_mix`, accepting its **width** and **depth_ismul** keys. Its **polarity**, **timestep**, and **generator** come from **codec_config** and are rejected here.

        The gate parameters belong to the streaming cell and are reachable
        through ``cell.core``.

        .. warning::

            Optimizers silently skip the gate ``Parameter`` objects. The spike
            comparison is not differentiable, so no gradient ever reaches them,
            ``.grad`` stays ``None``, and ``SGD.step()`` leaves the values
            bit-identical.
        """
        super().__init__(codec_config, ['polarity', 'timestep', 'generator'],
                         optional_key_list=['dim', 'seed', 'taps'], polarity_required=True)

        #: Number of timesteps in one complete run of this cell.
        self.timestep = codec_config['timestep']

        merged_core_config = _hub_core_config('mgu_hard_mix_hub', codec_config, core_config)
        #: Number-sequence dimension used by the input encoder.
        self.dim = codec_config.get('dim', 1)
        #: Number-sequence dimension used by the hidden-state encoder.
        self.dim_hx = self.dim + 1
        for dim in (self.dim, self.dim_hx):
            if dim in _CORE_DIMS:
                message = (f'Invalid dim: <{self.dim}>; legal values: any dimension placing the '
                           f'input dim and the hidden dim <{self.dim_hx}> outside the gate dims '
                           f'<{list(_CORE_DIMS)}>, so the port streams decorrelate from the gate '
                           f'weight streams.')
                logger.error(message)
                raise AssertionError(message)

        #: Streaming MGU cell that owns the gate parameters and the hidden value.
        self.core = mgu_hard_mix(weight_f, bias_f, weight_n, bias_n, hx_value,
                                 merged_core_config)
        #: Input-port encoder turning numeric inputs into the cell's input stream.
        self.reference_encode_input = encode(dict(codec_config, dim=self.dim))
        #: Hidden-port encoder turning the numeric hidden value into the cell's hidden stream.
        self.reference_encode_hx = encode(dict(codec_config, dim=self.dim_hx))
        #: Output-port decoder holding the progressively decoded next hidden state.
        self.decoder = decode(dict(codec_config))

        # Encoding, the cell, and decoding are combinational within one timestep.
        #: Hardware latency and timing metadata for the wrapped cell.
        self.hw.pp_delay = self.core.hw.pp_delay
        #: Whether the RTL counterpart must hold its own encoder, true when any part does.
        self.internal_encode = any(part.internal_encode for part in self.children())

        #: Empty, since the numeric ports carry no stream encoding.
        self.encoding_io = {}
        self.polarity_io = {'input': self.polarity, 'output': self.polarity}
        self.correlation_i = {}


    def _reset(self):
        """Reset state owned directly by this cell.

        This class holds no local mutable state. The inherited ``reset()``
        method restarts both port encoders, the streaming cell, and the output
        decoder before this hook returns ``None``.
        """
        pass


    @napl_sim_timesteps
    def forward(self, input):
        """Run a complete **timestep**-cycle run for a fixed numeric input.

        Args:
            input: Numeric tensor shaped ``(batch, input_size)`` in
                ``[-1, 1]``.

        Returns:
            The decoded next hidden state shaped ``(batch, hidden_size)``,
            taken after the final timestep of the run.

        The call is a fresh run: it resets this cell and its children, holds the
        numeric input fixed for **timestep** timesteps, and counts those
        timesteps on the streaming encoders, cell, and decoder rather than on
        this cell wrapper's ``timestep_cur``. Any state a caller set beforehand is
        discarded, so repeating the call on the same input returns the same
        value. The numeric hidden value bound at construction stays unchanged,
        and its stream is generated fresh each timestep instead of being decoded
        and re-encoded. Call ``forward_timestep(input)`` instead to advance one
        timestep and read the progressively refined value.
        """
        reference_encode_input_bit = self.reference_encode_input(input)
        reference_encode_hx_bit = self.reference_encode_hx(self.core.hx_value)
        self.decoder(self.core(reference_encode_input_bit, reference_encode_hx_bit))
        return self.decoder.spike_value

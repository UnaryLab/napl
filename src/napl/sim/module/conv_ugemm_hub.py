from napl.sim.base import napl_base
from napl.sim.module._shared import _CORE_DIM, _hub_core_config
from napl.sim.operation import decode, encode
from .conv_ugemm import conv_ugemm
from loguru import logger


class conv_ugemm_hub(napl_base):
    r"""Apply a hybrid unary-binary convolution with numeric input and output ports.

    Use this layer when the caller works in the binary domain and wants the
    streaming uGEMM convolution to handle the whole stream. Encoding happens
    only at the input port and decoding only at the output port, so
    :class:`~napl.sim.module.conv_ugemm` runs purely on spikes in between.

    Each call processes one timestep and returns the progressively decoded

    .. math::

       y = \frac{\mathrm{conv2d}(x, W) + b}{s},

    with **scale** :math:`s` defaulting to the kernel fan-in plus the bias. The
    returned value refines toward that target over a run of **timestep** calls,
    within the stochastic-computing error of the streams. The numeric input is
    held fixed for the whole run.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import conv_ugemm_hub

        layer = conv_ugemm_hub(torch.zeros(2, 1, 3, 3), padding=1)
        for _ in range(256):
            output_value = layer(torch.ones(1, 1, 4, 4))

    .. container:: api-references

        .. rubric:: References

        *uGEMM: Unary Computing Architecture for GEMM Applications*, ISCA, 2020.

        *uBrain: A Unary Brain Computer Interface*, ISCA, 2022.
    """


    def __init__(
            self,
            weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            codec_config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'dim': 2,
            },
            core_config={
                'scale': None,
                'width': 8,
            }
        ):
        """Construct the input encoder, the streaming core, and the output decoder.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight** – Numeric convolution weight shaped ``(out_channels, in_channels, kernel_height, kernel_width)``.
            - **bias** – Optional numeric tensor shaped ``(out_channels,)``; the default is ``None``.
            - **stride** – Convolution stride; the default is ``1``.
            - **padding** – Symmetric zero padding; the default is ``0``.
            - **dilation** – Kernel dilation; the default is ``1``.
            - **codec_config** – Configuration mapping for the numeric ports.

              - **polarity**: Stream encoding, ``"unipolar"`` or ``"bipolar"``; the default is ``"bipolar"``.
              - **timestep**: Positive stream length; the default is ``256``.
              - **generator**: Number-sequence generator name; the default is ``"sobol"``.
              - **dim**: Input-encoder number-sequence dimension, which must differ from the core's weight dimension ``1`` so the input stream decorrelates from the weight stream; the default is ``2``.
              - **seed**: Optional integer seed for the ``lfsr`` and ``sys`` generators.
              - **taps**: Optional LFSR feedback-tap list.
              - **name**: Optional instance label.

            - **core_config** – Configuration mapping passed to :class:`~napl.sim.module.conv_ugemm`, accepting its **scale** and **width** keys. Its **polarity**, **timestep**, and **generator** come from **codec_config** and are rejected here.

        Weight and bias belong to the streaming core and are reachable as
        ``layer.core.weight`` and ``layer.core.bias``.

        .. warning::

            Optimizers silently skip those ``Parameter`` objects. The spike
            comparison is not differentiable, so no gradient ever reaches them,
            ``.grad`` stays ``None``, and ``SGD.step()`` leaves the values
            bit-identical.
        """
        super().__init__(codec_config, ['polarity', 'timestep', 'generator'],
                         optional_key_list=['dim', 'seed', 'taps'], polarity_required=True)

        #: Number of timesteps in one complete run of this layer.
        self.timestep = codec_config['timestep']

        merged_core_config = _hub_core_config('conv_ugemm_hub', codec_config, core_config)
        #: Number-sequence dimension used by the input encoder.
        self.dim = codec_config.get('dim', 2)
        if self.dim == _CORE_DIM:
            message = (f'Invalid dim: <{self.dim}>; legal values: any dimension other than the '
                       f'core dim <{_CORE_DIM}>, so the input stream decorrelates from the '
                       f'weight stream.')
            logger.error(message)
            raise AssertionError(message)

        #: Input-port encoder turning numeric values into the core's input stream.
        self.reference_encode = encode(dict(codec_config, dim=self.dim))
        #: Streaming uGEMM convolution that owns the weight and bias parameters.
        self.core = conv_ugemm(weight, bias, stride=stride, padding=padding,
                               dilation=dilation, config=merged_core_config)
        #: Output-port decoder holding the progressively decoded result.
        self.decoder = decode(dict(codec_config))

        #: Divisor implemented by the streaming core's unary adder.
        self.scale = self.core.scale
        #: Parallel-count fan-in of the streaming core, including the bias when present.
        self.entry = self.core.entry

        # Encoding, the core, and decoding are combinational within one timestep.
        #: Hardware latency and timing metadata for the wrapped layer.
        self.hw.pp_delay = self.core.hw.pp_delay

        #: Empty, since the numeric ports carry no stream encoding.
        self.encoding_io = {}
        self.polarity_io = {'input_value': self.polarity, 'output': self.polarity}
        self.correlation_i = {}
        self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by this layer.

        This class holds no local mutable state. The inherited ``reset()``
        method restarts the input encoder, the streaming core, and the output
        decoder before this hook returns ``None``.
        """
        pass


    def forward(self, input_value):
        """Process one timestep for a fixed numeric NCHW input.

        Args:
            input_value: Numeric tensor shaped
                ``(batch, in_channels, height, width)``, in ``[0, 1]`` for
                unipolar or ``[-1, 1]`` for bipolar streams.

        Returns:
            The progressively decoded output tensor shaped
            ``(batch, out_channels, output_height, output_width)``.

        The call advances the encoder, the core, the decoder, and this layer's
        ``timestep_cur`` once. Pass the same numeric input on every call of a
        run; the returned value refines as the run proceeds.
        """
        reference_encode_bit = self.reference_encode(input_value)
        self.decoder(self.core(reference_encode_bit))
        return self.decoder.spike_value

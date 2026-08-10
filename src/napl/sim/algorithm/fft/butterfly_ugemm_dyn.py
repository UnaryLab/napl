import torch
from loguru import logger

from napl.sim.operation import add_any_dyn

from .butterfly_ugemm import butterfly_ugemm


class butterfly_ugemm_dyn(butterfly_ugemm):
    r"""Evaluate a fully streaming radix-2 butterfly with a runtime adder scale.

    Use this class when a butterfly must stay in the spike domain on every data
    port and the carry scale is chosen on each timestep. Like
    :class:`butterfly_ugemm`, it takes bipolar 0/1 spike tensors and returns
    bipolar 0/1 spike tensors and holds no encoder or decoder, so it is not the
    numeric-port class :class:`fft_dyn_hub`.

    With the scale held constant from reset through a run, the output streams
    encode the exact butterfly divided by that scale, reported after each call
    as :attr:`compensation`. ``add_any_dyn`` preserves accumulator mass at the
    scale active when a spike fires, so a scale may change without reset, but
    the stream then mixes segments produced under different scales and no single
    compensation factor describes the whole run. Only bipolar encoding is
    supported.

    As in :class:`butterfly_ugemm`, instances built from the same ``mul_config``
    share a bit-identical weight sequence and decorrelate through the
    conditional sequence-index advance on distinct data.
    """


    def __init__(self, twiddle_real, twiddle_imag, mul_config, add_config):
        """Configure the constant twiddle factor, the multiplier, and the dynamic adder.

        Args:
            twiddle_real: Non-empty 1-D tensor holding the real part of the
                constant twiddle factor, one entry per butterfly lane.
            twiddle_imag: Imaginary part of the same twiddle factor, with the
                same length as ``twiddle_real``.
            mul_config: Bipolar conditional-spike multiplier configuration
                containing ``polarity``, ``timestep``, and ``generator``.
            add_config: Dynamic adder configuration containing ``polarity``,
                positive integer ``scale_max``, and ``width``. The width bound
                documented on :class:`butterfly_ugemm` is checked against
                ``scale_max``.
        """
        if 'scale_max' not in add_config:
            message = 'Missing key <scale_max> in the dynamic adder configuration.'
            logger.error(message)
            raise AssertionError(message)
        scale_max = add_config['scale_max']
        if type(scale_max) is not int or scale_max < 1:
            message = (
                f'butterfly_ugemm_dyn scale_max must be a positive int: '
                f'got <{scale_max}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        static_add_config = dict(add_config)
        static_add_config['scale'] = static_add_config.pop('scale_max')
        super().__init__(twiddle_real, twiddle_imag, mul_config, static_add_config)
        #: Dynamic scaled adder used by the butterfly output paths.
        self.add_y = add_any_dyn(dict(add_config))
        #: Largest runtime carry scale accepted by this butterfly.
        self.scale_max = self.add_y.scale_max
        #: Factor the output streams of the most recent call are divided by.
        self.compensation = None


    def _reset(self):
        """Clear the reported compensation after inherited children reset.

        The reported factor returns to ``None`` because no call has selected a
        scale yet. This hook returns ``None``.
        """
        super()._reset()
        self.compensation = None


    def forward(self, x0r, x0i, x1r, x1i, scale):
        """Process one butterfly timestep at the requested carry scale.

        Args:
            x0r: Real-part spike tensor of the first complex input, shaped
                ``(lane, ...)``.
            x0i: Imaginary-part spike tensor of the first complex input.
            x1r: Real-part spike tensor of the second complex input.
            x1i: Imaginary-part spike tensor of the second complex input.
            scale: Integer carry scale in ``1`` through ``scale_max`` for this
                timestep. This control value is not a spike port.

        Returns:
            ``(y0r, y0i, y1r, y1i)`` as bipolar 0/1 spike tensors, each encoding
            its exact butterfly output divided by :attr:`compensation`.

        The call advances every child once and updates the dynamic adder
        accumulator at ``scale`` without modifying the input spikes. Hold
        ``scale`` constant after reset so that one compensation factor describes
        the whole stream.
        """
        scale = self._runtime_scale(scale)
        if not (x0r.shape == x0i.shape == x1r.shape == x1i.shape):
            message = (
                f'butterfly_ugemm input shapes must match: got <{x0r.shape}>, '
                f'<{x0i.shape}>, <{x1r.shape}>, and <{x1i.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        if x0r.ndim == 0 or x0r.shape[0] != self.lane:
            message = (
                f'butterfly_ugemm first input dimension must equal lane <{self.lane}>: '
                f'got shape <{x0r.shape}>.'
            )
            logger.error(message)
            raise AssertionError(message)

        lane = self.lane
        tail = (1,) * (x0r.ndim - 1)
        x0_spike = torch.cat([x0r, x0i], 0)
        x1_spike = torch.cat([x1r, x1i], 0)

        product = self.mul_wx(
            torch.cat([x1_spike, x1_spike], 0),
            self.twiddle_stack.reshape((-1,) + tail),
        )
        product_01 = product.narrow(0, 0, 2 * lane)
        product_32 = torch.cat([
            product.narrow(0, 3 * lane, lane),
            product.narrow(0, 2 * lane, lane),
        ], 0)

        # Bias terms absorb inverted spikes in the real-minus and imaginary-plus lanes.
        term = product_01 + self.sign.reshape((-1,) + tail) * product_32
        y0_sum = x0_spike + term + self.bias0.reshape((-1,) + tail)
        y1_sum = x0_spike - term + self.bias1.reshape((-1,) + tail)

        y_spike = self.add_y(
            torch.cat([y0_sum, y1_sum], 0), scale, entry=self.add_entry, dim=None
        )
        self.compensation = scale
        return (
            y_spike.narrow(0, 0, lane),
            y_spike.narrow(0, lane, lane),
            y_spike.narrow(0, 2 * lane, lane),
            y_spike.narrow(0, 3 * lane, lane),
        )


    def __call__(self, x0r, x0i, x1r, x1i, scale):
        """Validate the runtime scale, then process one butterfly timestep.

        Args:
            x0r: Real-part spike tensor of the first complex input.
            x0i: Imaginary-part spike tensor of the first complex input.
            x1r: Real-part spike tensor of the second complex input.
            x1i: Imaginary-part spike tensor of the second complex input.
            scale: Exact Python int in ``1`` through ``scale_max``.

        Returns:
            ``(y0r, y0i, y1r, y1i)`` as bipolar 0/1 spike tensors.

        An invalid scale raises before this module or any child advances or
        updates persistent state. A valid call advances each exactly once.
        """
        self._runtime_scale(scale)
        return super().__call__(x0r, x0i, x1r, x1i, scale)


    def _runtime_scale(self, scale):
        """Return a validated runtime carry scale."""
        if type(scale) is not int:
            message = f'butterfly_ugemm_dyn scale must be an int: got <{scale}>.'
            logger.error(message)
            raise AssertionError(message)
        if scale < 1 or scale > self.add_y.scale_max:
            message = (
                f'butterfly_ugemm_dyn scale <{scale}> outside the supported '
                f'range <1> to scale_max <{self.add_y.scale_max}>.'
            )
            logger.error(message)
            raise AssertionError(message)
        return scale

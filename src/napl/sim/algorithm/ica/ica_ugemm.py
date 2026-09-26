from loguru import logger

from napl.sim.base import napl_base
from napl.sim.module import linear_ugemm
from napl.sim.operation import tanh_pn


class ica_ugemm(napl_base):
    r"""Apply a pre-learned ICA unmixing matrix to spike streams.

    The input carries one bipolar mixed-signal spike per feature. One call
    applies the fixed unmixing matrix for one timestep,

    .. math::

       y = W x.

    The fixed ``scale=1`` composition requires the numeric transform to stay
    within the bipolar output range, ``|W x| <= 1``. Choose ``dim`` to differ
    from the Sobol dimension used by the upstream input encoder.

    When **contrast** is enabled, the call returns ``(source, contrast)`` and
    the secondary output applies :class:`tanh_pn` elementwise to the recovered
    source stream. Otherwise it returns only the recovered source stream.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.algorithm.ica import ica_ugemm

        separator = ica_ugemm(
            torch.eye(2),
            {'polarity': 'bipolar', 'timestep': 256,
             'generator': 'sobol'},
        )
        source_spike = separator(torch.tensor([0, 1]))

    .. container:: api-references

        .. rubric:: References

        *Independent Component Analysis: Algorithms and Applications*, Neural
        Networks, 2000.
    """
    streaming = True


    def __init__(
            self,
            weight,
            config={
                'polarity': 'bipolar',
                'timestep': 256,
                'generator': 'sobol',
                'dim': 1,
                'width': 8,
                'contrast': False,
                'contrast_depth': 5,
            },
        ):
        """Configure the fixed unmixing transform and optional contrast output.

        .. container:: api-parameter-list

            **Parameters:**

            - **weight** – Pre-learned unmixing matrix shaped
              ``(sources, mixed_signals)``.
            - **config** – Configuration mapping.

              - **polarity**: Required stream encoding, which must be
                ``"bipolar"``.
              - **timestep**: Required positive stream length.
              - **generator**: Required number-sequence generator name.
              - **dim**: Optional number-sequence dimension; the default is
                ``1``. It must differ from the upstream input encoder's Sobol
                dimension.
              - **width**: Optional signed accumulator width; the default is
                ``8``.
              - **contrast**: Return a secondary tanh contrast stream when
                ``True``; the default is ``False``.
              - **contrast_depth**: Counter width of the optional
                :class:`tanh_pn`; the default is ``5``.
              - **name**: Optional instance label.
        """
        super().__init__(
            config,
            ['polarity', 'timestep', 'generator'],
            optional_key_list=['dim', 'width', 'contrast', 'contrast_depth'],
            polarity_required=True,
        )

        if self.polarity != 'bipolar':
            message = "ica_ugemm supports only bipolar spike streams."
            logger.error(message)
            raise AssertionError(message)
        contrast = config.get('contrast', False)
        if not isinstance(contrast, bool):
            message = f'Invalid contrast: <{contrast}>; legal values: a boolean.'
            logger.error(message)
            raise AssertionError(message)
        contrast_depth = config.get('contrast_depth', 5)
        if type(contrast_depth) is not int or contrast_depth < 1:
            message = (
                f'Invalid contrast_depth: <{contrast_depth}>; legal values: '
                'an integer of at least 1.'
            )
            logger.error(message)
            raise AssertionError(message)

        #: Whether :meth:`forward` returns the secondary contrast stream.
        self.contrast = contrast
        #: Streaming linear transform that applies the fixed unmixing matrix.
        self.unmix = linear_ugemm(
            weight,
            None,
            {
                'polarity': self.polarity,
                'timestep': config['timestep'],
                'generator': config['generator'],
                'dim': config.get('dim', 1),
                'scale': 1,
                'width': config.get('width', 8),
            },
        )
        #: Optional bipolar tanh contrast transform.
        self.contrast_op = (
            tanh_pn({'depth': contrast_depth})
            if self.contrast else None
        )

        #: Encoder the hardware counterpart carries, derived from the registered
        #: parts: ``'private'`` when any part carries an encoder of its own,
        #: ``'none'`` otherwise.
        self.internal_encode = 'private' if any(part.internal_encode != 'none' for part in self.children()) else 'none'
        self.hw.pp_delay = self.unmix.hw.pp_delay + (
            self.contrast_op.hw.pp_delay if self.contrast_op is not None else 0
        )
        self.encoding_io = {'input': 'rc', 'source': 'rc'}
        self.polarity_io = {'input': 'bipolar', 'source': 'bipolar'}
        if self.contrast:
            self.encoding_io['contrast'] = 'rc'
            self.polarity_io['contrast'] = 'bipolar'
        self.correlation_i = {}


    def _reset(self):
        """Reset state owned directly by the separator.

        This class has no extra local state. The inherited ``reset()`` method
        restarts the held unmixing layer and optional contrast operation.
        """
        pass


    def forward(self, input):
        """Process one mixed-signal spike timestep.

        Args:
            input: Bipolar 0/1 spike tensor whose last dimension equals the
                unmixing matrix input dimension.

        Returns:
            Recovered-source spike tensor, or ``(source, contrast)`` when
            contrast output is enabled.
        """
        source = self.unmix(input)
        if self.contrast_op is None:
            return source
        return source, self.contrast_op(source)

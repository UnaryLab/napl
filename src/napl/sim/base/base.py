import os
import napl
import torch

from loguru import logger
from dataclasses import dataclass, field
from napl.utils import *
from functools import wraps


torch_dtype_map = {
    'torch.float32': torch.float32,
    'torch.float': torch.float32,
    'torch.float64': torch.float64,
    'torch.double': torch.float64,
    'torch.float16': torch.float16,
    'torch.half': torch.float16,
    'torch.bfloat16': torch.bfloat16,
    'torch.int64': torch.int64,
    'torch.long': torch.int64,
    'torch.int32': torch.int32,
    'torch.int': torch.int32,
    'torch.int16': torch.int16,
    'torch.short': torch.int16,
    'torch.int8': torch.int8,
    'torch.uint8': torch.uint8,
    'torch.bool': torch.bool,
}


_GLOBAL_ROOT_PATH = os.path.dirname(os.path.abspath(napl.__file__))
_GLOBAL_CONFIG_FILE = os.path.join(_GLOBAL_ROOT_PATH, 'sim/base/global_config.yaml')


@dataclass
class global_config_check:
    """Load and validate the process-wide tensor dtypes used by NAPL.

    Use this class when inspecting or validating the global configuration. Most
    applications use the module-level ``global_config`` instance instead of
    constructing another instance.

    The YAML ``global_config`` mapping must define **spike_type** as
    ``torch.float``, ``torch.bfloat16``, or ``torch.int8`` and
    **non_spike_type** as ``torch.float`` or ``torch.bfloat16``.

    .. rubric:: Example

    .. code-block:: python

        from napl import global_config_check

        config = global_config_check()
        print(config.stype, config.ntype)
    """
    #: Installed NAPL package directory used to locate shared data files.
    root_path: str = field(default_factory=lambda: _GLOBAL_ROOT_PATH)
    #: YAML file that defines the process-wide spike and non-spike dtypes.
    config_file: str = field(default_factory=lambda: _GLOBAL_CONFIG_FILE)
    assert os.path.exists(_GLOBAL_CONFIG_FILE), logger.error(
        f'Global configuration file <{_GLOBAL_CONFIG_FILE}> does not exist.'
    )
    #: Parsed contents of :attr:`config_file`.
    config = read_yaml(_GLOBAL_CONFIG_FILE)

    #: PyTorch dtype used for spike tensors.
    stype = torch_dtype_map.get(config['global_config']['spike_type'], None)
    assert stype in [torch.float, torch.bfloat16, torch.int8], \
        logger.error(f'Invalid spike type: <{stype}>; legal types: [torch.float, torch.bfloat16, torch.int8].')

    #: PyTorch dtype used for non-spike values and accumulated results.
    ntype = torch_dtype_map.get(config['global_config']['non_spike_type'], None)
    assert ntype in [torch.float, torch.bfloat16], \
        logger.error(f'Invalid non-spike type: <{ntype}>; legal types: [torch.float, torch.bfloat16].')


global_config = global_config_check()


@dataclass(frozen=True)
class pvt_corner:
    """Describe one process, voltage, temperature, RC, and mode scenario.

    Use this immutable value as a key in :class:`hw_params` timing maps when a
    hardware implementation has been characterized at multiple sign-off corners.

    .. rubric:: Example

    .. code-block:: python

        from napl import pvt_corner

        corner = pvt_corner("asap7", "ss", 0.63, 125.0, rc="cworst")
    """
    #: Technology-node identifier, such as ``"asap7"``.
    node: str
    #: Process corner or standard-cell library set, such as ``"ss"``.
    process: str
    #: Supply voltage in volts.
    voltage: float
    #: Junction temperature in degrees Celsius.
    temp: float
    #: Interconnect-resistance/capacitance corner.
    rc: str = 'typ'
    #: Multi-corner analysis mode, such as ``"func"`` or ``"scan"``.
    mode: str = 'func'


@dataclass
class timing:
    """Store non-overlapping timing segments for one hardware module.

    Use this value for pre-synthesis path estimates or to attach characterized
    nanosecond delays to a :class:`pvt_corner`. Segment delays add across module
    boundaries without double counting.

    .. rubric:: Example

    .. code-block:: python

        from napl import timing

        path = timing(cp_delay=0.42, ir_delay=0.08, or_delay=0.05)
    """
    #: Worst internal combinational delay in nanoseconds.
    cp_delay: float = 0.0
    #: Input-port-to-first-register delay in nanoseconds.
    ir_delay: float = 0.0
    #: Last-register-to-output-port delay in nanoseconds.
    or_delay: float = 0.0


@dataclass
class hw_params:
    """Describe the latency and characterized timing of a NAPL hardware block.

    Use this contract when connecting a functional simulation class to generated
    RTL or when balancing reconvergent paths. ``pp_delay`` affects stream timing;
    entries in ``timing`` record implementation measurements and do not change
    functional results.

    .. rubric:: Example

    .. code-block:: python

        from napl import hw_params, pvt_corner, timing

        corner = pvt_corner("asap7", "tt", 0.70, 25.0)
        hw = hw_params(pp_delay=1, timing={corner: timing(cp_delay=0.35)})
    """
    #: Input-to-output pipeline latency in clock cycles.
    pp_delay: int = 0
    #: Timing data indexed by :class:`pvt_corner`.
    timing: dict = field(default_factory=dict)


class napl_base(torch.nn.Module):
    """Provide shared execution state and reset behavior for NAPL modules.

    Use this class as the base of a new NAPL simulation module. Streaming
    subclasses advance ``timestep_cur`` once per call. Single-shot subclasses set
    ``streaming = False`` and leave the timestep at ``0``.

    .. rubric:: Example

    .. code-block:: python

        from napl import napl_base

        module = napl_base()
        module.tick()
        assert module.valid
        module.reset()
        assert not module.valid
    """
    #: Whether each call represents one streaming timestep.
    streaming = True


    def __init__(self, config: dict={}, key_list: list=[], polarity_required: bool=False):
        """Initialize shared configuration, execution state, and hardware metadata.

        Args:
            config: Module configuration. The shared optional keys are
                **polarity**, ``"unipolar"`` or ``"bipolar"``, and **name**, a
                user label. Their defaults are supplied by ``check_polarity()``
                and ``check_name()`` when the caller permits them to be absent.
            key_list: Configuration keys accepted by the subclass. Defaults to
                an empty list.
            polarity_required: Require **polarity** to be present when ``True``.
                Defaults to ``False``.

        The constructor sets ``timestep_cur`` to ``0`` and initializes ``hw`` to
        a zero-latency :class:`hw_params` value.
        """
        super().__init__()
        #: PyTorch dtype used for spike tensors in this module.
        self.stype = global_config.stype
        #: PyTorch dtype used for non-spike values in this module.
        self.ntype = global_config.ntype

        if polarity_required is True:
            assert 'polarity' in config, logger.error('Missing key <polarity> in the input configuration.')
        check_config(config, key_list)
        #: Stream encoding, either ``"unipolar"``, ``"bipolar"``, or ``None``.
        self.polarity = check_polarity(config)
        #: User-facing module label derived from the configuration.
        self.name = check_name(config)

        #: Number of streaming timesteps processed since the last reset.
        self.timestep_cur = 0

        #: Hardware latency and characterized timing metadata for this module.
        self.hw = hw_params()


    def _reset(self):
        """Reset state owned directly by the base class.

        The base implementation has no additional local state and returns
        ``None``. Subclasses override this hook for their own mutable state;
        callers use :meth:`reset` instead.
        """
        pass


    @property
    def valid(self):
        """Report whether a streaming module has processed at least one timestep.

        Returns:
            ``True`` when ``timestep_cur > 0``; otherwise ``False``.

        Reading this property does not change state. It remains ``False`` for
        single-shot modules because they do not advance ``timestep_cur``.
        """
        return self.timestep_cur > 0


    def tick(self):
        """Advance a streaming module by one timestep.

        This method increments ``timestep_cur`` in place and returns ``None``.
        User code normally calls the module itself, which invokes ``tick()``
        automatically for streaming classes.

        **Example:**

        .. code-block:: python

            module.tick()
            assert module.timestep_cur == 1
        """
        self.timestep_cur += 1


    def __call__(self, *args, **kwargs):
        """Run ``forward()`` and update streaming execution state.

        Args:
            *args: Positional inputs forwarded to ``forward()``.
            **kwargs: Keyword inputs forwarded to ``forward()``.

        Returns:
            The value returned by ``forward()``.

        For streaming modules, the call increments ``timestep_cur`` before
        ``forward()`` runs. Single-shot modules with ``streaming = False`` do not
        change the timestep. Call the module normally instead of invoking this
        method directly.
        """
        if self.streaming:
            self.tick()
        return super().__call__(*args, **kwargs)


    def reset(self, verbose=False):
        """
        Start a new run by resetting the timestep, registered child modules, and class-owned state.

        Args:
            verbose: Log the reset when ``True``. Defaults to ``False``.

        Returns:
            ``None``.

        This method sets ``timestep_cur`` to ``0``, resets registered child
        modules that expose ``reset()``, and then calls the subclass ``_reset()``
        hook.

        **Example:**

        .. code-block:: python

            module.reset()
        """
        self.timestep_cur = 0
        if verbose:
            logger.info(f'Reset module <{self.__class__.__name__}>.')
        for module in self.children():
            if hasattr(module, 'reset'):
                module.reset()
        self._reset()


def napl_sim_timesteps(timestep_func):
    """Repeat a NAPL method or free function for a requested number of timesteps."""
    @wraps(timestep_func)
    def timesteps_wrapper(*args, **kwargs):
        assert 'timesteps' in kwargs, \
            logger.error('Timesteps not specified in the arguments. Please provide <timesteps> as a keyword argument.')

        timesteps = kwargs.pop('timesteps', 256)
        verbose = kwargs.pop('verbose', False)
        if verbose:
            if args and isinstance(args[0], napl_base):
                target = f'class <{args[0].__class__.__name__}>'
            else:
                target = f'function <{timestep_func.__name__}>'
            logger.info(f'Simulating <{timesteps}> timesteps in NAPL {target}...')

        for _ in range(timesteps):
            output = timestep_func(*args, **kwargs)
        return output

    return timesteps_wrapper

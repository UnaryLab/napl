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


# Initialize global configuration
# This will be used throughout the NAPL framework to maintain consistent data types across different modules and operations.
# The global configuration can be loaded from a YAML file or set directly.
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
    root_path: str = os.path.dirname(os.path.abspath(napl.__file__))
    config_file: str = str(os.path.join(root_path, 'sim/base/global_config.yaml'))
    assert os.path.exists(config_file), logger.error(f'Global configuration file <{config_file}> does not exist.')
    config = read_yaml(config_file)

    stype = torch_dtype_map.get(config['global_config']['spike_type'], None)
    assert stype in [torch.float, torch.bfloat16, torch.int8], \
        logger.error(f'Invalid spike type: <{stype}>; legal types: [torch.float, torch.bfloat16, torch.int8].')
        
    ntype = torch_dtype_map.get(config['global_config']['non_spike_type'], None)
    assert ntype in [torch.float, torch.bfloat16], \
        logger.error(f'Invalid non-spike type: <{ntype}>; legal types: [torch.float, torch.bfloat16].')


global_config_check.__init__.__doc__ = """Initialize global dtype validation.

Args:
    root_path: NAPL package root. Defaults to the installed package path.
    config_file: YAML file to read. Defaults to
        ``sim/base/global_config.yaml`` below ``root_path``.
"""

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
    node: str                    # tech node, e.g. 'asap7', 'tsmc28'
    process: str                 # process corner / libset, e.g. 'ss'/'tt'/'ff'
    voltage: float               # supply voltage in V, e.g. 0.63
    temp: float                  # junction temperature in degC, e.g. 125.0 (may be < 0)
    rc: str = 'typ'              # interconnect (RC) corner, e.g. 'cworst'/'rcworst'/'typ'
    mode: str = 'func'           # MCMM mode, e.g. 'func'/'scan'/'sleep'


pvt_corner.__init__.__doc__ = """Initialize an immutable hardware corner.

Args:
    node: Technology node identifier, such as ``"asap7"``.
    process: Process corner or library set, such as ``"ss"``.
    voltage: Supply voltage in volts.
    temp: Junction temperature in degrees Celsius.
    rc: Interconnect corner. Defaults to ``"typ"``.
    mode: Functional mode. Defaults to ``"func"``.
"""


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
    cp_delay: float = 0.0
    ir_delay: float = 0.0
    or_delay: float = 0.0


timing.__init__.__doc__ = """Initialize timing-segment delays.

Args:
    cp_delay: Worst internal combinational delay in nanoseconds. For a registered
        module this is the register-to-register path; for a combinational module
        it is the input-to-output delay. Defaults to ``0.0``.
    ir_delay: Input-port-to-first-register delay in nanoseconds. Defaults to
        ``0.0``.
    or_delay: Last-register-to-output-port delay in nanoseconds. Defaults to
        ``0.0``.
"""


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
    pp_delay: int = 0
    timing: dict = field(default_factory=dict)   # {pvt_corner: timing}


hw_params.__init__.__doc__ = """Initialize hardware latency and timing data.

Args:
    pp_delay: Input-to-output latency in clock cycles, equal to the number of
        register stages. ``0`` means purely combinational. Defaults to ``0``.
    timing: Mapping from :class:`pvt_corner` objects to :class:`timing` values.
        Defaults to an empty mapping.
"""


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
    # Streaming modules advance timestep_cur once per call; single-shot
    # binary-domain classes override with False.
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
        # Load global configuration
        self.stype = global_config.stype
        self.ntype = global_config.ntype

        # check config
        if polarity_required is True:
            assert 'polarity' in config, logger.error('Missing key <polarity> in the input configuration.')
        check_config(config, key_list)
        self.polarity = check_polarity(config)
        self.name = check_name(config)
        
        self.timestep_cur = 0

        # hardware contract for RTL generation; ops override with their own values
        self.hw = hw_params()


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
        # __call__ override, not register_forward_pre_hook: a hook on every module
        # forces nn.Module's slow call path per timestep.
        if self.streaming:
            self.tick()
        return super().__call__(*args, **kwargs)


    @property
    def valid(self):
        """Report whether a streaming module has processed at least one timestep.

        Returns:
            ``True`` when ``timestep_cur > 0``; otherwise ``False``.

        Reading this property does not change state. It remains ``False`` for
        single-shot modules because they do not advance ``timestep_cur``.
        """
        return self.timestep_cur > 0


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


    def _reset(self):
        """Reset state owned directly by the base class.

        The base implementation has no additional local state and returns
        ``None``. Subclasses override this hook for their own mutable state;
        callers use :meth:`reset` instead.
        """
        pass


def napl_sim_timesteps(timestep_func):
    """
    This function is a decorator to simulate multiple timesteps in the NAPL framework.
    """
    @wraps(timestep_func)
    def timesteps_wrapper(self, *args, **kwargs):
        assert 'timesteps' in kwargs, \
            logger.error('Timesteps not specified in the arguments. Please provide <timesteps> as a keyword argument.')
        
        timesteps = kwargs.pop('timesteps', 256)  # Remove 'timesteps' from kwargs
        verbose = kwargs.pop('verbose', False)  # Remove 'timesteps' from kwargs
        if verbose:
            logger.info(f'Simulating <{timesteps}> timesteps in NAPL class <{self.__class__.__name__}>...')

        for _ in range(timesteps):
            output = timestep_func(self, *args, **kwargs)
        return output
    
    return timesteps_wrapper


def napl_sim_timesteps_func(timestep_func):
    """
    This function is a decorator to simulate multiple timesteps in the NAPL framework.
    """
    @wraps(timestep_func)
    def timesteps_wrapper(*args, **kwargs):
        assert 'timesteps' in kwargs, \
            logger.error('Timesteps not specified in the arguments. Please provide <timesteps> as a keyword argument.')
        
        timesteps = kwargs.pop('timesteps', 256)  # Remove 'timesteps' from kwargs
        verbose = kwargs.pop('verbose', False)  # Remove 'timesteps' from kwargs
        if verbose:
            logger.info(f'Simulating <{timesteps}> timesteps in NAPL function...')

        for _ in range(timesteps):
            out = timestep_func(*args, **kwargs)
        return out
    
    return timesteps_wrapper

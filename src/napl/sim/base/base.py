import os
import napl
import torch

from loguru import logger
from dataclasses import dataclass, field
from napl.utils import read_yaml
from functools import lru_cache, wraps
from importlib.resources import files
from inspect import unwrap


@lru_cache(maxsize=None)
def _load_flux_map(package):
    """Return the package's flux_stability.yaml as class -> polarity -> float, or {} if absent."""
    try:
        resource = files(package).joinpath('flux_stability.yaml')
        if not resource.is_file():
            return {}
        path = str(resource)
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError, OSError,
            AttributeError, ValueError):
        return {}
    # A present-but-corrupt yaml raises loud at first construction; read_yaml is outside the guard.
    return read_yaml(path) or {}


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


def check_config(config: dict, key_list: list, optional_key_list: list = []):
    """
    Check that every key in ``key_list`` is present in ``config``.

    Keys in ``optional_key_list`` and the key ``name`` may be absent; any key
    outside these three groups is rejected. Raises ``AssertionError`` on a
    missing required key or an unknown key.
    """
    for key in key_list:
        if key not in config:
            message = f'Missing key <{key}> in the input configuration.'
            logger.error(message)
            raise AssertionError(message)
    accepted = set(key_list) | set(optional_key_list) | {'name'}
    for key in config:
        if key not in accepted:
            message = (f'Unknown key <{key}> in the input configuration; '
                       f'accepted keys: <{sorted(accepted)}>.')
            logger.error(message)
            raise AssertionError(message)


def check_polarity(config: dict):
    """
    Check that the optional ``polarity`` key of ``config`` is legal.

    Returns the lowercased polarity, or ``None`` when the key is absent. Raises
    ``AssertionError`` for anything other than ``"unipolar"`` or ``"bipolar"``.
    """
    polarity = config.get('polarity', None)
    if polarity is not None:
        if not isinstance(polarity, str):
            message = f'Invalid polarity: <{polarity}>; polarity should be a string.'
            logger.error(message)
            raise AssertionError(message)
        polarity = polarity.lower()
        legal_polarity = ['unipolar', 'bipolar']
        if polarity not in legal_polarity:
            message = f'Invalid polarity: <{polarity}>; legal values: <{str(legal_polarity)}>.'
            logger.error(message)
            raise AssertionError(message)
    return polarity


def check_name(config: dict):
    """
    Check that the optional ``name`` key of ``config`` is a string.

    Returns the lowercased name, or ``None`` when the key is absent. Raises
    ``AssertionError`` when the value is not a string.
    """
    name = config.get('name', None)
    if name is not None:
        if not isinstance(name, str):
            message = f'Invalid name: <{name}>; name should be a string.'
            logger.error(message)
            raise AssertionError(message)
        name = name.lower()
    return name


_GLOBAL_ROOT_PATH = os.path.dirname(os.path.abspath(napl.__file__))
_GLOBAL_CONFIG_FILE = os.path.join(_GLOBAL_ROOT_PATH, 'sim/base/global_config.yaml')


@dataclass
class global_config_check:
    """Hold the process-wide spike and non-spike dtypes read from the global YAML configuration."""
    #: Installed NAPL package directory used to locate shared data files.
    root_path: str = field(default_factory=lambda: _GLOBAL_ROOT_PATH)
    #: YAML file that defines the process-wide spike and non-spike dtypes.
    config_file: str = field(default_factory=lambda: _GLOBAL_CONFIG_FILE)
    if not os.path.exists(_GLOBAL_CONFIG_FILE):
        message = f'Global configuration file <{_GLOBAL_CONFIG_FILE}> does not exist.'
        logger.error(message)
        raise AssertionError(message)
    #: Parsed contents of :attr:`config_file`.
    config = read_yaml(_GLOBAL_CONFIG_FILE)

    #: PyTorch dtype used for spike tensors.
    stype = torch_dtype_map.get(config['global_config']['spike_type'], None)
    if stype not in [torch.float, torch.bfloat16, torch.int8]:
        message = f'Invalid spike type: <{stype}>; legal types: [torch.float, torch.bfloat16, torch.int8].'
        logger.error(message)
        raise AssertionError(message)

    #: PyTorch dtype used for non-spike values and accumulated results.
    ntype = torch_dtype_map.get(config['global_config']['non_spike_type'], None)
    if ntype not in [torch.float, torch.bfloat16]:
        message = f'Invalid non-spike type: <{ntype}>; legal types: [torch.float, torch.bfloat16].'
        logger.error(message)
        raise AssertionError(message)


global_config = global_config_check()


@dataclass(frozen=True)
class pvt_corner:
    """Identify one immutable process, voltage, temperature, RC, and mode scenario used to key timing data."""
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
    """Hold the nanosecond path delays measured for a module at one ``pvt_corner``."""
    #: Worst internal combinational delay in nanoseconds.
    cp_delay: float = 0.0
    #: Input-port-to-first-register delay in nanoseconds.
    ir_delay: float = 0.0
    #: Last-register-to-output-port delay in nanoseconds.
    or_delay: float = 0.0


@dataclass
class hw_params:
    """Describe a module's hardware counterpart: its pipeline latency and its characterized timing data."""
    #: Input-to-output pipeline latency in clock cycles.
    pp_delay: int = 0
    #: Timing data indexed by ``pvt_corner``.
    timing: dict = field(default_factory=dict)


class napl_base(torch.nn.Module):
    """Provide shared execution state and reset behavior for NAPL modules.

    Use this class as the base of a new NAPL simulation module. Streaming
    subclasses advance ``timestep_cur`` once per call. Non-streaming subclasses set
    ``streaming = False`` and leave the timestep at ``0``.

    Every module takes its tensor dtypes from the module-level ``global_config``
    instance of ``global_config_check``, which loads them from the YAML
    ``global_config`` mapping in ``sim/base/global_config.yaml``. That mapping
    must define **spike_type** as ``torch.float``, ``torch.bfloat16``, or
    ``torch.int8`` and **non_spike_type** as ``torch.float`` or
    ``torch.bfloat16``. Most applications read that module-level instance rather
    than constructing another one.

    The ``hw`` attribute holds a ``hw_params`` value describing the module's
    hardware counterpart, the contract to use when connecting a functional
    simulation class to generated RTL or when balancing reconvergent paths. Its
    ``pp_delay`` is the input-to-output pipeline latency in clock cycles and
    affects stream timing; its ``timing`` map records implementation
    measurements and does not change functional results. Each key of that map is
    a ``pvt_corner``, one immutable process, voltage, temperature, RC, and mode
    scenario, and each value is a ``timing`` value holding nanosecond delays for
    pre-synthesis path estimates or characterized sign-off numbers. Timing
    segments do not overlap, so segment delays add across module boundaries
    without double counting.

    .. rubric:: Example

    .. code-block:: python

        from napl.sim.base import napl_base

        module = napl_base()
        module.tick()
        assert module.valid
        module.reset()
        assert not module.valid
    """
    #: Whether each call represents one streaming timestep.
    streaming = True

    #: Whether the RTL counterpart holds its own encoder.
    #: True when the hardware counterpart carries the encoder itself, covering
    #: both encoding that advances conditionally on data and operands such as
    #: weights and biases that it encodes internally from held numeric codes.
    #: The binary-domain fxp classes have no encoder, so they are False.
    internal_encode = False


    def __init__(self, config: dict={}, key_list: list=[], optional_key_list: list=[], polarity_required: bool=False):
        """Initialize shared configuration, execution state, and hardware metadata.

        Args:
            config: Module configuration. **name**, a user label, is always
                accepted and defaults to ``None``. **polarity**, ``"unipolar"``
                or ``"bipolar"``, is accepted only when the subclass lists it in
                ``key_list`` or ``optional_key_list``, and defaults to ``None``
                when absent.
            key_list: Configuration keys accepted by the subclass. Defaults to
                an empty list.
            optional_key_list: Configuration keys the subclass accepts but does
                not require. Defaults to an empty list.
            polarity_required: Require **polarity** to be present when ``True``.
                Defaults to ``False``.

        The constructor sets ``timestep_cur`` to ``0`` and initializes ``hw`` to
        a zero-latency ``hw_params`` value.
        """
        super().__init__()
        #: PyTorch dtype used for spike tensors in this module.
        self.stype = global_config.stype
        #: PyTorch dtype used for non-spike values in this module.
        self.ntype = global_config.ntype

        if polarity_required is True and 'polarity' not in config:
            message = 'Missing key <polarity> in the input configuration.'
            logger.error(message)
            raise AssertionError(message)
        check_config(config, key_list, optional_key_list)
        #: Stream encoding, either ``"unipolar"``, ``"bipolar"``, or ``None``.
        self.polarity = check_polarity(config)
        #: User-facing module label derived from the configuration.
        self.name = check_name(config)

        parts = type(self).__module__.split('.')
        #: Simulation layer this class belongs to, given by the ``napl.sim`` subdirectory
        #: of the defining module, such as ``"operation"``, ``"module"``,
        #: ``"metric"``, ``"structure"``, or ``"algorithm"``. A class defined
        #: outside ``napl.sim`` reports ``""``.
        self.layer = parts[2] if len(parts) > 2 and parts[:2] == ['napl', 'sim'] else ''

        #: Number of streaming timesteps processed since the last reset.
        self.timestep_cur = 0

        #: Hardware latency and characterized timing metadata for this module.
        self.hw = hw_params()

        #: Stream encoding required per input and produced per output, keyed by
        #: ``forward()`` parameter name and output name; ``"rc"`` for rate coding
        #: and ``"tc"`` for temporal coding. Unconstrained ports are omitted.
        self.encoding_io = {}

        #: Stream polarity required per input and produced per output, keyed by
        #: ``forward()`` parameter name and output name; ``"unipolar"`` or
        #: ``"bipolar"``. Unconstrained ports are omitted.
        self.polarity_io = {}

        #: Cross-correlation required between input streams, keyed by a tuple of
        #: ``forward()`` parameter names; ``"zero"``, ``"pos"``, or ``"neg"``.
        self.correlation_i = {}

        if not isinstance(getattr(type(self), 'flux_stability', None), property):
            #: Relative output flux stability of this module. A subclass may
            #: expose ``flux_stability`` as a property instead, as the
            #: stability_flux metric does for its measured value, and then no
            #: placeholder is set here. flux_stability comes from the class's
            #: package profiling yaml when present, else 1.0; the class's own
            #: package is already imported by construction time and the yaml is
            #: data-only, so no import cycle.
            flux_map = _load_flux_map(type(self).__module__.rsplit('.', 1)[0])
            entry = flux_map.get(type(self).__name__)
            polarity = getattr(self, 'polarity', None)
            self.flux_stability = entry.get(polarity, 1.0) if isinstance(entry, dict) else 1.0


    def _reset(self):
        """Reset state owned directly by the base class.

        The base implementation has no additional local state and returns
        ``None``; callers use :meth:`reset` instead.

        Every streaming subclass defines this hook. A streaming subclass with
        no local mutable state uses a documented ``pass``. A non-streaming
        subclass may inherit this no-op implementation.
        """
        pass


    @property
    def valid(self):
        """Report whether a streaming module has processed at least one timestep.

        Returns:
            ``True`` when ``timestep_cur > 0``; otherwise ``False``.

        Reading this property does not change state. It remains ``False`` for
        non-streaming modules because they do not advance ``timestep_cur``.
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


    def forward_timestep(self, *args, **kwargs):
        """Advance ``forward()`` by exactly one timestep.

        Args:
            *args: Positional inputs forwarded to ``forward()``.
            **kwargs: Keyword inputs forwarded to ``forward()``.

        Returns:
            The value ``forward()`` produces for one timestep.

        The call increments ``timestep_cur`` once for a streaming module and
        runs a single timestep even when ``forward()`` carries the
        ``napl_sim_timesteps`` decorator, so a caller can step a run and read
        the progressively refined result after each timestep. It never resets,
        so consecutive calls continue the current run.

        **Example:**

        .. code-block:: python

            import torch
            from napl.sim.operation import shiftreg

            delay = shiftreg({'depth': 2})
            for _ in range(4):
                output = delay.forward_timestep(torch.tensor([1], dtype=torch.int8))
            assert delay.timestep_cur == 4
        """
        if self.streaming:
            self.tick()
        return unwrap(type(self).forward)(self, *args, **kwargs)


    def __call__(self, *args, **kwargs):
        """Run ``forward()`` and update streaming execution state.

        Args:
            *args: Positional inputs forwarded to ``forward()``.
            **kwargs: Keyword inputs forwarded to ``forward()``.

        Returns:
            The value returned by ``forward()``.

        For streaming modules, the call increments ``timestep_cur`` before
        ``forward()`` runs. Non-streaming modules with ``streaming = False`` do not
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
    """Repeat a NAPL method or free function for a run of timesteps.

    Args:
        timestep_func: Callable that advances one timestep per invocation.

    Returns:
        A wrapper accepting the same arguments plus two keyword-only entries:
        **timesteps**, the number of repetitions, and **verbose**, which logs
        the run when ``True`` and defaults to ``False``. The wrapper returns the
        value produced by the final repetition.

    A ``napl_base`` module carrying a ``timestep`` attribute, such as a hub
    wrapper, may omit **timesteps**. One such call is then a complete fresh run:
    the wrapper resets the module and its children, repeats the callable for the
    configured ``timestep`` cycles on the same arguments, and counts those cycles
    on ``timestep_cur`` for a streaming module. A non-streaming module keeps
    ``timestep_cur`` at ``0``, and its streaming children hold the cycle count
    instead. Any state the caller set beforehand,
    including a partly advanced run, is discarded by that reset, so repeating
    the call on the same input returns the same result. Every other target must
    pass **timesteps**, and the wrapper then repeats the callable without
    resetting it. Use :meth:`napl_base.forward_timestep` to advance a decorated
    ``forward()`` one timestep at a time instead.
    """
    @wraps(timestep_func)
    def timesteps_wrapper(*args, **kwargs):
        module = args[0] if args and isinstance(args[0], napl_base) else None
        # A module that carries its own run length needs no timesteps argument.
        fresh_run = 'timesteps' not in kwargs and getattr(module, 'timestep', None) is not None
        if 'timesteps' not in kwargs and not fresh_run:
            message = 'Timesteps not specified in the arguments. Please provide <timesteps> as a keyword argument.'
            logger.error(message)
            raise AssertionError(message)

        timesteps = kwargs.pop('timesteps', None)
        verbose = kwargs.pop('verbose', False)
        if fresh_run:
            timesteps = module.timestep
            module.reset()
        if verbose:
            if module is not None:
                target = f'class <{module.__class__.__name__}>'
            else:
                target = f'function <{timestep_func.__name__}>'
            logger.info(f'Simulating <{timesteps}> timesteps in NAPL {target}...')

        for _ in range(timesteps):
            # A streaming module counts a fresh run's cycles on its own counter,
            # since __call__ charges its tick and the reset cleared the one charged
            # for this run, while a non-streaming hub keeps its counter at 0 and its
            # streaming children carry the cycle count instead.
            if fresh_run and module.streaming:
                module.tick()
            output = timestep_func(*args, **kwargs)
        return output

    return timesteps_wrapper

import os
import napl
import torch

from loguru import logger
from dataclasses import dataclass, field
from napl.utils import read_yaml
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


def check_config(config: dict, key_list: list, optional_key_list: list = []):
    """
    Check if all key in the key_list exists in the config.

    Keys in the optional_key_list may be absent; any other key is rejected.
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
    Check if polarity is legal.
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
    Check if name is available.
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
    #: Worst internal combinational delay in nanoseconds.
    cp_delay: float = 0.0
    #: Input-port-to-first-register delay in nanoseconds.
    ir_delay: float = 0.0
    #: Last-register-to-output-port delay in nanoseconds.
    or_delay: float = 0.0


@dataclass
class hw_params:
    #: Input-to-output pipeline latency in clock cycles.
    pp_delay: int = 0
    #: Timing data indexed by ``pvt_corner``.
    timing: dict = field(default_factory=dict)


class napl_base(torch.nn.Module):
    """Provide shared execution state and reset behavior for NAPL modules.

    Use this class as the base of a new NAPL simulation module. Streaming
    subclasses advance ``timestep_cur`` once per call. Single-shot subclasses set
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

        from napl import napl_base

        module = napl_base()
        module.tick()
        assert module.valid
        module.reset()
        assert not module.valid
    """
    #: Whether each call represents one streaming timestep.
    streaming = True

    #: Whether the RTL counterpart must hold its own encoder.
    #: True when encoding advances conditionally on data, as in conditional
    #: bitstream generation, so the encoder cannot be shared with other
    #: operations.
    internal_encode = False


    def __init__(self, config: dict={}, key_list: list=[], optional_key_list: list=[], polarity_required: bool=False):
        """Initialize shared configuration, execution state, and hardware metadata.

        Args:
            config: Module configuration. The shared optional keys are
                **polarity**, ``"unipolar"`` or ``"bipolar"``, and **name**, a
                user label. Their defaults are supplied by ``check_polarity()``
                and ``check_name()`` when the caller permits them to be absent.
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
        #: Simulation layer this class belongs to: the ``napl.sim`` subdirectory
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

        if not isinstance(getattr(type(self), 'stability_flux', None), property):
            #: Relative output stability flux of this module. A subclass may
            #: expose ``stability_flux`` as a property instead, as the
            #: stability_flux metric does for its measured value, and then no
            #: placeholder is set here.
            self.stability_flux = 1.0


    def _reset(self):
        """Reset state owned directly by the base class.

        The base implementation has no additional local state and returns
        ``None``; callers use :meth:`reset` instead.

        Every subclass defines this hook. A subclass that owns no reset state
        gives it a ``pass`` body whose docstring states that the class owns no
        reset state.
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
    """Repeat a NAPL method or free function for a requested number of timesteps.

    Args:
        timestep_func: Callable that advances one timestep per invocation.

    Returns:
        A wrapper accepting the same arguments plus two keyword-only entries:
        **timesteps**, the required number of repetitions, and **verbose**,
        which logs the run when ``True`` and defaults to ``False``. The wrapper
        returns the value produced by the final repetition.
    """
    @wraps(timestep_func)
    def timesteps_wrapper(*args, **kwargs):
        if 'timesteps' not in kwargs:
            message = 'Timesteps not specified in the arguments. Please provide <timesteps> as a keyword argument.'
            logger.error(message)
            raise AssertionError(message)

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

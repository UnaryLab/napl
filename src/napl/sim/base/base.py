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
    

global_config = global_config_check()


@dataclass(frozen=True)
class pvt_corner:
    """
    A full MCMM sign-off scenario at a given tech node: a P/V/T + RC corner
    crossed with a functional mode. Frozen so it is hashable and usable as a key
    in hw_params.timing. Mirrors what STA tools (PrimeTime/Tempus) call a
    scenario = corner (process libset + voltage + temperature + parasitic corner)
    x mode.
    """
    node: str                    # tech node, e.g. 'asap7', 'tsmc28'
    process: str                 # process corner / libset, e.g. 'ss'/'tt'/'ff'
    voltage: float               # supply voltage in V, e.g. 0.63
    temp: float                  # junction temperature in degC, e.g. 125.0 (may be < 0)
    rc: str = 'typ'              # interconnect (RC) corner, e.g. 'cworst'/'rcworst'/'typ'
    mode: str = 'func'           # MCMM mode, e.g. 'func'/'scan'/'sleep'


@dataclass
class timing:
    """
    Corner-dependent timing of one module, in ns, defined as non-overlapping
    segments of every input->output route so they add across module boundaries
    for pre-synthesis critical-path estimation.

    - cp_delay: worst *internal* combinational arc. For a registered module this
      is the reg-to-reg path (its own fmax limiter); for a purely combinational
      module (pp_delay == 0) it is the input->output through-delay, which adds
      into whatever combinational cloud the module sits in.
    - ir_delay: input port -> first capturing register. 0 for a purely
      combinational module (no register to terminate at; its delay is in cp_delay).
      A boundary stub: it belongs to one cloud, so an interior node must keep it 0.
    - or_delay: last launching register -> output port. 0 for a purely
      combinational module. Also a boundary stub (interior nodes keep it 0).

    The combinational path created when upstream U drives downstream D is
    U.or_delay + wire + (cp_delay of any combinational ops between) + D.ir_delay.
    """
    cp_delay: float = 0.0
    ir_delay: float = 0.0
    or_delay: float = 0.0


@dataclass
class hw_params:
    """
    Hardware contract bridging the functional model to generated RTL.

    Technology-independent (must match the sim exactly, identical at every corner):
    pp_delay is input->output latency in clock cycles (== register stages; 0 ==
    purely combinational). Composing ops with mismatched pp_delay desynchronizes
    streams on reconvergent paths, so pp_delay drives both stage count and path
    balancing.

    Technology- & corner-dependent (synthesis numbers, never affect functional
    equivalence): timing maps each pvt_corner to its measured ns delays. Empty
    until characterized by the RTL/STA flow.
    """
    pp_delay: int = 0
    timing: dict = field(default_factory=dict)   # {pvt_corner: timing}


class napl_base(torch.nn.Module):
    """
    Base class for all NAPL modules.
    This class initializes the global configuration and provides a common interface for all modules.
    """
    # Streaming modules advance timestep_cur once per call; single-shot
    # binary-domain classes override with False.
    streaming = True

    def __init__(self, config: dict={}, key_list: list=[], polarity_required: bool=False):
        super().__init__()
        # Load global configuration
        self.stype = global_config.stype
        self.ntype = global_config.ntype

        # check config
        if polarity_required is True:
            assert 'polarity' in config, logger.error(f'Missing key <polarity> in the input configuration.')
        check_config(config, key_list)
        self.polarity = check_polarity(config)
        self.name = check_name(config)
        
        self.timestep_cur = 0

        # hardware contract for RTL generation; ops override with their own values
        self.hw = hw_params()


    def tick(self):
        self.timestep_cur += 1


    def __call__(self, *args, **kwargs):
        """
        On streaming modules, every call advances timestep_cur by one before
        forward runs; single-shot modules (streaming = False) never tick.
        """
        # __call__ override, not register_forward_pre_hook: a hook on every module
        # forces nn.Module's slow call path per timestep.
        if self.streaming:
            self.tick()
        return super().__call__(*args, **kwargs)


    @property
    def valid(self):
        """
        True once forward() has run at least once since __init__/reset().
        Meaningful only for streaming modules; single-shot modules never tick,
        so it stays False.
        """
        return self.timestep_cur > 0


    def reset(self, verbose=False):
        """
        Reset the timestep and registered child modules.
        """
        self.timestep_cur = 0
        if verbose:
            logger.info(f'Reset module <{self.__class__.__name__}>.')
        for module in self.children():
            if hasattr(module, 'reset'):
                module.reset()
        self._reset()


    def _reset(self):
        pass


def napl_sim_timesteps(timestep_func):
    """
    This function is a decorator to simulate multiple timesteps in the NAPL framework.
    """
    @wraps(timestep_func)
    def timesteps_wrapper(self, *args, **kwargs):
        assert 'timesteps' in kwargs, \
            logger.error(f'Timesteps not specified in the arguments. Please provide <timesteps> as a keyword argument.')
        
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
            logger.error(f'Timesteps not specified in the arguments. Please provide <timesteps> as a keyword argument.')
        
        timesteps = kwargs.pop('timesteps', 256)  # Remove 'timesteps' from kwargs
        verbose = kwargs.pop('verbose', False)  # Remove 'timesteps' from kwargs
        if verbose:
            logger.info(f'Simulating <{timesteps}> timesteps in NAPL function...')

        for _ in range(timesteps):
            out = timestep_func(*args, **kwargs)
        return out
    
    return timesteps_wrapper

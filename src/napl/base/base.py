import os
import napl
import torch

from loguru import logger
from dataclasses import dataclass
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


# Initialize global configuration
# This will be used throughout the NAPL framework to maintain consistent data types across different modules and operations.
# The global configuration can be loaded from a YAML file or set directly.
@dataclass
class global_config_check:
    root_path: str = os.path.dirname(os.path.abspath(napl.__file__))
    config_file: str = str(os.path.join(root_path, 'base/global_config.yaml'))
    assert os.path.exists(config_file), logger.error(f'Global configuration file <{config_file}> does not exist.')
    config = read_yaml(config_file)

    stype = torch_dtype_map.get(config['global_config']['spike_type'], None)
    assert stype in [torch.float, torch.bfloat16, torch.int8], \
        logger.error(f'Invalid spike type: <{stype}>; legal types: [torch.float, torch.bfloat16, torch.int8].')
        
    ntype = torch_dtype_map.get(config['global_config']['non_spike_type'], None)
    assert ntype in [torch.float, torch.bfloat16], \
        logger.error(f'Invalid non-spike type: <{ntype}>; legal types: [torch.float, torch.bfloat16].')
    

global_config = global_config_check()


class napl_base(torch.nn.Module):
    """
    Base class for all NAPL modules.
    This class initializes the global configuration and provides a common interface for all modules.
    """
    def __init__(self):
        super().__init__()
        # Load global configuration
        self.stype = global_config.stype
        self.ntype = global_config.ntype


    def reset(self, verbose=False):
        """
        Reset the module to its initial state.
        This method should be overridden by subclasses to reset their specific parameters.
        """
        if verbose:
            logger.info(f'Reset module <{self.__class__.__name__}>.')
        for name, module in self.named_modules():
            if module is self:
                # skip self to prevent infinite recursion
                continue
            if hasattr(module, 'reset'):
                if verbose:
                    logger.info(f'    Reset module <{name}>.')
                module.reset()


def napl_sim_timesteps_class(timestep_func):
    """
    This function is a decorator to simulate multiple timesteps in the NAPL framework.
    """
    @wraps(timestep_func)
    def timesteps_wrapper(self, *args, **kwargs):
        assert 'timesteps' in kwargs, \
            logger.error(f'Timesteps not specified in the arguments. Please provide "timesteps" as a keyword argument.')
        
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
            logger.error(f'Timesteps not specified in the arguments. Please provide "timesteps" as a keyword argument.')
        
        timesteps = kwargs.pop('timesteps', 256)  # Remove 'timesteps' from kwargs
        verbose = kwargs.pop('verbose', False)  # Remove 'timesteps' from kwargs
        if verbose:
            logger.info(f'Simulating <{timesteps}> timesteps in NAPL function...')

        for _ in range(timesteps):
            out = timestep_func(*args, **kwargs)
        return out
    
    return timesteps_wrapper


import os
import torch

import napl
from napl.sim import algorithm, base, metric, module, operation, structure
from napl.sim.base import global_config


def test_global_config():
    """Verify the global configuration file exists and selects supported tensor types."""
    assert global_config.config_file is not None, 'Global config file should be set.'
    assert os.path.exists(global_config.config_file), f'Global config file {global_config.config_file} does not exist.'
    
    assert global_config.stype in [torch.float, torch.bfloat16, torch.int8], \
        f'Invalid spike type {global_config.stype}; legal types are: [torch.float, torch.bfloat16, torch.int8].'
    
    assert global_config.ntype in [torch.float, torch.bfloat16], \
        f'Invalid non-spike type {global_config.ntype}; legal types are: [torch.float, torch.bfloat16].'

    print(f'Global config file: {global_config.config_file}')
    print(f'Global non-spike type: {global_config.ntype}')
    print(f'Global spike type: {global_config.stype}')
    print('Test passed.')


def test_package_exports():
    """Verify the package export list covers exactly the simulation subpackage exports."""
    subpackage = set().union(*(
        pkg.__all__ for pkg in (base, operation, module, metric, structure, algorithm)
    ))

    assert set(napl.__all__) == subpackage, (
        'napl.__all__ diverged from the subpackage exports; '
        f'missing: {sorted(subpackage - set(napl.__all__))}, '
        f'extra: {sorted(set(napl.__all__) - subpackage)}.'
    )

    print(f'Exported names: {len(napl.__all__)}')
    print('Test passed.')


if __name__ == '__main__':
    test_global_config()
    test_package_exports()

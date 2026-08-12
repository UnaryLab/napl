import importlib.util
import os
import sys

import torch

import napl
import napl.utils as napl_utils
from napl.sim.base import global_config


def test_global_config():
    """Verify the global configuration file exists and selects supported tensor types."""
    spike_types = {
        'torch.float32': torch.float32,
        'torch.float': torch.float32,
        'torch.bfloat16': torch.bfloat16,
        'torch.int8': torch.int8,
    }
    non_spike_types = {
        'torch.float32': torch.float32,
        'torch.float': torch.float32,
        'torch.bfloat16': torch.bfloat16,
    }
    yaml_config = global_config.config['global_config']

    assert set(yaml_config) == {'spike_type', 'non_spike_type'}
    assert yaml_config['spike_type'] in spike_types
    assert yaml_config['non_spike_type'] in non_spike_types
    assert global_config.stype is spike_types[yaml_config['spike_type']]
    assert global_config.ntype is non_spike_types[yaml_config['non_spike_type']]
    assert os.path.isdir(global_config.root_path)
    assert os.path.isfile(global_config.config_file)
    assert os.path.commonpath(
        (global_config.root_path, global_config.config_file)
    ) == global_config.root_path

    print(f'Global config file: {global_config.config_file}')
    print(f'Global non-spike type: {global_config.ntype}')
    print(f'Global spike type: {global_config.stype}')
    print('Test passed.')


def _expect_invalid_global_config(config, fragment):
    module_name = 'napl.sim.base._invalid_global_config_test'
    module_path = os.path.join(
        os.path.dirname(global_config.config_file),
        'base.py',
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    original_read_yaml = napl_utils.read_yaml

    try:
        napl_utils.read_yaml = lambda _: {'global_config': config}
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except AssertionError as error:
            assert fragment in str(error), error
        else:
            raise AssertionError(f'{fragment!r} error was not raised')
    finally:
        napl_utils.read_yaml = original_read_yaml
        sys.modules.pop(module_name, None)


def test_global_config_rejects_invalid_dtypes():
    """Verify isolated module loading rejects invalid spike and non-spike dtypes."""
    _expect_invalid_global_config(
        {'spike_type': 'torch.int16', 'non_spike_type': 'torch.float32'},
        'Invalid spike type',
    )
    _expect_invalid_global_config(
        {'spike_type': 'torch.int8', 'non_spike_type': 'torch.int8'},
        'Invalid non-spike type',
    )


def test_package_exports():
    """Verify the package root re-exports no simulation classes."""
    for name in ('mul_gaines', 'linear_ugemm', 'accuracy', 'napl_base'):
        assert not hasattr(napl, name), (
            f'napl still re-exports {name!r}; public classes are imported from '
            'napl.sim.<subpackage>, not the package root.'
        )
    assert not getattr(napl, '__all__', None), 'napl.__all__ must not list re-exports.'

    print('Test passed.')


if __name__ == '__main__':
    test_global_config()
    test_global_config_rejects_invalid_dtypes()
    test_package_exports()

"""Tests for the base constructor filling flux_stability from packaged profiling results.

Verifies that napl_base fills flux_stability from each class's package
profiling_results.yaml keyed by class name and polarity, always as a list holding one
value per profiled input-stability level in ascending order, falling back to ``[1.0]``
when the class, polarity, or yaml is absent, and that the loader is cached.
"""

import shutil
import sys
import tempfile
from importlib import invalidate_caches
from importlib.resources import files
from pathlib import Path

import numpy
import torch
import yaml

from napl.sim.base.base import _load_flux_map
from napl.sim.metric import stability_flux
from napl.sim.module import linear_ugemm
from napl.sim.operation import mul_gaines, relu_fxp
from napl.utils._shared_test import devices

PROBE_SOURCE = """
from napl.sim.base import napl_base


class probe_op(napl_base):
    def __init__(self, config):
        super().__init__(config, ['polarity'])
"""


def _yaml_flux(package, cls, polarity):
    """Return the packaged flux stability of one class and polarity, as the yaml holds it."""
    text = files(package).joinpath('profiling_results.yaml').read_text()
    return yaml.safe_load(text)[cls][polarity]['flux_stability']


def _expected_levels(raw):
    """State the expected level list for a packaged value, via numpy rather than the loader's rule."""
    return list(numpy.atleast_1d(raw))


def test_profiled_operation():
    """A profiled op's flux_stability equals its operation profiling level list on every device."""
    expected = _expected_levels(_yaml_flux('napl.sim.operation', 'mul_gaines', 'bipolar'))
    for device in devices():
        operation = mul_gaines({'polarity': 'bipolar'}).to(device)
        assert operation.flux_stability == expected


def test_profiled_module():
    """A profiled module's flux_stability equals its module profiling level list on every device."""
    expected = _expected_levels(_yaml_flux('napl.sim.module', 'linear_ugemm', 'bipolar'))
    config = {'polarity': 'bipolar', 'timestep': 64, 'generator': 'sobol',
              'dim': 1, 'scale': None, 'width': 12}
    for device in devices():
        weight = torch.ones(2, 2).to(device)
        module = linear_ugemm(weight, None, config).to(device)
        assert module.flux_stability == expected


def test_multi_level_yaml_fills_every_level():
    """A multi-level yaml reaches a constructed instance as the whole ordered level list."""
    levels = [0.25, 0.5, 0.75]
    root = Path(tempfile.mkdtemp())
    package = root / 'flux_probe'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'probe.py').write_text(PROBE_SOURCE)
    (package / 'profiling_results.yaml').write_text(yaml.safe_dump(
        {'probe_op': {'bipolar': {'flux_stability': levels,
                                  'input_stability': [0.1, 0.5, 0.9],
                                  'rmse': [0.3, 0.2, 0.1]}}}))
    sys.path.insert(0, str(root))
    invalidate_caches()
    _load_flux_map.cache_clear()
    try:
        from flux_probe.probe import probe_op
        assert probe_op({'polarity': 'bipolar'}).flux_stability == levels
    finally:
        sys.path.remove(str(root))
        for module in ['flux_probe.probe', 'flux_probe']:
            sys.modules.pop(module, None)
        _load_flux_map.cache_clear()
        shutil.rmtree(root)


def test_unprofiled_metric_falls_back():
    """A metric (its package has no profiling_results.yaml) falls back to [1.0] on every device."""
    for device in devices():
        metric = stability_flux(torch.ones(1), torch.ones(1)).to(device)
        assert metric.flux_stability == [1.0]


def test_polarity_less_falls_back():
    """A polarity-less class (polarity None, no matching yaml key) falls back to [1.0] on every device."""
    for device in devices():
        operation = relu_fxp().to(device)
        assert operation.polarity is None
        assert operation.flux_stability == [1.0]


def test_fill_is_a_copy():
    """Editing one instance's flux_stability leaves the next construction unchanged."""
    first = mul_gaines({'polarity': 'bipolar'})
    expected = list(first.flux_stability)
    first.flux_stability[0] = -1.0
    assert mul_gaines({'polarity': 'bipolar'}).flux_stability == expected


def test_loader_is_cached():
    """A second construction of a same-package class hits the cache instead of re-reading the yaml."""
    _load_flux_map('napl.sim.operation')
    before = _load_flux_map.cache_info().hits
    mul_gaines({'polarity': 'bipolar'})
    mul_gaines({'polarity': 'unipolar'})
    assert _load_flux_map.cache_info().hits > before


if __name__ == '__main__':
    test_profiled_operation()
    test_profiled_module()
    test_multi_level_yaml_fills_every_level()
    test_unprofiled_metric_falls_back()
    test_polarity_less_falls_back()
    test_fill_is_a_copy()
    test_loader_is_cached()
    print('Test passed.')

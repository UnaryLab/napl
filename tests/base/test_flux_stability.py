"""Tests for the base constructor filling flux_stability from packaged profiling yamls.

Verifies that napl_base fills flux_stability from each class's package
flux_stability.yaml keyed by class name and polarity, falling back to 1.0 when
the class, polarity, or yaml is absent, and that the loader is cached.
"""

from importlib.resources import files

import torch
import yaml

from napl.sim.base.base import _load_flux_map
from napl.sim.metric import stability_flux
from napl.sim.module import linear_ugemm
from napl.sim.operation import mul_gaines, relu_fxp
from napl.utils._shared_test import devices


def _yaml_value(package, cls, polarity):
    text = files(package).joinpath('flux_stability.yaml').read_text()
    return yaml.safe_load(text)[cls][polarity]


def test_profiled_operation():
    """A profiled op's flux_stability equals its operation-yaml value on every device."""
    expected = _yaml_value('napl.sim.operation', 'mul_gaines', 'bipolar')
    for device in devices():
        operation = mul_gaines({'polarity': 'bipolar'}).to(device)
        assert operation.flux_stability == expected


def test_profiled_module():
    """A profiled module's flux_stability equals its module-yaml value on every device."""
    expected = _yaml_value('napl.sim.module', 'linear_ugemm', 'bipolar')
    config = {'polarity': 'bipolar', 'timestep': 64, 'generator': 'sobol',
              'dim': 1, 'scale': None, 'width': 12}
    for device in devices():
        weight = torch.ones(2, 2).to(device)
        module = linear_ugemm(weight, None, config).to(device)
        assert module.flux_stability == expected


def test_unprofiled_metric_falls_back():
    """A metric (its package has no flux_stability.yaml) falls back to 1.0 on every device."""
    for device in devices():
        metric = stability_flux(torch.ones(1), torch.ones(1)).to(device)
        assert metric.flux_stability == 1.0


def test_polarity_less_falls_back():
    """A polarity-less class (polarity None, no matching yaml key) falls back to 1.0 on every device."""
    for device in devices():
        operation = relu_fxp().to(device)
        assert operation.polarity is None
        assert operation.flux_stability == 1.0


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
    test_unprofiled_metric_falls_back()
    test_polarity_less_falls_back()
    test_loader_is_cached()
    print('Test passed.')

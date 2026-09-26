"""Tests for correlation-contract metadata and decorrelator output correlation."""

import torch

from napl.sim.base import napl_base
from napl.sim.metric import correlation
from napl.sim.operation import decorr, desync, encode, sync, sync_skewed


def test_correlation_output_contracts():
    """Verify producer output contracts and the empty base defaults."""
    assert napl_base().correlation_i == {}
    assert napl_base().correlation_o == {}
    assert sync().correlation_o == {('output_0', 'output_1'): 'pos'}
    assert sync_skewed().correlation_o == {('output_0', 'output_1'): 'pos'}
    assert desync().correlation_o == {('output_0', 'output_1'): 'neg'}
    depth_one = decorr({'polarity': 'unipolar', 'depth': 1, 'timestep': 256,
                        'generator': 'sys'})
    assert depth_one.correlation_o == {('output_0', 'output_1'): 'pos'}
    for depth in (2, 4, 16):
        assert decorr({'polarity': 'unipolar', 'depth': depth, 'timestep': 256,
                       'generator': 'sys'}).correlation_o == {}


def test_decorr_correlation_evidence():
    """Print the output SCC at the supported decorrelator depths."""
    timestep = 256
    codec = {'polarity': 'unipolar', 'timestep': timestep,
             'generator': 'sobol', 'dim': 1}
    value = torch.tensor([0.3])
    for depth in (1, 2, 4, 16):
        encoder_0 = encode(codec)
        encoder_1 = encode(codec)
        operation = decorr({'polarity': 'unipolar', 'depth': depth,
                            'timestep': timestep, 'generator': 'sys', 'seed': 7})
        metric = correlation()
        for _ in range(timestep):
            output_0, output_1 = operation(encoder_0(value), encoder_1(value))
            metric(output_0, output_1)
        print(f'[decorr depth={depth}] output_scc={metric.correlation.item():.4f}')


if __name__ == '__main__':
    test_correlation_output_contracts()
    test_decorr_correlation_evidence()
    print('Test passed.')

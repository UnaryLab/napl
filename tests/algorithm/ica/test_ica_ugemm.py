import torch

from napl.sim.algorithm.ica import ica_ugemm
from napl.sim.base import global_config
from napl.sim.module import linear_ugemm
from napl.sim.operation import encode, tanh_pn
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256
MIXING = torch.tensor([
    [0.8, 0.6],
    [-0.6, 0.8],
], dtype=global_config.ntype)
UNMIXING = torch.linalg.inv(MIXING)


def _sources(rows=512):
    phase = torch.linspace(-1.0, 1.0, rows, dtype=global_config.ntype)
    return torch.stack((0.5 * phase, 0.5 * torch.sign(torch.sin(phase * 9))), dim=-1)


def _mixed(sources):
    return sources @ MIXING.T


def make_operation(polarity, timestep, device):
    return ica_ugemm(
        UNMIXING,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
            'width': 8,
        },
    )


def make_values(polarity):
    return (_mixed(_sources()),)


def make_random_perf_values(polarity):
    return (_mixed(_sources(65536)),)


def analytic_reference(values, polarity):
    return values[0] @ UNMIXING.T


def known_answer_case(polarity):
    sources = torch.tensor(
        [[-0.5, 0.5], [0.25, -0.25]], dtype=global_config.ntype
    )
    return (_mixed(sources),), sources


def _contrast_composition_check():
    """Verify both ICA outputs bit-exactly match their bare child operations."""
    mixed_cpu = _mixed(_sources(32))
    for device in devices():
        mixed = mixed_cpu.to(device)
        encoder = encode({
            'polarity': 'bipolar', 'timestep': 32,
            'generator': 'sobol', 'dim': 2,
        }).to(device)
        unmix_reference = linear_ugemm(
            UNMIXING,
            None,
            {
                'polarity': 'bipolar', 'timestep': 32,
                'generator': 'sobol', 'dim': 1, 'scale': 1, 'width': 8,
            },
        ).to(device)
        contrasted = ica_ugemm(
            UNMIXING,
            {
                'polarity': 'bipolar', 'timestep': 32,
                'generator': 'sobol', 'dim': 1, 'width': 8,
                'contrast': True, 'contrast_depth': 3,
            },
        ).to(device)
        contrast_reference = tanh_pn({'depth': 3}).to(device)
        first_trace = []
        for _ in range(32):
            input_spike = encoder(mixed)
            expected_source = unmix_reference(input_spike)
            source, contrast_spike = contrasted(input_spike)
            assert torch.equal(source, expected_source)
            assert torch.equal(contrast_spike, contrast_reference(expected_source))
            assert source.shape == (32, 2)
            assert contrast_spike.shape == (32, 2)
            assert contrasted.unmix.mul.seq_idx.shape == (32, 1, 2)
            assert contrasted.unmix.mul.seq_idx_inv.shape == (32, 1, 2)
            assert contrasted.unmix.acc.accumulator.shape == (32, 2)
            assert contrasted.contrast_op.cnt.shape == (32, 2)
            first_trace.append((source.clone(), contrast_spike.clone()))

        encoder.reset()
        unmix_reference.reset()
        contrasted.reset()
        contrast_reference.reset()
        for expected_source, expected_contrast in first_trace:
            input_spike = encoder(mixed)
            plain_source = unmix_reference(input_spike)
            source, contrast_spike = contrasted(input_spike)
            assert torch.equal(source, plain_source)
            assert torch.equal(source, expected_source)
            assert torch.equal(contrast_spike, expected_contrast)
            assert torch.equal(contrast_spike, contrast_reference(plain_source))


CONFIG = {
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [2],
    'timesteps': TIMESTEPS,
    'warmup_runs': 1,
    'trials': 3,
    'extra_checks': _contrast_composition_check,
}


def test_ica_ugemm():
    """Verify bipolar-only ICA on mixed inputs restricted to [-0.7, 0.7] so |W x| <= 1."""
    streaming_suite(CONFIG)


def test_ica_ugemm_rejects_invalid_config():
    """Verify ICA rejects unsupported polarity, contrast, and contrast depth values."""
    base = {
        'polarity': 'bipolar', 'timestep': 32,
        'generator': 'sobol', 'dim': 1, 'width': 8,
    }
    cases = (
        ({**base, 'polarity': 'unipolar'},
         'ica_ugemm supports only bipolar spike streams.'),
        ({**base, 'contrast': 1},
         'Invalid contrast: <1>; legal values: a boolean.'),
        ({**base, 'contrast_depth': 0},
         'Invalid contrast_depth: <0>; legal values: an integer of at least 1.'),
        ({**base, 'contrast_depth': True},
         'Invalid contrast_depth: <True>; legal values: an integer of at least 1.'),
    )
    for config, message in cases:
        try:
            ica_ugemm(UNMIXING, config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'ica_ugemm accepted invalid config {config}')


if __name__ == '__main__':
    test_ica_ugemm()
    test_ica_ugemm_rejects_invalid_config()

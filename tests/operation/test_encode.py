import math

import torch

from napl.sim.base import global_config
from napl.sim.metric import accuracy
from napl.sim.operation import (
    encode,
    gen_num_seq,
    get_lfsr_seq,
    get_sysrand_seq,
    input_scale,
)
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import benchmark, devices


def _run(stream_encoder, stream_accuracy, input, timestep):
    for _ in range(timestep):
        stream_accuracy(stream_encoder(input))


def test_encode():
    """Verify encoder fidelity, reset replay, device coverage, and timing."""
    torch.manual_seed(0)
    config = {
        'polarity': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
        'name': 'spike_accuracy',
        'dim': 1,
    }
    input_cpu = gen_rand_tensor(
        config['polarity'],
        shape=(4096,),
        width=math.log2(config['timestep']),
    ).type(global_config.ntype)

    cpu_runtime = None
    for device in devices():
        stream_encoder = encode(config).to(device)
        stream_accuracy = accuracy(config).to(device)
        input = input_cpu.to(device)

        _run(stream_encoder, stream_accuracy, input, config['timestep'])
        error, _ = stream_accuracy.analyze(input, verbose=True)
        print(f'[{device}] rmse={error.pow(2).mean().sqrt():.4f}')
        assert stream_encoder.timestep_cur == config['timestep']
        first = stream_accuracy.spike_value.detach().cpu().clone()

        stream_encoder.reset()
        stream_accuracy.reset()
        assert stream_encoder.timestep_cur == 0
        assert not stream_accuracy.valid
        _run(stream_encoder, stream_accuracy, input, config['timestep'])
        assert torch.equal(first, stream_accuracy.spike_value.cpu())

        device_runtime = benchmark(
            lambda values: _run(
                stream_encoder,
                stream_accuracy,
                values[0],
                config['timestep'],
            ),
            (input_cpu,),
            device,
            warmup_runs=1,
            trials=3,
            prepare=lambda: (
                stream_encoder.reset(),
                stream_accuracy.reset(),
            ),
        )
        if device == 'cpu':
            cpu_runtime = device_runtime
        assert cpu_runtime is not None
        print(
            f'[{device}] device_runtime={device_runtime * 1e3:.1f}ms, '
            f'cpu_runtime={cpu_runtime * 1e3:.1f}ms, '
            f'speedup={cpu_runtime / device_runtime:.2f}x'
        )


def test_encode_rank2():
    """Verify encoder preserves rank-two shape and meets stochastic error tolerance."""
    config = {
        'polarity': 'bipolar',
        'timestep': 16,
        'generator': 'sobol',
    }
    input_cpu = torch.tensor([[-0.75, -0.25], [0.25, 0.75]])

    for device in devices():
        stream_encoder = encode(config).to(device)
        stream_accuracy = accuracy(config).to(device)
        input = input_cpu.to(device)

        _run(stream_encoder, stream_accuracy, input, config['timestep'])
        error, _ = stream_accuracy.analyze(input)

        assert stream_accuracy.spike_value.shape == input.shape
        assert error.shape == input.shape
        print(f'[{device}] rmse={error.pow(2).mean().sqrt():.4f}')


def test_number_sequences():
    """Verify supported random-number generators produce valid deterministic sequences."""
    width = 4
    length = 2**width

    torch.manual_seed(0)
    sys_seq = get_sysrand_seq(width)
    assert sys_seq.dtype == torch.float32
    assert torch.equal(
        (sys_seq * length).sort().values,
        torch.arange(length, dtype=sys_seq.dtype),
    )

    lfsr_a = get_lfsr_seq(width=width, seed=1)
    lfsr_b = get_lfsr_seq(width=width, seed=1)
    assert torch.equal(lfsr_a, lfsr_b)
    assert lfsr_a.shape == (length,)
    assert torch.all((lfsr_a >= 0) & (lfsr_a < 1))

    # The classic generator remains the maximal 15-state cycle followed by its
    # repeated opening state. The extended generator replaces that repeat with
    # zero, so all k/16 rates are exactly representable in one 16-step period.
    classic = gen_num_seq({'width': width, 'generator': 'lfsr'})
    expected_classic_states = torch.tensor(
        [1, 8, 12, 14, 15, 7, 11, 5, 10, 13, 6, 3, 9, 4, 2, 1],
        dtype=torch.long,
    )
    assert torch.equal(classic.mul(length).round().type(torch.long),
                       expected_classic_states)

    extended = gen_num_seq({'width': width, 'generator': 'lfsr_ext'})
    expected_extended_states = torch.tensor(
        [1, 0, 8, 12, 14, 15, 7, 11, 5, 10, 13, 6, 3, 9, 4, 2],
        dtype=torch.long,
    )
    assert torch.equal(extended.mul(length).round().type(torch.long),
                       expected_extended_states)
    assert torch.equal(
        extended.mul(length).round().sort().values.type(torch.long),
        torch.arange(length),
    )
    try:
        gen_num_seq({'width': width, 'generator': 'lfsr_ext', 'taps': [4, 2]})
    except AssertionError as error:
        assert str(error) == (
            'Invalid extended LFSR cycle for width <4>; feedback taps must '
            'produce all 15 nonzero states.'
        )
    else:
        raise AssertionError('lfsr_ext accepted a non-maximal feedback cycle')

    grid = torch.arange(length + 1, dtype=global_config.ntype).div(length)
    extended_encoder = encode({
        'polarity': 'unipolar',
        'timestep': length,
        'generator': 'lfsr_ext',
    })
    extended_counts = torch.stack([extended_encoder(grid) for _ in range(length)]).sum(0)
    assert torch.equal(extended_counts, torch.arange(length + 1))

    classic_encoder = encode({
        'polarity': 'unipolar',
        'timestep': length,
        'generator': 'lfsr',
    })
    classic_counts = torch.stack([classic_encoder(grid) for _ in range(length)]).sum(0)
    assert classic_counts[1].item() == 0
    print(
        f'[lfsr width={width}] classic_count@1/{length}={classic_counts[1].item()}, '
        f'extended_exact_grid={torch.equal(extended_counts, torch.arange(length + 1))}'
    )

    # A seed reducing to zero has no LFSR state of its own and takes the all-ones
    # state; callers deriving one seed per input feature otherwise fail on
    # whichever term lands on a multiple of the period.
    for seed in [0, length, 2 * length]:
        zero_seed = get_lfsr_seq(width=width, seed=seed)
        assert zero_seed.shape == (length,)
        assert torch.all((zero_seed > 0) & (zero_seed < 1))
        # The substitution aliases onto the all-ones seed. Some collision is
        # unavoidable, since 2**width seeds map onto 2**width - 1 states.
        assert torch.equal(zero_seed, get_lfsr_seq(width=width, seed=length - 1))

    # A seeded sys sequence must repeat, an unseeded one must not, and seeding
    # must not disturb the global generator other callers draw from.
    seeded = gen_num_seq({'width': width, 'generator': 'sys', 'seed': 42})
    assert torch.equal(seeded, gen_num_seq({'width': width, 'generator': 'sys', 'seed': 42}))
    assert not torch.equal(seeded, gen_num_seq({'width': width, 'generator': 'sys', 'seed': 43}))
    torch.manual_seed(0)
    assert not torch.equal(gen_num_seq({'width': width, 'generator': 'sys'}),
                           gen_num_seq({'width': width, 'generator': 'sys'}))
    torch.manual_seed(7)
    before = torch.randn(3)
    torch.manual_seed(7)
    gen_num_seq({'width': width, 'generator': 'sys', 'seed': 42})
    assert torch.equal(before, torch.randn(3))

    # Ascending thresholds put the falling edge later for larger values.
    temporal = gen_num_seq({'width': width, 'generator': 'temporal'})
    expected = torch.arange(length, dtype=global_config.ntype)
    expected.div_(length)
    assert torch.equal(temporal, expected)


def test_lfsr_taps_validation():
    """Verify get_lfsr_seq rejects taps that are not a non-empty list and accepts valid taps."""
    for taps in ([], 'not-a-list', 4, (4, 1)):
        try:
            get_lfsr_seq(width=4, taps=taps)
        except AssertionError as error:
            assert str(error) == f'Error: the input taps {taps} needs to be a non-empty list.'
        else:
            raise AssertionError(f'get_lfsr_seq accepted the invalid taps <{taps}>.')

    assert get_lfsr_seq(width=4, taps=[4, 1]).numel() == 16
    assert get_lfsr_seq(width=5, taps=[5, 3]).numel() == 32


def test_input_scale():
    """Verify input_scale normalizes a tensor by its largest absolute value."""
    input = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
    assert torch.equal(input_scale(input), input / 4)


if __name__ == '__main__':
    test_encode()
    test_encode_rank2()
    test_number_sequences()
    test_lfsr_taps_validation()
    test_input_scale()
    print('Test passed.')

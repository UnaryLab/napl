import torch

from napl.utils._shared_test import (
    _multirank_inputs,
    timer,
    assert_inputs_equal,
    benchmark,
    clone_inputs,
    single_shot_suite,
    streaming_suite,
)
from napl.utils._shared_test import devices


def test_timer():
    """Verify timer records successful and exceptional regions without swallowing errors."""
    for device in devices():
        with timer(device) as elapsed:
            torch.arange(64, dtype=torch.float32, device=device).square().sum()
        assert elapsed.seconds > 0

        try:
            with timer(device) as failed:
                raise RuntimeError('expected')
        except RuntimeError:
            pass
        else:
            raise AssertionError('timer swallowed an exception')
        assert failed.seconds > 0


def _expect_mismatch(candidate_inputs, baseline_inputs):
    try:
        assert_inputs_equal(candidate_inputs, baseline_inputs)
    except AssertionError:
        return
    raise AssertionError('mismatched inputs were accepted')


def test_clone_inputs_and_equality():
    """Verify cloned inputs are independent, device-local, and compared with configured tolerances."""
    shared_cpu = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.tensor([1, 0, 1], dtype=torch.int8),
    )
    for device in devices():
        candidate_inputs = clone_inputs(shared_cpu, device)
        baseline_inputs = clone_inputs(shared_cpu, device)

        assert all(value.device.type == device for value in candidate_inputs)
        assert all(
            candidate is not baseline
            for candidate, baseline in zip(
                candidate_inputs, baseline_inputs
            )
        )
        assert all(
            candidate.data_ptr() != baseline.data_ptr()
            for candidate, baseline in zip(
                candidate_inputs, baseline_inputs
            )
        )
        assert_inputs_equal(candidate_inputs, baseline_inputs)

        candidate_inputs[0].add_(1)
        _expect_mismatch(candidate_inputs, baseline_inputs)
        torch.testing.assert_close(
            baseline_inputs[0], shared_cpu[0].to(device)
        )


def test_multirank_inputs():
    """Verify multirank input expansion preserves scalars and adds higher-rank variants."""
    scalar = torch.tensor(1.0)
    vector = torch.arange(3)
    matrix = torch.arange(6).reshape(2, 3)

    result = _multirank_inputs((scalar, vector, matrix))

    assert [value.shape for value in result] == [
        torch.Size([1, 1]),
        torch.Size([1, 3]),
        torch.Size([2, 3]),
    ]
    assert all(value.ndim >= 2 for value in result)
    assert torch.equal(result[0].reshape(()), scalar)
    assert torch.equal(result[1].reshape(3), vector)
    assert result[2] is matrix


def test_benchmark():
    """Verify benchmarking reports positive runtimes and enforces configured device bounds."""
    warmup_runs = 2
    trials = 4
    runtimes = {}
    assert devices()[0] == 'cpu'

    for device in devices():
        shared_inputs = (torch.arange(64, dtype=torch.float32),)
        prepare_count = 0
        seen_inputs = []

        def prepare():
            nonlocal prepare_count
            prepare_count += 1

        def run(inputs):
            assert inputs[0].device.type == device
            seen_inputs.append(inputs[0])
            inputs[0].add_(1)
            inputs[0].square().sum()

        runtimes[device] = benchmark(
            run,
            shared_inputs,
            device,
            warmup_runs=warmup_runs,
            trials=trials,
            prepare=prepare,
        )

        run_count = warmup_runs + trials
        assert prepare_count == run_count
        assert len(seen_inputs) == run_count
        assert_inputs_equal(
            shared_inputs,
            (torch.arange(64, dtype=torch.float32),),
        )
        assert runtimes[device] > 0
        speedup = runtimes['cpu'] / runtimes[device]
        assert speedup > 0
        print(
            f'[{device}] device_runtime={runtimes[device]:.6f}s, '
            f'cpu_runtime={runtimes["cpu"]:.6f}s, '
            f'speedup={speedup:.2f}x'
        )


def _mul_and_operation(polarity, timestep, device):
    from napl.sim.operation import mul_and
    return mul_and({'polarity': polarity})


def _mul_and_values(polarity):
    from napl.utils import gen_rand_tensor
    return (
        gen_rand_tensor(polarity, shape=(128,), width=6),
        gen_rand_tensor(polarity, shape=(128,), width=6),
    )


def _mul_and_reference(values, polarity):
    return values[0] * values[1]


def _mul_and_known_answer(polarity):
    # Both polarities encode 1.0 as all ones, so the product is exact.
    return (torch.tensor([1.0]), torch.tensor([1.0])), torch.tensor([1.0]), 0.0


def test_streaming_suite():
    """Verify streaming_suite exercises fidelity, known answers, reset replay, rank, and timing."""
    streaming_suite({
        'polarities': ('unipolar', 'bipolar'),
        'tolerance_scale': 1.0,
        'make_operation': _mul_and_operation,
        'make_values': _mul_and_values,
        'analytic_reference': _mul_and_reference,
        'known_answer_case': _mul_and_known_answer,
        'timesteps': 64,
        'warmup_runs': 1,
        'trials': 3,
    })


def _fxp_pair():
    from napl.sim.module import linear_fxp
    torch.manual_seed(1)
    in_features, out_features = 8, 4
    reference = torch.nn.Linear(in_features, out_features)
    candidate = linear_fxp(
        in_features,
        out_features,
        bias=True,
        weight_ext=reference.weight.data,
        bias_ext=reference.bias.data,
        config={
            'widthi': 8,
            'widthw': 8,
            'quantilei': 1,
            'quantilew': 1,
            'rounding': 'round',
        },
    )
    return candidate, reference


def _fxp_inputs():
    return (torch.rand(8, 8) * 2 - 1,)


def _fxp_known_answer():
    # Zero input isolates the bias exactly.
    candidate, reference = _fxp_pair()
    expected = reference.bias.data.unsqueeze(0).expand(2, 4).clone()
    return candidate, (torch.zeros(2, 8),), expected


def _fxp_gradient_case():
    candidate, _ = _fxp_pair()
    return candidate, (torch.rand(4, 8) * 2 - 1,)


def _fxp_expected_gradients(candidate, inputs, grad_output):
    # linear_fxp uses the exact linear STE gradient.
    grad_input = grad_output.matmul(candidate.weight.detach())
    grad_weight = grad_output.t().matmul(inputs[0].detach())
    grad_bias = grad_output.sum(0)
    return (grad_input,), {'weight': grad_weight, 'bias': grad_bias}


def test_single_shot_suite():
    """Verify single_shot_suite exercises quantization, known answers, gradients, rank, and timing."""
    single_shot_suite({
        'quantization_atol': 0.05,
        'known_answer_atol': 0.0,
        'gradient_atol': 1e-5,
        'gradient_rtol': 1e-5,
        'make_module_pair': _fxp_pair,
        'make_inputs': _fxp_inputs,
        'known_answer_case': _fxp_known_answer,
        'gradient_case': _fxp_gradient_case,
        'expected_ste_gradients': _fxp_expected_gradients,
        'warmup_runs': 1,
        'trials': 3,
    })


def test_suite_config_validation():
    """Verify shared suites reject missing requirements and unknown configuration keys."""
    def expect(error_type, fragment, cfg, suite):
        try:
            suite(cfg)
        except error_type as error:
            assert fragment in str(error), error
            return
        raise AssertionError(f'{fragment!r} error was not raised')

    expect(NotImplementedError, "set 'polarities'", {}, streaming_suite)
    expect(
        NotImplementedError,
        "set 'quantization_atol'",
        {},
        single_shot_suite,
    )
    # Removed options fail explicitly by name.
    expect(
        ValueError,
        "unknown keys: ['seed']",
        {'polarities': ('unipolar',), 'seed': 1},
        streaming_suite,
    )
    expect(
        ValueError,
        "unknown keys: ['bogus_key']",
        {'bogus_key': 1},
        single_shot_suite,
    )


if __name__ == '__main__':
    test_timer()
    test_clone_inputs_and_equality()
    test_multirank_inputs()
    test_benchmark()
    test_streaming_suite()
    test_single_shot_suite()
    test_suite_config_validation()

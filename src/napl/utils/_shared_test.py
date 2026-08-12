import copy
from inspect import signature
from statistics import median
from time import perf_counter
from typing import Callable, Optional, Sequence, Tuple

import torch

InputTuple = Tuple[torch.Tensor, ...]
Runner = Callable[[InputTuple], object]
Prepare = Optional[Callable[[], None]]
Device = str


# List available compute devices.
def devices():
    dev = ['cpu']
    if torch.cuda.is_available():
        dev.append('cuda')
    if torch.backends.mps.is_available():
        dev.append('mps')
    return dev


# Synchronize pending device work.
def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


# Measure elapsed time for device operations.
class timer:
    # Prepare timing for the selected device.
    def __init__(self, device):
        self.device = device
        self.seconds = None


    # Start device-aware timing.
    def __enter__(self):
        sync(self.device)
        self._start = perf_counter()
        return self


    # Stop timing and store elapsed seconds.
    def __exit__(self, exc_type, exc_value, traceback):
        sync(self.device)
        self.seconds = perf_counter() - self._start


# Clone inputs, optionally moving them to a device.
def clone_inputs(
    inputs: Sequence[torch.Tensor], device: Optional[Device] = None
) -> InputTuple:
    if device is None:
        return tuple(value.clone() for value in inputs)
    return tuple(value.clone().to(device) for value in inputs)


# Normalize inputs to at least two dimensions.
def _multirank_inputs(inputs: Sequence[torch.Tensor]) -> InputTuple:
    return tuple(
        value.reshape((1,) * (2 - value.ndim) + tuple(value.shape))
        if value.ndim < 2 else value
        for value in inputs
    )


# Draw uniform random inputs over each operand's min-to-max legal span, keeping shape, dtype, and device.
def _legal_range_random(inputs: Sequence[torch.Tensor]) -> InputTuple:
    drawn = []
    for value in inputs:
        low = value.min()
        high = value.max()
        drawn.append(low + (high - low) * torch.rand_like(value))
    return tuple(drawn)


# Verify that two input sequences match exactly.
def assert_inputs_equal(
    candidate_inputs: Sequence[torch.Tensor],
    baseline_inputs: Sequence[torch.Tensor],
) -> None:
    assert len(candidate_inputs) == len(baseline_inputs), (
        f'input count differs: {len(candidate_inputs)} != '
        f'{len(baseline_inputs)}'
    )
    for index, (candidate, baseline) in enumerate(
        zip(candidate_inputs, baseline_inputs)
    ):
        assert candidate.shape == baseline.shape, (
            f'input[{index}] shape differs: '
            f'{candidate.shape} != {baseline.shape}'
        )
        assert candidate.dtype == baseline.dtype, (
            f'input[{index}] dtype differs: '
            f'{candidate.dtype} != {baseline.dtype}'
        )
        assert candidate.device == baseline.device, (
            f'input[{index}] device differs: '
            f'{candidate.device} != {baseline.device}'
        )
        assert torch.equal(candidate, baseline), (
            f'input[{index}] values differ'
        )


# Measure median runner runtime.
def benchmark(
    runner: Runner,
    shared_inputs: Sequence[torch.Tensor],
    device: Device,
    *,
    warmup_runs: int,
    trials: int,
    prepare: Prepare = None,
) -> float:
    assert warmup_runs >= 0
    assert trials > 0
    times = []

    for run_index in range(warmup_runs + trials):
        inputs = clone_inputs(shared_inputs, device)
        if prepare is not None:
            prepare()
        with timer(device) as elapsed:
            runner(inputs)
        if run_index >= warmup_runs:
            times.append(elapsed.seconds)

    runtime = median(times)
    assert runtime > 0
    return runtime


# Validate and merge suite configuration.
def _check_config(cfg, required, defaults, suite):
    unknown = set(cfg) - set(required) - set(defaults)
    if unknown:
        raise ValueError(f'{suite} config has unknown keys: {sorted(unknown)}')
    for key in required:
        if cfg.get(key) is None:
            raise NotImplementedError(f"{suite} config: set '{key}'")
    merged = dict(defaults)
    merged.update(cfg)
    return merged


_SEED = 0
_STATE_TIMESTEPS = 16

_STREAMING_REQUIRED = (
    'polarities',
    'make_operation',
    'make_values',
    'analytic_reference',
    'known_answer_case',
)
_STREAMING_DEFAULTS = {
    'make_random_perf_values': None,
    'apply_operation': None,
    'input_polarities': None,
    'output_polarity': None,
    'encoder_dims': None,
    'encoder_generators': None,
    'make_readout': None,
    'extra_checks': None,
    'timesteps': 256,
    'state_timesteps': _STATE_TIMESTEPS,
    'warmup_runs': 2,
    'trials': 7,
}


# Normalize apply_operation to the (operation, spikes, values) form.
def _wrap_apply_operation(apply_operation):
    if apply_operation is None:
        return lambda operation, spikes, values: operation(*spikes)
    parameters = signature(apply_operation).parameters
    if len(parameters) >= 3:
        return apply_operation
    return lambda operation, spikes, values: apply_operation(operation, spikes)


# Build codec settings for a pipeline.
def _codec_config(polarity, timestep, dim):
    return {
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': dim,
    }


# Resolve a polarity-dependent option.
def _resolve_streaming_option(option, polarity):
    if callable(option):
        return option(polarity)
    return option


# Construct the encoder, operation, and decoder pipeline.
def _make_pipeline(polarity, timestep, device, input_count, cfg):
    input_polarities = _resolve_streaming_option(
        cfg['input_polarities'], polarity
    )
    if input_polarities is None:
        input_polarities = [polarity] * input_count
    assert len(input_polarities) == input_count

    encoder_dims = _resolve_streaming_option(cfg['encoder_dims'], polarity)
    if encoder_dims is None:
        encoder_dims = list(range(1, input_count + 1))
    assert len(encoder_dims) == input_count

    encoder_generators = _resolve_streaming_option(
        cfg['encoder_generators'], polarity
    )
    if encoder_generators is None:
        encoder_generators = ['sobol'] * input_count
    assert len(encoder_generators) == input_count

    output_polarity = _resolve_streaming_option(
        cfg['output_polarity'], polarity
    )
    if output_polarity is None:
        output_polarity = polarity

    from napl.sim.operation import decode, encode
    encoders = []
    for input_polarity, dim, generator in zip(
        input_polarities, encoder_dims, encoder_generators
    ):
        codec_config = _codec_config(input_polarity, timestep, dim=dim)
        codec_config['generator'] = generator
        encoders.append(encode(codec_config).to(device))
    operation = cfg['make_operation'](polarity, timestep, device).to(device)
    if cfg['make_readout'] is None:
        dec = decode(
            _codec_config(output_polarity, timestep, dim=1)
        ).to(device)
    else:
        dec = cfg['make_readout'](
            output_polarity, timestep, device
        ).to(device)
    assert operation.streaming is True
    return encoders, operation, dec


# Run a streaming pipeline for a fixed number of timesteps.
def _run_pipeline(cfg, pipeline, values, timestep, check_progress,
                  capture_trace):
    encoders, operation, dec = pipeline
    trace = []
    for step in range(1, timestep + 1):
        spikes = tuple(enc(value) for enc, value in zip(encoders, values))
        spike_out = cfg['apply_operation'](operation, spikes, values)
        dec(spike_out)
        if check_progress:
            for module in (*encoders, operation, dec):
                assert module.timestep_cur == step
        if capture_trace:
            trace.append(spike_out.detach().cpu().clone())
    result = dec.spike_value.detach().cpu().clone()
    return result, trace


# Return pipeline modules to their initial timestep state.
def _reset_pipeline(pipeline):
    encoders, operation, dec = pipeline
    for module in (*encoders, operation, dec):
        module.reset()
        assert module.timestep_cur == 0


# Run the known-answer inputs across every device and polarity and print the decoded value for the user to inspect, asserting no fidelity bound.
def _streaming_known_answer(cfg):
    torch.manual_seed(_SEED)
    for polarity in cfg['polarities']:
        values_cpu, expected = cfg['known_answer_case'](polarity)[:2]
        for device in devices():
            values = clone_inputs(values_cpu, device)
            pipeline = _make_pipeline(
                polarity, cfg['timesteps'], device, len(values), cfg,
            )
            result, _ = _run_pipeline(
                cfg, pipeline, values, cfg['timesteps'],
                check_progress=False, capture_trace=False,
            )
            error = (result - expected.cpu()).abs().max().item()
            print(f'[{device}][{polarity}] known-answer max_error={error:.6f}')


# Run the fixed RTL-fidelity vectors and print the rmse and largest per-element error for the user to inspect, asserting no fidelity bound.
def _streaming_rtl_fidelity(cfg):
    torch.manual_seed(_SEED)
    timesteps = cfg['timesteps']
    for polarity in cfg['polarities']:
        raw_values = cfg['make_values'](polarity)
        reference = cfg['analytic_reference'](raw_values, polarity).cpu()
        values_cpu = _multirank_inputs(raw_values)
        assert all(value.ndim >= 2 for value in values_cpu)
        if any(value.shape != raw.shape for value, raw in zip(values_cpu, raw_values)):
            reference = _multirank_inputs((reference,))[0]
        for device in devices():
            values = clone_inputs(values_cpu, device)
            pipeline = _make_pipeline(
                polarity, timesteps, device, len(values), cfg,
            )
            result, _ = _run_pipeline(
                cfg, pipeline, values, timesteps,
                check_progress=False, capture_trace=False,
            )
            error = (result - reference).abs()
            rmse = error.pow(2).mean().sqrt().item()
            max_error = error.max().item()
            print(
                f'[{device}][{polarity}] seed={_SEED}, '
                f'N={timesteps}, elements={reference.numel()}, '
                f'rmse={rmse:.6f}, max_error={max_error:.6f}'
            )


# Check streaming reset and replay determinism.
def _streaming_reset_replay(cfg):
    torch.manual_seed(_SEED)
    timesteps = cfg['state_timesteps']
    for polarity in cfg['polarities']:
        values_cpu = cfg['make_values'](polarity)
        for device in devices():
            values = clone_inputs(values_cpu, device)
            pipeline = _make_pipeline(
                polarity, timesteps, device, len(values), cfg,
            )
            first_result, first_trace = _run_pipeline(
                cfg, pipeline, values, timesteps,
                check_progress=True, capture_trace=True,
            )
            encoders, operation, dec = pipeline
            for module in (*encoders, operation, dec):
                assert module.timestep_cur == timesteps
            _reset_pipeline(pipeline)
            assert dec.spike_count.abs().sum().item() == 0

            replay_result, replay_trace = _run_pipeline(
                cfg, pipeline, values, timesteps,
                check_progress=True, capture_trace=True,
            )
            assert torch.equal(first_result, replay_result)
            assert len(first_trace) == len(replay_trace)
            assert all(
                torch.equal(first, replay)
                for first, replay in zip(first_trace, replay_trace)
            )


# Benchmark the pipeline on large random inputs drawn over the legal spike range per polarity and print the measured error beside the runtime.
def _streaming_random_perf(cfg):
    torch.manual_seed(_SEED)
    timesteps = cfg['timesteps']
    for polarity in cfg['polarities']:
        # Draw the large performance inputs at random over the legal spike range
        # per polarity, falling back to the fidelity inputs when none is set.
        if cfg['make_random_perf_values'] is None:
            values_cpu = _legal_range_random(cfg['make_values'](polarity))
        else:
            values_cpu = _legal_range_random(
                cfg['make_random_perf_values'](polarity)
            )
        reference = cfg['analytic_reference'](values_cpu, polarity).cpu()
        cpu_runtime = None
        for device in devices():
            pipeline = _make_pipeline(
                polarity, timesteps, device, len(values_cpu), cfg,
            )

            # Run the configured pipeline for benchmark inputs.
            def run(inputs):
                _run_pipeline(
                    cfg, pipeline, inputs, timesteps,
                    check_progress=False, capture_trace=False,
                )

            device_runtime = benchmark(
                run,
                values_cpu,
                device,
                warmup_runs=cfg['warmup_runs'],
                trials=cfg['trials'],
                prepare=lambda: _reset_pipeline(pipeline),
            )
            if device == 'cpu':
                cpu_runtime = device_runtime
            assert cpu_runtime is not None
            speedup = cpu_runtime / device_runtime
            print(
                f'[{device}][{polarity}] '
                f'device_runtime={device_runtime:.6f}s, '
                f'cpu_runtime={cpu_runtime:.6f}s, '
                f'warmup={cfg["warmup_runs"]}, trials={cfg["trials"]}, '
                f'median, speedup={speedup:.2f}x'
            )

            # Report the measured error on the performance inputs beside the
            # runtime, asserting nothing, for the user to inspect by eye.
            _reset_pipeline(pipeline)
            result, _ = _run_pipeline(
                cfg, pipeline, clone_inputs(values_cpu, device), timesteps,
                check_progress=False, capture_trace=False,
            )
            error = (result - reference).abs()
            print(
                f'[{device}][{polarity}] perf error: '
                f'rmse={error.pow(2).mean().sqrt().item():.6f}, '
                f'max_error={error.max().item():.6f}'
            )


def streaming_suite(cfg):
    """
    Run the streaming-kernel suite (known answer, analytic fidelity, reset
    and replay, performance) over every device and supported polarity.

    The known-answer and fidelity checks exercise the kernel and print the
    measured decoded error for the user to inspect by eye; they assert no
    numerical fidelity. The fidelity and performance runs draw their inputs at
    random over the legal spike range per polarity. Execution state, reset and
    replay, per-device timing, and any ``extra_checks`` still assert.

    An ``apply_operation`` callback takes ``(operation, spikes)``, or
    ``(operation, spikes, values)`` when the operation also consumes a raw
    unencoded input, as ``mul_ugemm`` does for its binary-domain operand.
    """
    cfg = _check_config(
        cfg, _STREAMING_REQUIRED, _STREAMING_DEFAULTS, 'streaming suite'
    )
    if not cfg['polarities']:
        raise NotImplementedError("streaming suite config: set 'polarities'")
    cfg['apply_operation'] = _wrap_apply_operation(cfg['apply_operation'])
    _streaming_known_answer(cfg)
    _streaming_rtl_fidelity(cfg)
    _streaming_reset_replay(cfg)
    _streaming_random_perf(cfg)
    if cfg['extra_checks'] is not None:
        cfg['extra_checks']()


_NON_STREAMING_REQUIRED = (
    'gradient_atol',
    'gradient_rtol',
    'make_module_pair',
    'make_inputs',
    'known_answer_case',
    'gradient_case',
    'expected_ste_gradients',
)
_NON_STREAMING_DEFAULTS = {
    'make_random_perf_values': None,
    'extra_checks': None,
    'warmup_runs': 2,
    'trials': 7,
}


# Assert non-streaming module state.
def _assert_non_streaming(module):
    assert module.streaming is False
    assert module.timestep_cur == 0


# Build the default gradient output.
def _default_gradient_output(output):
    ramp = torch.arange(
        1, output.numel() + 1, dtype=output.dtype, device=output.device
    )
    return (ramp / output.numel()).reshape(output.shape)


# Run the known-answer inputs across every device and print the error against the expected answer for the user to inspect, asserting no fidelity bound.
def _non_streaming_known_answer(cfg):
    torch.manual_seed(_SEED)
    candidate_cpu, inputs_cpu, expected = cfg['known_answer_case']()
    for device in devices():
        candidate = copy.deepcopy(candidate_cpu).to(device)
        inputs = clone_inputs(inputs_cpu, device)
        _assert_non_streaming(candidate)
        result = candidate(*inputs)
        _assert_non_streaming(candidate)
        error = (result - expected.to(device)).abs().max().item()
        print(f'[{device}] known-answer max_error={error:.6f}')


# Run the fixed RTL-fidelity vectors and print the error against the PyTorch reference for the user to inspect, asserting no fidelity bound.
def _non_streaming_rtl_fidelity(cfg):
    torch.manual_seed(_SEED)
    candidate_cpu, reference_cpu = cfg['make_module_pair']()
    inputs_cpu = _multirank_inputs(cfg['make_inputs']())
    assert all(value.ndim >= 2 for value in inputs_cpu)
    for device in devices():
        candidate = copy.deepcopy(candidate_cpu).to(device)
        reference = copy.deepcopy(reference_cpu).to(device)
        candidate_inputs = clone_inputs(inputs_cpu, device)
        reference_inputs = clone_inputs(inputs_cpu, device)
        assert_inputs_equal(candidate_inputs, reference_inputs)
        _assert_non_streaming(candidate)
        result = candidate(*candidate_inputs)
        expected = reference(*reference_inputs)
        _assert_non_streaming(candidate)
        max_error = (result - expected).abs().max().item()
        print(
            f'[{device}] seed={_SEED}, dtype={result.dtype}, '
            f'max_error={max_error:.6f}'
        )


# Check straight-through estimator gradients.
def _non_streaming_gradients(cfg):
    torch.manual_seed(_SEED)
    atol = cfg['gradient_atol']
    rtol = cfg['gradient_rtol']
    candidate_cpu, raw_inputs = cfg['gradient_case']()
    for device in devices():
        candidate = copy.deepcopy(candidate_cpu).to(device)
        inputs = tuple(
            value.detach().clone().to(device).requires_grad_(True)
            for value in raw_inputs
        )
        _assert_non_streaming(candidate)
        output = candidate(*inputs)
        grad_output = _default_gradient_output(output).to(device)
        expected_inputs, expected_parameters = cfg['expected_ste_gradients'](
            candidate, inputs, grad_output
        )
        output.backward(grad_output)
        _assert_non_streaming(candidate)

        assert len(expected_inputs) == len(inputs)
        for index, (value, expected) in enumerate(
            zip(inputs, expected_inputs)
        ):
            assert value.grad is not None, f'input[{index}] gradient is missing'
            max_error = (value.grad - expected).abs().max().item()
            torch.testing.assert_close(
                value.grad, expected, atol=atol, rtol=rtol
            )
            print(
                f'[{device}] input[{index}] gradient '
                f'max_error={max_error:.6g}, atol={atol}, rtol={rtol}'
            )

        parameters = {
            name: parameter
            for name, parameter in candidate.named_parameters()
            if parameter.requires_grad
        }
        assert set(expected_parameters) == set(parameters)
        for name, parameter in parameters.items():
            assert parameter.grad is not None, f'{name} gradient is missing'
            expected = expected_parameters[name]
            max_error = (parameter.grad - expected).abs().max().item()
            torch.testing.assert_close(
                parameter.grad, expected, atol=atol, rtol=rtol
            )
            print(
                f'[{device}] {name} gradient '
                f'max_error={max_error:.6g}, atol={atol}, rtol={rtol}'
            )


# Benchmark the module on large random inputs drawn over the legal range and print the measured error beside the runtime.
def _non_streaming_random_perf(cfg):
    torch.manual_seed(_SEED)
    candidate_cpu, reference_cpu = cfg['make_module_pair']()
    # Draw the large performance inputs at random over the legal range, falling
    # back to the fidelity inputs when none is set.
    if cfg['make_random_perf_values'] is None:
        shared_inputs_cpu = _legal_range_random(cfg['make_inputs']())
    else:
        shared_inputs_cpu = _legal_range_random(cfg['make_random_perf_values']())
    cpu_runtime = None
    for device in devices():
        candidate = copy.deepcopy(candidate_cpu).to(device)
        reference = copy.deepcopy(reference_cpu).to(device)
        _assert_non_streaming(candidate)

        with torch.no_grad():
            device_runtime = benchmark(
                lambda inputs: candidate(*inputs),
                shared_inputs_cpu,
                device,
                warmup_runs=cfg['warmup_runs'],
                trials=cfg['trials'],
            )

        _assert_non_streaming(candidate)
        if device == 'cpu':
            cpu_runtime = device_runtime
        assert cpu_runtime is not None
        speedup = cpu_runtime / device_runtime
        print(
            f'[{device}] device_runtime={device_runtime:.6f}s, '
            f'cpu_runtime={cpu_runtime:.6f}s, '
            f'warmup={cfg["warmup_runs"]}, trials={cfg["trials"]}, '
            f'median, speedup={speedup:.2f}x'
        )

        # Report the measured error on the performance inputs beside the
        # runtime, asserting nothing, for the user to inspect by eye.
        with torch.no_grad():
            inputs = clone_inputs(shared_inputs_cpu, device)
            error = (candidate(*inputs) - reference(*inputs)).abs()
        _assert_non_streaming(candidate)
        print(
            f'[{device}] perf error: '
            f'max_error={error.max().item():.6f}'
        )


def non_streaming_suite(cfg):
    """
    Run the non-streaming trainable-kernel suite (known answer, PyTorch
    reference fidelity, STE gradients, performance) over every device.
    """
    cfg = _check_config(
        cfg, _NON_STREAMING_REQUIRED, _NON_STREAMING_DEFAULTS, 'non-streaming suite'
    )
    _non_streaming_known_answer(cfg)
    _non_streaming_rtl_fidelity(cfg)
    _non_streaming_gradients(cfg)
    _non_streaming_random_perf(cfg)
    if cfg['extra_checks'] is not None:
        cfg['extra_checks']()

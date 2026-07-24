import copy
import math
from statistics import median
from time import perf_counter
from typing import Callable, Optional, Sequence, Tuple

import torch

InputTuple = Tuple[torch.Tensor, ...]
Runner = Callable[[InputTuple], object]
Prepare = Optional[Callable[[], None]]
Device = str


def devices():
    dev = ['cpu']
    if torch.cuda.is_available():
        dev.append('cuda')
    if torch.backends.mps.is_available():
        dev.append('mps')
    return dev


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


class timer:
    def __init__(self, device):
        self.device = device
        self.seconds = None

    def __enter__(self):
        sync(self.device)
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sync(self.device)
        self.seconds = perf_counter() - self._start


def clone_inputs(
    inputs: Sequence[torch.Tensor], device: Optional[Device] = None
) -> InputTuple:
    if device is None:
        return tuple(value.clone() for value in inputs)
    return tuple(value.clone().to(device) for value in inputs)


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
    'tolerance_scale',
    'make_operation',
    'make_values',
    'analytic_reference',
    'known_answer_case',
)
_STREAMING_DEFAULTS = {
    'apply_operation': None,
    'timesteps': 256,
    'warmup_runs': 2,
    'trials': 7,
}


def _codec_config(polarity, timestep, dim):
    return {
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': dim,
    }


def _make_pipeline(polarity, timestep, device, input_count,
                   operation_factory):
    from napl.sim.module import decoder, encoder
    encoders = [
        encoder(_codec_config(polarity, timestep, dim=index + 1)).to(device)
        for index in range(input_count)
    ]
    operation = operation_factory(polarity, timestep, device).to(device)
    dec = decoder(_codec_config(polarity, timestep, dim=1)).to(device)
    assert operation.streaming is True
    return encoders, operation, dec


def _run_pipeline(cfg, pipeline, values, timestep, check_progress,
                  capture_trace):
    encoders, operation, dec = pipeline
    trace = []
    for step in range(1, timestep + 1):
        spikes = tuple(enc(value) for enc, value in zip(encoders, values))
        spike_out = cfg['apply_operation'](operation, spikes)
        dec(spike_out)
        if check_progress:
            for module in (*encoders, operation, dec):
                assert module.timestep_cur == step
        if capture_trace:
            trace.append(spike_out.detach().cpu().clone())
    result = dec.spike_value.detach().cpu().clone()
    return result, trace


def _reset_pipeline(pipeline):
    encoders, operation, dec = pipeline
    for module in (*encoders, operation, dec):
        module.reset()
        assert module.timestep_cur == 0


def _streaming_known_answer(cfg):
    torch.manual_seed(_SEED)
    for polarity in cfg['polarities']:
        values_cpu, expected, tolerance = cfg['known_answer_case'](polarity)
        for device in devices():
            values = clone_inputs(values_cpu, device)
            pipeline = _make_pipeline(
                polarity, cfg['timesteps'], device,
                len(values), cfg['make_operation'],
            )
            result, _ = _run_pipeline(
                cfg, pipeline, values, cfg['timesteps'],
                check_progress=False, capture_trace=False,
            )
            torch.testing.assert_close(
                result, expected.cpu(), atol=tolerance, rtol=0
            )


def _streaming_fidelity(cfg):
    torch.manual_seed(_SEED)
    timesteps = cfg['timesteps']
    tolerance = cfg['tolerance_scale'] / math.sqrt(timesteps)
    for polarity in cfg['polarities']:
        values_cpu = cfg['make_values'](polarity)
        reference = cfg['analytic_reference'](values_cpu, polarity).cpu()
        for device in devices():
            values = clone_inputs(values_cpu, device)
            pipeline = _make_pipeline(
                polarity, timesteps, device, len(values),
                cfg['make_operation'],
            )
            result, _ = _run_pipeline(
                cfg, pipeline, values, timesteps,
                check_progress=False, capture_trace=False,
            )
            rmse = (result - reference).pow(2).mean().sqrt().item()
            assert rmse <= tolerance, (
                f'[{device}][{polarity}] rmse={rmse:.6f}, '
                f'bound={tolerance:.6f}'
            )
            print(
                f'[{device}][{polarity}] seed={_SEED}, '
                f'N={timesteps}, rmse={rmse:.6f}, bound={tolerance:.6f}'
            )


def _streaming_reset_replay(cfg):
    torch.manual_seed(_SEED)
    timesteps = _STATE_TIMESTEPS
    for polarity in cfg['polarities']:
        values_cpu = cfg['make_values'](polarity)
        for device in devices():
            values = clone_inputs(values_cpu, device)
            pipeline = _make_pipeline(
                polarity, timesteps, device, len(values),
                cfg['make_operation'],
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


def _streaming_performance(cfg):
    torch.manual_seed(_SEED)
    timesteps = cfg['timesteps']
    for polarity in cfg['polarities']:
        values_cpu = cfg['make_values'](polarity)
        cpu_runtime = None
        for device in devices():
            pipeline = _make_pipeline(
                polarity, timesteps, device, len(values_cpu),
                cfg['make_operation'],
            )

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


def streaming_suite(cfg):
    """
    Run the streaming-kernel suite (known answer, analytic fidelity, reset
    and replay, performance) over every device and supported polarity.
    """
    cfg = _check_config(
        cfg, _STREAMING_REQUIRED, _STREAMING_DEFAULTS, 'streaming suite'
    )
    if not cfg['polarities']:
        raise NotImplementedError("streaming suite config: set 'polarities'")
    if cfg['apply_operation'] is None:
        cfg['apply_operation'] = lambda operation, spikes: operation(*spikes)
    _streaming_known_answer(cfg)
    _streaming_fidelity(cfg)
    _streaming_reset_replay(cfg)
    _streaming_performance(cfg)


_SINGLE_SHOT_REQUIRED = (
    'quantization_atol',
    'known_answer_atol',
    'gradient_atol',
    'gradient_rtol',
    'make_module_pair',
    'make_inputs',
    'known_answer_case',
    'gradient_case',
    'expected_ste_gradients',
)
_SINGLE_SHOT_DEFAULTS = {
    'warmup_runs': 2,
    'trials': 7,
}


def _assert_single_shot(module):
    assert module.streaming is False
    assert module.timestep_cur == 0


def _default_gradient_output(output):
    ramp = torch.arange(
        1, output.numel() + 1, dtype=output.dtype, device=output.device
    )
    return (ramp / output.numel()).reshape(output.shape)


def _single_shot_known_answer(cfg):
    torch.manual_seed(_SEED)
    candidate_cpu, inputs_cpu, expected = cfg['known_answer_case']()
    for device in devices():
        candidate = copy.deepcopy(candidate_cpu).to(device)
        inputs = clone_inputs(inputs_cpu, device)
        _assert_single_shot(candidate)
        result = candidate(*inputs)
        _assert_single_shot(candidate)
        torch.testing.assert_close(
            result, expected.to(device), atol=cfg['known_answer_atol'], rtol=0
        )


def _single_shot_fidelity(cfg):
    torch.manual_seed(_SEED)
    tolerance = cfg['quantization_atol']
    candidate_cpu, reference_cpu = cfg['make_module_pair']()
    inputs_cpu = cfg['make_inputs']()
    for device in devices():
        candidate = copy.deepcopy(candidate_cpu).to(device)
        reference = copy.deepcopy(reference_cpu).to(device)
        candidate_inputs = clone_inputs(inputs_cpu, device)
        reference_inputs = clone_inputs(inputs_cpu, device)
        assert_inputs_equal(candidate_inputs, reference_inputs)
        _assert_single_shot(candidate)
        result = candidate(*candidate_inputs)
        expected = reference(*reference_inputs)
        _assert_single_shot(candidate)
        max_error = (result - expected).abs().max().item()
        torch.testing.assert_close(result, expected, atol=tolerance, rtol=0)
        print(
            f'[{device}] seed={_SEED}, dtype={result.dtype}, '
            f'max_error={max_error:.6f}, quantization_bound={tolerance:.6f}'
        )


def _single_shot_gradients(cfg):
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
        _assert_single_shot(candidate)
        output = candidate(*inputs)
        grad_output = _default_gradient_output(output).to(device)
        expected_inputs, expected_parameters = cfg['expected_ste_gradients'](
            candidate, inputs, grad_output
        )
        output.backward(grad_output)
        _assert_single_shot(candidate)

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


def _single_shot_performance(cfg):
    torch.manual_seed(_SEED)
    candidate_cpu, _ = cfg['make_module_pair']()
    shared_inputs_cpu = cfg['make_inputs']()
    cpu_runtime = None
    for device in devices():
        candidate = copy.deepcopy(candidate_cpu).to(device)
        _assert_single_shot(candidate)

        with torch.no_grad():
            device_runtime = benchmark(
                lambda inputs: candidate(*inputs),
                shared_inputs_cpu,
                device,
                warmup_runs=cfg['warmup_runs'],
                trials=cfg['trials'],
            )

        _assert_single_shot(candidate)
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


def single_shot_suite(cfg):
    """
    Run the single-shot trainable-kernel suite (known answer, PyTorch
    reference fidelity, STE gradients, performance) over every device.
    """
    cfg = _check_config(
        cfg, _SINGLE_SHOT_REQUIRED, _SINGLE_SHOT_DEFAULTS, 'single-shot suite'
    )
    _single_shot_known_answer(cfg)
    _single_shot_fidelity(cfg)
    _single_shot_gradients(cfg)
    _single_shot_performance(cfg)

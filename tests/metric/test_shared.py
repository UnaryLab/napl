from io import StringIO

import torch
from loguru import logger

from napl.sim.metric import (
    accuracy,
    correlation,
    stability,
    stability_builder,
    stability_flux,
    stability_norm,
)
from napl.sim.metric._shared import Analysis, analyze
from napl.utils._shared_test import devices


def capture_log(call):
    output = StringIO()
    sink = logger.add(output, format='{message}')
    try:
        call()
    finally:
        logger.remove(sink)
    return output.getvalue().splitlines()


def test_analyze_known_answer():
    """Verify tensor analysis returns the expected extrema, means, RMS, and indices."""
    for device in devices():
        input = torch.tensor([[-2.0, 1.0, 0.0]], device=device)
        result = analyze(input)

        assert torch.equal(result.absolute, input.abs())
        assert result.absolute_min == 0.0
        assert result.absolute_max == 2.0
        assert torch.allclose(result.mean, torch.tensor(-1.0 / 3, device=device))
        assert result.mean_absolute == 1.0
        assert torch.allclose(
            result.root_mean_square, torch.sqrt(torch.tensor(5.0 / 3, device=device))
        )
        assert result.max_absolute_index == 0


def test_analyze_report():
    """Verify tensor analysis logs a labeled report only when verbose output is requested."""
    input = torch.tensor([[-2.0, 1.0, 0.0]])
    result = analyze(input)
    assert capture_log(lambda: analyze(input)) == []
    messages = capture_log(
        lambda: analyze(
            input,
            verbose=True,
            report='Analysis',
            value='value',
        )
    )

    assert messages == [
        'Analysis report: ',
        f'    Max absolute value:     <{result.absolute_max.item()}>',
        f'    Min absolute value:     <{result.absolute_min.item()}>',
        f'    Mean value:             <{result.mean.item()}>',
        f'    Mean absolute value:    <{result.mean_absolute.item()}>',
        f'    Root mean square value: <{result.root_mean_square.item()}>',
        '',
    ]

    messages = capture_log(
        lambda: analyze(
            input,
            verbose=True,
            report='Analysis',
            value='value',
            timestep=3,
        )
    )
    assert messages[0] == 'Analysis report over <3> timesteps: '


def test_accuracy_analyze_returns_local_results():
    """Verify accuracy analysis returns local results without registering derived tensor state."""
    for device in devices():
        metric = accuracy({'polarity': 'unipolar'}).to(device)
        metric(torch.tensor([[1, 0]], dtype=metric.stype, device=device))

        result, analysis_result = metric.analyze(
            torch.tensor([[0.0, 1.0]], device=device)
        )

        assert not hasattr(metric, 'spike_error')
        assert list(metric.named_parameters()) == []
        assert set(dict(metric.named_buffers())) == {'spike_count'}
        assert set(metric.state_dict()) == {'spike_count'}
        summary = analyze(result)
        assert isinstance(analysis_result, Analysis)
        for actual, expected in zip(analysis_result, summary):
            assert torch.equal(actual, expected)

        messages = capture_log(
            lambda: metric.analyze(
                torch.tensor([[0.0, 1.0]], device=device),
                verbose=True,
            )
        )
        assert messages == [
            f'Progressive Error report over <{metric.timestep_cur}> timesteps: ',
            f'    Max absolute progressive error:     <{summary.absolute_max.item()}>',
            f'    Min absolute progressive error:     <{summary.absolute_min.item()}>',
            f'    Mean progressive error:             <{summary.mean.item()}>',
            f'    Mean absolute progressive error:    <{summary.mean_absolute.item()}>',
            f'    Root mean square progressive error: <{summary.root_mean_square.item()}>',
            '',
        ]
        metric.reset()
        assert not hasattr(metric, 'spike_error')


def test_metric_analyze_returns_local_results():
    """Verify metric analyses return local summaries without retaining derived tensor state."""
    for device in devices():
        source = torch.ones(2)
        spike = source.to(device)
        metrics = (
            (correlation().to(device), 'correlation'),
            (
                stability(
                    source,
                    {'polarity': 'bipolar', 'threshold': 0.05},
                ).to(device),
                'stability',
            ),
            (
                stability_flux(
                    source,
                    source,
                    {'polarity': 'bipolar', 'threshold': 0.05},
                ).to(device),
                'stability_flux',
            ),
            (
                stability_norm(
                    source,
                    {'polarity': 'bipolar', 'threshold': 0.05},
                ).to(device),
                'stability_norm',
            ),
        )

        for metric, prefix in metrics:
            assert not metric.valid
            for _ in range(4):
                if isinstance(metric, (correlation, stability_flux)):
                    metric(spike, spike)
                else:
                    metric(spike)
            assert metric.valid
            assert metric.timestep_cur == 4
            value, analysis_result = metric.analyze()
            summary = analyze(value)
            assert isinstance(analysis_result, Analysis)
            for actual, expected in zip(analysis_result, summary):
                assert torch.equal(actual, expected)
            for suffix in ('abs_min', 'abs_max', 'avg', 'mae', 'rmse'):
                assert not hasattr(metric, f'{prefix}_{suffix}')


def test_metric_persistent_tensor_state():
    """Verify metric tensor state is buffer-backed, non-trainable, and device portable."""
    source = torch.tensor([-1.0, 1.0])
    config = {'polarity': 'bipolar', 'threshold': 0.05}
    metrics = (
        accuracy(),
        correlation(),
        stability(source, config),
        stability_norm(source, config),
        stability_flux(source, source, config),
        stability_builder(
            source,
            {
                **config,
                'normstability': 0.5,
                'timestep': 4,
                'generator': 'sobol',
            },
        ),
    )
    for metric in metrics:
        assert list(metric.named_parameters()) == []
        assert all(not buffer.requires_grad for _, buffer in metric.named_buffers())
        for module in metric.modules():
            assert all(
                not isinstance(value, torch.Tensor)
                for value in vars(module).values()
            )


if __name__ == '__main__':
    test_analyze_known_answer()
    test_analyze_report()
    test_accuracy_analyze_returns_local_results()
    test_metric_analyze_returns_local_results()
    test_metric_persistent_tensor_state()
    print('Test passed.')

from io import StringIO

import torch
from loguru import logger

from napl.metric import (
    accuracy,
    correlation,
    stability,
    stability_flux,
    stability_norm,
)
from napl.metric._shared import analyze
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
    for device in devices():
        input = torch.tensor([-2.0, 1.0, 0.0], device=device)
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
    input = torch.tensor([-2.0, 1.0, 0.0])
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


def test_accuracy_analyze_preserves_parameters():
    for device in devices():
        metric = accuracy({'polarity': 'unipolar'}).to(device)
        spike_error = metric.spike_error
        metric(torch.tensor([1, 0], dtype=metric.stype, device=device))

        result, _ = metric.analyze(torch.tensor([0.0, 1.0], device=device))

        assert result is spike_error
        assert dict(metric.named_parameters())['spike_error'] is spike_error
        summary = analyze(result)
        assert torch.equal(metric.spike_error_abs_min, summary.absolute_min)
        assert torch.equal(metric.spike_error_abs_max, summary.absolute_max)
        assert torch.equal(metric.spike_error_avg, summary.mean)
        assert torch.equal(metric.spike_error_mae, summary.mean_absolute)
        assert torch.equal(metric.spike_error_rmse, summary.root_mean_square)

        messages = capture_log(
            lambda: metric.analyze(
                torch.tensor([0.0, 1.0], device=device),
                verbose=True,
            )
        )
        assert messages == [
            f'Accuracy report over <{metric.timestep_cur}> timesteps: ',
            f'    Max absolute error:     <{summary.absolute_max.item()}>',
            f'    Min absolute error:     <{summary.absolute_min.item()}>',
            f'    Mean error:             <{summary.mean.item()}>',
            f'    Mean absolute error:    <{summary.mean_absolute.item()}>',
            f'    Root mean square error: <{summary.root_mean_square.item()}>',
            '',
        ]


def test_metric_analyze_callers():
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
                'flux',
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
            value, _ = metric.analyze()
            summary = analyze(value)
            assert torch.equal(
                getattr(metric, f'{prefix}_abs_min'),
                summary.absolute_min,
            )
            assert torch.equal(
                getattr(metric, f'{prefix}_abs_max'),
                summary.absolute_max,
            )
            assert torch.equal(
                getattr(metric, f'{prefix}_avg'),
                summary.mean,
            )
            assert torch.equal(
                getattr(metric, f'{prefix}_mae'),
                summary.mean_absolute,
            )
            assert torch.equal(
                getattr(metric, f'{prefix}_rmse'),
                summary.root_mean_square,
            )


if __name__ == '__main__':
    test_analyze_known_answer()
    test_analyze_report()
    test_accuracy_analyze_preserves_parameters()
    test_metric_analyze_callers()
    print('Test passed.')

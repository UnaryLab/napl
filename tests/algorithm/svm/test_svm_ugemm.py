import torch

from napl.sim.algorithm.svm import svm_ugemm
from napl.sim.base import global_config
from napl.sim.module import linear_ugemm
from napl.sim.operation import encode, gt
from napl.utils._shared_test import devices, streaming_suite


WEIGHT = torch.tensor([[0.75, -0.5]])
BIAS = torch.tensor([0.125])


def _config(timestep):
    return {
        'polarity': 'bipolar',
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 2,
        'scale': None,
        'width': 8,
    }


def make_operation(_polarity, timestep, _device):
    return svm_ugemm(WEIGHT.clone(), BIAS.clone(), _config(timestep))


def make_values(_polarity):
    return (torch.tensor([
        [-1.0, -1.0],
        [-0.75, 0.25],
        [0.0, 0.0],
        [0.5, -0.5],
        [1.0, 1.0],
    ]),)


def make_random_perf_values(_polarity):
    return (torch.linspace(-1.0, 1.0, 131072).reshape(-1, 2),)


def analytic_reference(values, _polarity):
    score = values[0] @ WEIGHT.T + BIAS
    return score.gt(0).type(global_config.ntype)


def known_answer_case(_polarity):
    values = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    return (values,), torch.tensor([[1.0], [0.0]])


def _composition_checks():
    """Pin both returned streams to the declared NAPL primitive composition."""
    timestep = 32
    config = _config(timestep)
    values = torch.tensor([[0.75, -0.25], [-0.5, 0.5]])
    for device in devices():
        operation = svm_ugemm(
            WEIGHT.clone(), BIAS.clone(), config
        ).to(device)
        score_layer = linear_ugemm(
            WEIGHT.clone(), BIAS.clone(), config
        ).to(device)
        reference_encode = encode({
            'polarity': 'bipolar',
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 3,
        }).to(device)
        decision_compare = gt({'polarity': 'bipolar'}).to(device)
        input_encode = encode({
            'polarity': 'bipolar',
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
        }).to(device)
        zero = torch.tensor(0.0, dtype=global_config.ntype, device=device)
        for _ in range(timestep):
            input_spike = input_encode(values.to(device))
            score, decision = operation(input_spike)
            expected_score = score_layer(input_spike)
            expected_decision = decision_compare(
                expected_score, reference_encode(zero)
            )
            assert torch.equal(score, expected_score)
            assert torch.equal(decision, expected_decision)

        assert operation.timestep_cur == timestep
        assert operation.score_layer.timestep_cur == timestep
        assert operation.reference_encode.timestep_cur == timestep
        assert operation.decision_compare.timestep_cur == timestep
        first_score = score.clone()
        first_decision = decision.clone()
        operation.reset()
        input_encode.reset()
        for _ in range(timestep):
            score, decision = operation(input_encode(values.to(device)))
        assert torch.equal(first_score, score)
        assert torch.equal(first_decision, decision)


CONFIG = {
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'apply_operation': lambda operation, spikes: operation(spikes[0])[1],
    'output_polarity': 'unipolar',
    'encoder_dims': [1],
    'timesteps': 256,
    'state_timesteps': 16,
    'warmup_runs': 1,
    'trials': 3,
    'extra_checks': _composition_checks,
}


def test_svm_ugemm():
    """Verify linear SVM decisions, primitive composition, and reset replay."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_svm_ugemm()

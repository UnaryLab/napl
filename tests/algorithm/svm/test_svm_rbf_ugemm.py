import math

import torch

from napl.sim.algorithm import svm_rbf_ugemm
from napl.sim.base import global_config
from napl.sim.operation import (
    add_scale,
    bi2uni,
    encode,
    exp_n1,
    gt,
    mul_scale,
    mul_unibi,
    pow_delay,
    sub_scale,
)
from napl.utils._shared_test import devices, streaming_suite


SUPPORT = torch.tensor([
    [-0.85, -0.70],
    [-0.45, 0.55],
    [0.35, -0.35],
    [0.80, 0.75],
], dtype=global_config.ntype)
BETA = torch.tensor([-0.70, 0.60, -0.50, 0.40], dtype=global_config.ntype)
BIAS = 0.05
GAMMA = 0.0625
POINTS = torch.tensor([
    [-1.00, -0.90],
    [-0.70, 0.40],
    [-0.15, -0.10],
    [0.25, -0.55],
    [0.65, 0.60],
    [1.00, 0.95],
], dtype=global_config.ntype)


def _config(timestep):
    return {
        'polarity': 'bipolar',
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 2,
        'score_scale': 2.25390625,
        'intwidth': 12,
        'fracwidth': 8,
        # Extra delay and conversion width reduce finite-stream correlation error.
        'square_depth': 5,
        'bi2uni_width': 6,
    }


def _exact_score(points, gamma=GAMMA):
    distance = (points.unsqueeze(-2) - SUPPORT).square().sum(-1)
    return torch.exp(-gamma * distance) @ BETA + BIAS


def make_operation(_polarity, timestep, _device):
    return svm_rbf_ugemm(
        SUPPORT.clone(), BETA.clone(), BIAS, GAMMA, _config(timestep)
    )


def make_values(_polarity):
    generator = torch.Generator().manual_seed(7)
    return (torch.rand(512, 2, generator=generator).mul(2).sub(1),)


def make_random_perf_values(_polarity):
    generator = torch.Generator().manual_seed(11)
    return (torch.rand(65536, 2, generator=generator).mul(2).sub(1),)


def analytic_reference(values, _polarity):
    return _exact_score(values[0]).gt(0).type(global_config.ntype)


def known_answer_case(_polarity):
    return (POINTS,), _exact_score(POINTS).gt(0).type(global_config.ntype)


def _bare_pipeline(timestep, device):
    config = _config(timestep)
    dim = config['dim']
    codec = {
        'polarity': 'bipolar', 'timestep': timestep,
        'generator': 'sobol',
    }
    feature_count = SUPPORT.shape[1]
    return {
        'support_encoders': [
            encode({**codec, 'dim': dim + index}).to(device)
            for index in range(feature_count)
        ],
        'difference_ops': [
            sub_scale({
                'polarity': 'bipolar', 'scale': 2,
                'intwidth': config['intwidth'], 'fracwidth': config['fracwidth'],
            }).to(device) for _ in range(feature_count)
        ],
        'square_ops': [
            pow_delay({
                'polarity': 'bipolar', 'n': 2,
                'depth': config['square_depth'],
            }).to(device) for _ in range(feature_count)
        ],
        'square_converters': [
            bi2uni({'width': config['bi2uni_width']}).to(device)
            for _ in range(feature_count)
        ],
        'norm_add': add_scale({
            'polarity': 'unipolar', 'scale': feature_count,
            'intwidth': config['intwidth'], 'fracwidth': config['fracwidth'],
        }).to(device),
        'kernel_scale': mul_scale({
            'polarity': 'unipolar', 'scale': 4 * feature_count * GAMMA,
            'intwidth': config['intwidth'], 'fracwidth': config['fracwidth'],
        }).to(device),
        'kernel_exp': exp_n1({
            'polarity': 'unipolar', 'timestep': timestep,
            'generator': 'sobol', 'dim': dim + feature_count,
        }).to(device),
        'beta_encoder': encode({
            **codec, 'dim': dim + feature_count + 4,
        }).to(device),
        'kernel_beta_mul': mul_unibi().to(device),
        'bias_encoder': encode({
            **codec, 'dim': dim + feature_count + 5,
        }).to(device),
        'score_add': add_scale({
            'polarity': 'bipolar', 'scale': config['score_scale'],
            'intwidth': config['intwidth'], 'fracwidth': config['fracwidth'],
        }).to(device),
        'reference_encode': encode({
            **codec, 'dim': dim + feature_count + 6,
        }).to(device),
        'decision_compare': gt({'polarity': 'bipolar'}).to(device),
    }


def _bare_step(parts, input, support, beta, bias, zero):
    squares = []
    for index in range(support.shape[1]):
        support_bit = parts['support_encoders'][index](support[:, index])
        input_bit, support_bit = torch.broadcast_tensors(
            input[..., index].unsqueeze(-1), support_bit
        )
        difference = parts['difference_ops'][index](
            input_bit, support_bit
        )
        square = parts['square_ops'][index](difference)
        squares.append(parts['square_converters'][index](square))
    norm = parts['norm_add'](torch.stack(squares, dim=-1), dim=-1)
    kernel = parts['kernel_exp'](parts['kernel_scale'](norm))
    terms = parts['kernel_beta_mul'](kernel, parts['beta_encoder'](beta))
    bias_bit = parts['bias_encoder'](bias).expand(terms.shape[:-1])
    score = parts['score_add'](
        torch.cat((terms, bias_bit.unsqueeze(-1)), dim=-1), dim=-1
    )
    decision = parts['decision_compare'](
        score, parts['reference_encode'](zero)
    )
    if decision.shape != score.shape:
        decision = decision.expand(score.shape)
    return score, decision


def _composition_check():
    """Pin score and decision bit-exactly to the declared bare pipeline."""
    timestep = 32
    for device in devices():
        operation = make_operation('bipolar', timestep, device).to(device)
        parts = _bare_pipeline(timestep, device)
        input_encoder = encode({
            'polarity': 'bipolar', 'timestep': timestep,
            'generator': 'sobol', 'dim': 20,
        }).to(device)
        support = SUPPORT.to(device)
        beta = BETA.to(device)
        bias = torch.tensor(BIAS, dtype=global_config.ntype, device=device)
        zero = torch.tensor(0.0, dtype=global_config.ntype, device=device)
        assert operation.kernel_scale == 4 * SUPPORT.shape[1] * GAMMA
        assert operation.effective_gamma == GAMMA
        assert operation.score_scale == _config(timestep)['score_scale']
        first_trace = []
        for _ in range(timestep):
            input_spike = input_encoder(POINTS.to(device))
            score, decision = operation(input_spike)
            expected_score, expected_decision = _bare_step(
                parts, input_spike, support, beta, bias, zero
            )
            assert torch.equal(score, expected_score)
            assert torch.equal(decision, expected_decision)
            assert score.shape == (POINTS.shape[0],)
            assert decision.shape == score.shape
            first_trace.append((score.clone(), decision.clone()))

        state_shape = (POINTS.shape[0], SUPPORT.shape[0])
        for index in range(SUPPORT.shape[1]):
            assert operation.difference_ops[index].add_scale.accumulator.shape == state_shape
            delay_line = operation.square_ops[index].delays[0]
            assert delay_line.reg.shape == (delay_line.depth, *state_shape)
            assert operation.square_converters[index].accumulator.shape == state_shape
        assert operation.norm_add.accumulator.shape == state_shape
        assert operation.kernel_scale_op.accumulator.shape == state_shape
        assert operation.kernel_exp.input_d1.shape == state_shape
        assert operation.kernel_beta_mul.state.shape == state_shape
        assert operation.score_add.accumulator.shape == (POINTS.shape[0],)
        assert operation.decision_compare.decision.shape == (POINTS.shape[0],)

        operation.reset()
        input_encoder.reset()
        for expected_score, expected_decision in first_trace:
            score, decision = operation(input_encoder(POINTS.to(device)))
            assert torch.equal(score, expected_score)
            assert torch.equal(decision, expected_decision)


CONFIG = {
    # RBF differences, signed beta, and bias require bipolar input and score streams.
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'apply_operation': lambda operation, spikes: operation(spikes[0])[1],
    'output_polarity': 'unipolar',
    'encoder_dims': [20],
    'timesteps': 256,
    'state_timesteps': 16,
    'warmup_runs': 1,
    'trials': 3,
    'extra_checks': _composition_check,
}


def test_svm_rbf_ugemm():
    """Verify bipolar-only RBF decisions because differences and beta are signed."""
    streaming_suite(CONFIG)


def test_svm_rbf_fidelity():
    """Print requested-gamma score RMSE and decision agreement at three budgets."""
    reference = _exact_score(POINTS)
    reference_decision = reference.gt(0)
    for timestep in (256, 1024, 4096):
        for device in devices():
            operation = make_operation('bipolar', timestep, device).to(device)
            input_encoder = encode({
                'polarity': 'bipolar', 'timestep': timestep,
                'generator': 'sobol', 'dim': 20,
            }).to(device)
            score_sum = torch.zeros(POINTS.shape[0], device=device)
            decision_sum = torch.zeros_like(score_sum)
            points = POINTS.to(device)
            for _ in range(timestep):
                score, decision = operation(input_encoder(points))
                score_sum.add_(score)
                decision_sum.add_(decision)
            decoded_score = score_sum.div(timestep).mul(2).sub(1)
            decoded_score.mul_(operation.score_scale)
            decoded_decision = decision_sum.div(timestep).ge(0.5)
            rmse = torch.sqrt(torch.mean(
                (decoded_score.cpu() - reference).square()
            )).item()
            agreement = decoded_decision.cpu().eq(reference_decision).float().mean().item()
            print(
                f'RBF fidelity [{device}] timesteps={timestep}: '
                f'score_rmse={rmse:.6f}, decision_agreement={agreement:.6f}'
            )


def test_svm_rbf_rejects_invalid_inputs():
    """Verify every RBF-specific constructor guard and its exact message."""
    config = _config(32)
    cases = (
        (SUPPORT, BETA, BIAS, GAMMA, {**config, 'polarity': 'unipolar'},
         "Invalid polarity: <unipolar>; legal values: <['bipolar']>."),
        ([], BETA, BIAS, GAMMA, config,
         'Invalid support_vectors: <list>; legal values: a non-empty 2-D tensor shaped (support_count, in_features).'),
        (torch.empty(0, 2), torch.empty(0), BIAS, GAMMA, config,
         'Invalid support_vectors: <(0, 2)>; legal values: a non-empty 2-D tensor shaped (support_count, in_features).'),
        (torch.tensor([[math.nan, 0.0]]), torch.zeros(1), BIAS, GAMMA, config,
         'Invalid support_vectors: all values must be finite and in [-1, 1].'),
        (SUPPORT, torch.zeros(3), BIAS, GAMMA, config,
         'Invalid beta: <(3,)>; legal values: a 1-D tensor shaped (4,).'),
        (SUPPORT, torch.tensor([math.inf, 0.0, 0.0, 0.0]), BIAS, GAMMA, config,
         'Invalid beta: all values must be finite and in [-1, 1].'),
        (SUPPORT, BETA, math.inf, GAMMA, config,
         'Invalid bias: <inf>; legal values: a finite scalar in [-1, 1].'),
        (SUPPORT, BETA, BIAS, 0.0, config,
         'Invalid gamma: <0.0>; legal values: a positive finite int or float.'),
        (SUPPORT, BETA, BIAS, 0.2, config,
         'Invalid gamma: <0.2> gives kernel scale <1.6>; legal values satisfy 4 * in_features * gamma <= 1.'),
        (SUPPORT, BETA, BIAS, GAMMA, {**config, 'square_depth': 0},
         'Invalid square_depth: <0>; legal values: an integer of at least 1.'),
        (SUPPORT, BETA, BIAS, GAMMA, {**config, 'bi2uni_width': 1},
         'Invalid bi2uni_width: <1>; legal values: an integer of at least 2.'),
        (SUPPORT, BETA, BIAS, GAMMA, {**config, 'score_scale': 2.0},
         'Invalid score_scale: <2.0>; legal values: a positive finite number at least <2.2500000476837156>.'),
        (SUPPORT, BETA, BIAS, GAMMA, {**config, 'score_scale': 2.251},
         'Quantized score_scale <2.25> is below the coefficient L1 bound <2.2500000476837156>; increase score_scale or fracwidth.'),
    )
    for support, beta, bias, gamma, case_config, message in cases:
        try:
            svm_rbf_ugemm(support, beta, bias, gamma, case_config)
        except AssertionError as error:
            assert str(error) == message, error
        else:
            raise AssertionError(f'svm_rbf_ugemm accepted invalid case: {message}')


if __name__ == '__main__':
    test_svm_rbf_ugemm()
    test_svm_rbf_fidelity()
    test_svm_rbf_rejects_invalid_inputs()
    print('Test passed.')

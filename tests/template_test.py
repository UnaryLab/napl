# copy this file to tests/<subpackage>/test_<name>.py and fill in the TODOs; the name
# template_test.py is deliberate so tests/sweep_test.py's test_*.py glob skips it.
import math
import time

import torch

from napl.module import decoder, encoder
from napl.metric import analyze_error
from napl.utils import devices, sync, gen_rand_tensor

# TODO: import the streaming operation under test and assign its class here.
OP_TYPE = None
# TODO: provide a baseline runner with the same signature as run_op.
BASELINE_RUNNER = None


def run_op(val, device, timestep):
    val_1, val_2 = (item.to(device) for item in val)
    polarity = 'bipolar'  # TODO: set the operation polarity.
    codec_config_1 = {
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config_2 = {
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 2,
    }
    # distinct sobol dims decorrelate independent operand streams
    enc_1 = encoder(codec_config_1).to(device)
    enc_2 = encoder(codec_config_2).to(device)
    op_config = {'polarity': polarity}  # TODO: add the operation config and parameters.
    if OP_TYPE is None:
        raise NotImplementedError('TODO: set OP_TYPE to the operation under test')
    operation = OP_TYPE(op_config).to(device)  # TODO: adjust constructor arguments.
    dec = decoder(codec_config_1).to(device)

    sync(device)
    start = time.time()
    for _ in range(timestep):
        spike_1 = enc_1(val_1)
        spike_2 = enc_2(val_2)
        spike_out = operation(spike_1, spike_2)  # TODO: adjust the input/output arity.
        dec(spike_out)
    result = dec.spike_value.detach().cpu().clone()
    sync(device)
    elapsed = time.time() - start

    return result, elapsed, (enc_1, enc_2, operation, dec)


def test_fidelity():
    timestep = 256  # TODO: set the timestep used by this operation.
    val = (
        gen_rand_tensor('bipolar', shape=(1000,), width=8),
        gen_rand_tensor('bipolar', shape=(1000,), width=8),
    )
    reference_fn = None  # TODO: provide the analytic reference function.
    if reference_fn is None:
        raise NotImplementedError('TODO: provide the analytic reference function')
    reference = reference_fn(*val)

    for device in devices():
        result, elapsed, _ = run_op(val, device, timestep)
        error, _ = analyze_error(result, reference)
        rmse = error.pow(2).mean().sqrt()
        tolerance_scale = 1.0  # TODO: set the justified constant for this operation.
        tolerance = tolerance_scale / math.sqrt(timestep)
        assert rmse <= tolerance, f'[{device}] rmse={rmse.item():.4f}, bound={tolerance:.4f}'
        print(f'[{device}] rmse={rmse.item():.4f}, time={elapsed * 1000:.1f}ms')


def test_known_answer():
    timestep = 256  # TODO: set the known-answer timestep.
    val = (
        torch.tensor([0.0, 0.5, 1.0]),
        torch.tensor([1.0, 0.5, 0.0]),
    )
    expected = None  # TODO: enter the hand-computed expected output values.
    tolerance = 0.0  # TODO: set zero for exact checks or the justified tolerance.
    if expected is None:
        raise NotImplementedError('TODO: enter the known-answer expected values')

    for device in devices():
        result, _, _ = run_op(val, device, timestep)
        assert torch.allclose(result, expected, atol=tolerance, rtol=0)


def test_reset():
    timestep = 16  # TODO: set a short reset-check timestep.
    val = (torch.tensor([0.25]), torch.tensor([0.75]))

    for device in devices():
        _, _, modules = run_op(val, device, timestep)
        for module in modules:
            assert module.timestep_cur == timestep
            module.reset()
            assert module.timestep_cur == 0
        assert modules[-1].spike_count.abs().sum() == 0
        # TODO: assert that operation-specific streaming state returned to its initial value.


def test_performance():
    timestep = 256  # TODO: set the performance-test timestep.
    shared_val = (
        gen_rand_tensor('bipolar', shape=(1000,), width=8),
        gen_rand_tensor('bipolar', shape=(1000,), width=8),
    )
    if BASELINE_RUNNER is None:
        raise NotImplementedError('TODO: provide the baseline variant runner')

    for device in devices():
        candidate_val = tuple(item.clone() for item in shared_val)
        baseline_val = tuple(item.clone() for item in shared_val)
        assert all(torch.equal(a, b) for a, b in zip(candidate_val, baseline_val))
        _, candidate_elapsed, _ = run_op(candidate_val, device, timestep)
        _, baseline_elapsed = BASELINE_RUNNER(baseline_val, device, timestep)
        speedup = baseline_elapsed / candidate_elapsed
        minimum_speedup = 1.0  # TODO: set the required speedup against the baseline.
        assert speedup >= minimum_speedup
        print(f'[{device}] speedup={speedup:.2f}x')


if __name__ == '__main__':
    test_fidelity()
    test_known_answer()
    test_reset()
    test_performance()

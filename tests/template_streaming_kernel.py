# Copy to tests/<subpackage>/test_<name>.py and fill in every TODO.
# The template_ prefix keeps this file out of sweep_test.py's test_*.py glob.
#
# streaming_suite checks known answers, rank-2-or-higher analytic fidelity,
# timestep state, reset and replay, and performance on every device and
# polarity. Speedup is the saved CPU runtime divided by each device runtime.
# Use timer only for a separate unpaired elapsed measurement.
from napl.sim.metric import accuracy
from napl.utils._shared_test import streaming_suite


def make_operation(polarity, timestep, device):
    # TODO: return the configured streaming operation.
    raise NotImplementedError('TODO: implement make_operation')


def make_values(polarity):
    # TODO: return pre-generated CPU input tensors as a tuple. Fidelity inputs
    # are promoted to at least rank 2 by streaming_suite.
    raise NotImplementedError('TODO: implement make_values')


def analytic_reference(values, polarity):
    # TODO: return the analytic result as a CPU tensor.
    raise NotImplementedError('TODO: implement analytic_reference')


def known_answer_case(polarity):
    # TODO: return CPU (input_tuple, expected_output, absolute_tolerance).
    raise NotImplementedError('TODO: implement known_answer_case')


CONFIG = {
    # Required.
    'polarities': None,  # TODO: list all supported polarities.
    'tolerance_scale': None,  # TODO: justify C for C / sqrt(timesteps).
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Optional. Delete unchanged entries.
    # 'apply_operation': lambda operation, spikes: operation(*spikes),
    # 'input_polarities': lambda polarity: [polarity, polarity],
    # 'output_polarity': lambda polarity: polarity,
    # 'encoder_dims': lambda polarity: [1, 2],
    # 'encoder_generators': lambda polarity: ['sobol', 'sobol'],
    # 'make_readout': lambda _polarity, _timestep, _device: accuracy(
    #     {'polarity': 'unipolar'}
    # ),
    # 'extra_checks': extra_checks,
    # 'timesteps': 256,
    # 'warmup_runs': 2,
    # 'trials': 7,
}


def test_streaming_kernel():
    """TODO: state the behavior this test verifies."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_streaming_kernel()

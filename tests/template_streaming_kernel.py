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
    # TODO: return pre-generated CPU input tensors as a tuple over the full
    # legal range for the selected polarity. State any math restriction.
    # Fidelity inputs are promoted to at least rank 2 by streaming_suite.
    raise NotImplementedError('TODO: implement make_values')


def analytic_reference(values, polarity):
    # TODO: return the analytic result as a CPU tensor.
    raise NotImplementedError('TODO: implement analytic_reference')


def known_answer_case(polarity):
    # TODO: return CPU (input_tuple, expected_output, absolute_tolerance),
    # using the full legal range unless a stated math restriction applies.
    raise NotImplementedError('TODO: implement known_answer_case')


CONFIG = {
    # Required.
    # Polarities the whole suite loops over; list every supported polarity.
    'polarities': None,  # TODO: list all supported polarities.
    # C in the fidelity RMSE bound C / sqrt(timesteps).
    'tolerance_scale': None,  # TODO: justify C for C / sqrt(timesteps).
    # Builds the streaming operation for (polarity, timestep, device).
    'make_operation': make_operation,
    # Returns pre-generated CPU input value tensors, one per encoder.
    'make_values': make_values,
    # Optional larger inputs used only by the performance check.
    # 'make_performance_values': make_performance_values,
    # Exact expected decoded value for the make_values inputs.
    'analytic_reference': analytic_reference,
    # Per polarity: CPU (input_tuple, expected_output, absolute_tolerance).
    'known_answer_case': known_answer_case,
    # Optional. Delete unchanged entries.
    # Calls the operation on the encoded spikes; the default unpacks them.
    # 'apply_operation': lambda operation, spikes: operation(*spikes),
    # Encoder polarity per input; the default repeats the suite polarity.
    # 'input_polarities': lambda polarity: [polarity, polarity],
    # Decoder polarity; the default is the suite polarity.
    # 'output_polarity': lambda polarity: polarity,
    # Sobol dimension per encoder; the default is 1..input_count.
    # 'encoder_dims': lambda polarity: [1, 2],
    # Spike generator per encoder; the default is all 'sobol'.
    # 'encoder_generators': lambda polarity: ['sobol', 'sobol'],
    # Replaces the default decoder with a custom readout module.
    # 'make_readout': lambda _polarity, _timestep, _device: accuracy(
    #     {'polarity': 'unipolar'}
    # ),
    # Callable run after the four suite checks pass.
    # 'extra_checks': extra_checks,
    # Stream length for the known-answer, fidelity, and performance checks;
    # reset and replay always uses a short fixed stream.
    # 'timesteps': 256,
    # Untimed runs before timing each device.
    # 'warmup_runs': 2,
    # Timed runs per device; the median is reported.
    # 'trials': 7,
}


def test_streaming_kernel():
    """TODO: state the behavior this test verifies."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_streaming_kernel()

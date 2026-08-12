# Copy to tests/<subpackage>/test_<name>.py and fill in every TODO.
# The template_ prefix keeps this file out of sweep_test.py's test_*.py glob.
#
# streaming_suite checks known answers, rank-2-or-higher analytic fidelity,
# timestep state, reset and replay, and performance on every device and
# polarity. Speedup is the saved CPU runtime divided by each device runtime.
# Use timer only for a separate unpaired elapsed measurement.
#
# Every one of those checks compares a decoded value to an expected value, or a
# run to its own replay, so all of them pass for a kernel whose defect leaves
# the decoded value intact. Gate 17 of RULE_SIM.md requires a check that fails
# on an identity wire for that case; extra_checks below is one place to put it,
# a separate structural test in this file is another.
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
    # TODO: return CPU (input_tuple, expected_output), using the full legal
    # range unless a stated math restriction applies.
    raise NotImplementedError('TODO: implement known_answer_case')


CONFIG = {
    # Required.
    # Polarities the whole suite loops over; list every supported polarity.
    'polarities': None,  # TODO: list all supported polarities.
    # Gate 5 of RULE_SIM.md: the suite prints the measured fidelity error every
    # run and asserts no bound; the user reads the error by eye, so no tolerance,
    # ceiling, or scale is configured.
    # Builds the streaming operation for (polarity, timestep, device).
    'make_operation': make_operation,
    # Returns pre-generated CPU input value tensors, one per encoder.
    'make_values': make_values,
    # Optional larger inputs used only by the performance check.
    # 'make_random_perf_values': make_random_perf_values,
    # Exact expected decoded value for the make_values inputs.
    'analytic_reference': analytic_reference,
    # Per polarity: CPU (input_tuple, expected_output).
    'known_answer_case': known_answer_case,
    # Gate 17 of RULE_SIM.md: a callable asserting whatever the decoded value
    # cannot show, run after the four suite checks pass. Keep None only for a
    # kernel that a wrong decoded value already catches, or when a separate
    # structural test in this file carries that check instead. Confirm it by
    # returning the input unchanged from the kernel while its real state
    # updates keep running, so the wire neuters only the emitted values, and
    # rerunning this file, where at least one assertion must fail.
    'extra_checks': None,  # TODO: supply extra_checks, or state where it lives.
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
    # Stream length for the known-answer, fidelity, and performance checks.
    # 'timesteps': 256,
    # Stream length for the reset-and-replay check.
    # 'state_timesteps': 16,
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

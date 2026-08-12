# Copy to tests/<subpackage>/test_<name>.py and fill in every TODO.
# The template_ prefix keeps this file out of sweep_test.py's test_*.py glob.
#
# non_streaming_suite checks known answers, rank-2-or-higher PyTorch-reference
# fidelity, execution state, every declared STE gradient, and performance on
# every device. Speedup is the saved CPU runtime divided by each device runtime.
# Use timer only for a separate unpaired elapsed measurement.
#
# Every one of those checks compares a computed value to an expected value, so
# all of them pass for a kernel whose defect leaves those values intact. Gate 17
# of RULE_SIM.md requires a check that fails on an identity wire for that case;
# extra_checks below is one place to put it, a separate structural test in this
# file is another.
from napl.utils._shared_test import non_streaming_suite


def make_module_pair():
    # TODO: return CPU (candidate, PyTorch reference) with matched parameters.
    raise NotImplementedError('TODO: implement make_module_pair')


def make_inputs():
    # TODO: return pre-generated CPU input tensors as a tuple. Fidelity inputs
    # are promoted to at least rank 2 by non_streaming_suite.
    raise NotImplementedError('TODO: implement make_inputs')


def known_answer_case():
    # TODO: return CPU (candidate, input_tuple, expected_output).
    raise NotImplementedError('TODO: implement known_answer_case')


def gradient_case():
    # TODO: return CPU (candidate, input_tuple) for the STE equations.
    raise NotImplementedError('TODO: implement gradient_case')


def expected_ste_gradients(candidate, inputs, grad_output):
    # TODO: return (input_gradient_tuple, named_parameter_gradient_dict)
    # from the defining STE equations.
    raise NotImplementedError('TODO: implement expected_ste_gradients')


CONFIG = {
    # Required.
    # Gate 5 of RULE_SIM.md: the suite prints the measured candidate-versus-
    # PyTorch error and the known-answer error, and asserts no bound.
    # Absolute tolerance for each STE gradient versus its defining equation.
    'gradient_atol': None,  # TODO: set the STE absolute tolerance.
    # Relative tolerance for each STE gradient versus its defining equation.
    'gradient_rtol': None,  # TODO: set the STE relative tolerance.
    # Builds CPU (candidate, PyTorch reference) with matched parameters.
    'make_module_pair': make_module_pair,
    # Returns pre-generated CPU input tensors for fidelity and performance.
    'make_inputs': make_inputs,
    # Optional larger inputs used only by the performance check.
    # 'make_random_perf_values': make_random_perf_values,
    # Returns CPU (candidate, input_tuple, expected_output).
    'known_answer_case': known_answer_case,
    # Returns CPU (candidate, input_tuple) for the gradient check.
    'gradient_case': gradient_case,
    # Returns the expected input and parameter gradients from the STE equations.
    'expected_ste_gradients': expected_ste_gradients,
    # Gate 17 of RULE_SIM.md: a callable asserting whatever the compared output
    # and gradient values cannot show, run after the suite checks pass. Keep
    # None only for a kernel that a wrong output or gradient already catches, or
    # when a separate structural test in this file carries that check instead.
    # Confirm it by returning the input unchanged from the kernel while its
    # real state updates keep running, so the wire neuters only the emitted
    # values, and rerunning this file, where at least one assertion must fail.
    'extra_checks': None,  # TODO: supply extra_checks, or state where it lives.
    # Optional. Delete unchanged entries.
    # Untimed runs before timing each device.
    # 'warmup_runs': 2,
    # Timed runs per device; the median is reported.
    # 'trials': 7,
}


def test_non_streaming_trainable():
    """TODO: state the behavior this test verifies."""
    non_streaming_suite(CONFIG)


if __name__ == '__main__':
    test_non_streaming_trainable()

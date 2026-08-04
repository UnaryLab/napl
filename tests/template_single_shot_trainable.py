# Copy to tests/<subpackage>/test_<name>.py and fill in every TODO.
# The template_ prefix keeps this file out of sweep_test.py's test_*.py glob.
#
# single_shot_suite checks known answers, rank-2-or-higher PyTorch-reference
# fidelity, execution state, every declared STE gradient, and performance on
# every device. Speedup is the saved CPU runtime divided by each device runtime.
# Use timer only for a separate unpaired elapsed measurement.
from napl.utils._shared_test import single_shot_suite


def make_module_pair():
    # TODO: return CPU (candidate, PyTorch reference) with matched parameters.
    raise NotImplementedError('TODO: implement make_module_pair')


def make_inputs():
    # TODO: return pre-generated CPU input tensors as a tuple. Fidelity inputs
    # are promoted to at least rank 2 by single_shot_suite.
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
    # Absolute tolerance for candidate output versus the PyTorch reference.
    'quantization_atol': None,  # TODO: set the justified output tolerance.
    # Absolute tolerance for the known-answer output.
    'known_answer_atol': None,  # TODO: use 0 for an exact known answer.
    # Absolute tolerance for each STE gradient versus its defining equation.
    'gradient_atol': None,  # TODO: set the STE absolute tolerance.
    # Relative tolerance for each STE gradient versus its defining equation.
    'gradient_rtol': None,  # TODO: set the STE relative tolerance.
    # Builds CPU (candidate, PyTorch reference) with matched parameters.
    'make_module_pair': make_module_pair,
    # Returns pre-generated CPU input tensors for fidelity and performance.
    'make_inputs': make_inputs,
    # Optional larger inputs used only by the performance check.
    # 'make_performance_values': make_performance_values,
    # Returns CPU (candidate, input_tuple, expected_output).
    'known_answer_case': known_answer_case,
    # Returns CPU (candidate, input_tuple) for the gradient check.
    'gradient_case': gradient_case,
    # Returns the expected input and parameter gradients from the STE equations.
    'expected_ste_gradients': expected_ste_gradients,
    # Optional. Delete unchanged entries.
    # Callable run after the suite checks pass.
    # 'extra_checks': extra_checks,
    # Untimed runs before timing each device.
    # 'warmup_runs': 2,
    # Timed runs per device; the median is reported.
    # 'trials': 7,
}


def test_single_shot_trainable():
    """TODO: state the behavior this test verifies."""
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_single_shot_trainable()

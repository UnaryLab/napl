# Copy to tests/<subpackage>/test_<name>.py and fill in every TODO.
# The template_ prefix keeps this file out of sweep_test.py's test_*.py glob.
#
# single_shot_suite checks known answers, PyTorch-reference fidelity,
# execution state, every declared STE gradient, and performance on every
# device. Speedup is the saved CPU runtime divided by each device runtime for
# the same NAPL module and identical pre-generated inputs.
# Use timer only for a separate unpaired elapsed measurement.
from napl.utils._shared_test import single_shot_suite


def make_module_pair():
    # TODO: return CPU (candidate, PyTorch reference) with matched parameters.
    raise NotImplementedError('TODO: implement make_module_pair')


def make_inputs():
    # TODO: return pre-generated CPU input tensors as a tuple.
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
    'quantization_atol': None,  # TODO: set the justified output tolerance.
    'known_answer_atol': None,  # TODO: use 0 for an exact known answer.
    'gradient_atol': None,  # TODO: set the STE absolute tolerance.
    'gradient_rtol': None,  # TODO: set the STE relative tolerance.
    'make_module_pair': make_module_pair,
    'make_inputs': make_inputs,
    'known_answer_case': known_answer_case,
    'gradient_case': gradient_case,
    'expected_ste_gradients': expected_ste_gradients,
    # Optional. Delete unchanged entries.
    # 'warmup_runs': 2,
    # 'trials': 7,
}


def test_single_shot_trainable():
    single_shot_suite(CONFIG)


if __name__ == '__main__':
    test_single_shot_trainable()

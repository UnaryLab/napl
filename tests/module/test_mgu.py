import torch

from napl.sim.base import global_config
from napl.sim.module import mgu, mgu_hard
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256
ISZ, HSZ, BATCH = 4, 3, 2


def _reference_cell():
    """Return an mgu_hard cell holding the fixed gate parameters used by the suite."""
    cell = mgu_hard(ISZ, HSZ, bias=True)
    torch.manual_seed(0)
    with torch.no_grad():
        cell.weight_f.copy_(torch.linspace(-0.5, 0.5, HSZ * (HSZ + ISZ)).view(HSZ, HSZ + ISZ))
        cell.weight_n.copy_(torch.linspace(0.5, -0.5, HSZ * (HSZ + ISZ)).view(HSZ, HSZ + ISZ))
        cell.bias_f.copy_(torch.linspace(-0.2, 0.2, HSZ))
        cell.bias_n.copy_(torch.linspace(0.2, -0.2, HSZ))
    return cell


def _hx_value():
    return torch.linspace(-0.6, 0.6, BATCH * HSZ).view(BATCH, HSZ).type(global_config.ntype)


def _input_value():
    return torch.linspace(0.6, -0.6, BATCH * ISZ).view(BATCH, ISZ).type(global_config.ntype)


def make_operation(polarity, timestep, _device):
    ref = _reference_cell()
    return mgu(
        ref.weight_f.data, ref.bias_f.data, ref.weight_n.data, ref.bias_n.data, _hx_value(),
        {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol'},
    )


def make_values(_polarity):
    return (_input_value(), _hx_value())


def analytic_reference(values, _polarity):
    """The binary-domain cell the streaming run approximates."""
    return _reference_cell()(values[0], values[1])


def known_answer_case(_polarity):
    """Zero input against the fixed gates still follows the binary-domain cell."""
    values = (torch.zeros(BATCH, ISZ), _hx_value())
    return values, _reference_cell()(values[0], values[1]), 0.2


def check_bipolar_only():
    """Verify mgu rejects unipolar configuration."""
    ref = _reference_cell()
    try:
        mgu(ref.weight_f.data, ref.bias_f.data, ref.weight_n.data, ref.bias_n.data, _hx_value(),
            {'polarity': 'unipolar', 'timestep': 16, 'generator': 'sobol'})
    except AssertionError:
        return
    raise AssertionError('mgu accepted a unipolar configuration')


CONFIG = {
    'polarities': ['bipolar'],
    'tolerance_scale': 3.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [1, 2],
    'timesteps': TIMESTEPS,
    'extra_checks': check_bipolar_only,
}


def test_mgu():
    """Verify the streaming MGU cell against its binary-domain reference, with reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_mgu()

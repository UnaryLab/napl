import torch
import torch.nn.functional as F

from napl.sim.base import global_config
from napl.sim.module import mgu_hard_mix
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256
ISZ, HSZ, BATCH = 4, 3, 2


def _reference_parameters():
    """Return fixed gate tensors used by the suite's binary-domain reference."""
    torch.manual_seed(0)
    weight_f = torch.linspace(-0.5, 0.5, HSZ * (HSZ + ISZ)).view(HSZ, HSZ + ISZ)
    weight_n = torch.linspace(0.5, -0.5, HSZ * (HSZ + ISZ)).view(HSZ, HSZ + ISZ)
    bias_f = torch.linspace(-0.2, 0.2, HSZ)
    bias_n = torch.linspace(0.2, -0.2, HSZ)
    return weight_f, bias_f, weight_n, bias_n


def _reference_mgu(input_value, hx_value):
    """Direct hard-activation MGU equation used as the binary-domain reference."""
    weight_f, bias_f, weight_n, bias_n = _reference_parameters()
    fg_in = F.hardtanh(F.linear(torch.cat((hx_value, input_value), 1), weight_f, bias_f))
    fg = F.hardsigmoid(fg_in * 3)
    fg_hx = fg * hx_value
    ng = F.hardtanh(F.linear(torch.cat((fg_hx, input_value), 1), weight_n, bias_n))
    return F.hardtanh(ng - fg * ng + fg_hx)


def _hx_value():
    return torch.linspace(-0.6, 0.6, BATCH * HSZ).view(BATCH, HSZ).type(global_config.ntype)


def _input_value():
    return torch.linspace(0.6, -0.6, BATCH * ISZ).view(BATCH, ISZ).type(global_config.ntype)


def make_operation(polarity, timestep, _device):
    weight_f, bias_f, weight_n, bias_n = _reference_parameters()
    return mgu_hard_mix(
        weight_f, bias_f, weight_n, bias_n, _hx_value(),
        {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol'},
    )


def make_values(_polarity):
    return (_input_value(), _hx_value())


def analytic_reference(values, _polarity):
    """The binary-domain cell the streaming run approximates."""
    return _reference_mgu(values[0], values[1])


def known_answer_case(_polarity):
    """Zero input against the fixed gates still follows the binary-domain cell."""
    values = (torch.zeros(BATCH, ISZ), _hx_value())
    return values, _reference_mgu(values[0], values[1])


def check_bipolar_only():
    """Verify mgu_hard_mix rejects unipolar configuration and reports a held encoder."""
    weight_f, bias_f, weight_n, bias_n = _reference_parameters()
    # The gate multipliers encode their own operands.
    assert make_operation('bipolar', TIMESTEPS, 'cpu').internal_encode is True
    try:
        mgu_hard_mix(weight_f, bias_f, weight_n, bias_n, _hx_value(),
            {'polarity': 'unipolar', 'timestep': 16, 'generator': 'sobol'})
    except AssertionError:
        return
    raise AssertionError('mgu_hard_mix accepted a unipolar configuration')


CONFIG = {
    'polarities': ['bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [1, 2],
    'timesteps': TIMESTEPS,
    # The default depth_ismul-6 multiplier needs more than 64 timesteps to flush.
    'state_timesteps': 128,
    'extra_checks': check_bipolar_only,
}


def test_mgu_hard_mix():
    """Verify the streaming MGU cell against its binary-domain reference, with reset and timing."""
    streaming_suite(CONFIG)


def test_mgu_hard_mix_width_drives_every_adder():
    """Verify the width key sets the forget-gate sigmoid adder as well as the output adder."""
    weight_f, bias_f, weight_n, bias_n = _reference_parameters()
    cell = mgu_hard_mix(weight_f, bias_f, weight_n, bias_n, _hx_value(),
               {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol', 'width': 9,
                'depth_ismul': 3})
    assert cell.fg_sigmoid.scaled_add.intwidth == 9
    assert cell.hy_add.intwidth == 9
    print('Test passed.')


def test_mgu_hard_mix_run_length():
    """Verify mgu_hard_mix rejects a run that does not outlast the multiplier shift register."""
    weight_f, bias_f, weight_n, bias_n = _reference_parameters()

    def build(timestep, depth_ismul):
        return mgu_hard_mix(weight_f, bias_f, weight_n, bias_n,
                   _hx_value(),
                   {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol',
                    'depth_ismul': depth_ismul})

    for timestep, depth_ismul in ((16, 6), (64, 6), (4, 2)):
        try:
            build(timestep, depth_ismul)
        except AssertionError as error:
            print(f'expected AssertionError at timestep={timestep}, '
                  f'depth_ismul={depth_ismul}: {error}')
        else:
            raise AssertionError('a run that does not outlast the shift register must raise')

    # A run one step longer than the shift register is legal and still emits spikes.
    cell = build(5, 2)
    assert cell(torch.ones(BATCH, ISZ), torch.ones(BATCH, HSZ)).shape == (BATCH, HSZ)
    print('Test passed.')


if __name__ == '__main__':
    test_mgu_hard_mix()
    test_mgu_hard_mix_width_drives_every_adder()
    test_mgu_hard_mix_run_length()

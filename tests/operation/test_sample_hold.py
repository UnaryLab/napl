import torch

from napl.sim.operation import decode, encode, sample_hold
from napl.utils._shared_test import devices, streaming_suite


# The suite latches at half the run length, so the pass-through half and the
# re-emitted half both decode to the input value and the whole output stream
# reads back that value (fidelity is printed, never asserted).
def make_operation(polarity, timestep, _device):
    return sample_hold({'polarity': polarity, 'timestep': timestep,
                        'trigger_timestep': max(1, timestep // 2)})


def make_values(polarity):
    # Full legal range: unipolar [0, 1], bipolar [-1, 1].
    if polarity == 'unipolar':
        return (torch.tensor([0.0, 0.25, 0.5, 1.0]),)
    return (torch.tensor([-1.0, -0.25, 0.5, 1.0]),)


def make_random_perf_values(polarity):
    if polarity == 'unipolar':
        return (torch.rand(131072),)
    return (2.0 * torch.rand(131072) - 1.0,)


def analytic_reference(values, _polarity):
    # The frozen value reused across the run is the input value itself.
    return values[0].clone()


def known_answer_case(polarity):
    # Hold a known value; the re-emitted stream decodes back to it.
    if polarity == 'unipolar':
        values = torch.tensor([0.25, 0.5, 0.75])
    else:
        values = torch.tensor([-0.5, 0.0, 0.5])
    return (values,), values.clone()


def _constant_reemission_check():
    """
    Gate 17 identity-wire check: after the trigger the output is a constant
    re-emission of the latched value that does not change with further input.

    A wire (output == input) would fail this: the re-emitted stream is driven by
    the frozen estimate and is decorrelated from the live input fed after the
    trigger, so its decoded value tracks the held value, not the changed input.
    """
    torch.manual_seed(0)
    timestep = 256
    trigger = 64
    for polarity in ('unipolar', 'bipolar'):
        codec = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol'}
        for device in devices():
            enc = encode(codec).to(device)
            op = sample_hold({'polarity': polarity, 'timestep': timestep,
                              'trigger_timestep': trigger}).to(device)
            dec = decode(codec).to(device)

            # Rank-2 input covers gate 4 for the structural check.
            value = torch.full((2, 3), 0.75 if polarity == 'unipolar' else 0.25,
                               device=device)
            # Drive the trigger phase with the real value, then feed a contradicting
            # value afterward; the output must ignore the change.
            switched = torch.full_like(value, 0.1 if polarity == 'unipolar' else -0.6)
            frozen = None
            for step in range(1, timestep + 1):
                src = value if step <= trigger else switched
                out = op(enc(src))
                assert out.shape == value.shape
                if step == trigger:
                    # Snapshot the latched value; later switched input must not move it.
                    frozen = op.held.clone()
                if step > trigger:
                    dec(out)
                    # The frozen value is stable across every later timestep.
                    assert torch.equal(op.held, frozen)
            assert op.latched
            # The post-trigger decode reuses the latched value, not the switched input.
            held = op.held
            decoded = dec.spike_value
            reuse_error = (decoded - held).abs().max().item()
            switched_error = (decoded - switched).abs().max().item()
            assert reuse_error < switched_error, (
                f'[{device}][{polarity}] output tracks live input, not the frozen value: '
                f'reuse_error={reuse_error:.4f} switched_error={switched_error:.4f}')
    print('sample_hold constant-re-emission check passed.')


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Gate 17 applies: sample_hold is not identity-blind in the ordinary sense
    # (an identity wire changes the decoded value the suite reads), but the
    # constant-re-emission property is what defines the kernel, so the check
    # asserts that the post-trigger output reuses the frozen value rather than
    # the live input, and fails when the op is replaced by a pass-through wire.
    'extra_checks': _constant_reemission_check,
}


def test_sample_hold():
    """Verify sample_hold latch-and-reuse for both polarities across devices."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_sample_hold()

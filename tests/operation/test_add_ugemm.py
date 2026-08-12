import torch
import math

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import add_ugemm, decode, encode
from napl.sim.metric import accuracy


class napl_add_ugemm(napl_base):
    def __init__(self, codec_config, add_ugemm_config):
        super().__init__()
        self.encoder = encode(codec_config)
        self.decoder = decode(codec_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})
        self.add_ugemm = add_ugemm(add_ugemm_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.add_ugemm(i_spike, dim=-1)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def run_case(polarity, scaled, input, device, timestep=256):
    codec_config = {
        'polarity': polarity,
        'timestep': timestep,
        'generator': 'sobol',
    }
    add_ugemm_config = {
        'polarity': polarity,
        'scaled': scaled,
        'width': 16,
    }
    input = input.to(device)
    inst = napl_add_ugemm(codec_config, add_ugemm_config).to(device)

    with timer(device) as elapsed:
        inst(input, timesteps=timestep)

    entry = input.size(-1)
    if scaled:
        # Scaled mode averages across entries.
        r_value = torch.sum(input, dim=-1) / entry
    else:
        # Non-scaled mode clips the sum to the polarity range.
        lo = -1.0 if polarity == 'bipolar' else 0.0
        r_value = torch.sum(input, dim=-1).clamp(lo, 1.0)

    error, _ = inst.accuracy.analyze(r_value, verbose=True)
    mae = error.abs().mean().item()

    assert inst.add_ugemm.timestep_cur == timestep
    inst.reset()
    assert inst.add_ugemm.timestep_cur == 0

    print(f'[{device}] polarity={polarity} scaled={scaled}: MAE={mae:.5f} time={elapsed.seconds:.3f}s')


def _kernel_specific_checks():
    """
    Test add_ugemm (scaled and non-scaled, unipolar and bipolar) on every device.
    """
    timestep = 256
    entry = 8
    width = math.log2(timestep)

    # Reuse inputs across devices and variants.
    input_scaled_uni = gen_rand_tensor('unipolar', shape=(2000, entry), width=width).type(global_config.ntype)
    input_scaled_bi = gen_rand_tensor('bipolar', shape=(2000, entry), width=width).type(global_config.ntype)
    # Keep non-scaled sums within the output range.
    input_ns_uni = input_scaled_uni / entry
    input_ns_bi = input_scaled_bi / entry

    for device in devices():
        run_case('unipolar', True, input_scaled_uni, device, timestep)
        run_case('bipolar', True, input_scaled_bi, device, timestep)
        run_case('unipolar', False, input_ns_uni, device, timestep)
        run_case('bipolar', False, input_ns_bi, device, timestep)

    print('Test passed.')


def _suite_config(polarity, scaled):
    entry = 8
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    scale = 1.0 if scaled else 1.0 / entry

    def make_operation(_polarity, _timestep, _device):
        return add_ugemm({'polarity': polarity, 'scaled': scaled, 'width': 16})

    def make_values(_polarity):
        values = torch.linspace(low, high, 512).reshape(64, entry) * scale
        return (values,)

    def make_random_perf_values(_polarity):
        values = torch.linspace(low, high, 131072).reshape(16384, entry) * scale
        return (values,)

    def analytic_reference(values, _polarity):
        result = values[0].sum(dim=-1)
        if scaled:
            result = result / entry
        return result.clamp(-1.0 if polarity == 'bipolar' else 0.0, 1.0)

    def known_answer_case(_polarity):
        value = 1.0
        values = torch.full((8, entry), value * scale)
        return (values,), analytic_reference((values,), polarity)

    return {
        'make_operation': make_operation,
        'make_values': make_values,
        'make_random_perf_values': make_random_perf_values,
        'analytic_reference': analytic_reference,
        'known_answer_case': known_answer_case,
        'polarities': [polarity],
        'timesteps': 256,
        'apply_operation': lambda operation, spikes: operation(spikes[0], dim=-1),
    }


CONFIGS = [
    _suite_config('unipolar', True),
    _suite_config('bipolar', True),
    _suite_config('unipolar', False),
    _suite_config('bipolar', False),
]
CONFIGS[-1]['extra_checks'] = _kernel_specific_checks


def test_add_ugemm():
    """Verify add_ugemm against analytic and known-answer streams, including reset and timing."""
    for config in CONFIGS:
        streaming_suite(config)


def test_add_ugemm_width_saturation():
    """Verify a narrow width saturates the accumulator and lowers the realized output rate."""
    # Unipolar scaled with entry 3 and a per-timestep sum of 2: with a generous width the
    # accumulator walks 2, 4->1, 3->0 and emits 2 spikes per 3 timesteps; with acc_max = 3
    # the walk is clipped to 2, 4->3->0 and emits 1 spike per 2 timesteps.
    steps = 8
    spike = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=global_config.stype)
    expected_generous = torch.tensor([0, 1, 1, 0, 1, 1, 0, 1], dtype=global_config.stype)
    expected_narrow = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=global_config.stype)

    for device in devices():
        spike_dev = spike.to(device)
        generous = add_ugemm({'polarity': 'unipolar', 'scaled': True, 'width': 10}).to(device)
        narrow = add_ugemm({'polarity': 'unipolar', 'scaled': True, 'width': 3}).to(device)
        assert narrow.acc_max == 3 and narrow.acc_min == -4

        for step in range(steps):
            o_generous = generous(spike_dev, dim=-1)
            o_narrow = narrow(spike_dev, dim=-1)
            assert o_generous.shape == (2,), f'{device}: generous output shape {o_generous.shape}'
            assert o_narrow.shape == (2,), f'{device}: narrow output shape {o_narrow.shape}'
            assert torch.equal(o_generous.cpu(), expected_generous[step].expand(2)), (
                f'{device}: generous output at step {step} is {o_generous}'
            )
            assert torch.equal(o_narrow.cpu(), expected_narrow[step].expand(2)), (
                f'{device}: narrow output at step {step} is {o_narrow}'
            )
        # Saturation discards accumulated credit, so the narrow stream emits fewer spikes.
        assert expected_narrow.sum() < expected_generous.sum()
        assert narrow.accumulator.abs().max().item() <= narrow.acc_max

        print(f'[{device}] width saturation: narrow rate {expected_narrow.float().mean():.3f} '
              f'< generous rate {expected_generous.float().mean():.3f}')

    print('Test passed.')


def test_add_ugemm_width_required():
    """Verify a config without 'width' is rejected by the napl_base key check."""
    try:
        add_ugemm({'polarity': 'unipolar', 'scaled': True})
    except AssertionError as error:
        assert str(error) == 'Missing key <width> in the input configuration.', error
    else:
        raise AssertionError('add_ugemm accepted a config without <width>')

    print('missing width rejected as expected.')
    print('Test passed.')


if __name__ == '__main__':
    test_add_ugemm()
    test_add_ugemm_width_saturation()
    test_add_ugemm_width_required()

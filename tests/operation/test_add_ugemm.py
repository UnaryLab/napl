import torch
import math

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import encode, decode
from napl.sim.operation import add_ugemm
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
    bound = 2 / math.sqrt(timestep)
    assert mae < bound, f'{device} {polarity} scaled={scaled}: MAE {mae} exceeds SC bound {bound}'

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
        return add_ugemm({'polarity': polarity, 'scaled': scaled})

    def make_values(_polarity):
        values = torch.linspace(low, high, 512).reshape(64, entry) * scale
        return (values,)

    def make_performance_values(_polarity):
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
        return (values,), analytic_reference((values,), polarity), 2.0 / math.sqrt(256)

    return {
        'make_operation': make_operation,
        'make_values': make_values,
        'make_performance_values': make_performance_values,
        'analytic_reference': analytic_reference,
        'known_answer_case': known_answer_case,
        'polarities': [polarity],
        'timesteps': 256,
        'tolerance_scale': 2.0,
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


if __name__ == '__main__':
    test_add_ugemm()

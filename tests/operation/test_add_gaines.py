import time
import math
import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import add_any
# add_gaines is not exported from operation/__init__.py.
from napl.sim.operation.add_gaines import add_gaines
from napl.sim.metric import accuracy


class napl_add_gaines_scaled(napl_base):
    """One shared-RNG encoder over the (entry, col) tensor, MUX add over dim 0."""


    def __init__(self, codec_config, add_config):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})
        self.add_gaines = add_gaines(add_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = self.encoder(input)
        o_spike = self.add_gaines(i_spike, dim=0)
        self.decoder(o_spike)
        self.accuracy(o_spike)


class napl_add_gaines_or(napl_base):
    """Per-row decorrelated encoders (distinct Sobol dims), OR add over dim 0."""


    def __init__(self, codec_configs, add_config):
        super().__init__()
        self.encoders = torch.nn.ModuleList([encoder(c) for c in codec_configs])
        self.decoder = decoder(codec_configs[0])
        self.accuracy = accuracy({'polarity': codec_configs[0]['polarity']})
        self.add_gaines = add_gaines(add_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        i_spike = torch.stack([enc(input[i]) for i, enc in enumerate(self.encoders)], 0)
        o_spike = self.add_gaines(i_spike, dim=0)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def _kernel_specific_checks():
    """
    Gaines addition: the scaled MUX recovers mean(input, dim) within the SC bound for both
    polarities; the non-scaled OR recovers 1 - prod(1 - p) for decorrelated unipolar streams.
    UnarySim reference: GainesAdd.
    """
    timestep = 256
    entry = 8
    col = 10000
    bound = 3.0 / (timestep ** 0.5)
    scaled_input_cpu = {
        polarity: gen_rand_tensor(
            polarity,
            shape=(entry, col),
            width=math.log2(timestep),
        ).type(global_config.ntype)
        for polarity in ['unipolar', 'bipolar']
    }
    or_input_cpu = gen_rand_tensor(
        'unipolar',
        shape=(4, col),
        width=math.log2(timestep),
    ).type(global_config.ntype) * 0.15

    for device in devices():
        # Scaled MUX addition supports both polarities.
        for polarity in ['unipolar', 'bipolar']:
            codec_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
            add_config = {'polarity': polarity, 'scaled': True, 'entry': entry,
                          'generator': 'sobol', 'dim': 5}

            input = scaled_input_cpu[polarity].to(device)

            inst = napl_add_gaines_scaled(codec_config, add_config).to(device)
            inst(input, timesteps=timestep)

            r_value = input.mean(0)
            inst.accuracy.analyze(r_value, verbose=True)
            err = torch.sqrt((inst.decoder.spike_value - r_value).pow(2).mean()).item()
            assert err < bound, f'{device}/{polarity}/scaled: rmse {err} >= {bound}'
            assert inst.add_gaines.timestep_cur == timestep
            inst.reset()
            assert inst.add_gaines.idx == 0

            # Identical rows pass through the MUX for every select value.
            mux = add_gaines(dict(add_config)).to(device)
            ones = torch.ones(entry, 4, dtype=global_config.stype, device=device)
            assert torch.equal(mux(ones, dim=0).cpu(), torch.ones(4, dtype=global_config.stype))
            zeros = torch.zeros(entry, 4, dtype=global_config.stype, device=device)
            assert torch.equal(mux(zeros, dim=0).cpu(), torch.zeros(4, dtype=global_config.stype))
            mux.reset()

            print(f'{device}/{polarity}/scaled: rmse {err:.4f} (bound {bound:.4f})')

        # Non-scaled OR addition uses small, decorrelated unipolar inputs.
        or_entry = 4
        codec_configs = [{'polarity': 'unipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': d + 1}
                         for d in range(or_entry)]
        add_config = {'polarity': 'unipolar', 'scaled': False}

        input = or_input_cpu.to(device)

        inst = napl_add_gaines_or(codec_configs, add_config).to(device)
        inst(input, timesteps=timestep)

        r_value = 1 - torch.prod(1 - input, 0)
        inst.accuracy.analyze(r_value, verbose=True)
        err = torch.sqrt((inst.decoder.spike_value - r_value).pow(2).mean()).item()
        assert err < bound, f'{device}/or: rmse {err} >= {bound}'
        inst.reset()

        gate = add_gaines(dict(add_config)).to(device)
        a = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=global_config.stype, device=device)
        assert torch.equal(gate(a, dim=0).cpu(), torch.tensor([0, 1, 1, 1], dtype=global_config.stype))
        gate.reset()

        print(f'{device}/unipolar/or: rmse {err:.4f} (bound {bound:.4f})')

    # Non-scaled bipolar mode is invalid.
    try:
        add_gaines({'polarity': 'bipolar', 'scaled': False})
        assert False, 'bipolar non-scaled should be rejected'
    except AssertionError as e:
        if 'rejected' in str(e):
            raise

    print('Test passed.')


def _kernel_specific_perf():
    """Per device, time add_gaines (scaled MUX) against add_any (the obvious baseline) on identical spikes."""
    iters = 200
    entry = 8
    spikes_cpu = (torch.rand(entry, 100000) > 0.5).type(global_config.stype)
    for device in devices():
        spikes = spikes_cpu.to(device)
        results = {}

        op = add_any({'polarity': 'unipolar', 'scale': entry, 'width': 10}).to(device)
        op(spikes, dim=0)  # Warm up before timing.
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            op(spikes, dim=0)
        sync(device)
        results['add_any'] = time.perf_counter() - t0

        op = add_gaines({'polarity': 'unipolar', 'scaled': True, 'entry': entry,
                         'generator': 'sobol', 'dim': 5}).to(device)
        op(spikes, dim=0)  # Warm up before timing.
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            op(spikes, dim=0)
        sync(device)
        results['add_gaines'] = time.perf_counter() - t0

        ratio = results['add_any'] / results['add_gaines']
        print(f'{device}: add_gaines {results["add_gaines"]:.4f}s vs add_any {results["add_any"]:.4f}s, speedup x{ratio:.2f}')

    print('Perf test passed.')


def _scaled_operation(polarity, _timestep, _device):
    return add_gaines({
        'polarity': polarity,
        'scaled': True,
        'entry': 8,
        'generator': 'sobol',
        'dim': 5,
    })


def _scaled_values(polarity):
    lo = -0.75 if polarity == 'bipolar' else 0.0
    return (torch.linspace(lo, 0.75, 64).repeat(8, 1),)


def _scaled_reference(values, _polarity):
    return values[0].mean(dim=0)


def _scaled_known_answer(polarity):
    value = -0.25 if polarity == 'bipolar' else 0.25
    values = torch.full((8, 8), value)
    return (values,), values.mean(dim=0), 3.0 / math.sqrt(256)


SCALED_CONFIG = {
    'make_operation': _scaled_operation,
    'make_values': _scaled_values,
    'analytic_reference': _scaled_reference,
    'known_answer_case': _scaled_known_answer,
    'polarities': ['unipolar', 'bipolar'],
    'timesteps': 256,
    'tolerance_scale': 3.0,
    'apply_operation': lambda operation, spikes: operation(spikes[0], dim=0),
}


def _or_operation(_polarity, _timestep, _device):
    return add_gaines({'polarity': 'unipolar', 'scaled': False})


def _or_values(_polarity):
    base = torch.linspace(0.0, 0.15, 64)
    return tuple(base.roll(index * 7) for index in range(4))


def _or_reference(values, _polarity):
    return 1 - torch.prod(1 - torch.stack(values), dim=0)


def _or_known_answer(_polarity):
    values = (
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
    return values, expected, 0.0


def _all_kernel_specific_checks():
    _kernel_specific_checks()
    _kernel_specific_perf()


OR_CONFIG = {
    'make_operation': _or_operation,
    'make_values': _or_values,
    'analytic_reference': _or_reference,
    'known_answer_case': _or_known_answer,
    'polarities': ['unipolar'],
    'timesteps': 256,
    'tolerance_scale': 3.0,
    'apply_operation': lambda operation, spikes: operation(torch.stack(spikes), dim=0),
    'extra_checks': _all_kernel_specific_checks,
}


def test_add_gaines():
    """Verify add_gaines against analytic and known-answer streams, including reset and timing."""
    streaming_suite(SCALED_CONFIG)
    streaming_suite(OR_CONFIG)


if __name__ == '__main__':
    test_add_gaines()

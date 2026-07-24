import time
import math
import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation import add_any
# direct module import: not yet wired into operation/__init__.py
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


def test_add_gaines():
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
        # scaled MUX addition, both polarities
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

            # known-answer MUX check: identical rows pass through unchanged for any select
            mux = add_gaines(dict(add_config)).to(device)
            ones = torch.ones(entry, 4, dtype=global_config.stype, device=device)
            assert torch.equal(mux(ones, dim=0).cpu(), torch.ones(4, dtype=global_config.stype))
            zeros = torch.zeros(entry, 4, dtype=global_config.stype, device=device)
            assert torch.equal(mux(zeros, dim=0).cpu(), torch.zeros(4, dtype=global_config.stype))
            mux.reset()

            print(f'{device}/{polarity}/scaled: rmse {err:.4f} (bound {bound:.4f})')

        # non-scaled OR addition, unipolar only, small decorrelated inputs
        or_entry = 4
        codec_configs = [{'polarity': 'unipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': d + 1}
                         for d in range(or_entry)]
        add_config = {'polarity': 'unipolar', 'scaled': False}

        input = or_input_cpu.to(device)

        inst = napl_add_gaines_or(codec_configs, add_config).to(device)
        inst(input, timesteps=timestep)

        # analytic OR of independent streams
        r_value = 1 - torch.prod(1 - input, 0)
        inst.accuracy.analyze(r_value, verbose=True)
        err = torch.sqrt((inst.decoder.spike_value - r_value).pow(2).mean()).item()
        assert err < bound, f'{device}/or: rmse {err} >= {bound}'
        inst.reset()

        # known-answer OR check
        gate = add_gaines(dict(add_config)).to(device)
        a = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=global_config.stype, device=device)
        assert torch.equal(gate(a, dim=0).cpu(), torch.tensor([0, 1, 1, 1], dtype=global_config.stype))
        gate.reset()

        print(f'{device}/unipolar/or: rmse {err:.4f} (bound {bound:.4f})')

    # bipolar non-scaled is rejected
    try:
        add_gaines({'polarity': 'bipolar', 'scaled': False})
        assert False, 'bipolar non-scaled should be rejected'
    except AssertionError as e:
        if 'rejected' in str(e):
            raise

    print('Test passed.')


def test_add_gaines_perf():
    """Per device, time add_gaines (scaled MUX) against add_any (the obvious baseline) on identical spikes."""
    iters = 200
    entry = 8
    spikes_cpu = (torch.rand(entry, 100000) > 0.5).type(global_config.stype)
    for device in devices():
        spikes = spikes_cpu.to(device)
        results = {}

        op = add_any({'polarity': 'unipolar', 'scale': entry, 'width': 10}).to(device)
        op(spikes, dim=0)  # warmup
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            op(spikes, dim=0)
        sync(device)
        results['add_any'] = time.perf_counter() - t0

        op = add_gaines({'polarity': 'unipolar', 'scaled': True, 'entry': entry,
                         'generator': 'sobol', 'dim': 5}).to(device)
        op(spikes, dim=0)  # warmup
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            op(spikes, dim=0)
        sync(device)
        results['add_gaines'] = time.perf_counter() - t0

        ratio = results['add_any'] / results['add_gaines']
        print(f'{device}: add_gaines {results["add_gaines"]:.4f}s vs add_any {results["add_any"]:.4f}s, speedup x{ratio:.2f}')

    print('Perf test passed.')


if __name__ == '__main__':
    test_add_gaines()
    test_add_gaines_perf()

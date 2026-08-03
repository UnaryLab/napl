import torch
import math
import time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation.sync_skewed_int import sync_skewed_int
from napl.sim.metric import accuracy


class napl_sync_skewed_int(napl_base):
    def __init__(self, codec_config1, codec_config2, sync_skewed_int_config):
        super().__init__()
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder0 = decoder(codec_config1)
        self.decoder1 = decoder(codec_config1)
        self.sync_skewed_int = sync_skewed_int(sync_skewed_int_config)
        self.accuracy0 = accuracy(codec_config1)
        self.accuracy1 = accuracy(codec_config1)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.sync_skewed_int(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)
        self.accuracy0(o_spike0)
        self.accuracy1(o_spike1)


def _kernel_specific_checks():
    """
    Test sync_skewed_int on every available device: streaming accuracy with input_0 <= input_1
    (spike conservation bounds the error by cnt residual), pass-through exactness of output 1,
    a known-answer check of the integer aggregation path, and a timing report.
    """
    codec_config1={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 1,
    }
    codec_config2={
        'polarity': 'unipolar',
        'timestep': 256,
        'generator': 'sobol',
        'dim': 3,
    }
    sync_skewed_int_config={
        'width' : 4,
    }
    timestep = codec_config1['timestep']

    # Keep input_0 <= input_1 so all input_0 spikes are released within the residual bound.
    input_0 = gen_rand_tensor(codec_config1['polarity'], shape=(10000,), width=math.log2(timestep)).type(global_config.ntype)
    input_1 = gen_rand_tensor(codec_config2['polarity'], shape=(10000,), width=math.log2(timestep)).type(global_config.ntype)
    input_mask = input_0 < input_1
    input_0_new = torch.where(input_mask, input_0, input_1)
    input_1_new = torch.where(~input_mask, input_0, input_1)
    input_1_new = torch.where(input_1_new==0, 1, input_1_new)
    input_0 = input_0_new
    input_1 = input_1_new

    cnt_max = 2**sync_skewed_int_config['width'] - 1
    for device in devices():
        in_0 = input_0.to(device)
        in_1 = input_1.to(device)

        inst = napl_sync_skewed_int(codec_config1, codec_config2, sync_skewed_int_config).to(device)
        sync(device)
        start = time.perf_counter()
        inst(in_0, in_1, timesteps=timestep)
        sync(device)
        elapsed = time.perf_counter() - start
        print(f'[{device}] {timestep} timesteps x {in_0.numel()} elems: {elapsed:.3f} s')

        # Output 1 conserves input 0 up to the counter residual.
        inst.accuracy0.analyze(in_0, verbose=True)
        err0 = (inst.decoder0.spike_value - in_0).abs().max().item()
        assert err0 <= (cnt_max + 1) / timestep, f'[{device}] output_1 error {err0} exceeds bound'
        # Output 2 passes input 2 through bit-exactly.
        error1, _ = inst.accuracy1.analyze(in_1, verbose=True)
        err1 = error1.abs().max().item()
        assert err1 == 0, f'[{device}] output_2 not a pass-through (err {err1})'

        assert inst.sync_skewed_int.timestep_cur == timestep
        inst.reset()
        assert inst.sync_skewed_int.timestep_cur == 0

        # Alternating input 2 spikes make output 1 release integer digits of 2.
        op = sync_skewed_int(sync_skewed_int_config).to(device)
        one = torch.ones(4, dtype=global_config.stype, device=device)
        zero = torch.zeros(4, dtype=global_config.stype, device=device)
        for t in range(8):
            in_2 = one if t % 2 else zero
            out_1, out_2 = op(one, in_2)
            expected = one * 2 if t % 2 else zero
            assert torch.equal(out_1, expected), f'[{device}] t={t}: {out_1} != {expected}'
            assert torch.equal(out_2, in_2)

    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return sync_skewed_int({'width': 4})


def make_values(_polarity):
    first = torch.linspace(0.0, 0.75, 128)
    second = torch.linspace(0.25, 1.0, 128)
    return first, second


def analytic_reference(values, _polarity):
    return values[1]


def known_answer_case(_polarity):
    values = (torch.tensor([0.0, 0.5]), torch.tensor([0.5, 1.0]))
    return values, values[1], 0.0


CONFIG = {
    'polarities': ['unipolar'],
    'tolerance_scale': 2.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [1, 3],
    'apply_operation': lambda operation, spikes: operation(*spikes)[1],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_sync_skewed_int():
    """Verify sync_skewed_int against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)
    try:
        sync_skewed_int({'width': 8})
    except AssertionError:
        return
    raise AssertionError('sync_skewed_int must reject width 8 for int8 spike digits')


if __name__ == '__main__':
    test_sync_skewed_int()

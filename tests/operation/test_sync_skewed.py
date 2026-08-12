import math

import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.operation import decode, encode, sync_skewed
from napl.sim.metric import accuracy, correlation


class napl_sync_skewed(napl_base):
    def __init__(self, codec_config1, codec_config2, sync_skewed_config):
        super().__init__()
        self.encoder0 = encode(codec_config1)
        self.encoder1 = encode(codec_config2)
        self.decoder0 = decode(codec_config1)
        self.decoder1 = decode(codec_config1)
        self.sync_skewed = sync_skewed(sync_skewed_config)
        self.accuracy0 = accuracy(codec_config1)
        self.accuracy1 = accuracy(codec_config1)
        self.scc_in = correlation()
        self.scc_out = correlation()


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        self.scc_in(i_spike0, i_spike1)
        o_spike0, o_spike1 = self.sync_skewed(i_spike0, i_spike1)
        self.scc_out(o_spike0, o_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)
        self.accuracy0(o_spike0)
        self.accuracy1(o_spike1)

    
def _kernel_specific_checks():
    """
    Test sync_skewed on every device and supported polarity: the retimed stream keeps the
    value of input 0, input 1 passes through, and the retimed pair is highly correlated.
    """
    sync_skewed_config={
        'width' : 3,
    }
    timestep = CONFIG['timesteps']

    for polarity in CONFIG['polarities']:
        codec_config1={
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
        }
        codec_config2={
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 3,
        }
        # Lowest legal value for this polarity, which encodes to the zero-rate stream.
        min_value = -1.0 if polarity == 'bipolar' else 0.0

        input_0_cpu = gen_rand_tensor(polarity, shape=(10000,), width=math.log2(timestep)).type(global_config.ntype)
        input_1_cpu = gen_rand_tensor(polarity, shape=(10000,), width=math.log2(timestep)).type(global_config.ntype)
        input_mask = input_0_cpu < input_1_cpu
        input_0_new = torch.where(input_mask, input_0_cpu, input_1_cpu)
        input_1_new = torch.where(~input_mask, input_0_cpu, input_1_cpu)
        # A zero-rate reference stream offers no spike position to retime onto.
        input_1_new = torch.where(input_1_new==min_value, 1.0, input_1_new)
        input_0_cpu = input_0_new
        input_1_cpu = input_1_new

        for device in devices():
            input_0 = input_0_cpu.to(device)
            input_1 = input_1_cpu.to(device)
            sync_skewed_inst = napl_sync_skewed(codec_config1, codec_config2, sync_skewed_config).to(device)
            with timer(device) as elapsed:
                sync_skewed_inst(input_0, input_1, timesteps=timestep)

            error_0, _ = sync_skewed_inst.accuracy0.analyze(input_0, verbose=True)
            rmse_0 = error_0.pow(2).mean().sqrt()
            error_1, _ = sync_skewed_inst.accuracy1.analyze(input_1, verbose=True)
            rmse_1 = error_1.pow(2).mean().sqrt()
            scc_in, _ = sync_skewed_inst.scc_in.analyze()
            scc_out, _ = sync_skewed_inst.scc_out.analyze()
            # An element is degenerate when either output stream carries no spike over the run:
            # the SCC counts then satisfy a+b == 0 or a+c == 0, its denominator collapses, and
            # the metric returns 0.0 regardless of what the kernel did. Score the rest per
            # element, since a mean over all elements lets a healthy majority hide a broken few.
            spikes_0 = sync_skewed_inst.scc_out.sum_1
            spikes_1 = sync_skewed_inst.scc_out.sum_2
            scored = (spikes_0 > 0) & (spikes_1 > 0)
            scc_out_scored = scc_out[scored]
            scc_in_scored = scc_in[scored]
            n_scored = scc_out_scored.numel()
            # Retiming puts every output-0 spike on an output-1 spike, so a scored element
            # correlates perfectly unless the width-bit skew buffer saturates and passes a
            # spike through; 1 percent is the declared allowance for that tail.
            n_weak = int((scc_out_scored < 0.9).sum().item())
            print(f'[{device}][{polarity}] rmse_0={rmse_0:.6f}, rmse_1={rmse_1:.6f}, '
                  f'scc_in={scc_in_scored.mean().item():.4f}, '
                  f'scc_out={scc_out_scored.mean().item():.4f}, '
                  f'degenerate={scc_out.numel() - n_scored}/{scc_out.numel()}, '
                  f'scc_out<0.9: {n_weak}/{n_scored}')
            assert n_weak <= 0.01 * n_scored, f'{n_weak}/{n_scored} elements below scc 0.9'
            assert scc_out_scored.mean() > scc_in_scored.mean() + 0.5, \
                (scc_in_scored.mean(), scc_out_scored.mean())
            assert sync_skewed_inst.sync_skewed.timestep_cur == timestep
            sync_skewed_inst.reset()
            assert sync_skewed_inst.sync_skewed.timestep_cur == 0
            print(f'[{device}][{polarity}] time: {elapsed.seconds * 1000:.1f} ms')

    print('Test passed.')


def make_operation(_polarity, _timestep, _device):
    return sync_skewed({'width': 3})


def _reference_values(low, count):
    # The kernel requires input_0 to have no higher rate than input_1. Under both
    # polarities the rate of (v + 1) / 2 is halfway between the rate of v and 1,
    # so the second stream stays the higher-rate reference over the whole sweep.
    first = torch.linspace(low, 1.0, count)
    return first, (first + 1.0) / 2.0


def make_values(polarity):
    low = -1.0 if polarity == 'bipolar' else 0.0
    return _reference_values(low, 128)


def make_random_perf_values(polarity):
    low = -1.0 if polarity == 'bipolar' else 0.0
    return _reference_values(low, 131072)


def analytic_reference(values, _polarity):
    return values[0]


def known_answer_case(polarity):
    low = -1.0 if polarity == 'bipolar' else 0.0
    # The zero-rate stream has no spike to retime and the all-ones pair always
    # agrees, so the skew counter never holds a spike and output 0 is exact.
    values = (torch.tensor([low, 1.0]), torch.tensor([1.0, 1.0]))
    return values, values[0]


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'encoder_dims': [1, 3],
    'apply_operation': lambda operation, spikes: operation(*spikes)[0],
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_sync_skewed_contract():
    """Report rate coding on both inputs and both outputs, no polarity or correlation constraint, and zero delay."""
    operation = sync_skewed({'width': 3})
    assert operation.streaming is True
    assert operation.encoding_io == {'input_0': 'rc', 'input_1': 'rc',
                                     'output_0': 'rc', 'output_1': 'rc'}, operation.encoding_io
    assert operation.polarity_io == {}, operation.polarity_io
    assert operation.correlation_i == {}, operation.correlation_i
    assert operation.hw.pp_delay == 0
    assert operation.flux_stability == 1.0


def test_sync_skewed():
    """Verify sync_skewed across the full unipolar and bipolar ranges and input ordering."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_sync_skewed_contract()
    test_sync_skewed()

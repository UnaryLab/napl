import torch, math, time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
# import directly from the module: not yet wired into operation/__init__.py
from napl.sim.operation.sync_skewed_int import sync_skewed_int
from napl.sim.metric import accuracy


class napl_sync_skewed_int(napl_base):
    def __init__(self, codec_config1, codec_config2, sync_skewed_int_config):
        super().__init__()
        # set up encoder, decoder, and accuracy
        self.encoder0 = encoder(codec_config1)
        self.encoder1 = encoder(codec_config2)
        self.decoder0 = decoder(codec_config1)
        self.decoder1 = decoder(codec_config1)
        self.sync_skewed_int = sync_skewed_int(sync_skewed_int_config)
        self.accuracy0 = accuracy(codec_config1)
        self.accuracy1 = accuracy(codec_config1)


    @napl_sim_timesteps
    def forward(self, input_0, input_1, timesteps=256):
        # forward is a description of the circuit
        i_spike0 = self.encoder0(input_0)
        i_spike1 = self.encoder1(input_1)
        o_spike0, o_spike1 = self.sync_skewed_int(i_spike0, i_spike1)
        self.decoder0(o_spike0)
        self.decoder1(o_spike1)
        self.accuracy0(o_spike0)
        self.accuracy1(o_spike1)


def test_sync_skewed_int():
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

    # Generate random inputs based on polarity; keep input_0 <= input_1 so all input_0 spikes
    # are released (residual cnt <= cnt_max bounds the error)
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

        # numerical check: output 1 conserves input 0's spikes up to the counter residual
        inst.accuracy0.analyze(in_0, verbose=True)
        err0 = (inst.decoder0.spike_value - in_0).abs().max().item()
        assert err0 <= (cnt_max + 1) / timestep, f'[{device}] output_1 error {err0} exceeds bound'
        # output 2 is a bit-exact pass-through of input 2
        error1, _ = inst.accuracy1.analyze(in_1, verbose=True)
        err1 = error1.abs().max().item()
        assert err1 == 0, f'[{device}] output_2 not a pass-through (err {err1})'

        assert inst.sync_skewed_int.timestep_cur == timestep
        inst.reset()
        assert inst.sync_skewed_int.timestep_cur == 0

        # known-answer aggregation: input 1 spikes every step, input 2 every other step,
        # so output 1 must release integer digits of 2 on input 2's spikes
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


if __name__ == '__main__':
    test_sync_skewed_int()

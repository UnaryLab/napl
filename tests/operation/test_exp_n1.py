import torch, math, time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
from napl.sim.operation.exp_n1 import exp_n1
from napl.sim.metric import accuracy


class napl_exp_n1(napl_base):
    def __init__(self, codec_config, exp_n1_config):
        super().__init__()
        # set up encoder, decoder, op, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})
        self.exp_n1 = exp_n1(exp_n1_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
        i_spike = self.encoder(input)
        o_spike = self.exp_n1(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)

def test_exp_n1():
    """
    Test exp_n1 (exp(-x), unipolar) on every available device: correctness
    against torch.exp(-x) within the SC bound, plus per-device runtime.
    """
    timestep = 256

    codec_config={
        'polarity': 'unipolar',
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 5,   # distinct from the op's internal dims 1..4
    }
    exp_n1_config={
        'polarity': 'unipolar',
        'timestep': timestep,
        'generator': 'sobol',
        'dim': 1,
    }

    # identical inputs on every device: generate once on CPU
    input_cpu = gen_rand_tensor('unipolar', shape=(10000,), width=math.log2(timestep)).type(global_config.ntype)
    # known-answer corners: exp(0)=1, exp(-1)=0.3679
    input_cpu[0] = 0.0
    input_cpu[1] = 1.0

    for device in devices():
        input = input_cpu.to(device)

        exp_n1_inst = napl_exp_n1(codec_config, exp_n1_config).to(device)
        sync(device)
        start = time.perf_counter()
        exp_n1_inst(input, timesteps=timestep)
        sync(device)
        elapsed = time.perf_counter() - start

        # calculate the reference output
        r_value = torch.exp(-input)

        # report the error
        exp_n1_inst.accuracy.analyze(r_value, verbose=True)

        out = exp_n1_inst.decoder.spike_value.cpu()
        ref = r_value.cpu()
        rmse = (out - ref).pow(2).mean().sqrt().item()
        bound = 1.5 / math.sqrt(timestep)   # SC bound; series truncation < 0.002
        assert rmse < bound, f'[{device}] RMSE {rmse:.4f} exceeds bound {bound:.4f}'
        # known-answer corners within the SC bound
        assert abs(out[0].item() - 1.0) < bound
        assert abs(out[1].item() - math.exp(-1)) < bound

        assert exp_n1_inst.exp_n1.timestep_cur == timestep
        exp_n1_inst.reset()
        assert exp_n1_inst.exp_n1.timestep_cur == 0

        # performance: streaming kernel vs the single-shot float reference
        sync(device)
        start_ref = time.perf_counter()
        torch.exp(-input)
        sync(device)
        elapsed_ref = time.perf_counter() - start_ref
        print(f'[{device}] rmse={rmse:.4f} (bound {bound:.4f}), '
              f'kernel {elapsed*1e3:.1f} ms for {timestep} timesteps, '
              f'torch.exp {elapsed_ref*1e3:.3f} ms (ratio {elapsed/max(elapsed_ref, 1e-9):.0f}x)')

    print('Test passed.')


if __name__ == '__main__':
    test_exp_n1()

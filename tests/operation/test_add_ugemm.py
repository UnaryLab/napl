import torch, math, time

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, sync
from napl.sim.module import encoder, decoder
# import directly from the module: not wired into operation/__init__.py yet
from napl.sim.operation.add_ugemm import add_ugemm
from napl.sim.metric import accuracy


class napl_add_ugemm(napl_base):
    def __init__(self, codec_config, add_ugemm_config):
        super().__init__()
        # set up encoder, decoder, adder, and accuracy
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.accuracy = accuracy({'polarity': codec_config['polarity']})
        self.add_ugemm = add_ugemm(add_ugemm_config)


    @napl_sim_timesteps
    def forward(self, input, timesteps=256):
        # forward is a description of the circuit
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

    sync(device)
    start = time.perf_counter()
    inst(input, timesteps=timestep)
    sync(device)
    elapsed = time.perf_counter() - start

    entry = input.size(-1)
    if scaled:
        # scaled mode averages: output ~= sum/entry
        r_value = torch.sum(input, dim=-1) / entry
    else:
        # non-scaled mode: output ~= sum, clipped to the polarity range
        lo = -1.0 if polarity == 'bipolar' else 0.0
        r_value = torch.sum(input, dim=-1).clamp(lo, 1.0)

    error, _ = inst.accuracy.analyze(r_value, verbose=True)
    mae = error.abs().mean().item()
    bound = 2 / math.sqrt(timestep)
    assert mae < bound, f'{device} {polarity} scaled={scaled}: MAE {mae} exceeds SC bound {bound}'

    assert inst.add_ugemm.timestep_cur == timestep
    inst.reset()
    assert inst.add_ugemm.timestep_cur == 0

    print(f'[{device}] polarity={polarity} scaled={scaled}: MAE={mae:.5f} time={elapsed:.3f}s')


def test_add_ugemm():
    """
    Test add_ugemm (scaled and non-scaled, unipolar and bipolar) on every device.
    """
    timestep = 256
    entry = 8
    width = math.log2(timestep)

    # identical inputs across devices and variants
    input_scaled_uni = gen_rand_tensor('unipolar', shape=(2000, entry), width=width).type(global_config.ntype)
    input_scaled_bi = gen_rand_tensor('bipolar', shape=(2000, entry), width=width).type(global_config.ntype)
    # keep the sum inside the output range for non-scaled mode
    input_ns_uni = input_scaled_uni / entry
    input_ns_bi = input_scaled_bi / entry

    for device in devices():
        run_case('unipolar', True, input_scaled_uni, device, timestep)
        run_case('bipolar', True, input_scaled_bi, device, timestep)
        run_case('unipolar', False, input_ns_uni, device, timestep)
        run_case('bipolar', False, input_ns_bi, device, timestep)

    print('Test passed.')


if __name__ == '__main__':
    test_add_ugemm()

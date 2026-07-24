import sys, time, math, importlib.util
import torch
sys.path.insert(0, '/Users/diwu/Projects')

from UnarySim.kernel.add import FSUAdd
from napl.sim.base import global_config
from napl.sim.module import encoder, decoder
from napl.sim.operation import add_any
from napl.utils import gen_rand_tensor

# load the committed test wiring
spec = importlib.util.spec_from_file_location('t', '/Users/diwu/Projects/napl/tests/operation/test_add_any.py')
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
NaplWiring = t.napl_add_any


def run(device, polarity, scale, width, entry, T, n=2000, seed=0):
    torch.manual_seed(seed)
    codec_config = {'polarity': polarity, 'timestep': T, 'generator': 'sobol'}
    add_cfg = {'polarity': polarity, 'scale': scale, 'width': width}

    # input value generated ONCE, fed to both sides
    inp = gen_rand_tensor(polarity, shape=(n, entry), width=math.log2(T)).type(global_config.ntype).to(device)

    # napl side via committed wiring
    nap = NaplWiring(codec_config, add_cfg).to(device)
    nap(inp, timesteps=T)
    nap_out = nap.decoder.spike_value  # decoded value

    # UnarySim side: feed the SAME spike stream (mirror napl encoder)
    enc = encoder(codec_config).to(device)
    dec = decoder(codec_config).to(device)
    hwcfg = {'mode': polarity, 'scale': scale, 'dima': -1, 'depth': width, 'entry': entry}
    swcfg = {'btype': torch.float, 'stype': torch.float}
    uadd = FSUAdd(hwcfg, swcfg).to(device)
    enc.reset(); dec.reset()
    for _ in range(T):
        s = enc(inp)
        o = uadd(s.type(torch.float))
        dec(o.type(global_config.stype))
    ref_out = dec.spike_value

    bitexact = torch.equal(nap_out, ref_out)
    maxdiff = (nap_out - ref_out).abs().max().item()

    # analytic reference
    r_value = torch.sum(inp, dim=-1) / scale
    rmse_nap = torch.sqrt(torch.mean((nap_out - r_value) ** 2)).item()
    rmse_ref = torch.sqrt(torch.mean((ref_out - r_value) ** 2)).item()
    sc_bound = 1.0 / math.sqrt(T)
    return bitexact, maxdiff, rmse_nap, rmse_ref, sc_bound


def time_napl(device, polarity, scale, width, entry, T, n=2000, seed=0, iters=3):
    torch.manual_seed(seed)
    codec_config = {'polarity': polarity, 'timestep': T, 'generator': 'sobol'}
    add_cfg = {'polarity': polarity, 'scale': scale, 'width': width}
    inp = gen_rand_tensor(polarity, shape=(n, entry), width=math.log2(T)).type(global_config.ntype).to(device)
    nap = NaplWiring(codec_config, add_cfg).to(device)
    # warmup
    nap(inp, timesteps=T); nap.reset()
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        nap(inp, timesteps=T); nap.reset()
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms


if __name__ == '__main__':
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else []) + (['mps'] if torch.backends.mps.is_available() else [])
    T = 256
    scale, width, entry = 128, 20, 128
    print('devices:', devices)
    for device in devices:
        for polarity in ['bipolar', 'unipolar']:
            be, md, rn, rr, sc = run(device, polarity, scale, width, entry, T)
            print(f'[{device}] {polarity}: bitexact={be} maxdiff={md:.3e} '
                  f'rmse_nap={rn:.4e} rmse_ref={rr:.4e} sc_bound={sc:.4e}')
    # timing on the test's default config (scale=128 entry=128 width=20)
    for device in devices:
        ms = time_napl(device, 'bipolar', scale, width, entry, T, n=2000)
        print(f'TIMING [{device}] bipolar n=2000 entry=128 T=256: {ms:.2f} ms')

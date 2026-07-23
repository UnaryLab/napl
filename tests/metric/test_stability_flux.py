import time

import torch

from napl.metric import stability, stability_flux
from napl.module import encoder
from napl.utils import devices, gen_rand_tensor, sync


def run_flux(val_1, val_2, device, timestep):
    stability_config = {'polarity': 'bipolar', 'threshold': 0.05}
    val_1 = val_1.to(device)
    val_2 = val_2.to(device)
    enc_1 = encoder(
        {
            'polarity': 'bipolar',
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 1,
        }
    ).to(device)
    enc_2 = encoder(
        {
            'polarity': 'bipolar',
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 2,
        }
    ).to(device)
    flux = stability_flux(val_1, val_2, stability_config).to(device)
    stab_1 = stability(val_1, stability_config).to(device)
    stab_2 = stability(val_2, stability_config).to(device)

    sync(device)
    start = time.time()
    for _ in range(timestep):
        spike_1 = enc_1(val_1)
        spike_2 = enc_2(val_2)
        flux(spike_1, spike_2)
        stab_1(spike_1)
        stab_2(spike_2)
    result = flux.analyze()[0].detach().cpu().clone()
    expected = stab_1.analyze()[0].div(stab_2.analyze()[0])
    sync(device)
    elapsed = time.time() - start
    return result, expected.detach().cpu(), elapsed, (enc_1, enc_2, flux)


def test_fidelity():
    timestep = 256
    val_1 = gen_rand_tensor('bipolar', shape=(1000,), width=8)
    val_2 = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        result, expected, _, _ = run_flux(val_1, val_2, device, timestep)
        assert torch.equal(result, expected), 'flux != stability_1 / stability_2'
        print(
            f'[{device}] flux mean={result.mean().item():.4f}, '
            f'min={result.min().item():.4f}, max={result.max().item():.4f}'
        )


def test_known_answer():
    timestep = 256
    val = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        ones = torch.ones(4, device=device)
        flux = stability_flux(
            ones, ones, {'polarity': 'bipolar', 'threshold': 0.05}
        ).to(device)
        for _ in range(timestep):
            spike = torch.ones(4, device=device)
            flux(spike, spike)
        assert torch.equal(flux.analyze()[0], torch.ones(4, device=device))

        device_val = val.to(device)
        enc = encoder(
            {
                'polarity': 'bipolar',
                'timestep': timestep,
                'generator': 'sobol',
                'dim': 1,
            }
        ).to(device)
        ones = torch.ones(1000, device=device)
        flux = stability_flux(
            ones, device_val, {'polarity': 'bipolar', 'threshold': 0.05}
        ).to(device)
        for _ in range(timestep):
            flux(ones, enc(device_val))
        result = flux.analyze()[0]
        assert (result >= 1.0).all(), result.min()
        assert result.mean() > 1.0, result.mean()


def test_reset():
    timestep = 256
    val_1 = gen_rand_tensor('bipolar', shape=(1000,), width=8)
    val_2 = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        _, _, _, modules = run_flux(val_1, val_2, device, timestep)
        flux = modules[-1]
        flux.reset()
        assert flux.timestep_cur == 0 and flux.flux.abs().sum() == 0
        assert (
            flux.stability_1.timestep_cur == 0
            and flux.stability_2.timestep_cur == 0
        )


def test_performance():
    timestep = 256
    val_1 = gen_rand_tensor('bipolar', shape=(1000,), width=8)
    val_2 = gen_rand_tensor('bipolar', shape=(1000,), width=8)

    for device in devices():
        result, _, elapsed, _ = run_flux(val_1, val_2, device, timestep)
        print(
            f'[{device}] flux mean={result.mean().item():.4f}, '
            f'time={elapsed * 1000:.1f}ms'
        )


if __name__ == '__main__':
    test_fidelity()
    test_known_answer()
    test_reset()
    test_performance()
    print('Test passed.')

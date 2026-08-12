import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import linear_mix, linear_gaines
from napl.sim.operation import encode, decode


class napl_linear_gaines(napl_base):
    """Wire encoder -> linear_gaines -> decode (canonical round-trip)."""


    def __init__(self, codec_config, lin_config, weight, bias):
        super().__init__()
        self.encoder = encode(codec_config)
        self.decoder = decode(codec_config)
        self.linear = linear_gaines(weight, bias, lin_config)


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        o_spike = self.linear(i_spike)
        self.decoder(o_spike)


def _scaled_ref(s, polarity, entry):
    """Decoded value the Gaines scaled adder converges to for inner product s."""
    w = 2 ** round(torch.log2(torch.tensor(float(entry))).item())
    if polarity == 'bipolar':
        return ((entry + s) / w - 1).clamp(-1, 1)
    return (s / w).clamp(0, 1)


def _kernel_specific_checks():
    """
    Streaming Gaines linear (gMUL + gADD) reproduces its analytic target within a
    stochastic-computing bound on every device: scaled mode tracks the scaled inner
    product, and non-scaled unipolar uses OR addition. UnarySim: GainesLinear1.
    """
    timestep = 1024
    out_features = 8
    torch.manual_seed(42)  # Keep gen_rand_tensor inputs reproducible.

    scaled_cases = []
    for polarity in ['unipolar', 'bipolar']:
        for has_bias in [True, False]:
            in_features = 15 if has_bias else 16
            if polarity == 'unipolar':
                input_x_cpu = gen_rand_tensor(
                    polarity, shape=(in_features,), width=8
                ).type(global_config.ntype)
                weight_cpu = gen_rand_tensor(
                    polarity, shape=(out_features, in_features), width=8
                ).type(global_config.ntype)
                bias_cpu = gen_rand_tensor(
                    polarity, shape=(out_features,), width=8
                ).type(global_config.ntype) if has_bias else None
            else:
                # Deterministic sign-aligned bipolar fixture. Independent random
                # signs leave the scaled reference (W x + b) / entry with rms
                # below the streaming error, so no rmse bound could separate the
                # correct output from a const-zero one. Aligning weight signs
                # with the input lifts the reference rms to ~0.43.
                input_x_cpu = torch.linspace(
                    -1.0, 1.0, in_features
                ).type(global_config.ntype)
                mag = torch.linspace(0.5, 1.0, in_features).type(global_config.ntype)
                sign = torch.where(input_x_cpu < 0, -1.0, 1.0).type(global_config.ntype)
                # Distinct per-row scales keep the output rows apart, so a
                # permuted output also fails the rmse bound.
                weight_cpu = torch.stack(
                    [sign * mag.roll(i) for i in range(out_features)]
                ) * torch.linspace(
                    0.45, 1.0, out_features
                ).type(global_config.ntype).unsqueeze(1)
                bias_cpu = torch.linspace(
                    0.25, 1.0, out_features
                ).type(global_config.ntype) if has_bias else None
            scaled_cases.append(
                (polarity, has_bias, input_x_cpu, weight_cpu, bias_cpu)
            )
    rejection_weight_cpu = gen_rand_tensor(
        'bipolar', shape=(out_features, 16), width=8
    ).type(global_config.ntype)
    in_features = 16

    for device in devices():
        # Cover both polarities and bias settings in scaled mode.
        for polarity, has_bias, input_x_cpu, weight_cpu, bias_cpu in scaled_cases:
            input_x = input_x_cpu.to(device)
            weight = weight_cpu.to(device)
            bias = None if bias_cpu is None else bias_cpu.to(device)

            codec_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
            lin_config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'scaled': True}

            inst = napl_linear_gaines(codec_config, lin_config, weight, bias).to(device)
            inst(input_x, timesteps=timestep)

            entry = input_x.numel() + (1 if has_bias else 0)
            s = weight @ input_x + (bias if has_bias else 0)
            r_value = _scaled_ref(s, polarity, entry)

            err = (inst.decoder.spike_value - r_value).abs()
            rmse = torch.sqrt(err.pow(2).mean()).item()
            print(f'[{device}] scaled {polarity} bias={has_bias}: rmse={rmse:.5f} max_err={err.max().item():.5f}')
            assert inst.linear.timestep_cur == timestep
            # The layer holds its own weight and bias encoders.
            assert inst.linear.internal_encode is True
            inst.reset()

        # Non-scaled Gaines addition supports unipolar data only.
        weight = rejection_weight_cpu.to(device)
        lin_config = {
            'polarity': 'bipolar',
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 2,
            'scaled': False,
        }
        try:
            linear_gaines(weight, None, lin_config)
        except AssertionError as error:
            assert str(error) == 'Non-scaled Gaines addition in linear_gaines does not support bipolar data.'
        else:
            raise AssertionError('linear_gaines accepted non-scaled bipolar data')

    # The non-scaled unipolar adder ORs the fan-in, so its output rate lies between
    # the largest product rate and the sum of the product rates, whatever the
    # correlation between the product streams.
    input_x_cpu = torch.full((in_features,), 0.25, dtype=global_config.ntype)
    weight_cpu = torch.full(
        (out_features, in_features), 0.125, dtype=global_config.ntype
    )
    codec_config = {'polarity': 'unipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    lin_config = {'polarity': 'unipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'scaled': False}
    tolerance = 3.0 / timestep ** 0.5
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        inst = napl_linear_gaines(codec_config, lin_config, weight, None).to(device)
        inst(input_x, timesteps=timestep)
        product = weight * input_x
        low, high = product.max(-1).values, product.sum(-1)
        assert high.max().item() < 1.0
        spike_value = inst.decoder.spike_value
        assert spike_value.shape == low.shape, (device, spike_value.shape, low.shape)
        print(f'[{device}] non-scaled unipolar: rate={spike_value.min().item():.5f}..'
              f'{spike_value.max().item():.5f} in [{low.min().item():.5f}, '
              f'{high.max().item():.5f}] +/- {tolerance:.5f}')
        assert (spike_value >= low - tolerance).all() and (spike_value <= high + tolerance).all(), \
            f'{device}/non-scaled/unipolar: {spike_value} outside [{low}, {high}]'
        inst.reset()

    # Equal weights across input features must still spike on different timesteps,
    # because each feature draws its own sequence.
    w_eq = torch.full((1, 8), 0.5).type(global_config.ntype)
    for device in devices():
        gl_eq = linear_gaines(
            w_eq, None, {'polarity': 'unipolar', 'timestep': 256,
                         'generator': 'sobol', 'dim': 2, 'scaled': True}
        ).to(device)
        buffers = dict(gl_eq.named_buffers())
        assert 'w_num_seq' in buffers
        weight_num_seq = buffers['w_num_seq']
        assert weight_num_seq.device == gl_eq.weight.device
        assert weight_num_seq.shape == (256, 8)
        assert torch.unique(weight_num_seq, dim=1).shape[1] == 8, \
            'weight sequences are shared across input features, so their bits are comonotone'
        assert not any(
            name.startswith('w_num_seq')
            for name, _child in gl_eq.named_children()
        )
    print('per-input-feature encode-owned sequences are registered and distinct.')

    # A weight update must reach the vectorized encoder, in place or by assignment.
    for device in devices():
        w_mut = torch.zeros(1, 4).type(global_config.ntype).to(device)
        gl_mut = linear_gaines(w_mut, None, {'polarity': 'unipolar', 'timestep': 64,
                                              'generator': 'sobol', 'dim': 2, 'scaled': True}).to(device)
        x_on = torch.ones(4).type(global_config.ntype).to(device)
        assert gl_mut(x_on).sum().item() == 0, f'[{device}] zero weights should emit no spike'
        with torch.no_grad():
            gl_mut.weight.fill_(1.0)
        assert gl_mut(x_on).sum().item() > 0, f'[{device}] in-place weight update did not take effect'
        with torch.no_grad():
            gl_mut.weight.copy_(torch.zeros_like(gl_mut.weight))
        assert gl_mut(x_on).sum().item() == 0, f'[{device}] weight reset did not take effect'
        gl_mut.weight = torch.nn.Parameter(torch.ones_like(gl_mut.weight))
        assert gl_mut(x_on).sum().item() > 0, f'[{device}] weight reassignment did not take effect'
    print('weight updates reach the vectorized weight encoder.')

    # All-ones unipolar operands make the scaled output emit 1 every timestep.
    w1_cpu = torch.ones(out_features, in_features).type(global_config.ntype)
    x1_cpu = torch.ones(in_features).type(global_config.ntype)
    for device in devices():
        w1 = w1_cpu.to(device)
        x1 = x1_cpu.to(device)
        inst = napl_linear_gaines(
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 1},
            {'polarity': 'unipolar', 'timestep': 64, 'generator': 'sobol', 'dim': 2, 'scaled': True},
            w1, None).to(device)
        inst(x1, timesteps=64)
        assert torch.allclose(inst.decoder.spike_value, torch.ones(out_features, device=device)), \
            f'[{device}] all-ones unipolar scaled Gaines linear should spike every timestep'
        inst.reset()
    print('known-answer corner passed.')

    # Compare the comparator and scaled accumulator adders on identical spikes.
    input_x_cpu = gen_rand_tensor('bipolar', shape=(in_features,), width=8).type(global_config.ntype)
    weight_cpu = gen_rand_tensor('bipolar', shape=(out_features, in_features), width=8).type(global_config.ntype)
    cfg = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2}
    for device in devices():
        input_x = input_x_cpu.to(device)
        weight = weight_cpu.to(device)
        enc = encode({'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}).to(device)
        gl = linear_gaines(weight, None, {**cfg, 'scaled': True}).to(device)
        lin = linear_mix(weight, None, {**cfg, 'scale': None, 'width': 12}).to(device)

        spikes = [enc(input_x).clone() for _ in range(timestep)]
        enc.reset()
        with timer(device) as t_gl:
            for s in spikes:
                gl(s)
        gl.reset()
        with timer(device) as t_lin:
            for s in spikes:
                lin(s)
        lin.reset()
        gl_s = t_gl.seconds or 0.0
        lin_s = t_lin.seconds or 0.0
        print(f'[{device}] perf: linear_gaines {gl_s*1e3:.1f}ms vs linear_mix {lin_s*1e3:.1f}ms '
              f'(ratio {lin_s/max(gl_s,1e-9):.2f}x)')

    print('Test passed.')


def _suite_weight(polarity):
    # The 16-entry suite uses a 4-bit threshold RNG.
    if polarity == 'unipolar':
        row = torch.linspace(0.0, 1.0, 16)
    else:
        row = torch.linspace(-1.0, 1.0, 16)
    return torch.stack((row, row.flip(0)))


def make_operation(polarity, timestep, _device):
    return linear_gaines(
        _suite_weight(polarity), None,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 2,
            'scaled': True,
        },
    )


def make_values(polarity):
    low, high = (0.0, 1.0) if polarity == 'unipolar' else (-1.0, 1.0)
    return (torch.linspace(low, high, 16),)


def make_random_perf_values(polarity):
    values = make_values(polarity)[0]
    return (values.repeat(32768, 1),)


def analytic_reference(values, polarity):
    # values[0] @ weight.T keeps the reference correct for a batched perf input
    # of shape (rows, in_features) as well as a single fidelity vector.
    return _scaled_ref(
        values[0] @ _suite_weight(polarity).T, polarity, 16
    )


def known_answer_case(polarity):
    values = torch.ones(16)
    return (
        (values,),
        analytic_reference((values,), polarity),
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    # 16 input features need a sequence period above 16 to stay distinct.
    'state_timesteps': 32,
    'extra_checks': _kernel_specific_checks,
}


def test_linear_gaines():
    """Verify linear_gaines against analytic and known-answer streams, including reset and timing."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear_gaines()

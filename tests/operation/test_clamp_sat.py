import torch

from napl.sim.operation import add_scale, clamp_sat, encode, gen_num_seq
from napl.utils._shared_test import devices, streaming_suite


# Fixed band per polarity, both bounds inside the polarity's legal value range.
BOUNDS = {'bipolar': (-0.5, 0.5), 'unipolar': (0.25, 0.75)}
# Documented clamp_sat defaults, which the pins below rebuild the band-constant
# streams from without reading them off the operation.
FRACWIDTH = 8
FIRST_DIM = 2
# Band and timestep count on which a three-integer-bit stage accumulator departs
# from a wider one, while four, five, six, eight, and twelve agree bit for bit.
WIDTH_PIN_BAND = (0.9, 1.0)
WIDTH_PIN_TIMESTEPS = 1024
WIDE_INTWIDTH = 12


def make_operation(polarity, _timestep, _device):
    lo, hi = BOUNDS[polarity]
    return clamp_sat({'polarity': polarity, 'lo': lo, 'hi': hi})


def make_values(polarity):
    # Full legal range, so values below lo and above hi are both exercised.
    if polarity == 'bipolar':
        return (torch.linspace(-1.0, 1.0, 128),)
    return (torch.linspace(0.0, 1.0, 128),)


def make_random_perf_values(polarity):
    if polarity == 'bipolar':
        return (torch.linspace(-1.0, 1.0, 131072),)
    return (torch.linspace(0.0, 1.0, 131072),)


def analytic_reference(values, polarity):
    lo, hi = BOUNDS[polarity]
    return torch.clamp(values[0], lo, hi)


def known_answer_case(polarity):
    lo, hi = BOUNDS[polarity]
    if polarity == 'bipolar':
        values = torch.tensor([-0.9, 0.1, 0.9])
    else:
        values = torch.tensor([0.1, 0.5, 0.9])
    # A value below lo clamps up to lo, above hi clamps down to hi, inside passes.
    return (values,), torch.clamp(values, lo, hi)


CONFIG = {
    'polarities': ['bipolar', 'unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Gate 17 does not apply: clamp_sat returns a single output, which the suite
    # reads. extra_checks pins the two stage saturations that place the band and
    # the stage accumulator width, none of which the printed fidelity error
    # asserts.
    'extra_checks': lambda: (_saturation_pin_checks(), _accumulator_width_check()),
    'timesteps': 256,
}


def _bipolar_band(polarity):
    """Return the band as the internal bipolar stages see it."""
    lo, hi = BOUNDS[polarity]
    if polarity == 'bipolar':
        return lo, hi
    return 2.0 * lo - 1.0, 2.0 * hi - 1.0


def _bipolar_constants(polarity, lo, hi):
    """Return the three stage constants a band places on its Sobol streams."""
    grid = 2.0 ** -FRACWIDTH
    lo, hi = round(lo / grid) * grid, round(hi / grid) * grid
    if polarity == 'unipolar':
        lo, hi = 2.0 * lo - 1.0, 2.0 * hi - 1.0
    return (-lo, 1.0 + lo - hi, hi)


def _constant_bits(polarity, lo, hi):
    """Rebuild the three band-constant spike sequences from the band alone."""
    bits = []
    for index, constant in enumerate(_bipolar_constants(polarity, lo, hi)):
        num_seq = gen_num_seq(
            {'width': FRACWIDTH + 1, 'generator': 'sobol', 'dim': FIRST_DIM + index}
        )
        probability = torch.tensor((constant + 1.0) / 2.0)
        bits.append([float(b) for b in torch.gt(probability, num_seq).tolist()])
    return bits


def _accumulator_width_check():
    """Assert clamp_sat's spikes match a wide-accumulator stage chain bit for bit."""
    lo, hi = WIDTH_PIN_BAND
    timesteps = WIDTH_PIN_TIMESTEPS
    bits = _constant_bits('unipolar', lo, hi)
    stage_config = {'polarity': 'bipolar', 'scale': 1, 'intwidth': WIDE_INTWIDTH, 'fracwidth': 0}
    for device in devices():
        operation = clamp_sat({'polarity': 'unipolar', 'lo': lo, 'hi': hi}).to(device)
        stages = [add_scale(stage_config).to(device) for _ in range(3)]
        encoder = encode({
            'polarity': 'unipolar', 'timestep': timesteps, 'generator': 'sobol', 'dim': 1,
        }).to(device)
        values = torch.linspace(0.0, 1.0, 128, device=device)
        for step in range(timesteps):
            index = step % len(bits[0])
            spike = encoder(values)
            floor_out = stages[0](spike + bits[0][index], entry=3, dim=None)
            ceiling_out = stages[1](floor_out + bits[1][index] + 1.0, entry=3, dim=None)
            reference = stages[2](ceiling_out + bits[2][index], entry=3, dim=None)
            assert torch.equal(operation(spike), reference), (
                f'clamp_sat stage accumulators depart from {WIDE_INTWIDTH} integer bits '
                f'at timestep {step} on {device}'
            )
    print(f'stage accumulators match {WIDE_INTWIDTH} integer bits on every device.')


def _saturation_pin_checks():
    """Assert both stage saturations place the band, on a constant full-rate and a constant zero-rate input."""
    timesteps = CONFIG['timesteps']
    for device in devices():
        for polarity in CONFIG['polarities']:
            band_lo, _ = _bipolar_band(polarity)
            ceiling_bits = _constant_bits(polarity, *BOUNDS[polarity])[2]
            # A constant full-rate input sits above hi, so the ceiling stage saturates
            # every timestep and the shift stage passes its hi constant stream through
            # unchanged. Widening or removing that saturation lets the input through.
            operation = make_operation(polarity, timesteps, device).to(device)
            spike = torch.ones(2, 3, dtype=operation.stype, device=device)
            for step in range(timesteps):
                expected = ceiling_bits[step % len(ceiling_bits)]
                output = operation(spike)
                assert torch.equal(output, torch.full_like(output, expected)), (
                    f'{polarity} ceiling saturation broke at timestep {step} on {device}'
                )
            # A constant zero-rate input sits below lo, so the floor stage saturates
            # every timestep and the output rate is lo. Widening or removing that
            # saturation raises the emitted count.
            operation = make_operation(polarity, timesteps, device).to(device)
            spike = torch.zeros(2, 3, dtype=operation.stype, device=device)
            count = torch.zeros(2, 3, dtype=torch.int64, device=device)
            for _ in range(timesteps):
                count += operation(spike).type(torch.int64)
            expected = round(timesteps * (band_lo + 1.0) / 2.0)
            assert torch.equal(count, torch.full_like(count, expected)), (
                f'{polarity} floor saturation emitted {count.flatten().tolist()} '
                f'spikes on {device}, expected {expected}'
            )
    print('stage saturations place the band on every device.')


def _assert_rejected(bad):
    try:
        clamp_sat(bad)
    except AssertionError:
        return
    raise AssertionError(f'expected AssertionError for config {bad}')


def _config_validation_checks():
    """Assert that an invalid band or Sobol dimension is rejected and legal extremes are accepted."""
    # lo >= hi is rejected.
    _assert_rejected({'polarity': 'bipolar', 'lo': 0.5, 'hi': -0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': 0.3, 'hi': 0.3})
    # A bound outside the polarity's legal value range is rejected.
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5, 'hi': 1.5})
    _assert_rejected({'polarity': 'unipolar', 'lo': -0.25, 'hi': 0.75})
    # A NaN bound compares False against both ends of the range, so it is rejected
    # by the finiteness test rather than reaching the grid quantization.
    _assert_rejected({'polarity': 'bipolar', 'lo': float('nan'), 'hi': 0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5, 'hi': float('nan')})
    # An infinite bound is caught by the finiteness test and by the range comparison
    # alike, so it is rejected with an AssertionError rather than reaching the grid
    # quantization, where round() would raise OverflowError.
    _assert_rejected({'polarity': 'bipolar', 'lo': float('inf'), 'hi': 0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': float('-inf'), 'hi': 0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5, 'hi': float('inf')})
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5, 'hi': float('-inf')})
    # dim names three consecutive Sobol dimensions, so it is an integer in [1, 21199].
    for bad_dim in (2.5, 0, 21200):
        _assert_rejected({'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5, 'dim': bad_dim})
    # The legal region is the whole value range of the polarity, so a band spanning
    # it is accepted, as is the largest dim leaving room for all three streams.
    for good in ({'polarity': 'bipolar', 'lo': -1.0, 'hi': 1.0},
                 {'polarity': 'unipolar', 'lo': 0.0, 'hi': 1.0},
                 {'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5, 'dim': 21199}):
        clamp_sat(good)


def test_clamp_sat():
    """Verify clamp_sat against analytic and known-answer streams, pin the stage saturations and accumulator width, and reject invalid configs."""
    streaming_suite(CONFIG)
    _config_validation_checks()


if __name__ == '__main__':
    test_clamp_sat()

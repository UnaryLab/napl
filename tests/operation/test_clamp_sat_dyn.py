import torch

from napl.sim.operation import add_scale, clamp_sat_dyn, encode
from napl.utils._shared_test import devices, streaming_suite


# Band per polarity, both bounds inside the polarity's legal value range. The
# bounds ride their own Sobol dimensions, supplied to the operation as streams.
# encode thresholds a unipolar rate p and a bipolar value 2p - 1 against the same
# Sobol sequence, so a unipolar bound at (v + 1) / 2 is the same spike stream as
# the bipolar bound v. Every unipolar band and value grid in this file is chosen
# off that map from its bipolar counterpart, so each polarity drives streams the
# other polarity never drives and its pass is separate evidence.
BOUNDS = {'bipolar': (-0.5, 0.5), 'unipolar': (0.125, 0.6875)}
# Sobol dimensions the suite gives the input, the lo stream, and the hi stream.
INPUT_DIM, LO_DIM, HI_DIM = 1, 2, 3
# Integer bits clamp_sat_dyn builds its three stages at.
STAGE_INTWIDTH = 4
# Band and timestep count on which a three-integer-bit stage accumulator departs
# from a wider one, while four and twelve agree bit for bit.
WIDTH_PIN_BAND = (-0.875, -0.75)
WIDTH_PIN_TIMESTEPS = 512
NARROW_INTWIDTH = 3
WIDE_INTWIDTH = 12
# Timesteps per phase of the two-phase saturation pins, and the exact spike
# count each pin expects over the second phase.
PHASE_TIMESTEPS = 128
FLOOR_PIN_COUNT = {'bipolar': 91, 'unipolar': 83}
CEILING_PIN_COUNT = {'bipolar': 38, 'unipolar': 22}
# Value grid, band sweep, and inverted bands the runtime-band checks drive, per
# polarity. The sweep's second and third bands are the mid-run change; neither
# band pair is one on which four and twelve integer bits happen to agree. The
# wide inverted band is the first sweep band reversed, and the narrow one is a
# single grid step of the bound encoder wide.
SWEEP = {
    'bipolar': ((-0.75, 0.75), (-0.5, 0.0), (0.25, 0.5)),
    'unipolar': ((0.0625, 0.9375), (0.1875, 0.4375), (0.5625, 0.6875)),
}
NARROW_INVERTED = {'bipolar': (-0.25, -0.375), 'unipolar': (0.75, 0.6875)}


def make_operation(polarity, _timestep, _device):
    return clamp_sat_dyn({'polarity': polarity})


def _band_values(polarity, values):
    lo, hi = BOUNDS[polarity]
    return (values, torch.full_like(values, lo), torch.full_like(values, hi))


def make_values(polarity):
    # Full legal range, so values below lo and above hi are both exercised.
    if polarity == 'bipolar':
        return _band_values(polarity, torch.linspace(-1.0, 1.0, 128))
    return _band_values(polarity, torch.linspace(0.0, 1.0, 128))


def make_random_perf_values(polarity):
    if polarity == 'bipolar':
        return _band_values(polarity, torch.linspace(-1.0, 1.0, 131072))
    return _band_values(polarity, torch.linspace(0.0, 1.0, 131072))


def analytic_reference(values, polarity):
    return torch.clamp(values[0], BOUNDS[polarity][0], BOUNDS[polarity][1])


def known_answer_case(polarity):
    lo, hi = BOUNDS[polarity]
    if polarity == 'bipolar':
        values = torch.tensor([-0.9, 0.1, 0.9])
    else:
        values = torch.tensor([0.1, 0.5, 0.9])
    # A value below lo clamps up to lo, above hi clamps down to hi, inside passes.
    return _band_values(polarity, values), torch.clamp(values, lo, hi)


CONFIG = {
    'polarities': ['bipolar', 'unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Gate 17 does not apply: clamp_sat_dyn returns a single output, which the suite
    # reads. extra_checks pins the three stage saturations, the declared spike
    # dtype and pipeline delay, the behavior on a runtime band that moves and on
    # both a wide and a narrow inverted one, and the stage accumulator width, none
    # of which the printed fidelity error asserts.
    'extra_checks': lambda: (
        _saturation_pin_checks(),
        _dynamic_band_checks(),
        _accumulator_width_check(),
    ),
    'timesteps': 256,
}


def _bound_encoders(polarity, timesteps, device):
    """Return one encoder per bound stream, each on its own Sobol dimension."""
    return tuple(
        encode({
            'polarity': polarity, 'timestep': timesteps,
            'generator': 'sobol', 'dim': dim,
        }).to(device)
        for dim in (LO_DIM, HI_DIM)
    )


def _hand_chain(intwidths, device):
    """Return three saturating stages built at the given per-stage integer widths."""
    return [
        add_scale({
            'polarity': 'bipolar', 'scale': 1,
            'intwidth': intwidth, 'fracwidth': 0,
        }).to(device)
        for intwidth in intwidths
    ]


def _chain_step(stages, spike, lo_bit, hi_bit):
    """Run one timestep of the floor, ceiling, and shift stages."""
    floor_out = stages[0](spike + (1.0 - lo_bit), entry=3, dim=None)
    ceiling_out = stages[1](floor_out + 2.0 + lo_bit + (1.0 - hi_bit), entry=5, dim=None)
    return stages[2](ceiling_out + hi_bit, entry=3, dim=None)


def _band_schedule(bands, timesteps):
    """Return the band each timestep drives, the bands split evenly over the run."""
    span = timesteps // len(bands)
    return [bands[min(step // span, len(bands) - 1)] for step in range(timesteps)]


def _chain_outputs(intwidths, polarity, bands, values, timesteps, device):
    """Return the stacked spikes a hand-built stage chain emits over a band schedule, with no reset between bands."""
    stages = _hand_chain(intwidths, device)
    encoder = encode({
        'polarity': polarity, 'timestep': timesteps,
        'generator': 'sobol', 'dim': INPUT_DIM,
    }).to(device)
    lo_encoder, hi_encoder = _bound_encoders(polarity, timesteps, device)
    return torch.stack([
        _chain_step(
            stages, encoder(values),
            lo_encoder(torch.full_like(values, band[0])),
            hi_encoder(torch.full_like(values, band[1])),
        )
        for band in _band_schedule(bands, timesteps)
    ])


def _operation_outputs(polarity, bands, values, timesteps, device):
    """Return the stacked spikes clamp_sat_dyn emits over a band schedule, with no reset between bands."""
    operation = clamp_sat_dyn({'polarity': polarity}).to(device)
    encoder = encode({
        'polarity': polarity, 'timestep': timesteps,
        'generator': 'sobol', 'dim': INPUT_DIM,
    }).to(device)
    lo_encoder, hi_encoder = _bound_encoders(polarity, timesteps, device)
    return torch.stack([
        operation(
            encoder(values),
            lo_encoder(torch.full_like(values, band[0])),
            hi_encoder(torch.full_like(values, band[1])),
        )
        for band in _band_schedule(bands, timesteps)
    ])


def _pin_values(polarity, device):
    """Return the rank-2 band values the two-phase saturation pins drive."""
    lo, hi = BOUNDS[polarity]
    shape = (2, 3)
    return (torch.full(shape, lo, device=device), torch.full(shape, hi, device=device))


def _phase_count(operation, polarity, device, first_spike, second_spike):
    """Drive one constant-rate phase, then count the spikes of a second one."""
    timesteps = 2 * PHASE_TIMESTEPS
    lo_value, hi_value = _pin_values(polarity, device)
    lo_encoder, hi_encoder = _bound_encoders(polarity, timesteps, device)
    for _ in range(PHASE_TIMESTEPS):
        operation(first_spike, lo_encoder(lo_value), hi_encoder(hi_value))
    count = torch.zeros(2, 3, dtype=torch.int64, device=device)
    for _ in range(PHASE_TIMESTEPS):
        output = operation(second_spike, lo_encoder(lo_value), hi_encoder(hi_value))
        count += output.type(torch.int64)
    return count


def _saturation_pin_checks():
    """Assert the three stage saturations place the band on exact spike counts and an exact hi-stream sequence, and pin the declared spike dtype and pipeline delay."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            operation = clamp_sat_dyn({'polarity': polarity}).to(device)
            zeros = torch.zeros(2, 3, dtype=operation.stype, device=device)
            ones = torch.ones(2, 3, dtype=operation.stype, device=device)
            # The emitted dtype comes from the shift stage, so the spike dtype
            # clamp_sat_dyn declares is only a contract until something reads it against
            # what the chain emits. The pin stimulus is built from that declared
            # dtype, which no assertion below can see, so it is checked here.
            probe = operation(
                zeros,
                *[e(torch.zeros(2, 3, device=device)) for e in _bound_encoders(polarity, 2, device)],
            )
            assert probe.dtype == operation.stype, (
                f'{polarity} clamp_sat_dyn emitted {probe.dtype} spikes while declaring '
                f'{operation.stype} on {device}'
            )
            # clamp_sat_dyn adds no register to the three combinational stages it chains,
            # so its pipeline delay is their sum.
            stage_delay = sum(
                stage.hw.pp_delay for stage in
                (operation.floor_stage, operation.ceiling_stage, operation.shift_stage)
            )
            assert operation.hw.pp_delay == stage_delay, (
                f'{polarity} clamp_sat_dyn reports pp_delay {operation.hw.pp_delay} against '
                f'a stage chain delay of {stage_delay}'
            )
            operation.reset()
            # A constant zero-rate input sits below lo, so the floor stage saturates
            # low; the input then jumps to a constant full rate above hi, and the
            # floor stage's saturation is what bounds the debt it carried out of the
            # first phase, so it sets how many spikes the second phase emits. A wider
            # floor accumulator keeps more of that debt and emits fewer.
            count = _phase_count(operation, polarity, device, zeros, ones)
            expected = FLOOR_PIN_COUNT[polarity]
            assert torch.equal(count, torch.full_like(count, expected)), (
                f'{polarity} floor saturation emitted {count.flatten().tolist()} '
                f'spikes on {device}, expected {expected}'
            )
            # The mirror case: a constant full-rate input saturates the ceiling stage
            # high, and the drop to a constant zero rate reads the credit that stage
            # carried out of the first phase, which its high saturation bounds.
            operation = clamp_sat_dyn({'polarity': polarity}).to(device)
            count = _phase_count(operation, polarity, device, ones, zeros)
            expected = CEILING_PIN_COUNT[polarity]
            assert torch.equal(count, torch.full_like(count, expected)), (
                f'{polarity} ceiling saturation emitted {count.flatten().tolist()} '
                f'spikes on {device}, expected {expected}'
            )
            # A constant full-rate input holds the ceiling stage saturated high, so the
            # shift stage sees a stream that spikes every timestep and its own rail
            # and hi entries are all that place the output: it must emit the hi
            # stream, bit for bit. A shift stage that shifted by anything other than
            # hi - 1 departs from the hi stream, though not always at once: shifting by
            # hi + 1 holds for three timesteps before departing, so the pin compares
            # every timestep of the run rather than only the first.
            operation = clamp_sat_dyn({'polarity': polarity}).to(device)
            timesteps = 2 * PHASE_TIMESTEPS
            lo_value, hi_value = _pin_values(polarity, device)
            lo_encoder, hi_encoder = _bound_encoders(polarity, timesteps, device)
            for step in range(timesteps):
                hi_bit = hi_encoder(hi_value)
                output = operation(ones, lo_encoder(lo_value), hi_bit)
                assert torch.equal(output, hi_bit), (
                    f'{polarity} shift stage departed from the hi stream at '
                    f'timestep {step} on {device}'
                )
    print('stage saturations place the band on every device.')


def _sweep_values(polarity, device):
    """Return the value grid the runtime-band checks drive for one polarity."""
    if polarity == 'bipolar':
        return torch.linspace(-1.0, 1.0, 33, device=device)
    # Off the bipolar grid's unipolar image, linspace(0, 1, 33), at every element: the
    # image's k-th value is k / 32 = 32k / 1024 and this grid's is (33 + 30k) / 1024,
    # whose numerator is odd at every k and so is never a multiple of 32. The parity
    # holds for the whole grid rather than at a shared index, so no element of this grid
    # encodes to a stream the bipolar pass already drove. A grid whose step is odd, or
    # whose endpoint is on the image, does not have that property: at 29 / 1024 steps
    # from 1 / 32 to 15 / 16, both endpoints land on the image.
    return torch.linspace(33.0 / 1024.0, 993.0 / 1024.0, 33, device=device)


def _dynamic_band_checks():
    """Assert the band follows the bound streams at runtime, across a band sweep, a mid-run change with no reset, and both a wide and a narrow inverted band."""
    timesteps = 1024
    for device in devices():
        for polarity in CONFIG['polarities']:
            values = _sweep_values(polarity, device)
            sweep = SWEEP[polarity]
            # Each band is placed by the bound streams alone: the instance is reset
            # before each one, and a stage chain built here from the same three
            # add_scale stages, driven by its own encoders on the same dimensions,
            # emits the same spikes. That chain is the reference the assertion uses,
            # so the pin is exact rather than a tolerance on the decoded rate.
            operation = clamp_sat_dyn({'polarity': polarity}).to(device)
            for band in sweep:
                operation.reset()
                emitted = _operation_outputs(polarity, (band,), values, timesteps, device)
                expected = _chain_outputs(
                    (STAGE_INTWIDTH,) * 3, polarity, (band,), values, timesteps, device,
                )
                assert torch.equal(emitted, expected), (
                    f'{polarity} band {band} departed from the stage chain on {device}'
                )
                rate = emitted.type(torch.float32).mean(0)
                decoded = rate if polarity == 'unipolar' else 2.0 * rate - 1.0
                reference = torch.clamp(values, band[0], band[1])
                print(
                    f'[{device}][{polarity}] band={band} '
                    f'max_error={(decoded - reference).abs().max().item():.6f}'
                )
            # A band change mid-run: the stages are not reset at the change, so the
            # second half is driven by the new bound streams through state the first
            # half left behind. The reference chain runs the same schedule through the
            # same state, so the assertion pins every spike of both halves.
            change = sweep[1:]
            emitted = _operation_outputs(polarity, change, values, timesteps, device)
            expected = _chain_outputs(
                (STAGE_INTWIDTH,) * 3, polarity, change, values, timesteps, device,
            )
            assert torch.equal(emitted, expected), (
                f'{polarity} mid-run band change {change} departed from the stage '
                f'chain on {device}'
            )
            second_half = emitted[timesteps // 2:]
            rate = second_half.type(torch.float32).mean(0)
            decoded = rate if polarity == 'unipolar' else 2.0 * rate - 1.0
            reference = torch.clamp(values, change[1][0], change[1][1])
            print(
                f'[{device}][{polarity}] mid-run band change to {change[1]} '
                f'max_error={(decoded - reference).abs().max().item():.6f}'
            )
            # An inverted band cannot be guarded, since it is a property of the
            # streams and not of the config. The ceiling stage saturates high on every
            # value, so the shift stage sees a saturated input and passes the hi stream
            # through. On a wide inversion it passes bit for bit; on an inversion one
            # grid step of the bound encoder wide, the lo stream is low on a timestep
            # where the hi stream is high, which zeroes the ceiling delta, drops that
            # stage out of saturation for the step, and drops the spike. The output is
            # ceiling_out and hi together, so it can only drop hi spikes, never add.
            wide = (sweep[0][1], sweep[0][0])
            _assert_hi_passthrough(polarity, wide, values, timesteps, device, exact=True)
            _assert_hi_passthrough(
                polarity, NARROW_INVERTED[polarity], values, timesteps, device, exact=False,
            )
    print('the runtime band follows the bound streams on every device.')


def _assert_hi_passthrough(polarity, inverted, values, timesteps, device, exact):
    """Assert an inverted band emits a subset of the hi stream, bit-exactly when exact is set and strictly short of it when it is not."""
    operation = clamp_sat_dyn({'polarity': polarity}).to(device)
    lo_encoder, hi_encoder = _bound_encoders(polarity, timesteps, device)
    encoder = encode({
        'polarity': polarity, 'timestep': timesteps,
        'generator': 'sobol', 'dim': INPUT_DIM,
    }).to(device)
    departures = 0
    for step in range(timesteps):
        hi_bit = hi_encoder(torch.full_like(values, inverted[1]))
        output = operation(
            encoder(values),
            lo_encoder(torch.full_like(values, inverted[0])),
            hi_bit,
        )
        assert torch.all(output <= hi_bit), (
            f'{polarity} inverted band {inverted} emitted a spike outside the hi '
            f'stream at timestep {step} on {device}'
        )
        if not torch.equal(output, hi_bit):
            departures += 1
            assert not exact, (
                f'{polarity} inverted band {inverted} departed from the hi stream '
                f'at timestep {step} on {device}'
            )
    assert exact or departures > 0, (
        f'{polarity} narrowly inverted band {inverted} matched the hi stream bit for '
        f'bit on {device}, so it does not cover the departing case'
    )


def _accumulator_width_check():
    """Assert clamp_sat_dyn matches a wide-accumulator stage chain bit for bit while a narrower chain does not."""
    timesteps = WIDTH_PIN_TIMESTEPS
    for device in devices():
        values = torch.linspace(-1.0, 1.0, 64, device=device)
        wide = _chain_outputs(
            (WIDE_INTWIDTH,) * 3, 'bipolar', (WIDTH_PIN_BAND,), values, timesteps, device,
        )
        emitted = _operation_outputs(
            'bipolar', (WIDTH_PIN_BAND,), values, timesteps, device,
        )
        assert torch.equal(emitted, wide), (
            f'clamp_sat_dyn stage accumulators depart from {WIDE_INTWIDTH} integer bits '
            f'on {device}'
        )
        narrow = _chain_outputs(
            (NARROW_INTWIDTH,) * 3, 'bipolar', (WIDTH_PIN_BAND,), values, timesteps, device,
        )
        # A self-check on the reference, not coverage of clamp_sat_dyn: both sides are
        # built here, so no mutation of clamp_sat_dyn.py can fail it. It states that this
        # band and timestep count are ones on which the width matters, without which
        # the assertion above would pass at any width.
        assert not torch.equal(narrow, wide), (
            f'{NARROW_INTWIDTH} integer bits already matches {WIDE_INTWIDTH} on '
            f'{device}, so the width pin proves nothing'
        )
    print(f'stage accumulators match {WIDE_INTWIDTH} integer bits on every device.')


def _assert_rejected(bad):
    try:
        clamp_sat_dyn(bad)
    except AssertionError:
        return
    raise AssertionError(f'expected AssertionError for config {bad}')


def _config_validation_checks():
    """Assert that a missing or invalid polarity and an unexpected key are rejected."""
    _assert_rejected({})
    _assert_rejected({'polarity': 'tripolar'})
    # No band lives in the config, so a band key is not accepted.
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5})
    _assert_rejected({'polarity': 'bipolar', 'hi': 0.5})
    for good in ({'polarity': 'bipolar'}, {'polarity': 'unipolar'},
                 {'polarity': 'bipolar', 'name': 'band'}):
        clamp_sat_dyn(good)


def test_clamp_sat_dyn():
    """Verify clamp_sat_dyn against analytic and known-answer streams, pin the stage saturations, the runtime band, and the accumulator width, and reject invalid configs."""
    streaming_suite(CONFIG)
    _config_validation_checks()


if __name__ == '__main__':
    test_clamp_sat_dyn()

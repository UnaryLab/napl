import torch

from napl.sim.operation import clamp_comp_dyn, encode, max, min
from napl.utils._shared_test import devices, streaming_suite


# Grid entries every value and bound in this file is drawn from; a bipolar entry
# sits on an even multiple of one threshold step and a unipolar entry on an odd
# one, which is what keeps the two polarities off each other's spike streams.
GRID_LEN = 256
# Band per polarity, as grid step indices, both bounds inside the polarity's
# legal value range. The bounds ride their own Sobol dimensions, supplied to the
# operation as streams.
BAND_STEPS = {'bipolar': (32, 96), 'unipolar': (16, 88)}
# Sobol dimensions the suite gives the input, the lo stream, and the hi stream.
INPUT_DIM, LO_DIM, HI_DIM = 1, 2, 3
# Band sweep per polarity, as grid step indices. The second and third bands are
# the mid-run change, driven with no reset between them.
SWEEP_STEPS = {
    'bipolar': ((16, 112), (32, 64), (80, 96)),
    'unipolar': ((8, 120), (24, 56), (72, 88)),
}
# Inverted bands, as grid step indices: the wide one is the first sweep band
# reversed and the narrow one is a single grid step of the value grid wide.
NARROW_INVERTED_STEPS = {'bipolar': (48, 47), 'unipolar': (48, 47)}
# Timesteps the exact pins run for, and the value grid they drive.
PIN_TIMESTEPS = 512
SWEEP_LEN = 33
# Grid step indices the 0-dim check drives, one value inside BAND_STEPS and one
# above its hi bound, with the per-element output spike counts they place over a
# PIN_TIMESTEPS run on that band. A 0-dim input carries no lane to place a value
# grid on, so the check reads one value at a time.
RANK0_STEPS = (70, 121)
RANK0_COUNT = {'bipolar': (281, 384), 'unipolar': (283, 354)}
# Inverted-band measurements over a PIN_TIMESTEPS run on the SWEEP_LEN grid: the
# timesteps whose output differs from the hi stream, the element-timestep pairs
# carrying a spike the hi stream does not, and the pairs missing one it does.
# Each entry names one (lo, hi, input) rate triple, since that triple and not the
# polarity label is what the measurement follows; the two polarities appear here
# as separate entries because _grid_value puts them on different rate grids, so
# their bands of the same step index are different rate triples.
INVERTED_PIN = {
    ('bipolar', 'wide'): (4, 4, 0),
    ('bipolar', 'narrow'): (72, 478, 428),
    ('unipolar', 'wide'): (0, 0, 0),
    ('unipolar', 'narrow'): (152, 788, 721),
}


def _grid_value(polarity, step):
    """Return the grid value of one step index, in the polarity's own value range."""
    # The unscrambled Sobol thresholds of a 2**k-long sequence are the 2**k grid
    # points, each once, so a rate 2*step/GRID_LEN carries an even spike count over
    # the period and (2*step + 1)/GRID_LEN an odd one. A bipolar value v and the
    # unipolar rate (v + 1) / 2 threshold against the same sequence, so giving the
    # two polarities opposite parities keeps every stream one polarity drives off
    # every stream the other drives.
    if polarity == 'bipolar':
        return 2.0 * (2.0 * step / GRID_LEN) - 1.0
    return (2.0 * step + 1.0) / GRID_LEN


def _grid(polarity, count):
    """Return a value grid of the requested length, spanning the polarity's range."""
    steps = torch.arange(count, dtype=torch.float32).mul(GRID_LEN / 2 / count).floor()
    return _grid_value(polarity, steps)


def _band(polarity, steps):
    """Return the band value pair of a step-index pair."""
    return tuple(_grid_value(polarity, step) for step in steps)


def make_operation(polarity, _timestep, _device):
    return clamp_comp_dyn({'polarity': polarity})


def _band_values(polarity, values):
    lo, hi = _band(polarity, BAND_STEPS[polarity])
    return (values, torch.full_like(values, lo), torch.full_like(values, hi))


def make_values(polarity):
    # Whole legal range, so values below lo and above hi are both exercised.
    return _band_values(polarity, _grid(polarity, 128))


def make_random_perf_values(polarity):
    return _band_values(polarity, _grid(polarity, 131072))


def analytic_reference(values, polarity):
    lo, hi = _band(polarity, BAND_STEPS[polarity])
    return torch.clamp(values[0], lo, hi)


def known_answer_case(polarity):
    values = torch.tensor([_grid_value(polarity, step) for step in (6, 64, 121)])
    lo, hi = _band(polarity, BAND_STEPS[polarity])
    # A value below lo clamps up to lo, above hi clamps down to hi, inside passes.
    return _band_values(polarity, values), torch.clamp(values, lo, hi)


CONFIG = {
    'polarities': ['bipolar', 'unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Gate 17 does not apply: clamp_comp_dyn returns a single output, which the
    # suite reads. extra_checks pins the declared spike dtype and pipeline delay,
    # the exact spikes a runtime band places, a mid-run band change made with no
    # reset, the two rails on which the output is a bound stream bit for bit, the
    # measured behavior on a wide and a narrow inverted band, the rate triple rather
    # than the polarity label that this behavior follows, and the separation of the
    # two polarities' spike streams, none of which the printed fidelity error
    # asserts.
    'extra_checks': lambda: (
        _contract_checks(),
        _dynamic_band_checks(),
        _rail_pin_checks(),
        _inverted_band_checks(),
        _rate_triple_checks(),
        _stream_separation_check(),
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


def _band_schedule(bands, timesteps):
    """Return the band each timestep drives, the bands split evenly over the run."""
    span = timesteps // len(bands)
    last = len(bands) - 1
    return [
        bands[index if index < last else last]
        for index in (step // span for step in range(timesteps))
    ]


def _run(step_fn, polarity, bands, values, timesteps, device):
    """Drive one timestep per band in the schedule and return the stacked spikes and hi stream."""
    encoder = encode({
        'polarity': polarity, 'timestep': timesteps,
        'generator': 'sobol', 'dim': INPUT_DIM,
    }).to(device)
    lo_encoder, hi_encoder = _bound_encoders(polarity, timesteps, device)
    outputs, hi_bits = [], []
    for band in _band_schedule(bands, timesteps):
        spike = encoder(values)
        lo_bit = lo_encoder(torch.full_like(values, band[0]))
        hi_bit = hi_encoder(torch.full_like(values, band[1]))
        outputs.append(step_fn(spike, lo_bit, hi_bit))
        hi_bits.append(hi_bit)
    return torch.stack(outputs), torch.stack(hi_bits)


def _chain_step_fn(device):
    """Return a step function driving a hand-built selector chain, held across timesteps."""
    floor_stage, ceiling_stage = max().to(device), min().to(device)

    def step(spike, lo_bit, hi_bit):
        floor_out, _ = floor_stage(spike, lo_bit)
        ceiling_out, _ = ceiling_stage(floor_out, hi_bit)
        return ceiling_out

    return step


def _operation_step_fn(polarity, device):
    """Return a step function driving one clamp_comp_dyn instance, held across timesteps."""
    operation = clamp_comp_dyn({'polarity': polarity}).to(device)
    return lambda spike, lo_bit, hi_bit: operation(spike, lo_bit, hi_bit)


def _contract_checks():
    """Assert the emitted spike dtype matches the declared one and the pipeline delay is the selector chain's."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            operation = clamp_comp_dyn({'polarity': polarity}).to(device)
            zeros = torch.zeros(2, 3, dtype=operation.stype, device=device)
            ones = torch.ones(2, 3, dtype=operation.stype, device=device)
            # The emitted dtype comes from the ceiling selector, so the spike dtype
            # clamp_comp_dyn declares is only a contract until something reads it
            # against what the chain emits.
            probe = operation(zeros, zeros, ones)
            assert probe.dtype == operation.stype, (
                f'{polarity} clamp_comp_dyn emitted {probe.dtype} spikes while declaring '
                f'{operation.stype} on {device}'
            )
            assert probe.shape == (2, 3), (
                f'{polarity} clamp_comp_dyn emitted shape {tuple(probe.shape)} on a '
                f'rank-2 input on {device}'
            )
            # clamp_comp_dyn adds no register to the two combinational selectors it
            # chains, so its pipeline delay is their sum.
            stage_delay = (
                operation.floor_stage.hw.pp_delay + operation.ceiling_stage.hw.pp_delay
            )
            assert operation.hw.pp_delay == stage_delay, (
                f'{polarity} clamp_comp_dyn reports pp_delay {operation.hw.pp_delay} '
                f'against a selector chain delay of {stage_delay}'
            )
    print('the declared spike dtype and pipeline delay match the selector chain.')


def _dynamic_band_checks():
    """Assert the band follows the bound streams across a sweep and a mid-run change made with no reset, against a hand-built selector chain."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            values = _grid(polarity, SWEEP_LEN).to(device)
            sweep = [_band(polarity, steps) for steps in SWEEP_STEPS[polarity]]
            # Each band is placed by the bound streams alone: a chain built here from
            # bare max and min selectors, driven by its own encoders on the same
            # dimensions, emits the same spikes. That chain is the reference the
            # assertion uses, so the pin is exact rather than a tolerance on the
            # decoded rate.
            for band in sweep:
                emitted, _ = _run(
                    _operation_step_fn(polarity, device), polarity, (band,),
                    values, PIN_TIMESTEPS, device,
                )
                expected, _ = _run(
                    _chain_step_fn(device), polarity, (band,),
                    values, PIN_TIMESTEPS, device,
                )
                assert torch.equal(emitted, expected), (
                    f'{polarity} band {band} departed from the selector chain on {device}'
                )
                rate = emitted.type(torch.float32).mean(0)
                decoded = rate if polarity == 'unipolar' else 2.0 * rate - 1.0
                reference = torch.clamp(values, band[0], band[1])
                print(
                    f'[{device}][{polarity}] band={band} '
                    f'max_error={(decoded - reference).abs().max().item():.6f}'
                )
            # A band change mid-run: the selectors are not reset at the change, so the
            # second half is driven by the new bound streams through the latched
            # decisions the first half left behind. The reference chain runs the same
            # schedule through its own latched state, so the assertion pins every
            # spike of both halves.
            change = tuple(sweep[1:])
            emitted, _ = _run(
                _operation_step_fn(polarity, device), polarity, change,
                values, PIN_TIMESTEPS, device,
            )
            expected, _ = _run(
                _chain_step_fn(device), polarity, change, values, PIN_TIMESTEPS, device,
            )
            assert torch.equal(emitted, expected), (
                f'{polarity} mid-run band change {change} departed from the selector '
                f'chain on {device}'
            )
            # The wrong source for the same comparison: the same two bands driven in
            # the opposite order, which a run that ignored the runtime band would
            # reproduce.
            swapped, _ = _run(
                _chain_step_fn(device), polarity, change[::-1],
                values, PIN_TIMESTEPS, device,
            )
            assert not torch.equal(emitted, swapped), (
                f'{polarity} mid-run band change {change} matched the same bands in '
                f'the opposite order on {device}, so the run does not read the band'
            )
            second_half = emitted[PIN_TIMESTEPS // 2:]
            rate = second_half.type(torch.float32).mean(0)
            decoded = rate if polarity == 'unipolar' else 2.0 * rate - 1.0
            reference = torch.clamp(values, change[1][0], change[1][1])
            print(
                f'[{device}][{polarity}] mid-run band change to {change[1]} '
                f'max_error={(decoded - reference).abs().max().item():.6f}'
            )
    print('the runtime band follows the bound streams on every device.')


def _rail_pin_checks():
    """Assert a zero-rate input emits the lo stream and a full-rate input the hi stream, bit for bit, and that neither matches the other bound."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            band = _band(polarity, BAND_STEPS[polarity])
            values = torch.full((2, 3), band[0], device=device)
            lo_encoder, hi_encoder = _bound_encoders(polarity, PIN_TIMESTEPS, device)
            operation = clamp_comp_dyn({'polarity': polarity}).to(device)
            for rail, bound_index in (
                (torch.zeros(2, 3, dtype=operation.stype, device=device), 0),
                (torch.ones(2, 3, dtype=operation.stype, device=device), 1),
            ):
                operation.reset()
                lo_encoder.reset()
                hi_encoder.reset()
                for step in range(PIN_TIMESTEPS):
                    bits = (
                        lo_encoder(torch.full_like(values, band[0])),
                        hi_encoder(torch.full_like(values, band[1])),
                    )
                    output = operation(rail, *bits)
                    assert torch.equal(output, bits[bound_index]), (
                        f'{polarity} rail input departed from its bound stream at '
                        f'timestep {step} on {device}'
                    )
                # The wrong source for the same comparison: the other bound's stream.
                # The two bound streams differ, so a rail output that matched both
                # would say nothing about which bound the rail selects.
                assert not torch.equal(bits[0], bits[1]), (
                    f'{polarity} both bound streams carried the same spike on {device}, '
                    f'so the rail pin does not name a bound'
                )
    print('a saturated input emits its bound stream bit for bit on every device.')


def _inverted_band_checks():
    """Pin the measured behavior on a wide and a narrow inverted band, where the output tracks the hi rate without being the hi stream."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            values = _grid(polarity, SWEEP_LEN).to(device)
            inverted = {
                'wide': _band(polarity, SWEEP_STEPS[polarity][0][::-1]),
                'narrow': _band(polarity, NARROW_INVERTED_STEPS[polarity]),
            }
            for label, band in inverted.items():
                emitted, hi_bits = _run(
                    _operation_step_fn(polarity, device), polarity, (band,),
                    values, PIN_TIMESTEPS, device,
                )
                expected, _ = _run(
                    _chain_step_fn(device), polarity, (band,),
                    values, PIN_TIMESTEPS, device,
                )
                assert torch.equal(emitted, expected), (
                    f'{polarity} {label} inverted band {band} departed from the selector '
                    f'chain on {device}'
                )
                measured = (
                    int((emitted != hi_bits).any(-1).sum()),
                    int((emitted > hi_bits).sum()),
                    int((emitted < hi_bits).sum()),
                )
                # The measured contract that separates this circuit from the
                # saturating-adder clamp_sat_dyn, whose inverted-band output is the hi
                # stream masked and so can only drop spikes: here the output carries
                # spikes the hi stream does not. Whether a given inversion does so
                # follows its (lo, hi, input) rate triple rather than its polarity
                # label or its width, so the claim is pinned on the narrow band that
                # carries it under both labels, from the values this run measured.
                if label == 'narrow':
                    assert measured[1] > 0, (
                        f'{polarity} narrow inverted band {band} emitted no spike '
                        f'outside the hi stream on {device}'
                    )
                pinned = INVERTED_PIN[(polarity, label)]
                assert measured == pinned, (
                    f'{polarity} {label} inverted band {band} measured {measured} '
                    f'against the pinned {pinned} on {device}'
                )
                # The wrong source for the same pin: the other inverted band of this
                # polarity, whose run must not reproduce these numbers.
                other = inverted['narrow' if label == 'wide' else 'wide']
                other_emitted, other_hi = _run(
                    _chain_step_fn(device), polarity, (other,),
                    values, PIN_TIMESTEPS, device,
                )
                assert (
                    int((other_emitted != other_hi).any(-1).sum()),
                    int((other_emitted > other_hi).sum()),
                    int((other_emitted < other_hi).sum()),
                ) != pinned, (
                    f'{polarity} inverted band {other} reproduced the pinned {pinned} '
                    f'on {device}, so the pin does not name a band'
                )
                rate = emitted.type(torch.float32).mean(0)
                hi_rate = hi_bits.type(torch.float32).mean(0)
                print(
                    f'[{device}][{polarity}] {label} inverted band={band} '
                    f'max_rate_deviation_from_hi={(rate - hi_rate).abs().max().item():.6f}'
                )
    print('the inverted band emits the pinned spikes on every device.')


def _rate(polarity, value):
    """Return the spike rate a value of one polarity is encoded at."""
    return value if polarity == 'unipolar' else (value + 1.0) / 2.0


def _value(polarity, rate):
    """Return the value of one polarity that is encoded at a spike rate."""
    return rate if polarity == 'unipolar' else 2.0 * rate - 1.0


def _inverted_verdict(polarity, band, values, device):
    """Return how a run's output stands against the hi stream: differing timesteps, extra spikes, missing spikes."""
    emitted, hi_bits = _run(
        _operation_step_fn(polarity, device), polarity, (band,),
        values, PIN_TIMESTEPS, device,
    )
    return (
        int((emitted != hi_bits).any(-1).sum()),
        int((emitted > hi_bits).sum()),
        int((emitted < hi_bits).sum()),
    )


def _rate_triple_checks():
    """Assert the inverted-band verdict follows the (lo, hi, input) rate triple rather than the polarity label: one rate lattice read in either polarity gives one verdict."""
    for device in devices():
        verdicts = {}
        for source in CONFIG['polarities']:
            # The rate lattice and the two bound rates this source's own pins drive,
            # taken as rates so either polarity can be asked to read them.
            lattice = _rate(source, _grid(source, SWEEP_LEN).to(device))
            bands = {
                'wide': SWEEP_STEPS[source][0][::-1],
                'narrow': NARROW_INVERTED_STEPS[source],
            }
            for label, steps in bands.items():
                rates = tuple(_rate(source, _grid_value(source, step)) for step in steps)
                for driven in CONFIG['polarities']:
                    # forward reads no polarity, so a label change must move nothing as
                    # long as the three streams keep their rates. Reading the same rates
                    # back out as values of the driven polarity is what holds them fixed.
                    verdicts[(source, label, driven)] = _inverted_verdict(
                        driven, tuple(_value(driven, rate) for rate in rates),
                        _value(driven, lattice), device,
                    )
                assert (
                    verdicts[(source, label, 'bipolar')]
                    == verdicts[(source, label, 'unipolar')]
                ), (
                    f'the {source} {label} inverted rate triple {rates} measured '
                    f'{verdicts[(source, label, "bipolar")]} read as bipolar and '
                    f'{verdicts[(source, label, "unipolar")]} read as unipolar on '
                    f'{device}, so the verdict follows the label'
                )
                # The verdict a polarity label carries in INVERTED_PIN is the one its
                # own rate triple gives, which is what makes that table a statement
                # about rates rather than about labels.
                assert verdicts[(source, label, source)] == INVERTED_PIN[(source, label)], (
                    f'the {source} {label} rate triple {rates} measured '
                    f'{verdicts[(source, label, source)]} against the pinned '
                    f'{INVERTED_PIN[(source, label)]} on {device}'
                )
        # The wrong source for the same comparison: the two polarities' wide inverted
        # rate triples, read in one fixed label. Agreement across labels would say
        # nothing if every rate triple gave the same verdict.
        assert (
            verdicts[('bipolar', 'wide', 'bipolar')]
            != verdicts[('unipolar', 'wide', 'bipolar')]
        ), (
            f'the two wide inverted rate triples measured the same verdict read as '
            f'bipolar on {device}, so the verdict names no rate triple'
        )
    print('the inverted-band verdict follows the rate triple on every device.')


def _streams(polarity, dim, values, timesteps):
    """Return the set of spike streams a value list encodes to on one Sobol dimension."""
    encoder = encode({
        'polarity': polarity, 'timestep': timesteps, 'generator': 'sobol', 'dim': dim,
    })
    stacked = torch.stack([encoder(values) for _ in range(timesteps)])
    return {tuple(column.tolist()) for column in stacked.t()}


def _driven_bounds(polarity, position):
    """Return every bound value this file drives on one side of the band."""
    steps = [BAND_STEPS[polarity][position], NARROW_INVERTED_STEPS[polarity][position]]
    steps += [band[position] for band in SWEEP_STEPS[polarity]]
    # The inverted bands drive each sweep bound on the opposite side as well.
    steps += [band[1 - position] for band in SWEEP_STEPS[polarity]]
    steps += [NARROW_INVERTED_STEPS[polarity][1 - position]]
    return torch.tensor([_grid_value(polarity, step) for step in steps])


def _stream_separation_check():
    """Assert no value or bound one polarity drives encodes to a stream the other polarity drives."""
    for timesteps in (CONFIG['timesteps'], PIN_TIMESTEPS):
        driven = {
            INPUT_DIM: lambda polarity: torch.cat((
                make_values(polarity)[0],
                known_answer_case(polarity)[0][0],
                _grid(polarity, SWEEP_LEN),
            )),
            LO_DIM: lambda polarity: _driven_bounds(polarity, 0),
            HI_DIM: lambda polarity: _driven_bounds(polarity, 1),
        }
        for dim, values_of in driven.items():
            shared = (
                _streams('bipolar', dim, values_of('bipolar'), timesteps)
                & _streams('unipolar', dim, values_of('unipolar'), timesteps)
            )
            assert not shared, (
                f'{len(shared)} stream(s) on dimension {dim} are driven by both '
                f'polarities at {timesteps} timesteps'
            )
    print('the two polarities drive disjoint spike streams.')


def _rank0_checks():
    """Assert a 0-dim input is carried at rank 1 and places the pinned spike count on the fixed band."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            band = _band(polarity, BAND_STEPS[polarity])
            for index, step in enumerate(RANK0_STEPS):
                value = torch.tensor(_grid_value(polarity, step), device=device)
                outputs, _ = _run(
                    _operation_step_fn(polarity, device), polarity, (band,), value,
                    PIN_TIMESTEPS, device,
                )
                assert outputs.shape == (PIN_TIMESTEPS, 1), (
                    f'{polarity} clamp_comp_dyn emitted per-timestep shape '
                    f'{tuple(outputs.shape[1:])} on a 0-dim input on {device}, not the '
                    f'rank-1 shape the 0-dim bound bits leave it at'
                )
                count = int(outputs.sum())
                assert count == RANK0_COUNT[polarity][index], (
                    f'{polarity} clamp_comp_dyn counted {count} spikes on a 0-dim grid '
                    f'step {step} on {device}, against the pinned '
                    f'{RANK0_COUNT[polarity][index]}'
                )
    print('a 0-dim input is carried at rank 1 and keeps the pinned spike count.')


def _assert_rejected(bad):
    try:
        clamp_comp_dyn(bad)
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
        clamp_comp_dyn(good)


def test_clamp_comp_dyn():
    """Verify clamp_comp_dyn against analytic and known-answer streams, pin the runtime band, the rails, the inverted-band behavior, and the 0-dim input rank, and reject invalid configs."""
    streaming_suite(CONFIG)
    _rank0_checks()
    _config_validation_checks()


if __name__ == '__main__':
    test_clamp_comp_dyn()

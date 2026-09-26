import torch

from napl.sim.operation import clamp_comp, encode, gen_num_seq, max, min
from napl.utils._shared_test import devices, streaming_suite


# Band per polarity, both bounds inside the polarity's legal value range. The
# operation encodes them itself, on the Sobol dimensions BOUND_DIM names.
BOUNDS = {'bipolar': (-0.5, 0.5), 'unipolar': (0.125, 0.6875)}
# A second band, used as the wrong source every pinned constant below is
# rechecked against: the same reference chain driven by this band must produce a
# different number, or the pin would hold for a band it was not derived from.
TWIN_BOUNDS = {'bipolar': (-0.25, 0.25), 'unipolar': (0.0625, 0.5625)}
# Documented clamp_comp defaults, which the pins below rebuild the bound streams
# from without reading them off the operation. PIN_COUNT holds under both
# constants, because a Sobol dimension permutes one period's thresholds among
# themselves and a wider sequence carries this period as its prefix, so neither
# moves a spike count taken over a whole period: dim 3, dim 7, width 8 and width
# 10 all reproduce the pinned counts. The bit-exact comparisons are what carry the
# constants. Both constants redden _contract_checks, where dim 3, dim 7, width 8
# and width 10 each fail the rebuild. The chain comparison in _band_pin_checks
# reddens on dim 3, dim 7 and width 8, but not on width 10: a width-10 stream
# carries the width-9 one as its first 512 entries, which is all the chain reads.
# _rail_pin_checks splits the three it does catch by how they fail: a wrong
# dimension fails its bound-stream comparison, while width 8 leaves a bound stream
# shorter than PIN_TIMESTEPS and so raises a shape error there instead. Width 10
# leaves _rail_pin_checks green too, for the same prefix reason.
BOUND_DIM = 2
BOUND_WIDTH = 9
# Sobol dimension the suite gives the input.
INPUT_DIM = 1
# Timesteps the exact pins run for, and the value grid they drive, one grid entry
# below lo, one above hi, and the rest inside and around the band.
PIN_TIMESTEPS = 512
PIN_STEPS = ((6, 48, 70), (100, 121, 127))
# Per-element output spike counts over a PIN_TIMESTEPS run on the PIN_STEPS grid.
PIN_COUNT = {
    'bipolar': (128, 193, 281, 384, 384, 384),
    'unipolar': (64, 195, 283, 352, 352, 352),
}
# Grid entries the value builders read; a bipolar entry sits on an even multiple
# of one threshold step and a unipolar entry on an odd one, which is what keeps
# the two polarities off each other's spike streams.
GRID_LEN = 256
# Flat indices into PIN_STEPS and PIN_COUNT that the 0-dim check drives, one
# value inside the band and one above hi. A 0-dim input carries no lane to place
# a rank-2 grid on, so the check reads these entries one value at a time.
RANK0_INDEX = (2, 4)
# Bound grid the band-box accuracy report sweeps, as a count of evenly spaced grid
# step indices spanning the polarity's whole legal value range. Every ordered pair
# of them is one band, so the report covers the band box the class documents
# rather than a single band. The value grid it drives is BOX_VALUES long.
BOX_LEN = 17
# Value grid length of that sweep, long enough that the reported worst case has
# stopped moving: 33 values report a bipolar 0.062500, 65 and 129 report 0.066406,
# and 257 and 513 report the same, while the unipolar line reaches 0.025391 only
# at 129 and holds there at 257 and 513.
BOX_VALUES = 129
# Bounds of the separately reported off-grid band sweep, as rates of the bound
# streams. Every box band above lands on the 1/2**BOUND_WIDTH threshold grid the
# bound streams resolve, so the box says nothing about a band that does not; these
# rates have no power-of-two denominator and _band_box_report asserts that.
OFF_GRID_RATES = (1.0 / 7.0, 1.0 / 3.0, 3.0 / 7.0, 5.0 / 9.0, 6.0 / 7.0)


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


def _pin_values(polarity, device):
    """Return the rank-2 value grid the exact pins drive."""
    return torch.tensor(
        [[_grid_value(polarity, step) for step in row] for row in PIN_STEPS],
        device=device,
    )


def make_operation(polarity, _timestep, _device):
    lo, hi = BOUNDS[polarity]
    return clamp_comp({'polarity': polarity, 'lo': lo, 'hi': hi})


def make_values(polarity):
    # Whole legal range, so values below lo and above hi are both exercised.
    return (_grid(polarity, 128),)


def make_random_perf_values(polarity):
    return (_grid(polarity, 131072),)


def analytic_reference(values, polarity):
    lo, hi = BOUNDS[polarity]
    return torch.clamp(values[0], lo, hi)


def known_answer_case(polarity):
    values = torch.tensor([_grid_value(polarity, step) for step in (6, 64, 121)])
    lo, hi = BOUNDS[polarity]
    # A value below lo clamps up to lo, above hi clamps down to hi, inside passes.
    return (values,), torch.clamp(values, lo, hi)


CONFIG = {
    'polarities': ['bipolar', 'unipolar'],
    'make_operation': make_operation,
    'make_values': make_values,
    'make_random_perf_values': make_random_perf_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    # Gate 17 does not apply: clamp_comp returns a single output, which the suite
    # reads. extra_checks pins the declared spike dtype and pipeline delay, the
    # documented bound-stream defaults, the exact spikes and per-element spike
    # counts the band places, the two rails on which the output is a bound stream
    # bit for bit, and the separation of the two polarities' spike streams, none
    # of which the printed fidelity error asserts. It also prints, and does not
    # assert, the band-box accuracy the class docstring points at.
    'extra_checks': lambda: (
        _contract_checks(),
        _band_pin_checks(),
        _rail_pin_checks(),
        _stream_separation_check(),
        _band_box_report(),
    ),
    'timesteps': 256,
}


def _bound_bits(polarity, band, device):
    """Return the two bound spike sequences, rebuilt from the documented defaults."""
    bits = []
    for index, bound in enumerate(band):
        num_seq = gen_num_seq({
            'width': BOUND_WIDTH, 'generator': 'sobol', 'dim': BOUND_DIM + index,
        })
        # A unipolar rate p and the bipolar value 2p - 1 threshold against the same
        # sequence, so a bipolar bound moves to the rate that denotes it.
        probability = bound if polarity == 'unipolar' else (bound + 1.0) / 2.0
        bits.append(torch.gt(torch.tensor(probability), num_seq).type(torch.int8).to(device))
    return bits


def _chain_outputs(polarity, band, values, timesteps, device, input_spikes=None):
    """Return the stacked spikes a hand-built selector chain emits, one timestep per row."""
    floor_stage, ceiling_stage = max().to(device), min().to(device)
    lo_bits, hi_bits = _bound_bits(polarity, band, device)
    encoder = encode({
        'polarity': polarity, 'timestep': timesteps,
        'generator': 'sobol', 'dim': INPUT_DIM,
    }).to(device)
    outputs = []
    for step in range(timesteps):
        spike = encoder(values) if input_spikes is None else input_spikes
        floor_out, _ = floor_stage(spike, lo_bits[step % lo_bits.numel()].expand_as(spike))
        ceiling_out, _ = ceiling_stage(floor_out, hi_bits[step % hi_bits.numel()].expand_as(spike))
        outputs.append(ceiling_out)
    return torch.stack(outputs)


def _operation_outputs(polarity, band, values, timesteps, device, input_spikes=None):
    """Return the stacked spikes clamp_comp emits, one timestep per row."""
    operation = clamp_comp({
        'polarity': polarity, 'lo': band[0], 'hi': band[1],
    }).to(device)
    encoder = encode({
        'polarity': polarity, 'timestep': timesteps,
        'generator': 'sobol', 'dim': INPUT_DIM,
    }).to(device)
    return torch.stack([
        operation(encoder(values) if input_spikes is None else input_spikes)
        for _ in range(timesteps)
    ])


def _contract_checks():
    """Assert the emitted spike dtype matches the declared one, the pipeline delay is the selector chain's, and the documented bound dimension and width rebuild the operation's own bound streams."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            lo, hi = BOUNDS[polarity]
            operation = clamp_comp({'polarity': polarity, 'lo': lo, 'hi': hi}).to(device)
            # The emitted dtype comes from the ceiling selector, so the spike dtype
            # clamp_comp declares is only a contract until something reads it against
            # what the chain emits.
            probe = operation(torch.zeros(2, 3, dtype=operation.stype, device=device))
            assert probe.dtype == operation.stype, (
                f'{polarity} clamp_comp emitted {probe.dtype} spikes while declaring '
                f'{operation.stype} on {device}'
            )
            assert probe.shape == (2, 3), (
                f'{polarity} clamp_comp emitted shape {tuple(probe.shape)} on a rank-2 '
                f'input on {device}'
            )
            # clamp_comp adds no register to the two combinational selectors it chains,
            # so its pipeline delay is their sum.
            stage_delay = (
                operation.floor_stage.hw.pp_delay + operation.ceiling_stage.hw.pp_delay
            )
            assert operation.hw.pp_delay == stage_delay, (
                f'{polarity} clamp_comp reports pp_delay {operation.hw.pp_delay} against '
                f'a selector chain delay of {stage_delay}'
            )
            # The pins rebuild the bound streams from BOUND_DIM and BOUND_WIDTH instead
            # of reading them off the operation, so those two constants are a claim
            # about the documented defaults that only this comparison can contradict:
            # a wrong dimension names a different Sobol sequence and a wrong width a
            # different period, and either leaves the rebuilt stream unequal to the one
            # the operation built for the same band.
            rebuilt = _bound_bits(polarity, (lo, hi), device)
            for name, bits in (('lo_bits', rebuilt[0]), ('hi_bits', rebuilt[1])):
                built = getattr(operation, name)
                assert torch.equal(built.type(torch.int8), bits.type(torch.int8)), (
                    f'{polarity} clamp_comp built a {name} stream of length '
                    f'{built.numel()} that the rebuild from dim {BOUND_DIM} and width '
                    f'{BOUND_WIDTH} does not reproduce on {device}'
                )
    print('the declared spike dtype and pipeline delay match the selector chain, and '
          'the documented bound defaults rebuild the operation\'s own bound streams.')


def _band_pin_checks():
    """Assert the band places exact spikes and exact per-element spike counts, and that the twin band and the other polarity both move those counts."""
    for device in devices():
        counts = {}
        for polarity in CONFIG['polarities']:
            values = _pin_values(polarity, device)
            band = BOUNDS[polarity]
            # The reference chain is built here from bare max and min selectors and
            # bound streams rebuilt from the documented defaults, so it calls nothing
            # of clamp_comp. It is what the pinned counts below are derived from.
            expected = _chain_outputs(polarity, band, values, PIN_TIMESTEPS, device)
            emitted = _operation_outputs(polarity, band, values, PIN_TIMESTEPS, device)
            assert torch.equal(emitted, expected), (
                f'{polarity} band {band} departed from the selector chain on {device}'
            )
            reference_count = expected.sum(0).type(torch.int64).flatten()
            pinned = torch.tensor(PIN_COUNT[polarity], dtype=torch.int64, device=device)
            assert torch.equal(reference_count, pinned), (
                f'{polarity} reference chain counted {reference_count.tolist()} spikes '
                f'against the pinned {pinned.tolist()} on {device}'
            )
            emitted_count = emitted.sum(0).type(torch.int64).flatten()
            assert torch.equal(emitted_count, pinned), (
                f'{polarity} clamp_comp counted {emitted_count.tolist()} spikes against '
                f'the pinned {pinned.tolist()} on {device}'
            )
            # First wrong source: the same reference chain on the twin band. A count
            # this run reproduces would be one the band never entered.
            twin = _chain_outputs(
                polarity, TWIN_BOUNDS[polarity], values, PIN_TIMESTEPS, device,
            ).sum(0).type(torch.int64).flatten()
            assert not torch.equal(twin, pinned), (
                f'{polarity} twin band {TWIN_BOUNDS[polarity]} reproduced the pinned '
                f'counts {pinned.tolist()} on {device}, so the pin does not name a band'
            )
            counts[polarity] = pinned
            rate = emitted.type(torch.float32).mean(0)
            decoded = rate if polarity == 'unipolar' else 2.0 * rate - 1.0
            reference = torch.clamp(values, band[0], band[1])
            print(
                f'[{device}][{polarity}] band={band} '
                f'max_error={(decoded - reference).abs().max().item():.6f}'
            )
        # Second wrong source: the other polarity, which drives the same grid step
        # indices through its own bounds and its own streams.
        assert not torch.equal(counts['bipolar'], counts['unipolar']), (
            f'both polarities counted {counts["bipolar"].tolist()} spikes on {device}, '
            f'so the pinned counts do not name a polarity'
        )
    print('the band places exact spike counts on every device.')


def _rail_pin_checks():
    """Assert a zero-rate input emits the lo stream and a full-rate input the hi stream, bit for bit, and that neither matches the other bound."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            band = BOUNDS[polarity]
            lo_bits, hi_bits = _bound_bits(polarity, band, device)
            shape = (2, 3)
            for rail, expected_bits, other_bits in (
                (torch.zeros(shape, dtype=torch.int8, device=device), lo_bits, hi_bits),
                (torch.ones(shape, dtype=torch.int8, device=device), hi_bits, lo_bits),
            ):
                emitted = _operation_outputs(
                    polarity, band, None, PIN_TIMESTEPS, device, input_spikes=rail,
                )
                expected = expected_bits[:PIN_TIMESTEPS].reshape(-1, 1, 1).expand_as(emitted)
                assert torch.equal(emitted, expected), (
                    f'{polarity} rail input departed from its bound stream on {device}'
                )
                # The wrong source for the same comparison: the other bound's stream,
                # which the rail output must not match, or matching a bound stream
                # would say nothing about which bound the rail selects.
                other = other_bits[:PIN_TIMESTEPS].reshape(-1, 1, 1).expand_as(emitted)
                assert not torch.equal(emitted, other), (
                    f'{polarity} rail input matched both bound streams on {device}, so '
                    f'the pin does not name a bound'
                )
    print('a saturated input emits its bound stream bit for bit on every device.')


def _box_bands(polarity):
    """Return every band of the documented band box, as value pairs."""
    # A unipolar grid entry sits on an odd multiple of the threshold step, so its
    # top step is one below the bipolar one, which lands on the range end exactly.
    top = GRID_LEN // 2 if polarity == 'bipolar' else GRID_LEN // 2 - 1
    steps = [round(index * top / (BOX_LEN - 1)) for index in range(BOX_LEN)]
    return [
        (_grid_value(polarity, steps[low]), _grid_value(polarity, steps[high]))
        for low in range(BOX_LEN) for high in range(low + 1, BOX_LEN)
    ]


def _off_grid_bands(polarity):
    """Return every band built from the off-grid rates, as value pairs."""
    for rate in OFF_GRID_RATES:
        assert (rate * 2 ** BOUND_WIDTH) % 1.0 != 0.0, (
            f'rate {rate} lands on the 1/{2 ** BOUND_WIDTH} threshold grid, so the '
            f'off-grid sweep would repeat the band box'
        )
    values = [rate if polarity == 'unipolar' else 2.0 * rate - 1.0
              for rate in OFF_GRID_RATES]
    return [(values[low], values[high])
            for low in range(len(values)) for high in range(low + 1, len(values))]


def _worst_band(polarity, bands, values, device):
    """Return the largest decoded error over a band list, and the band carrying it."""
    worst, worst_band = 0.0, None
    for band in bands:
        emitted = _operation_outputs(polarity, band, values, PIN_TIMESTEPS, device)
        rate = emitted.type(torch.float32).mean(0)
        decoded = rate if polarity == 'unipolar' else 2.0 * rate - 1.0
        error = (decoded - torch.clamp(values, band[0], band[1])).abs().max().item()
        if error > worst:
            worst, worst_band = error, band
    return worst, worst_band


def _band_box_report():
    """Print the largest decoded error over the whole legal band box, which is the accuracy the class docstring points at, and the same figure over off-grid bands on its own line."""
    for device in devices():
        for polarity in CONFIG['polarities']:
            values = _grid(polarity, BOX_VALUES).to(device)
            for label, bands in (('band box', _box_bands(polarity)),
                                 ('off-grid', _off_grid_bands(polarity))):
                worst, worst_band = _worst_band(polarity, bands, values, device)
                print(
                    f'[{device}][{polarity}] {label} bands={len(bands)} '
                    f'values={BOX_VALUES} timesteps={PIN_TIMESTEPS} '
                    f'worst_band={worst_band} worst_max_error={worst:.6f}'
                )


def _streams(polarity, dim, values, timesteps):
    """Return the set of spike streams a value list encodes to on one Sobol dimension."""
    encoder = encode({
        'polarity': polarity, 'timestep': timesteps, 'generator': 'sobol', 'dim': dim,
    })
    stacked = torch.stack([encoder(values) for _ in range(timesteps)])
    return {tuple(column.tolist()) for column in stacked.t()}


def _stream_separation_check():
    """Assert no value or bound one polarity drives encodes to a stream the other polarity drives."""
    for timesteps in (CONFIG['timesteps'], PIN_TIMESTEPS):
        driven = {
            polarity: torch.cat((
                make_values(polarity)[0],
                known_answer_case(polarity)[0][0],
                _pin_values(polarity, 'cpu').flatten(),
            ))
            for polarity in CONFIG['polarities']
        }
        shared = (
            _streams('bipolar', INPUT_DIM, driven['bipolar'], timesteps)
            & _streams('unipolar', INPUT_DIM, driven['unipolar'], timesteps)
        )
        assert not shared, (
            f'{len(shared)} input stream(s) are driven by both polarities at '
            f'{timesteps} timesteps'
        )
    # The bound streams do not come from an encoder, so they are compared as the
    # operation builds them, one dimension at a time.
    for index in range(2):
        bipolar = {
            tuple(_bound_bits('bipolar', band, 'cpu')[index].tolist())
            for band in (BOUNDS['bipolar'], TWIN_BOUNDS['bipolar'])
        }
        unipolar = {
            tuple(_bound_bits('unipolar', band, 'cpu')[index].tolist())
            for band in (BOUNDS['unipolar'], TWIN_BOUNDS['unipolar'])
        }
        assert not (bipolar & unipolar), (
            f'a bound stream on dimension {BOUND_DIM + index} is driven by both polarities'
        )
    print('the two polarities drive disjoint spike streams.')


def _rank0_checks():
    """Assert a 0-dim input is carried at rank 1 and keeps the pinned spike count of the same value on the rank-2 grid."""
    steps = [step for row in PIN_STEPS for step in row]
    for device in devices():
        for polarity in CONFIG['polarities']:
            for index in RANK0_INDEX:
                value = torch.tensor(_grid_value(polarity, steps[index]), device=device)
                outputs = _operation_outputs(
                    polarity, BOUNDS[polarity], value, PIN_TIMESTEPS, device,
                )
                assert outputs.shape == (PIN_TIMESTEPS, 1), (
                    f'{polarity} clamp_comp emitted per-timestep shape '
                    f'{tuple(outputs.shape[1:])} on a 0-dim input on {device}, not the '
                    f'rank-1 shape the 0-dim bound bits leave it at'
                )
                count = int(outputs.sum())
                assert count == PIN_COUNT[polarity][index], (
                    f'{polarity} clamp_comp counted {count} spikes on a 0-dim grid step '
                    f'{steps[index]} on {device}, against the pinned '
                    f'{PIN_COUNT[polarity][index]}'
                )
    print('a 0-dim input is carried at rank 1 and keeps the pinned spike count.')


def _assert_rejected(bad):
    try:
        clamp_comp(bad)
    except AssertionError:
        return
    raise AssertionError(f'expected AssertionError for config {bad}')


def _config_validation_checks():
    """Assert that a missing key, an invalid polarity, an out-of-range or inverted band, and a bad dimension are rejected."""
    _assert_rejected({})
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5})
    _assert_rejected({'polarity': 'tripolar', 'lo': -0.5, 'hi': 0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': -1.5, 'hi': 0.5})
    _assert_rejected({'polarity': 'unipolar', 'lo': -0.5, 'hi': 0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': 0.5, 'hi': -0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': 0.5, 'hi': 0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': float('nan'), 'hi': 0.5})
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5, 'dim': 0})
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5, 'dim': 21201})
    _assert_rejected({'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5, 'fracwidth': 8})
    for good in ({'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5},
                 {'polarity': 'unipolar', 'lo': 0.125, 'hi': 0.6875},
                 {'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5, 'dim': 21200},
                 {'polarity': 'bipolar', 'lo': -0.5, 'hi': 0.5, 'name': 'band'}):
        clamp_comp(good)


def test_clamp_comp():
    """Verify clamp_comp against analytic and known-answer streams, pin the band, the rails, the stream separation, and the 0-dim input rank, and reject invalid configs."""
    streaming_suite(CONFIG)
    _rank0_checks()
    _config_validation_checks()


if __name__ == '__main__':
    test_clamp_comp()

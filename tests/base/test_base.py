"""Tests for non-tensor infrastructure dataclasses.

Device/rank, streaming/reset, gradient, and performance gates do not apply.
"""

from dataclasses import FrozenInstanceError

from napl.sim.base import hw_params, pvt_corner, timing


def test_pvt_corner():
    """Verify PVT-corner defaults, overrides, immutable equality, and hashing."""
    default = pvt_corner('asap7', 'ss', 0.7, 125.0)
    assert default == pvt_corner('asap7', 'ss', 0.7, 125.0)
    assert default.rc == 'typ'
    assert default.mode == 'func'
    assert len({default, pvt_corner('asap7', 'ss', 0.7, 125.0)}) == 1

    custom = pvt_corner('sky130', 'ff', 1.8, -40.0, rc='rcworst', mode='scan')
    assert custom == pvt_corner('sky130', 'ff', 1.8, -40.0, 'rcworst', 'scan')
    assert custom != default
    try:
        custom.voltage = 1.9
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError('pvt_corner accepted mutation')


def test_timing():
    """Verify timing defaults and exact custom delay values."""
    assert timing() == timing(cp_delay=0.0, ir_delay=0.0, or_delay=0.0)
    assert timing(1.25, 0.5, 0.75) == timing(
        cp_delay=1.25,
        ir_delay=0.5,
        or_delay=0.75,
    )


def test_hw_params():
    """Verify hardware defaults, corner timing lookup, and independent timing maps."""
    first = hw_params()
    second = hw_params()
    assert first.pp_delay == 0
    assert first.timing == {}
    assert first.timing is not second.timing

    corner = pvt_corner('asap7', 'tt', 0.7, 25.0)
    delays = timing(cp_delay=0.8, ir_delay=0.2, or_delay=0.3)
    custom = hw_params(pp_delay=3, timing={corner: delays})
    assert custom.pp_delay == 3
    assert custom.timing == {corner: delays}
    assert custom.timing[corner] is delays


if __name__ == '__main__':
    test_pvt_corner()
    test_timing()
    test_hw_params()
    print('Test passed.')

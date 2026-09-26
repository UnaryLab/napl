"""Tests for a 0-dim operand with a nonzero storage offset reaching an operation on MPS.

On torch >= 2.14 the MPS backend reads the element a 0-dim operand's nonzero storage
offset selects, for every dtype and for a CPU-resident scalar operand of an MPS tensor.
These tests are the regression guard for that read.
"""

import pytest
import torch

from napl.sim.operation import max_tc


MPS_ONLY = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='no MPS device on this machine'
)

#: Dtypes an MPS 0-dim bitwise operand is checked against. The 8-bit entries are the
#: range ``spike_type`` admits; the wider ones extend the check past it.
MPS_DTYPES = (
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _xor_with_zeros(dtype, device, index=2, scalar_device=None):
    """Return the elementwise XOR of a zero vector with the 0-dim slice ``bits[index]``.

    ``bits`` is ``[1, 0, 0, 0]``, so the default ``index=2`` gives the zero vector when
    the operand reads its storage offset and the one vector when it reads element 0 of
    the same storage. ``index=0`` names element 0 directly, which is what
    ``test_probe_operator_separates_the_two_reads`` compares the default against.
    ``scalar_device`` places ``bits`` on a device other than the zero vector's.
    """
    bits = torch.tensor([1, 0, 0, 0], dtype=dtype, device=scalar_device or device)
    scalar = bits[index]
    assert scalar.storage_offset() == index, 'the probe input is not the intended 0-dim view'
    return (torch.zeros(4, dtype=dtype, device=device) ^ scalar).cpu().tolist()


def test_probe_operator_separates_the_two_reads():
    """Verify the probe itself tells the offset element apart from element 0.

    The dtype-scope probe is only evidence if its operator maps ``bits[2] == 0`` and
    ``bits[0] == 1`` to different outputs. This calls ``_xor_with_zeros`` on CPU, where
    both reads are known correct, so replacing its operator with a non-discriminating
    one (OR against a one operand, AND against a zero operand) fails here.
    """
    for dtype in MPS_DTYPES:
        at_offset = _xor_with_zeros(dtype, 'cpu', 2)
        at_element_0 = _xor_with_zeros(dtype, 'cpu', 0)
        assert at_offset != at_element_0, (
            f'the probe operator does not separate the two candidate reads for {dtype}: '
            f'bits[2] gives {at_offset}, bits[0] gives {at_element_0}'
        )

    print('probe operator separates the two candidate reads for every dtype under test.')
    print('Test passed.')


@MPS_ONLY
def test_offset_scalar_dtype_scope():
    """Verify MPS reads the storage offset of a 0-dim bitwise operand for every dtype.

    Each MPS read is compared against the CPU read of the same probe, so the check
    fails if MPS returns element 0 of the storage instead of the offset element.
    """
    for dtype in MPS_DTYPES:
        cpu_read = _xor_with_zeros(dtype, 'cpu')
        assert cpu_read == torch.zeros(4, dtype=dtype).tolist(), (
            f'CPU regression, not an MPS one: CPU no longer reads the storage offset '
            f'of a 0-dim {dtype} operand: got {cpu_read}'
        )

        mps_read = _xor_with_zeros(dtype, 'mps')
        assert mps_read == cpu_read, (
            f'MPS no longer reads the storage offset of a 0-dim {dtype} operand: got '
            f'{mps_read}, CPU gives {cpu_read}'
        )

        print(f'{dtype}: cpu and mps both read the offset element')

    print('Test passed.')


@MPS_ONLY
def test_offset_cpu_scalar_operand_of_mps_tensor():
    """Verify an offset 0-dim CPU operand of an MPS tensor reads the offset element.

    A CPU-resident 0-dim tensor is a legal operand of an MPS tensor, and it reaches the
    MPS kernel by a path of its own, so it is checked separately from the same-device
    case in ``test_offset_scalar_dtype_scope``.
    """
    cpu_read = _xor_with_zeros(torch.int8, 'cpu')
    mixed_read = _xor_with_zeros(torch.int8, 'mps', scalar_device='cpu')
    assert mixed_read == cpu_read, (
        f'an offset 0-dim CPU operand of an MPS tensor read the wrong storage element: '
        f'got {mixed_read}, CPU gives {cpu_read}'
    )

    print(f'offset 0-dim cpu operand of an mps tensor: {mixed_read}')
    print('Test passed.')


@MPS_ONLY
def test_offset_scalar_operand_on_mps():
    """Verify both entry points carry a sliced 0-dim spike into ``forward()`` unchanged.

    ``max_tc`` ORs its operands against the quiet zero stream, so a read of element 0
    instead of the offset element would turn the sliced value ``bits[2] == 0`` into
    ``bits[0] == 1``. Both ``__call__`` and ``forward_timestep`` are checked.

    ``int8`` is the only dtype this can be written for. ``max_tc`` casts its operands
    with ``.type(torch.int8)`` (``max_tc.py:91``), which materializes fresh storage for
    a ``uint8`` or ``bool`` operand, so those dtypes return the correct answer whatever
    the offset read does and a ``uint8`` or ``bool`` arm here would be vacuous. Only
    ``int8``, where the cast is a no-op, keeps the offset view alive into the operator.
    """
    bits = torch.tensor([1, 0, 0, 0], dtype=torch.int8, device='mps')
    scalar = bits[2]
    assert scalar.storage_offset() == 2, 'the test input is not an offset 0-dim view'

    quiet = torch.zeros(4, dtype=torch.int8, device='mps')
    expected = torch.zeros(4, dtype=torch.int8)

    called = max_tc()(quiet, scalar)
    assert torch.equal(called.cpu(), expected), (
        f'__call__ read the wrong storage element for an offset 0-dim operand: '
        f'got {called.cpu().tolist()}, expected {expected.tolist()}'
    )

    stepped = max_tc().forward_timestep(quiet, scalar)
    assert torch.equal(stepped.cpu(), expected), (
        f'forward_timestep read the wrong storage element for an offset 0-dim operand: '
        f'got {stepped.cpu().tolist()}, expected {expected.tolist()}'
    )

    print(f'offset 0-dim operand: storage_offset {scalar.storage_offset()}, '
          f'__call__ {called.cpu().tolist()}, forward_timestep {stepped.cpu().tolist()}')
    print('Test passed.')


if __name__ == '__main__':
    test_probe_operator_separates_the_two_reads()
    if torch.backends.mps.is_available():
        test_offset_scalar_dtype_scope()
        test_offset_cpu_scalar_operand_of_mps_tensor()
        test_offset_scalar_operand_on_mps()
    else:
        print('no MPS device on this machine; MPS tests skipped.')

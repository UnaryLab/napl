import math

import torch

from napl.sim.base import global_config, napl_base
from napl.sim.operation import wta
from napl.utils._shared_test import devices, streaming_suite


TIMESTEPS = 256
_EDGE_TIMESTEPS = 6
_EDGE_SHAPE = (2, 3)


class first_fall_readout(napl_base):
    """Decode a wta output, which is one spike followed by a latched silence."""


    def __init__(self, config):
        super().__init__(config, ['polarity', 'timestep'], polarity_required=True)
        self.timestep = config['timestep']
        # Timesteps observed before the winning spike; the temporal falling edge
        # of the winner sits at this index of the threshold grid.
        self.spike_count: torch.Tensor
        self.register_buffer('spike_count', torch.zeros(1, dtype=self.ntype))
        self.fired: torch.Tensor
        self.register_buffer('fired', torch.zeros(1, dtype=self.stype))


    def _reset(self):
        """Clear the winner latch and the pre-spike timestep count."""
        self.spike_count.resize_(1).zero_()
        self.fired.resize_(1).zero_()


    def forward(self, spike):
        """Accumulate one wta output timestep."""
        if self.fired.shape != spike.shape:
            self.spike_count.resize_as_(spike).zero_()
            self.fired.resize_as_(spike).zero_()
        self.fired.bitwise_or_(spike.to(self.stype))
        self.spike_count.add_(self.fired.eq(0).to(self.ntype))


    @property
    def spike_value(self):
        """Return the winner value carried by the first output spike."""
        value = self.spike_count.div(2 ** math.ceil(math.log2(self.timestep)))
        if self.polarity == 'bipolar':
            value.mul_(2).sub_(1)
        return value


def make_operation(polarity, timestep, device):
    return wta({'polarity': polarity})


def apply_operation(operation, spikes, values):
    return operation(torch.stack(spikes, dim=-1), dim=-1)


def make_readout(polarity, timestep, device):
    return first_fall_readout({'polarity': polarity, 'timestep': timestep})


def make_values(polarity):
    low = 0.0 if polarity == 'unipolar' else -1.0
    values = torch.linspace(low, 1.0, 128, dtype=global_config.ntype)
    return values, values.roll(31), values.roll(67)


def make_performance_values(polarity):
    low = 0.0 if polarity == 'unipolar' else -1.0
    values = torch.linspace(low, 1.0, 131072, dtype=global_config.ntype)
    return values, values.roll(31), values.roll(67)


def analytic_reference(values, polarity):
    return torch.minimum(torch.minimum(values[0], values[1]), values[2])


def known_answer_case(polarity):
    if polarity == 'unipolar':
        values = (
            torch.tensor([0.0, 0.25, 1.0, 0.75]),
            torch.tensor([1.0, 0.75, 1.0, 0.25]),
            torch.tensor([0.5, 0.50, 1.0, 0.50]),
        )
        expected = torch.tensor([0.0, 0.25, 1.0, 0.25])
    else:
        values = (
            torch.tensor([-1.0, -0.5, 1.0, 0.5]),
            torch.tensor([1.0, 0.5, 1.0, -0.5]),
            torch.tensor([0.0, 0.0, 1.0, 0.0]),
        )
        expected = torch.tensor([-1.0, -0.5, 1.0, -0.5])
    return values, expected, 0.0


def _edge_stream(fall_time, shape=_EDGE_SHAPE):
    """Build a run-of-ones temporal stream with an optional falling edge."""
    stream = torch.ones((_EDGE_TIMESTEPS,) + shape, dtype=global_config.stype)
    if fall_time is not None:
        stream[fall_time:] = 0
    return stream


def _run_edges(operation, streams, device, dim=-1):
    """Run one wta instance over stacked current-timestep input streams."""
    outputs = []
    for timestep in range(_EDGE_TIMESTEPS):
        inputs = torch.stack(
            [stream[timestep].to(device) for stream in streams], dim=dim
        )
        outputs.append(operation(inputs, dim=dim))
    return torch.stack(outputs)


def check_edges():
    """Verify bit-exact falling-edge selection, shapes, devices, and reset."""
    for device in devices():
        # No falling edge at all.
        operation = wta().to(device)
        output = _run_edges(operation, [_edge_stream(None)], device)
        assert torch.equal(output, torch.zeros_like(output))

        # A single stream pulses once at its own falling edge.
        operation = wta().to(device)
        output = _run_edges(operation, [_edge_stream(3)], device)
        expected = torch.zeros_like(output)
        expected[3] = 1
        assert torch.equal(output, expected)

        # The earliest falling edge wins and later edges stay suppressed.
        operation = wta().to(device)
        streams = [_edge_stream(4), _edge_stream(2), _edge_stream(None)]
        output = _run_edges(operation, streams, device)
        expected = torch.zeros_like(output)
        expected[2] = 1
        assert torch.equal(output, expected)

        # Simultaneous earliest edges merge into one shared pulse.
        operation = wta().to(device)
        output = _run_edges(operation, [_edge_stream(2), _edge_stream(2)], device)
        expected = torch.zeros_like(output)
        expected[2] = 1
        assert torch.equal(output, expected)
        assert output[2].sum().item() == _EDGE_SHAPE[0] * _EDGE_SHAPE[1]

        # Shape, dtype, device placement, and replay after reset.
        operation = wta().to(device)
        streams = [_edge_stream(1), _edge_stream(3)]
        first = _run_edges(operation, streams, device)
        assert first.shape == (_EDGE_TIMESTEPS,) + _EDGE_SHAPE
        assert first.dtype == global_config.stype
        assert first.device.type == device
        assert operation.previous.device.type == device
        assert operation.fired.device.type == device
        operation.reset()
        replay = _run_edges(operation, streams, device)
        assert torch.equal(first, replay)
        assert operation.timestep_cur == _EDGE_TIMESTEPS

        # An explicitly selected stack dimension gives the same result.
        operation = wta().to(device)
        output = _run_edges(operation, streams, device, dim=0)
        expected = torch.zeros_like(output)
        expected[1] = 1
        assert torch.equal(output, expected)
        print(f'[{device}] wta falling-edge, shape, device, and reset checks passed')


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    # Threshold encoding is deterministic, so the decoded winner carries no
    # stochastic variance and the bound needs no widening.
    'tolerance_scale': 1.0,
    'make_operation': make_operation,
    'apply_operation': apply_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'make_readout': make_readout,
    'encoder_generators': ['temporal', 'temporal', 'temporal'],
    'extra_checks': check_edges,
    'timesteps': TIMESTEPS,
}


def test_wta():
    """Verify wta selects the smallest temporally encoded value, the one whose falling edge arrives first."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_wta()
    print('Test passed.')

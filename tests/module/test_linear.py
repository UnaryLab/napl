import torch

from napl.sim.base import global_config, napl_base, napl_sim_timesteps
from napl.utils import gen_rand_tensor
from napl.utils._shared_test import devices, streaming_suite, timer
from napl.sim.module import encoder, decoder, linear
from napl.sim.metric import accuracy


class napl_linear(napl_base):
    def __init__(self, codec_config, lin_config, weight, bias):
        super().__init__()
        self.encoder = encoder(codec_config)
        self.decoder = decoder(codec_config)
        self.accuracy = accuracy(codec_config)
        self.linear = linear(weight, bias, lin_config)


    @napl_sim_timesteps
    def forward(self, input_x, timesteps=256):
        i_spike = self.encoder(input_x)
        o_spike = self.linear(i_spike)
        self.decoder(o_spike)
        self.accuracy(o_spike)


def _kernel_specific_checks():
    """
    Streaming unary linear layer reproduces (W x + b) / (in_features + bias) within a
    stochastic-computing error bound.
    """
    timestep = 256
    in_features, out_features = 16, 8

    # Separate Sobol dimensions decorrelate inputs, weights, and biases.
    codec_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 1}
    lin_config = {'polarity': 'bipolar', 'timestep': timestep, 'generator': 'sobol', 'dim': 2, 'scale': None, 'width': 12}

    input_cpu = gen_rand_tensor(
        'bipolar', shape=(in_features,), width=8
    ).type(global_config.ntype)
    weight_cpu = gen_rand_tensor(
        'bipolar', shape=(out_features, in_features), width=8
    ).type(global_config.ntype)
    bias_cpu = gen_rand_tensor(
        'bipolar', shape=(out_features,), width=8
    ).type(global_config.ntype)

    for device in devices():
        input_x = input_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        inst = napl_linear(
            codec_config, lin_config, weight, bias
        ).to(device)
        with timer(device) as elapsed:
            inst(input_x, timesteps=timestep)

        entry = in_features + 1
        r_value = (weight @ input_x + bias) / entry
        err, _ = inst.accuracy.analyze(r_value, verbose=True)
        rmse = torch.sqrt(err.abs().pow(2).mean())
        print(
            f'[{device}] linear rmse={rmse.item():.4f}, '
            f'max={err.abs().max().item():.4f}, time={elapsed.seconds * 1000:.1f}ms'
        )
        assert rmse < 0.05, (device, rmse)
        assert inst.linear.timestep_cur == timestep
        inst.reset()
        assert inst.linear.timestep_cur == 0
        assert inst.accuracy.timestep_cur == 0

    print('Test passed.')


def _suite_weight(polarity):
    if polarity == 'unipolar':
        return torch.tensor([
            [0.25, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.25],
        ])
    return torch.tensor([
        [-0.75, -0.25, 0.25, 0.75],
        [0.75, 0.25, -0.25, -0.75],
    ])


def make_operation(polarity, timestep, _device):
    return linear(
        _suite_weight(polarity), None,
        {
            'polarity': polarity,
            'timestep': timestep,
            'generator': 'sobol',
            'dim': 2,
            'scale': None,
            'width': 12,
        },
    )


def make_values(polarity):
    if polarity == 'unipolar':
        return (torch.tensor([0.0, 0.25, 0.75, 1.0]),)
    return (torch.tensor([-1.0, -0.25, 0.25, 1.0]),)


def make_performance_values(polarity):
    values = make_values(polarity)[0]
    return (values.repeat(32768, 1),)


def analytic_reference(values, polarity):
    return _suite_weight(polarity) @ values[0] / 4


def known_answer_case(polarity):
    values = torch.ones(4)
    return (
        (values,),
        _suite_weight(polarity) @ values / 4,
        4.0 / (256 ** 0.5),
    )


CONFIG = {
    'polarities': ['unipolar', 'bipolar'],
    'tolerance_scale': 4.0,
    'make_operation': make_operation,
    'make_values': make_values,
    'make_performance_values': make_performance_values,
    'analytic_reference': analytic_reference,
    'known_answer_case': known_answer_case,
    'timesteps': 256,
    'extra_checks': _kernel_specific_checks,
}


def test_linear():
    """Verify linear for both polarities against analytic and known-answer streams."""
    streaming_suite(CONFIG)


if __name__ == '__main__':
    test_linear()

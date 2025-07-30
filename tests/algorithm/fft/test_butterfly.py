import torch, math

from napl.base import global_config
from napl.utils import *
from napl.algorithm.fft.butterfly import butterfly_spike, butterfly_binary
from napl.metric.accuracy import report_error


def test_butterfly_spike():
    """
    Test the bfu_1_add forward method with random inputs.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    codec_config = {
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    width = math.log2(codec_config['timestep'])
    mul_config = {
        'mode': 'bipolar',
        'timestep': 256,
        'generator': 'sobol',
    }
    add_config = {
        'mode': 'bipolar',
        'scale': 3,
        'width' : width+1,
    }
    acc_config = codec_config

    batch_size = 1024

    # Generate random input tensors
    x0r = gen_rand_tensor(codec_config['mode'], shape=(batch_size, 1), width=width).type(global_config.ntype).to(device)
    x0i = gen_rand_tensor(codec_config['mode'], shape=(batch_size, 1), width=width).type(global_config.ntype).to(device)
    x1r = gen_rand_tensor(codec_config['mode'], shape=(batch_size, 1), width=width).type(global_config.ntype).to(device)
    x1i = gen_rand_tensor(codec_config['mode'], shape=(batch_size, 1), width=width).type(global_config.ntype).to(device)
    wr  = gen_rand_tensor(codec_config['mode'], shape=(batch_size, 1), width=width).type(global_config.ntype).to(device)
    wi  = gen_rand_tensor(codec_config['mode'], shape=(batch_size, 1), width=width).type(global_config.ntype).to(device)

    # Instantiate reference butterfly unit
    butterfly_binary_unit = butterfly_binary()
    # y0r_ref, y0i_ref, y1r_ref, y1i_ref, wr_x1r_ref, wr_x1i_ref, wi_x1r_ref, wi_x1i_ref = butterfly_binary_unit(x0r, x0i, x1r, x1i, wr, wi)
    y0r_ref, y0i_ref, y1r_ref, y1i_ref = butterfly_binary_unit(x0r, x0i, x1r, x1i, wr, wi)

    # Instantiate the butterfly unit
    butterfly_spike_unit = butterfly_spike(codec_config, mul_config, add_config, acc_config).to(device)

    # Run forward pass
    y0r_value, y0i_value, y1r_value, y1i_value = butterfly_spike_unit(x0r, x0i, x1r, x1i, wr, wi, timesteps=codec_config['timestep'])

    report_error(y0r_value, y0r_ref / add_config['scale'])
    report_error(y0i_value, y0i_ref / add_config['scale'])
    report_error(y1r_value, y1r_ref / add_config['scale'])
    report_error(y1i_value, y1i_ref / add_config['scale'])


    # Reset and see whether results are the same
    butterfly_spike_unit.reset()

    # Run forward pass
    y0r_value, y0i_value, y1r_value, y1i_value = butterfly_spike_unit(x0r, x0i, x1r, x1i, wr, wi, timesteps=codec_config['timestep'])

    report_error(y0r_value, y0r_ref / add_config['scale'])
    report_error(y0i_value, y0i_ref / add_config['scale'])
    report_error(y1r_value, y1r_ref / add_config['scale'])
    report_error(y1i_value, y1i_ref / add_config['scale'])

    print('Test passed.')


if __name__ == '__main__':
    test_butterfly_spike()


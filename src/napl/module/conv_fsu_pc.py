import torch

from napl.utils import *
from napl.base import napl_base
from loguru import logger


class conv_fsu_pc(napl_base):
    """
    Streaming (FSU) unary conv2d *parallel counter*: the per-timestep binary inner-product
    count of the im2col'd input spikes against freshly encoded weight spikes, before any
    accumulation into a bitstream. This is the `conv_fsu` partial sum without its scaled
    unary adder, the conv counterpart of `linear_fsu_pc`.

    Each timestep the weights (and bias) are encoded into spikes on a distinct RNG dimension
    from the input (decorrelated operands), the input is unfolded to patches, and the spike
    product is counted: unipolar returns the AND-count sum(input & weight) (+ bias spike),
    bipolar the XNOR-count sum(input == weight) (+ bias spike, added on the input-1 path
    only, matching FSUConv2dPC). The count is folded back to NCHW. Per timestep the count
    per output element lies in [0, entry] with entry = in*kh*kw + has_bias; accumulating the
    count over T timesteps and dividing by T recovers the unipolar conv directly, or the
    bipolar conv as 2*mean - entry. Bipolar zero-padding uses a decorrelated rate-0.5 pad
    stream (a separate pad encoder), not a deterministic toggle. References: uGEMM.
    UnarySim: FSUConv2dPC. groups=1, zero padding only.
    """
    def __init__(self, weight, bias=None, stride=1, padding=0, dilation=1,
                 config={'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol', 'dim': 2}):
        super().__init__(config, ['polarity', 'timestep', 'generator'], polarity_required=True)
        # lazy import: operation.mul imports module.encoder, so importing at module top would
        # create an import cycle with module/__init__.
        from napl.module.encoder import encoder

        assert weight.dim() == 4, logger.error(f'conv_fsu_pc weight must be 4D (out,in,kh,kw), got {tuple(weight.shape)}.')
        self.weight = weight
        self.out_channels, self.in_channels = weight.shape[0], weight.shape[1]
        self.kernel_size = (weight.shape[2], weight.shape[3])
        self.stride, self.dilation = stride, dilation
        self.padding = num2tuple(padding)
        self.has_bias = bias is not None
        self.weight_flat = weight.view(self.out_channels, -1)                 # (out, K)
        self.K = self.weight_flat.shape[1]                                    # in*kh*kw
        self.entry = self.K + (1 if self.has_bias else 0)

        dim = config.get('dim', 2)
        if config['generator'].lower() not in ['sobol', 'rc', 'rate']:
            logger.warning(
                f'conv_fsu_pc decorrelates operands via distinct sobol dimensions, but generator '
                f'<{config["generator"]}> does not decorrelate by dim. Use a sobol-family generator.')

        cfg = {'polarity': self.polarity, 'timestep': config['timestep'], 'generator': config['generator']}
        self.w_encoder = encoder({**cfg, 'dim': dim})
        if self.has_bias:
            self.bias = bias
            self.b_encoder = encoder({**cfg, 'dim': dim + 1})
        # decorrelated rate-0.5 pad stream for bipolar zero-padding (mirrors conv_fsu)
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            self.pad_encoder = encoder({**cfg, 'dim': dim + 2})
            # The pad bit feeds F.pad's scalar `value`, so it must be a Python float; precompute
            # the full rate-0.5 period to plain floats once to avoid a per-step device->host sync.
            self.pad_len = self.pad_encoder.len
            pad_seq = torch.gt(torch.tensor(0.5, dtype=self.ntype),
                               self.pad_encoder.num_seq.detach()).type(self.stype)
            self.pad_bits = [float(b) for b in pad_seq.tolist()]

    def reset(self, verbose=False):
        self.timestep_cur = 0
        self.w_encoder.reset()
        if self.has_bias:
            self.b_encoder.reset()
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            self.pad_encoder.reset()

    def forward(self, input_spike):
        # input_spike: (batch, in_channels, H, W) spike tensor for the current timestep
        self.tick()
        ph, pw = self.padding
        out_hw = conv2d_output_shape((input_spike.size(2), input_spike.size(3)), kernel_size=self.kernel_size,
                                     dilation=self.dilation, pad=self.padding, stride=self.stride)
        # unfold/fold are not implemented for int8 spikes; cast to float (spikes are 0/1, exact)
        xf = input_spike.type(self.ntype)
        if self.polarity == 'bipolar' and self.padding != (0, 0):
            # pad with a decorrelated rate-0.5 spike (bipolar 0), then unfold with no extra padding
            pad_bit = self.pad_bits[(self.timestep_cur - 1) % self.pad_len]
            xf = torch.nn.functional.pad(xf, (pw, pw, ph, ph), value=pad_bit)
            im2col = torch.nn.functional.unfold(xf, self.kernel_size, self.dilation, 0, self.stride)
        else:
            im2col = torch.nn.functional.unfold(xf, self.kernel_size, self.dilation, self.padding, self.stride)
        inp = im2col.transpose(1, 2).reshape(-1, im2col.size(1))              # (batch*L, K)
        w_spike = self.w_encoder(self.weight_flat)                           # (out, K)
        wf = w_spike.type(self.ntype)
        # AND-count of the input-1 path, without materializing the (batch*L, out, K) product
        pc = torch.matmul(inp, wf.t())                                       # (batch*L, out)
        if self.has_bias:
            # bias spike joins the input-1 path only (matches FSUConv2dPC bias placement)
            pc = pc + self.b_encoder(self.bias).type(self.ntype)
        if self.polarity == 'bipolar':
            # XNOR-count: add the input-0 path sum((1-input)&(1-weight)) to the input-1 path
            pc = pc + torch.matmul(1 - inp, (1 - wf).t())
        out = pc.reshape(input_spike.size(0), -1, pc.size(-1)).transpose(1, 2)  # (batch, out, L)
        return torch.nn.functional.fold(out, out_hw, (1, 1))                 # (batch, out, H, W)

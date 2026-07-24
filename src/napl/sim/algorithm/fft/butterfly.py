import torch

from napl.sim.base import napl_base, napl_sim_timesteps
from napl.utils import *
from napl.sim.module import encoder, decoder
from napl.sim.metric import accuracy
from napl.sim.operation import mul_csg, add_any


class butterfly_spike(napl_base):
    def __init__(
            self,
            codec_config,
            mul_config,
            add_config,
            acc_config,
        ):
        super().__init__()

        # butterfly equation
        # y0r = x0r + (wr * x1r - wi * x1i)
        # y0i = x0i + (wr * x1i + wi * x1r)
        # y1r = x0r - (wr * x1r - wi * x1i)
        # y1i = x0i - (wr * x1i + wi * x1r)

        # One batched instance per stage: the four parallel lanes are stacked
        # along dim 0. Bit-identical to four separate instances because every
        # submodule's state/RNG is elementwise (encoder threshold is a
        # per-timestep scalar; mul_csg seq_idx, add_any accumulator, and
        # decoder count are per-element) and all lanes share the same config.
        self.encoder_x = encoder(codec_config)
        self.decoder_y = decoder(codec_config)
        self.accuracy_y = accuracy(acc_config)
        self.mul_wx = mul_csg(mul_config)
        self.add_y = add_any(add_config)

        # cache of input-derived stacked tensors: inputs are constant across
        # timesteps, so build once and reuse (keyed on id/version/shape)
        self._stack_cache = None

    def _reset(self):
        self._stack_cache = None

    def _stacks(self, x0r, x0i, x1r, x1i, wr, wi):
        key = tuple((t.data_ptr(), t._version, t.shape) for t in (x0r, x0i, x1r, x1i, wr, wi))
        if self._stack_cache is not None and self._stack_cache[0] == key:
            return self._stack_cache[1:]
        shape = torch.broadcast_shapes(x0r.shape, x0i.shape, x1r.shape, x1i.shape, wr.shape, wi.shape)
        x0r, x0i, x1r, x1i, wr, wi = (t.expand(shape) for t in (x0r, x0i, x1r, x1i, wr, wi))
        x_stack = torch.cat([x0r, x0i, x1r, x1i], 0)  # encoder input
        w_stack = torch.cat([wr, wr, wi, wi], 0)      # mul weights for [x1r, x1i, x1r, x1i]
        b = shape[0]
        # per-lane constants over the [r; i] half-stack, broadcast over trailing dims
        tail = (1,) * (len(shape) - 1)
        sign = torch.cat([torch.full((b,) + tail, -1), torch.full((b,) + tail, 1)]).type(self.stype).to(x_stack.device)
        bias0 = sign.eq(-1).type(self.stype)                # [1; 0]
        bias1 = torch.cat([bias0.narrow(0, 0, b), bias0.narrow(0, 0, b) + 1])  # [1; 2]
        self._stack_cache = (key, x_stack, w_stack, sign, bias0, bias1, b)
        return x_stack, w_stack, sign, bias0, bias1, b

    @napl_sim_timesteps
    def forward(self, x0r, x0i, x1r, x1i, wr, wi):
        # all inputs and outputs are binary tensors
        x_stack, w_stack, sign, bias0, bias1, b = self._stacks(x0r, x0i, x1r, x1i, wr, wi)

        # encode [x0r; x0i; x1r; x1i] in one shot
        x_spike = self.encoder_x(x_stack)
        x0_spike = x_spike.narrow(0, 0, 2 * b)   # [x0r; x0i]
        x1_spike = x_spike.narrow(0, 2 * b, 2 * b)

        # w * x1 in one shot: lanes [wr*x1r; wr*x1i; wi*x1r; wi*x1i]
        m = self.mul_wx(torch.cat([x1_spike, x1_spike], 0), w_stack)
        m01 = m.narrow(0, 0, 2 * b)                                        # [wr*x1r; wr*x1i]
        m32 = torch.cat([m.narrow(0, 3 * b, b), m.narrow(0, 2 * b, b)], 0)  # [wi*x1i; wi*x1r]

        # twiddle term t = [wr*x1r - wi*x1i; wr*x1i + wi*x1r]; the inverted-spike
        # (1 - s) offsets from the original per-lane sums fold into bias0/bias1,
        # so each lane's spike sum (in {0..3}) is bit-identical to the unfused form
        t = m01 + sign * m32
        y0_sum = x0_spike + t + bias0
        y1_sum = x0_spike - t + bias1

        # three-input scaled add + decode, all four lanes at once
        y_spike = self.add_y(torch.cat([y0_sum, y1_sum], 0), entry=3, dim=None)
        y = self.decoder_y(y_spike)
        self.accuracy_y(y_spike)

        return y.narrow(0, 0, b), y.narrow(0, b, b), y.narrow(0, 2 * b, b), y.narrow(0, 3 * b, b)


class butterfly_binary(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x0r, x0i, x1r, x1i, wr, wi):
        # butterfly equation
        t_r = wr * x1r - wi * x1i
        t_i = wr * x1i + wi * x1r

        y0r = x0r + t_r
        y0i = x0i + t_i
        y1r = x0r - t_r
        y1i = x0i - t_i

        return y0r, y0i, y1r, y1i

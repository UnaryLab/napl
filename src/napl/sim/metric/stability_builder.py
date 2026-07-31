import torch
import math

from loguru import logger
from napl.sim.base import napl_base
from napl.sim.metric.stability_norm import search_max_stab
from napl.sim.module.encoder import gen_num_seq


class stability_builder(napl_base):
    """
    Generate a spike stream with a requested normalized stability.

    Use this builder to create controlled stability inputs for experiments. It
    emits an unstable prefix followed by a stable tail while preserving the
    configured source value over the designed stream length.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import stability_builder

        builder = stability_builder(torch.tensor([0.0]))
        spike = builder()

    .. container:: api-references

        .. rubric:: References

        *Normalized Stability: A Cross-Level Design Metric for Early Termination in Stochastic Computing*.
    """
    def __init__(
            self,
            source,
            config:dict={
                'polarity': 'bipolar',
                'threshold': 0.05,
                'normstability': 0.5,
                'timestep': 256,
                'generator': 'sobol',
                }
        ):
        """
        Configure the source value, stream length, and target stability.

        Construction performs the stability search on CPU. Move the constructed
        module to its execution device with :meth:`torch.nn.Module.to`. If
        ``config`` is supplied, it must contain ``polarity``, ``threshold``,
        ``normstability``, ``timestep``, and ``generator``.

        .. container:: api-parameter-list

            **Parameters:**

            - **source** – Tensor of values to encode. Use ``[0, 1]`` for
              unipolar encoding or ``[-1, 1]`` for bipolar encoding.
            - **config** – Configuration mapping. The defaults are shown below.

              - **polarity**: Stream encoding, either ``"unipolar"`` or
                ``"bipolar"``; the default is ``"bipolar"``.
              - **threshold**: Absolute decoded-value tolerance used to define
                stability; the default is ``0.05``.
              - **normstability**: Requested normalized stability; the default
                is ``0.5``.
              - **timestep**: Positive designed stream length; the default is
                ``256``.
              - **generator**: Number-sequence generator. Accepted values are
                ``"sobol"``, ``"lfsr"``, ``"sys"``, ``"rc"``, ``"tc"``,
                ``"rate"``, and ``"temporal"``; the default is ``"sobol"``.
              - **dim**: Sobol dimension; the default is ``1``.
              - **name**: Optional instance label; the default is ``None``.
        """
        super().__init__(config, ['polarity', 'threshold', 'normstability', 'timestep', 'generator'], polarity_required=True)

        self.timestep = config['timestep']
        assert self.timestep > 0, logger.error(f'Invalid timestep: <{self.timestep}>; legal values: a positive integer.')
        self.width = math.ceil(math.log2(self.timestep))
        self.normstability = config['normstability']

        # the build-time search below runs on CPU (torch.gcd on integers); move
        # the module to the target device after construction for streaming.
        source = source.detach().to('cpu', self.ntype)
        seq_len = 2**self.width
        if self.polarity == 'bipolar':
            val = (source + 1) / 2
            threshold = config['threshold'] / 2
        else:
            val = source
            threshold = config['threshold']

        # per-element value band [p_low, p_up] allowed by the threshold
        p_low = torch.max(val - threshold, torch.zeros_like(val))
        p_up = torch.min(torch.ones_like(val), val + threshold)
        lower = torch.max(torch.floor(seq_len * p_low), torch.zeros_like(val))
        upper = torch.min(torch.ceil(seq_len * p_up), torch.full_like(val, seq_len))
        search_range = int(threshold * 2 * seq_len + 1)

        max_stable, _, _ = search_max_stab(lower.to(torch.int64), upper.to(torch.int64),
                                           seq_len, search_range)
        max_stable = max_stable.to(val.dtype)

        # split the stream: unstable prefix of new_ns_len spikes, stable tail
        max_st_len = seq_len - max_stable
        new_st_len = torch.ceil(max_st_len * self.normstability)
        new_ns_len = seq_len - new_st_len

        # one-count per segment: the prefix parks the value just outside the
        # band on the far side, the tail supplies the remaining ones
        val_gt_half = (val > 0.5).to(val.dtype)
        new_ns_one = val_gt_half * (p_up * (new_ns_len + 1)) \
                   + (1 - val_gt_half) * torch.max(p_low * (new_ns_len + 1) - 1, torch.zeros_like(val))
        new_st_one = val * seq_len - new_ns_one

        # per-segment stream values, quantized to seq_len levels for comparison
        # against the integer-valued number sequence
        src_ns = (new_ns_one / new_ns_len).mul(seq_len).round()
        src_st = (new_st_one / new_st_len).mul(seq_len).round()

        num_seq = gen_num_seq(config={'width': self.width, 'generator': config['generator'], 'dim': config.get('dim', 1)})
        self.register_buffer('num_seq', num_seq.mul(seq_len).floor())
        self.register_buffer('src_ns', src_ns)
        self.register_buffer('src_st', src_st)
        self.register_buffer('new_ns_len', new_ns_len)
        # per-segment spike counters, doubling as indices into num_seq
        self.register_buffer('out_cnt_ns', torch.zeros_like(val, dtype=torch.long))
        self.register_buffer('out_cnt_st', torch.zeros_like(val, dtype=torch.long))


    def _reset(self):
        """
        Reset both class-local segment counters to zero.

        The generated sequence and segment values remain unchanged. The inherited
        reset method resets the timestep before calling this hook. This hook
        returns ``None``.
        """
        self.out_cnt_ns.zero_()
        self.out_cnt_st.zero_()


    def forward(self):
        """
        Emit the next spike from the designed stream.

        Returns:
            A spike tensor with the source shape and the global spike dtype.

        Calling the builder increments its timestep and advances either the
        unstable-prefix counter or the stable-tail counter for each element.

        **Example:**

        .. code-block:: python

            spike = builder()
        """
        in_prefix = self.out_cnt_ns < self.new_ns_len
        # select the active segment first, then gather/compare once (exactly
        # equivalent to comparing both segments and where-selecting the spikes)
        src = torch.where(in_prefix, self.src_ns, self.src_st)
        cnt = torch.where(in_prefix, self.out_cnt_ns, self.out_cnt_st)
        spike = torch.gt(src, self.num_seq[cnt]).type(self.stype)
        # in-place is safe: counters are full-shape longs from __init__ (no
        # scalar broadcast), long += bool involves no dtype promotion change
        self.out_cnt_ns.add_(in_prefix)
        self.out_cnt_st.add_(~in_prefix)
        return spike

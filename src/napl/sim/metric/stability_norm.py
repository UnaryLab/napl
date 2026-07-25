import torch

from napl.sim.base import napl_base
from napl.sim.metric._shared import analyze
from napl.sim.metric.stability import stability
from loguru import logger


def search_max_stab(p_low_L, p_high_L, L, search_range):
    """
    For each element, search the probability grid [p_low_L, p_high_L] (out of L) for the
    segmented-uniform stream with the longest stable tail, returning the shortest
    unstable prefix length, its repetition count R, and its period l_p. Shared torch port
    of UnarySim's search_best_stab_parallel_numpy, in float32 arithmetic; each iteration
    overwrites (not keeps) the running best exactly as the original does, so the
    iteration count must match the reference too.
    p_low_L/p_high_L are integer CPU tensors; L (power of 2) and search_range are ints.
    """
    L_t = torch.full_like(p_low_L, L)
    max_stab_len = torch.full(p_low_L.shape, float(L), dtype=torch.float32)
    max_stab_R = torch.ones_like(max_stab_len)
    max_stab_l_p = torch.ones_like(max_stab_len)
    p_low = p_low_L.float()
    p_high = p_high_L.float()
    # i-invariant parts of the padding terms, for all-0s (B_L=0) and all-1s (B_L=L)
    # padding: the out-of-window penalty and the distance numerators.
    pads = []
    for B_L in (0.0, float(L)):
        low_pen = (1 - (p_low <= B_L).float()) * L
        high_pen = (1 - (B_L <= p_high).float()) * L
        pads.append((low_pen, p_low - B_L, high_pen, B_L - p_high))
    # Batch the candidate loop into (rows, N) 2D chunks: one kernel per op across many
    # candidates instead of ~30 tiny launches per i. All ops are elementwise float32,
    # so the values are bit-identical to the per-i loop; only the running-best scan is
    # inherently sequential (the reference overwrites, not keeps, the best) and stays a
    # cheap per-row loop. ponytail: chunk rows to cap peak memory at O(rows*N) floats.
    rows = max(1, min(search_range + 1, (1 << 20) // max(1, p_low_L.numel())))
    for start in range(0, search_range + 1, rows):
        i_idx = torch.arange(start, min(start + rows, search_range + 1),
                             dtype=p_low_L.dtype).unsqueeze(1)
        p_L = torch.minimum(p_low_L + i_idx, p_high_L)
        # gcd with L (a power of 2) is the largest power of 2 dividing p_L,
        # i.e. p_L & -p_L, capped by p_L <= L; 0 maps to L. Same values as
        # torch.gcd(p_L, L_t) but without its per-element Euclid loop.
        gcd = p_L & p_L.neg()
        l_p = L / torch.where(gcd == 0, L_t, gcd).float()
        denom_low = l_p * i_idx.float()
        denom_high = l_p * (p_high - p_L.float())
        denom_low[denom_low == 0] = 1
        denom_high[denom_high == 0] = 1
        p_L_eq_low = (p_L == p_low_L).float()
        p_L_eq_high = (p_high_L == p_L).float()
        # R needed so the tail stays in-threshold when padded with all-0s (B_L=0) or
        # all-1s (B_L=L); take the cheaper padding.
        R_pad = []
        for low_pen, num_low, high_pen, num_high in pads:
            R_low = p_L_eq_low * low_pen + (1 - p_L_eq_low) * num_low / denom_low
            R_high = p_L_eq_high * high_pen + (1 - p_L_eq_high) * num_high / denom_high
            R_pad.append(torch.ceil(torch.maximum(R_low, R_high)))
        R = torch.minimum(R_pad[0], R_pad[1])
        R_l_p = R * l_p
        R_l_p_c = R_l_p.clamp(min=1)
        # Vectorized form of the reference's sequential overwrite scan. R_l_p >= 0
        # (R is a min of ceils of non-negative maxima given p_high_L <= L, and
        # l_p >= 1), so a failed compare zeroes the state and it stays zero
        # (nothing is ever < 0). The final state therefore equals the last row's
        # values iff every row's compare succeeded: row 0 against the carried
        # state, row j against R_l_p_c[j-1]. Bit-identical to the per-row loop.
        alive = (R_l_p < torch.cat((max_stab_len.unsqueeze(0), R_l_p_c[:-1]))).all(dim=0).float()
        max_stab_len = alive * R_l_p_c[-1]
        max_stab_R = alive * R[-1]
        max_stab_l_p = alive * l_p[-1]
    return max_stab_len, max_stab_R, max_stab_l_p


class stability_norm(napl_base):
    """
    Normalized, value-independent stability of a spike stream: actual stability (from the
    stability metric) over the maximum stability achievable for the source value at this
    stream length, per element in [0, 1]. Call forward(spike) once per timestep, then
    analyze(). Reference: "Normalized Stability: A Cross-Level Design Metric for
    Early Termination in Stochastic Computing".
    """
    def __init__(
            self,
            source,
            config={
                'polarity': 'bipolar',
                'threshold': 0.05,
            }
        ):
        super().__init__(config, ['polarity', 'threshold'], polarity_required=True)

        self.register_buffer('source', source)
        self.threshold = config['threshold']
        # inner actual-stability monitor
        self.stability = stability(source, {'polarity': self.polarity, 'threshold': self.threshold})
        if self.polarity == 'bipolar':
            prob = (source + 1) / 2
            half = self.threshold / 2
        else:
            prob = source
            half = self.threshold
        # in-threshold probability window of the source value
        self.min_prob = torch.nn.Parameter((prob - half).clamp(min=0), requires_grad=False)
        self.max_prob = torch.nn.Parameter((prob + half).clamp(max=1), requires_grad=False)

    def forward(self, spike):
        self.stability(spike)
        # no return: readers access .stability_norm on demand.


    @property
    def stability_norm(self):
        """
        Normalized stability, computed on access from the accumulated state.
        Returns a fresh tensor; before any forward() it is the zeros seed.
        """
        if not self.valid:
            return torch.zeros_like(self.source)
        timestep = self.stability.accuracy.timestep_cur
        # float32 tensor math mirrors UnarySim exactly (L = next power of 2 >= timestep)
        len_t = torch.tensor([float(timestep)])
        L_t = torch.pow(2, torch.ceil(torch.log2(len_t)))
        L = int(L_t.item())
        # torch.gcd in the search is CPU-only territory; the search runs once on CPU
        p_low_L = torch.floor(self.min_prob.cpu() * L_t).clamp(0, L).to(torch.int64)
        p_high_L = torch.ceil(self.max_prob.cpu() * L_t).clamp(0, L).to(torch.int64)
        # float32 threshold arithmetic, truncated: matches the reference's search width
        search_range = int((torch.tensor([self.threshold], dtype=torch.float32) * 2 * L_t + 1).item())
        max_stab_len, _, _ = search_max_stab(p_low_L, p_high_L, L, search_range)
        max_stab = (1 - max_stab_len / float(timestep)).clamp(min=0).to(self.source.device)

        norm = self.stability.stability / max_stab
        norm[torch.isnan(norm)] = 0
        # segmented-uniform max is an approximation of the best case, so the ratio can
        # exceed 1; clamp like the reference
        return norm.clamp_(0, 1)


    def analyze(self, verbose=False):
        # return the normalized stability and index of max abs normalized stability
        assert self.valid, logger.error('Metric is not valid. Please call forward() before analyze().')
        # one property access: stability_norm computes from the accumulated state on each read
        stability_norm = self.stability_norm
        result = analyze(
            stability_norm,
            verbose=verbose,
            report='Normalized stability',
            value='normalized stability',
            timestep=self.timestep_cur,
        )
        self.stability_norm_abs_max = result.absolute_max
        self.stability_norm_abs_min = result.absolute_min
        self.stability_norm_avg = result.mean
        self.stability_norm_mae = result.mean_absolute
        self.stability_norm_rmse = result.root_mean_square

        return stability_norm, result.max_absolute_index

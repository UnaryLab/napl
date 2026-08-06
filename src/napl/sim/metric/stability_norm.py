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
    pads = []
    for B_L in (0.0, float(L)):
        low_pen = (1 - (p_low <= B_L).float()) * L
        high_pen = (1 - (B_L <= p_high).float()) * L
        pads.append((low_pen, p_low - B_L, high_pen, B_L - p_high))
    rows = max(1, min(search_range + 1, (1 << 20) // max(1, p_low_L.numel())))
    for start in range(0, search_range + 1, rows):
        i_idx = torch.arange(start, min(start + rows, search_range + 1),
                             dtype=p_low_L.dtype).reshape(
                                 (-1,) + (1,) * p_low_L.ndim)
        p_L = torch.minimum(p_low_L + i_idx, p_high_L)
        # For power-of-two L, p_L & -p_L is gcd(p_L, L); zero maps to L.
        gcd = p_L & p_L.neg()
        l_p = L / torch.where(gcd == 0, L_t, gcd).float()
        denom_low = l_p * i_idx.float()
        denom_high = l_p * (p_high - p_L.float())
        denom_low[denom_low == 0] = 1
        denom_high[denom_high == 0] = 1
        p_L_eq_low = (p_L == p_low_L).float()
        p_L_eq_high = (p_high_L == p_L).float()
        # R is the smaller repetition count for all-zero or all-one padding.
        R_pad = []
        for low_pen, num_low, high_pen, num_high in pads:
            R_low = p_L_eq_low * low_pen + (1 - p_L_eq_low) * num_low / denom_low
            R_high = p_L_eq_high * high_pen + (1 - p_L_eq_high) * num_high / denom_high
            R_pad.append(torch.ceil(torch.maximum(R_low, R_high)))
        R = torch.minimum(R_pad[0], R_pad[1])
        R_l_p = R * l_p
        R_l_p_c = R_l_p.clamp(min=1)
        # Since R_l_p is nonnegative, one failed comparison zeros all later state.
        alive = (R_l_p < torch.cat((max_stab_len.unsqueeze(0), R_l_p_c[:-1]))).all(dim=0).float()
        max_stab_len = alive * R_l_p_c[-1]
        max_stab_R = alive * R[-1]
        max_stab_l_p = alive * l_p[-1]
    return max_stab_len, max_stab_R, max_stab_l_p


class stability_norm(napl_base):
    r"""
    Measure value-independent normalized stability of a spike stream.

    This metric divides observed stability by the estimated maximum stability for
    the source value and current stream length. Use it to compare streams whose
    encoded values have different best-case convergence behavior.

    The target divides observed stability :math:`S_T` by the best stability
    :math:`S^{\max}_T` any stream of the same length can reach for the same
    source value and threshold,

    .. math::

       N_T = \frac{S_T}{S^{\max}_T}.

    The maximum is approximated over segmented-uniform streams only, using the
    shortest unstable prefix :math:`\ell` among those whose rate stays within
    the threshold band around the encoded source probability,

    .. math::

       S^{\max}_T \approx \max\left(1 - \frac{\ell}{T},\, 0\right).

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import stability_norm

        metric = stability_norm(torch.ones(1))
        for _ in range(2):
            metric(torch.ones(1))
        value, result = metric.analyze()

    .. container:: api-references

        .. rubric:: References

        *Normalized Stability: A Cross-Level Design Metric for Early Termination in Stochastic Computing*, ASP-DAC, 2021.
    """


    def __init__(
            self,
            source,
            config={
                'polarity': 'bipolar',
                'threshold': 0.05,
            }
        ):
        """
        Configure the source value and stability threshold.

        If ``config`` is supplied, it must contain both ``polarity`` and
        ``threshold``.

        .. container:: api-parameter-list

            **Parameters:**

            - **source** – Tensor of expected decoded values. Use values in
              ``[0, 1]`` for unipolar streams or ``[-1, 1]`` for bipolar streams.
            - **config** – Configuration mapping. The default is
              ``{'polarity': 'bipolar', 'threshold': 0.05}``.

              - **polarity**: Stream encoding, either ``"unipolar"`` or
                ``"bipolar"``; the default is ``"bipolar"``.
              - **threshold**: Maximum absolute progressive error considered
                stable; the default is ``0.05``.
              - **name**: Optional instance label; the default is ``None``.
        """
        super().__init__(config, ['polarity', 'threshold'], polarity_required=True)

        #: Expected decoded value used as the per-element normalization reference.
        self.source: torch.Tensor
        self.register_buffer('source', source)
        #: Maximum absolute progressive error treated as stable.
        self.threshold = config['threshold']
        #: Child metric that measures the observed stream's unnormalized stability.
        self.stability = stability(source, {'polarity': self.polarity, 'threshold': self.threshold})
        if self.polarity == 'bipolar':
            prob = (source + 1) / 2
            half = self.threshold / 2
        else:
            prob = source
            half = self.threshold
        #: Lower encoded-probability bound accepted as stable for each source element.
        self.min_prob: torch.Tensor
        self.register_buffer('min_prob', (prob - half).clamp(min=0).detach())
        #: Upper encoded-probability bound accepted as stable for each source element.
        self.max_prob: torch.Tensor
        self.register_buffer('max_prob', (prob + half).clamp(max=1).detach())


    def _reset(self):
        """
        Perform the class-local reset, which has no additional mutable state.

        The inherited reset method resets the timestep and child stability metric
        before calling this hook. This hook returns ``None``.
        """
        pass


    def forward(self, spike):
        """
        Record one spike-stream timestep for normalized-stability analysis.

        Args:
            spike: Current 0/1 spike tensor with the same logical shape as
                ``source``.

        Calling the metric increments its timestep and advances the child
        stability metric. The method returns ``None``.

        **Example:**

        .. code-block:: python

            metric(torch.ones(1))
        """
        self.stability(spike)


    @property
    def stability_norm(self):
        """
        Return the current per-element normalized stability.

        The result is a fresh tensor with values clamped to ``[0, 1]``. Before
        the first timestep, it is all zeros. Reading this property performs the
        best-case search but does not change persistent metric state.

        **Example:**

        .. code-block:: python

            current = metric.stability_norm
        """
        if not self.valid:
            return torch.zeros_like(self.source)
        timestep = self.stability.accuracy.timestep_cur
        # Search arithmetic uses float32, with L as the next power of two.
        len_t = torch.tensor([float(timestep)])
        L_t = torch.pow(2, torch.ceil(torch.log2(len_t)))
        L = int(L_t.item())
        # Integer search bounds stay on CPU.
        p_low_L = torch.floor(self.min_prob.cpu() * L_t).clamp(0, L).to(torch.int64)
        p_high_L = torch.ceil(self.max_prob.cpu() * L_t).clamp(0, L).to(torch.int64)
        # The search width truncates a float32 threshold product.
        search_range = int((torch.tensor([self.threshold], dtype=torch.float32) * 2 * L_t + 1).item())
        max_stab_len, _, _ = search_max_stab(p_low_L, p_high_L, L, search_range)
        max_stab = (1 - max_stab_len / float(timestep)).clamp(min=0).to(self.source.device)

        norm = self.stability.stability / max_stab
        norm[torch.isnan(norm)] = 0
        # The estimated maximum can produce ratios above one.
        return norm.clamp_(0, 1)


    def analyze(self, verbose=False):
        """
        Summarize the current per-element normalized stability.

        Call this method after at least one timestep.

        Args:
            verbose: Set to ``True`` to print the analysis summary. The default
                is ``False``.

        Returns:
            A pair containing the per-element normalized-stability tensor and its
            complete :class:`napl.sim.metric._shared.Analysis` summary.

        This method does not change the accumulated metric state.

        **Example:**

        .. code-block:: python

            value, result = metric.analyze()
        """
        if not self.valid:
            message = 'Metric is not valid. Please call forward() before analyze().'
            logger.error(message)
            raise AssertionError(message)
        stability_norm = self.stability_norm
        result = analyze(
            stability_norm,
            verbose=verbose,
            report='Normalized stability',
            value='normalized stability',
            timestep=self.timestep_cur,
        )
        return stability_norm, result

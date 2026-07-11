# napl port-from-UnarySim (gen-sim) log

Records produced by the `napl-gen-sim` skill - one row per UnarySim class ported into napl. Each row
attests that a UnarySim implementation was reimplemented in napl's conventions (napl_base subclass,
config/key_list validation, stype/ntype dtypes, the streaming `@napl_sim_timesteps` paradigm or the
single-shot binary-domain form, the hw_params contract, napl naming), given a `test_<name>.py`, and
validated faithful against the UnarySim original. **Status** is `ported` (new napl class created and
validated), `skipped` (no sound napl mapping, e.g. a backward-only autograd helper), or `failed`.
**Validated** records the napl-vs-UnarySim agreement (bit-exact or within the SC/quant bound).

| Date | napl class | UnarySim source | Status | Test | Validated | Notes |
|------|------------|-----------------|--------|------|-----------|-------|
| 2026-06-13 | module.conv_fsu_pc | FSUConv2dPC | ported | tests/module/test_conv_fsu_pc.py | Faithful vs FSUConv2dPC on identical input spike streams (CPU+MPS), both polarities x bias on/off x pad 0/1. bit-exact=False as expected (each side encodes the weight/bias stream with its own RNG, so cannot match bit-for-bit). Both reproduce conv2d(x,W)+b within the SC bound: napl-vs-conv2d rmse 0.005-0.024, unarysim-vs-conv2d rmse 0.009-0.021, napl-vs-unarysim rmse 0.011-0.033, all well under the SC bound 0.133 (T=512). Stochastic-port regime: within-bound agreement, not bit-exact. | Streaming FSU paradigm. Parallel-counter variant of conv_fsu: per-timestep it encodes weights (and bias) on a distinct sobol dim, unfolds the input to patches, returns the AND-count (unipolar) / XNOR-count (bipolar) per output element via matmul + fold, WITHOUT conv_fsu's add_any scaled accumulator. The conv counterpart of the existing linear_fsu_pc and mirrors its bias placement (input-1 path only) and bipolar XNOR-count form (matmul(x,w)+matmul(1-x,1-w)). Reuses conv_fsu's decorrelated rate-0.5 pad stream for bipolar zero-padding. Per-timestep count lies in [0, entry] with entry=in*kh*kw+has_bias; accumulate/T then 2*mean-entry (bipolar) or mean (unipolar) recovers the conv. Lazy encoder import (module<->operation cycle). groups=1, zero padding only. Test passes on CPU+MPS (pytest 1 passed). NOT wired into module/__init__.py and mapping.md NOT edited per task override (workflow does the wiring); Step 6 recorder NOT run per override. Test imports the class directly from napl.module.conv_fsu_pc. |

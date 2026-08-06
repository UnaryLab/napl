# napl RTL generation log

Records produced by the `napl-gen-rtl` skill - one row per RTL-generation run - plus rows for the
module-layer trees, which RULE_IMP.md says are written by hand under that file rather than by the
skill. Each row attests that Verilog RTL was emitted for an `operation` class under
`src/napl/imp/operation/<op>/` or a `module` class under `src/napl/imp/module/<name>/` and
verified against the napl Python model via golden-vector co-simulation (`make test` PASS only on a
full match). **Status** is `verified` (make test PASSed), `skipped` (no sound gate-level mapping, so
no RTL), or `failed` (generation/verification did not pass). **pp_delay** is the RTL input-to-output
latency in `i_clk` cycles written back to the class's `self.hw.pp_delay` (min across paths when they
differ; blank for skipped). Golden vectors always come from the Python model, never a hand truth
table.

| Date | napl class | RTL module(s) | Status | make test | pp_delay (cyc) | Polarities | Notes |
|------|------------|---------------|--------|-----------|----------------|------------|-------|
| 2026-06-13 | add_any | add_any | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/add_any/ - Stateful any-scale accumulating adder. Two polarity variants: rtl/add_any_unipolar.v and rtl/add_any_bipolar.v (distinct modules; the only datapath difference is the bipolar offset constant). Per timestep the model does acc += (partial - offset); clamp(acc_min,acc_max); out = acc>=scale; acc -= scale*out, where partial is the reduced input partial sum (the model's dim=None path), so the bit-serial RTL drives the partial sum directly (i_in is a 3-bit value in [0,ENTRY]). offset = (entry-scale)/2 can be a half-integer (e.g. config scale=3,width=5,entry=4 -> offset=0.5), so the accumulator is kept at 2x scale (A=2*acc, always integer): A += 2*partial - (entry-scale); clamp to [-2^width, 2^width-2]; o_out = A>=2*scale; A -= 2*scale*out. Fixed config SCALE=3 WIDTH=5 ENTRY=4 (2*offset=1 bipolar / 0 unipolar, 2*scale=6, A in [-32,30]). gen/gen_add_any.py and tb/add_any_tb.v derive golden vectors from the napl Python model (200-cycle stream hitting both clamp rails). make test OP=add_any PASS 200/200 bit-exact for both variants. Key bug fixed during bring-up: signed-vs-unsigned compares (bare localparam slices like ACC_LO[7:0] compare as unsigned) clamped everything to the rails; fixed by routing all bounds through signed 8-bit wires. pp_delay=0: o_out is combinational from the current accumulator state, same cycle as the input (i_rst_n low maps to reset(), accumulator=0). Per batch overrides, did NOT edit add.py self.hw.pp_delay (returned as pipeline_delay=0) and did NOT run the Step 6 recorder. | 2026-08-06 update: the co-simulation regime moved from scale 8 / width 20 to the carry regime of tests/operation/test_add_any.py (scale 2, width 8) over the same 8-entry reduction, and a rail segment (80 low cycles then 110 high) was appended, for 18622 vectors. With scale == entry the bipolar offset (entry - scale) / 2 is 0, the accumulator is confined to [0, entry - 1] and neither clamp is reachable; at scale 2 the offset is 3 and the low rail drives it onto -128 in 43 cycles, the high rail onto 127. That is what makes ACC_LO observable: halving it in add_any_bipolar.v fails at cycle 18526, 12 mismatches over 18622 vectors, restoring it passes 18622/18622. The unipolar negative clamp stays unreachable by construction (offset 0, and a carry only subtracts down to 0): the same injection into add_any_unipolar.v PASSes 18622/18622, so that clamp is dead logic rather than untested. The generator asserts all four facts. | 2026-08-06 removal: on that evidence the negative clamp is gone from add_any_unipolar.v, which now clamps to [0, 2^WIDTH-2]. Removed the `ACC_LO` localparam, the `s_lo` signed wire it fed, and the `(sum < s_lo) ? s_lo` arm of `clmp`; nothing else read either name, so the removal orphaned nothing. The banner carries the invariant that replaces the branch: `acc >= 0` always holds, by induction from acc = 0 after reset, since the unipolar offset is 0 and every addend is non-negative so the sum is at least acc, and a fire needs sum >= 2*SCALE and subtracts exactly 2*SCALE. Only the bipolar variant, whose nonzero offset is subtracted every cycle, can drive a low clamp, and its `ACC_LO` is untouched. add_any_bipolar.v is untouched, and so are the generator and its negative-rail saturation block, which still passes unchanged and still asserts `rail_uni[0] == 0` and `rail_bi[0] == model_bi.acc_min`. Gates after the removal: `make test OP=add_any` PASS 18622/18622 (unipolar + bipolar), `operation/add_any/vec/add_any.vec` md5 893829a8e4f35be39f6684996ad87bac unchanged, `make guards` green, `iverilog -g2001 -Wall` 0 warnings.
| 2026-07-26 | add_gaines | add_gaines | verified | PASS | 0 | unipolar + bipolar | Backfill verification from `src/napl/imp`: ENTRY=8, SELECT_WIDTH=3, SCALED=0/1; power-on and mid-stream reset; 512/512 vectors matched. |
| 2026-07-26 | add_ugemm | add_ugemm_unipolar, add_ugemm_bipolar | verified | PASS | 0 | unipolar + bipolar | Backfill verification from `src/napl/imp`: ENTRY=8, COUNT_WIDTH=4, ACC_WIDTH=14, SCALED=0/1; power-on and mid-stream reset; 512/512 vectors matched. |
| 2026-06-13 | bi2uni | bi2uni | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/bi2uni/rtl/bi2uni.v - bipolar-to-unipolar stream converter, single variant (no polarity split, bare op name bi2uni). Mealy machine matching the Python forward(): addend = 2*i_in - 1 (+/-1), acc_clamped = clamp(acc + addend, ACC_MIN=-2, ACC_MAX=1), o_out = (acc_clamped >= 1), acc <= acc_clamped - o_out. width=2 from test config fixes acc range [-2,1]; pre-clamp sum spans [-3,2], so a 3-bit signed datapath ([-4,3]) covers both. Single register acc, active-low i_rst_n resets it to 0 matching reset(). o_out is combinational in i_in and acc, so pp_delay=0. Files: rtl/bi2uni.v, tb/bi2uni_tb.v, gen/gen_bi2uni.py (golden vectors from napl Python model). make test OP=bi2uni PASS 304/304 vectors bit-exact. Closely mirrors the sibling uni2bi converter. Per batch overrides: did NOT edit self.hw.pp_delay in bi2uni.py (returned here for serial Finalize) and did NOT run the Step 6 recorder. |
| 2026-08-04 | decode | decode | verified | PASS | 0 | none (bare op) | WIDTH+1-bit spike counter, combinational count in the arrival cycle; polarity only scales the count when read, so no polarity split; WIDTH inherited from the model width via vec/decode_params.vh; 208 vectors with mid-stream resets; the RTL covers forward() only, the divide and bipolar rescale in the spike_value property have no hardware counterpart; verified at WIDTH 4 |
| 2026-06-13 | dff | dff | verified | PASS | 1 |  | /Users/diwu/Projects/napl/src/napl/hw/dff/rtl/dff.v - dff is a per-timestep streaming op: a depth-D FIFO that delays the spike stream by depth cycles with zero reset state. Generated RTL for the class default depth=1, i.e. a single D flip-flop (module dff, bare op name; no polarity variants since polarity_required=False). One module: rtl/dff.v captures i_in on posedge i_clk and emits it the next cycle; active-low i_rst_n clears reg_q to 0, matching the model's reset() (FIFO initialized to zeros). gen/gen_dff.py drives a 64-cycle seeded pseudo-random stream through the napl Python model from reset() and records per-cycle (in, out); tb/dff_tb.v pulses reset then, per cycle, drives i_in, samples o_out (the value stored before this posedge), then clocks. conda run -n napl make test OP=dff PASSes bit-exactly 64/64. pp_delay=1 (single register stage, one input->output path). Per the batch overrides I did NOT edit dff.py's self.hw.pp_delay and did NOT run the Step 6 recorder; returning pipeline_delay=1 for the workflow to apply. Note: the Python class is parameterized by depth (default 1); higher depths are a deeper shift register (structurally like hw/shiftreg but with zero reset) and would need a DEPTH localparam if a non-default depth is targeted. Verified RTL covers the canonical depth=1 dff. |
| 2026-06-13 | div_cordiv | div_cordiv | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/div_cordiv/rtl/div_cordiv.v - Unipolar-only correlated division (polarity_required=False, no polarity branch -> single bare-named module, no _unipolar/_bipolar postfix). Built the depth-2 concrete variant (model's optimal/default, also what div_iscb uses). State: depth-2 circular quotient buffer + 1-bit cyclic idx; for depth 2 the Sobol index table rand_seq={0,1} so rand_q==buffer_q[idx]. o_quotient = i_divisor ? i_dividend : buffer_q[idx] is combinational from inputs+state (pp_delay=0); buffer shifts in the new quotient only where i_divisor spikes; idx advances each cycle. Active-low i_rst_n reproduces reset() exactly (buffer all-zero, idx=0). Golden vectors emitted from the napl Python model; make test PASS bit-exact 16/16. Per batch overrides: did NOT edit self.hw.pp_delay and did NOT run the Step 6 recorder; pp_delay returned for central application. |
| 2026-07-26 | div_gaines | div_gaines_unipolar, div_gaines_bipolar | verified | PASS | 1 | unipolar + bipolar | Backfill verification from `src/napl/imp`: DEPTH=5; power-on and mid-stream reset; 8448/8448 vectors matched. |
| 2026-06-13 | div_iscb | div_iscb | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/div_iscb/ - In-stream correlation-based division (iscbdiv), rate coding. Two polarity variants: div_iscb_unipolar and div_iscb_bipolar. Unipolar = sync_skewed(width=3) feeding div_cordiv(depth=2, sobol -> rand_seq_idx=[0,1], alternating buf0/buf1 read). Bipolar wraps the unipolar magnitude core (instantiated as div_iscb_unipolar) with signabs(width=3) x2, bi2uni(width=2) x2, uni2bi(width=3), and combines sign_d ^ sign_s ^ bi_abs_q. Internal helper modules (one per file, op-prefixed): div_iscb_signabs.v, div_iscb_bi2uni.v (signed acc [-2,1]), div_iscb_uni2bi.v (signed acc [-4,3]); all compiled together by the generic Makefile (rtl/*.v). Each sub-block updates its accumulator/counter on posedge then reads the updated value within the same timestep, matching the Python model; quotient is combinational from current inputs + current registered state, so pp_delay=0 (min across paths; all paths combinational). reset() state replicated exactly: sync cnt=0, cordiv buf={0,0}/idx=0, signabs acc=4, bi2uni/uni2bi acc=0. Before writing Verilog I validated a scalar Python spec against the napl model over 200 mixed cycles for both polarities (exact match), then make test OP=div_iscb co-sim passed 512/512 vectors bit-exact for both variants. Confirmed iverilog/vvp on PATH. Per batch overrides: did NOT edit div.py (self.hw.pp_delay) and did NOT run the report recorder; pipeline_delay=0 returned for the workflow to apply. Note: div.py currently sets self.delay=0 (legacy scalar) and does not yet import hw_params; Finalize should add 'from napl.sim.base import napl_base, hw_params' and set self.hw = hw_params(pp_delay=0). |
| 2026-08-04 | encode | encode | verified | PASS | 0 | none (bare op) | strict > against a Python-generated number-sequence ROM ($readmemb, mul_ugemm pattern); WIDTH=8 index, FRAC=9 shared fixed point so bipolar (x+1)/2 is exact; 3840 vectors incl. tie rails and a mid-stream reset; i_input is the already-derived probability p, so deriving p from the input value has no hardware counterpart, and p must sit on the 1/2**FRAC grid to stay bit-exact |
| 2026-07-26 | exp_n1 | exp_n1 | verified | PASS | 0 | unipolar | Backfill verification from `src/napl/imp`: WIDTH=8; power-on and mid-stream reset; 3328/3328 vectors matched. |
| 2026-07-26 | exp_ng | exp_ng | verified | PASS | 1 | bipolar | Backfill verification from `src/napl/imp`: DEPTH=5, GAIN=1; power-on and mid-stream reset; 3328/3328 vectors matched. |
| 2026-06-13 | gt_rc | gt_rc | verified | PASS | 1 |  | /Users/diwu/Projects/napl/src/napl/hw/gt_rc/rtl/gt_rc.v - gt_rc is a stateful per-timestep streaming circuit; single bare module gt_rc (no polarity branch: polarity_required=False, forward() never reads self.polarity). State = 1-bit result register dff (reset to 1) + the 2-bit skew counter cnt of sync_skewed(width=2) (cnt 0..3, reset to 0 via is_first_call on first forward after reset()). Per cycle the RTL reproduces sync_skewed's output_1 (skewed input_0) and passthrough input_1, then d_enable=sync_0^sync_1 and dff_next=d_enable?sync_0:dff, with cnt += diff?(a?+1:-1):0 saturating in [0,3]. o_out is the registered dff read BEFORE its update, so input->output latency is 1 cycle (matches the model's self.delay=1). i_rst_n (active-low) maps to reset(): dff<=1, cnt<=0. make test OP=gt_rc PASSes 2000/2000 golden vectors emitted from the napl Python model. iverilog -g2012 -Wall compiled clean; Verilator MODMISSING was isolated-lint noise only. Files: rtl/gt_rc.v, tb/gt_rc_tb.v, gen/gen_gt_rc.py under src/napl/hw/gt_rc/. Per batch overrides, did NOT edit compare.py self.hw.pp_delay (return pipeline_delay=1) and did NOT run the Step 6 recorder. |
| 2026-08-04 | inhibit | inhibit | verified | PASS | 0 | none (bare op) | temporal-code INHIBIT: sticky latch s_t = s_t-1 \| (in_1 & ~in_0), o_out = in_0 & ~s_t; combinational output, latch resets to 0; gen + tb from the Python model, per-segment resets |
| 2026-06-13 | jkff | jkff | verified | PASS | 1 |  | /Users/diwu/Projects/napl/src/napl/hw/jkff/ - JK flip-flop, clocked/stateful, no polarity variants -> single bare module jkff (rtl/jkff.v). RTL implements the model's reduced characteristic eq Q' = Q ? ~K : J as a synchronous register; active-low i_rst_n maps to reset() (q<-0). Ports: i_clk, i_rst_n, i_input_j, i_input_k, o_q. Generator (gen/gen_jkff.py) drives a 12-cycle stream from the napl Python model (napl.sim.operation.jkff) visiting both q states under every (J,K); testbench (tb/jkff_tb.v) pulses i_rst_n low then replays cycle-by-cycle. make test OP=jkff PASS 12/12 bit-exact. pp_delay=1 (single registered output stage; matches the class's legacy self.delay=1). Per batch overrides: did NOT write self.hw.pp_delay to jkff.py and did NOT run the Step 6 recorder. |
| 2026-06-13 | lt_rc | lt_rc | verified | PASS | 1 |  | /Users/diwu/Projects/napl/src/napl/hw/lt_rc/rtl/lt_rc.v - Per-timestep streaming compare op. Single bare module lt_rc (no polarity branch in forward -> no _unipolar/_bipolar split). RTL models the embedded sync_skewed(width=2) as a 2-bit saturating counter (cnt 0..3) plus the less-than dff; sync_0 = i_in_0 ^ i_in_1 ? (i_in_0 ? ~not_max : not_min) : i_in_0, dff_nxt = d_en ? sync_1 : dff. o_out is the registered dff sampled BEFORE this cycle's update -> pp_delay = 1 (one register stage; matches the class's existing self.delay = 1). i_rst_n (active-low) reloads cnt<=0, dff<=0 per Python reset(). Golden vectors emitted from the napl Python model across 5 reset segments / 65 cycles exercising counter saturation at both ends and all 00/01/10/11 input pairs; testbench uses an explicit per-line rst column to mark segment boundaries. make test OP=lt_rc PASS 65/65 bit-exact; Verilator --lint-only -Wall clean on the RTL. Per batch overrides: did NOT edit compare.py (pp_delay returned, not written) and did NOT run the Step 6 recorder. Files: src/napl/hw/lt_rc/{rtl/lt_rc.v, tb/lt_rc_tb.v, gen/gen_lt_rc.py}. |
| 2026-06-13 | max_rc | max_rc | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/max_rc/rtl/max_rc.v - max_rc streams max + argmax of two rate-coded spike streams via an embedded sync_skewed (width-2, saturating 0..3 counter). No polarity branch in the Python model, so a single bare module max_rc (no _unipolar/_bipolar postfix) with two outputs: o_max and o_arg. State held across cycles: the sync counter cnt and the argmax dff, both reset to 0 (i_rst_n active-low maps to reset()). o_max uses the OLD dff; o_arg is the NEW (post-update) dff; both are combinational from current inputs+state, so pp_delay=0 (min across both output paths), consistent with the class's self.delay=0. Bit-exact: conda run -n napl make test OP=max_rc PASS, 4096/4096 vectors matched from the Python model. One bug found and fixed during bring-up: the model term (cnt_not_min + cnt_not_max) is an arithmetic sum in {0,1,2}, initially mis-encoded in RTL as a bit concatenation (2*cnt_not_min + cnt_not_max). Verilator lint clean (one intentional UNUSEDSIGNAL pragma on the provably-{0,1} sync_0 sum's upper bits). Per batch overrides: did NOT write self.hw.pp_delay to compare.py and did NOT run the Step 6 recorder; pp_delay=0 returned here for the workflow's Finalize phase. iverilog/vvp present at /usr/local/bin. |
| 2026-06-13 | max_tc | max_tc | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/max_tc/rtl/max_tc.v - max_tc is temporal-coded max = bitwise OR of two spike streams; combinational, stateless, no polarity split, so a single bare-named module (no _unipolar/_bipolar postfix). RTL: assign o_out = i_in_0 \| i_in_1. pp_delay=0 (pure combinational). Generator gen/gen_max_tc.py emits 4 golden vectors from napl.sim.operation.max_tc (exhaustive 2x1-bit sweep); self-checking tb/max_tc_tb.v replays them. conda run -n napl make test OP=max_tc => PASS max_tc: 4/4 vectors bit-exact. Files: rtl/max_tc.v, tb/max_tc_tb.v, gen/gen_max_tc.py under /Users/diwu/Projects/napl/src/napl/hw/max_tc/. Per batch overrides, did NOT edit compare.py self.hw.pp_delay (returned here as pipeline_delay=0) and did NOT run the Step 6 recorder. iverilog/vvp on PATH. |
| 2026-06-13 | min_rc | min_rc | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/min_rc/rtl/min_rc.v - Stateful rate-coded min/argmin via sync_skewed (width=2, cnt in 0..3). Single module 'min_rc' (no polarity split: the class has polarity_required=False and no polarity branch). State: dff (reset 0) and 2-bit cnt (reset 0); i_rst_n maps to reset(). Two outputs: o_min and o_argmin. Subtlety: Python forward() reads o_min from PRE-update dff (dff*in_0 + (1-dff)*in_1) but returns o_argmin = 1 - self.dff AFTER the in-timestep dff update, so RTL uses o_argmin = ~dff_next. make test OP=min_rc PASSes bit-exact 4096/4096 cycles on a seed-0 randomized stream generated from the Python model; verilator --lint-only -Wall clean. pp_delay=0: both input->output paths are combinational w.r.t. current inputs (min over paths = 0), matching the model's self.delay=0. Files: rtl/min_rc.v, tb/min_rc_tb.v, gen/gen_min_rc.py. Did NOT edit compare.py pp_delay and did NOT run the Step 6 recorder per batch overrides. |
| 2026-06-13 | min_tc | min_tc | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/min_tc/rtl/min_tc.v - Temporal-coded min implemented as a 2-input AND gate (temporal streams are 1s then 0s, so bitwise AND = elementwise min). Purely combinational, no polarity split (bare op name), no state, single output o_out -> pp_delay=0. Golden vectors generated by driving the napl Python min_tc model over all 4 input combinations; tb/min_tc_tb.v replays them and prints PASS only on full bit-exact match. make test OP=min_tc -> PASS 4/4. Per batch overrides, did NOT edit compare.py self.hw.pp_delay (returned here) and did NOT run the Step 6 recorder. Files: rtl/min_tc.v, gen/gen_min_tc.py, tb/min_tc_tb.v under src/napl/hw/min_tc/. |
| 2026-06-13 | mul_and | mul_and | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/mul_and - Combinational op: unipolar = AND (mul_and_unipolar.v), bipolar = XNOR (mul_and_bipolar.v); no registers, pp_delay=0. The op dir (rtl/tb/gen) had been deleted in the working tree per git status; restored from git HEAD where it matched the current Python model exactly (mul.py: unipolar AND of int8 spikes, bipolar XNOR via double bitwise_xor). gen_mul_and.py emits 4 golden vectors (exhaustive 2x1-bit sweep) straight from napl.sim.operation.mul_and for both polarities; tb replays them and printed 'PASS mul_and: 4/4 vectors'. conda run -n napl make test OP=mul_and PASSed bit-exactly via iverilog/vvp. Per batch overrides, did NOT edit mul.py self.hw.pp_delay (already 0) and did NOT run the Step 6 recorder; returning pp_delay=0 for the workflow to finalize. |
| 2026-06-13 | mul_csg | mul_csg | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/mul_csg/ - mul_csg (conditional spike generation multiply), two polarity modules: mul_csg_unipolar (AND of i_in_0 with a ROM-compare spike) and mul_csg_bipolar (adds the inverse path with a second independent ROM-walking counter). RTL pinned to reference config timestep=256, generator='sobol' (width=8, len=256): the generator's float num_seq is a permutation of 0..255 over 1/256, baked into the .v as an 8-bit case ROM. Operand i_in_1 is 9-bit round(prob*256) in [0,256] (prob=value unipolar, (value+1)/2 bipolar; 9 bits so prob 1.0 -> 256 is representable). Bit-exact contract: the float gt(prob, num_seq) equals integer in_1 > num_seq only when prob is an exact 1/256 multiple, which is the same width-bit quantization the napl test feeds via gen_rand_tensor; the generator picks on-grid operands (value=j/128) and asserts exactness. pp_delay=0: o_out is combinational from the CURRENT seq_idx/seq_idx_inv registers + ROM; the counters are async-reset (i_rst_n -> 0, matching reset()) state that only advances the next cycle's index (enabled by i_in_0 / ~i_in_0). make test OP=mul_csg PASSes 2048/2048 vectors (unipolar+bipolar), exit 0. Per batch overrides: did NOT edit self.hw.pp_delay (already hw_params(pp_delay=0) in mul.py) and did NOT run the Step 6 recorder. gen/_emit_rtl.py is a one-shot author helper kept for ROM provenance; the Makefile only runs gen/gen_mul_csg.py. |
| 2026-07-26 | mul_gaines | mul_gaines_unipolar, mul_gaines_bipolar | verified | PASS | 0 | unipolar + bipolar | Backfill verification from `src/napl/imp`: combinational circuit with no sizing parameters; 4/4 vectors matched. |
| 2026-07-31 | mul_shiftreg | mul_shiftreg_unipolar,mul_shiftreg_bipolar | verified | PASS | 0 | unipolar + bipolar | WIDTH=4, DEPTH=16; model-derived Sobol ROM; alternating-register first-call reset state; power-on and mid-stream reset; 8448/8448 vectors matched |
| 2026-06-13 | relu_cnt | relu_cnt | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/relu_cnt/rtl/relu_cnt.v - Counter-based bipolar rate-coded ReLU. No polarity split (logic does not branch on polarity), so bare module name relu_cnt (no _unipolar/_bipolar). Stateful: a WIDTH-bit saturating up/down counter acc, parameter WIDTH=3 (the class default) wired in the tb/gen. Active-low i_rst_n reloads acc=HALF (2^(WIDTH-1)=4), matching the model's reset() state (not zero). Per cycle: below_half=(acc<HALF); o_out = i_in \| below_half (combinational in i_in and current acc -> pp_delay=0); acc advances on posedge i_clk by +1 if o_out else -1, clamped to [0, 2^WIDTH-1]. Golden vectors from the napl Python model (relu_cnt, seed 0, 256-cycle random bipolar stream exercising both clamp rails). make test OP=relu_cnt PASSes 256/256 bit-exactly. Files: src/napl/hw/relu_cnt/{rtl/relu_cnt.v, tb/relu_cnt_tb.v, gen/gen_relu_cnt.py}. Per batch overrides: did NOT edit self.hw.pp_delay in relu.py and did NOT run the report recorder; pipeline_delay=0 returned for the workflow to apply. |
| 2026-06-13 | relu_hub | relu_hub | skipped | n/a |  |  | relu_hub is a single-shot binary-domain HUB cell, not a per-timestep gate-level streaming circuit, so it has no sound RTL mapping. Its forward() is torch.nn.functional.hardtanh(input, 0.0, self.scale) applied to a whole floating-point (non_spike_type / float32) tensor at once: no spike stream, no tick()/timestep advance (timestep_cur stays 0), no held state, and no reset() of registers. It operates on binary-domain values rather than 0/1 spikes, and scale plus the clamp bounds are floats with no fixed-point width specified, so there is no defined bit-serial datapath to lower. Per the skill's Step 1/Step 4 guidance (operation subpackage gate circuits only; skip honestly rather than force an unsound design) this is reported as skipped. No RTL, testbench, or generator emitted; no make test run. Per batch overrides, did not write self.hw.pp_delay to relu.py and did not run the Step 6 recorder; returning pipeline_delay=null. |
| 2026-06-13 | relu_sat | relu_sat | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/relu_sat/rtl/relu_sat.v - Bipolar/rate-coded only, so a single bare-name module relu_sat (no polarity split). Models the two chained add_any accumulators (scale=1, width=3) from the Python forward(): a sub_1 stage shifting [-1,1]->[-1,0] then an add_1 stage shifting [-1,0]->[0,1]. The add_any 0.5 offsets are made integer by scaling each accumulator by 2 (half-units A=2*acc, signed, clamped to [-8,6]=2*[-4,3]); per cycle sub: A+=2*i_in-1, clamp, out_sub=A>=2, A-=2; add: A+=2*out_sub+1, clamp, o_out=A>=2, A-=2. Output is combinational from current state (same cycle as input) -> pp_delay=0; i_rst_n low maps to reset() (both accumulators=0). An integer reference model matched the Python model over 200 random streams before writing RTL. make test OP=relu_sat PASSes 218/218 vectors bit-exactly (golden vectors emitted from the napl Python model; stream covers both-direction saturation plus a pseudo-random tail). Verilator lint clean. Per batch overrides: did NOT edit relu.py self.hw.pp_delay and did NOT run the Step 6 recorder; pp_delay returned here for the workflow's Finalize phase. Note the legacy class still sets self.delay=0; Finalize should convert to self.hw=hw_params(pp_delay=0) and add the hw_params import. |
| 2026-07-31 | relu_shiftreg | relu_shiftreg | verified | PASS | 0 | bipolar only (bare op) | DEPTH=8 shift-register state; alternating-register first-call reset state; power-on and mid-stream reset; 3328/3328 vectors matched |
| 2026-07-31 | relu_tc | relu_tc | verified | PASS | 0 | bipolar only (bare op) | WIDTH=8 temporal accumulator; cycle and accumulator saturate after threshold without changing Python predicates; power-on and mid-stream reset; 3328/3328 vectors matched |
| 2026-06-13 | round_fxp | round_fxp | skipped | n/a |  |  | round_fxp (src/napl/sim/operation/round_fxp.py) is a single-shot, binary-domain op, not a per-timestep streaming spike circuit, so it has no sound gate-level (bit-serial) RTL mapping and is skipped per the skill's Step 1/Step 4 guidance. Evidence: the class docstring states 'Single-shot, binary-domain (no tick)'; forward() calls round_ste(input, fracwidth, min_val, max_val) once on the whole tensor (an STE wrapper over a torch.autograd.Function), with no tick()/reset()/timestep_cur advance and no 1-bit spike stream. It is a trainable quantizer of the same family as the HUB/FXP/TLUT/Hard cells, which CLAUDE.md describes as whole-tensor single-shot (timestep_cur stays 0). The napl-gen-rtl contract (one forward() timestep == one posedge i_clk, scalar 1-bit datapath, golden vectors driven per-timestep from model.reset()) does not apply: there is no per-cycle stream to record. The underlying operation round(x<<fracwidth).clamp(min,max)>>fracwidth is a combinational saturating clamp, but on a multi-bit fixed-point word of parameterized width (1+intwidth+fracwidth bits, e.g. 1+3+4), not a 1-bit spike datapath. A word-level clamp module would belong to a higher-level FXP block, not this skill's bit-serial per-operation RTL; emitting it would be an unsound design relative to the contract. The class itself is healthy: tests/operation/test_round.py passes (printed 'Test passed.'). It still carries the legacy scalar self.delay = 0 (line 64) rather than self.hw = hw_params(...); since this op is skipped, no pp_delay applies. No src/napl/hw/round_fxp/ directory exists or was created. Per batch overrides, did not edit the Python class and did not run the Step 6 recorder. |
| 2026-06-13 | shiftreg | shiftreg | verified | PASS | 4 |  | /Users/diwu/Projects/napl/src/napl/hw/shiftreg/rtl/shiftreg.v - Stateful depth-DEPTH bit-serial delay line; single module shiftreg (no polarity variants since polarity_required=False). RTL uses localparam DEPTH=4 matching the canonical example; reg_q[0] is the oldest cell read out each cycle, shift toward index 0, new input appended at reg_q[DEPTH-1]. Active-low i_rst_n async-reloads the model's exact reset() state reg[i]=i%2 (alternating 0,1,0,1), NOT all-zeros. Generator gen/gen_shiftreg.py drives napl.sim.operation.shiftreg from reset() over a 15-cycle 0/1 stream and records per-cycle (in,out); testbench replays them, checking o_out (oldest cell, read before the shift) on negedge then advancing one posedge per timestep. make test OP=shiftreg PASS 15/15 bit-exact. pp_delay = depth (input->output latency); for DEPTH=4 RTL that is 4 (single input->output path, so min==4). The Python class already sets self.hw=hw_params(pp_delay=self.depth); per batch overrides I did NOT edit the class or run the Step 6 recorder. iverilog/vvp present at /usr/local/bin. Files: rtl/shiftreg.v, tb/shiftreg_tb.v, gen/gen_shiftreg.py under /Users/diwu/Projects/napl/src/napl/hw/shiftreg/. |
| 2026-06-13 | sigmoid_hard | sigmoid_hard | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/sigmoid_hard/rtl/sigmoid_hard.v - sigmoid_hard = add_any(scale=2, width=3) fed the pre-reduced per-timestep sum (input+1). The bipolar offset is (entry-scale)/2 = (2-2)/2 = 0, identical to unipolar, so it is a SINGLE bare-named module (no _unipolar/_bipolar split); confirmed both polarities emit identical streams from the Python model. RTL: a 5-bit signed accumulator register mirroring add_any's width=3 clamp range [-4,3]; per posedge i_clk it computes acc_sum=clamp(acc+(i_in+1),-4,3), o_out=(acc_sum>=2), acc_next=acc_sum-(o_out?2:0). Output is combinational in (acc register, i_in) within the same cycle the input is applied, so input->output latency = 0 cycles (matches the class's self.delay=0). i_rst_n (active-low) maps to reset() which zeroes the accumulator. make test OP=sigmoid_hard PASSes bit-exactly 256/256 vectors, golden vectors driven from napl.sim.operation.sigmoid_hard over a 256-cycle deterministic pseudo-random stream from reset(). Verilator lint clean (TB MODMISSING is an isolation artifact). Per batch overrides: did NOT edit the Python class self.hw.pp_delay and did NOT run the Step 6 recorder; returning pp_delay=0 here. |
| 2026-06-13 | sigmoid_hub | sigmoid_hub | skipped | n/a |  |  | sigmoid_hub is a single-shot binary-domain HUB op, not a per-timestep streaming spike circuit. Its forward() is torch.nn.functional.hardsigmoid(input * scale) on a whole float tensor (non_spike_type) at once: no tick(), no reset state, timestep_cur stays 0, and the input/output are floating-point values rather than 1-bit spike streams. Per CLAUDE.md/skill Step 4, HUB cells are whole-tensor binary-domain functions with no natural gate-level (1-bit datapath) mapping. A Verilog datapath would require inventing a fixed-point format, bit-width, and rounding not defined by the float model, making bit-exact co-sim against the Python model unsound. No tests/operation/test_sigmoid_hub.py exists either. Skipped honestly rather than forcing an unsound design. (Per batch overrides: did not edit Python self.hw.pp_delay and did not run the Step 6 recorder.) |
| 2026-06-13 | signabs | signabs | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/signabs/rtl/signabs.v - Single bare-op module signabs (no polarity variants; polarity_required=False). Saturating up/down accumulator (WIDTH=3, ACC_MAX=7, ACC_MED=4): +1 on i_in=1, -1 on i_in=0, clamped to [0,7]. o_sign=(acc_next<ACC_MED), o_abs=o_sign^i_in. Reset state acc=ACC_MED=4 (non-zero), reloaded on active-low i_rst_n exactly as Python reset(). The accumulator is registered but the i_in->o_sign/o_abs path is combinational within the timestep (outputs use the updated acc), so pp_delay=0 across both outputs. gen/gen_signabs.py emits golden vectors straight from napl.sim.operation.signabs; tb/signabs_tb.v replays them cycle-by-cycle. make test OP=signabs => PASS 42/42 vectors bit-exact. Per batch overrides, did NOT edit the Python class self.hw.pp_delay and did NOT run the Step 6 recorder. Files: rtl/signabs.v, tb/signabs_tb.v, gen/gen_signabs.py under /Users/diwu/Projects/napl/src/napl/hw/signabs/. |
| 2026-07-31 | signabs_interleave | signabs_interleave | verified | PASS | 0 | bipolar only (bare op) | WIDTH=5 saturating accumulator with updated-state sign/magnitude outputs; power-on and mid-stream reset; 3328/3328 vectors matched |
| 2026-07-31 | signabs_shiftreg | signabs_shiftreg | verified | PASS | 0 | bipolar only (bare op) | DEPTH=8 shift-register count with alternating-register first-call reset state; power-on and mid-stream reset; 3328/3328 vectors matched |
| 2026-06-13 | sqrt_emit | sqrt_emit | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/sqrt_emit/ - Opportunistic-bit-inserting square root. Two polarity variants emitted: rtl/sqrt_emit_unipolar.v and rtl/sqrt_emit_bipolar.v. Both share the nsadd path (fixed UNIPOLAR add_any in __init__: scale=1, width=3, offset=0, acc clamped [-4,3]) -> o_out = (acc+i_in+emit_out >= 1); they differ only in the emit feedback: unipolar = scrambled & o_out, bipolar = scrambled & bi2uni(o_out) (bi2uni width=2, acc_b clamped [-2,1]). scrambled = depth-2 shiftreg reading the oldest cell, pushing ~o_out; i_rst_n reloads sr[i]=i%2 (sr=2'b10), acc=0, acc_b=0, emit_out=0 to match Python reset(). pp_delay=0: o_out is combinational in i_in given the cycle-t registers; the accumulators, emit feedback bit, and shiftreg are internal state, not on the input->output path. make test OP=sqrt_emit PASSes 64/64 vectors (unipolar+bipolar), iverilog -Wall clean. gen + tb derive expected outputs from the napl Python model. Per batch overrides: did NOT edit self.hw.pp_delay in sqrt.py and did NOT run the Step 6 recorder; returning pipeline_delay=0 for the workflow to apply. |
| 2026-07-26 | sqrt_gaines | sqrt_gaines_unipolar, sqrt_gaines_bipolar | verified | PASS | 1 | unipolar + bipolar | Backfill verification from `src/napl/imp`: WIDTH=5; power-on and mid-stream reset; 3328/3328 vectors matched. |
| 2026-06-13 | sqrt_traceiscb | sqrt_traceiscb | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/sqrt_traceiscb/ - Stateful bit-serial square root via stochastic bit inserting (iscb cordiv kernel). Two polarity modules emitted: rtl/sqrt_traceiscb_unipolar.v and rtl/sqrt_traceiscb_bipolar.v. State per cycle: trace (feedback to output), dff, cordiv depth-2 buffer (buf0/buf1 + idx toggle; rand_seq is deterministically [0,1]), plus a width-2 bi2uni signed accumulator (range [-2,1]) in the bipolar variant only. output = trace\|input; out = bi2uni(output) (bipolar) or output (unipolar); dividend = ~dff & out; divisor = dff \| dividend; trace' = cordiv(dividend,divisor); dff' = ~dff. Reset (active-low i_rst_n) zeroes all registers, matching the model reset(). Validated my scalar gate-level reference against the Python model (40-cycle random streams, both polarities) before writing RTL. make test OP=sqrt_traceiscb PASSes 128/128 vectors bit-exact for both variants (one testbench instantiates both DUTs). gen/gen_sqrt_traceiscb.py drives both models from reset() and emits 'in out_unipolar out_bipolar' per cycle. pp_delay=0: o_out is combinational from i_in and the registered trace bit (only output path). Per batch overrides, did NOT edit self.hw.pp_delay in sqrt.py and did NOT run the Step 6 recorder; returning pp_delay=0 for the workflow to apply. Note sqrt.py's class still sets legacy self.delay=0; the Finalize phase should replace it with self.hw=hw_params(pp_delay=0) and add the hw_params import. |
| 2026-06-13 | sqrt_tracejkff | sqrt_tracejkff | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/sqrt_tracejkff/ - Two polarity variant modules generated: rtl/sqrt_tracejkff_unipolar.v and rtl/sqrt_tracejkff_bipolar.v. Per-timestep streaming circuit. Output o_out = trace \| i_in (combinational in i_in given current registers). Unipolar: JK-FF trace update trace' = (~trace) & o_out (J=o_out, K=1). Bipolar: adds a width-2 bi2uni accumulator (signed 3-bit, range [-2,1]; acc_n = clamp(acc + 2*o_out - 1, -2, 1), out_uni = acc_n>=1, acc' = acc_n - out_uni), then trace' = (~trace) & out_uni. reset() maps to i_rst_n low -> trace=0 (and acc=0 bipolar). pp_delay=0: input->output path is purely combinational (trace/acc are clocked but not on the i_in->o_out path); min across paths is 0. make test OP=sqrt_tracejkff PASS 64/64 vectors bit-exact (gen + tb derived from the napl Python model, both polarity columns). Legacy self.delay=0 in the class already agrees. Per batch overrides: did NOT edit self.hw.pp_delay in sqrt.py and did NOT run the Step 6 recorder; returning pipeline_delay=0 for the workflow to apply centrally. iverilog/vvp present on PATH. |
| 2026-06-13 | square_dff | square_dff | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/square_dff/ - Stateful op: squares one spike stream by AND (unipolar) / XNOR (bipolar) with a depth=1 delayed copy of itself (a D flip-flop). RTL emits two polarity variants: square_dff_unipolar.v (o_out = i_in & in_d) and square_dff_bipolar.v (o_out = ~(i_in ^ in_d)), each with a 1-bit delay register in_d clocked on posedge i_clk and async active-low i_rst_n clearing it to 0 (matches the Python dff buffer reset to zeros). Generator gen_square_dff.py drives a seeded 64-cycle 0/1 stream through napl.sim.operation.square_dff from reset() and records per-cycle (i_in, out_unipolar, out_bipolar); tb pulses i_rst_n low then checks the combinational output each cycle before the rising edge advances the register. make test OP=square_dff PASSes 64/64 bit-exactly. Designed for the test config depth=1 (single-stage delay); larger depths would be an N-deep shift register, not covered. pipeline_delay=0: o_out is combinational w.r.t. the current i_in (min across the 0-cycle current-input path and the 1-cycle delayed path). Per batch overrides, did NOT write self.hw.pp_delay to square.py and did NOT run the Step 6 recorder. |
| 2026-06-13 | sync_skewed | sync_skewed | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/sync_skewed/rtl/sync_skewed.v - Stateful skewed synchronizer of two spike streams; single bare module sync_skewed (polarity_required=False, no _unipolar/_bipolar split). Datapath: saturating WIDTH-bit up/down counter cnt (WIDTH=3, CNT_MAX=7) buffering lead/lag between streams. diff=i_in_1^i_in_2; on 00/11 o_out_1=i_in_1 and cnt held; on a push (i_in_1=1,diff) emit only when saturated high and cnt+1 (sat); on a pop (i_in_1=0,diff) emit only when not empty and cnt-1 (sat). o_out_2=i_in_2 passthrough. Outputs are pure combinational assigns from current inputs + registered cnt (cnt read BEFORE update, like shiftreg), so input->output latency is 0 on both paths; cnt update is the only registered logic. Reset (active-low i_rst_n) loads cnt=0, matching the Python reset(). pp_delay=0 (min across both output paths). make test OP=sync_skewed PASSes 33/33 vectors bit-exactly; golden vectors generated from napl.sim.operation.sync_skewed via gen/gen_sync_skewed.py, exercising push/pop/00/11-passthrough and both saturation limits. Per batch overrides: did NOT edit self.hw.pp_delay in sync.py and did NOT run the Step 6 recorder; pp_delay returned here for the workflow to apply. Files: rtl/sync_skewed.v, tb/sync_skewed_tb.v, gen/gen_sync_skewed.py under /Users/diwu/Projects/napl/src/napl/hw/sync_skewed/. |
| 2026-07-26 | sync_skewed_int | sync_skewed_int | verified | PASS | 0 | n/a | Backfill verification from `src/napl/imp`: WIDTH=4; power-on and mid-stream reset; 8448/8448 vectors matched. |
| 2026-06-13 | tanh_hard | tanh_hard | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/tanh_hard/rtl/tanh_hard.v - tanh_hard.forward() is 'self.tick(); return input' - a stateless, polarity-agnostic identity pass-through. RTL is a single bare-name combinational module (o_out = i_in), no polarity split, no clock. Generator gen/gen_tanh_hard.py emits 68 golden vectors from the napl Python model (exhaustive both 1-bit values plus a 64-sample pseudo-random stream); self-checking tb tanh_hard_tb.v replays them. conda run -n napl make test OP=tanh_hard PASSes 68/68 bit-exactly. pp_delay=0 (purely combinational). Per batch overrides: did NOT edit tanh.py self.hw.pp_delay and did NOT run the Step 6 recorder; returning pipeline_delay for the workflow to finalize. Files: rtl/tanh_hard.v, tb/tanh_hard_tb.v, gen/gen_tanh_hard.py under /Users/diwu/Projects/napl/src/napl/hw/tanh_hard/. |
| 2026-06-13 | tanh_hub | tanh_hub | skipped | n/a |  |  | tanh_hub is a single-shot binary-domain (HUB) op, not a per-timestep streaming spike circuit. Its forward() is torch.nn.functional.hardtanh(input, -1.0, 1.0) applied once to a whole multi-bit float/fixed-point tensor: no self.tick(), no per-timestep spike stream, no reset()-advanced state, and it is tested/trained directly against nn.Hardtanh via STE (per CLAUDE.md's 'single-shot binary-domain' paradigm). There is no bit-serial 1-bit gate-level datapath to lower: hardtanh on a fixed-point word is a multi-bit saturation/clamp block whose bit-width and number format are defined by the surrounding HUB layer, not by this op, so no single scalar spike circuit reproduces forward(). Contrast with the streaming tanh_hard (which consumes 1-bit i_spike inside a @napl_sim_timesteps loop and is the kind of op that gets RTL like sigmoid_hard/relu_sat). Following the skill's 'skip honestly' rule (Step 4), status=skipped rather than forcing an unsound design. No files written. Source: src/napl/sim/operation/tanh_hard.py lines 29-42. |
| 2026-07-26 | tanh_p1 | tanh_p1 | verified | PASS | 0 | unipolar | Backfill verification from `src/napl/imp`: WIDTH=8; power-on and mid-stream reset; 3328/3328 vectors matched. |
| 2026-07-26 | tanh_pn | tanh_pn | verified | PASS | 1 | bipolar | Backfill verification from `src/napl/imp`: DEPTH=3; power-on and mid-stream reset; 3328/3328 vectors matched. |
| 2026-06-13 | uni2bi | uni2bi | verified | PASS | 0 |  | /Users/diwu/Projects/napl/src/napl/hw/uni2bi/rtl/uni2bi.v - uni2bi (unipolar-to-bipolar scaled-addition converter): single bare-name module (no polarity branch in the Python forward, so no _unipolar/_bipolar split). Implemented as a Mealy machine matching napl.sim.operation.uni2bi with width=3 (acc range [-4,3], pre-clamp sum [-3,5] -> 4-bit signed adder): addend=i_in?2:1, acc_clamped=clamp(acc+addend,-4,3), o_out=(acc_clamped>=2), acc<=acc_clamped-2*o_out. o_out is combinational (assign), the accumulator is the only register, reset (active-low i_rst_n) sets acc=0 matching reset(). gen drives the model per-cycle from reset() over a stimulus mixing runs of 1s/0s (to reach clamp rails) + a seeded random tail; tb replays cycle-by-cycle, checks combinational o_out then clocks to advance acc. make test OP=uni2bi PASS, 304/304 vectors bit-exact (outputs: 76 zeros, 228 ones). pp_delay=0: the only input->output path (i_in->o_out) is purely combinational; the accumulator register is internal state, not on an input->output path. Per batch overrides, did NOT edit the Python class self.hw.pp_delay and did NOT run the Step 6 recorder. |
| 2026-08-06 | avgpool2d | avgpool2d | verified | PASS | 0 | n/a (one variant) | Module layer, backfilled row. KERNEL_AREA=4 (kernel 2), DIVISOR 4 and 8 (divisor_override), LANES=192 over input (4, 3, 8, 8); 768 vectors as a unipolar block, a bipolar block, and the unipolar block replayed after a mid-stream reset of dirtied accumulators. Guarded on DIVISOR >= KERNEL_AREA. Every elaboration parameter is observable in a vector row: KERNEL_AREA and LANES through the golden-column width assertions, DIVISOR through the compared output. Bipolar stimulus range narrowed to [-1, 0.75] so its encoded rate is not a copy of the unipolar grid, with a generator assertion on it. |
| 2026-08-06 | conv | conv_unipolar, conv_bipolar | verified | PASS | 0 | unipolar + bipolar | Module layer, backfilled row. Input (1, 2, 6, 6), 3 out channels, 3x3 kernel, WIDTH=12; four geometries a (pad 1, bias, scale 19, 108 lanes), b (pad 0, no bias, scale 18, 48 lanes), c (pad 2, stride 2, dilation 2, bias, default scale 19, 27 lanes), d (c's geometry at an explicit scale of 9, 27 lanes); 1792 vectors over three reset-to-reset sequences plus a saturation block. Guarded on 2**(WIDTH-1) > ENTRY and LANES == output positions. a, b and c resolve the mapping's fan-in branch of SCALE and d the explicit-scale branch. Two saturation blocks now follow the three sequences, for 2452 vectors. The positive block charges geometry d's accumulators onto their 2047 clamp and drains them, which is what makes WIDTH observable: corrupting GEN_WIDTH 12 -> 11 fails at n=1359 on the d columns alone, 250 mismatches over 1792 vectors (measured before the negative block was appended), restoring it passes. The negative block rails the bias low as well, so an unpadded lane's popcount is 0 and the bipolar offset 5 drives geometry d's accumulator onto -2048 in 410 of its 460 silent cycles; the 200 spiking cycles that follow recharge it at 14 per cycle. That is what makes ACC_LO observable: halving it in add_any_bipolar.v fails at n=2332 on the bipolar d column, 79 mismatches over 2452 vectors, restoring it passes 2452/2452. In unipolar the offset is 0 and a carry only subtracts down to 0, so ACC_LO is unreachable there by construction; the generator asserts that too. The unipolar variant carries no pad port (a unipolar zero pad is a constant zero spike) and mapping.yaml maps pad_bits to no port for it. Geometry d was added under a per-column md5 comparison: over the original 768 rows the shared stimulus columns and the a, b and c output columns hash the same before and after, so only the new d columns moved. That comparison covers those 768 rows only. The 1024 saturation rows are stimulus the same change introduced, so they have no pre-change baseline and nothing about them follows from it. |
| 2026-08-06 | conv_pc | conv_pc_unipolar, conv_pc_bipolar | verified | PASS | 0 | unipolar + bipolar | Module layer, backfilled row. Same geometry as conv with the add_any stage removed and the popcount published on a LANES x COUNT_W count bus, COUNT_W=5; 768 vectors, combinational, no clock or reset port, so the three-sequence replay checks the Python encoders rather than an RTL reset. Guarded on COUNT_W == clog2(ENTRY + 1) and LANES == output positions. Every elaboration parameter is observable: geometry and COUNT_W through the guards, LANES through the count-bus width assertion. Neither of its ENTRY values is a power of two, so the + 1 in clog2(ENTRY + 1) is pinned by linear_pc's no-bias arm instead. The unipolar variant carries no pad port. |
| 2026-08-06 | conv_ugemm | conv_ugemm_unipolar, conv_ugemm_bipolar | verified | PASS | 0 | unipolar + bipolar | Module layer, backfilled row. Input (1, 1, 4, 4), 2 out channels, 2x2 kernel, SEQ_WIDTH=8, WIDTH=12; weight and bias are held fixed-point codes generated in hardware (internal_encode), bipolar pad is a jkff toggle, unipolar pad a zero spike; four geometries (50, 18, 8 and 18 lanes, scale 13, 12, 13 and an explicit 6); 1138 vectors over three reset-to-reset sequences plus a saturation block. Guarded on 2**(WIDTH-1) > ENTRY and LANES == output positions. Every elaboration parameter is observable, WIDTH included. | 2026-08-06 reversal: the WIDTH row recorded here was wrong, and so was the arithmetic under it. The excursion is bounded by the pre-carry sum, which add_any clamps before it subtracts a carry, and the budget is 2**SEQ_WIDTH spiking cycles rather than 2**SEQ_WIDTH - 1: the sequence index is read on every timestep and advanced only on an enabling one, so a maximal run uses index 0 through 2**SEQ_WIDTH - 1. The reachable unipolar peak is therefore (ENTRY - SCALE) * 2**SEQ_WIDTH + SCALE, which at the old ENTRY 5 and SCALE 1 is 4 * 256 + 1 = 1025, past the 1023 clamp of an 11-bit accumulator rather than short of it. A stimulus that shows a clamp must charge and then drain, and a run that spends its whole spiking budget has no cycle left to drain in, so at ENTRY 5 only 255 charging cycles are usable and the peak drops to 1021: pinnable in arithmetic, not in a vector row. The configuration was resized instead. IN_CHANNELS is 3, so K is 12, and geometry d (padding 0, stride 1, dilation 1, with bias, explicit SCALE 6) holds railed operands in vec/conv_ugemm_operand_d.hex: weights at the top code and bias at 0, so a railed input gives each of its 18 lanes a partial sum of 12 and a silent input gives 0. The block charges 180 cycles at 6 per cycle to a pre-carry 1086 and drains 190 at 6 per cycle. Measured against an instrumented model: unipolar stored peak 1080, bipolar stored peak 450 and trough -497. (Corrected 2026-08-06: the bipolar figure was recorded as 456, which is the pre-carry sum 2.5 * 180 + 6; the stored value is that less one carry of SCALE 6.) Fault-injection matrix below. |
| 2026-08-06 | linear | linear_unipolar, linear_bipolar | verified | PASS | 0 | unipolar + bipolar | Module layer, backfilled row. 8 lanes x 16 in_features, WIDTH=12, scale 17 (bias), 16 (no bias) and an explicit 9 in both polarities; 2296 vectors over three reset-to-reset sequences plus a positive and a negative saturation block. Guarded on 2**(WIDTH-1) > ENTRY. The scale-9 arms are the only accumulators that move at all: with scale == entry the bipolar offset (entry - scale) / 2 is 0 and a carry subtracts the whole entry, so the accumulator stays inside [0, entry - 1]. The scale-9 bipolar arm (column out_b_s) was added here, so WIDTH is now pinned from both polarities instead of one unipolar column: corrupting GEN_WIDTH 12 -> 11 fails at n=1365, 285 mismatches over 2296 vectors, 115 of them unipolar and 170 bipolar, restoring it passes. The negative block rails the bias low as well, so the popcount is 0 and the bipolar offset 4 drives the scale-9 accumulator onto -2048 in 512 of its 560 silent cycles; the 200 that follow recharge it at 13 per cycle. That is what makes ACC_LO observable: halving it in add_any_bipolar.v fails at n=2183 on out_b_s, 85 mismatches over 2296 vectors, restoring it passes 2296/2296. In unipolar the offset is 0 and a carry only subtracts down to 0, so ACC_LO is unreachable there by construction; the generator asserts that too. |
| 2026-08-06 | linear_pc | linear_pc_unipolar, linear_pc_bipolar | verified | PASS | 0 | unipolar + bipolar | Module layer, backfilled row. 8 lanes x 16 in_features, count bus COUNT_W=5 for both the with-bias (ENTRY 17) and no-bias (ENTRY 16) arms; 768 vectors, combinational, no clock or reset port. Guarded on COUNT_W == clog2(ENTRY + 1). Every elaboration parameter is observable. The no-bias arm's ENTRY is a power of two, so clog2(ENTRY) would be one bit narrower: that arm is what pins the + 1, now asserted in the generator. |
| 2026-08-06 | linear_ugemm | linear_ugemm_unipolar, linear_ugemm_bipolar | verified | PASS | 0 | unipolar + bipolar | Module layer, backfilled row. 8 lanes x 16 in_features, SEQ_WIDTH=8, WIDTH=12, scale 17, 16 and an explicit 5 in both polarities; held operand codes in vec/linear_ugemm_operand_{u,b}.hex with the scaled arms' railed codes in _s.hex, sequence ROM from the model; 1477 vectors over one sequence, a dirtied replay, and a positive and a negative saturation block. Guarded on 2**(WIDTH-1) > ENTRY. WIDTH is now observable: the earlier bound recorded here (2**SEQ_WIDTH * ENTRY / 4) was wrong, since SCALE == ENTRY made the bipolar offset 0 and confined the accumulator to [0, ENTRY - 1]. The scale-5 arms (columns out_u_s, out_b_s) hold their weights at the top code and their bias at 0, so a spiking input gives a partial sum of 16 and a silent one 0. The unipolar arm climbs at 11 per cycle and reaches the 2047 clamp in 186 of the 255 cycles the sequence index allows; the bipolar arm climbs and drains at the offset 6, which the same budget stops at 1275 and -1530, past the 1023 and -1024 of a width one bit narrower. Corrupting GEN_WIDTH 12 -> 11 fails at n=860, 126 mismatches over 1477 vectors, 52 unipolar and 74 bipolar; halving ACC_LO in add_any_bipolar.v fails at n=1380, 51 mismatches, all on out_b_s; restoring each passes 1477/1477. The generator asserts the unipolar peak, both bipolar excursions, and that each block parts from a shadow model one bit narrower. Input operand codes are drawn per polarity so the bipolar input column is not a copy of the unipolar one. |
| 2026-08-06 | mgu | mgu_bipolar | verified | PASS | 0 | bipolar only | Module layer, backfilled row. 3 lanes x 4 inputs, WIDTH=10, SEQ_WIDTH=8, SR_WIDTH=6; composes linear_bipolar, sigmoid_hard, mul_ugemm_bipolar, mul_ugemm_sr_bipolar and add_any_bipolar; 1359 vectors over three reset-to-reset sequences (dirty run of 77 steps, coprime with the shift-register depth) plus a positive-rail and a negative-rail block. Guarded on SEQ_WIDTH > SR_WIDTH. | 2026-08-06 negative rail: the module was listed as covering the accumulator's negative clamp while no stimulus reached it. Both gate biases were railed high, which keeps one addend on every cycle, so the forget-gate accumulator only ever charged. Proof of the gap: with the pre-change 1024-vector set, halving ACC_LO in add_any_bipolar.v PASSes 1024/1024. The negative-rail block rails both gate biases low instead, which takes the gate popcount to zero on a railed-low input, so the accumulator falls at the offset (ENTRY - SCALE) / 2 = 3.5 per cycle and reaches acc_min = -512 in 147 cycles; 180 drain cycles then 160 recharge cycles cover the with-bias cell and the no-bias one, whose ENTRY is 7 and which falls at 3 and needs 171. The positive block was also lengthened from 128 charge cycles to 146, which is what reaches the elaborated clamp at 511 rather than only the 255 of a narrower width; its 105 drain cycles keep the fg-driven mul_ugemm index inside its 2**SEQ_WIDTH budget. Injections against the new set: halved ACC_LO fails at n=1265, 90 mismatches over 1359 vectors; GEN_WIDTH 10 -> 9 fails at n=987, 107 mismatches (3 before this change). Both restored, PASS 1359/1359. Fault-injection entries below. |
| 2026-08-06 | linear_gaines1, linear_gaines2 | linear_gaines_unipolar, linear_gaines_bipolar | verified | PASS | 0 | unipolar + bipolar | Module layer, hand-written under RULE_IMP.md. One module per polarity serves BOTH classes: 8 lanes x 16 in_features, TIMESTEP 256 (SEQ_WIDTH 8), SCALE_WIDTH 4, DEPTH 8, three arms per polarity (scaled with bias, scaled no bias, non-scaled with bias); 870 vectors over three reset-to-reset sequences (dirty run of 129 steps, coprime with both the 256-entry threshold period and the 16-entry scaled period) plus a 102-cycle railed block of three 34-cycle blocks. Under a matched configuration the two classes build the same w_num_seq, b_encoder and reference_encode tables and the generator asserts all three, then requires the two models to agree bit for bit on every timestep of every arm, so the golden column is their common output; linear_gaines2's scale_seq and its unipolar non-scaled counter are dead and are not lowered. Four mapping.yaml entries (class x polarity) point at the two RTL modules; because the module base name is not the class name, _select_entry in syn/translate.py gained a fallback to the entry carrying the polarity suffix when exactly one does. Held operand ports, as in conv_ugemm: weight and bias arrive as fixed-point codes in vec/linear_gaines_operand.hex and the ROM images carry only the threshold sequences (vec/linear_gaines_w.hex, W_LEN x IN_FEATURES, shared outside the lane generate; vec/encode_rom.hex for the bias; vec/gaines_rom.hex for the scaled adder). Both classes now carry internal_encode = True, which is descriptive: their sim tests pass unchanged and no number moved. There are IN_FEATURES x LANES weight comparators, IN_FEATURES inside each lane; only the threshold ROM is shared. The lane composes mul_gaines_<polarity> per feature, encode for the bias, and either encode (scaled) or add_gaines at SCALED = 0 (non-scaled unipolar); the non-scaled bipolar counter is written in the module, the operation layer holding no such circuit. Guarded on SCALE_WIDTH + 1 >= clog2(ENTRY + 1) (ERROR_linear_gaines_SCALE_WIDTH_too_small_for_ENTRY, both polarities enrolled in test_guards.py). requires: the weight rows equal the lane count on all four entries, and a scaled arm on an lfsr image needs entry >= 3, since get_lfsr_seq rejects width < 2, on the two linear_gaines2 entries only -- linear_gaines1 rejects any non-sobol-family generator in its own constructor, so that clause could never fire on its entries. Both clauses have negative tests in tests/syn/test_translate.py, and the lfsr test drives a linear_gaines2 node, so dropping the clause from the linear_gaines1 entries leaves covered == requires_clauses() closed. Every elaboration parameter is observable, red-then-green: IN_FEATURES and LANES through the golden-column width assertions, SEQ_WIDTH, SCALE_WIDTH, HAS_BIAS and SCALED through the compared columns, and DEPTH through the railed block alone. The only accumulator is the non-scaled bipolar counter, cnt = clamp(cnt + 2*count - ENTRY, 0, 2**DEPTH - 1) from a reset value of 2**(DEPTH-1), with no carry, so the clamp acts on the same value that is stored and compared and the excursion arithmetic recorded for the add_any accumulators does not apply. Derived from that recurrence: lane 0 holds railed weight and bias codes, so an all-high input block gives count = ENTRY = 17 and climbs by 17 per cycle, reaching 255 in 8 cycles, and an all-low block leaves the railed bias addend on, so its count is 1 and it falls by 15 per cycle, reaching 0 in 17. That fall rate is the railed-high bias; a lane with no bias addend falls by ENTRY instead. An instrumented run of the model measured peak 255 at cycle 8 and floor 0 at cycle 17, agreeing with the derivation, and the generator asserts both rails. The block length is now derived from those rates, 2 * ceil(CNT_MAX / (ENTRY - 2)) = 34 cycles, rather than the hardcoded 40 it was. Both clamp arms are genuinely exposed, each proven on its own: raising the high clamp one bit fails at n=811 (17 mismatches), and comparing the low clamp against an unsigned zero fails at n=844 (15 mismatches) -- that second one was a real bug in the first draft, which the railed block caught. With the railed block removed both injections and GEN_DEPTH 8 -> 7 all PASS 768/768, so the railed block is what exposes them. What exposes each is a different part of that block, not 'both rails being charged': the opening charge block yields no DEPTH mismatch at all, the drain block is what parts the column (with the third block dropped, GEN_DEPTH 8 -> 7 still fails at n=807, 4 mismatches over 836 vectors), and the third block is the only stimulus that leaves the low rail, which is what makes the low clamp observable (deleting the low clamp fails at n=844 with the block present and PASSes 836/836 with it dropped). Fault-injection matrix below.

## Fault-injection matrix

Each row below was run: the named fault was applied to the named file, `conda run -n napl make test
MODULE=<module>` was run from `src/napl/imp/`, and the file was restored. The signature is the first
mismatch line the testbench printed plus its total, copied from that run (2026-08-06, against the
vector sets recorded in the table above). The `add_any` rows were run the same way with
`make test OP=add_any`. An injection into a generated `*_params.vh` was rerun without regenerating
the vectors, so the header no longer matches the model the vectors came from. A row that does not fire is recorded as such, since a
silent injection is a gap in the vector set rather than a passing result.

Two different injections appear below and they answer different questions. A **halved
constant** moves the condition as well as the result, so it tests only whether the
constant's *value* is observable; it cannot decide whether a branch is ever entered,
in either direction. A **branch oracle** keeps the condition exactly as written and
changes what the taken arm produces, so a green run proves the branch was never
entered. Every reachability claim below names which of the two produced it, and only
the oracle rows carry weight for a dead-or-live verdict. RULE_IMP.md's verification
gates state the rule.

The nine `conv_ugemm` injections an earlier review cited existed only in a transcript, in no file, so
they could not be reproduced. The set below is the one actually run here, chosen to hit each premise
the module's exactness argument rests on: the pad stream, the patch geometry, the lane packing, the
scaled accumulator, the `mul_ugemm` sequence-index update, and the `encode` counter.

| Module | File | Injected fault | Observed failure signature |
|--------|------|----------------|----------------------------|
| conv_ugemm | module/conv_ugemm/rtl/conv_ugemm_bipolar.v | bipolar pad tap tied to `1'b0` instead of the jkff toggle | `FAIL n=2 bipolar pad1 bias : got 0000000001... exp 1111110001...`; 1498 mismatches over 768 vectors |
| conv_ugemm | module/conv_ugemm/rtl/conv_ugemm_bipolar.v | pad jkff held at K = 0, so the pad stream sticks at 1 | `FAIL n=4 bipolar pad1 bias`; 1488 mismatches over 768 vectors |
| conv_ugemm | module/conv_ugemm/rtl/conv_ugemm_bipolar.v | bias `encode` FRAC widened to `SEQ_WIDTH + 1` | does not fire: PASS 768/768. `$readmemb` zero-extends the SEQ_WIDTH-bit ROM lines into the wider word, so the comparator reads the same values. Only a narrower FRAC truncates them. |
| conv_ugemm | module/conv_ugemm/rtl/conv_ugemm_bipolar.v | bias `encode` FRAC narrowed to `SEQ_WIDTH - 1` | `FAIL n=3 bipolar pad1 bias : got 0000001010... exp 0000000000...` and `FAIL n=3 bipolar strided : got 10000001 exp 10000000` |
| conv_ugemm | module/conv_ugemm/rtl/conv_ugemm_bipolar.v | lane accumulator `SCALE` off by one | `FAIL n=1 bipolar pad1 bias : got 0000000000... exp 0000001110...`; 2250 mismatches over 768 vectors |
| conv_ugemm | module/conv_ugemm/rtl/conv_ugemm_bipolar.v | patch tap order transposed (kernel row and column swapped) | `FAIL n=7 bipolar pad1 bias`; 2090 mismatches over 768 vectors |
| conv_ugemm | module/conv_ugemm/rtl/conv_ugemm_bipolar.v | padding added instead of subtracted on the input row index | `FAIL n=1 bipolar pad1 bias`; 1512 mismatches over 768 vectors |
| conv_ugemm | module/conv_ugemm/rtl/conv_ugemm_bipolar.v | lane index packs out_col before out_row | `FAIL n=3 bipolar pad0 nobias : got 101000110000000000 exp 101001100000000000`; 2004 mismatches over 768 vectors |
| conv_ugemm | operation/mul_ugemm/rtl/mul_ugemm_bipolar.v | sequence index advanced by the comparator output instead of `i_input_0` | `FAIL n=2 bipolar pad1 bias`; 2268 mismatches over 768 vectors |
| conv_ugemm | operation/encode/rtl/encode.v | counter made data-dependent (advances on `o_spike` instead of by one) | `FAIL n=8 unipolar pad1 bias`; 2498 mismatches over 768 vectors |
| mgu | module/mgu/rtl/mgu_bipolar.v | n gate reads `i_hx_spike` instead of `fg_hx` | `FAIL n=3 nobias : got 111 exp 101`; 815 mismatches over 1024 vectors |
| mgu | operation/mul_ugemm_sr/rtl/mul_ugemm_sr_bipolar.v | `reg_q` reset to zeros instead of `index % 2` | `FAIL n=9 nobias : got 111 exp 101`; 628 mismatches over 1024 vectors |
| add_any | operation/add_any/rtl/add_any_bipolar.v | `ACC_LO` halved to `-(2 ** WIDTH) / 2` | `FAIL cycle 18526 bipolar: i_input=11111111 got 1 exp 0`; 12 mismatches over 18622 vectors |
| add_any | operation/add_any/rtl/add_any_unipolar.v | `ACC_LO` halved to `-(2 ** WIDTH) / 2` | did not fire: PASS 18622/18622. The unipolar offset is 0 and a carry only subtracts down to 0, so the accumulator never goes negative at any stimulus; the clamp was dead logic, not untested logic, and has since been removed (see the add_any row above). The bipolar row directly above is the same injection into the reachable clamp, which fails, so the asymmetry is the evidence. |
| linear | operation/add_any/rtl/add_any_bipolar.v | `ACC_LO` halved to `-(2 ** WIDTH) / 2` | `FAIL n=2183 bipolar scale=9 : got 11111111 exp 00000000`; 85 mismatches over 2296 vectors |
| linear | module/linear/vec/linear_params.vh | `GEN_WIDTH` 12 -> 11 | `FAIL n=1365 bipolar scale=9 : got 00000000 exp 11111111`; 285 mismatches over 2296 vectors, 115 on out_u_s and 170 on out_b_s |
| conv | operation/add_any/rtl/add_any_bipolar.v | `ACC_LO` halved to `-(2 ** WIDTH) / 2` | `FAIL n=2332 bipolar scaled : got 101010101101010101101010101 exp 101000101101000101101000101`; 79 mismatches over 2452 vectors |
| linear_ugemm | operation/add_any/rtl/add_any_bipolar.v | `ACC_LO` halved to `-(2 ** WIDTH) / 2` | `FAIL n=1380 bipolar scale=5 : got 11111111 exp 00000000`; 51 mismatches over 1477 vectors, all on out_b_s |
| linear_ugemm | module/linear_ugemm/vec/linear_ugemm_params.vh | `GEN_WIDTH` 12 -> 11 | `FAIL n=860 bipolar scale=5 : got 00000000 exp 11111111`; 126 mismatches over 1477 vectors, 52 on out_u_s and 74 on out_b_s |
| conv_ugemm | module/conv_ugemm/vec/conv_ugemm_params.vh | `GEN_WIDTH` 12 -> 11 | `FAIL n=1118 unipolar scaled : got 000000000000000000 exp 111111111111111111`; 11 mismatches over 1138 vectors, all in the drain half of the saturation block; restored, PASS 1138/1138 |
| conv_ugemm | operation/add_any/rtl/add_any_bipolar.v | `ACC_LO` halved to `-(2 ** WIDTH) / 2` | does not fire: PASS 1138/1138. The unipolar accumulator has no negative clamp to halve (its offset is 0 and a carry only subtracts down to 0), and the bipolar one moves at the offset 3.5 per cycle, so the 2**SEQ_WIDTH - 1 silent cycles available take it to -497, short of the -1024 a halved clamp would hold. |
| mgu | operation/add_any/rtl/add_any_bipolar.v | `ACC_LO` halved to `-(2 ** WIDTH) / 2` | `FAIL n=1265 nobias : got 100 exp 000`; 90 mismatches over 1359 vectors; restored, PASS 1359/1359 |
| mgu | operation/add_any/rtl/add_any_bipolar.v | same, against the pre-change 1024-vector set | does not fire: PASS 1024/1024. The rail was listed as covered with no stimulus reaching it; the negative-rail block is what closed that. |
| uni2bi | operation/uni2bi/rtl/uni2bi.v | low clamp `ACC_MIN` halved to `-(1 << (WIDTH-2))` | does not fire: PASS 3329/3329. The addend is 1 or 2, so the accumulator never goes negative; the arm was dead logic and has been removed. |
| uni2bi | operation/uni2bi/rtl/uni2bi.v | high clamp `ACC_MAX` halved to `(1 << (WIDTH-2)) - 1` (**halved constant**) | `FAIL cyc=3 i_input=0 : got 0 exp 1`; 2431 mismatches over 3329 vectors. Superseded: this row was read as proof that the arm is live, which a halved constant cannot show. Halving moved the clamp down onto reachable sums; the arm at its own value is dead. |
| uni2bi | operation/uni2bi/rtl/uni2bi.v | high clamp result poisoned to 0, condition `sum > ACC_MAX` unchanged (**branch oracle**) | does not fire: PASS 3329/3329. The branch is never entered, agreeing with the BFS verdict, and the arm has been removed. |
| sigmoid_hard | operation/sigmoid_hard/rtl/sigmoid_hard.v | low clamp `ACC_MIN` halved from `-5'sd4` to `-5'sd2` | does not fire: PASS 6656/6656. The addend `i_input + 1` is 1 or 2, so the accumulator stays in [0, 3]; the arm was dead logic and has been removed. |
| sigmoid_hard | operation/sigmoid_hard/rtl/sigmoid_hard.v | high clamp `ACC_MAX` halved from `5'sd3` to `5'sd1` (**halved constant**) | `FAIL cycle 2: i_rst_n=1 i_input=0 got 0 exp 1`; 5121 mismatches over 6656 vectors. Superseded for the same reason as the `uni2bi` row above. |
| sigmoid_hard | operation/sigmoid_hard/rtl/sigmoid_hard.v | high clamp result poisoned to `5'sd0`, condition `raw_sum > ACC_MAX` unchanged (**branch oracle**) | does not fire: PASS 6656/6656. The branch is never entered, agreeing with the BFS verdict, and the arm has been removed. |
| div_iscb | operation/div_iscb/rtl/div_iscb_uni2bi.v | low clamp `ACC_MIN` halved from `-5'sd4` to `-5'sd2` | does not fire: PASS 8448/8448. The step is 1 or 2, so the helper's accumulator stays in [0, 3]; the arm was dead logic and has been removed. |
| div_iscb | operation/div_iscb/rtl/div_iscb_uni2bi.v | high clamp `ACC_MAX` halved from `5'sd3` to `5'sd1` (**halved constant**) | `FAIL[bi] t=0 dd=0 ds=0 : got 0 exp 1`; 6378 mismatches over 8448 vectors. Superseded for the same reason as the `uni2bi` row above. |
| div_iscb | operation/div_iscb/rtl/div_iscb_uni2bi.v | high clamp result poisoned to `5'sd0`, condition `acc_sum > ACC_MAX` unchanged (**branch oracle**) | does not fire: PASS 8448/8448. The branch is never entered, agreeing with the BFS verdict, and the arm has been removed. |
| div_iscb | operation/div_iscb/rtl/div_iscb_bi2uni.v | high clamp `ACC_MAX` halved from `4'sd3` to `4'sd1` | does not fire: PASS 8448/8448. The emit threshold is 1, so a fire leaves the accumulator at or below 2 and the sum at or below 3; the arm was dead logic and has been removed. |
| div_iscb | operation/div_iscb/rtl/div_iscb_bi2uni.v | low clamp `ACC_MIN` halved from `-4'sd4` to `-4'sd2` (**halved constant**) | `FAIL[bi] t=775 dd=1 ds=0 : got 1 exp 0`; 1555 mismatches over 8448 vectors. Superseded: the halved clamp at -2 is reachable while the real one at -4 was not, so this row proved only that -2 is observable. Reclassified LIVE BUT UNEXERCISED, see the rows below. |
| div_iscb | operation/div_iscb/rtl/div_iscb_bi2uni.v | low clamp result poisoned to `4'sd3`, condition `acc_sum < ACC_MIN` unchanged (**branch oracle**), against the pre-change 8448-vector set | does not fire: PASS 8448/8448. The branch was never entered. Through `div_iscb_bipolar` it cannot be: an exhaustive BFS over the composed (signabs, bi2uni) state space reaches 20 states with a minimum accumulator of -3, because signabs never holds its magnitude output low for more than three cycles in a row. |
| div_iscb | operation/div_iscb/rtl/div_iscb_bi2uni.v | the same branch oracle, against the 8456-vector set with the helper driven standalone | `FAIL[b2u] t=4 in=0 : got 1 exp 0`; 1057 mismatches over 8456 vectors; restored, PASS 8456/8456. Five consecutive silent cycles walk the standalone accumulator from 0 to -4 and present -5 to the clamp on the fifth. The arm is live and now exercised. |
| relu_sat | operation/relu_sat/rtl/relu_sat.v | sub-stage high clamp threshold halved from `6'sd6` to `6'sd3` | does not fire: PASS 6672/6672. `acc_sub` never exceeds 1, so `sum_sub` never exceeds 2; the arm was dead logic and has been removed. |
| relu_sat | operation/relu_sat/rtl/relu_sat.v | add-stage high clamp threshold halved from `6'sd6` to `6'sd3` | does not fire: PASS 6672/6672. `out_sub` cannot fire on two consecutive cycles, which caps `acc_add` at 2 and `sum_add` at 4; the arm was dead logic and has been removed. |
| relu_sat | operation/relu_sat/rtl/relu_sat.v | add-stage low clamp halved from `-6'sd8` to `-6'sd4` | does not fire: PASS 6672/6672. The add-stage addend, `2*out_sub + 1`, is always positive, so `acc_add` never goes negative; the arm was dead logic and has been removed. |
| relu_sat | operation/relu_sat/rtl/relu_sat.v | sub-stage low clamp halved from `-6'sd8` to `-6'sd4` | `FAIL cycle 801: rst=0 i_input=1 got 1 exp 0`; 14 mismatches over 6672 vectors. This is the LIVE fourth arm and it stays: a silent input decrements `acc_sub` without limit, and the vector set drives it onto -8. Its three siblings above, injected the same way in the same run, all stayed green. |
| relu_sat | operation/relu_sat/rtl/relu_sat.v | sub-stage low clamp result poisoned to `-6'sd4`, condition `sum_sub < -6'sd8` unchanged (**branch oracle**) | `FAIL cycle 805: rst=0 i_input=1 got 1 exp 0`; 96 mismatches over 6672 vectors. The oracle agrees with the halved-constant row above: this arm is genuinely entered, and it stays. |

The last two injections are the pair `conv_ugemm`'s exactness argument cites: the `mul_ugemm` index
update reading only the patch spike, and `mul_ugemm_sr`'s alternating reset state. The `encode` row
is the third premise, that the bias comparator's counter is data-independent.

## Elaboration-parameter observability

Every `` `define GEN_<PARAM> `` in each module's generated parameter header was corrupted by one,
without regenerating the vectors, and the co-simulation rerun. A corruption that still PASSes is a
parameter no vector row is sensitive to. Parameters swept, per module:

| Module | Parameters swept | Unobservable |
|--------|------------------|--------------|
| avgpool2d | KERNEL_AREA, DIVISOR, DIVISOR_D8, LANES | none |
| linear | IN_FEATURES, LANES, WIDTH, SCALE, SCALE_NB, SCALE_S | none |
| linear_pc | IN_FEATURES, LANES, COUNT_W, COUNT_W_NB | none |
| linear_ugemm | IN_FEATURES, LANES, SEQ_WIDTH, WIDTH, SCALE, SCALE_NB, SCALE_S | none |
| conv | BATCH, IN_CHANNELS, IN_H, IN_W, OUT_CHANNELS, KERNEL_H, KERNEL_W, WIDTH, and PADDING / STRIDE / DILATION / HAS_BIAS / SCALE / LANES per geometry | none |
| conv_pc | the same geometry set with COUNT_W per geometry in place of WIDTH and SCALE | none |
| conv_ugemm | the same geometry set plus SEQ_WIDTH and WIDTH | none |
| mgu | LANES, IN_SIZE, WIDTH, SEQ_WIDTH, SR_WIDTH | none |

Geometry and lane parameters fail through the elaboration guards or the golden-column width
assertions; scales and divisors fail through the compared outputs. `linear`, `conv`, `conv_ugemm`,
`linear_ugemm`, and `mgu` carry a saturation block that pins WIDTH from below: it drives the accumulator onto the clamp the
elaborated width sets, so a narrower WIDTH clips the excursion and changes a compared output, proven
red then green by the same corruption. A wider WIDTH stays unobservable, because the stimulus stops
at the elaborated clamp and never reaches the wider one:

| Module | Corruption | Before the saturation block | After |
|--------|-----------|-----------------------------|-------|
| linear | GEN_WIDTH 12 -> 11 | PASS 768/768 | `FAIL n=1407 unipolar scale=9 : got 00000000 exp 11111111`; restored, PASS 1536/1536 |
| conv | GEN_WIDTH 12 -> 11 | PASS 768/768 | `FAIL n=1359 bipolar scaled : got 010101010... exp 010111010...`; restored, PASS 1792/1792 |
| mgu | GEN_WIDTH 10 -> 9 | PASS 768/768 | `FAIL n=987 bias : got 111 exp 000`; 107 mismatches; restored, PASS 1359/1359 |
| conv_ugemm | GEN_WIDTH 12 -> 11 | PASS 768/768 (three-sequence set, no block) | `FAIL n=1118 unipolar scaled : got 0...0 exp 1...1`; 11 mismatches; restored, PASS 1138/1138 |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_IN_FEATURES` 16 -> 15 | `FAIL n=0 in_u: golden column is 16 bits, port is 15`; 3536 mismatches over 870 vectors |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_LANES` 8 -> 7 | `FAIL n=0 out_u_a: golden column is 8 bits, port is 7`; 6237 mismatches over 870 vectors |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_SEQ_WIDTH` 8 -> 7 | `FAIL n=1 unipolar scaled bias : got 11111110 exp 11111111`; 3587 mismatches over 870 vectors |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_SCALE_WIDTH` 4 -> 3 | build fails with `ERROR_linear_gaines_SCALE_WIDTH_too_small_for_ENTRY`: the elaboration guard catches it before the comparator can truncate the count |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_SCALE_WIDTH` 4 -> 5 | `FAIL n=17 unipolar scaled bias : got xxxxxxxx exp 11111111`; 1728 mismatches over 870 vectors |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_DEPTH` 8 -> 7 | `FAIL n=807 bipolar counter : got 10101000 exp 10101001`; 8 mismatches over 870 vectors, all inside the railed block |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_DEPTH` 8 -> 7 with the railed block removed | does not fire: PASS 768/768. Away from its clamps the counter's trajectory relative to cnt_half is the same at any DEPTH, so the railed block is the only stimulus that exposes it. |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_HAS_BIAS_B` 0 -> 1 | `FAIL n=2 unipolar scaled nobias : got 00000001 exp 00000000`; 167 mismatches over 870 vectors |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_SCALED_C` 0 -> 1 | `FAIL n=2 unipolar non-scaled : got 00000001 exp 11111111`; 1523 mismatches over 870 vectors |
| linear_gaines | module/linear_gaines/vec/linear_gaines_params.vh | `GEN_VECTORS` 870 -> 869 | `FAIL linear_gaines: consumed 870 vectors, generator wrote 869` |
| linear_gaines | module/linear_gaines/vec/linear_gaines.vec | last row deleted | `FAIL linear_gaines: consumed 869 vectors, generator wrote 870` |
| linear_gaines | module/linear_gaines/vec/linear_gaines.vec | one `0` prepended to `in_u` on one row | `FAIL n=1 in_u: golden column is 17 bits, port is 16` |
| linear_gaines | module/linear_gaines/vec/linear_gaines.vec | one character removed from `in_u` on one row | `FAIL n=1 in_u: golden column is 15 bits, port is 16` |
| linear_gaines | module/linear_gaines/tb/linear_gaines_tb.v | `MAX_CHARS` sized to the widest column instead of widest + 1, with the over-long row above | does not fire: PASS 870/870. `$fscanf("%s")` truncates the leading character, so the over-long column reads back at the expected length. The spare character is what makes the width check able to reject it. |
| linear_gaines | module/linear_gaines/rtl/linear_gaines_bipolar.v | counter high clamp raised one bit (`CNT_MAX` = 2**(DEPTH+1) - 1) | `FAIL n=811 bipolar counter : got 10101001 exp 10101000`; 17 mismatches over 870 vectors |
| linear_gaines | module/linear_gaines/rtl/linear_gaines_bipolar.v | same, with the railed block removed | does not fire: PASS 768/768, so the high rail is exposed by the railed block alone |
| linear_gaines | module/linear_gaines/rtl/linear_gaines_bipolar.v | counter low clamp compared against an unsigned zero (the first draft's bug) | `FAIL n=844 bipolar counter : got 00111000 exp 00111001`; 15 mismatches over 870 vectors |
| linear_gaines | module/linear_gaines/rtl/linear_gaines_bipolar.v | same, with the railed block removed | does not fire: PASS 768/768, so the low rail is exposed by the railed block alone |

`linear_ugemm` and `conv_ugemm` pin `WIDTH` one bit from below rather than against the elaborated
clamp: their `mul_ugemm` cycle budget stops the excursion between the two. RULE_IMP.md's
module-layer rule 9 states the arithmetic.

## `rep_values` bipolar range, 2026-08-06

`operation/_gen_common.py`'s `rep_values` drew its default bipolar operands from `[-1, 1]`, whose
probabilities `(v + 1) / 2` are exactly the unipolar `[0, 1]` grid, so a generator that built both
polarity streams from the helper got byte-identical stimulus in both arms. The default bipolar range
is now `[-1, 0.5]`, the offset `gen_div_gaines.py` and `gen_mul_ugemm_sr.py` already passed by hand,
and the three range branches collapse into one. Unipolar stimulus is unchanged, and every call site
that passes an explicit `value_range` is unchanged, `value_range` overriding rather than stacking.
The call-site asserts that catch identical polarity arms stay in place.

Grids, from the helper after the change:

```
rep_values('unipolar')         = [0.0, 0.25, 0.5, 0.75, 1.0, 0.1667, 0.3333, 0.6667, 0.8333, 0.4963, 0.7682, 0.0885, 0.132]
rep_values('bipolar')          = [-1.0, -0.625, -0.25, 0.125, 0.5, -0.75, -0.5, 0.0, 0.25, -0.2556, 0.1523, -0.8673, -0.802]
bipolar as probability (v+1)/2 = [0.0, 0.1875, 0.375, 0.5625, 0.75, 0.125, 0.25, 0.5, 0.625, 0.3722, 0.5762, 0.0664, 0.099]
grids equal as probabilities: False
```

`sigmoid_hard` is the only regenerated operation whose generator draws both polarity grids
(`rep_values("bipolar") + rep_values("unipolar")`). Its two encoded halves were byte-identical
before and now differ in 1283 of 3328 cycles. The other nineteen regenerated operations drive one
bipolar stream, which the change shifts and rescales rather than splits.

`add_any` builds its own probability-equivalent rows in `test_values()` and never calls the helper:
its vec is byte-identical before and after (`c37bd91a743919d6d9df7c6da68635f42e3924a4`), so the
documented single-stimulus, two-circuit intent is untouched.

`operation/max_rc/gen/gen_max_rc.py` passed a `timestep` key the `max_rc` model no longer accepts, so
`make test OP=max_rc` failed before this change for an unrelated reason. The stale key is dropped.

Regenerated vec files (20 operations, no module vec changed, no vector count changed):
`decode`, `dff`, `div_iscb`, `gt_rc`, `inhibit`, `lt_rc`, `max_rc`, `min_rc`, `relu_cnt`, `relu_sat`,
`relu_shiftreg`, `relu_tc`, `shiftreg`, `sigmoid_hard`, `signabs`, `signabs_interleave`,
`signabs_shiftreg`, `square_dff`, `tanh_hard`, `tanh_pn`.

Gates, from `src/napl/imp/`: all 43 `make test OP=` targets PASS, all 9 `make test MODULE=` targets
PASS, `make guards` and `tests/syn/test_translate.py` both print `Test passed.`, and `iverilog
-g2001 -Wall` emits zero warnings across all 52 builds.

## Accumulator excursion arithmetic, 2026-08-06

The bound this file and RULE_IMP.md quoted for a saturating `add_any` was wrong in three ways, and a
recorded `conv_ugemm` result rested on it. The corrected form is derived from `add_any.forward()`
and validated against the model.

Per timestep the model computes `acc_delta = partial - offset` with `offset = (entry - scale) / 2`
in bipolar and 0 in unipolar, adds it to the stored state, clamps that **pre-carry** sum to
`[acc_min, acc_max]`, compares the clamped value against `scale`, and subtracts one `scale` if it
fired. So with the partial sum railed at `p` and `drift = p - offset`, a state that fires every
cycle climbs at `rate = drift - scale`, and after `n` cycles from a cleared accumulator the
pre-carry sum is

    pre(n) = rate * n + scale          (rate >= 0)

which at `p = ENTRY` is `(ENTRY - SCALE) * n + SCALE` in unipolar and
`(ENTRY - SCALE) / 2 * n + SCALE` in bipolar. `pre(1) = drift` is the first cycle, as it must be.
Where `rate < 0` and `drift > 0` the accumulator does not grow with `n` at all: it climbs by `drift`
until it crosses `scale`, fires, and hovers below `scale + drift`, so the closed form does not apply
and the peak is that hover bound, never a negative number. Where `drift <= 0` the same recurrence
runs downward and the trough is `max(drift * n, acc_min)`.

The cycle budget that `n` is capped by is `2 ** SEQ_WIDTH`, not `2 ** SEQ_WIDTH - 1`:
`mul_ugemm.forward()` reads `num_seq[seq_idx]` on every timestep and advances the index only on an
enabling one, so a maximal run reads index 0 through `2 ** SEQ_WIDTH - 1` and the increment past the
end is never read back. A run that spends the whole budget has no cycle left, so a block that
charges and then drains is capped at `2 ** SEQ_WIDTH - 1` charging cycles. Measured, from
`mul_ugemm` at `timestep` 256: 256 spiking cycles run clean in both polarities, the 257th raises
`IndexError`, and a silent cycle after 256 spiking ones raises it as well.

Predicted against measured, over 22 parameter points. `form` is the closed form above, `rec` the
scalar recurrence read off `forward()`, and `meas` an instrumented `add_any` driven with the partial
sum held at `p` for `N` cycles from `reset()`; `hi` and `lo` are the extremes of the clamped
pre-carry sum. All 22 agree, including the three where the closed form correctly declines to apply.

```
pol       entry scale   W    N   p  form_hi   rec_hi  meas_hi |  form_lo   rec_lo  meas_lo
unipolar      5     1  12  256   5   1025.0   1025.0   1025.0 |      5.0      5.0      5.0
unipolar      5     1  11  256   5   1023.0   1023.0   1023.0 |      5.0      5.0      5.0
unipolar     13     6  12  180  12   1086.0   1086.0   1086.0 |     12.0     12.0     12.0
unipolar     13     6  11  180  12   1023.0   1023.0   1023.0 |     12.0     12.0     12.0
unipolar     17     5  12  255  16   2047.0   2047.0   2047.0 |     16.0     16.0     16.0
unipolar      8     8  12  100   8      8.0      8.0      8.0 |      8.0      8.0      8.0
unipolar      5     5  12   64   5      5.0      5.0      5.0 |      5.0      5.0      5.0
unipolar      4     9  12   64   4      n/a     12.0     12.0 |      n/a      4.0      4.0
unipolar      3     7   8   64   3      n/a      9.0      9.0 |      n/a      3.0      3.0
unipolar      5     5   4   64   5      5.0      5.0      5.0 |      5.0      5.0      5.0
bipolar       5     1  12  256   5    513.0    513.0    513.0 |      3.0      3.0      3.0
bipolar      13     6  12  180  12    456.0    456.0    456.0 |      8.5      8.5      8.5
bipolar       8     1  10  147   8    511.0    511.0    511.0 |      4.5      4.5      4.5
bipolar       8     8  12  100   8      8.0      8.0      8.0 |      8.0      8.0      8.0
bipolar       4     9  12   64   4      n/a     15.0     15.0 |      n/a      6.5      6.5
bipolar      17     5  12  255  16   1280.0   1280.0   1280.0 |     10.0     10.0     10.0
bipolar       8     1  10  200   0     -3.5     -3.5     -3.5 |   -512.0   -512.0   -512.0
bipolar       7     1  10  200   0     -3.0     -3.0     -3.0 |   -512.0   -512.0   -512.0
bipolar       3     1  10  256   0     -1.0     -1.0     -1.0 |   -256.0   -256.0   -256.0
bipolar       3     1  10  256   3    257.0    257.0    257.0 |      2.0      2.0      2.0
bipolar      13     6  12  255   0     -3.5     -3.5     -3.5 |   -892.5   -892.5   -892.5
unipolar     13     6  12  255   0      0.0      0.0      0.0 |      0.0      0.0      0.0
```

Three consequences the old text got wrong. The clamped quantity is the pre-carry sum, so a
`SCALE == ENTRY` arm, whose *stored* value is confined to `[0, ENTRY - 1]`, still presents
`2 * ENTRY - 1` to the clamp (row `unipolar 5 5 4`: `acc_max` 7 at WIDTH 4, and 9 would be the
unclamped sum). The budget is `2 ** SEQ_WIDTH`, which is what turns `conv_ugemm`'s old 1020 into
1025. And with `SCALE > ENTRY` the expression is not a negative excursion but a hover bound, which
rows `unipolar 4 9` and `bipolar 4 9` show.

### Per-rail coverage

Every accumulator in the two modules, and whether the vector set reaches its rails.

| Module | Accumulator | ACC_HI | ACC_LO |
|--------|-------------|--------|--------|
| conv_ugemm | unipolar geometries a, b, c (SCALE == ENTRY) | not reached: the stored value stays in `[0, ENTRY - 1]` and the clamped sum at `2*ENTRY - 1 = 25`, against 2047 | none: `add_any_unipolar` carries no negative clamp, and the offset is 0 so the state never goes negative |
| conv_ugemm | unipolar geometry d (SCALE 6, ENTRY 13) | reached past the next narrower clamp: pre-carry 1086 over 180 cycles, against 1023 at WIDTH 11 and 2047 at 12. Proven by injection (11 mismatches) | none, same construction; the generator asserts the measured trough is 0 |
| conv_ugemm | bipolar geometries a, b, c (SCALE == ENTRY) | not reached: offset 0, same confinement as unipolar | not reached: same confinement |
| conv_ugemm | bipolar geometry d | not reached: the offset is 3.5 per cycle, so `3.5 * 255 + 6 = 898` is all the budget allows against 1023. Measured stored peak 450, pre-carry 456 | not reached: same 3.5 per cycle downward, measured trough -497 against -1024. A `SCALE` of 1 would put both in reach, but its drain runs at 1 per cycle and no budget shows it |
| mgu | forget-gate `add_any` (ENTRY 8, SCALE 1) | reached exactly: pre-carry 511 in 146 cycles, stored 510. Generator asserts equality with `acc_max - SCALE` | reached exactly: -512 in 147 cycles with the gate biases railed low. Proven by injection (90 mismatches) |
| mgu | new-gate `add_any` (ENTRY 8, SCALE 1) | not reached: driven by `fg_hx` rather than by the input, measured peak 216 against 511 | reached past the next narrower clamp: measured trough -360 against -256 at WIDTH 9. Asserted in the generator |
| mgu | output `add_any` (ENTRY 3, SCALE 1) | not reached: the offset is 1, so no run inside the `2 ** SEQ_WIDTH` budget moves it past 256 against 511. Measured 24 | not reached: same bound, measured -2. Asserted in the generator as staying inside the narrower clamp |

Both modules' `WIDTH` is one parameter shared by every accumulator in the tree, so the ones that do
reach a rail are what make it observable; the ones that cannot are recorded here rather than counted
as covered.

## Dead clamp arms and sweep gates, 2026-08-06

### Six dead clamp arms removed

Following the `add_any_unipolar` precedent: each arm was first probed by fault injection (halve the
clamp, rerun the co-simulation), then removed only after the injection stayed green, then gated on a
byte-identical `vec/<op>.vec` and a `PASS` co-simulation. The method note below applies: halving is
not a reachability test, and an independent review has since re-derived all six of these removals
with a branch oracle and an exhaustive reachable-state BFS, both of which agree. Every removal carries an invariant banner in
the RTL header stating the induction that proves the arm unreachable. Injection rows, including the
live sibling each removal is contrasted against, are in the fault-injection matrix above.

| Operation | File | Arm removed | Invariant that proves it dead | Gate |
|-----------|------|-------------|-------------------------------|------|
| uni2bi | operation/uni2bi/rtl/uni2bi.v | low clamp `ACC_MIN` | acc >= 0: the addend is 1 or 2, `ACC_MAX` is itself >= 0, and a fire needs the clamped sum >= 2 and subtracts exactly 2 | PASS 3329/3329, vec byte-identical |
| sigmoid_hard | operation/sigmoid_hard/rtl/sigmoid_hard.v | low clamp `ACC_MIN` | acc in [0, 3] by the same induction at scale 2, width 3 | PASS 6656/6656, vec byte-identical |
| div_iscb | operation/div_iscb/rtl/div_iscb_uni2bi.v | low clamp `ACC_MIN` | acc in [0, 3]: the step is 1 or 2 and the emit threshold is 2 | PASS 8448/8448, vec byte-identical |
| div_iscb | operation/div_iscb/rtl/div_iscb_bi2uni.v | high clamp `ACC_MAX` | acc <= 2, so acc_sum <= 3: the emit threshold is 1, so a fire leaves at most 2, and a non-fire leaves at most 0 | PASS 8448/8448, vec byte-identical |
| relu_sat | operation/relu_sat/rtl/relu_sat.v | sub-stage high clamp | acc_sub <= 1, so sum_sub <= 2: the only rising step is +1 and it fires and subtracts 2 as soon as acc_sub reaches 1 | PASS 6672/6672, vec byte-identical |
| relu_sat | operation/relu_sat/rtl/relu_sat.v | add-stage high clamp | acc_add <= 2, so sum_add <= 4: `out_sub` requires acc_sub = 1 and leaves acc_sub = 0, so it never fires twice in a row, and acc_add = 2 is only reachable from an addend of 3 | PASS 6672/6672, vec byte-identical |
| relu_sat | operation/relu_sat/rtl/relu_sat.v | add-stage low clamp | acc_add >= 0: the addend `2*out_sub + 1` is always positive and a fire subtracts exactly 2 from a sum of at least 2 | PASS 6672/6672, vec byte-identical |

### Three more dead clamp arms removed, 2026-08-06 (branch oracle)

The same treatment for the high clamps of the same three helpers, this time decided by branch oracle
rather than by halving. Each arm's condition was left exactly as written and its taken-arm result
poisoned; every one stayed green, so the branch is never entered. That agrees with an independent
exhaustive reachable-state BFS, which finds each of them dead at every legal `WIDTH` rather than only
at the elaborated one. The oracle rows are in the fault-injection matrix above.

| Operation | File | Arm removed | Oracle verdict | BFS verdict | Invariant that proves it dead | Gate |
|-----------|------|-------------|----------------|-------------|-------------------------------|------|
| uni2bi | operation/uni2bi/rtl/uni2bi.v | high clamp `ACC_MAX` | never entered | dead at every legal WIDTH | acc is 0 or 1: from acc in [0,1] the addend is 1 or 2, so sum is in [1,3]; a fire needs sum >= 2 and subtracts exactly 2, a non-fire leaves sum = 1. The model rejects a width whose `acc_max` is below the threshold 2, so every legal WIDTH has `acc_max` >= 3 >= sum | PASS 3329/3329, vec byte-identical (`d6cf1010f8f88f4fef354fe66b53b9a5`) |
| sigmoid_hard | operation/sigmoid_hard/rtl/sigmoid_hard.v | high clamp `ACC_MAX` | never entered | dead | the same induction at the fixed scale 2, width 3: acc is 0 or 1 and `acc_sum` is in [1,3], inside the model's [-4, 3] | PASS 6656/6656, vec byte-identical (`bd6dcd4c3c4808149ba867ce7359b842`) |
| div_iscb | operation/div_iscb/rtl/div_iscb_uni2bi.v | high clamp `ACC_MAX` | never entered | dead | the same induction: step 1 or 2, threshold 2, so acc is 0 or 1 and `acc_sum` is in [1,3] | PASS 8448/8448, vec byte-identical (`24ab967d0f14dab390fceee52b53de93`) |

Each md5 is the vector file before and after the removal, taken from the run recorded above; the
co-simulation was rerun after each removal and passed on the same vectors.

### `div_iscb_bi2uni`'s low clamp: reclassified LIVE BUT UNEXERCISED, 2026-08-06

The matrix had recorded this arm as live because halving it to -2 fired. That is not what a halved
constant shows. A branch oracle against the same 8448-vector set stays green, so the clamp at -4 was
never entered: what the halved run proved is that -2 is reachable, not that -4 is.

The arm is nonetheless live, and the gap was in the stimulus. `div_iscb_bi2uni` is its own mapping
entry against `sim/operation/bi2uni.py`, so its contract is the standalone helper, but the only
stimulus it had came through `div_iscb_bipolar`, where it cannot reach the clamp: an exhaustive BFS
over the composed (signabs, bi2uni) state space reaches 20 states with a minimum accumulator of -3,
because `signabs` never holds its magnitude output low for more than three cycles in a row. Driven
directly, five consecutive silent cycles walk the accumulator from 0 to -4 and present -5 to the
clamp on the fifth.

`gen_div_iscb.py` now drives a standalone `bi2uni(width=3)` on its own stimulus column, on the
dividend stream plus a deterministic tail of six silent cycles and two spiking ones, and asserts that
the low clamp was taken at least once. The testbench instantiates a third DUT for it. Red then green
against the new 8456-vector set: the branch oracle fails at `FAIL[b2u] t=4 in=0` with 1057
mismatches, and the restored RTL passes 8456/8456. The arm was kept, not removed.

`relu_sat`'s fourth arm, the sub-stage low clamp at -8, is LIVE and was kept: a silent input
decrements `acc_sub` without limit and the vector set reaches the clamp. The injection asymmetry is
the evidence, all four arms halved one at a time in the same run against the same 6672-vector set:
the three above stayed green, the sub-stage low clamp failed at cycle 801 with 15 mismatches. All
five touched operations recompile with `iverilog -g2001 -Wall` at 0 warnings.

`add_any_unipolar`'s header claimed a clamp to `[0, 2^WIDTH-2]` and then explained that the low end
of that range is not a clamp at all. The claim now reads "clamps to at most 2^WIDTH-2", which is what
the removed-arm paragraph below it goes on to prove.

### The sweep could pass vacuously; a mapping floor now stops it

`make sweep` discovers targets by globbing `operation/*/rtl` and `module/*/rtl`, so a deleted or
renamed unit folder left the glob and the sweep reported every remaining target green. `make floor`
(`src/napl/imp/sweep_floor.py`) derives the expected unit set from `mapping.yaml`, the authoritative
registry, and fails on a registered unit the glob missed, a discovered folder with no mapping entry,
or an empty target list. It runs first in the sweep and joins the `*** SWEEP FAILED:` list.

Proof, red then green. With `operation/uni2bi/rtl` renamed to `rtl_moved` (the same on-disk condition
a `git rm` of the folder creates), the sweep goes red naming the unit instead of reporting 42
operations passed:

```
=== floor
*** SWEEP FLOOR FAILED
  - operation unit 'uni2bi' is registered in mapping.yaml but the sweep discovered no operation/uni2bi/rtl directory
make[1]: *** [floor] Error 1
=== guards
=== translate
*** SWEEP FAILED: floor
```

Restored, the sweep is green again and the floor states its own count:

```
sweep floor: 43 operations, 9 modules match mapping.yaml
SWEEP PASSED: 43 operations, 9 modules, floor, guards, translate
```

### The sweep now covers mapping.yaml and translate.py

`mapping.yaml` and `syn/translate.py` carry no RTL, so no unit target gated them and a mapping edit
could pass the documented commit gate untested. `make translate` runs
`tests/syn/test_translate.py` inside the sweep (`11 passed in 2.30s`), so a mapping edit that breaks
translation turns the sweep red.

### The commit gate is now mechanical

The gate was a sentence in RULE_IMP.md, and an operation target sat red for an unknown period because
nothing enforced it. `.githooks/pre-commit` is versioned in the repo and installed with one command,
`conda run -n napl make install-hooks` from `src/napl/imp/`, which points `core.hooksPath` at
`.githooks/`. The hook exits 0 immediately unless the staged file list touches `src/napl/imp/`; when
it does, it runs `make sweep` and refuses the commit on a nonzero exit. `git commit --no-verify`
bypasses it deliberately; nothing bypasses it by accident.

Proof, red then green, both against the same staged file (`src/napl/imp/Makefile`). With
`operation/uni2bi/rtl` renamed away so the sweep is red:

```
*** SWEEP FAILED: floor
make: *** [sweep] Error 1

*** COMMIT REFUSED: make sweep is not green.
*** Fix the red targets above, or commit with --no-verify to bypass deliberately.
```

`HEAD` was `012145b` before that attempt and `012145b` after it. With the folder restored, the same
commit succeeds (`SWEEP PASSED: 43 operations, 9 modules, floor, guards, translate`, then
`[porting-unarysim 72097c5] demo: clean commit`). That demonstration commit was undone with
`git reset --soft 012145b` followed by `git reset`, so `HEAD` is `012145b` again, nothing is staged,
and the working tree is unchanged. `git log --oneline -3` reads `012145b`, `e884138`, `d26271b`,
the same three commits as before the demonstration.

### Makefile and generator minors

1. `vvp | tee` reported `tee`'s exit status, so a nonzero `vvp` could read as success. The Makefile
   now runs recipes under `bash` with `.SHELLFLAGS := -o pipefail -c`, so the pipeline carries
   `vvp`'s status.
2. The sweep sent the guards output to `/dev/null`, so a guards failure was silent about why. The
   floor, guards, and translate steps now log to `src/napl/imp/build/<step>.log` and the sweep prints
   the log when that step fails.
3. `gen_add_any.py`'s `rail_uni[0] == 0` assertion re-checks the invariant that the unipolar
   accumulator never goes negative, so it cannot catch a dead stimulus: the unipolar rail segment has
   no low rail to reach. It is now annotated as an invariant check, the runnable form of the
   induction in the `add_any_unipolar` header, and the three assertions above it are named as the
   stimulus checks.

## Record corrections and non-blockers, 2026-08-06 (batch 11, part 1)

Each number below was re-measured here rather than transcribed.

1. **`conv_ugemm` bipolar peak is 450, not 456.** An instrumented run of geometry `d`'s accumulator
   reports `peak [1080.0, 450.0] bottom [0, -497.0]`. 456 is the pre-carry sum `2.5 * 180 + 6`; the
   stored value is that less one carry of `SCALE` 6, which is what the trough -497 alongside it was
   already reported at. `gen_conv_ugemm.py` now carries `SAT_PEAK_UNI`, `SAT_PEAK_BI` and
   `SAT_TROUGH_BI` and asserts the measured triple equals them exactly. The inequalities that were
   there bound the reaches and would not have caught a wrong figure; this assert would have.
2. **`relu_sat`'s live-arm injection is 14 mismatches, not 15.** Re-run: halving the sub-stage low
   clamp gives `FAIL relu_sat: 14 mismatch(es) over 6672 vectors`. A branch oracle on the same arm
   (condition kept, result poisoned) gives 96 mismatches, so the two methods agree that this arm is
   entered.
3. **RULE_IMP's `p = ENTRY` sentence** claimed every railed operand gives `p = ENTRY`, while its own
   worked example rails the bias low and uses `p = K = ENTRY - 1`. The sentence now names both cases.
4. **RULE_IMP's `mul_ugemm` charge cap** said a charge-and-drain block holds at most
   `2 ** SEQ_WIDTH - 1` "of each". Only the charging half is capped that way; the phrase is dropped.
5. **The DEPTH observability claim for `linear_gaines` was wrong.** It read as if both rails being
   charged is what exposes `DEPTH`. Measured: the opening charge block produces no `DEPTH` mismatch,
   because a counter of any `DEPTH` sits on its own high clamp there. The drain block is what parts
   the column and is sufficient on its own -- with the third block dropped, `GEN_DEPTH` 8 -> 7 still
   fails at n=807 with 4 mismatches over 836 vectors. The third block is not dead: it is the only
   stimulus that leaves the low rail, and deleting the low clamp fails at n=844 with it present and
   PASSes 836/836 with it dropped. It is kept on that justification.
6. **The `lfsr` requires clause is dropped from the two `linear_gaines1` mapping entries.**
   `linear_gaines1.__init__` rejects any generator outside the sobol family, so the clause could
   never fire on those entries. Its negative test drives a `linear_gaines2` node and both classes
   share one RTL module per polarity, so `requires_clauses()` still contains the
   `(linear_gaines_<polarity>, lfsr clause)` pairs through the `linear_gaines2` entries and
   `covered == requires_clauses()` stays closed with no test change. Fixing the class itself is a
   behavior change and is parked.
7. **`RAIL_STEPS` is derived, not hardcoded.** It was 40. It is now
   `2 * ceil(CNT_MAX / (ENTRY - 2))` = 34: the counter's whole span at the slower of its two rates,
   doubled so it holds on the clamp while the next block's departure is read against it. The vector
   set moves from 888 to 870 rows, so every `linear_gaines` injection above was re-measured against
   the new set.
8. **The `ENTRY - 2` fall rate is the railed-high bias.** With no bias addend, or a bias that is not
   railed, the counter falls by `ENTRY`. Stated in RULE_IMP and in the generator.
9. **The `SCALE_WIDTH` mapping assertion was vacuous** -- it compared the mapping expression against
   a second copy of the same rounding. It now compares against the model's own `scale_width`, the
   width the emitted threshold ROM is written at.
10. **`SUM_W`'s 32-bit ceiling is documented.** `CNT_MAX` and `CNT_HALF` are 32-bit signed
    elaboration constants sliced into the `SUM_W` datapath, so `DEPTH` is capped at 30; at 31 the
    `2 ** DEPTH - 1` constant is negative and the high clamp inverts. Recorded in
    `linear_gaines_bipolar.v`'s banner.
11. **`avgpool2d` no longer has a `DIVISOR`.** The config trim reduced the class to
    `(kernel_size, stride)`, so `DIVISOR` and `DIVISOR_D8`, the `DIVISOR >= KERNEL_AREA` elaboration
    guard, and the `out_d8` vector column are gone; `add_any_unipolar` now takes
    `SCALE = ENTRY = KERNEL_AREA`. This supersedes the `DIVISOR` parts of the `avgpool2d` ledger row
    and of its observability row above. Re-verified: `make test MODULE=avgpool2d` PASSes 768/768
    vectors (192 lanes, window area 4).

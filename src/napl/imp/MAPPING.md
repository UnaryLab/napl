# NAPL simulation to RTL mapping

Each row maps one Verilog module under `src/napl/imp/<op>/rtl/` to its Python simulation operation. Clock and reset ports are omitted. A forward argument with no data port is listed as `no RTL port`. `none` means the Verilog module header declares no parameters.

| Layer | Sim module | RTL module | Forward input(s) to RTL input port(s) | Forward output(s) to RTL output port(s) | Parameters and source |
| --- | --- | --- | --- | --- | --- |
| operation | `sim/operation/add_any.py` | `add_any_bipolar` | `input` → `i_input`; `entry`, `dim` → no RTL port | `output` → `o_out` | `SCALE` ← `config['scale']`; `WIDTH` ← `config['width']`; `ENTRY` ← forward reduction size |
| operation | `sim/operation/add_any.py` | `add_any_unipolar` | `input` → `i_input`; `entry`, `dim` → no RTL port | `output` → `o_out` | `SCALE` ← `config['scale']`; `WIDTH` ← `config['width']`; `ENTRY` ← forward reduction size |
| operation | `sim/operation/add_gaines.py` | `add_gaines` | `input` → `i_input`; `dim` → no RTL port | `output` → `o_out` | `SCALED` ← `config['scaled']`; `ENTRY` ← `config['entry']`; `SELECT_WIDTH` ← `log2(config['entry'])` |
| operation | `sim/operation/add_ugemm.py` | `add_ugemm_bipolar` | `input` → `i_input`; `dim` → no RTL port | `output` → `o_out` | `SCALED` ← `config['scaled']`; `ENTRY` ← `input.size(dim)`; `COUNT_WIDTH` ← `ENTRY`; `ACC_WIDTH` ← no config field |
| operation | `sim/operation/add_ugemm.py` | `add_ugemm_unipolar` | `input` → `i_input`; `dim` → no RTL port | `output` → `o_out` | `SCALED` ← `config['scaled']`; `ENTRY` ← `input.size(dim)`; `COUNT_WIDTH` ← `ENTRY`; `ACC_WIDTH` ← no config field |
| operation | `sim/operation/bi2uni.py` | `bi2uni` | `input` → `i_input` | `output` → `o_out` | `WIDTH` ← `config['width']` |
| operation | `sim/operation/dff.py` | `dff` | `input` → `i_input` | `output` → `o_out` | `DEPTH` ← `config['depth']` |
| operation | `sim/operation/div_cordiv.py` | `div_cordiv` | `dividend` → `i_dividend`; `divisor` → `i_divisor` | `quotient` → `o_quotient` | `DEPTH` ← `config['depth']`; `WIDTH` ← `log2(config['depth'])` |
| operation | `sim/operation/div_gaines.py` | `div_gaines_bipolar` | `dividend` → `i_dividend`; `divisor` → `i_divisor` | `output` → `o_out` | `DEPTH` ← `config['depth']` |
| operation | `sim/operation/div_gaines.py` | `div_gaines_unipolar` | `dividend` → `i_dividend`; `divisor` → `i_divisor` | `output` → `o_out` | `DEPTH` ← `config['depth']` |
| operation | `sim/operation/bi2uni.py` | `div_iscb_bi2uni` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/div_iscb.py` | `div_iscb_bipolar` | `dividend` → `i_dividend`; `divisor` → `i_divisor` | `output` → `o_quotient` | none |
| operation | `sim/operation/signabs.py` | `div_iscb_signabs` | `input` → `i_input` | `sign` → `o_sign`; `abs` → `o_abs` | none |
| operation | `sim/operation/uni2bi.py` | `div_iscb_uni2bi` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/div_iscb.py` | `div_iscb_unipolar` | `dividend` → `i_dividend`; `divisor` → `i_divisor` | `output` → `o_quotient` | none |
| operation | `sim/operation/exp_n1.py` | `exp_n1` | `input` → `i_input` | `output` → `o_out` | `WIDTH` ← `ceil(log2(config['timestep']))` |
| operation | `sim/operation/exp_ng.py` | `exp_ng` | `input` → `i_input` | `output` → `o_out` | `DEPTH` ← `config['depth']`; `GAIN` ← `config['gain']` |
| operation | `sim/operation/gt_rc.py` | `gt_rc` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | `output` → `o_out` | none |
| operation | `sim/operation/jkff.py` | `jkff` | `input_j` → `i_input_j`; `input_k` → `i_input_k` | `self.q` → `o_q` | none |
| operation | `sim/operation/lt_rc.py` | `lt_rc` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | `output` → `o_out` | none |
| operation | `sim/operation/max_rc.py` | `max_rc` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | `output` → `o_max`; `self.dff` → `o_arg` | none |
| operation | `sim/operation/max_tc.py` | `max_tc` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | `output` → `o_out` | none |
| operation | `sim/operation/min_rc.py` | `min_rc` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | `output` → `o_min`; `1 - self.dff` → `o_argmin` | none |
| operation | `sim/operation/min_tc.py` | `min_tc` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | `output` → `o_out` | none |
| operation | `sim/operation/mul_and.py` | `mul_and_bipolar` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | return value → `o_out` | none |
| operation | `sim/operation/mul_and.py` | `mul_and_unipolar` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | return value → `o_out` | none |
| operation | `sim/operation/mul_csg.py` | `mul_csg_bipolar` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | `path \| path_inv` → `o_out` | `WIDTH` ← `ceil(log2(config['timestep']))` |
| operation | `sim/operation/mul_csg.py` | `mul_csg_unipolar` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | `path` → `o_out` | `WIDTH` ← `ceil(log2(config['timestep']))` |
| operation | `sim/operation/mul_gaines.py` | `mul_gaines_bipolar` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | return value → `o_out` | none |
| operation | `sim/operation/mul_gaines.py` | `mul_gaines_unipolar` | `input_0` → `i_input_0`; `input_1` → `i_input_1` | return value → `o_out` | none |
| operation | `sim/operation/relu_cnt.py` | `relu_cnt` | `input` → `i_input` | `output` → `o_out` | `WIDTH` ← `config['width']` |
| operation | `sim/operation/relu_sat.py` | `relu_sat` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/shiftreg.py` | `shiftreg` | `input` → `i_input` | `output` → `o_out` | `DEPTH` ← `config['depth']` |
| operation | `sim/operation/sigmoid_hard.py` | `sigmoid_hard` | `input` → `i_input` | return value → `o_out` | none |
| operation | `sim/operation/signabs.py` | `signabs` | `input` → `i_input` | `sign` → `o_sign`; `abs` → `o_abs` | `WIDTH` ← `config['width']` |
| operation | `sim/operation/sqrt_emit.py` | `sqrt_emit_bipolar` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/sqrt_emit.py` | `sqrt_emit_unipolar` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/sqrt_gaines.py` | `sqrt_gaines_bipolar` | `input` → `i_input` | `output` → `o_out` | `WIDTH` ← `config['width']` |
| operation | `sim/operation/sqrt_gaines.py` | `sqrt_gaines_unipolar` | `input` → `i_input` | `output` → `o_out` | `WIDTH` ← `config['width']` |
| operation | `sim/operation/sqrt_traceiscb.py` | `sqrt_traceiscb_bipolar` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/sqrt_traceiscb.py` | `sqrt_traceiscb_unipolar` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/sqrt_tracejkff.py` | `sqrt_tracejkff_bipolar` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/sqrt_tracejkff.py` | `sqrt_tracejkff_unipolar` | `input` → `i_input` | `output` → `o_out` | none |
| operation | `sim/operation/square_dff.py` | `square_dff_bipolar` | `input` → `i_input` | `out` → `o_out` | `DEPTH` ← `config['depth']` |
| operation | `sim/operation/square_dff.py` | `square_dff_unipolar` | `input` → `i_input` | `out` → `o_out` | `DEPTH` ← `config['depth']` |
| operation | `sim/operation/sync_skewed.py` | `sync_skewed` | `input_1` → `i_input_1`; `input_2` → `i_input_2` | `output_1` → `o_out_1`; `input_2` → `o_out_2` | `WIDTH` ← `config['width']` |
| operation | `sim/operation/sync_skewed_int.py` | `sync_skewed_int` | `input_1` → `i_input_1`; `input_2` → `i_input_2` | `output_1` → `o_out_1`; `input_2` → `o_out_2` | `WIDTH` ← `config['width']` |
| operation | `sim/operation/tanh_hard.py` | `tanh_hard` | `input` → `i_input` | `input` → `o_out` | none |
| operation | `sim/operation/tanh_p1.py` | `tanh_p1` | `input` → `i_input` | `out` → `o_out` | `WIDTH` ← `ceil(log2(config['timestep']))` |
| operation | `sim/operation/tanh_pn.py` | `tanh_pn` | `input` → `i_input` | `output` → `o_out` | `DEPTH` ← `config['depth']` |
| operation | `sim/operation/uni2bi.py` | `uni2bi` | `input` → `i_input` | `output` → `o_out` | `WIDTH` ← `config['width']` |

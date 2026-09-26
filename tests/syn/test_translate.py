import re
from pathlib import Path

import pytest
import torch
import yaml

from napl.syn.translate import (_MAPPING_PATH, _check_requires, _resolve_parameters,
                                _RestrictedEvaluator)
from napl.syn import (
    TranslationError,
    load_mapping,
    translate_graph,
    translate_node,
)


def test_polarity_variant_and_derived_parameter():
    """Resolve a polarity-specific RTL module and derived sizing parameter."""
    add = translate_node({
        "name": "adder",
        "class": "add_scale",
        "config": {"polarity": "bipolar", "scale": 2, "intwidth": 8, "fracwidth": 0},
        "inputs": {"input": {"shape": (4, 16)}},
    })
    assert add.rtl_module == "add_scale_bipolar"
    assert add.parameters == {"SCALE": 2, "WIDTH": 8, "ENTRY": 16}
    assert add.port_map["inputs"]["input"] == "i_input"
    assert add.port_map["output"] == "o_output"
    assert Path(add.file_path).name == "add_scale_bipolar.v"

    divider = translate_node({
        "class": "div_cordiv",
        "config": {"depth": 8, "generator": "sobol"},
        "inputs": {"dividend": {"shape": (8,)}, "divisor": {"shape": (8,)}},
    })
    assert divider.parameters == {"DEPTH": 8, "WIDTH": 3}


def test_graph_translation_preserves_order():
    """Translate graph nodes in their supplied order."""
    bindings = translate_graph([
        {"name": "a", "class": "mul_gaines", "config": {"polarity": "unipolar"}},
        {"name": "b", "class": "relu_sat", "config": {}},
    ])
    assert [binding.name for binding in bindings] == ["a", "b"]
    assert [binding.rtl_module for binding in bindings] == ["mul_gaines_unipolar", "relu_sat"]


def test_missing_rtl_mapping_names_class():
    """Reject a simulation class that has no RTL mapping entry."""
    # A synthetic name keeps the check about the error message rather than about
    # which real classes happen to lack an RTL module today.
    absent = "operation_with_no_rtl_module"
    for entry in load_mapping():
        assert entry.get("rtl_module") != absent, f"{absent} must stay out of the RTL mapping"
        assert absent not in str(entry.get("sim_module", "")), \
            f"{absent} must stay out of the RTL mapping"
    with pytest.raises(TranslationError, match=absent):
        translate_node({"class": absent, "config": {}})


LAYER_CONFIG = {'polarity': 'unipolar', 'timestep': 256, 'generator': 'sobol',
                'dim': 1, 'scale': None, 'width': 12}
# The Gaines layers take no accumulator scale or width.
GAINES_CONFIG = {'polarity': 'unipolar', 'timestep': 256, 'generator': 'sobol', 'dim': 1}


def module_nodes():
    """Nodes for both module entries, built as RULE_IMP documents them.

    A module node carries the class's ``__init__`` arguments under their own
    names, so a `config=` argument nests, plus the caller-supplied `lanes`.
    """
    weight = torch.zeros(8, 16)
    layer_config = LAYER_CONFIG
    return [
        {"class": "avgpool2d_ugemm", "config": {"kernel_size": 2, "lanes": 48}},
        {"class": "avgpool2d_ugemm", "config": {"kernel_size": (2, 3), "lanes": 48}},
        {"class": "linear_ugemm",
         "config": {"weight": weight, "bias": torch.zeros(8),
                    "config": layer_config, "lanes": 8}},
        {"class": "linear_ugemm",
         "config": {"weight": weight, "bias": None,
                    "config": dict(layer_config, polarity='bipolar'), "lanes": 8}},
        {"class": "linear_mix",
         "config": {"weight": weight, "bias": torch.zeros(8),
                    "config": layer_config, "lanes": 8}},
        {"class": "linear_mix",
         "config": {"weight": weight, "bias": None,
                    "config": dict(layer_config, polarity='bipolar'), "lanes": 8}},
        # conv sizes itself from the weight and the input shape, so its node
        # carries the NCHW input the geometry is derived from.
        {"class": "conv_mix",
         "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": torch.zeros(3),
                    "stride": 1, "padding": 1, "dilation": 1,
                    "config": layer_config, "lanes": 108},
         "inputs": {"input": {"shape": (1, 2, 6, 6)}}},
        {"class": "conv_mix",
         "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": None,
                    "stride": 2, "padding": 2, "dilation": 2,
                    "config": dict(layer_config, polarity='bipolar'), "lanes": 27},
         "inputs": {"input": {"shape": (1, 2, 6, 6)}}},
    ]


def test_module_layer_parameters():
    """Resolve every module entry from constructor-name node configs."""
    (pool, pool_rect, ugemm, ugemm_nb, linear, linear_nb,
     convolution, convolution_nb) = [translate_node(node) for node in module_nodes()]

    assert pool.rtl_module == "avgpool2d_ugemm"
    assert Path(pool.file_path).parts[-3:] == ("avgpool2d_ugemm", "rtl", "avgpool2d_ugemm.v")
    assert pool.parameters == {"KERNEL_AREA": 4, "LANES": 48}
    # A tuple kernel resolves to the window area.
    assert pool_rect.parameters == {"KERNEL_AREA": 6, "LANES": 48}

    assert ugemm.rtl_module == "linear_ugemm_unipolar"
    assert ugemm.parameters == {"IN_FEATURES": 16, "LANES": 8, "SEQ_WIDTH": 8,
                                "WIDTH": 12, "HAS_BIAS": 1, "SCALE": 17}
    assert ugemm.port_map["inputs"]["weight"] == "i_weight"
    assert ugemm.port_map["output"] == "o_output"
    # No bias drops an addend, so the default scale shrinks with it.
    assert ugemm_nb.rtl_module == "linear_ugemm_bipolar"
    assert ugemm_nb.parameters["HAS_BIAS"] == 0
    assert ugemm_nb.parameters["SCALE"] == 16

    assert linear.rtl_module == "linear_mix_unipolar"
    assert linear.parameters == {"IN_FEATURES": 16, "LANES": 8, "SEQ_WIDTH": 8,
                                 "WIDTH": 12, "HAS_BIAS": 1, "SCALE": 17}
    assert linear.port_map["inputs"]["input"] == "i_input"
    assert linear.port_map["output"] == "o_output"
    assert linear_nb.rtl_module == "linear_mix_bipolar"
    assert linear_nb.parameters["HAS_BIAS"] == 0
    assert linear_nb.parameters["SCALE"] == 16

    # conv reads its lane geometry from the weight and the NCHW input shape.
    assert convolution.rtl_module == "conv_mix_unipolar"
    assert convolution.parameters == {"BATCH": 1, "IN_CHANNELS": 2, "IN_H": 6, "IN_W": 6,
                                      "OUT_CHANNELS": 3, "KERNEL_H": 3, "KERNEL_W": 3,
                                      "STRIDE": 1, "PADDING": 1, "DILATION": 1,
                                      "SEQ_WIDTH": 8, "WIDTH": 12, "HAS_BIAS": 1,
                                      "SCALE": 19, "LANES": 108}
    # Both variants generate the pad stream inside the module, so pad_bits maps
    # to no port for either polarity.
    assert convolution.port_map["inputs"]["pad_bits"] is None
    assert convolution.port_map["output"] == "o_output"
    # Stride and dilation shrink the output positions, and no bias shrinks the scale.
    assert convolution_nb.rtl_module == "conv_mix_bipolar"
    assert convolution_nb.port_map["inputs"]["pad_bits"] is None
    assert convolution_nb.parameters["HAS_BIAS"] == 0
    assert convolution_nb.parameters["SCALE"] == 18
    assert convolution_nb.parameters["LANES"] == 27


def test_linear_gaines_maps_per_polarity():
    """Resolve linear_gaines to the same RTL module base per polarity."""
    for polarity in ("unipolar", "bipolar"):
        expected = {"IN_FEATURES": 15, "LANES": 8, "SEQ_WIDTH": 8, "HAS_BIAS": 1,
                    "SCALE_WIDTH": 4, "SCALED": 1}
        if polarity == "bipolar":
            expected.pop("SCALED")
        binding = translate_node(gaines_node("linear_gaines", polarity))
        assert binding.rtl_module == f"linear_gaines_{polarity}"
        assert binding.parameters == expected, binding.parameters
    # No bias drops an addend, so the threshold width follows the smaller entry.
    no_bias = translate_node(gaines_node(
        "linear_gaines", "unipolar", has_bias=False, scaled=False
    ))
    assert no_bias.parameters["HAS_BIAS"] == 0
    assert no_bias.parameters["SCALE_WIDTH"] == 4


def test_module_rejects_unsupported_configuration():
    """Reject pooling geometry the RTL does not implement."""
    with pytest.raises(TranslationError, match="stride"):
        translate_node({"class": "avgpool2d_ugemm",
                        "config": {"kernel_size": 2, "stride": 3, "lanes": 48}})
    # A lane count the convolution geometry cannot fill is rejected before elaboration.
    conv_node = dict(module_nodes()[-1])
    conv_node["config"] = dict(conv_node["config"], lanes=28)
    with pytest.raises(TranslationError, match="LANES =="):
        translate_node(conv_node)


def test_layer_key_default_and_validation():
    """Default a missing `layer` to the operation tree and reject a bad one."""
    entry = next(item for item in load_mapping()
                 if item.get("rtl_module") == "mul_gaines_unipolar")
    node = {"class": "mul_gaines", "config": {"polarity": "unipolar"}}
    # The scratch mapping sits beside the real one: RTL paths resolve relative to it.
    scratch = _MAPPING_PATH.parent / "_layer_default_mapping.yaml"
    try:
        without_layer = {key: value for key, value in entry.items() if key != "layer"}
        scratch.write_text(yaml.safe_dump([without_layer]), encoding="utf-8")
        assert "layer" not in without_layer
        binding = translate_node(node, mapping_path=scratch)
        assert binding.file_path == translate_node(node).file_path

        scratch.write_text(yaml.safe_dump([dict(entry, layer="metric")]), encoding="utf-8")
        with pytest.raises(TranslationError, match="layer"):
            translate_node(node, mapping_path=scratch)
    finally:
        scratch.unlink(missing_ok=True)



def requires_clauses():
    """Every (rtl_module, condition) pair the mapping declares under `requires`."""
    return {(entry["rtl_module"], condition)
            for entry in load_mapping()
            for condition in (entry.get("requires") or [])}


def add_scale_node(polarity, intwidth, entry, fracwidth=0, scale=2):
    """One add_scale node with the given accumulator format and reduction size."""
    return {"class": "add_scale",
            "config": {"polarity": polarity, "scale": scale, "intwidth": intwidth,
                       "fracwidth": fracwidth},
            "inputs": {"input": {"shape": (4, entry)}}}


def add_scale_dyn_node(polarity, intwidth, entry, fracwidth=0, scale_max=2):
    """One add_scale_dyn node with the given accumulator format and reduction size."""
    return {"class": "add_scale_dyn",
            "config": {"polarity": polarity, "scale_max": scale_max,
                       "intwidth": intwidth, "fracwidth": fracwidth},
            "inputs": {"input": {"shape": (4, entry)}}}


def add_ugemm_node(polarity, scaled, width, entry, segment):
    """One add_ugemm node with the given accumulator width, reduction size, and run length."""
    return {"class": "add_ugemm",
            "config": {"polarity": polarity, "scaled": scaled, "width": width},
            "inputs": {"input": {"shape": (4, entry)}},
            "segment": segment}


def div_scale_node(polarity, intwidth=8, fracwidth=0, scale=2):
    """One div_scale node with the given accumulator format and divisor."""
    return {"class": "div_scale",
            "config": {"polarity": polarity, "scale": scale,
                       "intwidth": intwidth, "fracwidth": fracwidth}}


def div_scale_dyn_node(polarity, intwidth=8, fracwidth=0, scale_max=2):
    """One div_scale_dyn node with the given accumulator format and divisor bound."""
    return {"class": "div_scale_dyn",
            "config": {"polarity": polarity, "scale_max": scale_max,
                       "intwidth": intwidth, "fracwidth": fracwidth}}


def conv_gaines_node(polarity, kernel_h=3, has_bias=True, lanes=144, in_channels=1,
                     input_channels=None, scaled=True):
    """One conv_gaines node over a 3 x in_channels x kernel_h x 1 weight.

    The input is 1 x `input_channels` x 6 x 6 with unit stride and dilation and a
    symmetric pad of one, so the default geometry is a fan-in of four over 144
    output positions, the power-of-two fan-in the Gaines adder needs.
    """
    return {"class": "conv_gaines",
            "config": {"weight": torch.zeros(3, in_channels, kernel_h, 1),
                       "bias": torch.zeros(3) if has_bias else None,
                       "stride": 1, "padding": 1, "dilation": 1,
                       "config": dict(GAINES_CONFIG, polarity=polarity, scaled=scaled),
                       "lanes": lanes},
            "inputs": {"input": {"shape": (1, input_channels or in_channels, 6, 6)}}}


def linear_node(class_name, polarity, lanes=8, width=12, scale=None):
    """One linear_mix or linear_ugemm node over an 8x16 weight."""
    return {"class": class_name,
            "config": {"weight": torch.zeros(8, 16), "bias": torch.zeros(8),
                       "config": dict(LAYER_CONFIG, polarity=polarity, width=width,
                                      scale=scale),
                       "lanes": lanes}}


def gaines_node(class_name, polarity, in_features=15, has_bias=True, lanes=8,
                generator='sobol', scaled=True):
    """One linear_gaines node over a `lanes` x in_features weight."""
    config = dict(GAINES_CONFIG, polarity=polarity, generator=generator, scaled=scaled)
    return {"class": class_name,
            "config": {"weight": torch.zeros(8, in_features),
                       "bias": torch.zeros(8) if has_bias else None,
                       "config": config, "lanes": lanes}}


def conv_node(polarity, in_channels=2, lanes=108, width=12, class_name="conv_mix",
              scale=None):
    """One conv node over a 3x2x3x3 weight and a 1x2x6x6 input."""
    config = dict(LAYER_CONFIG, polarity=polarity, width=width, scale=scale)
    if class_name == "conv_ugemm":
        # The conv_ugemm constructor takes no `dim`.
        del config["dim"]
    return {"class": class_name,
            "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": torch.zeros(3),
                       "stride": 1, "padding": 1, "dilation": 1,
                       "config": config, "lanes": lanes},
            "inputs": {"input": {"shape": (1, in_channels, 6, 6)}}}


def mgu_node(lanes=3, hidden=3, in_size=4, width=10, depth_ismul=6, n_hidden=None,
             n_features=None, polarity="bipolar"):
    """One mgu_hard_mix node over gate weights shaped (hidden, hidden + in_size).

    `n_hidden` and `n_features` override the new-gate weight shape on its own, so
    a gate pair that disagrees can be built without touching the forget gate.
    """
    features = hidden + in_size
    config = {'polarity': polarity, 'timestep': 256, 'generator': 'sobol',
              'width': width, 'depth_ismul': depth_ismul}
    return {"class": "mgu_hard_mix",
            "config": {"weight_f": torch.zeros(hidden, features),
                       "bias_f": torch.zeros(hidden),
                       "weight_n": torch.zeros(n_hidden or hidden,
                                               n_features or features),
                       "bias_n": torch.zeros(hidden),
                       "hx_value": torch.zeros(1, hidden),
                       "config": config, "lanes": lanes}}


def clamp_sat_node(polarity, lo, hi, fracwidth=8, dim=2):
    """One clamp_sat node with the given saturation bounds, fractional grid, and Sobol dimension."""
    return {"class": "clamp_sat",
            "config": {"polarity": polarity, "lo": lo, "hi": hi,
                       "fracwidth": fracwidth, "dim": dim}}


# Bands on the 2**-9 bound-threshold grid clamp_comp encodes against, one per
# polarity, and half a threshold step, which moves a bound off that grid.
CLAMP_COMP_BOUNDS = {"bipolar": (-0.5, 0.5), "unipolar": (0.125, 0.6875)}
CLAMP_COMP_OFF_GRID = 1.0 / 1024.0


def clamp_comp_node(polarity, lo_offset=0.0, hi_offset=0.0, dim=2):
    """One clamp_comp node with its band nudged off the grid by the given offsets."""
    lo, hi = CLAMP_COMP_BOUNDS[polarity]
    return {"class": "clamp_comp",
            "config": {"polarity": polarity, "lo": lo + lo_offset,
                       "hi": hi + hi_offset, "dim": dim}}


def encode_node(timestep=256, dim=1, polarity="unipolar", generator="sobol", seed=None,
                taps=None):
    """One encode node at the given sequence length, Sobol dimension, and generator."""
    config = {"polarity": polarity, "timestep": timestep,
              "generator": generator, "dim": dim}
    if seed is not None:
        config["seed"] = seed
    if taps is not None:
        config["taps"] = taps
    return {"class": "encode", "config": config}


def mul_scale_node(polarity, intwidth, fracwidth, scale):
    """One mul_scale node with the given accumulator format and scale."""
    return {"class": "mul_scale",
            "config": {"polarity": polarity, "scale": scale,
                       "intwidth": intwidth, "fracwidth": fracwidth}}


def sub_scale_node(intwidth, fracwidth, scale, polarity="bipolar"):
    """One sub_scale node with the given accumulator format, scale, and polarity."""
    return {"class": "sub_scale",
            "config": {"polarity": polarity, "scale": scale, "intwidth": intwidth,
                       "fracwidth": fracwidth}}


def encode_hold_node(timestep=256, trigger_timestep=100, generator=None):
    """One encode_hold node with the given trigger period and re-encode generator."""
    config = {"polarity": "unipolar", "timestep": timestep,
              "trigger_timestep": trigger_timestep}
    if generator is not None:
        config["generator"] = generator
    return {"class": "encode_hold", "config": config}


def series_node(class_name, generator="sobol", dim=1, timestep=256):
    """One exp_n1, log_n1, or tanh_p1 node with the given coefficient-stream sequence and length."""
    return {"class": class_name,
            "config": {"polarity": "unipolar", "timestep": timestep,
                       "generator": generator, "dim": dim}}


def div_cordiv_node(depth=8, generator="sobol", dim=None):
    """One div_cordiv node at the given history depth and quotient sequence."""
    config = {"depth": depth, "generator": generator}
    if dim is not None:
        config["dim"] = dim
    return {"class": "div_cordiv", "config": config}


def add_gaines_node(scaled=True, entry=8, generator="sobol", dim=5):
    """One add_gaines node at the given mode and selector sequence."""
    return {"class": "add_gaines",
            "config": {"polarity": "bipolar", "scaled": scaled, "entry": entry,
                       "generator": generator, "dim": dim}}


def decorr_node(generator="sys", seed=7, depth=4, timestep=256):
    """One decorr node at the given buffer sequence."""
    return {"class": "decorr",
            "config": {"polarity": "unipolar", "depth": depth, "timestep": timestep,
                       "generator": generator, "seed": seed}}


def counter_nodes():
    """One node per counter-sized operation, at its smallest legal counter width."""
    nodes = [{"class": "tanh_pn", "config": {"depth": 1}},
             {"class": "exp_n2g", "config": {"depth": 1}},
             {"class": "relu_cnt", "config": {"width": 1}},
             {"class": "signabs", "config": {"width": 1}}]
    for class_name in ("div_gaines", "sqrt_gaines"):
        for polarity in ("unipolar", "bipolar"):
            nodes.append({"class": class_name,
                          "config": {"polarity": polarity, "width": 1, "generator": "sobol"}})
    return nodes


def sequence_nodes(timestep):
    """(rtl_module, clause, node) per entry whose sequence counter is ceil(log2(timestep)) bits."""
    nodes = [("encode_hold", "config['timestep'] >= 2",
              encode_hold_node(timestep=timestep, trigger_timestep=1))]
    for polarity in ("unipolar", "bipolar"):
        nodes.append((f"mul_ugemm_{polarity}", "config['timestep'] >= 2",
                      {"class": "mul_ugemm", "config": {"polarity": polarity,
                                                        "timestep": timestep,
                                                        "generator": "sobol"}}))
        for rtl_module, node in ((f"linear_mix_{polarity}", linear_node("linear_mix", polarity)),
                                 (f"linear_ugemm_{polarity}", linear_node("linear_ugemm", polarity)),
                                 (f"linear_gaines_{polarity}", gaines_node("linear_gaines", polarity)),
                                 (f"conv_mix_{polarity}", conv_node(polarity)),
                                 (f"conv_ugemm_{polarity}",
                                  conv_node(polarity, class_name="conv_ugemm")),
                                 (f"conv_gaines_{polarity}", conv_gaines_node(polarity))):
            node["config"]["config"]["timestep"] = timestep
            nodes.append((rtl_module, "config['config']['timestep'] >= 2", node))
    return nodes


def zero_size_cases():
    """(rtl_module, reason, node) per zero or one-bit sizing an existing check rejects."""
    cases = []
    for class_name in ("desync", "max_sync", "min_sync", "sync"):
        cases.append((class_name, Rejected("Invalid depth"),
                      {"class": class_name, "config": {"polarity": "unipolar", "depth": 0}}))
    for class_name in ("relu_delay", "signabs_delay"):
        cases.append((class_name, Rejected("Invalid depth"),
                      {"class": class_name, "config": {"depth": 0}}))
    for class_name in ("relu_tc", "signabs_interleave"):
        cases.append((class_name, Rejected("Invalid width"),
                      {"class": class_name, "config": {"width": 0}}))
    # bi2uni's accumulator needs two bits before its emission threshold is reachable.
    for width in (0, 1):
        cases.append(("bi2uni", Rejected("emission threshold <1> is unreachable"),
                      {"class": "bi2uni", "config": {"width": width}}))
    for polarity in ("unipolar", "bipolar"):
        cases.append((f"mul_ugemm_regen_{polarity}", Rejected("Invalid width"),
                      {"class": "mul_ugemm_regen",
                       "config": {"polarity": polarity, "width": 0, "generator": "sobol"}}))
        for n in (0, 1):
            cases.append((f"pow_delay_{polarity}", Rejected("Invalid n"),
                          {"class": "pow_delay", "config": {"polarity": polarity, "n": n}}))
        # One integer bit is the sign alone, which holds no scale of two.
        for name, build, scale_key in (
                ("mul_scale", lambda **kw: mul_scale_node(polarity, kw.get("intwidth", 8), 0,
                                                          kw.get("scale", 2)), "scale"),
                ("div_scale", lambda **kw: div_scale_node(polarity, **kw), "scale"),
                ("div_scale_dyn", lambda **kw: div_scale_dyn_node(polarity, **kw), "scale_max"),
                ("add_scale_dyn", lambda **kw: add_scale_dyn_node(polarity, **dict({"intwidth": 8, "entry": 16}, **kw)),
                 "scale_max")):
            cases.append((f"{name}_{polarity}", Rejected("Invalid intwidth"), build(intwidth=0)))
            cases.append((f"{name}_{polarity}", Rejected("exceeds accumulator maximum"),
                          build(intwidth=1)))
            cases.append((f"{name}_{polarity}", Rejected(f"Invalid {scale_key}"),
                          build(**{scale_key: 0})))
        # A bias-only layer builds in the simulator, but the RTL input bus needs a bit.
        for class_name in ("linear_mix", "linear_ugemm"):
            node = linear_node(class_name, polarity)
            node["config"]["weight"] = torch.zeros(8, 0)
            cases.append((f"{class_name}_{polarity}", "IN_FEATURES >= 1", node))
        node = gaines_node("linear_gaines", polarity, in_features=0)
        cases.append((f"linear_gaines_{polarity}", Rejected("Invalid in_features"), node))
    return cases


def lane_nodes(lanes):
    """(rtl_module, node) per module whose lane count is its output-feature or hidden count."""
    nodes = [("mgu_hard_mix_bipolar", mgu_node(lanes=lanes, hidden=lanes))]
    for polarity in ("unipolar", "bipolar"):
        for class_name in ("linear_mix", "linear_ugemm"):
            node = linear_node(class_name, polarity, lanes=lanes)
            node["config"]["weight"] = torch.zeros(lanes, 16)
            node["config"]["bias"] = torch.zeros(lanes)
            nodes.append((f"{class_name}_{polarity}", node))
        node = gaines_node("linear_gaines", polarity, lanes=lanes)
        node["config"]["weight"] = torch.zeros(lanes, 15)
        node["config"]["bias"] = torch.zeros(lanes)
        nodes.append((f"linear_gaines_{polarity}", node))
    return nodes


def avgpool_node(kernel_size=2, lanes=48):
    """One avgpool2d_ugemm node with the given window and lane count."""
    return {"class": "avgpool2d_ugemm", "config": {"kernel_size": kernel_size, "lanes": lanes}}


def entry_nodes(entry):
    """One node per operation sized by input.size(dim), reducing `entry` inputs."""
    shape = {"input": {"shape": (4, entry)}}
    nodes = [{"class": "wta_tc", "config": {"polarity": "unipolar"}, "inputs": shape}]
    for polarity in ("unipolar", "bipolar"):
        nodes += [add_scale_node(polarity, 8, entry), add_scale_dyn_node(polarity, 8, entry),
                  add_ugemm_node(polarity, True, 10, entry, 1),
                  add_ugemm_node(polarity, False, 10, entry, 1)]
    return nodes


class Rejected(str):
    """A fragment of the simulation constructor's message for a config it rejects."""


def rejection_cases():
    """One node per (rtl_module, requires clause), each violating that clause.

    A case whose reason is `Rejected` breaks a constructor check instead, which
    translation reaches before any clause.
    """
    cases = [
        # A one-row buffer builds in the simulator but needs a zero-bit index.
        ("div_cordiv", "DEPTH >= 2", div_cordiv_node(depth=1)),
        ("div_cordiv", Rejected("is not power of 2"),
         div_cordiv_node(depth=6)),
        # The quotient history index is hardwired to Sobol dimension 1, so
        # another generator and another dimension are both different sequences.
        ("div_cordiv", "lower(get(config, 'generator', 'sobol')) in ['sobol', 'rc', 'rate']",
         div_cordiv_node(depth=4, generator="lfsr")),
        # Dimension 3's first four Sobol points are not the hardwired index order.
        ("div_cordiv", "model_seq('div_cordiv', config) == gray_index(DEPTH)",
         div_cordiv_node(depth=4, generator="sobol", dim=3)),
        # The constructor has no unscaled bipolar adder, and its scaled adder needs
        # the entry count and the generator named in the config.
        ("add_gaines", Rejected("Non-scaled Gaines addition for bipolar data"),
         add_gaines_node(scaled=False)),
        ("add_gaines", Rejected("requires <entry> and <generator>"),
         {"class": "add_gaines",
          "config": {"polarity": "bipolar", "scaled": True, "entry": 8, "dim": 5}}),
        # The scaled selector ROM is eight rows of one Sobol selector order.
        ("add_gaines",
         "not config['scaled'] or lower(get(config, 'generator', 'sobol')) in ['sobol', 'rc', 'rate']",
         add_gaines_node(generator="lfsr")),
        ("add_gaines", "not config['scaled'] or ENTRY == 8", add_gaines_node(entry=16)),
        ("add_gaines",
         "not config['scaled'] or model_seq('add_gaines', config) == [0, 4, 6, 2, 3, 7, 5, 1]",
         add_gaines_node(dim=2)),
        # The buffer-position ROM holds the `sys` sequences at seeds 7 and 8,
        # scaled by depth 4 and one word per timestep over 256 timesteps.
        ("decorr", "lower(get(config, 'generator', 'sys')) == 'sys'",
         decorr_node(generator="lfsr")),
        ("decorr", "get(config, 'seed', None) == 7", decorr_node(seed=3)),
        ("decorr", "DEPTH == 4", decorr_node(depth=8)),
        ("decorr", "SEQ_LEN == 256", decorr_node(timestep=128)),
        ("avgpool2d_ugemm", "get(config, 'stride') is None or get(config, 'stride') == config['kernel_size']",
         {"class": "avgpool2d_ugemm", "config": {"kernel_size": 2, "stride": 3, "lanes": 48}}),
        ("linear_gaines_bipolar", Rejected("does not support bipolar data"),
         gaines_node("linear_gaines", "bipolar", scaled=False)),
        ("linear_gaines_bipolar", Rejected("scaled mode needs entry >= 2"),
         gaines_node("linear_gaines", "bipolar", in_features=1, has_bias=False)),
        ("linear_gaines_bipolar", Rejected("scaled mode needs power-of-two entry"),
         gaines_node("linear_gaines", "bipolar", in_features=2)),
        # mgu_hard_mix is bipolar only, so its cases are listed once rather than per
        # polarity. Each case breaks one check with every other one satisfied.
        ("mgu_hard_mix_bipolar", Rejected("Invalid polarity"),
         mgu_node(polarity="unipolar")),
        ("mgu_hard_mix_bipolar", "shape(config['weight_f'])[0] == config['lanes']",
         mgu_node(lanes=4)),
        ("mgu_hard_mix_bipolar", "shape(config['weight_n'])[0] == config['lanes']",
         mgu_node(n_hidden=4)),
        ("mgu_hard_mix_bipolar", "shape(config['weight_n'])[1] == shape(config['weight_f'])[1]",
         mgu_node(n_features=8)),
        # A gate width equal to the hidden size leaves no input features.
        ("mgu_hard_mix_bipolar", "IN_SIZE >= 1", mgu_node(in_size=0)),
        ("mgu_hard_mix_bipolar",
         Rejected("accumulator width <3> too small"),
         mgu_node(width=3)),
        ("mgu_hard_mix_bipolar", "WIDTH <= 30", mgu_node(width=31)),
        # timestep 256 gives SEQ_WIDTH 8, so depth_ismul 8 is a run that cannot
        # outlast the multiplier shift register.
        ("mgu_hard_mix_bipolar", Rejected("Invalid timestep"), mgu_node(depth_ismul=8)),
        # The width case keeps a legal tolerance and the tolerance case keeps a
        # legal width.
        ("eq", Rejected("Invalid width"), {"class": "eq", "config": {"width": 0, "tolerance": 0}}),
        ("eq", Rejected("Invalid tolerance"),
         {"class": "eq", "config": {"width": 3, "tolerance": 5}}),
        # sync_skewed builds a zero-bit counter as a pass-through, so only
        # negative and fractional widths break its constructor.
        ("sync_skewed", Rejected("Invalid width"), {"class": "sync_skewed", "config": {"width": -1}}),
        ("sync_skewed", Rejected("Invalid width"), {"class": "sync_skewed", "config": {"width": 0.5}}),
        # A single-timestep stream sizes the sequence counter at zero bits.
        ("encode", "config['timestep'] >= 2", encode_node(timestep=1)),
        # decode counts a single-timestep stream in zero bits, which its RTL
        # implements, so only a non-integral or non-positive length is rejected.
        ("decode", Rejected("Invalid timestep"),
         {"class": "decode", "config": {"polarity": "unipolar", "timestep": 0.5}}),
        # The Sobol engine has no dimension below 1.
        ("encode", Rejected("dimensionality"), encode_node(dim=0)),
        ("encode", Rejected("dimensionality"), encode_node(dim=-1)),
        # Sobol dimension 5 builds in the simulator, but the RTL holds
        # direction-vector tables for dimensions 1 to 4 only.
        ("encode", "get(config, 'dim', 1) <= 4 or GENERATOR != 0", encode_node(dim=5)),
        # `sys` is a permutation drawn at construction, which no circuit rebuilds.
        ("encode",
         "lower(get(config, 'generator', 'sobol')) in "
         "['sobol', 'rc', 'rate', 'lfsr', 'lfsr_ext', 'tc', 'temporal']",
         encode_node(generator="sys")),
        # Two timesteps give WIDTH 1, which has no feedback polynomial, so the
        # simulator rejects it; 8192 timesteps give WIDTH 13, which the simulator
        # builds but which is past the last verified tap row.
        ("encode", Rejected("Invalid lfsr width"), encode_node(timestep=2, generator="lfsr")),
        ("encode",
         "lower(get(config, 'generator', 'sobol')) not in ['lfsr', 'lfsr_ext'] or "
         "(ceil(log2(config['timestep'])) >= 2 and ceil(log2(config['timestep'])) <= 12)",
         encode_node(timestep=8192, generator="lfsr_ext")),
        # A custom seed walks a different cycle than the resolved SEED.
        ("encode",
         "lower(get(config, 'generator', 'sobol')) not in ['lfsr', 'lfsr_ext'] or "
         "'seed' not in config",
         encode_node(generator="lfsr", seed=7)),
        # Custom taps walk a different polynomial than the resolved TAPS.
        ("encode",
         "lower(get(config, 'generator', 'sobol')) not in ['lfsr', 'lfsr_ext'] or "
         "'taps' not in config",
         encode_node(generator="lfsr", taps=[3, 1])),
        # sub_scale is single-variant. Clauses resolve in order: a unipolar
        # stream, then a too-wide accumulator, then a fractional grid, then a
        # non-integral scale.
        ("sub_scale", Rejected("Invalid polarity"),
         sub_scale_node(8, 0, 2, polarity="unipolar")),
        ("sub_scale", "config['intwidth'] <= 30", sub_scale_node(31, 0, 2)),
        ("sub_scale", "config['fracwidth'] == 0", sub_scale_node(8, 4, 2)),
        ("sub_scale", "config['scale'] == int(config['scale'])",
         sub_scale_node(8, 0, 2.5)),
        # negate.v is a bipolar circuit and subabs.v a unipolar one.
        ("negate", Rejected("Invalid polarity"),
         {"class": "negate", "config": {"polarity": "unipolar"}}),
        ("subabs", Rejected("Invalid polarity"),
         {"class": "subabs", "config": {"polarity": "bipolar"}}),
        # encode_hold clocks its sampler on a power-of-two trigger period.
        ("encode_hold", "2 ** int(log2(config['trigger_timestep'])) == config['trigger_timestep']",
         encode_hold_node(trigger_timestep=100)),
        # The re-encoder inside encode_hold.v is a Sobol encode instance, so the
        # trigger clause is satisfied here and the generator is what rejects.
        ("encode_hold", "lower(get(config, 'generator', 'sobol')) in ['sobol', 'rc', 'rate']",
         encode_hold_node(trigger_timestep=128, generator="lfsr")),
    ]
    # exp_n1.v, log_n1.v, and tanh_p1.v read coefficient streams baked from Sobol
    # dimensions 1 to 4, so a non-Sobol generator or another first dimension has no
    # hardware form. The ROM is baked for one WIDTH 8 period, so a 64-step or a
    # 512-step run has no verified form either, and each circuit is unipolar. Each
    # case keeps the other clauses satisfied.
    for class_name in ("exp_n1", "log_n1", "tanh_p1"):
        bipolar = series_node(class_name)
        bipolar["config"]["polarity"] = "bipolar"
        cases += [
            (class_name, Rejected("Invalid polarity"), bipolar),
            (class_name, "lower(get(config, 'generator', 'sobol')) in ['sobol', 'rc', 'rate']",
             series_node(class_name, generator="lfsr")),
            (class_name, "get(config, 'dim', 1) == 1", series_node(class_name, dim=5)),
            (class_name, "WIDTH == 8", series_node(class_name, timestep=64)),
            (class_name, "WIDTH == 8", series_node(class_name, timestep=512)),
        ]
    for polarity in ("unipolar", "bipolar"):
        # The clamp constructors take bounds from `low` to 1 with lo strictly
        # below hi, and each case keeps its other bound legal and on the grid.
        low = -1 if polarity == "bipolar" else 0
        cases += [
            (f"clamp_sat_{polarity}", Rejected("Invalid lo"),
             clamp_sat_node(polarity, lo=low - 0.5, hi=0.5)),
            (f"clamp_sat_{polarity}", Rejected("Invalid hi"),
             clamp_sat_node(polarity, lo=0.0, hi=1.5)),
            (f"clamp_sat_{polarity}", Rejected("Invalid band"),
             clamp_sat_node(polarity, lo=0.5, hi=0.5)),
            ("clamp_comp", Rejected("Invalid lo"),
             {"class": "clamp_comp", "config": {"polarity": polarity, "lo": low - 0.5,
                                                "hi": 0.5}}),
            ("clamp_comp", Rejected("Invalid hi"),
             {"class": "clamp_comp", "config": {"polarity": polarity, "lo": 0.0,
                                                "hi": 1.5}}),
            ("clamp_comp", Rejected("Invalid band"),
             {"class": "clamp_comp", "config": {"polarity": polarity, "lo": 0.5,
                                                "hi": 0.5}}),
        ]
        # WIDTH 31 overflows the 32-bit constants on its own; WIDTH 30 with a
        # reduction of 400 million entries overflows only through the ENTRY term.
        cases += [
            (f"add_scale_{polarity}", "WIDTH <= 30", add_scale_node(polarity, 31, 16)),
            (f"add_scale_{polarity}", "2 ** WIDTH + 3 * ENTRY + 1 <= 2 ** 31 - 1",
             add_scale_node(polarity, 30, 400_000_000)),
            # The RTL accumulator is integer-only, so a fractional grid has no hardware form.
            (f"add_scale_{polarity}", "config['fracwidth'] == 0",
             add_scale_node(polarity, 8, 16, fracwidth=4)),
            # On the integer grid the model rounds the scale, so only an integral
            # request reaches the RTL unchanged.
            (f"add_scale_{polarity}", "config['scale'] == int(config['scale'])",
             add_scale_node(polarity, 8, 16, scale=2.5)),
            # add_scale_dyn carries the scale bus alongside the accumulator, so its
            # 32-bit bound holds one more power-of-two term than add_scale's.
            (f"add_scale_dyn_{polarity}", "WIDTH <= 30",
             add_scale_dyn_node(polarity, 31, 16)),
            (f"add_scale_dyn_{polarity}",
             "2 ** WIDTH + 2 ** (WIDTH - 1) + 3 * ENTRY + 1 <= 2 ** 31 - 1",
             add_scale_dyn_node(polarity, 30, 400_000_000)),
            (f"add_scale_dyn_{polarity}", "config['fracwidth'] == 0",
             add_scale_dyn_node(polarity, 8, 16, fracwidth=4)),
            (f"div_scale_{polarity}", "config['fracwidth'] == 0",
             div_scale_node(polarity, fracwidth=4)),
            (f"div_scale_{polarity}", "config['scale'] == int(config['scale'])",
             div_scale_node(polarity, scale=2.5)),
            (f"div_scale_dyn_{polarity}", "config['fracwidth'] == 0",
             div_scale_dyn_node(polarity, fracwidth=4)),
            # conv_gaines sizes itself from the weight and the NCHW input shape, so a
            # channel count the weight disagrees with is rejected before the geometry.
            (f"conv_gaines_{polarity}", "shape(config['weight'])[1] == input.size(1)",
             conv_gaines_node(polarity, input_channels=2)),
            (f"conv_gaines_{polarity}",
             "LANES == BATCH * OUT_CHANNELS * ((IN_H + 2 * PADDING - DILATION * (KERNEL_H - 1) - 1) // STRIDE + 1) * ((IN_W + 2 * PADDING - DILATION * (KERNEL_W - 1) - 1) // STRIDE + 1)",
             conv_gaines_node(polarity, lanes=28)),
            (f"linear_mix_{polarity}", "shape(config['weight'])[0] == config['lanes']",
             linear_node("linear_mix", polarity, lanes=4)),
            (f"linear_mix_{polarity}", Rejected("too small for fan-in"),
             linear_node("linear_mix", polarity, width=4)),
            # The inner add_scale runs on the integer grid, so only an integral
            # scale reaches the RTL unchanged.
            (f"linear_mix_{polarity}", "SCALE == int(SCALE)",
             linear_node("linear_mix", polarity, scale=3.4)),
            (f"linear_gaines_{polarity}", "shape(config['weight'])[0] == config['lanes']",
             gaines_node("linear_gaines", polarity, lanes=4)),
            (f"linear_ugemm_{polarity}", "shape(config['weight'])[0] == config['lanes']",
             linear_node("linear_ugemm", polarity, lanes=4)),
            (f"linear_ugemm_{polarity}", Rejected("too small for fan-in"),
             linear_node("linear_ugemm", polarity, width=4)),
            (f"linear_ugemm_{polarity}", "WIDTH <= 30",
             linear_node("linear_ugemm", polarity, width=31)),
            (f"linear_ugemm_{polarity}", "SCALE == int(SCALE)",
             linear_node("linear_ugemm", polarity, scale=3.4)),
            (f"conv_mix_{polarity}", "shape(config['weight'])[1] == input.size(1)",
             conv_node(polarity, in_channels=4)),
            (f"conv_mix_{polarity}", Rejected("too small for fan-in"),
             conv_node(polarity, width=4)),
            (f"conv_mix_{polarity}",
             "LANES == BATCH * OUT_CHANNELS * ((IN_H + 2 * PADDING - DILATION * (KERNEL_H - 1) - 1) // STRIDE + 1) * ((IN_W + 2 * PADDING - DILATION * (KERNEL_W - 1) - 1) // STRIDE + 1)",
             conv_node(polarity, lanes=28)),
            (f"conv_mix_{polarity}", "SCALE == int(SCALE)",
             conv_node(polarity, scale=3.4)),
            # conv_ugemm generates its weight and bias streams in hardware, so it
            # carries conv's clauses plus the add_scale width cap its lanes inherit.
            (f"conv_ugemm_{polarity}", "shape(config['weight'])[1] == input.size(1)",
             conv_node(polarity, in_channels=4, class_name="conv_ugemm")),
            (f"conv_ugemm_{polarity}",
             Rejected("too small for fan-in"),
             conv_node(polarity, width=4, class_name="conv_ugemm")),
            (f"conv_ugemm_{polarity}",
             "LANES == BATCH * OUT_CHANNELS * ((IN_H + 2 * PADDING - DILATION * (KERNEL_H - 1) - 1) // STRIDE + 1) * ((IN_W + 2 * PADDING - DILATION * (KERNEL_W - 1) - 1) // STRIDE + 1)",
             conv_node(polarity, lanes=28, class_name="conv_ugemm")),
            (f"conv_ugemm_{polarity}", "WIDTH <= 30",
             conv_node(polarity, width=31, class_name="conv_ugemm")),
            (f"conv_ugemm_{polarity}", "SCALE == int(SCALE)",
             conv_node(polarity, scale=3.4, class_name="conv_ugemm")),
            # The Gaines weight-threshold table paths hold a two-digit feature
            # index, so a 127-tap patch and a 127-feature weight are past the
            # last table the index can name.
            (f"conv_gaines_{polarity}", "IN_CHANNELS * KERNEL_H * KERNEL_W <= 100",
             conv_gaines_node(polarity, in_channels=127, kernel_h=1, lanes=192)),
            (f"linear_gaines_{polarity}", "IN_FEATURES <= 100",
             gaines_node("linear_gaines", polarity, in_features=127)),
            # clamp_sat's bounds each round onto the fractional grid, so a bound off
            # that grid has no integer form. The lo clause resolves before the hi
            # clause, so each case keeps the other bound on the grid.
            (f"clamp_sat_{polarity}",
             "config['lo'] * 2 ** FRACWIDTH == floor(config['lo'] * 2 ** FRACWIDTH + 0.5)",
             clamp_sat_node(polarity, lo=0.001, hi=1.0)),
            (f"clamp_sat_{polarity}",
             "config['hi'] * 2 ** FRACWIDTH == floor(config['hi'] * 2 ** FRACWIDTH + 0.5)",
             clamp_sat_node(polarity, lo=0.0, hi=0.501)),
            # The band constants ride Sobol dimensions dim, dim+1, dim+2, whose
            # direction vectors the RTL bakes in, so only the default dimension
            # translates.
            (f"clamp_sat_{polarity}", "get(config, 'dim', 2) == 2",
             clamp_sat_node(polarity, lo=0.0, hi=1.0, dim=5)),
            # The direction-vector tables are baked for the default fracwidth, so a
            # narrower grid has no hardware form. The bounds stay on that narrower
            # grid and the dimension stays default, so only this clause rejects.
            (f"clamp_sat_{polarity}", "get(config, 'fracwidth', 8) == 8",
             clamp_sat_node(polarity, lo=0.0, hi=1.0, fracwidth=4)),
            # clamp_comp encodes each bound as a probability code on the 2**-SEQ_W
            # threshold grid, so a bound off that grid has no integer code. The lo
            # clause resolves before the hi clause, so each case keeps the other
            # bound on the grid and the dimension at its default.
            ("clamp_comp",
             "(config['lo'] if config['polarity'] == 'unipolar' else (config['lo'] + 1) / 2) * 2 ** SEQ_W == LO_CODE",
             clamp_comp_node(polarity, lo_offset=CLAMP_COMP_OFF_GRID)),
            ("clamp_comp",
             "(config['hi'] if config['polarity'] == 'unipolar' else (config['hi'] + 1) / 2) * 2 ** SEQ_W == HI_CODE",
             clamp_comp_node(polarity, hi_offset=CLAMP_COMP_OFF_GRID)),
            # The two bound streams ride Sobol dimensions dim and dim + 1, whose
            # direction vectors the RTL bakes in, so only the default dimension
            # translates. Both bounds stay on the grid, so only this clause rejects.
            ("clamp_comp", "get(config, 'dim', 2) == 2",
             clamp_comp_node(polarity, dim=5)),
            # mul_scale runs on the integer grid. Clauses resolve in order: a
            # too-wide accumulator, then a fractional grid, then a non-integral
            # scale, then a scale past the accumulator's signed range.
            (f"mul_scale_{polarity}", "config['intwidth'] <= 30",
             mul_scale_node(polarity, 31, 0, 2)),
            (f"mul_scale_{polarity}", "config['fracwidth'] == 0",
             mul_scale_node(polarity, 8, 4, 2)),
            (f"mul_scale_{polarity}", "config['scale'] == int(config['scale'])",
             mul_scale_node(polarity, 8, 0, 2.5)),
            (f"mul_scale_{polarity}", Rejected("exceeds accumulator maximum"),
             mul_scale_node(polarity, 3, 0, 4)),
            # The RTL add_ugemm has no clamp, so the width must hold the run's peak:
            # scaled, 2*5-1 = 9 passes width 4's maximum of 7.
            (f"add_ugemm_{polarity}",
             "not config['scaled'] or ENTRY + (ENTRY - 1) * (len(SEGMENT) > 1) <= 2 ** (get(config, 'width', 10) - 1) - 1",
             add_ugemm_node(polarity, True, width=4, entry=5, segment=2)),
        ]
        if polarity == "unipolar":
            cases += [
                # Non-scaled, the unipolar gap peaks at 7*256 + 1 = 1793, past
                # width 11's maximum of 1023.
                ("add_ugemm_unipolar",
                 "config['scaled'] or (ENTRY - 1) * len(SEGMENT) + 1 <= 2 ** (get(config, 'width', 10) - 1) - 1",
                 add_ugemm_node(polarity, False, width=11, entry=8, segment=256)),
                ("linear_gaines_unipolar", Rejected("scaled mode needs entry >= 2"),
                 gaines_node("linear_gaines", polarity, in_features=1, has_bias=False)),
                ("linear_gaines_unipolar",
                 Rejected("scaled mode needs power-of-two entry"),
                 gaines_node("linear_gaines", polarity, in_features=2)),
                # The constructor checks the fan-in only for the scaled adder; a
                # kernel of one tap with no bias leaves a single addend and a
                # three-tap kernel leaves an odd fan-in.
                ("conv_gaines_unipolar",
                 Rejected("scaled mode needs entry >= 2"),
                 conv_gaines_node(polarity, kernel_h=1, has_bias=False, lanes=192)),
                ("conv_gaines_unipolar",
                 Rejected("scaled mode needs power-of-two entry"),
                 conv_gaines_node(polarity, has_bias=False)),
            ]
        else:
            # The unipolar accumulator never reaches its clamp, so only the bipolar
            # div_scale variants carry WIDTH and its 32-bit bound.
            cases += [
                # Non-scaled, twice the bipolar gap peaks at 7*256 + 2 = 1794, past
                # width 10's bound of 2**10 - 2 = 1022.
                ("add_ugemm_bipolar",
                 "config['scaled'] or (ENTRY - 1) * len(SEGMENT) + 2 <= 2 ** get(config, 'width', 10) - 2",
                 add_ugemm_node(polarity, False, width=10, entry=8, segment=256)),
                ("div_scale_bipolar", "WIDTH <= 30", div_scale_node(polarity, intwidth=31)),
                ("div_scale_dyn_bipolar", "WIDTH <= 30",
                 div_scale_dyn_node(polarity, intwidth=31)),
                # A bipolar conv_gaines has no OR adder, so the constructor rejects
                # unscaled mode and checks the fan-in unconditionally.
                ("conv_gaines_bipolar", Rejected("does not support bipolar data"),
                 conv_gaines_node(polarity, scaled=False)),
                ("conv_gaines_bipolar",
                 Rejected("scaled mode needs entry >= 2"),
                 conv_gaines_node(polarity, kernel_h=1, has_bias=False, lanes=192)),
                ("conv_gaines_bipolar",
                 Rejected("scaled mode needs power-of-two entry"),
                 conv_gaines_node(polarity, has_bias=False)),
            ]
    # A zero-bit counter has no half-scale state, so each constructor rejects it.
    for node in counter_nodes():
        key = "depth" if "depth" in node["config"] else "width"
        for value in (0, -1, 0.5):
            broken = dict(node, config=dict(node["config"], **{key: value}))
            cases.append((node["class"], Rejected(f"Invalid {key}"), broken))
    # A single-timestep stream sizes the sequence counter at zero bits.
    cases += sequence_nodes(1)
    cases += zero_size_cases()
    # A zero-lane layer builds in the simulator, but the RTL has no lane to generate.
    cases += [(rtl_module, "LANES >= 1", node) for rtl_module, node in lane_nodes(0)]
    # lanes is the caller's elaboration count, which no constructor reads.
    for lanes in (0, -1, 0.5):
        cases.append(("avgpool2d_ugemm", "LANES == int(LANES) and LANES >= 1",
                      avgpool_node(lanes=lanes)))
    for polarity in ("unipolar", "bipolar"):
        node = conv_gaines_node(polarity)
        node["config"]["weight"] = torch.zeros(0, 1, 3, 1)
        node["config"]["bias"] = torch.zeros(0)
        cases.append((f"conv_gaines_{polarity}", Rejected("Invalid out_channels"), node))
        for class_name in ("conv_mix", "conv_ugemm"):
            node = conv_node(polarity, class_name=class_name)
            node["config"]["weight"] = torch.zeros(0, 2, 3, 3)
            node["config"]["bias"] = torch.zeros(0)
            cases.append((f"{class_name}_{polarity}", Rejected("Invalid out_channels"), node))
    # Unscaled mode skips conv_gaines' fan-in check, so its linear_gaines core rejects it.
    node = conv_gaines_node("unipolar", in_channels=0, scaled=False)
    cases.append(("conv_gaines_unipolar", Rejected("Invalid in_features"), node))
    for kernel_size in (-1, (2, -1), 0, (2, 0), 2.0, True, (2, 2, 2)):
        cases.append(("avgpool2d_ugemm", Rejected("Invalid kernel_size"),
                      avgpool_node(kernel_size=kernel_size)))
    # A zero-bit skew counter builds in the simulator as a pass-through.
    cases.append(("sync_skewed", "WIDTH >= 1", {"class": "sync_skewed", "config": {"width": 0}}))
    return cases


def test_every_requires_clause_rejects():
    """Every requires clause in the mapping rejects a configuration that breaks it."""
    covered = set()
    for rtl_module, clause, node in rejection_cases():
        with pytest.raises(TranslationError) as raised:
            translate_node(node)
        message = str(raised.value)
        if isinstance(clause, Rejected):
            assert "rejects this configuration" in message, f"{clause}: {message}"
            assert clause in message, f"{clause}: {message}"
            continue
        assert rtl_module in message, f"{clause}: {message}"
        # A clause that raises is reported with its text too, so only this
        # wording shows the clause evaluated to false.
        assert f"{clause} is false" in message, f"{clause}: {message}"
        covered.add((rtl_module, clause))
    assert covered == requires_clauses()


def test_smallest_counter_and_entry_translate():
    """The smallest legal counter width, reduction size, timestep, lane count, and window resolve."""
    for node in counter_nodes():
        parameters = translate_node(node).parameters
        assert parameters.get("DEPTH", parameters.get("WIDTH")) == 1, node
    for node in entry_nodes(1):
        assert translate_node(node).parameters["ENTRY"] == 1, node
    assert translate_node({"class": "sync_skewed", "config": {"width": 1}}).parameters["WIDTH"] == 1
    for rtl_module, node in lane_nodes(1):
        binding = translate_node(node)
        assert (binding.rtl_module, binding.parameters["LANES"]) == (rtl_module, 1)
    assert translate_node(avgpool_node(lanes=1)).parameters["LANES"] == 1
    for kernel_size, area in ((1, 1), ((1, 1), 1), ([1, 2], 2)):
        assert translate_node(avgpool_node(kernel_size=kernel_size)).parameters["KERNEL_AREA"] == area
    for polarity in ("unipolar", "bipolar"):
        assert translate_node(mul_scale_node(polarity, 8, 0, 1)).parameters["SCALE"] == 1
        assert translate_node(div_scale_node(polarity, scale=1)).parameters["SCALE"] == 1
        assert translate_node(div_scale_dyn_node(polarity, scale_max=1)).parameters["SCALE_W"] == 1
        assert translate_node(add_scale_dyn_node(polarity, 8, 16, scale_max=1)).parameters["SCALE_W"] == 1
        for class_name in ("linear_mix", "linear_ugemm"):
            node = linear_node(class_name, polarity)
            node["config"]["weight"] = torch.zeros(8, 1)
            assert translate_node(node).parameters["IN_FEATURES"] == 1
    for rtl_module, _, node in sequence_nodes(2):
        binding = translate_node(node)
        assert binding.rtl_module == rtl_module
        assert binding.parameters.get("SEQ_WIDTH", binding.parameters.get("WIDTH")) == 1, rtl_module


def test_empty_input_dimension_rejects():
    """An input.size(dim) of zero is rejected, since the RTL has no zero-input form."""
    nodes = entry_nodes(0)
    for polarity in ("unipolar", "bipolar"):
        for class_name in ("conv_mix", "conv_ugemm"):
            node = conv_node(polarity, in_channels=0, class_name=class_name)
            node["config"]["weight"] = torch.zeros(3, 0, 3, 3)
            nodes.append(node)
    for node in nodes:
        with pytest.raises(TranslationError, match="is empty"):
            translate_node(node)


def test_encode_resolves_a_one_bit_sequence():
    """A two-timestep encode stream resolves, so its one-bit counter is legal hardware."""
    binding = translate_node(encode_node(timestep=2))
    assert binding.rtl_module == "encode"
    assert binding.parameters["WIDTH"] == 1


def test_encode_accepts_a_cased_generator_name():
    """The simulation model lowercases the generator name, so `Sobol` is the Sobol circuit."""
    assert translate_node(encode_node(generator="Sobol")).parameters["GENERATOR"] == 0
    assert translate_node(encode_node(generator="LFSR")).parameters["GENERATOR"] == 1
    with pytest.raises(TranslationError, match="generator"):
        translate_node(encode_node(generator="SYS"))


def test_encode_selects_one_sequence_circuit_per_generator():
    """Each permitted generator resolves its own GENERATOR branch, not Sobol's."""
    modes = {"sobol": 0, "rc": 0, "rate": 0, "lfsr": 1, "lfsr_ext": 2,
             "tc": 3, "temporal": 3}
    for generator, expected in modes.items():
        parameters = translate_node(encode_node(generator=generator)).parameters
        assert parameters["GENERATOR"] == expected, generator
        # Only the LFSR modes read a tap mask; the rest resolve the unused 0.
        assert parameters["TAPS"] == (113 if expected in (1, 2) else 0), generator


def test_requires_clause_error_names_the_entry_and_clause():
    """A clause that cannot be evaluated is reported against its entry and text."""
    entry = {"rtl_module": "probe_module", "requires": ["log2(0) == 0"]}
    with pytest.raises(TranslationError,
                       match="requires clause 'log2\\(0\\) == 0' for RTL module 'probe_module'"):
        _check_requires(entry, _RestrictedEvaluator({}))


def test_exp_n2g_gain_defaults_to_the_simulator_default():
    """A node without `gain` resolves the simulation class's default of 1."""
    binding = translate_node({"class": "exp_n2g", "config": {"depth": 5}})
    assert binding.parameters == {"DEPTH": 5, "GAIN": 1}


def test_reserved_name_shadows_a_config_key_of_the_same_name():
    """Keep a config key named `input` readable, and let a real input shape win.

    No simulation class accepts a config key named `input`, so translation
    rejects such a node at construction; the parameter resolver is called
    directly to reach the name lookup.
    """
    entry = next(item for item in load_mapping()
                 if item.get("rtl_module") == "mul_gaines_unipolar")
    node = {"class": "mul_gaines", "config": {"polarity": "unipolar", "input": 5}}
    assert _resolve_parameters(dict(entry, parameters={"WIDTH": "input"}),
                               node, node["config"]) == {"WIDTH": 5}
    shaped = dict(node, inputs={"input": {"shape": (4, 16)}})
    assert _resolve_parameters(dict(entry, parameters={"WIDTH": "input.size(-1)"}),
                               shaped, shaped["config"]) == {"WIDTH": 16}
    with pytest.raises(TranslationError, match="Unknown key <input>"):
        translate_node(node)


def test_bindings_match_rtl_module_headers():
    """Check resolved parameters and directional ports against real RTL headers."""
    nodes = module_nodes() + [
        # conv_ugemm holds its own weight and bias encoders, so its header carries
        # SEQ_WIDTH on top of conv's parameters and no pad port.
        conv_node("unipolar", class_name="conv_ugemm"),
        {
            "class": "add_scale",
            "config": {"polarity": "bipolar", "scale": 2, "intwidth": 8, "fracwidth": 0},
            "inputs": {"input": {"shape": (4, 16)}},
        },
        {
            "class": "add_scale",
            "config": {"polarity": "unipolar", "scale": 2, "intwidth": 8, "fracwidth": 0},
            "inputs": {"input": {"shape": (4, 16)}},
        },
        # mgu_hard_mix composes two linear layers, so its header carries the gate widths
        # and the multiplier sequence widths rather than a single SCALE.
        mgu_node(),
        # conv_gaines reduces one im2col patch through the Gaines adder, so its
        # header carries SCALE_WIDTH rather than an accumulator SCALE, and only the
        # unipolar variant carries SCALED.
        conv_gaines_node("unipolar"),
        conv_gaines_node("bipolar"),
        {"class": "delay", "config": {"depth": 4, "init": "alternate"}},
        div_cordiv_node(),
        {"class": "mul_gaines", "config": {"polarity": "unipolar"}},
        {"class": "relu_sat", "config": {}},
        {"class": "sync_skewed", "config": {"width": 4}},
        {"class": "jkff", "config": {}},
    ]
    port_pattern = re.compile(
        r"\b(input|output)\s+(?:(?:wire|reg|logic|signed)\s+)*"
        r"(?:\[[^\]]+\]\s*)?([A-Za-z_]\w*)"
    )
    for node in nodes:
        binding = translate_node(node)
        source = binding.file_path.read_text(encoding="utf-8")
        source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)
        module_start = re.search(
            rf"\bmodule\s+{re.escape(binding.rtl_module)}\b", source
        )
        assert module_start is not None
        module_end = re.search(r"\)\s*;", source[module_start.end():])
        assert module_end is not None
        header_end = module_start.end() + module_end.end()
        header = source[module_start.start():header_end]
        parameter_names = set(re.findall(
            r"\bparameter(?:\s+\w+)*\s+(\w+)", header
        ))
        ports = {"input": set(), "output": set()}
        for direction, port in port_pattern.findall(header):
            ports[direction].add(port)
        assert set(binding.parameters) <= parameter_names
        for port in binding.port_map["inputs"].values():
            if port is not None:
                assert port in ports["input"]
        for port in binding.port_map["outputs"].values():
            if port is not None:
                assert port in ports["output"]
        # A null-mapped argument must have no port at all: the RTL port name is
        # the prefix plus the argument name, so a module that grew one the
        # mapping nulls out fails here.
        for name, port in binding.port_map["inputs"].items():
            if port is None:
                assert f"i_{name}" not in ports["input"], \
                    f"{binding.rtl_module} has port i_{name} that the mapping nulls out"
        for name, port in binding.port_map["outputs"].items():
            if port is None:
                assert f"o_{name}" not in ports["output"], \
                    f"{binding.rtl_module} has port o_{name} that the mapping nulls out"


def test_output_ports_follow_the_o_prefix_convention():
    """Every mapped output port is ``o_`` plus its simulation output key."""
    # No entry deviates, so the convention holds with no exception list: a port
    # that needs a different name needs its key renamed instead.
    pairs = 0
    for entry in load_mapping():
        for key, port in (entry.get("outputs") or {}).items():
            if port is None:
                continue
            pairs += 1
            assert port == f"o_{key}", (
                f"{entry['rtl_module']} maps output key {key} to port {port}, "
                f"but the convention is o_{key}"
            )
    assert pairs > 0


def test_null_mapped_port_rejects_a_supplied_source():
    """Reject a node that drives an input the RTL variant has no port for."""
    unipolar_conv = dict(module_nodes()[-2])
    assert translate_node(unipolar_conv).port_map["inputs"]["pad_bits"] is None
    unipolar_conv["inputs"] = dict(unipolar_conv["inputs"],
                                   pad_bits={"shape": (1,)})
    with pytest.raises(TranslationError, match="pad_bits"):
        translate_node(unipolar_conv)


if __name__ == "__main__":
    test_polarity_variant_and_derived_parameter()
    test_graph_translation_preserves_order()
    test_missing_rtl_mapping_names_class()
    test_module_layer_parameters()
    test_module_rejects_unsupported_configuration()
    test_layer_key_default_and_validation()
    test_every_requires_clause_rejects()
    test_smallest_counter_and_entry_translate()
    test_empty_input_dimension_rejects()
    test_encode_resolves_a_one_bit_sequence()
    test_encode_accepts_a_cased_generator_name()
    test_encode_selects_one_sequence_circuit_per_generator()
    test_reserved_name_shadows_a_config_key_of_the_same_name()
    test_bindings_match_rtl_module_headers()
    test_output_ports_follow_the_o_prefix_convention()
    test_null_mapped_port_rejects_a_supplied_source()
    print("Test passed.")

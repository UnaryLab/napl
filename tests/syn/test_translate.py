import re
from pathlib import Path

import pytest
import torch
import yaml

from napl.syn.translate import _MAPPING_PATH
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
        "class": "add_any",
        "config": {"polarity": "bipolar", "scale": 2, "width": 8},
        "inputs": {"input": {"shape": (4, 16)}},
    })
    assert add.rtl_module == "add_any_bipolar"
    assert add.parameters == {"SCALE": 2, "WIDTH": 8, "ENTRY": 16}
    assert add.port_map["inputs"]["input"] == "i_input"
    assert add.port_map["output"] == "o_out"
    assert Path(add.file_path).name == "add_any_bipolar.v"

    divider = translate_node({
        "class": "div_cordiv",
        "config": {"depth": 8},
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
         "inputs": {"input_spike": {"shape": (1, 2, 6, 6)}}},
        {"class": "conv_mix",
         "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": None,
                    "stride": 2, "padding": 2, "dilation": 2,
                    "config": dict(layer_config, polarity='bipolar'), "lanes": 27},
         "inputs": {"input_spike": {"shape": (1, 2, 6, 6)}}},
    ]


def test_module_layer_parameters():
    """Resolve every module entry from constructor-name node configs."""
    (pool, pool_rect, ugemm, ugemm_nb, linear, linear_nb,
     convolution, convolution_nb) = [translate_node(node) for node in module_nodes()]

    assert pool.rtl_module == "avgpool2d"
    assert Path(pool.file_path).parts[-3:] == ("avgpool2d", "rtl", "avgpool2d.v")
    assert pool.parameters == {"KERNEL_AREA": 4, "LANES": 48}
    # A tuple kernel resolves to the window area.
    assert pool_rect.parameters == {"KERNEL_AREA": 6, "LANES": 48}

    assert ugemm.rtl_module == "linear_ugemm_unipolar"
    assert ugemm.parameters == {"IN_FEATURES": 16, "LANES": 8, "SEQ_WIDTH": 8,
                                "WIDTH": 12, "HAS_BIAS": 1, "SCALE": 17}
    assert ugemm.port_map["inputs"]["weight"] == "i_weight"
    assert ugemm.port_map["output"] == "o_out"
    # No bias drops an addend, so the default scale shrinks with it.
    assert ugemm_nb.rtl_module == "linear_ugemm_bipolar"
    assert ugemm_nb.parameters["HAS_BIAS"] == 0
    assert ugemm_nb.parameters["SCALE"] == 16

    # linear encodes its weight outside the RTL, so it carries no SEQ_WIDTH.
    assert linear.rtl_module == "linear_unipolar"
    assert linear.parameters == {"IN_FEATURES": 16, "LANES": 8, "WIDTH": 12,
                                 "HAS_BIAS": 1, "SCALE": 17}
    assert linear.port_map["inputs"]["input_spike"] == "i_input_spike"
    assert linear.port_map["output"] == "o_out"
    assert linear_nb.rtl_module == "linear_bipolar"
    assert linear_nb.parameters["HAS_BIAS"] == 0
    assert linear_nb.parameters["SCALE"] == 16

    # conv reads its lane geometry from the weight and the NCHW input shape.
    assert convolution.rtl_module == "conv_unipolar"
    assert convolution.parameters == {"BATCH": 1, "IN_CHANNELS": 2, "IN_H": 6, "IN_W": 6,
                                      "OUT_CHANNELS": 3, "KERNEL_H": 3, "KERNEL_W": 3,
                                      "STRIDE": 1, "PADDING": 1, "DILATION": 1,
                                      "WIDTH": 12, "HAS_BIAS": 1, "SCALE": 19, "LANES": 108}
    # A unipolar zero pad is a constant inside the module, so the unipolar entry
    # maps pad_bits to no port; the bipolar entry carries one.
    assert convolution.port_map["inputs"]["pad_bits"] is None
    assert convolution.port_map["output"] == "o_out"
    # Stride and dilation shrink the output positions, and no bias shrinks the scale.
    assert convolution_nb.rtl_module == "conv_bipolar"
    assert convolution_nb.port_map["inputs"]["pad_bits"] == "i_pad_bits"
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


def add_any_node(polarity, width, entry):
    """One add_any node with the given accumulator width and reduction size."""
    return {"class": "add_any",
            "config": {"polarity": polarity, "scale": 2, "width": width},
            "inputs": {"input": {"shape": (4, entry)}}}


def linear_node(class_name, polarity, lanes=8, width=12):
    """One linear_mix or linear_ugemm node over an 8x16 weight."""
    return {"class": class_name,
            "config": {"weight": torch.zeros(8, 16), "bias": torch.zeros(8),
                       "config": dict(LAYER_CONFIG, polarity=polarity, width=width),
                       "lanes": lanes}}


def gaines_node(class_name, polarity, in_features=15, has_bias=True, lanes=8,
                generator='sobol', scaled=True):
    """One linear_gaines node over a `lanes` x in_features weight."""
    config = dict(LAYER_CONFIG, polarity=polarity, generator=generator, scaled=scaled)
    return {"class": class_name,
            "config": {"weight": torch.zeros(8, in_features),
                       "bias": torch.zeros(8) if has_bias else None,
                       "config": config, "lanes": lanes}}


def conv_node(polarity, in_channels=2, lanes=108, width=12, class_name="conv_mix"):
    """One conv node over a 3x2x3x3 weight and a 1x2x6x6 input."""
    return {"class": class_name,
            "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": torch.zeros(3),
                       "stride": 1, "padding": 1, "dilation": 1,
                       "config": dict(LAYER_CONFIG, polarity=polarity, width=width),
                       "lanes": lanes},
            "inputs": {"input_spike": {"shape": (1, in_channels, 6, 6)}}}


def mgu_node(lanes=3, hidden=3, in_size=4, width=10, depth_ismul=6, n_hidden=None,
             n_features=None):
    """One mgu_hard_mix node over gate weights shaped (hidden, hidden + in_size).

    `n_hidden` and `n_features` override the new-gate weight shape on its own, so
    a gate pair that disagrees can be built without touching the forget gate.
    """
    features = hidden + in_size
    config = {'polarity': 'bipolar', 'timestep': 256, 'generator': 'sobol',
              'width': width, 'depth_ismul': depth_ismul}
    return {"class": "mgu_hard_mix",
            "config": {"weight_f": torch.zeros(hidden, features),
                       "bias_f": torch.zeros(hidden),
                       "weight_n": torch.zeros(n_hidden or hidden,
                                               n_features or features),
                       "bias_n": torch.zeros(hidden),
                       "hx_value": torch.zeros(1, hidden),
                       "config": config, "lanes": lanes}}


def rejection_cases():
    """One node per (rtl_module, requires clause), each violating that clause."""
    cases = [
        ("div_cordiv", "2 ** int(WIDTH) == DEPTH",
         {"class": "div_cordiv", "config": {"depth": 6}}),
        ("avgpool2d", "get(config, 'stride') is None or get(config, 'stride') == config['kernel_size']",
         {"class": "avgpool2d_ugemm", "config": {"kernel_size": 2, "stride": 3, "lanes": 48}}),
        ("linear_gaines_bipolar", "get(config['config'], 'scaled', True)",
         gaines_node("linear_gaines", "bipolar", scaled=False)),
        ("linear_gaines_bipolar", "IN_FEATURES + HAS_BIAS >= 2",
         gaines_node("linear_gaines", "bipolar", in_features=1, has_bias=False)),
        ("linear_gaines_bipolar", "2 ** SCALE_WIDTH == IN_FEATURES + HAS_BIAS",
         gaines_node("linear_gaines", "bipolar", in_features=2)),
        # mgu_hard_mix is bipolar only, so its clauses are listed once rather than per
        # polarity. Clauses resolve in order, so each case below breaks its own
        # clause with every earlier one still satisfied.
        ("mgu_bipolar", "shape(config['weight_f'])[0] == config['lanes']",
         mgu_node(lanes=4)),
        ("mgu_bipolar", "shape(config['weight_n'])[0] == config['lanes']",
         mgu_node(n_hidden=4)),
        ("mgu_bipolar", "shape(config['weight_n'])[1] == shape(config['weight_f'])[1]",
         mgu_node(n_features=8)),
        # A gate width equal to the hidden size leaves no input features.
        ("mgu_bipolar", "IN_SIZE >= 1", mgu_node(in_size=0)),
        ("mgu_bipolar",
         "2 ** (WIDTH - 1) > LANES + IN_SIZE + (1 if HAS_BIAS_F or HAS_BIAS_N else 0)",
         mgu_node(width=3)),
        ("mgu_bipolar", "WIDTH <= 30", mgu_node(width=31)),
        # timestep 256 gives SEQ_WIDTH 8, so depth_ismul 8 is a run that cannot
        # outlast the multiplier shift register.
        ("mgu_bipolar", "SEQ_WIDTH > SR_WIDTH", mgu_node(depth_ismul=8)),
    ]
    for polarity in ("unipolar", "bipolar"):
        # WIDTH 31 overflows the 32-bit constants on its own; WIDTH 30 with a
        # reduction of 400 million entries overflows only through the ENTRY term.
        cases += [
            (f"add_any_{polarity}", "WIDTH <= 30", add_any_node(polarity, 31, 16)),
            (f"add_any_{polarity}", "2 ** WIDTH + 3 * ENTRY + 1 <= 2 ** 31 - 1",
             add_any_node(polarity, 30, 400_000_000)),
            (f"linear_{polarity}", "shape(config['weight'])[0] == config['lanes']",
             linear_node("linear_mix", polarity, lanes=4)),
            (f"linear_{polarity}", "2 ** (WIDTH - 1) > IN_FEATURES + HAS_BIAS",
             linear_node("linear_mix", polarity, width=4)),
            (f"linear_gaines_{polarity}", "shape(config['weight'])[0] == config['lanes']",
             gaines_node("linear_gaines", polarity, lanes=4)),
            (f"linear_ugemm_{polarity}", "shape(config['weight'])[0] == config['lanes']",
             linear_node("linear_ugemm", polarity, lanes=4)),
            (f"linear_ugemm_{polarity}", "2 ** (WIDTH - 1) > IN_FEATURES + HAS_BIAS",
             linear_node("linear_ugemm", polarity, width=4)),
            (f"linear_ugemm_{polarity}", "WIDTH <= 30",
             linear_node("linear_ugemm", polarity, width=31)),
            (f"conv_{polarity}", "shape(config['weight'])[1] == input.size(1)",
             conv_node(polarity, in_channels=4)),
            (f"conv_{polarity}", "2 ** (WIDTH - 1) > IN_CHANNELS * KERNEL_H * KERNEL_W + HAS_BIAS",
             conv_node(polarity, width=4)),
            (f"conv_{polarity}",
             "LANES == BATCH * OUT_CHANNELS * ((IN_H + 2 * PADDING - DILATION * (KERNEL_H - 1) - 1) // STRIDE + 1) * ((IN_W + 2 * PADDING - DILATION * (KERNEL_W - 1) - 1) // STRIDE + 1)",
             conv_node(polarity, lanes=28)),
            # conv_ugemm generates its weight and bias streams in hardware, so it
            # carries conv's clauses plus the add_any width cap its lanes inherit.
            (f"conv_ugemm_{polarity}", "shape(config['weight'])[1] == input.size(1)",
             conv_node(polarity, in_channels=4, class_name="conv_ugemm")),
            (f"conv_ugemm_{polarity}",
             "2 ** (WIDTH - 1) > IN_CHANNELS * KERNEL_H * KERNEL_W + HAS_BIAS",
             conv_node(polarity, width=4, class_name="conv_ugemm")),
            (f"conv_ugemm_{polarity}",
             "LANES == BATCH * OUT_CHANNELS * ((IN_H + 2 * PADDING - DILATION * (KERNEL_H - 1) - 1) // STRIDE + 1) * ((IN_W + 2 * PADDING - DILATION * (KERNEL_W - 1) - 1) // STRIDE + 1)",
             conv_node(polarity, lanes=28, class_name="conv_ugemm")),
            (f"conv_ugemm_{polarity}", "WIDTH <= 30",
             conv_node(polarity, width=31, class_name="conv_ugemm")),
        ]
        if polarity == "unipolar":
            cases += [
                ("linear_gaines_unipolar", "SCALED == 0 or IN_FEATURES + HAS_BIAS >= 2",
                 gaines_node("linear_gaines", polarity, in_features=1, has_bias=False)),
                ("linear_gaines_unipolar",
                 "SCALED == 0 or 2 ** SCALE_WIDTH == IN_FEATURES + HAS_BIAS",
                 gaines_node("linear_gaines", polarity, in_features=2)),
            ]
    return cases


def test_every_requires_clause_rejects():
    """Every requires clause in the mapping rejects a configuration that breaks it."""
    covered = set()
    for rtl_module, clause, node in rejection_cases():
        with pytest.raises(TranslationError) as raised:
            translate_node(node)
        message = str(raised.value)
        assert rtl_module in message, f"{clause}: {message}"
        assert clause in message, f"{clause}: {message}"
        covered.add((rtl_module, clause))
    assert covered == requires_clauses()


def test_reserved_name_shadows_a_config_key_of_the_same_name():
    """Keep a config key named `input` readable, and let a real input shape win."""
    entry = next(item for item in load_mapping()
                 if item.get("rtl_module") == "mul_gaines_unipolar")
    scratch = _MAPPING_PATH.parent / "_reserved_name_mapping.yaml"
    node = {"class": "mul_gaines", "config": {"polarity": "unipolar", "input": 5}}
    try:
        scratch.write_text(yaml.safe_dump([dict(entry, parameters={"WIDTH": "input"})]),
                           encoding="utf-8")
        assert translate_node(node, mapping_path=scratch).parameters == {"WIDTH": 5}

        scratch.write_text(
            yaml.safe_dump([dict(entry, parameters={"WIDTH": "input.size(-1)"})]),
            encoding="utf-8")
        shaped = dict(node, inputs={"input": {"shape": (4, 16)}})
        assert translate_node(shaped, mapping_path=scratch).parameters == {"WIDTH": 16}
    finally:
        scratch.unlink(missing_ok=True)


def test_bindings_match_rtl_module_headers():
    """Check resolved parameters and directional ports against real RTL headers."""
    nodes = module_nodes() + [
        # conv_ugemm holds its own weight and bias encoders, so its header carries
        # SEQ_WIDTH on top of conv's parameters and no pad port.
        {"class": "conv_ugemm",
         "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": torch.zeros(3),
                    "stride": 1, "padding": 1, "dilation": 1,
                    "config": LAYER_CONFIG, "lanes": 108},
         "inputs": {"input_spike": {"shape": (1, 2, 6, 6)}}},
        {
            "class": "add_any",
            "config": {"polarity": "bipolar", "scale": 2, "width": 8},
            "inputs": {"input": {"shape": (4, 16)}},
        },
        {
            "class": "add_any",
            "config": {"polarity": "unipolar", "scale": 2, "width": 8},
            "inputs": {"input": {"shape": (4, 16)}},
        },
        # mgu_hard_mix composes two linear layers, so its header carries the gate widths
        # and the multiplier sequence widths rather than a single SCALE.
        mgu_node(),
        {"class": "shiftreg", "config": {"depth": 4}},
        {"class": "div_cordiv", "config": {"depth": 8}},
        {"class": "mul_gaines", "config": {"polarity": "unipolar"}},
        {"class": "relu_sat", "config": {}},
        {"class": "sync_skewed_int", "config": {"width": 4}},
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
    test_reserved_name_shadows_a_config_key_of_the_same_name()
    test_bindings_match_rtl_module_headers()
    test_null_mapped_port_rejects_a_supplied_source()
    print("Test passed.")

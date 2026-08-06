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
        {"class": "avgpool2d", "config": {"kernel_size": 2, "lanes": 48}},
        {"class": "avgpool2d",
         "config": {"kernel_size": (2, 3), "divisor_override": 8, "lanes": 48}},
        {"class": "linear_ugemm",
         "config": {"weight": weight, "bias": torch.zeros(8),
                    "config": layer_config, "lanes": 8}},
        {"class": "linear_ugemm",
         "config": {"weight": weight, "bias": None,
                    "config": dict(layer_config, polarity='bipolar'), "lanes": 8}},
        {"class": "linear",
         "config": {"weight": weight, "bias": torch.zeros(8),
                    "config": layer_config, "lanes": 8}},
        {"class": "linear",
         "config": {"weight": weight, "bias": None,
                    "config": dict(layer_config, polarity='bipolar'), "lanes": 8}},
        # conv sizes itself from the weight and the input shape, so its node
        # carries the NCHW input the geometry is derived from.
        {"class": "conv",
         "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": torch.zeros(3),
                    "stride": 1, "padding": 1, "dilation": 1,
                    "config": layer_config, "lanes": 108},
         "inputs": {"input_spike": {"shape": (1, 2, 6, 6)}}},
        {"class": "conv",
         "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": None,
                    "stride": 2, "padding": 2, "dilation": 2,
                    "config": dict(layer_config, polarity='bipolar'), "lanes": 27},
         "inputs": {"input_spike": {"shape": (1, 2, 6, 6)}}},
    ]


def test_module_layer_parameters():
    """Resolve every module entry from constructor-name node configs."""
    (pool, pool_d8, ugemm, ugemm_nb, linear, linear_nb,
     convolution, convolution_nb) = [translate_node(node) for node in module_nodes()]

    assert pool.rtl_module == "avgpool2d"
    assert Path(pool.file_path).parts[-3:] == ("avgpool2d", "rtl", "avgpool2d.v")
    assert pool.parameters == {"KERNEL_AREA": 4, "DIVISOR": 4, "LANES": 48}
    # A tuple kernel keeps the window area, and divisor_override wins over it.
    assert pool_d8.parameters == {"KERNEL_AREA": 6, "DIVISOR": 8, "LANES": 48}

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
    assert convolution.port_map["inputs"]["pad_bits"] == "i_pad_bits"
    assert convolution.port_map["output"] == "o_out"
    # Stride and dilation shrink the output positions, and no bias shrinks the scale.
    assert convolution_nb.rtl_module == "conv_bipolar"
    assert convolution_nb.parameters["HAS_BIAS"] == 0
    assert convolution_nb.parameters["SCALE"] == 18
    assert convolution_nb.parameters["LANES"] == 27


def test_module_rejects_unsupported_configuration():
    """Reject pooling geometry the RTL does not implement."""
    with pytest.raises(TranslationError, match="padding"):
        translate_node({"class": "avgpool2d",
                        "config": {"kernel_size": 2, "padding": 1, "lanes": 48}})
    with pytest.raises(TranslationError, match="DIVISOR >= KERNEL_AREA"):
        translate_node({"class": "avgpool2d",
                        "config": {"kernel_size": 2, "divisor_override": 2, "lanes": 48}})
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
    """One linear or linear_ugemm node over an 8x16 weight."""
    return {"class": class_name,
            "config": {"weight": torch.zeros(8, 16), "bias": torch.zeros(8),
                       "config": dict(LAYER_CONFIG, polarity=polarity, width=width),
                       "lanes": lanes}}


def conv_node(polarity, in_channels=2, lanes=108, width=12, class_name="conv"):
    """One conv or conv_pc node over a 3x2x3x3 weight and a 1x2x6x6 input."""
    return {"class": class_name,
            "config": {"weight": torch.zeros(3, 2, 3, 3), "bias": torch.zeros(3),
                       "stride": 1, "padding": 1, "dilation": 1,
                       "config": dict(LAYER_CONFIG, polarity=polarity, width=width),
                       "lanes": lanes},
            "inputs": {"input_spike": {"shape": (1, in_channels, 6, 6)}}}


def rejection_cases():
    """One node per (rtl_module, requires clause), each violating that clause."""
    cases = [
        ("div_cordiv", "2 ** int(WIDTH) == DEPTH",
         {"class": "div_cordiv", "config": {"depth": 6}}),
        ("avgpool2d", "get(config, 'padding', 0) in (0, (0, 0))",
         {"class": "avgpool2d", "config": {"kernel_size": 2, "padding": 1, "lanes": 48}}),
        ("avgpool2d", "get(config, 'stride') is None or get(config, 'stride') == config['kernel_size']",
         {"class": "avgpool2d", "config": {"kernel_size": 2, "stride": 3, "lanes": 48}}),
        ("avgpool2d", "get(config, 'ceil_mode', False) == False",
         {"class": "avgpool2d", "config": {"kernel_size": 2, "ceil_mode": True, "lanes": 48}}),
        ("avgpool2d", "get(config, 'count_include_pad', True) == True",
         {"class": "avgpool2d",
          "config": {"kernel_size": 2, "count_include_pad": False, "lanes": 48}}),
        ("avgpool2d", "DIVISOR >= KERNEL_AREA",
         {"class": "avgpool2d",
          "config": {"kernel_size": 2, "divisor_override": 2, "lanes": 48}}),
    ]
    for polarity in ("unipolar", "bipolar"):
        # WIDTH 31 overflows the 32-bit constants on its own; WIDTH 30 with a
        # reduction of 400 million entries overflows only through the ENTRY term.
        cases += [
            (f"add_any_{polarity}", "WIDTH <= 30", add_any_node(polarity, 31, 16)),
            (f"add_any_{polarity}", "2 ** WIDTH + 3 * ENTRY + 1 <= 2 ** 31 - 1",
             add_any_node(polarity, 30, 400_000_000)),
            (f"linear_{polarity}", "shape(config['weight'])[0] == config['lanes']",
             linear_node("linear", polarity, lanes=4)),
            (f"linear_{polarity}", "2 ** (WIDTH - 1) > IN_FEATURES + HAS_BIAS",
             linear_node("linear", polarity, width=4)),
            # linear_pc drops the accumulator, so it carries the lane clause but
            # no accumulator-width one.
            (f"linear_pc_{polarity}", "shape(config['weight'])[0] == config['lanes']",
             linear_node("linear_pc", polarity, lanes=4)),
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
            # conv_pc drops the accumulator, so it carries the geometry clauses
            # but no accumulator-width one.
            (f"conv_pc_{polarity}", "shape(config['weight'])[1] == input.size(1)",
             conv_node(polarity, in_channels=4, class_name="conv_pc")),
            (f"conv_pc_{polarity}",
             "LANES == BATCH * OUT_CHANNELS * ((IN_H + 2 * PADDING - DILATION * (KERNEL_H - 1) - 1) // STRIDE + 1) * ((IN_W + 2 * PADDING - DILATION * (KERNEL_W - 1) - 1) // STRIDE + 1)",
             conv_node(polarity, lanes=28, class_name="conv_pc")),
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
    print("Test passed.")

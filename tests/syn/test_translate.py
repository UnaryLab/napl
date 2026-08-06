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


def module_nodes():
    """Nodes for both module entries, built as RULE_IMP documents them.

    A module node carries the class's ``__init__`` arguments under their own
    names, so a `config=` argument nests, plus the caller-supplied `lanes`.
    """
    weight = torch.zeros(8, 16)
    layer_config = {'polarity': 'unipolar', 'timestep': 256, 'generator': 'sobol',
                    'dim': 1, 'scale': None, 'width': 12}
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
    ]


def test_module_layer_parameters():
    """Resolve both module entries from constructor-name node configs."""
    pool, pool_d8, linear, linear_nb = [translate_node(node) for node in module_nodes()]

    assert pool.rtl_module == "avgpool2d"
    assert Path(pool.file_path).parts[-3:] == ("avgpool2d", "rtl", "avgpool2d.v")
    assert pool.parameters == {"KERNEL_AREA": 4, "DIVISOR": 4, "LANES": 48}
    # A tuple kernel keeps the window area, and divisor_override wins over it.
    assert pool_d8.parameters == {"KERNEL_AREA": 6, "DIVISOR": 8, "LANES": 48}

    assert linear.rtl_module == "linear_ugemm_unipolar"
    assert linear.parameters == {"IN_FEATURES": 16, "LANES": 8, "SEQ_WIDTH": 8,
                                 "WIDTH": 12, "HAS_BIAS": 1, "SCALE": 17}
    assert linear.port_map["inputs"]["weight"] == "i_weight"
    assert linear.port_map["output"] == "o_out"
    # No bias drops an addend, so the default scale shrinks with it.
    assert linear_nb.rtl_module == "linear_ugemm_bipolar"
    assert linear_nb.parameters["HAS_BIAS"] == 0
    assert linear_nb.parameters["SCALE"] == 16


def test_module_rejects_unsupported_configuration():
    """Reject pooling geometry the RTL does not implement."""
    with pytest.raises(TranslationError, match="padding"):
        translate_node({"class": "avgpool2d",
                        "config": {"kernel_size": 2, "padding": 1, "lanes": 48}})
    with pytest.raises(TranslationError, match="DIVISOR >= KERNEL_AREA"):
        translate_node({"class": "avgpool2d",
                        "config": {"kernel_size": 2, "divisor_override": 2, "lanes": 48}})


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


def test_bindings_match_rtl_module_headers():
    """Check resolved parameters and directional ports against real RTL headers."""
    nodes = module_nodes() + [
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
    test_bindings_match_rtl_module_headers()
    print("Test passed.")

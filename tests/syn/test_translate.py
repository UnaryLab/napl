import re
from pathlib import Path

import pytest

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


def test_bindings_match_rtl_module_headers():
    """Check resolved parameters and directional ports against real RTL headers."""
    nodes = [
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
    test_bindings_match_rtl_module_headers()
    print("Test passed.")

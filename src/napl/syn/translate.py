"""Translate NAPL computation-graph nodes into RTL bindings."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_MAPPING_PATH = Path(__file__).resolve().parents[1] / "imp" / "mapping.yaml"

#: Mapping-entry layers, each the `imp/` subdirectory holding that RTL.
_LAYERS = {"operation", "module"}


class TranslationError(ValueError):
    """Raised when a graph node cannot be mapped to an RTL operation."""


class _Shape:
    """Small shape object used by the restricted parameter evaluator."""

    def __init__(self, shape):
        self.shape = tuple(int(value) for value in shape)

    def size(self, dim=-1):
        if dim is None:
            dim = -1
        if not isinstance(dim, int):
            raise TypeError(f"dimension must be an integer, got {dim!r}")
        try:
            return self.shape[dim]
        except IndexError as exc:
            raise IndexError(
                f"dimension {dim} is out of range for input shape {self.shape}"
            ) from exc


class _Count:
    """Represent a scalar segment count for ``len(SEGMENT)`` expressions."""

    def __init__(self, count):
        self.count = int(count)

    def __len__(self):
        return self.count


class PortMap(dict):
    """Port mappings grouped by input and output direction."""

    def __init__(self, inputs=None, outputs=None):
        super().__init__(inputs=dict(inputs or {}), outputs=dict(outputs or {}))

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return any(key in ports for ports in (self["inputs"], self["outputs"]))

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        for ports in (super().__getitem__("inputs"), super().__getitem__("outputs")):
            if key in ports:
                return ports[key]
        raise KeyError(key)


@dataclass(frozen=True)
class RtlBinding(Mapping):
    """Resolved RTL information for one computation-graph node."""

    rtl_module: str
    file_path: Path
    parameters: dict[str, Any]
    port_map: PortMap
    node_name: str | None = None

    @property
    def rtl_file(self):
        """Return the resolved RTL file path."""
        return self.file_path

    @property
    def rtl_path(self):
        """Return the resolved RTL file path."""
        return self.file_path

    @property
    def param_values(self):
        """Return the resolved Verilog parameter values."""
        return self.parameters

    @property
    def ports(self):
        """Return the resolved input and output port maps."""
        return self.port_map

    @property
    def name(self):
        """Return the graph node name, when one was supplied."""
        return self.node_name

    def __getitem__(self, key):
        aliases = {
            "rtl_module": "rtl_module",
            "file_path": "file_path",
            "rtl_file": "file_path",
            "rtl_path": "file_path",
            "parameters": "parameters",
            "param_values": "parameters",
            "port_map": "port_map",
            "ports": "port_map",
            "name": "node_name",
            "node_name": "node_name",
        }
        try:
            return getattr(self, aliases[key])
        except KeyError as exc:
            raise KeyError(key) from exc

    def __iter__(self):
        return iter(("rtl_module", "file_path", "parameters", "port_map"))

    def __len__(self):
        return 4

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self):
        """Return the binding in a serializable dictionary form."""
        result = {
            "rtl_module": self.rtl_module,
            "file_path": str(self.file_path),
            "rtl_file": str(self.file_path),
            "parameters": dict(self.parameters),
            "port_map": {
                "inputs": dict(self.port_map["inputs"]),
                "outputs": dict(self.port_map["outputs"]),
            },
        }
        if self.node_name is not None:
            result["name"] = self.node_name
        return result


def load_mapping(mapping_path=None):
    """Load the simulation-to-RTL mapping table."""
    path = Path(mapping_path) if mapping_path is not None else _MAPPING_PATH
    try:
        with path.open(encoding="utf-8") as mapping_file:
            mapping = yaml.safe_load(mapping_file)
    except OSError as exc:
        raise TranslationError(f"Unable to read RTL mapping {path}: {exc}") from exc
    if not isinstance(mapping, list):
        raise TranslationError(f"RTL mapping {path} must contain a list of entries")
    return mapping


def _name_from_sim_module(sim_module):
    text = str(sim_module).replace("\\", "/").rstrip("/")
    text = text.rsplit("/", 1)[-1]
    if text.endswith(".py"):
        text = text[:-3]
    return text.rsplit(".", 1)[-1]


def _node_class_name(value):
    if isinstance(value, type):
        return value.__name__
    if not isinstance(value, str) or not value.strip():
        raise TranslationError("Graph node class must be a non-empty string")
    return _name_from_sim_module(value.strip())


def _entries_for_class(mapping, class_name):
    entries = [
        entry for entry in mapping
        if isinstance(entry, Mapping)
        and _name_from_sim_module(entry.get("sim_module", "")) == class_name
    ]
    if entries:
        return entries
    return [
        entry for entry in mapping
        if isinstance(entry, Mapping) and entry.get("rtl_module") == class_name
    ]


def _select_entry(mapping, class_name, config, requested_rtl=None):
    entries = _entries_for_class(mapping, class_name)
    if not entries:
        raise TranslationError(
            f"No RTL mapping for simulation class {class_name!r}"
        )

    requested_rtl = requested_rtl or config.get("rtl_module")
    if requested_rtl is not None:
        matches = [entry for entry in entries if entry.get("rtl_module") == requested_rtl]
        if not matches:
            raise TranslationError(
                f"No RTL variant {requested_rtl!r} for simulation class {class_name!r}"
            )
        return matches[0]

    polarity = config.get("polarity")
    if polarity is None and isinstance(config.get("config"), Mapping):
        polarity = config["config"].get("polarity")
    if polarity is not None:
        expected = f"{class_name}_{polarity}"
        matches = [entry for entry in entries if entry.get("rtl_module") == expected]
        if matches:
            return matches[0]

    exact = [entry for entry in entries if entry.get("rtl_module") == class_name]
    if len(exact) == 1:
        return exact[0]
    if len(entries) == 1:
        return entries[0]
    variants = ", ".join(str(entry.get("rtl_module")) for entry in entries)
    if polarity is None:
        raise TranslationError(
            f"Simulation class {class_name!r} needs config['polarity']; available RTL variants: {variants}"
        )
    raise TranslationError(
        f"No RTL variant for simulation class {class_name!r} and polarity {polarity!r}; available RTL variants: {variants}"
    )


def _shape_from_value(value):
    if isinstance(value, _Shape):
        return value.shape
    if isinstance(value, Mapping):
        for key in ("shape", "size", "input_shape"):
            if key in value:
                shape = _shape_from_value(value[key])
                if shape is not None:
                    return shape
        return None
    shape = getattr(value, "shape", None)
    if shape is not None and not callable(shape):
        try:
            return tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            pass
    size = getattr(value, "size", None)
    if callable(size):
        try:
            result = size()
            return tuple(int(item) for item in result)
        except (TypeError, ValueError):
            pass
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return (0,)
        if all(isinstance(item, int) for item in value):
            return tuple(int(item) for item in value)
        return (len(value),)
    return None


def _input_descriptor(node):
    inputs = node.get("inputs")
    if isinstance(inputs, Mapping):
        if "input" in inputs:
            return inputs["input"]
        for key, value in inputs.items():
            if key not in {"dim", "entry", "segment"} and value is not None:
                return value
        return inputs
    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        return inputs[0] if inputs else None
    return inputs


def _make_eval_context(node, config, entry):
    shape = _shape_from_value(_input_descriptor(node))
    if shape is None:
        for key in ("input_shape", "shape", "input_size"):
            if key in node:
                shape = _shape_from_value(node[key])
                if shape is not None:
                    break
    if shape is None:
        for key in ("input_shape", "shape", "input_size"):
            if key in config:
                shape = _shape_from_value(config[key])
                if shape is not None:
                    break
    if shape is None and "entry" in config:
        shape = _shape_from_value(config["entry"])

    dim = node.get("dim", config.get("dim", -1))
    if dim is None:
        dim = -1

    segment = node.get("segment", config.get("segment"))
    if segment is None:
        segment = config.get("timestep", [])
    if isinstance(segment, int):
        segment = _Count(segment)

    context = dict(config)
    # The reserved names are set after the config keys, so no config key can
    # shadow one of them. A module node carries the class's own `config` mapping
    # under an __init__ argument of that name, so the node config keeps `config`.
    context.update({
        "config": config,
        "input": _Shape(shape) if shape is not None else None,
        "dim": dim,
        "SEGMENT": segment,
    })
    if context["input"] is None:
        # With no shape to bind, the reserved name has nothing to contribute, so
        # a config key of that name stays readable instead of being dropped.
        if "input" in config:
            context["input"] = config["input"]
        else:
            context.pop("input")
    return context


def _safe_len(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return len(value)


def _safe_get(container, key, default=None):
    """Return ``container[key]``, or ``default`` when the key is absent."""
    if not isinstance(container, Mapping):
        raise TypeError(f"get() expects a mapping, got {type(container).__name__}")
    return container.get(key, default)


def _safe_shape(value):
    """Return the shape tuple of a tensor, sequence, or shape-like value."""
    shape = _shape_from_value(value)
    if shape is None:
        raise TypeError(f"value {value!r} has no shape")
    return shape


def _safe_area(value):
    """Element count of a window size given as an int or a per-axis sequence."""
    if isinstance(value, bool):
        raise TypeError("window size must not be a boolean")
    if isinstance(value, int):
        return value * value
    return math.prod(int(item) for item in value)


class _RestrictedEvaluator:
    """Evaluate the small expression language used by mapping.yaml."""

    _binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
    _compare = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda left, right: left in right,
        ast.NotIn: lambda left, right: left not in right,
    }
    _functions = {
        "ceil": math.ceil,
        "floor": math.floor,
        "log2": math.log2,
        "len": _safe_len,
        "int": int,
        "get": _safe_get,
        "shape": _safe_shape,
        "area": _safe_area,
    }

    def __init__(self, names):
        self.names = names

    def evaluate(self, expression):
        try:
            tree = ast.parse(str(expression), mode="eval")
            value = self._visit(tree.body)
        except (
            ArithmeticError,
            IndexError,
            KeyError,
            NameError,
            TypeError,
            ValueError,
            SyntaxError,
        ) as exc:
            raise TranslationError(
                f"Unsupported or invalid parameter expression {expression!r}: {exc}"
            ) from exc
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _visit(self, node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool, type(None))):
                return node.value
            raise TypeError(f"constant {node.value!r} is not allowed")
        if isinstance(node, ast.Name):
            if node.id not in self.names:
                raise NameError(f"name {node.id!r} is not allowed")
            return self.names[node.id]
        if isinstance(node, ast.List):
            return [self._visit(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._visit(item) for item in node.elts)
        if isinstance(node, ast.Subscript):
            value = self._visit(node.value)
            key = self._visit(node.slice)
            return value[key]
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary:
            return self._binary[type(node.op)](self._visit(node.left), self._visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary:
            return self._unary[type(node.op)](self._visit(node.operand))
        if isinstance(node, ast.Compare):
            left = self._visit(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                if type(op) not in self._compare:
                    raise TypeError(f"comparison {type(op).__name__} is not allowed")
                right = self._visit(comparator)
                if not self._compare[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            values = [self._visit(value) for value in node.values]
            if isinstance(node.op, ast.And):
                result = True
                for value in values:
                    result = value
                    if not value:
                        break
                return result
            result = False
            for value in values:
                result = value
                if value:
                    break
            return result
        if isinstance(node, ast.IfExp):
            if self._visit(node.test):
                return self._visit(node.body)
            return self._visit(node.orelse)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in self._functions:
                if node.keywords:
                    raise TypeError("keyword arguments are not allowed")
                return self._functions[node.func.id](*[self._visit(arg) for arg in node.args])
            if isinstance(node.func, ast.Attribute) and node.func.attr == "size":
                target = self._visit(node.func.value)
                if not isinstance(target, _Shape):
                    raise TypeError("only input.size(...) is allowed")
                if node.keywords or len(node.args) > 1:
                    raise TypeError("input.size accepts at most one positional argument")
                dim = self._visit(node.args[0]) if node.args else -1
                return target.size(dim)
            raise TypeError("function is not allowed")
        raise TypeError(f"syntax {type(node).__name__} is not allowed")


def _resolve_parameters(entry, node, config):
    parameters = {}
    names = _make_eval_context(node, config, entry)
    evaluator = _RestrictedEvaluator(names)
    for name, expression in (entry.get("parameters") or {}).items():
        try:
            value = evaluator.evaluate(expression)
        except TranslationError as exc:
            raise TranslationError(
                f"Could not resolve parameter {name!r} for RTL module {entry.get('rtl_module')!r}: {exc}"
            ) from exc
        parameters[name] = value
        evaluator.names[name] = value
    _check_requires(entry, evaluator)
    return parameters


def _check_requires(entry, evaluator):
    """Reject a configuration the RTL module does not implement."""
    conditions = entry.get("requires") or []
    if isinstance(conditions, str):
        conditions = [conditions]
    for condition in conditions:
        if not evaluator.evaluate(condition):
            raise TranslationError(
                f"RTL module {entry.get('rtl_module')!r} does not support this "
                f"configuration: {condition} is false"
            )


def _resolve_rtl_file(mapping_path, entry):
    root = Path(mapping_path).parent
    rtl_module = entry.get("rtl_module")
    if not isinstance(rtl_module, str) or not rtl_module:
        raise TranslationError("RTL mapping entry has no rtl_module")
    layer = entry.get("layer") or "operation"
    if layer not in _LAYERS:
        raise TranslationError(
            f"RTL mapping entry {rtl_module!r} has layer {entry.get('layer')!r}; "
            f"'layer' must be one of {sorted(_LAYERS)}"
        )
    candidates = sorted(root.glob(f"{layer}/*/rtl/{rtl_module}.v"))
    if len(candidates) == 1:
        return candidates[0]
    sim_name = _name_from_sim_module(entry.get("sim_module", ""))
    direct = root / layer / sim_name / "rtl" / f"{rtl_module}.v"
    if direct.is_file():
        return direct
    if not candidates:
        raise TranslationError(
            f"RTL file for module {rtl_module!r} is not present under {root / layer}"
        )
    raise TranslationError(
        f"RTL module {rtl_module!r} has multiple candidate files: {candidates}"
    )


def _translate_node(node, mapping, mapping_path):
    if not isinstance(node, Mapping):
        raise TranslationError("Each graph node must be a mapping")
    if "class" not in node:
        raise TranslationError("Graph node is missing its class")
    class_name = _node_class_name(node["class"])
    config = node.get("config") or {}
    if not isinstance(config, Mapping):
        raise TranslationError(f"Configuration for class {class_name!r} must be a mapping")
    config = dict(config)
    entry = _select_entry(mapping, class_name, config, node.get("rtl_module"))
    file_path = _resolve_rtl_file(mapping_path, entry)
    parameters = _resolve_parameters(entry, node, config)
    port_map = PortMap(entry.get("inputs"), entry.get("outputs"))
    return RtlBinding(
        rtl_module=entry["rtl_module"],
        file_path=file_path,
        parameters=parameters,
        port_map=port_map,
        node_name=node.get("name"),
    )


def translate_node(node, mapping_path=None):
    """Translate one computation-graph node into an :class:`RtlBinding`."""
    path = Path(mapping_path) if mapping_path is not None else _MAPPING_PATH
    mapping = load_mapping(path)
    return _translate_node(node, mapping, path)


def translate_graph(nodes, mapping_path=None):
    """Translate graph nodes in input order."""
    if isinstance(nodes, Mapping):
        nodes = nodes.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        raise TranslationError("A computation graph must provide a sequence of nodes")
    path = Path(mapping_path) if mapping_path is not None else _MAPPING_PATH
    mapping = load_mapping(path)
    return [_translate_node(node, mapping, path) for node in nodes]


def translate(graph_or_node, mapping_path=None):
    """Translate either one node or a graph of nodes."""
    if isinstance(graph_or_node, Mapping) and "class" in graph_or_node:
        return translate_node(graph_or_node, mapping_path)
    return translate_graph(graph_or_node, mapping_path)


__all__ = [
    "PortMap",
    "RtlBinding",
    "TranslationError",
    "load_mapping",
    "translate",
    "translate_graph",
    "translate_node",
]

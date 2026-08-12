"""Require the sweep to cover every unit mapping.yaml registers.

`make sweep` discovers its targets by globbing `operation/*/rtl` and
`module/*/rtl`, so a deleted or renamed unit folder leaves the glob and the
sweep still reports every remaining target green. `mapping.yaml` is the
authoritative registry of implemented units, so this floor derives the expected
set from it and rejects a discovered set that is missing a unit, empty, or a
different size. A folder that exists but no longer holds the registered
`rtl_module`'s `.v` file is rejected too, so an emptied directory fails here
instead of passing as present.

It also rejects a `.v` basename that appears in more than one rtl directory,
since `-y` resolves an instantiated name to the first match along its path.

Run as `python sweep_floor.py "<operations>" "<modules>"`, each argument the
space-separated list the Makefile discovered for that layer.
"""

import pathlib
import sys

import yaml

MAPPING = pathlib.Path(__file__).with_name("mapping.yaml")
ROOT = MAPPING.parent
LAYERS = ("operation", "module")


def _candidates(entry):
    """Unit folder names an entry accepts: the simulation class name."""
    return [pathlib.Path(entry["sim_module"]).stem]


def expected_units():
    """Map each layer to the (rtl_module, accepted folder names) it registers."""
    units = {layer: [] for layer in LAYERS}
    for entry in yaml.safe_load(MAPPING.read_text()):
        layer = entry.get("layer") or "operation"
        if layer not in units:
            sys.exit(f"*** SWEEP FLOOR: {entry['rtl_module']} has unknown layer {layer!r}")
        units[layer].append((entry["rtl_module"], _candidates(entry)))
    return units


def rtl_modules(layer, discovered):
    """Verilog module names the discovered folders of one layer actually hold."""
    return {
        path.stem
        for name in discovered
        for path in (ROOT / layer / name / "rtl").glob("*.v")
    }


def basename_collisions():
    """Report every .v basename that appears in more than one rtl directory.

    A module compiles with every `operation/*/rtl` and `module/*/rtl` directory
    passed to `iverilog` as a `-y` library, and `-y` resolves an instantiated
    name to the first `<name>.v` it finds along that path, so two files sharing
    a basename link one module where the other was meant instead of failing at
    elaboration. Testbenches live in `<unit>/tb/`, outside the `-y` path, so
    globbing the rtl directories alone is the set `-y` can resolve.
    """
    seen = {}
    problems = []
    for layer in LAYERS:
        for path in sorted((ROOT / layer).glob("*/rtl/*.v")):
            first = seen.setdefault(path.stem, path)
            if first != path:
                problems.append(
                    f"module basename '{path.stem}' is not unique across the -y "
                    f"search path: {first.relative_to(ROOT)} and {path.relative_to(ROOT)}"
                )
    return problems


def check(discovered_lists):
    """Report every registered unit the sweep would skip, and every extra unit."""
    units = expected_units()
    problems = basename_collisions()
    for layer, discovered_arg in zip(LAYERS, discovered_lists):
        discovered = set(discovered_arg.split())
        if not discovered:
            problems.append(f"the {layer} target list is empty")
        present = rtl_modules(layer, discovered)
        for name in sorted(discovered):
            if not any((ROOT / layer / name / "rtl").glob("*.v")):
                problems.append(f"{layer} unit '{name}' has an rtl directory holding no .v file")
        covered = set()
        for rtl_module, candidates in units[layer]:
            if rtl_module not in present:
                problems.append(
                    f"{layer} module '{rtl_module}' is registered in mapping.yaml "
                    f"but no {layer}/*/rtl/{rtl_module}.v file was found"
                )
            hit = next((name for name in candidates if name in discovered), None)
            if hit is None:
                problems.append(
                    f"{layer} unit '{candidates[0]}' is registered in mapping.yaml "
                    f"but the sweep discovered no {layer}/{candidates[0]}/rtl directory"
                )
            else:
                covered.add(hit)
        for name in sorted(discovered - covered):
            problems.append(
                f"{layer} unit '{name}' has an rtl directory but no mapping.yaml entry"
            )
    # Polarity variants share a folder, so one missing folder raises one problem.
    return list(dict.fromkeys(problems))


def main(argv):
    if len(argv) != len(LAYERS):
        sys.exit(f"usage: sweep_floor.py {' '.join(f'<{layer}s>' for layer in LAYERS)}")
    problems = check(argv)
    if problems:
        print("*** SWEEP FLOOR FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    counts = ", ".join(
        f"{len(set(arg.split()))} {layer}s" for arg, layer in zip(argv, LAYERS)
    )
    print(f"sweep floor: {counts} match mapping.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

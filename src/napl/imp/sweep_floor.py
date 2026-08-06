"""Require the sweep to cover every unit mapping.yaml registers.

`make sweep` discovers its targets by globbing `operation/*/rtl` and
`module/*/rtl`, so a deleted or renamed unit folder leaves the glob and the
sweep still reports every remaining target green. `mapping.yaml` is the
authoritative registry of implemented units, so this floor derives the expected
set from it and rejects a discovered set that is missing a unit, empty, or a
different size. A folder that exists but no longer holds the registered
`rtl_module`'s `.v` file is rejected too, so an emptied directory fails here
instead of passing as present.

It also warns when `core.hooksPath` is not `.githooks`, since the pre-commit
gate is bypassed whenever it is not installed.

Run as `python sweep_floor.py "<operations>" "<modules>"`, each argument the
space-separated list the Makefile discovered for that layer.
"""

import pathlib
import subprocess
import sys

import yaml

MAPPING = pathlib.Path(__file__).with_name("mapping.yaml")
ROOT = MAPPING.parent
LAYERS = ("operation", "module")
HOOKS_PATH = ".githooks"


def _candidates(entry):
    """Unit folder names an entry accepts, most specific first.

    The folder is the simulation class name. `linear_gaines` is the documented
    exception: one RTL base name serves `linear_gaines1` and `linear_gaines2`,
    so the polarity-stripped RTL module name is accepted as a fallback.
    """
    names = [pathlib.Path(entry["sim_module"]).stem]
    rtl_module = entry["rtl_module"]
    for suffix in ("_unipolar", "_bipolar"):
        if rtl_module.endswith(suffix):
            rtl_module = rtl_module[: -len(suffix)]
            break
    if rtl_module not in names:
        names.append(rtl_module)
    return names


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


def hooks_warning():
    """Warn when the pre-commit gate is not installed, so a commit can bypass it."""
    installed = subprocess.run(
        ["git", "config", "core.hooksPath"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if installed == HOOKS_PATH:
        return None
    return (
        f"core.hooksPath is {installed or '<unset>'}, not {HOOKS_PATH}: the pre-commit "
        "sweep gate is not installed. Run `make install-hooks`."
    )


def check(discovered_lists):
    """Report every registered unit the sweep would skip, and every extra unit."""
    units = expected_units()
    problems = []
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
    warning = hooks_warning()
    if warning:
        print(f"*** SWEEP FLOOR WARNING: {warning}")
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

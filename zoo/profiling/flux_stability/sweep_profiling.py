"""Sweep layer-specific profile_<name>.py files and write profiling_results.yaml."""

import argparse
import glob
import importlib
import os
import sys
import time
from importlib.resources import files

import yaml

import napl.sim.module as napl_module
import napl.sim.operation as napl_operation

# This directory holds the shared runner and the per-name profile modules.
_HERE = os.path.dirname(os.path.abspath(__file__))

# Fields copied verbatim from each result dict into the yaml (some are optional).
_META_KEYS = ('timesteps', 'shape', 'seed', 'n_inputs', 'n_outputs', 'model')

# Per-layer taxonomy source: layer name -> (package, yaml file, membership module).
_TAXONOMIES = (
    ('operation', 'napl.sim.operation', 'operation_type_mapping.yaml', napl_operation),
    ('module', 'napl.sim.module', 'module_type_mapping.yaml', napl_module),
)
_LAYERS = tuple(layer for layer, _package, _filename, _member in _TAXONOMIES)


def select_layers(selector=None):
    """Return the selected profiling layers, defaulting to both layers."""
    if selector is None:
        return _LAYERS
    if selector not in _LAYERS:
        raise ValueError(f'unknown layer {selector!r}; choose operation or module')
    return (selector,)


def selected_taxonomies(selector=None):
    """Return taxonomy records for the selected profiling layers."""
    layers = select_layers(selector)
    return tuple(record for record in _TAXONOMIES if record[0] in layers)


def load_taxonomies(selector=None):
    """Return name -> (layer, group) for the selected profiling layers."""
    name_map = {}
    for layer, package, filename, _member in selected_taxonomies(selector):
        groups = yaml.safe_load(files(package).joinpath(filename).read_text())
        for group, names in groups.items():
            for name in names:
                name_map[name] = (layer, group)
    return name_map


def load_layer_names(selector=None):
    """Return selected layer -> set of taxonomy names for validation."""
    per_layer = {}
    for layer, package, filename, _member in selected_taxonomies(selector):
        groups = yaml.safe_load(files(package).joinpath(filename).read_text())
        per_layer[layer] = {name for names in groups.values() for name in names}
    return per_layer


def discover_names(selector=None):
    """Return selected layer -> sorted names from its profile directory."""
    return {
        layer: sorted(
            os.path.basename(path)[len('profile_'):-len('.py')]
            for path in glob.glob(os.path.join(_HERE, layer, 'profile_*.py'))
        )
        for layer in select_layers(selector)
    }


def validate(name_map, discovered, selector=None):
    """Validate selected profiles against their layer taxonomy and package members."""
    discovered_layers = {}
    for layer in select_layers(selector):
        for name in discovered[layer]:
            previous_layer = discovered_layers.setdefault(name, layer)
            if previous_layer != layer:
                raise ValueError(
                    f'profile_{name}.py is in both {previous_layer} and {layer}; '
                    'it must appear in exactly one selected layer')
    per_layer = load_layer_names(selector)
    for layer, _package, filename, member in selected_taxonomies(selector):
        names = discovered[layer]
        unknown = sorted(name for name in names if name not in per_layer[layer])
        if unknown:
            raise ValueError(
                f'{layer}/profile_<name>.py has names not listed in {filename}: ' +
                ', '.join(unknown))
        discovered_set = set(names)
        unknown = sorted(
            name for name in per_layer[layer]
            if name not in discovered_set or name not in name_map or not hasattr(member, name))
        if unknown:
            raise ValueError(
                f'{filename} lists names that do not exist '
                f'(no {layer}/profile_<name>.py or not a napl.sim.{layer} member): ' +
                ', '.join(unknown))


def sweep(selector=None, discovered=None):
    """Profile selected names, printing progress, and return (results, failures)."""
    layers = select_layers(selector)
    if discovered is None:
        discovered = discover_names(selector)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    for layer in layers:
        layer_path = os.path.join(_HERE, layer)
        if layer_path not in sys.path:
            sys.path.insert(0, layer_path)
    results, failures = {}, {}
    names = sorted(name for layer in layers for name in discovered[layer])
    for name in names:
        try:
            module = importlib.import_module('profile_' + name)
            entries = module.profile()
            results[name] = entries
            fluxes = ', '.join(
                f'{e["polarity"]}={e["flux_stability"]}' for e in entries)
            print(f'{name} -> {fluxes}')
        except Exception as exc:
            failures[name] = repr(exc)
            print(f'{name} -> FAILED: {exc!r}')
    return results, failures


def to_yaml_fields(result):
    """Build one polarity's rounded profiling fields plus present metadata."""
    fields = {
        'flux_stability': round(float(result['flux_stability']), 6),
        'rmse': round(float(result['rmse']), 6),
    }
    for key in _META_KEYS:
        if key in result:
            fields[key] = result[key]
    return fields


_PACKAGE_HEADER = (
    '# Profiling results generated by zoo/profiling/flux_stability/sweep_profiling.py.\n'
    '# Regenerate all layers: cd zoo/profiling/flux_stability && conda run -n napl python sweep_profiling.py\n'
    '# Regenerate one layer: cd zoo/profiling/flux_stability && conda run -n napl python sweep_profiling.py operation  (or module)\n'
)


def write_package_files(doc, selector=None):
    """Write selected class metrics into the corresponding package directory."""
    for layer, _package, _filename, member in selected_taxonomies(selector):
        flat = {}
        for names in doc.get(layer, {}).values():
            for name, polarities in names.items():
                flat[name] = {
                    pol: {
                        'flux_stability': fields['flux_stability'],
                        'rmse': fields['rmse'],
                    }
                    for pol, fields in polarities.items()
                }
        path = os.path.join(os.path.dirname(member.__file__), 'profiling_results.yaml')
        with open(path, 'w') as handle:
            handle.write(_PACKAGE_HEADER)
            yaml.safe_dump(flat, handle, sort_keys=True, default_flow_style=False)


def main(argv=None):
    """Run the selected layer sweep, write profiling_results.yaml, and print a summary line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layer', nargs='?', choices=_LAYERS,
                        help='layer to sweep; omitted sweeps operation and module')
    args = parser.parse_args(argv)

    name_map = load_taxonomies(args.layer)
    discovered = discover_names(args.layer)
    validate(name_map, discovered, args.layer)

    start = time.perf_counter()
    results, failures = sweep(args.layer, discovered)
    elapsed = time.perf_counter() - start

    # Nest each polarity's fields under layer -> group -> name -> polarity.
    doc, per_layer_count = {}, {'operation': 0, 'module': 0}
    for name, entries in results.items():
        layer, group = name_map[name]
        for result in entries:
            doc.setdefault(layer, {}).setdefault(group, {}).setdefault(name, {})[result['polarity']] = \
                to_yaml_fields(result)
            per_layer_count[layer] += 1
    out_path = os.path.join(_HERE, 'profiling_results.yaml')
    merged_doc = {}
    if os.path.exists(out_path):
        with open(out_path) as handle:
            merged_doc = yaml.safe_load(handle) or {}
    for layer in select_layers(args.layer):
        merged_doc[layer] = doc.get(layer, {})
    with open(out_path, 'w') as handle:
        yaml.safe_dump(merged_doc, handle, sort_keys=True, default_flow_style=False)

    # Emit the flat per-layer package files the base-ctor fill reads.
    write_package_files(doc, args.layer)

    entry_count = sum(per_layer_count.values())
    if failures:
        print('Failed names: ' + ', '.join(f'{name} ({err})' for name, err in failures.items()))
    print(f'Sweep done: {len(results)} succeeded / {len(failures)} failed, '
          f'{entry_count} (name,polarity) entries written '
          f'(operation {per_layer_count["operation"]}, module {per_layer_count["module"]}) in {elapsed:.2f}s')


if __name__ == '__main__':
    main()

import copy
import inspect

import torch

import napl
from napl.sim.base import napl_base


# A superset of the configuration keys used across the simulation classes. Each
# class reads the keys it declares and ignores the rest, so one mapping probes
# nearly every constructor.
CONFIG = {
    'polarity': 'unipolar', 'timestep': 16, 'generator': 'sobol', 'dim': 1, 'seed': 1,
    'width': 4, 'depth': 4, 'entry': 4, 'scaled': True, 'scale': 2, 'bitwidth': 8,
    'cycle': 8, 'widthi': 4, 'widthw': 4, 'quantilei': 1, 'quantilew': 1,
    'rounding': 'round', 'rngi': 'sobol', 'rngw': 'sobol', 'temporal': 'i',
    'widtht': 4, 'formati': 'fxp', 'formatw': 'fxp', 'threshold': 0.05,
    'intwidth': 3, 'fracwidth': 4, 'depth_ismul': 6, 'entry_bit': 2,
    'normstability': 0.5, 'acc': 'scaled', 'entry_cnt': 4, 'static': True,
}

WEIGHT_4D = torch.zeros(4, 2, 3, 3)
WEIGHT_2D = torch.zeros(4, 4)
VECTOR = torch.zeros(4)

# Positional arguments preceding the configuration mapping, tried in order until
# one constructs. Covers the no-argument, weight, geometry, and gate-parameter
# constructor shapes.
ARGUMENT_SHAPES = [
    [], [WEIGHT_2D], [WEIGHT_2D, None], [WEIGHT_4D], [2, 4, 3], [4, 4], [VECTOR],
    [VECTOR, VECTOR], [WEIGHT_2D, None, 1, 0, 1], [WEIGHT_4D, None, 1, 0, 1],
    [WEIGHT_2D, VECTOR, WEIGHT_2D, VECTOR, 0.0],
]

# Classes whose configuration shape the generic probe cannot reach.
SPECIAL_CASES = {
    'conv': ([WEIGHT_4D, None, 1, 0, 1],
             {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol', 'dim': 2,
              'scale': None, 'width': 12}),
    'conv_ugemm': ([WEIGHT_4D, None, 1, 0, 1],
                   {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol', 'dim': 2}),
}

# butterfly_spike takes four separate configuration mappings instead of one.
BUTTERFLY_CONFIGS = (
    {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol', 'dim': 1},
    {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol', 'dim': 2},
    {'polarity': 'bipolar', 'scale': 2, 'width': 10, 'scaled': True, 'entry': 2,
     'generator': 'sobol', 'dim': 3},
    {'polarity': 'bipolar', 'timestep': 16},
)


class recording_dict(dict):
    """A configuration mapping that records every write made to it.

    Comparing a mapping before and after construction misses a write that stores
    a value already present, so writes are recorded as they happen instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: Names of the write operations performed on this mapping.
        self.writes = []

    def __setitem__(self, key, value):
        self.writes.append(f'config[{key!r}] = {value!r}')
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self.writes.append(f'del config[{key!r}]')
        super().__delitem__(key)

    def update(self, *args, **kwargs):
        self.writes.append('config.update(...)')
        super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        self.writes.append(f'config.setdefault({key!r}, ...)')
        return super().setdefault(key, default)

    def pop(self, *args):
        self.writes.append(f'config.pop({args[0]!r})')
        return super().pop(*args)

    def clear(self):
        self.writes.append('config.clear()')
        super().clear()


def simulation_classes():
    """Every ``napl_base`` subclass exported at the package top level."""
    return sorted(
        name for name in dir(napl)
        if inspect.isclass(getattr(napl, name))
        and issubclass(getattr(napl, name), napl_base)
        and getattr(napl, name) is not napl_base
    )


def _probe(cls, arguments, config):
    """Construct ``cls`` and return the writes it made, or ``None`` when it did not construct."""
    probe = recording_dict(copy.deepcopy(config))
    try:
        cls(*arguments, probe) if arguments else cls(probe)
    except Exception:
        return None
    return probe.writes


def _probe_generic(cls):
    """Try each argument shape and polarity until one constructs."""
    for polarity in ('unipolar', 'bipolar'):
        config = dict(CONFIG, polarity=polarity)
        for arguments in ARGUMENT_SHAPES:
            writes = _probe(cls, arguments, config)
            if writes is not None:
                return writes
    return None


def test_config_not_mutated():
    """Verify no simulation class writes into the configuration mapping it is given."""
    names = simulation_classes()
    mutated = []
    unreached = []

    for name in names:
        if name == 'butterfly_spike':
            probes = [recording_dict(copy.deepcopy(config)) for config in BUTTERFLY_CONFIGS]
            napl.butterfly_spike(*probes)
            writes = [write for probe in probes for write in probe.writes]
        elif name in SPECIAL_CASES:
            arguments, config = SPECIAL_CASES[name]
            writes = _probe(getattr(napl, name), arguments, config)
        else:
            writes = _probe_generic(getattr(napl, name))

        if writes is None:
            unreached.append(name)
        elif writes:
            mutated.append(f'{name} ({"; ".join(writes)})')

    # A class the probe cannot construct is not evidence of anything, so an
    # unreachable class fails here and needs an entry in SPECIAL_CASES.
    assert not unreached, (
        f'{len(unreached)} of {len(names)} classes were not constructed, so the invariant is '
        f'unproven for them; add a probe to SPECIAL_CASES: {unreached}'
    )
    assert not mutated, (
        'constructors must treat the configuration mapping as read-only, but these wrote into '
        f'the caller dict: {mutated}'
    )

    print(f'classes enumerated: {len(names)}')
    print(f'classes probed: {len(names) - len(unreached)}')
    print('no class mutated the caller configuration.')
    print('Test passed.')


if __name__ == '__main__':
    test_config_not_mutated()

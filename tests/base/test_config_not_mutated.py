import copy
import importlib
import inspect
import re

import torch

from napl.sim.base import napl_base


# The simulation subpackages whose sorted ``__all__`` names every public class.
_SIM_SUBPACKAGES = ('base', 'operation', 'module', 'metric', 'structure', 'algorithm')


def _simulation_classes():
    """Map every exported ``napl_base`` subclass name to its class."""
    classes = {}
    for sub in _SIM_SUBPACKAGES:
        module = importlib.import_module(f'napl.sim.{sub}')
        for name in getattr(module, '__all__', ()):
            obj = getattr(module, name)
            if inspect.isclass(obj) and issubclass(obj, napl_base) and obj is not napl_base:
                classes[name] = obj
    return classes


_CLASSES = _simulation_classes()


# A superset of the configuration keys used across the simulation classes. A
# class rejects any key it does not declare, so the probe below trims this
# mapping down to what each constructor accepts. Every class accepts 'name', so
# the trimmed mapping always keeps at least that key.
CONFIG = {
    'name': 'probe', 'polarity': 'unipolar', 'timestep': 16, 'generator': 'sobol', 'dim': 1, 'seed': 1,
    'width': 4, 'depth': 4, 'entry': 4, 'scaled': True, 'scale': 2, 'bitwidth': 8,
    'cycle': 8, 'widthi': 4, 'widthw': 4, 'quantilei': 1, 'quantilew': 1,
    'rounding': 'round', 'rngi': 'sobol', 'rngw': 'sobol', 'temporal': 'i',
    'widtht': 4, 'formati': 'fxp', 'formatw': 'fxp', 'threshold': 0.05,
    'intwidth': 3, 'fracwidth': 4, 'depth_ismul': 6, 'entry_bit': 2,
    'normstability': 0.5, 'acc': 'scaled', 'entry_cnt': 4, 'static': True,
    'scale_max': 2,
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
    # avgpool2d_ugemm needs an accumulator width satisfying 2 ** (width - 1) - 1 >= 2 * kernel_area - 1.
    'avgpool2d_ugemm': ([2], {'polarity': 'bipolar', 'width': 12}),
    # conv_gaines needs a power-of-two adder entry, which this 2x2x2 kernel gives.
    'conv_gaines': ([torch.zeros(4, 2, 2, 2), None, 1, 0, 1],
                    {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol', 'dim': 2,
                     'scaled': True}),
    'conv_mix': ([WEIGHT_4D, None, 1, 0, 1],
             {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol', 'dim': 2,
              'scale': None, 'width': 12}),
    'conv_ugemm': ([WEIGHT_4D, None, 1, 0, 1],
                   {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol'}),
    # mgu_hard_mix is bipolar only, holds its hidden value as a buffer, so it needs a tensor, and
    # its run must outlast the depth_ismul multiplier shift register.
    'mgu_hard_mix': ([WEIGHT_2D, VECTOR, WEIGHT_2D, VECTOR, torch.zeros(4, 4)],
            {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol', 'depth_ismul': 3}),
}

# Building blocks of the multi-configuration probes below.
CODEC = {'polarity': 'bipolar', 'timestep': 16, 'generator': 'sobol'}
ADD_SCALED = {'polarity': 'bipolar', 'scale': 2, 'intwidth': 12, 'fracwidth': 0}
ADD_DYNAMIC = {'polarity': 'bipolar', 'scale_max': 3, 'intwidth': 12, 'fracwidth': 0}

# Classes taking several configuration mappings instead of one, keyed by the
# constructor parameter each mapping is bound to. Every mapping is recorded, so a
# write into any of them is caught.
MULTI_CONFIG_CASES = {
    'butterfly_mix': ([VECTOR, VECTOR], {'mul_config': CODEC, 'add_config': ADD_SCALED}),
    'butterfly_mix_dyn': ([VECTOR, VECTOR], {'mul_config': CODEC, 'add_config': ADD_DYNAMIC}),
    'butterfly_ugemm': ([VECTOR, VECTOR], {'mul_config': CODEC, 'add_config': ADD_SCALED}),
    'butterfly_ugemm_dyn': ([VECTOR, VECTOR], {'mul_config': CODEC, 'add_config': ADD_DYNAMIC}),
    'conv_ugemm_hub': ([WEIGHT_4D, None, 1, 0, 1],
                       {'codec_config': dict(CODEC, dim=2),
                        'core_config': {'scale': None, 'width': 12}}),
    'fft': ([8], {'mul_config': CODEC, 'add_config': ADD_SCALED}),
    'fft_dyn': ([8], {'mul_config': CODEC, 'add_config': ADD_DYNAMIC}),
    'fft_dyn_hub': ([8], {'codec_config': dict(CODEC, dim=1), 'mul_config': CODEC,
                          'add_config': ADD_DYNAMIC}),
    'fft_hub': ([8], {'codec_config': dict(CODEC, dim=1), 'mul_config': CODEC,
                      'add_config': ADD_SCALED}),
    'linear_ugemm_hub': ([WEIGHT_2D, None],
                         {'codec_config': dict(CODEC, dim=2),
                          'core_config': {'dim': 1, 'scale': None, 'width': 12}}),
    'mgu_hard_mix_hub': ([WEIGHT_2D, VECTOR, WEIGHT_2D, VECTOR, torch.zeros(4, 4)],
                         {'codec_config': dict(CODEC, dim=1),
                          'core_config': {'width': 10, 'depth_ismul': 3}}),
}


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
    """Every ``napl_base`` subclass exported by the simulation subpackages."""
    return sorted(_CLASSES)


# The key named by the rejection raised from check_config for an unaccepted key.
UNKNOWN_KEY = re.compile(r'Unknown key <([^>]+)>')


def _probe(cls, arguments, config):
    """Construct ``cls`` and return ``(writes, error)``; ``writes`` is ``None`` on failure."""
    probe = recording_dict(copy.deepcopy(config))
    try:
        cls(*arguments, config=probe)
    except Exception as error:
        return None, error
    return probe.writes, None


def _probe_multi(cls, arguments, configs):
    """Construct ``cls`` from several configuration mappings; ``None`` on failure.

    Each mapping is bound by the constructor parameter naming it, so a probe
    cannot land on a parameter that is not a configuration.
    """
    probes = {name: recording_dict(copy.deepcopy(config)) for name, config in configs.items()}
    try:
        cls(*arguments, **probes)
    except Exception:
        return None
    return [write for probe in probes.values() for write in probe.writes]


def _probe_generic(cls):
    """Try each argument shape and polarity until one constructs.

    A class rejects any configuration key it does not declare, so a rejected key
    is dropped from the candidate configuration and the shape is retried. The
    retry count is bounded by the number of keys, and a configuration that would
    become empty is abandoned so every probe carries at least one key.
    """
    for polarity in ('unipolar', 'bipolar'):
        config = dict(CONFIG, polarity=polarity)
        for arguments in ARGUMENT_SHAPES:
            for _ in range(len(CONFIG)):
                writes, error = _probe(cls, arguments, config)
                if writes is not None:
                    # A construction that rejected no key never validated the
                    # probe, so the class ran on its own default configuration
                    # and no write into it would have been recorded.
                    assert len(config) < len(CONFIG), (
                        f'{cls.__name__} constructed without rejecting any key, so the probe '
                        'never reached its configuration; add a SPECIAL_CASES entry'
                    )
                    return writes
                match = UNKNOWN_KEY.search(str(error))
                if match is None or match.group(1) not in config or len(config) == 1:
                    break
                del config[match.group(1)]
    return None


def test_config_not_mutated():
    """Verify no simulation class writes into the configuration mapping it is given."""
    names = simulation_classes()
    mutated = []
    unreached = []

    for name in names:
        if name in MULTI_CONFIG_CASES:
            arguments, configs = MULTI_CONFIG_CASES[name]
            writes = _probe_multi(_CLASSES[name], arguments, configs)
        elif name in SPECIAL_CASES:
            arguments, config = SPECIAL_CASES[name]
            writes, _ = _probe(_CLASSES[name], arguments, config)
        else:
            writes = _probe_generic(_CLASSES[name])

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

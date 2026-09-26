"""Tests for every streaming operation class declaring a mechanism from the legal vocabulary."""

import importlib
import inspect
import pathlib
import re

from napl.sim.base.base import legal_mechanism, napl_base


# An instance assignment of the field, the form the operation sources must not use:
# it hides the value from a class-level read and so from this test.
SELF_MECHANISM = re.compile(r'self\.mechanism\s*=')


def operation_classes():
    """Map every ``napl_base`` subclass exported by ``napl.sim.operation`` to its class."""
    module = importlib.import_module('napl.sim.operation')
    return {
        name: obj
        for name in getattr(module, '__all__', ())
        for obj in [getattr(module, name)]
        if inspect.isclass(obj) and issubclass(obj, napl_base) and obj is not napl_base
    }


def operation_sources():
    """Return every module file of the ``napl.sim.operation`` package."""
    directory = pathlib.Path(importlib.import_module('napl.sim.operation').__file__).parent
    return sorted(directory.glob('*.py'))


def test_mechanism_declared():
    """Verify every streaming operation class declares a mechanism drawn from legal_mechanism."""
    classes = {
        name: cls for name, cls in operation_classes().items()
        if getattr(cls, 'streaming', True)
    }
    assert classes, 'no streaming operation classes were enumerated'

    offenders = [
        f'{name} declared mechanism <{getattr(cls, "mechanism", None)!r}>'
        for name, cls in sorted(classes.items())
        if getattr(cls, 'mechanism', None) not in legal_mechanism
    ]

    assert not offenders, (
        f'{len(offenders)} of {len(classes)} streaming operation classes declare a mechanism outside '
        f'the legal vocabulary <{legal_mechanism}>: {offenders}'
    )

    print(f'streaming operation classes enumerated: {len(classes)}')
    print('every streaming operation class declares a legal mechanism.')
    print('Test passed.')


def test_mechanism_declared_on_class_body():
    """Verify no operation source assigns the mechanism on an instance."""
    sources = operation_sources()
    assert sources, 'no operation sources were enumerated'

    offenders = [
        f'{path.name}:{number}: {line.strip()}'
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        if SELF_MECHANISM.search(line)
    ]

    assert not offenders, (
        f'{len(offenders)} instance assignments of mechanism found; declare it on the class '
        f'body so a class-level read observes it: {offenders}'
    )

    print(f'operation sources scanned: {len(sources)}')
    print('no operation source assigns mechanism on an instance.')
    print('Test passed.')


if __name__ == '__main__':
    test_mechanism_declared()
    test_mechanism_declared_on_class_body()

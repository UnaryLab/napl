import ast
import contextlib
import io
import pathlib
import re
import textwrap


SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2] / 'src' / 'napl' / 'sim'
DIRECTIVE = '.. code-block:: python'
DIRECTIVE_LINE = re.compile(r'^([ \t]*)' + re.escape(DIRECTIVE) + r'\s*$')


def code_blocks(docstring):
    """Yield the dedented body of every Python code block in ``docstring``."""
    if not docstring:
        return
    lines = docstring.splitlines()
    index = 0
    while index < len(lines):
        match = DIRECTIVE_LINE.match(lines[index])
        if not match:
            index += 1
            continue
        indent = len(match.group(1))
        index += 1
        body = []
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                body.append('')
                index += 1
                continue
            if len(line) - len(line.lstrip()) <= indent:
                break
            body.append(line)
            index += 1
        code = textwrap.dedent('\n'.join(body)).strip('\n')
        if code:
            yield code


def example_units():
    """Group every documented example by the class or module that owns it.

    Each unit holds the owner's own examples, which run first and bind the names
    the member examples use, and the member examples that follow them.
    """
    units = []
    for path in sorted(SOURCE_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text())
        relative = path.relative_to(SOURCE_ROOT.parent.parent.parent)

        for node in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            owner = list(code_blocks(ast.get_docstring(node, clean=False)))
            members = [
                (member.name, code)
                for member in node.body if isinstance(member, ast.FunctionDef)
                for code in code_blocks(ast.get_docstring(member, clean=False))
            ]
            if owner or members:
                units.append((f'{relative}::{node.name}', owner, members))

        loose = [
            (node.name, code)
            for node in tree.body if isinstance(node, ast.FunctionDef)
            for code in code_blocks(ast.get_docstring(node, clean=False))
        ]
        loose += [('<module>', code) for code in code_blocks(ast.get_docstring(tree, clean=False))]
        if loose:
            units.append((f'{relative}::<module>', [], loose))
    return units


def declared_example_count():
    """Count example directives straight from the source text.

    This is deliberately independent of the parser above: if the extraction stops
    matching, the two counts diverge instead of the suite silently running fewer
    examples.
    """
    return sum(path.read_text().count(DIRECTIVE) for path in sorted(SOURCE_ROOT.rglob('*.py')))


def _run(code, label, namespace):
    """Execute one example, keeping its output out of the test log."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(code, label, 'exec'), namespace)


def test_docstring_examples():
    """Verify every documented example executes and its assertions hold."""
    units = example_units()
    extracted = sum(len(owner) + len(members) for _, owner, members in units)
    declared = declared_example_count()
    assert extracted == declared, (
        f'extracted {extracted} examples but the source declares {declared}; the extraction '
        'is missing examples that would then go unchecked'
    )
    assert extracted > 0, 'no documented examples were found at all'

    failures = []
    for label, owner, members in units:
        namespace = {}
        owner_ok = True
        for code in owner:
            try:
                _run(code, f'<{label}>', namespace)
            except Exception as error:
                owner_ok = False
                failures.append(f'{label} [class example] {type(error).__name__}: {error}')

        # Member examples continue in the owner's namespace, so one that needs a
        # fresh object calls reset() itself rather than relying on the harness.
        for name, code in members:
            if not owner_ok:
                failures.append(f'{label}.{name} [not run, class example failed]')
                continue
            member_namespace = dict(namespace)
            try:
                _run(code, f'<{label}.{name}>', member_namespace)
            except Exception as error:
                failures.append(f'{label}.{name} [member example] {type(error).__name__}: {error}')

    assert not failures, (
        f'{len(failures)} documented example(s) failed:\n  ' + '\n  '.join(failures)
    )

    print(f'units carrying examples: {len(units)}')
    print(f'examples executed: {extracted} (source declares {declared})')
    print('Test passed.')


if __name__ == '__main__':
    test_docstring_examples()

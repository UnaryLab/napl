"""Guard the agent tooling text under `.claude/` against stale references.

The skill and workflow files instruct future agents; nothing else in CI reads them,
so a renamed class or moved file rots there silently. This test sweeps every
backticked token in `.claude/**` and fails on any that

- names a napl API that does not exist (lowercase and dotted `napl.*` tokens),
- names a repository path that does not exist,
- cites a line range outside the cited file's length,
- names a CamelCase UnarySim class that the local clone does not define. The clone
  is not part of this repo; when it is absent this check is skipped, and the skip
  is announced on stderr and as a warning so a run never reports it as a pass.

What it deliberately does not cover, so a reader does not mistake a pass for proof
that the `.claude/` text is accurate:

- **Unbackticked names in prose, except variant names.** Bare text is checked only
  for the `stem_variant` shape: a known napl stem, an underscore, then a closed set
  of variant suffixes (`hub`, `fxp`, `mix`, `pc`, `gaines`, `ugemm`, `hard`,
  `scale`), flagged when the whole name is not importable from `napl`. That shape is
  why the check cannot cry wolf: ordinary English in these docs (`add`, `round`,
  `encode`, `min`, `max`) carries no underscore and no variant suffix, so it is not
  a candidate at all rather than a candidate that happens to pass. Any other stale
  bare name, one that does not take a variant suffix, stays a review responsibility.
- **Whether a cited line says what the citing text claims.** Only the range's
  existence is checked, not its content.
- **Whether a claim about an API is true.** A sentence asserting the wrong
  execution mode, polarity, or default for a class that does exist is built
  entirely from resolving tokens and passes.
- **Which module a name lives in.** A token resolves if any scanned module
  exports it, so attributing a real name to the wrong subpackage passes.
- **Code inside fenced blocks.** Fences are read as plain text; a snippet that
  would raise on import is not executed.
"""

import importlib
import os
import re
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC_ROOT = REPO / '.claude'
DOC_SUFFIXES = {'.md', '.js', '.json', '.py'}

# Modules whose public names a backticked token may legitimately refer to.
NAME_MODULES = (
    'napl',
    'napl.utils',
    'napl.sim.base',
    'napl.sim.metric',
    'napl.sim.module',
    'napl.sim.module._shared',
    'napl.sim.operation',
)

# Backticked lowercase tokens that are deliberately not napl API names.
NAME_ALLOWLIST = {
    # English words used in prose or as report field values.
    'compare', 'entry', 'failed', 'forward', 'import', 'input', 'make', 'mode',
    'ported', 'skipped', 'square', 'thunk', 'timesteps', 'verified',
    # Parameter names in the bundled helper scripts' docstrings.
    'device', 'fn', 'header_marker', 'iters',
    # napl subpackage directories, not classes.
    'algorithm', 'metric', 'module', 'operation',
    # napl_base attributes and config keys, not module-level names.
    'codec_config', 'config', 'depth', 'dim', 'fracwidth', 'generator', 'hw_params',
    'intwidth', 'ntype', 'polarity', 'pp_delay', 'scale', 'spike_value',
    'stype', 'timestep', 'timestep_cur', 'width',
    # UnarySim identifiers quoted from the upstream source.
    'add_', 'hwcfg', 'hx', 'in_0', 'in_1', 'in_1_prob', 'in_2', 'rng_idx', 'swcfg',
    # Verilog keywords, port names, and generator helpers.
    'i_', 'i_clk', 'i_rst_n', 'localparam', 'make_random_perf_values',
    'make_values', 'o_', 'parameter', 'posedge',
    # Shell words, tool names, torch names, device names.
    'cuda', 'float32', 'importlib', 'iverilog', 'matmul', 'mps', 'python',
    'rg', 'vvp', 'xnor',
}

# Only paths under these roots are checked. Everything else spelled like a path
# points outside this repo (the UnarySim clone's `kernel/*.py`, `/tmp` scripts).
PATH_ROOTS = ('src/', 'tests/', 'docs/', 'zoo/', '.claude/', '.github/')

# Checked-root paths that are deliberately not expected to exist: they are
# placeholders for files an agent is told to create during a run.
PATH_ALLOWLIST_PREFIXES = (
    'src/napl/imp/operation/<',
    'tests/<',
)

# UnarySim is the upstream simulator napl was ported from; the mapping reference
# names its classes throughout. The clone lives outside this repo, so the class
# check runs only when it is present. `NAPL_UNARYSIM_CLONE` overrides the location;
# an empty or whitespace-only value is an error, since it would otherwise resolve to
# the current directory and scan this repo as if it were the clone. A path that is
# set but does not exist is not an error: that is the skipped-clone case below.
_CLONE_DEFAULT = '/Users/diwu/Projects/UnarySim'
_CLONE_SETTING = os.environ.get('NAPL_UNARYSIM_CLONE', _CLONE_DEFAULT)
if not _CLONE_SETTING.strip():
    raise ValueError('NAPL_UNARYSIM_CLONE is set to an empty value; give it a clone path '
                     f'or unset it to use the default {_CLONE_DEFAULT}')
UNARYSIM_CLONE = Path(_CLONE_SETTING)

# Backticked CamelCase tokens that are deliberately not UnarySim class names.
CLASS_ALLOWLIST = {
    # Agent tooling, build files, and report field values.
    'Agent', 'Makefile', 'PASS', 'Read', 'Task',
    # Verilog parameters.
    'DEPTH',
    # Python and torch names.
    'Function', 'FutureWarning',
    # Named in the mapping reference precisely to record that upstream lacks it.
    'FSUSquare',
    # An UnarySim method, not a class.
    'FSUMul_forward',
}

# Variant suffixes a napl class name may end in. Closed set: a bare word qualifies as
# a candidate only when it is a known stem plus one or more of these, so no ordinary
# English word can reach the importability check.
VARIANT_SUFFIXES = ('hub', 'fxp', 'mix', 'pc', 'gaines', 'ugemm', 'hard', 'scale')

# Bare `stem_variant` names that are deliberately not napl API names.
VARIANT_ALLOWLIST = {
    # The porting skill and workflow name it as an example of a name a future port creates.
    'linear_mix_pc',
}

NAME_RE = re.compile(r'^[a-z][a-z0-9_]*$')
SUFFIX_RE = re.compile(f"^(?:{'|'.join(VARIANT_SUFFIXES)})+$")
# A bare lowercase word carrying at least one underscore.
BARE_RE = re.compile(r'(?<![`\w.])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![`\w])')
BACKTICKED_RE = re.compile(r'`[^`\n]+`')
CLASS_RE = re.compile(r'^[A-Z][A-Za-z0-9_]*$')
UNARYSIM_CLASS_RE = re.compile(r'^class +([A-Za-z0-9_]+)', re.MULTILINE)
DOTTED_RE = re.compile(r'^napl(\.[a-z_][a-z0-9_]*)+$')
# A path token, with an optional `:12-34` line-range suffix.
PATH_RE = re.compile(r'^([A-Za-z0-9_.\-/]+/[A-Za-z0-9_.\-]+\.[a-z]+)(:\d+(-\d+)?)?$')


def _known_names():
    names = set()
    for module_name in NAME_MODULES:
        module = importlib.import_module(module_name)
        names.update(n for n in dir(module) if not n.startswith('__'))
    return names


def _doc_lines():
    for path in sorted(DOC_ROOT.rglob('*')):
        if not path.is_file() or path.suffix not in DOC_SUFFIXES:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            yield f'{path.relative_to(REPO)}:{number}', line


def _tokens():
    for site, line in _doc_lines():
        for match in re.finditer(r'`([^`\n]+)`', line):
            yield site, match.group(1)


def _bare_tokens():
    """Every bare `stem_variant`-shaped word outside backticks, with its site."""
    for site, line in _doc_lines():
        for match in BARE_RE.finditer(BACKTICKED_RE.sub(' ', line)):
            yield site, match.group(1)


def _stems(names):
    """Every underscore-delimited prefix of a known napl name, e.g. `mgu` from `mgu_hard_mix`."""
    stems = set()
    for name in names:
        parts = name.split('_')
        stems.update('_'.join(parts[:i]) for i in range(1, len(parts) + 1))
    return stems


def _is_variant(token, stems):
    """True when the token is a known stem followed by closed-set variant suffixes."""
    parts = token.split('_')
    return any('_'.join(parts[:i]) in stems and all(SUFFIX_RE.match(p) for p in parts[i:])
               for i in range(1, len(parts)))


def _unarysim_classes():
    """Every class name the local UnarySim clone defines, or None when it is absent."""
    if not UNARYSIM_CLONE.is_dir():
        notice = (f'CHECK NOT RUN: backticked CamelCase tokens in `.claude/` are not '
                  f'checked against UnarySim class names, because no UnarySim clone '
                  f'exists at {UNARYSIM_CLONE}. Every other check in this file ran.')
        print('!' * 78, f'!! {notice}', '!' * 78, sep='\n', file=sys.stderr, flush=True)
        warnings.warn(notice, stacklevel=2)
        return None
    names = set()
    for path in UNARYSIM_CLONE.rglob('*.py'):
        names.update(UNARYSIM_CLASS_RE.findall(path.read_text(errors='ignore')))
    return names


def _bad_line_range(path, suffix):
    """Report a cited `:12` or `:12-34` range that falls outside the file."""
    bounds = [int(part) for part in suffix.lstrip(':').split('-')]
    first, last = bounds[0], bounds[-1]
    length = len(path.read_text().splitlines())
    if first < 1 or last < first or last > length:
        return f'file has {length} lines'
    return None


def _resolve_dotted(token):
    parts = token.split('.')
    for split in range(len(parts), 0, -1):
        try:
            module = importlib.import_module('.'.join(parts[:split]))
        except ImportError:
            continue
        target = module
        for attribute in parts[split:]:
            if not hasattr(target, attribute):
                return False
            target = getattr(target, attribute)
        return True
    return False


def test_agent_doc_names():
    """Every backticked napl name, UnarySim class, cited path, and bare variant name resolves."""
    names = _known_names()
    classes = _unarysim_classes()
    stale = []
    for site, token in _tokens():
        if NAME_RE.match(token):
            if token not in NAME_ALLOWLIST and token not in names:
                stale.append(f'{site}: unknown napl name `{token}`')
        elif DOTTED_RE.match(token):
            if not _resolve_dotted(token):
                stale.append(f'{site}: unknown napl path `{token}`')
        elif CLASS_RE.match(token):
            if (classes is not None and token not in CLASS_ALLOWLIST
                    and token not in classes and token not in names):
                stale.append(f'{site}: unknown UnarySim class `{token}`')
        else:
            match = PATH_RE.match(token)
            if (match and token.startswith(PATH_ROOTS)
                    and not token.startswith(PATH_ALLOWLIST_PREFIXES)):
                target = REPO / match.group(1)
                if not target.exists():
                    stale.append(f'{site}: missing repository path `{token}`')
                elif match.group(2) and target.is_file():
                    problem = _bad_line_range(target, match.group(2))
                    if problem:
                        stale.append(f'{site}: line range outside `{token}` ({problem})')
    stems = _stems(names)
    for site, token in _bare_tokens():
        if (token not in names and token not in VARIANT_ALLOWLIST
                and _is_variant(token, stems)):
            stale.append(f'{site}: unknown napl variant name {token}')
    assert not stale, 'stale references in .claude:\n' + '\n'.join(stale)


if __name__ == '__main__':
    test_agent_doc_names()
    print('test_agent_doc_names: PASS')

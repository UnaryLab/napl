"""Guard the RULE_IMP.md rule 6 ledger of streaming operations that await RTL."""

import inspect
import re
from pathlib import Path

import napl.sim.operation as operation
from napl.sim.base import napl_base

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULE_PATH = _REPO_ROOT / "RULE_IMP.md"
_RTL_ROOT = _REPO_ROOT / "src" / "napl" / "imp" / "operation"

# The ledger sentence spells its count as an English word.
_COUNT_WORDS = (
    "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
    "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
    "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty",
)

_LEDGER_RE = re.compile(r"(\w+) streaming operations await RTL: ([^.]*)\.")


def streaming_ops_without_rtl():
    """Return the streaming operations that have no imp/operation folder."""
    names = set()
    for name, member in vars(operation).items():
        # napl_base is the abstract base re-exported into the package, not an
        # operation, so skip the base class itself rather than its name.
        if not inspect.isclass(member) or member is napl_base:
            continue
        if issubclass(member, napl_base) and member.streaming is True:
            if not (_RTL_ROOT / name).is_dir():
                names.add(name)
    return names


def parse_ledger(text):
    """Return (count_word, names) from the rule 6 ledger sentence in ``text``."""
    # Collapse whitespace so line wrapping and indentation cannot hide the
    # sentence, while the wording itself still has to match exactly.
    matches = _LEDGER_RE.findall(" ".join(text.split()))
    assert len(matches) == 1, \
        f"expected exactly one ledger sentence in RULE_IMP.md rule 6, found {len(matches)}"
    count_word, listed = matches[0]
    names = re.findall(r"`(\w+)`", listed)
    assert names, f"ledger sentence lists no backticked operation names: {listed!r}"
    return count_word, set(names)


def check_ledger(text):
    """Assert the ledger sentence in ``text`` matches the live enumeration."""
    count_word, listed = parse_ledger(text)
    actual = streaming_ops_without_rtl()

    missing = sorted(actual - listed)
    stale = sorted(listed - actual)
    assert not missing and not stale, (
        "RULE_IMP.md rule 6 ledger is stale: "
        f"missing from the ledger {missing}, listed but has RTL or is not streaming {stale}"
    )

    assert count_word in _COUNT_WORDS, \
        f"ledger count word {count_word!r} is not a recognized English number word"
    assert _COUNT_WORDS.index(count_word) == len(actual), (
        f"ledger count word {count_word!r} disagrees with the {len(actual)} "
        f"operations listed: {sorted(actual)}"
    )


def test_rule_6_ledger_matches_streaming_ops_without_rtl():
    """The rule 6 ledger names and counts exactly the streaming ops lacking RTL."""
    check_ledger(_RULE_PATH.read_text())


if __name__ == "__main__":
    test_rule_6_ledger_matches_streaming_ops_without_rtl()
    print("Test passed.")

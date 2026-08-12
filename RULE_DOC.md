# NAPL documentation rules

This file is the canonical policy for building and maintaining NAPL's Sphinx documentation. Documentation changes must keep the source, generated API pages, local preview, and GitHub Pages output consistent.

## Source ownership

1. **User-facing API claims live closest to the API.** Class, method, property, and function behavior belongs in the corresponding Python docstring. Follow the class-docstring and member-order requirements in [RULE_SIM.md](RULE_SIM.md) for simulation classes.
2. **Narrative pages live in Sphinx source files.** Put design, development, and API index text under `docs/source/`. Do not duplicate API behavior from docstrings in hand-written pages.
3. **Generated files are outputs.** Sphinx autosummary generates the per-class stub pages under `docs/source/api/sim/`, and Sphinx writes HTML to `docs/_build/html/`. Do not hand-edit either location. Change the owning docstring, source page, template, or stylesheet instead.
4. **Templates own page structure.** `docs/source/_templates/autosummary/class.rst` controls class-page member order and the divider before the constructor. `layout.html` owns the site header, and `about.html` owns the sidebar introduction and repository link.
5. **CSS owns presentation.** Keep site-wide fonts, spacing, colors, navigation, API headings, parameter alignment, and responsive behavior in `docs/source/_static/napl.css`. Do not place presentation-only markup in API docstrings.
6. **API pages mirror the package tree.** The `napl.sim` API documentation follows the package layout: each subpackage has an index page at `docs/source/api/sim/<subpackage>.rst` (singular name matching the subpackage) sitting beside its per-class stub directory `docs/source/api/sim/<subpackage>/`, whose `<name>.rst` files autosummary generates one per public export. Stub filenames are the bare export name, kept short by the programmatic `autosummary_filename_map` in `conf.py` that maps each `napl.sim.<subpackage>.<name>` to `<name>`, built from every subpackage's `__all__`. Rendered pages title each class by its full import path, `napl.sim.<subpackage>.<name>`.
7. **Comments state a goal in one sentence.** Each `#` comment is a single sentence giving the purpose or constraint behind the code it precedes, not a multi-sentence narration of what the code does; a comment that only restates the code is deleted rather than kept.

## API page format

- Render the class overview, minimum full example, and references before the constructor divider.
- Render `__init__()` after the divider, then public members in source order. Include the inherited public `reset()` documentation where the class template provides it.
- Keep section headings at one visual level within a class page. References use the same heading level as methods and properties, and paper titles use reStructuredText emphasis: `*Paper title*`.
- Keep every reference entry. When a venue or year cannot be verified, leave it empty rather than guessing it or dropping the entry.
- Write every reference entry as title, venue, year, with no author names.
- Validate every reference entry against a real source, such as the publisher page, IEEE Xplore, the ACM Digital Library, arXiv, or DBLP, before writing it. Never cite from memory.
- Class pages document only the public API surface. Internal helper methods use underscore-prefixed names and are excluded from generated pages.
- API pages document classes only. Functions, module instances, and other non-class names are not listed.
- Class-overview math stays high-level: state what the operation computes, never how the code computes it. No step-by-step narration of the implementation.
- Give the target equation alone. Add an approximated or implemented equation only when the implementation genuinely approximates the target, such as a truncated series expansion.
- Use bold text for parameter and configuration-variable labels, including parameter names in API signatures and nested configuration keys. Use inline code for literal values, tensor ranges, identifiers referenced inside prose, and code expressions.
- Keep examples close to the API item they demonstrate. Code blocks retain syntax highlighting and do not receive prose emphasis.
- Every import shown in a docstring example, a `docs/` page, a README, or any other documentation uses exactly `from napl.sim.<subpackage> import <name>`, where the subpackage (`base`, `operation`, `module`, `metric`, `structure`, `algorithm`) is the one that defines the name, as in `from napl.sim.operation import mul_gaines`. The flat `from napl import <name>` and the module-object forms (`import napl.sim.operation`, `from napl.sim import operation`) never appear in documentation.

## Build and preview

Install the pinned documentation dependency in the project environment:

```sh
conda run -n napl python -m pip install -r docs/requirements.txt
```

Run a full strict build after changing docstrings, Sphinx source, templates, configuration, or CSS:

```sh
conda run -n napl python -m sphinx -E -a -W --keep-going -b html docs/source docs/_build/html
```

The build passes only when it exits zero with no warnings. The `-E -a` options rebuild the Sphinx environment and every page so stale autosummary or docstring output cannot hide a problem.

Preview the generated site locally:

```sh
conda run -n napl python -m http.server 8765 --directory docs/_build/html
```

Open `http://localhost:8765/`. Check the changed page at desktop width and a narrow viewport. Confirm heading levels, member order, parameter indentation and emphasis, code-block rendering, navigation, and links. Inspect the generated HTML when a visual style depends on specific Sphinx markup such as `strong`, `field-list`, or `sig-param`.

## Deployment

`.github/workflows/docs.yml` performs a warning-strict build for pull requests and deploys `docs/_build/html` to GitHub Pages after a push to `main`. Keep deployment configuration in that workflow and set the repository's Pages source to **GitHub Actions**. Never commit `docs/_build/`.

## Pass criteria

A documentation change is complete when the strict build exits zero, the generated page contains the intended content and semantic markup, and the local preview has no layout regression. For API changes, verify at least the changed class page; for shared templates, CSS, navigation, or configuration, verify representative class, index, and narrative pages.

## Review tiers

A change confined to documentation, `#` comments, or README prose takes a one-pass spot check, with a verdict of at most three sentences, rather than the full mutation or bite proof a behavior change under `src/napl/` takes. The pass criteria above still apply: the strict build exits zero and the changed page renders as intended.

Mechanical work is verified by command rather than by re-derivation. Work is mechanical when its correctness is establishable from the command output alone, without reading the diff: renames, formatting passes, and moves or anchor updates qualify. Work whose correctness depends on what the changed text means is not mechanical, however small the diff. The test is whether the attached command can fail in a way that proves the change wrong. For a rename, a link or anchor update, or a file move, the author attaches the command output that covers the claim: an `rg --hidden` sweep over the scope the claim is stated at, the strict docs build, or both. The reviewer confirms the commands ran and that their scope covers the claim, spot-checks the results (a zero-hit sweep proves the old name is gone, not that the new name is right), and does not repeat the change by hand.

A measured number appearing in documentation states its construction beside the number: the experiment, configuration, and command that produced it. The lighter review tier does not license a number with no stated source, and a number whose construction cannot be stated is removed rather than published.

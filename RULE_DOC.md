# NAPL documentation rules

This file is the canonical policy for building and maintaining NAPL's Sphinx documentation. Documentation changes must keep the source, generated API pages, local preview, and GitHub Pages output consistent.

## Source ownership

1. **User-facing API claims live closest to the API.** Class, method, property, and function behavior belongs in the corresponding Python docstring. Follow the class-docstring and member-order requirements in [RULE_SIM.md](RULE_SIM.md) for simulation classes.
2. **Narrative pages live in Sphinx source files.** Put design, development, and API index text under `docs/source/`. Do not duplicate API behavior from docstrings in hand-written pages.
3. **Generated files are outputs.** Sphinx autosummary generates `docs/source/api/generated/`, and Sphinx writes HTML to `docs/_build/html/`. Do not hand-edit either location. Change the owning docstring, source page, template, or stylesheet instead.
4. **Templates own page structure.** `docs/source/_templates/autosummary/class.rst` controls class-page member order and the divider before the constructor. `layout.html` owns the site header, and `about.html` owns the sidebar introduction and repository link.
5. **CSS owns presentation.** Keep site-wide fonts, spacing, colors, navigation, API headings, parameter alignment, and responsive behavior in `docs/source/_static/napl.css`. Do not place presentation-only markup in API docstrings.

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

# NAPL documentation

The documentation uses Sphinx. Design pages are written in reStructuredText,
and API pages import NAPL to render class signatures and docstrings.

## Build HTML

Install the documentation dependency in the `napl` environment:

```sh
conda run -n napl python -m pip install -r docs/requirements.txt
```

Build with warnings treated as errors:

```sh
conda run -n napl make -C docs html
```

Open `docs/_build/html/index.html` in a browser. The generated directory is
ignored by Git.

## Deploy

The `docs.yml` GitHub Actions workflow builds the same HTML and deploys it to
GitHub Pages after a push to `main`. In the repository settings, set the Pages
source to **GitHub Actions** before the first deployment.

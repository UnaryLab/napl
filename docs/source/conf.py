import importlib
from pathlib import Path
import sys

from sphinx import addnodes
from sphinx.util.docstrings import prepare_docstring


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

project = 'NAPL'
author = 'UnaryLab'
copyright = 'UnaryLab'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

# Links resolve against the live torch docs; the inventory is read from the
# checked-in copy, so the build needs no network access.
intersphinx_mapping = {
    'torch': (
        'https://docs.pytorch.org/docs/stable/',
        str(Path(__file__).parent / '_inventory' / 'torch.inv'),
    ),
}

autosummary_generate = True
autosummary_generate_overwrite = True

# Name each stub after the short class/function name, so autosummary writes
# api/sim/<subpackage>/<name>.rst instead of the resolved full dotted path.
autosummary_filename_map = {}
for _sub in ('base', 'operation', 'module', 'metric', 'structure', 'algorithm'):
    try:
        _mod = importlib.import_module(f'napl.sim.{_sub}')
    except ModuleNotFoundError:
        continue
    for _name in getattr(_mod, '__all__', []):
        autosummary_filename_map[f'napl.sim.{_sub}.{_name}'] = _name
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

root_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
templates_path = ['_templates']

html_theme = 'alabaster'
html_title = 'NAPL documentation'
html_static_path = ['_static']
html_css_files = ['napl.css']
html_theme_options = {
    'description': 'Neuro-Adaptive Programming Language',
    'github_repo': 'napl',
    'github_user': 'UnaryLab',
}

_SELF_PREFIXED = {'attribute', 'method', 'property'}


def _prefix_self(app, doctree, docname):
    """Render class members as ``self.<name>`` in the built pages."""
    for desc in doctree.findall(addnodes.desc):
        if desc.get('domain') != 'py' or desc.get('objtype') not in _SELF_PREFIXED:
            continue
        for signature in desc.findall(addnodes.desc_signature):
            if not signature.get('class'):
                continue
            for name in signature.findall(addnodes.desc_name):
                index = name.parent.index(name)
                name.parent.insert(index, addnodes.desc_addname('self.', 'self.'))
                break


def _property_doc(app, what, name, obj, options, lines):
    """Document a property from its own docstring, not an inherited attribute comment."""
    if what == 'attribute' and isinstance(obj, property) and obj.__doc__:
        lines[:] = prepare_docstring(obj.__doc__)


def setup(app):
    app.connect('autodoc-process-docstring', _property_doc)
    app.connect('doctree-resolved', _prefix_self)

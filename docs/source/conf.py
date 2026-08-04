from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

project = 'NAPL'
author = 'UnaryLab'
copyright = 'UnaryLab'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
]

autosummary_generate = True
autosummary_generate_overwrite = True
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

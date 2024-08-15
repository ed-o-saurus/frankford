# Copyright (C) 2024 Edward F. Behn, Jr.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys

sys.path.insert(0, os.path.abspath("../../../frankford/src"))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "frankford"
copyright = "2024, Ed Behn"
author = "Ed Behn"
release = "0.1"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for EPUB output
epub_show_urls = "footnote"

autodoc_member_order = "bysource"

autodoc_mock_imports = ["cffi", "numba", "numpy", "pynvjitlink"]

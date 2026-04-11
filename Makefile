# Copyright (C) 2026 Edward F. Behn, Jr.

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

.PHONY: all clean wheel

all: wheel

wheel: frankford/__init__.py frankford/_bridge.py frankford/_common.py frankford/_dataset.py frankford/_dtypes.py frankford/_fitter.py frankford/_kernel.py frankford/_parameters.py frankford/fit.cu LICENSE pyproject.toml
	python3 -m build

clean:
	rm -fr build/ dist/ *.egg-info/ frankford/__pycache__

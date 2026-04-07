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

(
    "Use the CUDA system to fit multiple datasets simultaneously using the Levenberg-Marquardt"
    "algorithm."
)

__version__ = "0.3"


from ._dataset import Dataset
from ._parameters import (
    Side,
    FreeParameter,
    FixedParameter,
    TiedParameter,
    FreeParameterSetting,
    FixedParameterSetting,
)
from ._fitter import Result, Fitter
from ._dtypes import point_dtype

__all__ = [
    "Side",
    "Result",
    "FreeParameter",
    "FixedParameter",
    "TiedParameter",
    "Dataset",
    "FreeParameterSetting",
    "FixedParameterSetting",
    "Fitter",
    "point_dtype",
]

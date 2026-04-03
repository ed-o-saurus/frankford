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

"All static dtypes defined in the project"

# Third party packages
import numpy as np

point_dtype = np.dtype([("value", np.double), ("uncertainty", np.double)], align=True)

param_array_dtype = np.dtype(
    [
        ("values", np.uintp),  # array of float64
        ("out_offsets", np.uintp),  # array of int64
    ],
    align=True,
)

free_param_dtype = np.dtype(
    [
        ("init_value", np.uintp),  # ptr to param_array
        ("lower", np.uintp),  # ptr to param_array
        ("upper", np.uintp),  # ptr to param_array
        ("step", np.double),
        ("relstep", np.double),
        ("side", np.int8),
    ],
    align=True,
)

ind_var_dtype = np.dtype(
    [
        ("values", np.uintp),  # array of float64
        ("fit_offsets", np.uintp),  # array of int64
        ("out_offsets", np.uintp),  # array of int64
    ],
    align=True,
)

dataset_dtype = np.dtype(
    [
        ("fit_size", np.int64),
        ("points_fit_offsets", np.uintp),  # array of int64
        ("points_out_offsets", np.uintp),  # array of int64
        ("points", np.uintp),  # array of points
        ("n_ind_vars", np.int64),
        ("ind_vars", np.uintp),  # array of ptrs to ind_vars
    ],
    align=True,
)

# Used for return value
position_offset_dtype = np.dtype(
    [("position", np.int64), ("offset", np.uintp)], align=True
)

positions_offset_dtype = np.dtype(
    [("position1", np.int64), ("position2", np.int64), ("offset", np.uintp)],
    align=True,
)

SIZEOF_DOUBLE = np.double().itemsize
SIZEOF_POINT = point_dtype.itemsize

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

"Misc. functions"

# Python packages
from itertools import product
from math import prod
import re

# Third party packages
import numpy as np
from numba import cuda

FUNC_PAT = re.compile(r"\.visible\s+\.func\s+\(\.param\s+.b32\s+\w+\)\s+(\w+)")


def get_ptx_function_name(ptx, func_name):
    "Find the name of the visible function in PTX code"

    names = [match for match in FUNC_PAT.findall(ptx) if func_name in match]

    if len(names) != 1:
        raise ValueError("Cannot find function name")

    return names[0]


def convert_array(val):
    "Convert val to np array of doubles"
    return np.array(val, dtype=np.double, order="C")


def build_ptr_array(structs, d_arys):
    "Load list structs and array of pointer to them"

    ptr_ary = np.empty(len(structs), dtype=np.uintp)

    for i, struct in enumerate(structs):
        ptr_ary[i] = device_load(struct, d_arys)

    return ptr_ary


def device_load(h_ary, d_arys):
    """
    Load a host-array to the device and return a pointer to the address
    Appended to d_arys to prevent deletion on going out of scope.
    """
    d_ary = cuda.to_device(h_ary)
    d_arys.append(d_ary)
    return get_ptr(d_ary)


def get_ptr(d_ary):
    "Get the device address of a device-array"
    return np.uintp(d_ary.__cuda_array_interface__["data"][0])


def scalar(dtype):
    "Create scalar of dtype"
    return np.empty((), dtype=dtype)


def mk_offset_ary(shape, strides, sizeof):
    "Return offsets of an array for elements along the strides passed"
    return np.array(
        [
            sum(key * stride for (key, stride) in zip(keys, strides)) // sizeof
            for keys in product(*(range(size) for size in shape))
        ],
        dtype=np.int64,
    )


def mk_offset_ary_zeros(shape):
    "Return zero offsets of a shape passed. (Used for scalar values.)"
    return np.zeros(prod(shape), dtype=np.int64)

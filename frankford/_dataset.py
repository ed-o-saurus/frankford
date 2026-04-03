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

"Module for datasets and associated classes"

# Python packages
from inspect import signature
from collections import namedtuple

# Third party packages
import numpy as np
from numba import cuda, double

# Local
from ._common import convert_array, get_ptx_function_name
from ._dtypes import point_dtype, SIZEOF_DOUBLE


class ArgInfo(namedtuple("ArgInfo", ("is_param", "position"))):
    """
    Store information about a dataset model argument
    For internal use
    """

    @property
    def offset(self):
        "Memory offset of argument relative to start of array"
        return SIZEOF_DOUBLE * self.position


# pylint: disable-next=too-few-public-methods
class DatasetSetup:
    """
    Represent a dataset model
    For internal use
    """

    def __init__(self, model, parameter_infos):
        self.model = model
        sig = signature(model)

        for arg in sig.parameters.values():
            if arg.kind not in (arg.POSITIONAL_ONLY, arg.POSITIONAL_OR_KEYWORD):
                raise TypeError("Only Position Only arguments allowed in functions")

        args = list(sig.parameters)

        self.ptx_code, resty = cuda.compile_for_current_device(
            model,
            sig=double(*(double for _arg in args)),
            abi="numba",
        )
        if resty != double:
            raise TypeError("model does not return double")

        self.func_name = get_ptx_function_name(self.ptx_code, model.__name__)

        self.ind_var_names = []
        self.arg_infos = []
        ind_var_position = 0
        for arg in args:
            if arg in parameter_infos:
                self.arg_infos.append(
                    ArgInfo(is_param=True, position=parameter_infos[arg].position)
                )
            else:
                self.ind_var_names.append(arg)

                self.arg_infos.append(
                    ArgInfo(is_param=False, position=ind_var_position)
                )
                ind_var_position += 1


# pylint: disable-next=too-few-public-methods
class Dataset:
    "Represent data to fit to a model."

    def __init__(self, points, axis, ind_vars):
        """
        :param points: Dependent values along with uncertainties
        :type points: `numpy array
          <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_ of
          ``frankford.point_dtype``
        :param axis: Axis or axes along which a fit is performed. If axis is negative it counts
          from the last to the first axis. If axis is a tuple of ints, a fit is performed on all
          of the axes specified in the tuple instead of a single axis.
        :type axis: int | tuple of int
        :param ind_vars: ``dict`` of arrays repressing the independent variables to be passed to the
          model
        :type ind_vars: dict of str to `numpy array`_
        """

        if not (
            isinstance(points, np.ndarray)
            and points.dtype == point_dtype
            and points.flags.c_contiguous
        ):
            raise ValueError("points must be np array of point_dtype")

        self._points = points.copy()

        if not np.isfinite(points[np.isfinite(points["value"])]["uncertainty"]).all():
            raise ValueError("uncertainties must be positive finite")

        if not (points[np.isfinite(points["value"])]["uncertainty"] > 0.0).all():
            raise ValueError("uncertainties must be positive finite")

        ndim = self._points.ndim

        if isinstance(axis, (int, np.integer)):
            self._axis = {self._normalize_axis(ndim, axis)}
        elif not hasattr(axis, "__iter__"):
            raise TypeError(f"integer argument expected, got {type(self._axis)}")
        else:
            self._axis = set()
            for dim in axis:
                if isinstance(dim, (int, np.integer)):
                    dim = self._normalize_axis(ndim, dim)
                else:
                    raise TypeError(f"integer argument expected, got {type(dim)}")

                if dim in self._axis:
                    raise ValueError("duplicate value in 'axis'")

                self._axis.add(dim)

        self._axis = tuple(sorted(self._axis))

        self._full_shape = points.shape
        self._fit_shape = tuple(
            length
            for (idim, length) in enumerate(self._full_shape)
            if idim in self._axis
        )
        self._out_shape = tuple(
            length
            for (idim, length) in enumerate(self._full_shape)
            if idim not in self._axis
        )

        self._ind_vars = {}
        for name, ary in ind_vars.items():
            ary = convert_array(ary)

            if ary.shape not in (self._full_shape, self._fit_shape):
                raise ValueError(
                    f"independent variable '{name}' shape must be "
                    f"{self._full_shape} or {self._fit_shape}"
                )

            self._ind_vars[str(name)] = ary.copy()

    @property
    def points(self):
        """
        Dependent values along with uncertainties

        :rtype: `numpy array
          <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_ of
          ``frankford.point_dtype``
        """
        return self._points.copy()

    @property
    def axis(self):
        """
        Axis or axes along which a fit is performed.

        :rtype: tuple of int
        """
        return self._axis

    @property
    def full_shape(self):
        return self._full_shape

    @property
    def fit_shape(self):
        return self._fit_shape

    @property
    def out_shape(self):
        return self._out_shape

    @property
    def ind_vars(self):
        """
        param ind_vars: ``dict`` of arrays repressing the
         independent variables to be passed to the model

        :rtype:  dict of str to `numpy array`_
        """
        return {ind_var: val.copy() for ind_var, val in self._ind_vars.items()}

    @staticmethod
    def _normalize_axis(ndim, axis):
        "Convert axis number to value in range(ndim)"

        axis = int(axis)

        if 0 <= axis < ndim:
            return axis

        if -ndim <= axis < 0:
            return axis + ndim

        raise np.exceptions.AxisError(axis, ndim)

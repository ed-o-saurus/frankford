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

"Module for parameters and associated classes"

# Python packages
from enum import IntEnum
from math import inf
from inspect import signature

# Third party packages
import numpy as np
from numba import cuda, double

# Local
from ._common import convert_array, get_ptx_function_name
from ._dtypes import SIZEOF_DOUBLE


class Side(IntEnum):
    "Specify the sidedness of the finite difference when computing numerical derivatives."

    AUTO = 0
    "One-sided derivative with side chosen automatically"
    POS = 1
    r"""One-sided derivative: :math:`f^\prime \left( x \right) \approx
         \frac{f \left(x + h \right) - f \left( x \right)}{h}`"""
    NEG = -1
    r"""One-sided derivative: :math:`f^\prime \left( x \right) \approx
         \frac{f \left(x \right) - f \left( x - h \right)}{h}`"""
    BOTH = 2
    r"""Two-sided derivative: :math:`f^\prime \left( x \right) \approx
         \frac{f \left(x + h \right) - f \left( x - h \right)}{2 h}`"""


# pylint: disable-next=too-few-public-methods
class FreeParameter:
    """Represent a free parameter."""


# pylint: disable-next=too-few-public-methods
class FixedParameter:
    """Represent a fixed parameter."""


# pylint: disable-next=too-few-public-methods
class TiedParameter:
    """Represent a tied parameter."""

    def __init__(self, function):
        """
        :param function: function that takes other parameters as arguments
        :type function: callable
        """

        self._function = function

        sig = signature(function)

        for arg in sig.parameters.values():
            if arg.kind not in (arg.POSITIONAL_ONLY, arg.POSITIONAL_OR_KEYWORD):
                raise TypeError("Only Position Only arguments allowed in functions")

        self._args = tuple(sig.parameters)

        self._ptx_code, resty = cuda.compile_for_current_device(
            function,
            sig=double(*(double for _arg in self._args)),
            abi="numba",
        )
        if resty != double:
            raise TypeError("function does not return double")

        self._func_name = get_ptx_function_name(self.ptx_code, function.__name__)

    @property
    def args(self):
        """
        Names of arguments of passed function

        :rtype: list of str
        """
        return self._args

    @property
    def function(self):
        """
        Function

        :rtype: callable
        """
        return self._function

    @property
    def func_name(self):
        "Name of PTX function"
        return self._func_name

    @property
    def ptx_code(self):
        "PTX source code for fuction"
        return self._ptx_code


class ParameterInfo:
    """
    Stores additional information about how a parameter is used in context
    For internal use
    """

    def __init__(self, param, name):
        self.param = param
        self.name = name

        self.position = 0
        self.predecessors = set()

    @property
    def offset(self):
        "Memory offset of parameter relative to start of array"
        return SIZEOF_DOUBLE * self.position

    def add_predecessor(self, predecessor, param_infos):
        "Create set of names of all pamameter who's changes affect this parameter"
        if param_infos[predecessor].is_free:
            self.predecessors.add(predecessor)

        if param_infos[predecessor].is_tied:
            for arg in param_infos[predecessor].param.args:
                self.add_predecessor(arg, param_infos)

    @property
    def is_tied(self):
        "Is the parameter tied?"
        return isinstance(self.param, TiedParameter)

    @property
    def is_free(self):
        "Is the parameter free?"
        return isinstance(self.param, FreeParameter)

    @property
    def args(self):
        "Return list of argument of tied function (if applicable)"
        return self.param.args if self.is_tied else ()

    @property
    def func_name(self):
        "Name of PTX function"
        return self.param.func_name

    def __lt__(self, other):
        if isinstance(self.param, FreeParameter):
            return not isinstance(other.param, FreeParameter)

        if isinstance(self.param, FixedParameter):
            return isinstance(other.param, TiedParameter)

        return False


class FreeParameterSetting:
    "Store information about a free parameter."

    def __init__(
        self,
        init_values,
        *,
        lower=-inf,
        upper=+inf,
    ):
        r"""
        :param init_values: initial value(s)
        :type init_values: `np.double
         <https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double>`_ |
         `numpy array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_
         of `np.double`_
        :param lower: lower bound(s) for values, defaults to :math:`-\infty`
        :type lower: `np.double`_ | `numpy array`_ of `np.double`_
        :param upper: upper bound(s) for values, defaults to :math:`+\infty`
        :type upper: `np.double`_ | `numpy array`_ of `np.double`_
        """
        self._init_values = convert_array(init_values)

        self._lower = convert_array(lower)
        self._upper = convert_array(upper)

        if not np.isfinite(self._init_values).all():
            raise ValueError("initial values must be finite")

        if not (self._lower <= self._init_values).all():
            raise ValueError("lower bounds must be less than initial values")

        if not (self._init_values <= self._upper).all():
            raise ValueError("upper bounds must be greater than initial values")

        self._side = Side.AUTO
        self._step = None
        self._relative_step = np.sqrt(np.finfo(np.double).eps)  # 2^(-26)

    @property
    def init_values(self):
        """
        Initial value(s)

        :rtype: `numpy array`_ of `np.double`_
        """
        return self._init_values.copy()

    @property
    def lower(self):
        """
        Lower bound(s) for values

        :rtype: `numpy array`_ of `np.double`_
        """
        return self._lower.copy()

    @property
    def upper(self):
        """
        Upper bound(s) for values

        :rtype: `numpy array`_ of `np.double`_
        """
        return self._upper.copy()

    @property
    def step(self):
        """
        Determines the step side of the finite difference when computing numerical derivatives.
        Setting this overrides ``relative_step``.

        :type: `np.double
         <https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double>`_
        """
        return self._step

    @step.setter
    def step(self, value: np.double):
        value = np.double(value)

        if value <= 0.0 or not np.isfinite(value):
            raise ValueError("step must be positive finite")

        self._step = value
        self._relative_step = None

    @property
    def relative_step(self):
        r"""
        Determines the relative step side of the finite difference when computing numerical
        derivatives. Setting this overrides ``step``.
        Defaults to :math:`\sqrt{\varepsilon}` where :math:`\varepsilon = 2^{-52}`.

        :type: `np.double
          <https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double>`_
        """
        return self._relative_step

    @relative_step.setter
    def relative_step(self, value: np.double):
        value = np.double(value)

        if value <= 0.0 or not np.isfinite(value):
            raise ValueError("relative step must be positive finite")

        self._relative_step = value
        self._step = None

    @property
    def side(self):
        """
        Determines the sidedness of the finite difference when computing numerical derivatives.
        Defaults to :py:class:`frankford.Side.AUTO`

        :type: :py:class:`frankford.Side`
        """
        return self._side

    @side.setter
    def side(self, value):
        if not isinstance(value, Side):
            raise ValueError("side must be one of type Side")

        self._side = value


# pylint: disable-next=too-few-public-methods
class FixedParameterSetting:
    "Store value(s) of a fixed parameter."

    def __init__(self, values):
        """
        :param values: fixed parameter value(s)
        :type values: `np.double`_ | `numpy array`_ of `np.double`_
        """
        self._values = convert_array(values)

    @property
    def values(self):
        """
        Values

        :rtype: `numpy array`_ of `np.double`_
        """
        return self._values.copy()

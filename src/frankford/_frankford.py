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

# Python packages
from re import finditer as _finditer
from inspect import signature as _signature
from graphlib import TopologicalSorter as _TopologicalSorter
from io import StringIO as _StringIO
from collections import namedtuple as _namedtuple
from functools import reduce as _reduce
from itertools import product as _product
from importlib import resources as _resources
from enum import IntEnum as _IntEnum
from math import inf as _inf
import operator as _operator

# Third party packages
import numpy as _np
from pynvjitlink import patch as _patch
from numba import cuda as _cuda
from numba import double as _double
import cffi as _cffi

_patch.patch_numba_linker()

_ffi = _cffi.FFI()

_cu_source = _cuda.CUSource(
    _resources.files(__package__).joinpath("gpu_code.cu").read_text()
)


class Side(_IntEnum):
    """Specify the sidedness of the finite difference when computing numerical derivatives."""

    AUTO = 0
    """One-sided derivative with side chosen automatically"""
    POS = 1
    r"""One-sided derivative: :math:`f^\prime \left( x \right) \approx \frac{f \left(x + h \right) - f \left( x \right)}{h}`"""
    NEG = -1
    r"""One-sided derivative: :math:`f^\prime \left( x \right) \approx \frac{f \left(x \right) - f \left( x - h \right)}{h}`"""
    BOTH = 2
    r"""Two-sided derivative: :math:`f^\prime \left( x \right) \approx \frac{f \left(x + h \right) - f \left( x - h \right)}{2 h}`"""


class Result(_IntEnum):
    """Represent how a fit terminated."""

    ERR_UNKNOWN = -1
    """Unknown error"""
    ERR_DOF = -2
    """Not enough degrees of freedom"""
    ERR_USER_FUNC = -3
    """Error from user function"""
    OK_CHI_SQ = 1
    r"""Convergence in :math:`\chi^2` value"""
    OK_PAR = 2
    """Convergence in parameter value"""
    OK_BOTH = 3
    """Both :py:class:`frankford.Result.OK_CHI_SQ` and :py:class:`frankford.Result.OK_PAR` hold"""
    OK_DIR = 4
    """Convergence in orthogonality"""
    MAX_ITER = 5
    """Maximum number of iterations reached"""
    FTOL = 6
    """:py:class:`frankford.Fitter.ftol` is too small - no further improvement"""
    XTOL = 7
    """:py:class:`frankford.Fitter.xtol` is too small - no further improvement"""
    GTOL = 8
    """:py:class:`frankford.Fitter.gtol` is too small - no further improvement"""

    def __bool__(self):
        """Tell if result is successful."""
        return self.value > 0


point_dtype = _np.dtype(
    [("value", _np.double), ("uncertainty", _np.double)], align=True
)

_param_array_dtype = _np.dtype(
    [("values", _np.uintp), ("out_offsets", _np.uintp)], align=True  # array of float64
)  # array of int64

_free_param_dtype = _np.dtype(
    [
        ("init_value", _np.uintp),  # ptr to param_array
        ("lower", _np.uintp),  # ptr to param_array
        ("upper", _np.uintp),  # ptr to param_array
        ("step", _np.double),
        ("relstep", _np.double),
        ("side", _np.int8),
    ],
    align=True,
)

_ind_var_dtype = _np.dtype(
    [
        ("values", _np.uintp),  # array of float64
        ("fit_offsets", _np.uintp),  # array of int64
        ("out_offsets", _np.uintp),
    ],
    align=True,
)  # array of int64

_dataset_dtype = _np.dtype(
    [
        ("fit_size", _np.int64),
        ("points_fit_offsets", _np.uintp),  # array of int64
        ("points_out_offsets", _np.uintp),  # array of int64
        ("points", _np.uintp),  # array of points
        ("n_ind_vars", _np.int64),
        ("ind_vars", _np.uintp),
    ],
    align=True,
)  # array of ind_vars

# Used for return value
_position_offset_dtype = _np.dtype(
    [("position", _np.int64), ("offset", _np.uintp)], align=True
)

_positions_offset_dtype = _np.dtype(
    [("position1", _np.int64), ("position2", _np.int64), ("offset", _np.uintp)],
    align=True,
)

_SIZEOF_DOUBLE = _np.double().itemsize
_SIZEOF_POINT = point_dtype.itemsize


class _PTXFunctionData(
    _namedtuple("_PTXFunctionData", ("mangled_name", "arg_names", "ptx_source"))
):
    @staticmethod
    def process(function):
        sig = _signature(function)

        for arg in sig.parameters.values():
            if (
                arg.kind != arg.POSITIONAL_ONLY
                and arg.kind != arg.POSITIONAL_OR_KEYWORD
            ):
                raise TypeError("Only Position Only arguments allowed in functions")

        arg_names = list(sig.parameters)
        ptx, resty = _cuda.compile_ptx_for_current_device(
            function, sig=tuple(_double for _arg_name in arg_names), device=True
        )
        if resty != _double:
            raise TypeError("function does not return double")

        matches = list(
            _finditer(
                r"\.visible\s+\.func\s+(\(\.param\s+\.(b|s|u|f)(8|16|32|64)\s+\w+\)\s+)?(?P<mangled_name>[\w\$]+)",
                ptx,
            )
        )
        if len(matches) != 1:
            raise Exception("Unable to determine mangled_name")

        mangled_name = matches[0].group("mangled_name")

        ptx_source = _cuda.PTXSource(ptx.encode("ascii"))

        return _PTXFunctionData(mangled_name, arg_names, ptx_source)

    def __lt__(lhs, rhs):
        return lhs.mangled_name < rhs.mangled_name

    def __hash__(self):
        return hash(self.mangled_name)


class _Parameter:
    # byte offset in params array
    @property
    def _offset(self):
        return _SIZEOF_DOUBLE * self._position


class FreeParameter(_Parameter):
    """Represent a free parameter."""

    def __lt__(lhs, rhs):
        if isinstance(rhs, FixedParameter) or isinstance(rhs, TiedParameter):
            return True

        return False

    @property
    def _arg_names(self):
        yield from ()


class FixedParameter(_Parameter):
    """Represent a fixed parameter."""

    def __lt__(lhs, rhs):
        if isinstance(rhs, TiedParameter):
            return True

        return False

    @property
    def _arg_names(self):
        yield from ()


class TiedParameter(_Parameter):
    """Represent a tied parameter."""

    def __init__(self, function):
        """
        :param function: function that takes other parameters as arguments
        :type function: callable
        """
        self._ptx_function_data = _PTXFunctionData.process(function)

        # positions of Free Parameters that affect this parameter
        self._calcs_on_change = set()

    def __lt__(lhs, rhs):
        return False

    @property
    def _arg_names(self):
        return self._ptx_function_data.arg_names


class _DatasetSetup:
    def __init__(self, model, parameters):
        self.ptx_function_data = _PTXFunctionData.process(model)

        self.ind_vars_in_order = []
        self.argument_offsets = []
        for arg_name in self.ptx_function_data.arg_names:
            if arg_name in parameters:
                self.argument_offsets.append((True, parameters[arg_name]._offset))
            else:
                # Auto incement starting at zero
                ind_var_position = len(self.ind_vars_in_order)
                self.ind_vars_in_order.append(arg_name)

                self.argument_offsets.append((False, _SIZEOF_DOUBLE * ind_var_position))


class Dataset:
    """Represent data to fit to a model."""

    def __init__(self, points, axis, ind_vars):
        """
        :param points: Dependent values along with uncertainties
        :type points: `numpy array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_ of ``frankford.point_dtype``
        :param axis: Axis or axes along which a fit is performed. If axis is negative it counts from the last to the first axis. If axis is a tuple of ints, a fit is performed on all of the axes specified in the tuple instead of a single axis.
        :type axis: int | tuple of int
        :param ind_vars: ``dict`` of arrays repressing the independent variables to be passed to the model
        :type ind_vars: dict of str to `numpy array`_
        """
        self._points = points

        if not _np.isfinite(points[_np.isfinite(points["value"])]["uncertainty"]).all():
            raise ValueError("uncertainties must be positive finite")

        if not (points[_np.isfinite(points["value"])]["uncertainty"] > 0.0).all():
            raise ValueError("uncertainties must be positive finite")

        ndim = self._points.ndim

        if isinstance(axis, int) or isinstance(axis, _np.integer):
            self._axis = {_normalize_axis(ndim, axis)}
        elif not hasattr(axis, "__iter__"):
            raise TypeError(f"integer argument expected, got {type(self._axis)}")
        else:
            self._axis = set()
            for dim in axis:
                if isinstance(dim, int) or isinstance(dim, _np.integer):
                    dim = _normalize_axis(ndim, dim)
                else:
                    raise TypeError(f"integer argument expected, got {type(dim)}")

                if dim in self._axis:
                    raise ValueError("duplicate value in 'axis'")

                self._axis.add(dim)

        self._axis = sorted(self._axis)

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
            ary = _convert_array(ary)

            if ary.shape != self._full_shape and ary.shape != self._fit_shape:
                raise ValueError(
                    f"independent variable '{name}' shape must be {self._full_shape} or {self._fit_shape}"
                )

            self._ind_vars[str(name)] = ary

    def _build_struct(self, dataset_setup, d_arys):
        ret_val = _scalar(_dataset_dtype)

        ret_val["fit_size"] = _prod(self._fit_shape)

        points_fit_strides = tuple(
            stride
            for (idim, stride) in enumerate(self._points.strides)
            if idim in self._axis
        )
        ret_val["points_fit_offsets"] = _device_load(
            _mk_offset_ary(self._fit_shape, points_fit_strides, _SIZEOF_POINT), d_arys
        )

        points_out_strides = tuple(
            stride
            for (idim, stride) in enumerate(self._points.strides)
            if idim not in self._axis
        )
        ret_val["points_out_offsets"] = _device_load(
            _mk_offset_ary(self._out_shape, points_out_strides, _SIZEOF_POINT), d_arys
        )

        ret_val["points"] = _device_load(self._points, d_arys)

        ret_val["n_ind_vars"] = len(dataset_setup.ind_vars_in_order)
        ind_vars = []
        for name in dataset_setup.ind_vars_in_order:
            struct = _scalar(_ind_var_dtype)

            ary = self._ind_vars[name]

            struct["values"] = _device_load(ary, d_arys)

            if ary.shape == self._full_shape:
                fit_strides = tuple(
                    stride
                    for (idim, stride) in enumerate(ary.strides)
                    if idim in self._axis
                )

                fit_offsets = _mk_offset_ary(
                    self._fit_shape, fit_strides, _SIZEOF_DOUBLE
                )

                out_strides = tuple(
                    stride
                    for (idim, stride) in enumerate(ary.strides)
                    if idim not in self._axis
                )

                out_offsets = _mk_offset_ary(
                    self._out_shape, out_strides, _SIZEOF_DOUBLE
                )
            else:
                fit_offsets = _mk_offset_ary(
                    self._fit_shape, ary.strides, _SIZEOF_DOUBLE
                )

                out_offsets = _mk_offset_ary_zeros(self._out_shape)

            struct["fit_offsets"] = _device_load(fit_offsets, d_arys)
            struct["out_offsets"] = _device_load(out_offsets, d_arys)

            ind_vars.append(_device_load(struct, d_arys))

        ret_val["ind_vars"] = _device_load(_np.array(ind_vars, dtype=_np.uintp), d_arys)

        return _device_load(ret_val, d_arys)


class _ParameterSetting:
    pass


class FreeParameterSetting(_ParameterSetting):
    """Store information about a free parameter."""

    def __init__(
        self,
        init_values,
        *,
        lower=-_inf,
        upper=+_inf,
    ):
        """
        :param init_values: initial value(s)
        :type init_values: `np.double <https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double>`_ | `numpy array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_ of `np.double`_
        :param lower: lower bound(s) for values, defaults to -infinity
        :type lower: `np.double`_ | `numpy array`_ of `np.double`_
        :param upper: upper bound(s) for values, defaults to +infinity
        :type upper: `np.double`_ | `numpy array`_ of `np.double`_
        """
        self._init_values = _convert_array(init_values)

        self._lower = _convert_array(lower)
        self._upper = _convert_array(upper)

        if not _np.isfinite(self._init_values).all():
            raise ValueError("initial values must be finite")

        if not (self._lower <= self._init_values).all():
            raise ValueError("lower bounds must be less than initial values")

        if not (self._init_values <= self._upper).all():
            raise ValueError("upper bounds must be greater than initial values")

        self._side = Side.AUTO
        self._step = None
        self._relative_step = _np.sqrt(_np.finfo(_np.double).eps)

    @property
    def step(self):
        """
        Determine the step side of the finite difference when computing numerical derivatives.
        Setting this overrides ``relative_step``.

        :type: `np.double <https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double>`_
        """
        return self._step

    @step.setter
    def step(self, value):
        value = _np.double(value)

        if value <= 0.0 or not _np.isfinite(value):
            raise ValueError("step must be positive finite")

        self._step = value
        self._relative_step = None

    @property
    def relative_step(self):
        r"""
        Determine the relative step side of the finite difference when computing numerical derivatives.
        Setting this overrides ``step``.
        Defaults to :math:`\sqrt{\varepsilon}` where :math:`\varepsilon = 2^{-52}`.

        :type: `np.double <https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double>`_
        """
        return self._relative_step

    @relative_step.setter
    def relative_step(self, value):
        value = _np.double(value)

        if value <= 0.0 or not _np.isfinite(value):
            raise ValueError("relative step must be positive finite")

        self._relative_step = value
        self._step = None

    @property
    def side(self):
        """
        Determine the sidedness of the finite difference when computing numerical derivatives.
        Defaults to :py:class:`frankford.Side.AUTO`

        :type: :py:class:`frankford.Side`
        """
        return self._side

    @side.setter
    def side(self, value):
        if not isinstance(value, Side):
            raise ValueError("side must be one of type Side")

        self._side = value

    def _build_struct(self, out_shape, param_name, d_arys):
        ret_val = _scalar(_free_param_dtype)

        if self._init_values.shape != () and self._init_values.shape != out_shape:
            raise ValueError(
                f"Invalid shape for free parameter '{param_name}' initial values"
            )

        ret_val["init_value"] = _build_param_array(self._init_values, d_arys, out_shape)

        if self._lower.shape != () and self._lower.shape != out_shape:
            raise ValueError(
                f"Invalid shape for free parameter '{param_name}' lower bounds"
            )

        ret_val["lower"] = _build_param_array(self._lower, d_arys, out_shape)

        if self._upper.shape != () and self._upper.shape != out_shape:
            raise ValueError(
                f"Invalid shape for free parameter '{param_name}' upper bounds"
            )

        ret_val["upper"] = _build_param_array(self._upper, d_arys, out_shape)

        ret_val["step"] = 0.0 if self._step is None else self._step
        ret_val["relstep"] = 0.0 if self._relative_step is None else self._relative_step
        ret_val["side"] = self._side

        return _device_load(ret_val, d_arys)


class FixedParameterSetting(_ParameterSetting):
    """Store value(s) of a fixed parameter."""

    def __init__(self, values):
        """
        :param values: fixed parameter value(s)
        :type values: `np.double`_ | `numpy array`_ of `np.double`_
        """
        self._values = _convert_array(values)

    def _build_struct(self, out_shape, param_name, d_arys):
        if self._values.shape != () and self._values.shape != out_shape:
            raise ValueError(f"Invalid shape for tied parameter '{param_name}' values")

        return _build_param_array(self._values, d_arys, out_shape)


class Fitter:
    """Represent a fit set."""

    def __init__(
        self,
        parameters,
        models,
    ):
        """
        Setup an array of fits.

        :param parameters: Parameters to be passed to models
        :type parameters: dict of str to (:py:class:`frankford.FreeParameter | :py:class:`frankford.FixedParameter` | :py:class:`frankford.TiedParameter`)
        :param models: functions to fit to datasets
        :type models: dict of Any to callable
        """
        self._ftol = _np.double(1e-10)
        self._xtol = _np.double(1e-10)
        self._gtol = _np.double(1e-10)
        self._stepfactor = _np.double(100.0)
        self._covtol = _np.double(1e-14)
        self._maxiter = _np.int64(200)
        self._douserscale = _np.uint8(False)

        parameters = {
            str(param_name): parameters[param_name] for param_name in parameters
        }

        for param_name, param in parameters.items():
            if not isinstance(param, _Parameter):
                raise TypeError(f"parameter '{param_name}' is unknown type")

        self._n_free_param = sum(
            1 for param in parameters.values() if isinstance(param, FreeParameter)
        )

        self._n_fixed_param = sum(
            1 for param in parameters.values() if isinstance(param, FixedParameter)
        )

        self._n_tied_param = sum(
            1 for param in parameters.values() if isinstance(param, TiedParameter)
        )

        if not self._n_free_param:
            raise TypeError("At least one free parameter must be specified")

        for param in parameters.values():
            for arg_name in param._arg_names:
                if arg_name not in parameters:
                    raise TypeError(f"no param named '{arg_name}'")

        graph = _TopologicalSorter()
        for param_name, param in parameters.items():
            graph.add(param_name, *(param._arg_names))

        graph.prepare()

        _sorted_param_names = []
        generation = sorted(graph.get_ready())

        _sorted_param_names += generation
        graph.done(*generation)

        while graph.is_active():
            generation = sorted(graph.get_ready())
            assert all(
                isinstance(parameters[param_name], TiedParameter)
                for param_name in generation
            )

            _sorted_param_names += generation
            graph.done(*generation)

        for position, param_name in enumerate(_sorted_param_names):
            parameters[param_name]._position = position

        # Sort by type then by name
        _sorted_param_names.sort(key=parameters.get)
        self._parameters = {
            param_name: parameters[param_name] for param_name in _sorted_param_names
        }

        for position, param in enumerate(self._parameters.values()):
            param._position = position

        for param in self._parameters.values():
            if isinstance(param, TiedParameter):
                _set_calcs_on_change(param, param, self._parameters)

        if not models:
            raise ValueError("At least one dataset model must be specified")

        self._dataset_setups = {
            key: _DatasetSetup(models[key], self._parameters) for key in models
        }

        self.ptx_function_datas = set()

        for param in self._parameters.values():
            if isinstance(param, TiedParameter):
                self.ptx_function_datas.add(param._ptx_function_data)

        for dataset_setup in self._dataset_setups.values():
            self.ptx_function_datas.add(dataset_setup.ptx_function_data)

        ptx_bridge = _StringIO()

        print(".version 8.2", file=ptx_bridge)
        print(".target sm_50", file=ptx_bridge)
        print(".address_size 64", file=ptx_bridge)
        print(file=ptx_bridge)

        # Extern declarations
        for ptx_function_data in sorted(self.ptx_function_datas):
            print(
                f".extern .func (.param .b32 func_retval0) {ptx_function_data.mangled_name} (.param .b64 param_0",
                file=ptx_bridge,
                end="",
            )

            for i_arg, _arg_name in enumerate(ptx_function_data.arg_names, start=1):
                print(f", .param .b64 param_{i_arg}", file=ptx_bridge, end="")

            print(");", file=ptx_bridge)

        print(file=ptx_bridge)

        # Add setup_params function
        print(
            ".visible .func (.param .b32 func_retval0) setup_params(", file=ptx_bridge
        )
        print(".param .b64 setup_params_param_0", file=ptx_bridge)  # double *params
        print(")", file=ptx_bridge)
        print("{", file=ptx_bridge)

        print("  .reg .b64 %rd_params_addr;", file=ptx_bridge)
        print("  .reg .b32 %r_ret_val;", file=ptx_bridge)
        print("  .reg .pred %p_has_error;", file=ptx_bridge)
        print("  .reg .b32 %r_error_status;", file=ptx_bridge)
        print("  .reg .b64 %rd_out_addr;", file=ptx_bridge)
        print("  .reg .f64 %fd_out_val;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_lo;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_up;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_up_masked;", file=ptx_bridge)

        print(
            "  ld.param.u64 %rd_params_addr, [setup_params_param_0];", file=ptx_bridge
        )
        print("  mov.u32 %r_ret_val, 0;", file=ptx_bridge)

        for param in self._parameters.values():
            if not isinstance(param, TiedParameter):
                continue

            print("  {", file=ptx_bridge)

            print(
                f"    add.s64 %rd_out_addr, %rd_params_addr, {param._offset};",
                file=ptx_bridge,
            )

            for i_arg, arg_name in enumerate(param._arg_names, start=1):
                print(f"    .reg .f64 %fd_value_{i_arg};", file=ptx_bridge)

                offset = self._parameters[arg_name]._offset
                print(
                    f"    ld.f64 %fd_value_{i_arg}, [%rd_params_addr+{offset}];",
                    file=ptx_bridge,
                )

            print("    {", file=ptx_bridge)
            print("      .param .b64 param_0;", file=ptx_bridge)
            print(
                "      st.param.b64 [param_0], %rd_out_addr;",
                file=ptx_bridge,
            )

            for i_arg, _arg_name in enumerate(param._arg_names, start=1):
                print(f"      .param .b64 param_{i_arg};", file=ptx_bridge)
                print(
                    f"      st.param.f64 [param_{i_arg}], %fd_value_{i_arg};",
                    file=ptx_bridge,
                )

            print("      .param .b32 retval0;", file=ptx_bridge)

            print(
                f"      call.uni (retval0), {param._ptx_function_data.mangled_name}, (param_0",
                file=ptx_bridge,
                end="",
            )
            for i_arg, _arg_name in enumerate(param._arg_names, start=1):
                print(f", param_{i_arg}", file=ptx_bridge, end="")

            print(");", file=ptx_bridge)

            print(
                "      ld.param.b32 %r_error_status, [retval0];",
                file=ptx_bridge,
            )
            print("    }", file=ptx_bridge)
            print(
                "    setp.ne.s32 %p_has_error, %r_error_status, 0;",
                file=ptx_bridge,
            )
            print("    ld.f64 %fd_out_val, [%rd_out_addr];", file=ptx_bridge)
            print(
                "    mov.b64 {%r_out_val_lo, %r_out_val_up}, %fd_out_val;",
                file=ptx_bridge,
            )
            print(
                "    and.b32 %r_out_val_up_masked, %r_out_val_up, 2146435072;",
                file=ptx_bridge,
            )
            print(
                "    setp.eq.or.s32 %p_has_error, %r_out_val_up_masked, 2146435072, %p_has_error;",
                file=ptx_bridge,
            )
            print("    @%p_has_error bra $L_SETUP_END;", file=ptx_bridge)

            print("  }", file=ptx_bridge)

        print("  mov.u32 %r_ret_val, 1;", file=ptx_bridge)
        print("$L_SETUP_END:", file=ptx_bridge)
        print("  st.param.b32 [func_retval0], %r_ret_val;", file=ptx_bridge)
        print("  ret;", file=ptx_bridge)
        print("}", file=ptx_bridge)
        print(file=ptx_bridge)

        # Add update_params function
        print(
            ".visible .func (.param .b32 func_retval0) update_params(", file=ptx_bridge
        )
        print("  .param .b64 update_params_param_0,", file=ptx_bridge)  # double *params
        print("  .param .b64 update_params_param_1", file=ptx_bridge)  # int64 updated
        print(")", file=ptx_bridge)
        print("{", file=ptx_bridge)

        print("  .reg .b64 %rd_params_addr;", file=ptx_bridge)
        print("  .reg .b64 %r_updated;", file=ptx_bridge)
        print("  .reg .b32 %r_ret_val;", file=ptx_bridge)
        print("  .reg .pred %p_has_error;", file=ptx_bridge)
        print("  .reg .pred %p_skip;", file=ptx_bridge)
        print("  .reg .b32 %r_error_status;", file=ptx_bridge)
        print("  .reg .b64 %rd_out_addr;", file=ptx_bridge)
        print("  .reg .f64 %fd_out_val;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_lo;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_up;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_up_masked;", file=ptx_bridge)

        print(
            "  ld.param.u64 %rd_params_addr, [update_params_param_0];",
            file=ptx_bridge,
        )
        print(
            "  ld.param.u64 %r_updated, [update_params_param_1];",
            file=ptx_bridge,
        )
        print("  mov.u32 %r_ret_val, 0;", file=ptx_bridge)

        for i_param, param in enumerate(self._parameters.values()):
            if not isinstance(param, TiedParameter):
                continue

            if not param._calcs_on_change:
                continue

            print("  {", file=ptx_bridge)

            for j, calc_on_change in enumerate(param._calcs_on_change):
                if j:  # Is not first
                    print(
                        f"    setp.ne.and.s64 %p_skip, %r_updated, {calc_on_change}, %p_skip;",
                        file=ptx_bridge,
                    )
                else:
                    print(
                        f"    setp.ne.s64 %p_skip, %r_updated, {calc_on_change};",
                        file=ptx_bridge,
                    )

            print(
                f"    @%p_skip bra $L_FUNC_END_{i_param};",
                file=ptx_bridge,
            )
            print(
                f"    add.s64 %rd_out_addr, %rd_params_addr, {param._offset};",
                file=ptx_bridge,
            )

            for i_arg, arg_name in enumerate(param._arg_names, start=1):
                print(f"    .reg .f64 %fd_value_{i_arg};", file=ptx_bridge)

                offset = self._parameters[arg_name]._offset
                print(
                    f"    ld.f64 %fd_value_{i_arg}, [%rd_params_addr+{offset}];",
                    file=ptx_bridge,
                )

            print("    {", file=ptx_bridge)
            print("      .param .b64 param_0;", file=ptx_bridge)
            print(
                "      st.param.b64 [param_0], %rd_out_addr;",
                file=ptx_bridge,
            )

            for i_arg, _arg_name in enumerate(param._arg_names, start=1):
                print(f"      .param .b64 param_{i_arg};", file=ptx_bridge)
                print(
                    f"      st.param.f64 [param_{i_arg}], %fd_value_{i_arg};",
                    file=ptx_bridge,
                )

            print("      .param .b32 retval0;", file=ptx_bridge)

            print(
                f"      call.uni (retval0), {param._ptx_function_data.mangled_name}, (param_0",
                file=ptx_bridge,
                end="",
            )
            for i_arg, _arg_name in enumerate(param._arg_names, start=1):
                print(f", param_{i_arg}", file=ptx_bridge, end="")

            print(");", file=ptx_bridge)

            print(
                "      ld.param.b32 %r_error_status, [retval0];",
                file=ptx_bridge,
            )
            print("    }", file=ptx_bridge)
            print(
                "    setp.ne.s32 %p_has_error, %r_error_status, 0;",
                file=ptx_bridge,
            )
            print("    ld.f64 %fd_out_val, [%rd_out_addr];", file=ptx_bridge)
            print(
                "    mov.b64 {%r_out_val_lo, %r_out_val_up}, %fd_out_val;",
                file=ptx_bridge,
            )
            print(
                "    and.b32 %r_out_val_up_masked, %r_out_val_up, 2146435072;",
                file=ptx_bridge,
            )
            print(
                "    setp.eq.or.s32 %p_has_error, %r_out_val_up_masked, 2146435072, %p_has_error;",
                file=ptx_bridge,
            )
            print("    @%p_has_error bra $L_UPDATE_END;", file=ptx_bridge)

            print(f"$L_FUNC_END_{i_param}:", file=ptx_bridge)

            print("  }", file=ptx_bridge)

        print("  mov.u32 %r_ret_val, 1;", file=ptx_bridge)
        print("$L_UPDATE_END:", file=ptx_bridge)
        print("  st.param.b32 [func_retval0], %r_ret_val;", file=ptx_bridge)
        print("  ret;", file=ptx_bridge)
        print("}", file=ptx_bridge)
        print(file=ptx_bridge)

        # Add call_func function
        print(".visible .func (.param .b32 func_retval0) call_func(", file=ptx_bridge)
        print("  .param .b64 call_func_param_0,", file=ptx_bridge)  # double* out
        print("  .param .b64 call_func_param_1,", file=ptx_bridge)  # double* params
        print("  .param .b64 call_func_param_2,", file=ptx_bridge)  # double* ind_vars
        print("  .param .b64 call_func_param_3", file=ptx_bridge)  # int64_t i_dataset
        print(")", file=ptx_bridge)
        print("{", file=ptx_bridge)
        print("  .reg .b32 %r_error_status;", file=ptx_bridge)
        print("  .reg .b32 %r_ret_val;", file=ptx_bridge)
        print("  .reg .b64 %rd_out_addr;", file=ptx_bridge)
        print("  .reg .b64 %rd_params_addr;", file=ptx_bridge)
        print("  .reg .b64 %rd_ind_vars_addr;", file=ptx_bridge)
        print("  .reg .b64 %rd_i_dataset;", file=ptx_bridge)
        print("  .reg .pred %p_select_function;", file=ptx_bridge)
        print("  .reg .pred %p_no_error;", file=ptx_bridge)
        print("  .reg .f64 %fd_out_val;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_lo;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_up;", file=ptx_bridge)
        print("  .reg .b32 %r_out_val_up_masked;", file=ptx_bridge)

        print("  ld.param.u64 %rd_out_addr, [call_func_param_0];", file=ptx_bridge)
        print("  ld.param.u64 %rd_params_addr, [call_func_param_1];", file=ptx_bridge)
        print("  ld.param.u64 %rd_ind_vars_addr, [call_func_param_2];", file=ptx_bridge)
        print("  ld.param.u64 %rd_i_dataset, [call_func_param_3];", file=ptx_bridge)

        for i_dataset, _dataset_setup in enumerate(self._dataset_setups.values()):
            print(
                f"  setp.eq.s64 %p_select_function, %rd_i_dataset, {i_dataset};",
                file=ptx_bridge,
            )
            print(
                f"  @%p_select_function bra $L_RUN_{i_dataset};",
                file=ptx_bridge,
            )

        print(f"  bra.uni $L_UNKNOWN;", file=ptx_bridge)

        for i_dataset, dataset_setup in enumerate(self._dataset_setups.values()):
            print(f"$L_RUN_{i_dataset}:", file=ptx_bridge)

            print("  {", file=ptx_bridge)

            for i_arg, (is_param, offset) in enumerate(
                dataset_setup.argument_offsets, start=1
            ):
                print(f"    .reg .f64 %fd_value_{i_arg};", file=ptx_bridge)

                addr = "%rd_params_addr" if is_param else "%rd_ind_vars_addr"
                print(
                    f"    ld.f64 %fd_value_{i_arg}, [{addr}+{offset}];",
                    file=ptx_bridge,
                )

            print("    {", file=ptx_bridge)
            print("      .param .b64 param_0;", file=ptx_bridge)
            print("      st.param.b64 [param_0], %rd_out_addr;", file=ptx_bridge)

            for i_arg, _arg_name in enumerate(
                dataset_setup.ptx_function_data.arg_names, start=1
            ):
                print(f"      .param .b64 param_{i_arg};", file=ptx_bridge)
                print(
                    f"      st.param.f64 [param_{i_arg}], %fd_value_{i_arg};",
                    file=ptx_bridge,
                )

            print("      .param .b32 retval0;", file=ptx_bridge)

            print(
                f"      call.uni (retval0), {dataset_setup.ptx_function_data.mangled_name}, (param_0",
                file=ptx_bridge,
                end="",
            )
            for i_arg, _arg_name in enumerate(
                dataset_setup.ptx_function_data.arg_names, start=1
            ):
                print(f", param_{i_arg}", file=ptx_bridge, end="")

            print(");", file=ptx_bridge)

            print("      ld.param.b32 %r_error_status, [retval0];", file=ptx_bridge)
            print("    }", file=ptx_bridge)
            print("    bra.uni $L_RETURN;", file=ptx_bridge)

            print("  }", file=ptx_bridge)

        print("$L_UNKNOWN:", file=ptx_bridge)
        print("  mov.u32 %r_error_status, 1;", file=ptx_bridge)

        print("$L_RETURN:", file=ptx_bridge)
        print("  setp.eq.s32 %p_no_error, %r_error_status, 0;", file=ptx_bridge)
        print("  ld.f64 %fd_out_val, [%rd_out_addr];", file=ptx_bridge)
        print("  mov.b64 {%r_out_val_lo, %r_out_val_up}, %fd_out_val;", file=ptx_bridge)
        print(
            "  and.b32 %r_out_val_up_masked, %r_out_val_up, 2146435072;",
            file=ptx_bridge,
        )
        print(
            "  setp.ne.and.s32 %p_no_error, %r_out_val_up_masked, 2146435072, %p_no_error;",
            file=ptx_bridge,
        )
        print("  selp.u32 %r_ret_val, 1, 0, %p_no_error;", file=ptx_bridge)
        print("  st.param.b32 [func_retval0], %r_ret_val;", file=ptx_bridge)
        print("  ret;", file=ptx_bridge)
        print("}", file=ptx_bridge)

        self._ptx_bridge = _cuda.PTXSource(ptx_bridge.getvalue().encode("ascii"))

        link = [_cu_source, self._ptx_bridge] + [
            ptx_function_data.ptx_source
            for ptx_function_data in self.ptx_function_datas
        ]
        self._kernel = _cuda.jit(link=link)(_kernel)

    def __call__(
        self,
        parameter_settings,
        datasets,
        *,
        returned_parameters=None,
        returned_uncertainties=[],
        returned_covar=[],
        block_size=256,
    ):
        """
        Execute an array of fits on the GPU and return the output array.

        :param parameter_settings: settings for free and fixed parameters
        :type parameter_settings: dict of str to (:py:class:`frankfordFreeParameterSetting` | :py:class:`frankford.FixedParameterSetting`)
        :type datasets: dict of Any to :py:class:`frankford.Dataset`
        :param returned_parameters: list of parameters to return, defaults to None
        :type returned_parameters: None | list of str, optional
        :param returned_uncertainties: list of parameter uncertainties to return, defaults to []
        :type returned_uncertainties: list of str, optional
        :param returned_covar: list of parameter covariances to return, defaults to []
        :type returned_covar: list of (str, str), optional
        :param block_size: number of threads per block on GPU, defaults to 256
        :type block_size: int, optional
        """
        # To ensure the ref count of device arrays stays above zero
        d_arys = []

        parameter_settings = {
            str(param_name): parameter_settings[param_name]
            for param_name in parameter_settings
        }

        out_shape = None
        for key in self._dataset_setups:
            dataset_setting = datasets[key]

            if out_shape is None:
                out_shape = dataset_setting._out_shape
            elif out_shape != dataset_setting._out_shape:
                raise ValueError("Inconsistent output shape")

        free_param_ptrs = []
        fixed_param_ptrs = []
        for param_name, param in self._parameters.items():
            if isinstance(param, TiedParameter):
                continue

            parameter_setting = parameter_settings[param_name]

            if isinstance(param, FreeParameter):
                if not isinstance(parameter_setting, FreeParameterSetting):
                    raise TypeError(
                        f"Expected parameter setting for '{param_name}' to be FreeParameterSetting"
                    )

                free_param_ptrs.append(
                    parameter_setting._build_struct(out_shape, param_name, d_arys)
                )
            elif isinstance(param, FixedParameter):
                if not isinstance(parameter_setting, FixedParameterSetting):
                    raise TypeError(
                        f"Expected parameter setting for '{param_name}' to be FixedParameterSetting"
                    )

                fixed_param_ptrs.append(
                    parameter_setting._build_struct(out_shape, param_name, d_arys)
                )
            else:
                raise TypeError(f"Unknown parameter type {type(param)}")

        d_free_param_ptrs = _cuda.to_device(
            _np.array(free_param_ptrs + [0], dtype=_np.uintp)
        )
        d_fixed_param_ptrs = _cuda.to_device(
            _np.array(fixed_param_ptrs + [0], dtype=_np.uintp)
        )

        d_dataset_ptrs = _cuda.to_device(
            _np.array(
                [
                    datasets[key]._build_struct(dataset_setup, d_arys)
                    for key, dataset_setup in self._dataset_setups.items()
                ]
                + [0],
                dtype=_np.uintp,
            )
        )

        if returned_parameters is None:
            returned_parameters = list(self._parameters)

        returned_params_positions_names = []
        for param_name in returned_parameters:
            param = self._parameters[param_name]
            returned_params_positions_names.append((param._position, param_name))

        returned_params_dtype = _np.dtype(
            [
                (param_name, _np.double)
                for _position, param_name in returned_params_positions_names
            ],
            align=True,
        )

        returned_uncertainties_positions_names = []
        for param_name in returned_uncertainties:
            param = self._parameters[param_name]
            if not isinstance(param, FreeParameter):
                raise ValueError("Can only return uncertainties of free parameters")

            returned_uncertainties_positions_names.append((param._position, param_name))

        returned_uncertainties_dtype = _np.dtype(
            [
                (param_name, _np.double)
                for _position, param_name in returned_uncertainties_positions_names
            ],
            align=True,
        )

        returned_covar_positions_names = []
        for param1_name, param2_name in returned_covar:
            param1 = self._parameters[param1_name]
            if not isinstance(param1, FreeParameter):
                raise ValueError("Can only return covariance of free parameters")

            param2 = self._parameters[param2_name]
            if not isinstance(param2, FreeParameter):
                raise ValueError("Can only return covariance of free parameters")

            returned_covar_positions_names.append(
                (param1._position, param2._position, f"{param1_name}${param2_name}")
            )

        returned_covar_dtype = _np.dtype(
            [
                (params_name, _np.double)
                for _position1, _position2, params_name in returned_covar_positions_names
            ],
            align=True,
        )

        returned_dtype = _np.dtype(
            [
                ("result", _np.int8),
                ("chi_sq", _np.double),
                ("dof", _np.int64),
                ("num_iter", _np.int64),
                ("orig_chi_sq", _np.double),
                ("parameters", returned_params_dtype),
                ("uncertainties", returned_uncertainties_dtype),
                ("covar", returned_covar_dtype),
            ],
            align=True,
        )

        result_offset = _get_field_offset(returned_dtype, "result")
        chi_sq_offset = _get_field_offset(returned_dtype, "chi_sq")
        dof_offset = _get_field_offset(returned_dtype, "dof")
        num_iter_offset = _get_field_offset(returned_dtype, "num_iter")
        orig_chi_sq_offset = _get_field_offset(returned_dtype, "orig_chi_sq")
        parameters_offset = _get_field_offset(returned_dtype, "parameters")
        uncertainties_offset = _get_field_offset(returned_dtype, "uncertainties")
        covar_offset = _get_field_offset(returned_dtype, "covar")

        # Extra element added to avoid passing illegal empty array
        returned_params_offsets = _np.empty(
            len(returned_params_positions_names) + 1, dtype=_np.uintp
        )
        for i, (returned_param_offset, (position, param_name)) in enumerate(
            zip(returned_params_offsets, returned_params_positions_names)
        ):
            returned_param_offset = _scalar(_position_offset_dtype)
            returned_param_offset["position"] = position
            returned_param_offset["offset"] = (
                _get_field_offset(returned_params_dtype, param_name) + parameters_offset
            )

            returned_params_offsets[i] = _device_load(returned_param_offset, d_arys)

        returned_uncertainties_offsets = _np.empty(
            len(returned_uncertainties_positions_names) + 1, dtype=_np.uintp
        )
        for i, (returned_uncertainty_offset, (position, param_name)) in enumerate(
            zip(returned_uncertainties_offsets, returned_uncertainties_positions_names)
        ):
            returned_uncertainty_offset = _scalar(_position_offset_dtype)
            returned_uncertainty_offset["position"] = position
            returned_uncertainty_offset["offset"] = (
                _get_field_offset(returned_uncertainties_dtype, param_name)
                + uncertainties_offset
            )

            returned_uncertainties_offsets[i] = _device_load(
                returned_uncertainty_offset, d_arys
            )

        returned_covar_offsets = _np.empty(
            len(returned_covar_positions_names) + 1, dtype=_np.uintp
        )
        for i, (
            returned_covar_offset,
            (position1, position2, param_names),
        ) in enumerate(zip(returned_covar_offsets, returned_covar_positions_names)):
            returned_covar_offset = _scalar(_positions_offset_dtype)

            returned_covar_offset["position1"] = position1
            returned_covar_offset["position2"] = position2
            returned_covar_offset["offset"] = (
                _get_field_offset(returned_covar_dtype, param_names) + covar_offset
            )

            returned_covar_offsets[i] = _device_load(returned_covar_offset, d_arys)

        d_returned_params_offsets = _cuda.to_device(returned_params_offsets)
        d_returned_uncertainties_offsets = _cuda.to_device(
            returned_uncertainties_offsets
        )
        d_returned_covar_offsets = _cuda.to_device(returned_covar_offsets)

        returned_values = _np.empty(out_shape, dtype=returned_dtype)
        returned_values["result"] = Result.ERR_UNKNOWN
        d_returned_values = _cuda.to_device(returned_values)
        returned_values_ptr = d_returned_values.__cuda_array_interface__["data"][0]

        n_threads = _np.int64(_prod(out_shape))
        n_tied_param = _np.int64(self._n_tied_param)

        fit_count = int(
            sum(
                _prod(dataset_setting._fit_shape)
                for dataset_setting in datasets.values()
            )
        )

        n_param = self._n_free_param + self._n_fixed_param + self._n_tied_param

        max_n_ind_var = max(
            len(dataset_setup.ind_vars_in_order)
            for dataset_setup in self._dataset_setups.values()
        )

        d_fvec_block = _cuda.device_array((n_threads, fit_count), dtype=_np.double)
        d_qtf_block = _cuda.device_array(
            (n_threads, self._n_free_param), dtype=_np.double
        )
        d_params_all_block = _cuda.device_array((n_threads, n_param), dtype=_np.double)
        d_params_block = _cuda.device_array((n_threads, n_param), dtype=_np.double)
        d_params_new_block = _cuda.device_array((n_threads, n_param), dtype=_np.double)
        d_fjac_block = _cuda.device_array(
            (n_threads, self._n_free_param * fit_count), dtype=_np.double
        )
        d_diag_block = _cuda.device_array(
            (n_threads, self._n_free_param), dtype=_np.double
        )
        d_wa1_block = _cuda.device_array((n_threads, n_param), dtype=_np.double)
        d_wa2_block = _cuda.device_array((n_threads, fit_count), dtype=_np.double)
        d_wa3_block = _cuda.device_array((n_threads, n_param), dtype=_np.double)
        d_wa4_block = _cuda.device_array((n_threads, fit_count), dtype=_np.double)
        d_ipvt_block = _cuda.device_array(
            (n_threads, self._n_free_param), dtype=_np.int64
        )
        d_ind_vars_block = _cuda.device_array(
            (n_threads, max_n_ind_var), dtype=_np.double
        )
        d_fixed_block = _cuda.device_array(
            (n_threads, self._n_fixed_param + 1), dtype=_np.double
        )

        grid_size, threads = divmod(int(n_threads), block_size)
        if threads:
            grid_size += 1

        self._kernel[block_size, grid_size](
            n_threads,
            d_free_param_ptrs,
            d_fixed_param_ptrs,
            n_tied_param,
            d_dataset_ptrs,
            self._ftol,
            self._xtol,
            self._gtol,
            self._stepfactor,
            self._covtol,
            self._maxiter,
            self._douserscale,
            result_offset,
            chi_sq_offset,
            dof_offset,
            num_iter_offset,
            orig_chi_sq_offset,
            d_returned_params_offsets,
            d_returned_uncertainties_offsets,
            d_returned_covar_offsets,
            returned_values_ptr,
            _np.uintp(returned_values.itemsize),
            d_fvec_block,
            d_qtf_block,
            d_params_all_block,
            d_params_block,
            d_params_new_block,
            d_fjac_block,
            d_diag_block,
            d_wa1_block,
            d_wa2_block,
            d_wa3_block,
            d_wa4_block,
            d_ipvt_block,
            d_ind_vars_block,
            d_fixed_block,
        )

        d_returned_values.copy_to_host(returned_values)
        return returned_values

    @property
    def ftol(self):
        """
        Relative chi-square convergence criterium

        :rtype: np.double
        """
        return self._ftol

    @ftol.setter
    def ftol(self, value):
        value = _np.double(value)

        if value <= 0.0 or not _np.isfinite(value):
            raise ValueError("ftol must be positive finite")

        self._ftol = value

    @property
    def xtol(self):
        """
        Relative parameter convergence criterium

        :rtype: np.double
        """
        return self._xtol

    @xtol.setter
    def xtol(self, value):
        value = _np.double(value)

        if value <= 0.0 or not _np.isfinite(value):
            raise ValueError("xtol must be positive finite")

        self._xtol = value

    @property
    def gtol(self):
        """
        Orthogonality convergence criterium

        :rtype: np.double
        """
        return self._gtol

    @gtol.setter
    def gtol(self, value):
        value = _np.double(value)

        if value <= 0.0 or not _np.isfinite(value):
            raise ValueError("gtol must be positive finite")

        self._gtol = value

    @property
    def stepfactor(self):
        """
        Initial step bound

        :rtype: np.double
        """
        return self._stepfactor

    @stepfactor.setter
    def stepfactor(self, value):
        value = _np.double(value)

        if value <= 0.0 or not _np.isfinite(value):
            raise ValueError("stepfactor must be positive finite")

        self._stepfactor = value

    @property
    def covtol(self):
        """
        Range tolerance for covariance calculation

        :rtype: np.double
        """
        return self._covtol

    @covtol.setter
    def covtol(self, value):
        value = _np.double(value)

        if value <= 0.0 or not _np.isfinite(value):
            raise ValueError("covtol must be positive finite")

        self._covtol = value

    @property
    def maxiter(self):
        """
        Maximum number of iterations

        :rtype: np.int64
        """
        return self._maxiter

    @maxiter.setter
    def maxiter(self, value):
        value = _np.int64(value)

        if value < 0:
            raise ValueError("maxiter must not be negative")

        self._maxiter = value

    @property
    def douserscale(self):
        """
        Scale variables by user values?

        :rtype: bool
        """
        return bool(self._douserscale)

    @douserscale.setter
    def douserscale(self, value):
        self._douserscale = _np.uint8(bool(value))


def _set_calcs_on_change(target, dependency, parameters):
    if isinstance(dependency, FreeParameter):
        target._calcs_on_change.add(dependency._position)

    for param_name in dependency._arg_names:
        _set_calcs_on_change(target, parameters[param_name], parameters)


def _normalize_axis(ndim, val):
    val = int(val)

    if 0 <= val < ndim:
        return val
    elif -ndim <= val < 0:
        return val + ndim
    else:
        raise _np.AxisError(
            f"axis {val} is out of bounds for array of dimension {ndim}"
        )


def _convert_array(val):
    if isinstance(val, _np.ndarray):
        if val.dtype == _np.double and val.flags.c_contiguous:
            return val
        else:
            return _np.array(val, dtype=_np.double, order="C")
    else:
        return _np.array(val, dtype=_np.double, order="C")


def _get_field_offset(dtype, field):
    return _np.uintp(dtype.fields[field][1])


def _device_load(h_ary, d_arys):
    d_ary = _cuda.to_device(h_ary)
    d_arys.append(d_ary)
    return _np.uintp(d_ary.__cuda_array_interface__["data"][0])


def _build_param_array(values, d_arys, out_shape):
    ret_val = _scalar(_param_array_dtype)

    ret_val["values"] = _device_load(values, d_arys)

    if values.shape == ():
        out_offsets = _mk_offset_ary_zeros(out_shape)
    else:
        out_offsets = _mk_offset_ary(out_shape, values.strides, _SIZEOF_DOUBLE)

    ret_val["out_offsets"] = _device_load(out_offsets, d_arys)

    return _device_load(ret_val, d_arys)


def _scalar(dtype):
    return _np.empty((), dtype=dtype)


def _prod(vals):
    return _reduce(_operator.mul, vals, 1)


def _mk_offset_ary(shape, strides, sizeof):
    return _np.array(
        [
            sum(key * stride for (key, stride) in zip(keys, strides)) // sizeof
            for keys in _product(*(range(size) for size in shape))
        ],
        dtype=_np.int64,
    )


def _mk_offset_ary_zeros(shape):
    return _np.zeros(_prod(shape), dtype=_np.int64)


_fit_func_args = []

# i_thread
_fit_func_args.append("int64")

# n_free_param,  free_params
_fit_func_args += ["int64", "CPointer(uintp)"]

# n_fixed_param, fixed_params
_fit_func_args += ["int64", "CPointer(uintp)"]

# n_tied_param
_fit_func_args.append("int64")

# n_dataset, datasets
_fit_func_args += ["int64", "CPointer(uintp)"]

# ftol, xtol, gtol, eps, stepfactor, covtol, maxiter, douserscale
_fit_func_args += [
    "float64",
    "float64",
    "float64",
    "float64",
    "float64",
    "int64",
    "uint8",
]

# result_offset, chi_sq_offset, dof_offset, num_iter_offset, orig_chi_sq_offset
_fit_func_args += ["uintp", "uintp", "uintp", "uintp", "uintp"]

# n_returned_param, returned_params
_fit_func_args += ["int64", "CPointer(uintp)"]

# n_returned_uncertainties, returned_uncertaintiess
_fit_func_args += ["int64", "CPointer(uintp)"]

# n_returned_covar, returned_covars
_fit_func_args += ["int64", "CPointer(uintp)"]

# returned_value
_fit_func_args.append("uintp")

# fvec
_fit_func_args.append("CPointer(float64)")

# qtf
_fit_func_args.append("CPointer(float64)")

# params_all
_fit_func_args.append("CPointer(float64)")

# params
_fit_func_args.append("CPointer(float64)")

# params_new
_fit_func_args.append("CPointer(float64)")

# fjac
_fit_func_args.append("CPointer(float64)")

# diag
_fit_func_args.append("CPointer(float64)")

# wa1
_fit_func_args.append("CPointer(float64)")

# wa2
_fit_func_args.append("CPointer(float64)")

# wa3
_fit_func_args.append("CPointer(float64)")

# wa4
_fit_func_args.append("CPointer(float64)")

# ipvt
_fit_func_args.append("CPointer(int64)")

# ind_vars
_fit_func_args.append("CPointer(float64)")

# fixed
_fit_func_args.append("CPointer(float64)")


_fit_func_args = ", ".join(_fit_func_args)


_fit_func = _cuda.declare_device("fit", f"int32({_fit_func_args})")


def _kernel(
    n_threads,
    free_param_ptrs,
    fixed_param_ptrs,
    n_tied_param,
    dataset_ptrs,
    ftol,
    xtol,
    gtol,
    stepfactor,
    covtol,
    maxiter,
    douserscale,
    result_offset,
    chi_sq_offset,
    dof_offset,
    num_iter_offset,
    orig_chi_sq_offset,
    returned_params_offsets,
    returned_uncertainties_offsets,
    returned_covar_offsets,
    returned_values_ptr,
    returned_values_itemsize,
    fvec_block,
    qtf_block,
    params_all_block,
    params_block,
    params_new_block,
    fjac_block,
    diag_block,
    wa1_block,
    wa2_block,
    wa3_block,
    wa4_block,
    ipvt_block,
    ind_vars_block,
    fixed_block,
):
    i_thread = _cuda.grid(1)
    if i_thread < n_threads:
        qtf_block[i_thread] = 0.0

        _ = _fit_func(
            i_thread,
            free_param_ptrs.size - 1,
            _ffi.from_buffer(free_param_ptrs),
            fixed_param_ptrs.size - 1,
            _ffi.from_buffer(fixed_param_ptrs),
            n_tied_param,
            dataset_ptrs.size - 1,
            _ffi.from_buffer(dataset_ptrs),
            ftol,
            xtol,
            gtol,
            stepfactor,
            covtol,
            maxiter,
            douserscale,
            result_offset,
            chi_sq_offset,
            dof_offset,
            num_iter_offset,
            orig_chi_sq_offset,
            returned_params_offsets.size - 1,
            _ffi.from_buffer(returned_params_offsets),
            returned_uncertainties_offsets.size - 1,
            _ffi.from_buffer(returned_uncertainties_offsets),
            returned_covar_offsets.size - 1,
            _ffi.from_buffer(returned_covar_offsets),
            returned_values_ptr + i_thread * returned_values_itemsize,
            _ffi.from_buffer(fvec_block[i_thread]),
            _ffi.from_buffer(qtf_block[i_thread]),
            _ffi.from_buffer(params_all_block[i_thread]),
            _ffi.from_buffer(params_block[i_thread]),
            _ffi.from_buffer(params_new_block[i_thread]),
            _ffi.from_buffer(fjac_block[i_thread]),
            _ffi.from_buffer(diag_block[i_thread]),
            _ffi.from_buffer(wa1_block[i_thread]),
            _ffi.from_buffer(wa2_block[i_thread]),
            _ffi.from_buffer(wa3_block[i_thread]),
            _ffi.from_buffer(wa4_block[i_thread]),
            _ffi.from_buffer(ipvt_block[i_thread]),
            _ffi.from_buffer(ind_vars_block[i_thread]),
            _ffi.from_buffer(fixed_block[i_thread]),
        )

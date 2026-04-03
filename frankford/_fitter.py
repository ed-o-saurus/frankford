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

"Module for the Fitter class"

# Python packages
from graphlib import TopologicalSorter
from enum import IntEnum
from math import prod

# Third party packages
import numpy as np
from numba import cuda

# Local
from ._common import (
    scalar,
    get_ptr,
    build_ptr_array,
    device_load,
    mk_offset_ary_zeros,
    mk_offset_ary,
)
from ._parameters import (
    TiedParameter,
    FreeParameter,
    FixedParameter,
    FreeParameterSetting,
    FixedParameterSetting,
    ParameterInfo,
)
from ._dtypes import (
    position_offset_dtype,
    positions_offset_dtype,
    param_array_dtype,
    ind_var_dtype,
    dataset_dtype,
    free_param_dtype,
    SIZEOF_DOUBLE,
    SIZEOF_POINT,
)
from ._kernel import mk_kernel
from ._dataset import DatasetSetup, Dataset


class Result(IntEnum):
    "Represent how a fit terminated."

    ERR_UNKNOWN = -1
    "Unknown error"
    ERR_DOF = -2
    "Not enough degrees of freedom"
    ERR_USER_FUNC = -3
    "Error from user function"
    OK_CHI_SQ = 1
    r"Convergence in :math:`\chi^2` value"
    OK_PAR = 2
    "Convergence in parameter value"
    OK_BOTH = 3
    "Both :py:class:`frankford.Result.OK_CHI_SQ` and :py:class:`frankford.Result.OK_PAR` hold"
    OK_DIR = 4
    "Convergence in orthogonality"
    MAX_ITER = 5
    "Maximum number of iterations reached"
    FTOL = 6
    ":py:class:`frankford.Fitter.ftol` is too small - no further improvement"
    XTOL = 7
    ":py:class:`frankford.Fitter.xtol` is too small - no further improvement"
    GTOL = 8
    ":py:class:`frankford.Fitter.gtol` is too small - no further improvement"

    def __bool__(self):
        "Tell if result is successful."
        return self.value > 0


class Fitter:
    "Represent a fit set."

    def __init__(
        self,
        parameters,
        models,
    ):
        """
        Setup an array of fits.

        :param parameters: Parameters to be passed to models
        :type parameters: dict of str to (:py:class:`frankford.FreeParameter` |
          :py:class:`frankford.FixedParameter` | :py:class:`frankford.TiedParameter`)
        :param models: functions to fit to datasets
        :type models: dict of Any to callable
        """

        self._ftol = np.double(1e-10)
        self._xtol = np.double(1e-10)
        self._gtol = np.double(1e-10)
        self._stepfactor = np.double(100.0)
        self._covtol = np.double(1e-14)
        self._maxiter = np.int64(200)
        self._douserscale = False

        self._parameters = {
            str(param_name): parameters[param_name] for param_name in parameters
        }

        for param_name, param in self._parameters.items():
            if not isinstance(param, (FreeParameter, FixedParameter, TiedParameter)):
                raise TypeError(f"parameter '{param_name}' is unknown type")

        if not any(
            isinstance(param, FreeParameter) for param in self._parameters.values()
        ):
            raise ValueError("At least one free parameter must be specified")

        param_infos_no_order = {
            param_name: ParameterInfo(param, param_name)
            for param_name, param in self._parameters.items()
        }

        topological_sorter = TopologicalSorter()
        for param_info in sorted(param_infos_no_order.values()):
            topological_sorter.add(param_info.name, *param_info.args)

        self._param_infos = {}
        for position, param_name in enumerate(topological_sorter.static_order()):
            self._param_infos[param_name] = param_infos_no_order[param_name]
            self._param_infos[param_name].position = position

        for param_info in self._param_infos.values():
            for arg in param_info.args:
                param_info.add_predecessor(arg, self._param_infos)

        self._models = models

        if not self._models:
            raise ValueError("At least one dataset model must be specified")

        self._dataset_setups = {
            dataset_key: DatasetSetup(model, self._param_infos)
            for dataset_key, model in self._models.items()
        }

        self._kernel = mk_kernel(self._param_infos, self._dataset_setups)

    # pylint: disable-next=too-many-arguments
    def __call__(
        self,
        parameter_settings,
        datasets,
        *,
        returned_parameters=None,
        returned_uncertainties=(),
        returned_covar=(),
        block_size=256,
    ):
        """
        Execute an array of fits on the GPU and return the output array.

        :param parameter_settings: settings for free and fixed parameters
        :type parameter_settings: dict of str to
          (:py:class:`frankford.FreeParameterSetting` | :py:class:`frankford.FixedParameterSetting`)
        :type datasets: dict of Any to :py:class:`frankford.Dataset`
        :param returned_parameters: list of parameters to return,
         defaults to returning all parameters.
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

        for param_name, param_info in self._param_infos.items():
            if isinstance(param_info.param, FreeParameter):
                if param_name not in parameter_settings:
                    raise ValueError(f"Expected parameter setting for '{param_name}'")

                if not isinstance(parameter_settings[param_name], FreeParameterSetting):
                    raise TypeError(
                        f"Expected parameter setting for '{param_name}' to be FreeParameterSetting"
                    )
            elif isinstance(param_info.param, FixedParameter):
                if param_name not in parameter_settings:
                    raise ValueError(f"Expected parameter setting for '{param_name}'")

                if not isinstance(
                    parameter_settings[param_name], FixedParameterSetting
                ):
                    raise TypeError(
                        f"Expected parameter setting for '{param_name}' to be FixedParameterSetting"
                    )
            elif isinstance(param_info.param, TiedParameter):
                if param_name in parameter_settings:
                    raise ValueError(
                        f"Should not have parameter setting for '{param_name}'"
                    )

        out_shape = None
        for dataset_key, dataset in datasets.items():
            if out_shape is None:
                out_shape = dataset.out_shape
            elif out_shape != dataset.out_shape:
                raise ValueError("Inconsistent output shape")

        for param_name, parameter_setting in parameter_settings.items():
            if param_name not in self._param_infos:
                raise ValueError(f"Unknown parameter '{param_name}'")

            param_info = self._param_infos[param_name]

            if isinstance(param_info.param, FreeParameter):
                if parameter_setting.init_values.shape not in (
                    (),
                    out_shape,
                ):
                    raise ValueError(
                        f"Invalid shape for free parameter '{param_name}' initial values"
                    )

                if parameter_setting.lower.shape not in ((), out_shape):
                    raise ValueError(
                        f"Invalid shape for free parameter '{param_name}' lower bounds"
                    )

                if parameter_setting.upper.shape not in ((), out_shape):
                    raise ValueError(
                        f"Invalid shape for free parameter '{param_name}' upper bounds"
                    )
            elif isinstance(param_info.param, FixedParameter):
                if parameter_setting.values.shape not in ((), out_shape):
                    raise ValueError(
                        f"Invalid shape for tied parameter '{param_name}' values"
                    )

        for dataset_key in self._dataset_setups:
            if dataset_key not in datasets:
                raise KeyError(f"Missing dataset for key '{dataset_key}")

            if not isinstance(datasets[dataset_key], Dataset):
                raise TypeError(f"Entry for '{dataset_key}' must be type Dataset")

        for dataset_key in datasets:
            if dataset_key not in self._dataset_setups:
                raise ValueError(f"Unknown dataset '{dataset_key}'")

        free_param_structs = []
        fixed_param_structs = []
        for param_name, param_info in self._param_infos.items():
            if param_info.is_tied:
                continue

            parameter_setting = parameter_settings[param_name]

            if isinstance(param_info.param, FreeParameter):
                free_param_structs.append(
                    self._build_free_param_struct(parameter_setting, out_shape, d_arys)
                )
            elif isinstance(param_info.param, FixedParameter):
                fixed_param_structs.append(
                    self._build_param_struct(
                        parameter_setting.values, out_shape, d_arys
                    )
                )

        dataset_structs = [
            self._build_dataset_struct(
                self._dataset_setups[dataset_key], datasets[dataset_key], d_arys
            )
            for dataset_key in self._dataset_setups
        ]

        if returned_parameters is None:
            returned_parameters = list(self._parameters)

        for param_name in returned_parameters:
            if param_name not in self._param_infos:
                raise KeyError(f"Unknown parameter : {param_name}")

        returned_params_dtype = np.dtype(
            [(param_name, np.double) for param_name in returned_parameters],
            align=True,
        )

        for param_name in returned_uncertainties:
            if param_name not in self._param_infos:
                raise KeyError(f"Unknown parameter : {param_name}")

            if not isinstance(self._param_infos[param_name].param, FreeParameter):
                raise ValueError(
                    f"Can only return uncertainties of free parameters ('{param_name}')"
                )

        returned_uncertainties_dtype = np.dtype(
            [(param_name, np.double) for param_name in returned_uncertainties],
            align=True,
        )

        for param1_name, param2_name in returned_covar:
            if param1_name not in self._param_infos:
                raise KeyError(f"Unknown parameter : {param1_name}")

            if not isinstance(self._param_infos[param1_name].param, FreeParameter):
                raise ValueError(
                    f"Can only return covariance of free parameters ('{param1_name}')"
                )

            if param2_name not in self._param_infos:
                raise KeyError(f"Unknown parameter : {param2_name}")

            if not isinstance(self._param_infos[param2_name].param, FreeParameter):
                raise ValueError(
                    f"Can only return covariance of free parameters ('{param2_name}')"
                )

        returned_covar_dtype = np.dtype(
            [
                (f"{param1_name}${param2_name}", np.double)
                for param1_name, param2_name in returned_covar
            ],
            align=True,
        )

        returned_dtype = np.dtype(
            [
                ("result", np.int8),
                ("chi_sq", np.double),
                ("dof", np.int64),
                ("num_iter", np.int64),
                ("orig_chi_sq", np.double),
                ("parameters", returned_params_dtype),
                ("uncertainties", returned_uncertainties_dtype),
                ("covar", returned_covar_dtype),
            ],
            align=True,
        )

        result_offset = get_field_offset(returned_dtype, "result")
        chi_sq_offset = get_field_offset(returned_dtype, "chi_sq")
        dof_offset = get_field_offset(returned_dtype, "dof")
        num_iter_offset = get_field_offset(returned_dtype, "num_iter")
        orig_chi_sq_offset = get_field_offset(returned_dtype, "orig_chi_sq")
        parameters_offset = get_field_offset(returned_dtype, "parameters")
        uncertainties_offset = get_field_offset(returned_dtype, "uncertainties")
        covar_offset = get_field_offset(returned_dtype, "covar")

        returned_params_offset_structs = []
        for param_name in returned_parameters:
            returned_param_offset_struct = scalar(position_offset_dtype)
            returned_param_offset_struct["position"] = self._param_infos[
                param_name
            ].position
            returned_param_offset_struct["offset"] = (
                get_field_offset(returned_params_dtype, param_name) + parameters_offset
            )

            returned_params_offset_structs.append(returned_param_offset_struct)

        returned_uncertainties_offset_structs = []
        for param_name in returned_uncertainties:
            returned_uncertainty_offset_struct = scalar(position_offset_dtype)
            returned_uncertainty_offset_struct["position"] = self._param_infos[
                param_name
            ].position
            returned_uncertainty_offset_struct["offset"] = (
                get_field_offset(returned_uncertainties_dtype, param_name)
                + uncertainties_offset
            )

            returned_uncertainties_offset_structs.append(
                returned_uncertainty_offset_struct
            )

        returned_covar_offset_structs = []
        for param1_name, param2_name in returned_covar:
            returned_covar_offset_struct = scalar(positions_offset_dtype)
            returned_covar_offset_struct["position1"] = self._param_infos[
                param1_name
            ].position
            returned_covar_offset_struct["position2"] = self._param_infos[
                param2_name
            ].position
            returned_covar_offset_struct["offset"] = (
                get_field_offset(returned_covar_dtype, f"{param1_name}${param2_name}")
                + covar_offset
            )

            returned_covar_offset_structs.append(returned_covar_offset_struct)

        returned_values = np.empty(out_shape, dtype=returned_dtype)
        returned_values["result"] = Result.ERR_UNKNOWN
        d_returned_values = cuda.to_device(returned_values)
        returned_values_ptr = get_ptr(d_returned_values)

        n_threads = np.int64(prod(out_shape))

        fit_count = int(
            sum(
                prod(dataset_setting._fit_shape)
                for dataset_setting in datasets.values()
            )
        )

        n_free_param = sum(
            1 for param in self._parameters.values() if isinstance(param, FreeParameter)
        )

        n_fixed_param = sum(
            1
            for param in self._parameters.values()
            if isinstance(param, FixedParameter)
        )

        n_tied_param = sum(
            1 for param in self._parameters.values() if isinstance(param, TiedParameter)
        )

        n_param = n_free_param + n_fixed_param + n_tied_param

        max_n_ind_var = max(
            len(dataset_setup.ind_var_names)
            for dataset_setup in self._dataset_setups.values()
        )

        d_fvec_block = cuda.device_array((n_threads, fit_count), dtype=np.double)
        d_qtf_block = cuda.device_array((n_threads, n_free_param), dtype=np.double)
        d_params_all_block = cuda.device_array((n_threads, n_param), dtype=np.double)
        d_params_block = cuda.device_array((n_threads, n_param), dtype=np.double)
        d_params_new_block = cuda.device_array((n_threads, n_param), dtype=np.double)
        d_fjac_block = cuda.device_array(
            (n_threads, n_free_param * fit_count), dtype=np.double
        )
        d_diag_block = cuda.device_array((n_threads, n_free_param), dtype=np.double)
        d_wa1_block = cuda.device_array((n_threads, n_param), dtype=np.double)
        d_wa2_block = cuda.device_array((n_threads, fit_count), dtype=np.double)
        d_wa3_block = cuda.device_array((n_threads, n_param), dtype=np.double)
        d_wa4_block = cuda.device_array((n_threads, fit_count), dtype=np.double)
        d_ipvt_block = cuda.device_array((n_threads, n_free_param), dtype=np.int64)
        d_ind_vars_block = cuda.device_array(
            (n_threads, max_n_ind_var), dtype=np.double
        )
        d_fixed_block = cuda.device_array(
            (n_threads, n_fixed_param + 1), dtype=np.double
        )

        grid_size, threads = divmod(int(n_threads), block_size)
        if threads:
            grid_size += 1

        self._kernel[block_size, grid_size](
            n_threads,
            build_ptr_array(free_param_structs, d_arys),
            build_ptr_array(fixed_param_structs, d_arys),
            n_tied_param,
            build_ptr_array(dataset_structs, d_arys),
            self._ftol,
            self._xtol,
            self._gtol,
            self._stepfactor,
            self._covtol,
            self._maxiter,
            np.uint8(self._douserscale),
            result_offset,
            chi_sq_offset,
            dof_offset,
            num_iter_offset,
            orig_chi_sq_offset,
            build_ptr_array(returned_params_offset_structs, d_arys),
            build_ptr_array(returned_uncertainties_offset_structs, d_arys),
            build_ptr_array(returned_covar_offset_structs, d_arys),
            returned_values_ptr,
            np.uintp(returned_values.itemsize),
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
        value = np.double(value)

        if value <= 0.0 or not np.isfinite(value):
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
        value = np.double(value)

        if value <= 0.0 or not np.isfinite(value):
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
        value = np.double(value)

        if value <= 0.0 or not np.isfinite(value):
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
        value = np.double(value)

        if value <= 0.0 or not np.isfinite(value):
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
        value = np.double(value)

        if value <= 0.0 or not np.isfinite(value):
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
        value = np.int64(value)

        if value < 0:
            raise ValueError("maxiter must not be negative")

        self._maxiter = value

    @property
    def douserscale(self):
        """
        Scale variables by user values?

        :rtype: bool
        """
        return self._douserscale

    @douserscale.setter
    def douserscale(self, value):
        self._douserscale = bool(value)

    @property
    def parameters(self):
        """
        Loaded parameter

        :rtype: dict of str to (:py:class:`frankford.FreeParameter` | :py:class:`frankford.FixedParameter` | :py:class:`frankford.TiedParameter`)
        """
        return self._parameters.copy()

    @property
    def models(self):
        """
        Loaded models

        :rtype: dict of Any to callable
        """
        return self._models.copy()

    @staticmethod
    def _build_free_param_struct(parameter_setting, out_shape, d_arys):
        struct = scalar(free_param_dtype)

        struct["init_value"] = device_load(
            Fitter._build_param_struct(
                parameter_setting.init_values, out_shape, d_arys
            ),
            d_arys,
        )

        struct["lower"] = device_load(
            Fitter._build_param_struct(parameter_setting.lower, out_shape, d_arys),
            d_arys,
        )

        struct["upper"] = device_load(
            Fitter._build_param_struct(parameter_setting.upper, out_shape, d_arys),
            d_arys,
        )

        struct["step"] = (
            0.0 if parameter_setting.step is None else parameter_setting.step
        )
        struct["relstep"] = (
            0.0
            if parameter_setting.relative_step is None
            else parameter_setting.relative_step
        )
        struct["side"] = parameter_setting.side

        return struct

    @staticmethod
    def _build_param_struct(values, out_shape, d_arys):
        struct = scalar(param_array_dtype)

        struct["values"] = device_load(values, d_arys)

        if values.shape == ():
            out_offsets = mk_offset_ary_zeros(out_shape)
        else:
            out_offsets = mk_offset_ary(out_shape, values.strides, SIZEOF_DOUBLE)

        struct["out_offsets"] = device_load(out_offsets, d_arys)

        return struct

    @staticmethod
    def _build_dataset_struct(dataset_setup, dataset, d_arys):
        struct = scalar(dataset_dtype)

        struct["fit_size"] = prod(dataset.fit_shape)

        points_fit_strides = tuple(
            stride
            for (idim, stride) in enumerate(dataset.points.strides)
            if idim in dataset.axis
        )
        struct["points_fit_offsets"] = device_load(
            mk_offset_ary(dataset.fit_shape, points_fit_strides, SIZEOF_POINT), d_arys
        )

        points_out_strides = tuple(
            stride
            for (idim, stride) in enumerate(dataset.points.strides)
            if idim not in dataset.axis
        )
        struct["points_out_offsets"] = device_load(
            mk_offset_ary(dataset.out_shape, points_out_strides, SIZEOF_POINT), d_arys
        )

        struct["points"] = device_load(dataset.points, d_arys)

        ind_var_structs = [
            Fitter._build_ind_var_struct(
                dataset.ind_vars[ind_var_name], dataset, d_arys
            )
            for ind_var_name in dataset_setup.ind_var_names
        ]
        struct["n_ind_vars"] = len(ind_var_structs)
        struct["ind_vars"] = device_load(
            build_ptr_array(ind_var_structs, d_arys), d_arys
        )

        return struct

    @staticmethod
    def _build_ind_var_struct(values, dataset, d_arys):
        struct = scalar(ind_var_dtype)
        struct["values"] = device_load(values, d_arys)

        if values.shape == dataset.full_shape:
            fit_strides = tuple(
                stride
                for (idim, stride) in enumerate(values.strides)
                if idim in dataset.axis
            )

            fit_offsets = mk_offset_ary(dataset.fit_shape, fit_strides, SIZEOF_DOUBLE)

            out_strides = tuple(
                stride
                for (idim, stride) in enumerate(values.strides)
                if idim not in dataset.axis
            )

            out_offsets = mk_offset_ary(dataset.out_shape, out_strides, SIZEOF_DOUBLE)
        else:
            fit_offsets = mk_offset_ary(
                dataset.fit_shape, values.strides, SIZEOF_DOUBLE
            )

            out_offsets = mk_offset_ary_zeros(dataset.out_shape)

        struct["fit_offsets"] = device_load(fit_offsets, d_arys)
        struct["out_offsets"] = device_load(out_offsets, d_arys)

        return struct


def get_field_offset(dtype, field):
    "Return the memory offset of a dtype field"
    return np.uintp(dtype.fields[field][1])

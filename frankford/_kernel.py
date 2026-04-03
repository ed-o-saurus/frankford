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

"Module for the mk_kernel fuction"

# Python packages
from importlib import resources


# Third party packages
from numba import cuda, types
import cffi

# Local
from ._bridge import mk_bridge
from ._parameters import TiedParameter


FIT_FUNC_SIG = types.int32(
    types.int64,  # i_thread
    types.int64,  # n_free_param
    types.CPointer(types.uintp),  # free_params
    types.int64,  # n_fixed_param
    types.CPointer(types.uintp),  # fixed_params
    types.int64,  # n_tied_param
    types.int64,  # n_dataset
    types.CPointer(types.uintp),  # datasets
    types.float64,  # ftol
    types.float64,  # xtol
    types.float64,  # gtol
    types.float64,  # stepfactor
    types.float64,  # covtol
    types.int64,  # maxiter
    types.uint8,  # douserscale
    types.uintp,  # result_offset
    types.uintp,  # chi_sq_offset
    types.uintp,  # dof_offset
    types.uintp,  # num_iter_offset
    types.uintp,  # orig_chi_sq_offset
    types.int64,  # n_returned_param
    types.CPointer(types.uintp),  # returned_params
    types.int64,  # n_returned_uncertainties
    types.CPointer(types.uintp),  # returned_uncertaintiess
    types.int64,  # n_returned_covar
    types.CPointer(types.uintp),  # returned_covars
    types.uintp,  # returned_value
    types.CPointer(types.float64),  # fvec
    types.CPointer(types.float64),  # qtf
    types.CPointer(types.float64),  # params_all
    types.CPointer(types.float64),  # params
    types.CPointer(types.float64),  # params_new
    types.CPointer(types.float64),  # fjac
    types.CPointer(types.float64),  # diag
    types.CPointer(types.float64),  # wa1
    types.CPointer(types.float64),  # wa2
    types.CPointer(types.float64),  # wa3
    types.CPointer(types.float64),  # wa4
    types.CPointer(types.int64),  # ipvt
    types.CPointer(types.float64),  # ind_vars
    types.CPointer(types.float64),  # fixed
)


ffi = cffi.FFI()


def mk_kernel(param_infos, dataset_setups):
    "Build the kernel"

    fit_func = cuda.declare_device(
        "fit",
        FIT_FUNC_SIG,
        link=(
            [
                # pylint: disable-next=no-member
                cuda.CUSource(
                    resources.files(__package__).joinpath("fit.cu").read_bytes(),
                    name="fit_cu",
                ),
                # pylint: disable-next=no-member
                cuda.PTXSource(
                    mk_bridge(param_infos, dataset_setups).encode(), name="bridge_ptx"
                ),
            ]
            + [
                # pylint: disable-next=no-member
                cuda.PTXSource(
                    param_info.param.ptx_code.encode(),
                    name=f"param_{param_info.position}_ptx",
                )
                for param_info in param_infos.values()
                if param_info.is_tied
            ]
            + [
                # pylint: disable-next=no-member
                cuda.PTXSource(dataset_setup.ptx_code.encode(), name=f"dataset_{i}_ptx")
                for i, dataset_setup in enumerate(dataset_setups.values())
            ]
        ),
    )

    # pylint: disable-next=too-many-positional-arguments, too-many-locals, too-many-arguments
    @cuda.jit
    def kernel(
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
        # pylint: disable-next=no-value-for-parameter
        i_thread = cuda.grid(1)
        if i_thread < n_threads:
            qtf_block[i_thread] = 0.0

            _ = fit_func(
                i_thread,
                free_param_ptrs.size,
                ffi.from_buffer(free_param_ptrs),
                fixed_param_ptrs.size,
                ffi.from_buffer(fixed_param_ptrs),
                n_tied_param,
                dataset_ptrs.size,
                ffi.from_buffer(dataset_ptrs),
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
                returned_params_offsets.size,
                ffi.from_buffer(returned_params_offsets),
                returned_uncertainties_offsets.size,
                ffi.from_buffer(returned_uncertainties_offsets),
                returned_covar_offsets.size,
                ffi.from_buffer(returned_covar_offsets),
                returned_values_ptr + i_thread * returned_values_itemsize,
                ffi.from_buffer(fvec_block[i_thread]),
                ffi.from_buffer(qtf_block[i_thread]),
                ffi.from_buffer(params_all_block[i_thread]),
                ffi.from_buffer(params_block[i_thread]),
                ffi.from_buffer(params_new_block[i_thread]),
                ffi.from_buffer(fjac_block[i_thread]),
                ffi.from_buffer(diag_block[i_thread]),
                ffi.from_buffer(wa1_block[i_thread]),
                ffi.from_buffer(wa2_block[i_thread]),
                ffi.from_buffer(wa3_block[i_thread]),
                ffi.from_buffer(wa4_block[i_thread]),
                ffi.from_buffer(ipvt_block[i_thread]),
                ffi.from_buffer(ind_vars_block[i_thread]),
                ffi.from_buffer(fixed_block[i_thread]),
            )

    return kernel

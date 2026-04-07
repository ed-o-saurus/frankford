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

"Module to store mk_bridge"

# Python packages
from io import StringIO


def mk_bridge(param_infos, dataset_setups):
    "Write PTX code to bridge fit function to user supplied functions"

    ptx_bridge = StringIO()

    print(".version 8.8", file=ptx_bridge)
    print(".target sm_86", file=ptx_bridge)
    print(".address_size 64", file=ptx_bridge)
    print(file=ptx_bridge)

    external_declarations(ptx_bridge, param_infos, dataset_setups)
    setup_params(ptx_bridge, param_infos)
    update_params(ptx_bridge, param_infos)
    call_func(ptx_bridge, dataset_setups)

    return ptx_bridge.getvalue()


def external_declarations(ptx_bridge, param_infos, dataset_setups):
    "Declare user supplied functions"

    for param_info in param_infos.values():
        if not param_info.is_tied:
            continue

        print(
            f".extern .func (.param .b32 func_retval) {param_info.func_name}",
            file=ptx_bridge,
            end=" ",
        )
        print("(.param .b64 param_0", file=ptx_bridge, end="")

        for i_arg, _arg in enumerate(param_info.args, start=1):
            print(f", .param .b64 param_{i_arg}", file=ptx_bridge, end="")

        print(");", file=ptx_bridge)

    for dataset_key in dataset_setups:
        print(
            f".extern .func (.param .b32 func_retval) {dataset_setups[dataset_key].func_name}",
            file=ptx_bridge,
            end=" ",
        )
        print("(.param .b64 param_0", file=ptx_bridge, end="")

        for i_arg, _arg_info in enumerate(
            dataset_setups[dataset_key].arg_infos, start=1
        ):
            print(f", .param .b64 param_{i_arg}", file=ptx_bridge, end="")

        print(");", file=ptx_bridge)

    print(file=ptx_bridge)


def setup_params(ptx_bridge, param_infos):
    """
    Define set_params
    This function calculates tied parameter values
    """

    print(".visible .func (.param .b32 func_retval) setup_params(", file=ptx_bridge)
    print(".param .b64 setup_params_param_0", file=ptx_bridge)  # double *params
    print(")", file=ptx_bridge)
    print("{", file=ptx_bridge)

    print("  .reg .b64 %rd_params_addr;", file=ptx_bridge)
    print("  .reg .b32 %r_retval;", file=ptx_bridge)
    print("  .reg .pred %p_no_error;", file=ptx_bridge)

    print("  ld.param.u64 %rd_params_addr, [setup_params_param_0];", file=ptx_bridge)
    print("  mov.pred %p_no_error, 1;", file=ptx_bridge)

    for param_info in param_infos.values():
        if not param_info.is_tied:  # Only need to calculate tied parameters
            continue

        call_tied_param_func(ptx_bridge, param_info, param_infos)

    print("  selp.u32 %r_retval, 1, 0, %p_no_error;", file=ptx_bridge)
    print("  st.param.b32 [func_retval], %r_retval;", file=ptx_bridge)
    print("  ret;", file=ptx_bridge)
    print("}", file=ptx_bridge)
    print(file=ptx_bridge)


def update_params(ptx_bridge, param_infos):
    """
    Define update_params
    This function updated tied parameter values only if needed
    """

    print(".visible .func (.param .b32 func_retval) update_params(", file=ptx_bridge)
    print("  .param .b64 update_params_param_0,", file=ptx_bridge)  # double *params
    print("  .param .b64 update_params_param_1", file=ptx_bridge)  # int64 updated
    print(")", file=ptx_bridge)
    print("{", file=ptx_bridge)

    print("  .reg .b64 %rd_params_addr;", file=ptx_bridge)
    print("  .reg .b64 %r_updated;", file=ptx_bridge)
    print("  .reg .b32 %r_retval;", file=ptx_bridge)
    print("  .reg .pred %p_no_error;", file=ptx_bridge)
    print("  .reg .pred %p_skip;", file=ptx_bridge)
    print("  .reg .b64 %rd_out_addr;", file=ptx_bridge)

    print(
        "  ld.param.u64 %rd_params_addr, [update_params_param_0];",
        file=ptx_bridge,
    )
    print(
        "  ld.param.u64 %r_updated, [update_params_param_1];",
        file=ptx_bridge,
    )
    print("  mov.pred %p_no_error, 1;", file=ptx_bridge)

    for i_param, param_info in enumerate(param_infos.values()):
        if not param_info.is_tied:  # Only need to calculate tied parameters
            continue

        if not param_info.predecessors:
            continue  # not dependant on other parameters.

        print("  mov.pred %p_skip, 1;", file=ptx_bridge)

        # determine if %r_updated in predecessors
        # if not skip calling param func
        for predecessor in param_info.predecessors:
            predecessor_position = param_infos[predecessor].position

            print(
                f"  setp.ne.and.s64 %p_skip, %r_updated, {predecessor_position}, %p_skip;",
                file=ptx_bridge,
            )

        print(
            f"  @%p_skip bra $L_FUNC_END_{i_param};",
            file=ptx_bridge,
        )

        call_tied_param_func(ptx_bridge, param_info, param_infos)

        print(f"$L_FUNC_END_{i_param}:", file=ptx_bridge)

    print("  selp.u32 %r_retval, 1, 0, %p_no_error;", file=ptx_bridge)
    print("  st.param.b32 [func_retval], %r_retval;", file=ptx_bridge)
    print("  ret;", file=ptx_bridge)
    print("}", file=ptx_bridge)
    print(file=ptx_bridge)


def call_tied_param_func(ptx_bridge, param_info, param_infos):
    "Call the param function to update tied parameter"

    print("  {", file=ptx_bridge)

    for i_arg, _arg in enumerate(param_info.args, start=1):
        # Define arguments to pass
        print(f"    .reg .f64 %fd_value_{i_arg};", file=ptx_bridge)

    print("    .reg .pred %p_no_func_error;", file=ptx_bridge)
    print("    .reg .f64 %fd_out_val;", file=ptx_bridge)
    print("    .reg .b64 %rd_out_addr;", file=ptx_bridge)
    print("    .reg .b32 %r_error_status;", file=ptx_bridge)

    print(
        f"    add.s64 %rd_out_addr, %rd_params_addr, {param_info.offset};",
        file=ptx_bridge,
    )

    for i_arg, arg in enumerate(param_info.args, start=1):
        print(
            f"    ld.f64 %fd_value_{i_arg}, [%rd_params_addr+{param_infos[arg].offset}];",
            file=ptx_bridge,
        )

    print("    {", file=ptx_bridge)
    print("      .param .b64 param_0;", file=ptx_bridge)

    for i_arg, _arg in enumerate(param_info.args, start=1):
        print(f"      .param .b64 param_{i_arg};", file=ptx_bridge)

    print("      .param .b32 retval;", file=ptx_bridge)

    print(
        "      st.param.b64 [param_0], %rd_out_addr;",
        file=ptx_bridge,
    )  # numba ABI passes return value in zero arg slot

    for i_arg, _arg in enumerate(param_info.args, start=1):
        print(
            f"      st.param.f64 [param_{i_arg}], %fd_value_{i_arg};",
            file=ptx_bridge,
        )

    print(
        f"      call (retval), {param_info.func_name}, (param_0",
        file=ptx_bridge,
        end="",
    )
    for i_arg, _arg in enumerate(param_info.args, start=1):
        print(f", param_{i_arg}", file=ptx_bridge, end="")

    print(");", file=ptx_bridge)

    print(
        "      ld.param.b32 %r_error_status, [retval];",
        file=ptx_bridge,
    )
    print("    }", file=ptx_bridge)

    print("    ld.f64 %fd_out_val, [%rd_out_addr];", file=ptx_bridge)

    print(
        "    setp.eq.s32 %p_no_func_error, %r_error_status, 0;",
        file=ptx_bridge,
    )  # The numba ABI returns non-zero value if an error is raised

    print(
        "    and.pred %p_no_error, %p_no_error, %p_no_func_error;",
        file=ptx_bridge,
    )

    print("  }", file=ptx_bridge)


def call_func(ptx_bridge, dataset_setups):
    """
    Define call_func
    This function calls to appropriate model function
    """

    print(".visible .func (.param .b32 func_retval) call_func(", file=ptx_bridge)
    print("  .param .b64 call_func_param_0,", file=ptx_bridge)  # double* out
    print("  .param .b64 call_func_param_1,", file=ptx_bridge)  # double* params
    print("  .param .b64 call_func_param_2,", file=ptx_bridge)  # double* ind_vars
    print("  .param .b32 call_func_param_3", file=ptx_bridge)  # int64_t i_dataset
    print(")", file=ptx_bridge)
    print("{", file=ptx_bridge)
    print("  .reg .b32 %r_retval;", file=ptx_bridge)
    print("  .reg .b64 %rd_params_addr;", file=ptx_bridge)
    print("  .reg .b64 %rd_ind_vars_addr;", file=ptx_bridge)
    print("  .reg .b32 %rd_i_dataset;", file=ptx_bridge)
    print("  .reg .pred %p_no_func_error;", file=ptx_bridge)
    print("  .reg .f64 %fd_out_val;", file=ptx_bridge)
    print("  .reg .b64 %rd_out_addr;", file=ptx_bridge)
    print("  .reg .b32 %r_error_status;", file=ptx_bridge)

    print("  ld.param.u64 %rd_out_addr, [call_func_param_0];", file=ptx_bridge)
    print("  ld.param.u64 %rd_params_addr, [call_func_param_1];", file=ptx_bridge)
    print("  ld.param.u64 %rd_ind_vars_addr, [call_func_param_2];", file=ptx_bridge)
    print("  ld.param.u32 %rd_i_dataset, [call_func_param_3];", file=ptx_bridge)

    print("  ts: .branchtargets ", end="", file=ptx_bridge)
    print(
        *(
            f"$L_RUN_{i_dataset}"
            for i_dataset, _dataset_key in enumerate(dataset_setups)
        ),
        sep=", ",
        end="",
        file=ptx_bridge,
    )
    print(";", file=ptx_bridge)
    print("  brx.idx.uni %rd_i_dataset, ts;", file=ptx_bridge)

    for i_dataset, dataset_setup in enumerate(dataset_setups.values()):
        call_model_function(ptx_bridge, i_dataset, dataset_setup)

    print("$L_RETURN:", file=ptx_bridge)

    print("  ld.f64 %fd_out_val, [%rd_out_addr];", file=ptx_bridge)

    print(
        "  testp.finite.f64 %p_no_func_error, %fd_out_val;",
        file=ptx_bridge,
    )

    print(
        "  setp.eq.and.s32 %p_no_func_error, %r_error_status, 0, %p_no_func_error;",
        file=ptx_bridge,
    )  # The numba ABI returns non-zero value if an error is raised
    print("  selp.u32 %r_retval, 1, 0, %p_no_func_error;", file=ptx_bridge)
    print("  st.param.b32 [func_retval], %r_retval;", file=ptx_bridge)
    print("  ret.uni;", file=ptx_bridge)
    print("}", file=ptx_bridge)


def call_model_function(ptx_bridge, i_dataset, dataset_setup):
    "Call the model function"

    print(f"$L_RUN_{i_dataset}:", file=ptx_bridge)

    print("  {", file=ptx_bridge)

    for i_arg, _arg_info in enumerate(dataset_setup.arg_infos, start=1):
        print(f"    .reg .f64 %fd_value_{i_arg};", file=ptx_bridge)

    for i_arg, arg_info in enumerate(dataset_setup.arg_infos, start=1):
        if arg_info.is_param:
            # The argument is a parameter
            print(
                f"    ld.f64 %fd_value_{i_arg}, [%rd_params_addr+{arg_info.offset}];",
                file=ptx_bridge,
            )
        else:
            # The argument is an independent variable
            print(
                f"    ld.f64 %fd_value_{i_arg}, [%rd_ind_vars_addr+{arg_info.offset}];",
                file=ptx_bridge,
            )

    print("    {", file=ptx_bridge)
    print("      .param .b64 param_0;", file=ptx_bridge)

    for i_arg, _arg_info in enumerate(dataset_setup.arg_infos, start=1):
        print(f"      .param .b64 param_{i_arg};", file=ptx_bridge)

    print("      .param .b32 retval;", file=ptx_bridge)

    print("      st.param.b64 [param_0], %rd_out_addr;", file=ptx_bridge)

    for i_arg, _arg_info in enumerate(dataset_setup.arg_infos, start=1):
        print(
            f"      st.param.f64 [param_{i_arg}], %fd_value_{i_arg};",
            file=ptx_bridge,
        )

    print(
        f"      call (retval), {dataset_setup.func_name}, (param_0",
        file=ptx_bridge,
        end="",
    )
    for i_arg, _arg_info in enumerate(dataset_setup.arg_infos, start=1):
        print(f", param_{i_arg}", file=ptx_bridge, end="")

    print(");", file=ptx_bridge)

    print("      ld.param.b32 %r_error_status, [retval];", file=ptx_bridge)
    print("    }", file=ptx_bridge)
    print("  }", file=ptx_bridge)
    print("  bra.uni $L_RETURN;", file=ptx_bridge)

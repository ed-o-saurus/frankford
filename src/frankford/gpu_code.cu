// Copyright (C) 2024 Edward F. Behn, Jr.

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// The C code is based on MP-fit (https://pages.physics.wisc.edu/~craigm/idl/cmpfit.html) developed by Craig Markwardt.

// Translated from MINPACK-1 in FORTRAN, Apr-Jul 1998, CM
// Copyright (C) 1997-2002, Craig Markwardt
// This software is provided as is without any warranty whatsoever.
// Permission to use, copy, modify, and distribute modified or
// unmodified copies is granted, provided this copyright and disclaimer
// are included unchanged.

typedef unsigned long long uint64_t;
typedef uint64_t uintptr_t;
typedef int int32_t;
typedef unsigned char uint8_t;

#ifndef _BITS_STDINT_INTN_H
typedef long long int64_t;
typedef char int8_t;
#endif

#define DBL_EPSILON 0x1p-52  // 2.220446049250313e-16
#define LDBL_EPSILON 0x1p-26 // 1.4901161193847656e-08
#define DBL_MIN 0x1p-1022    // 2.2250738585072014e-308

#define SIDE_AUTO 0 // one-sided derivative computed automatically
#define SIDE_POS 1  // one-sided derivative (f(x+h) - f(x)  ) / h
#define SIDE_NEG -1 // one-sided derivative (f(x)   - f(x-h)) / h
#define SIDE_BOTH 2 // two-sided derivative (f(x+h) - f(x-h)) / (2 * h)

#define RESULT_ERR_UNKNOWN -1   // Other error
#define RESULT_ERR_DOF -2       // Not enough degrees of freedom
#define RESULT_ERR_USER_FUNC -3 // Error from user function
#define RESULT_OK_CHI_SQ 1      // Convergence in chi-square value
#define RESULT_OK_PAR 2         // Convergence in parameter value
#define RESULT_OK_BOTH 3        // Both RESULT_OK_PAR and RESULT_OK_CHI_SQ hold
#define RESULT_OK_DIR 4         // Convergence in orthogonality
#define RESULT_MAX_ITER 5       // Maximum number of iterations reached
#define RESULT_FTOL 6           // ftol is too small; no further improvement
#define RESULT_XTOL 7           // xtol is too small; no further improvement
#define RESULT_GTOL 8           // gtol is too small; no further improvement

extern "C"
{
    extern __device__ bool setup_params(double *params);
    extern __device__ bool update_params(double *params, int64_t updated_position);
    extern __device__ bool call_func(double *out, double *params, double *ind_vars, int64_t i_dataset);
}

__device__ int8_t calculate(struct CallConfig *call_config,
                            double *fvec,
                            double *qtf,
                            double *params_all,
                            double *params,
                            double *params_new,
                            double *fjac,
                            double *diag,
                            double *wa1,
                            double *wa2,
                            double *wa3,
                            double *wa4,
                            int64_t *ipvt,
                            double *ind_vars,
                            double *fixed);
__device__ struct Point *read_dataset(struct CallConfig *call_config, int64_t i_point, struct Dataset *dataset, double *ind_vars);
__device__ float read_param(struct CallConfig *call_config, struct ParamArray *param_array);
__device__ bool get_deviates(struct CallConfig *call_config, double *fvec, double *params, double *ind_vars);
__device__ bool fdjac(struct CallConfig *call_config, double *fjac, double *params, double *fvec, double *ind_vars, double *wa1, double *wa2, double *wa4);
__device__ void qrfac(struct CallConfig *call_config, double *a, int64_t *ipvt, double *rdiag, double *acnorm, double *wa);
__device__ void lmpar(struct CallConfig *call_config, double *r, int64_t *ipvt, double *diag, double *qtb, double delta, double &par, double *x, double *sdiag, double *wa3, double *wa4);
__device__ void qrsolv(struct CallConfig *call_config, double *r, int64_t *ipvt, double *diag, double *qtb, double *x, double *sdiag, double *wa);
__device__ void covar(struct CallConfig *call_config, double *r, int64_t *ipvt, double *wa);

struct Point
{
    double value;
    double uncertainty;
};

struct ParamArray
{
    double *values;
    int64_t *out_offsets;
};

struct FreeParam
{
    struct ParamArray *init_value;
    struct ParamArray *lower;
    struct ParamArray *upper;
    double step;
    double relstep;
    int8_t side;
};

struct IndVar
{
    double *values;
    int64_t *fit_offsets;
    int64_t *out_offsets;
};

struct Dataset
{
    int64_t fit_size;
    int64_t *points_fit_offsets;
    int64_t *points_out_offsets;
    struct Point *points;
    int64_t n_ind_vars;
    struct IndVar **ind_vars;
};

struct PositionOffset
{
    int64_t position;
    uintptr_t offset;
};

struct PositionsOffset
{
    int64_t position1;
    int64_t position2;
    uintptr_t offset;
};

struct CallConfig
{
    int64_t i_thread;
    int64_t n_free_param;
    struct FreeParam **free_params;
    int64_t n_fixed_param;
    struct ParamArray **fixed_params;
    int64_t n_tied_param;
    int64_t n_param;
    int64_t n_dataset;
    struct Dataset **datasets;
    int64_t m; // Number of finite points
    double ftol;
    double xtol;
    double gtol;
    double stepfactor;
    double covtol;
    int64_t maxiter;
    bool douserscale;
    int64_t iter;
    double fnorm;
    double fnorm1;
    double orig_chi_sq;
};

const double zero = 0.0;
const double p0001 = 1.0e-4;
const double p001 = 0.001;
const double p05 = 0.05;
const double p1 = 0.1;
const double p25 = 0.25;
const double p5 = 0.5;
const double p75 = 0.75;
const double one = 1.0;

extern "C" __device__ int32_t fit(
    int32_t *_, // Unused return value
    int64_t i_thread,
    int64_t n_free_param, struct FreeParam **free_params,
    int64_t n_fixed_param, struct ParamArray **fixed_params,
    int64_t n_tied_param,
    int64_t n_dataset, struct Dataset **datasets,
    double ftol, double xtol, double gtol, double stepfactor, double covtol, int64_t maxiter, uint8_t douserscale,
    uintptr_t result_offset, uintptr_t chi_sq_offset, uintptr_t dof_offset, uintptr_t num_iter_offset, uintptr_t orig_chi_sq_offset,
    int64_t n_returned_param, struct PositionOffset **returned_params,
    int64_t n_returned_uncertainty, struct PositionOffset **returned_uncertainties,
    int64_t n_returned_covar, struct PositionsOffset **returned_covars,
    uintptr_t ret_value,
    double *fvec,
    double *qtf,
    double *params_all,
    double *params,
    double *params_new,
    double *fjac,
    double *diag,
    double *wa1,
    double *wa2,
    double *wa3,
    double *wa4,
    int64_t *ipvt,
    double *ind_vars,
    double *fixed)
{
    *_ = 0;

    int64_t n_param = n_free_param + n_fixed_param + n_tied_param;

    struct CallConfig call_config;

    call_config.i_thread = i_thread;
    call_config.n_free_param = n_free_param;
    call_config.free_params = free_params;
    call_config.n_fixed_param = n_fixed_param;
    call_config.fixed_params = fixed_params;
    call_config.n_tied_param = n_tied_param;
    call_config.n_param = n_param;
    call_config.n_dataset = n_dataset;
    call_config.datasets = datasets;
    call_config.ftol = ftol;
    call_config.xtol = xtol;
    call_config.gtol = gtol;
    call_config.stepfactor = stepfactor;
    call_config.covtol = covtol;
    call_config.maxiter = maxiter;
    call_config.douserscale = douserscale; // Cast to boolean
    call_config.iter = 1;
    call_config.fnorm = -one;
    call_config.fnorm1 = -one;

    uint8_t result = calculate(&call_config,
                               fvec,
                               qtf,
                               params_all,
                               params,
                               params_new,
                               fjac,
                               diag,
                               wa1,
                               wa2,
                               wa3,
                               wa4,
                               ipvt,
                               ind_vars,
                               fixed);

    uint64_t m = call_config.m;

    *((double *)(ret_value + orig_chi_sq_offset)) = call_config.orig_chi_sq;
    *((int64_t *)(ret_value + num_iter_offset)) = call_config.iter;
    *((int8_t *)(ret_value + result_offset)) = result;

    if (result > 0)
    {
        double bestnorm = fmax(call_config.fnorm, call_config.fnorm1);
        *((double *)(ret_value + chi_sq_offset)) = bestnorm * bestnorm;

        *((int64_t *)(ret_value + dof_offset)) = m - n_free_param;

        memcpy(params + call_config.n_free_param, fixed, call_config.n_fixed_param * sizeof(double));
        if (!setup_params(params))
        {
            *((int8_t *)(ret_value + result_offset)) = RESULT_ERR_USER_FUNC;

            // Signal that no Python exception occurred
            return 0;
        }

        for (int64_t i_returned_param = 0; i_returned_param < n_returned_param; i_returned_param++)
        {
            struct PositionOffset *returned_param = returned_params[i_returned_param];
            *((double *)(ret_value + returned_param->offset)) = params[returned_param->position];
        }

        if (n_returned_uncertainty > 0 || n_returned_covar > 0)
        {
            covar(&call_config, fjac, ipvt, wa2);

            for (int64_t i_returned_uncertainty = 0; i_returned_uncertainty < n_returned_uncertainty; i_returned_uncertainty++)
            {
                struct PositionOffset *returned_uncertainty = returned_uncertainties[i_returned_uncertainty];
                double cc = fjac[(returned_uncertainty->position) * (m + 1)];

                *((double *)(ret_value + returned_uncertainty->offset)) = (cc > zero) ? sqrt(cc) : zero;
            }

            for (int64_t i_returned_covar = 0; i_returned_covar < n_returned_covar; i_returned_covar++)
            {
                struct PositionsOffset *returned_covar = returned_covars[i_returned_covar];
                *((double *)(ret_value + returned_covar->offset)) = fjac[(returned_covar->position1) * m + (returned_covar->position2)];
            }
        }
    }

    // Signal that no Python exception occurred
    return 0;
}

__device__ int8_t calculate(struct CallConfig *call_config,
                            double *fvec,
                            double *qtf,
                            double *params_all,
                            double *params,
                            double *params_new,
                            double *fjac,
                            double *diag,
                            double *wa1,
                            double *wa2,
                            double *wa3,
                            double *wa4,
                            int64_t *ipvt,
                            double *ind_vars,
                            double *fixed)
{
    double actred, dirder, gnorm;
    double pnorm, prered, ratio;
    double xnorm;

    double delta = zero;
    double par = zero;

    int64_t m = 0;
    for (int64_t i_dataset = 0; i_dataset < call_config->n_dataset; i_dataset++)
    {
        struct Dataset *dataset = call_config->datasets[i_dataset];

        int64_t out_offset = dataset->points_out_offsets[call_config->i_thread];

        for (int64_t i_point = 0; i_point < dataset->fit_size; i_point++)
        {
            int64_t fit_offset = dataset->points_fit_offsets[i_point];

            if (isfinite(dataset->points[out_offset + fit_offset].value))
                m++;
        }
    }

    if (m < call_config->n_free_param)
        return RESULT_ERR_DOF;

    call_config->m = m;

    for (int64_t j = 0; j < call_config->n_free_param; j++)
        params_all[j] = read_param(call_config, call_config->free_params[j]->init_value);

    for (int64_t j = 0; j < call_config->n_fixed_param; j++)
        fixed[j] = params_all[call_config->n_free_param + j] = read_param(call_config, call_config->fixed_params[j]);

    if (!setup_params(params_all))
        return RESULT_ERR_USER_FUNC;

    if (!get_deviates(call_config, fvec, params_all, ind_vars))
        return RESULT_ERR_USER_FUNC;

    call_config->fnorm = norm(call_config->m, fvec);
    call_config->orig_chi_sq = call_config->fnorm * call_config->fnorm;

    // Make a new copy
    // Transfer parameters to 'params'
    memcpy(params, params_all, call_config->n_param * sizeof(double));

    while (true) // start of outer loop
    {
        memcpy(params_new, params, call_config->n_param * sizeof(double));

        if (!fdjac(call_config, fjac, params_new, fvec, ind_vars, wa1, wa2, wa4))
            return RESULT_ERR_USER_FUNC;

        for (int64_t j = 0; j < call_config->n_free_param; j++)
        {
            // Determine if any of the parameters are pegged at the limits
            bool lpegged = params[j] == read_param(call_config, call_config->free_params[j]->lower);
            bool upegged = params[j] == read_param(call_config, call_config->free_params[j]->upper);
            float sum = zero;

            // If the parameter is pegged at a limit, compute the gradient direction
            if (lpegged || upegged)
                for (int64_t i = 0, ij = j * call_config->m; i < call_config->m; i++, ij++)
                    sum += fvec[i] * fjac[ij];

            // If pegged at lower limit and gradient is toward negative then reset gradient to zero
            // If pegged at upper limit and gradient is toward positive then reset gradient to zero
            if ((lpegged && sum > 0) || (upegged && sum < 0))
                for (int64_t i = 0, ij = j * call_config->m; i < call_config->m; i++, ij++)
                    fjac[ij] = zero;
        }

        // Compute the QR factorization of the jacobian
        qrfac(call_config, fjac, ipvt, wa1, wa2, wa3);

        // on the first iteration and if mode is 1, scale according
        // to the norms of the columns of the initial jacobian.
        if (call_config->iter == 1)
        {
            if (!call_config->douserscale)
                for (int64_t j = 0; j < call_config->n_free_param; j++)
                    if (wa2[j] == zero)
                        diag[j] = one;
                    else
                        diag[j] = wa2[j];

            // on the first iteration, calculate the norm of the scaled x
            // and initialize the step bound delta.
            for (int64_t j = 0; j < call_config->n_free_param; j++)
                wa3[j] = diag[j] * params[j];

            xnorm = norm(call_config->n_free_param, wa3);
            delta = call_config->stepfactor * xnorm;
            if (delta == zero)
                delta = call_config->stepfactor;
        }

        // form (q transpose)*fvec and store the first n components in qtf.
        for (int64_t i = 0; i < call_config->m; i++)
            wa4[i] = fvec[i];

        for (int64_t j = 0, jj = 0; j < call_config->n_free_param; j++, jj += (call_config->m) + 1)
        {
            double temp3 = fjac[jj];
            if (temp3 != zero)
            {
                double sum = zero;
                for (int64_t i = j, ij = jj; i < call_config->m; i++, ij++)
                    sum += fjac[ij] * wa4[i];

                double temp = -sum / temp3;
                for (int64_t i = j, ij = jj; i < call_config->m; i++, ij++)
                    wa4[i] += fjac[ij] * temp;
            }

            fjac[jj] = wa1[j];

            qtf[j] = wa4[j];
        }

        // (From this point on, only the square matrix, consisting of the triangle of R, is needed.)

        // compute the norm of the scaled gradient.
        gnorm = zero;
        if (call_config->fnorm != zero)
            for (int64_t j = 0, jj = 0; j < call_config->n_free_param; j++, jj += call_config->m)
            {
                int64_t l = ipvt[j];
                if (wa2[l] != zero)
                {
                    double sum = zero;
                    for (int64_t i = 0, ij = jj; i <= j; i++, ij++)
                        sum += fjac[ij] * (qtf[i] / call_config->fnorm);

                    gnorm = fmax(gnorm, fabs(sum / wa2[l]));
                }
            }

        // test for convergence of the gradient norm.
        if (gnorm <= call_config->gtol)
            return RESULT_OK_DIR;

        if (call_config->maxiter == 0)
            return RESULT_MAX_ITER;

        // rescale if necessary.
        if (!call_config->douserscale)
            for (int64_t j = 0; j < call_config->n_free_param; j++)
                diag[j] = fmax(diag[j], wa2[j]);

        do // beginning of the inner loop.
        {
            // determine the Levenberg-Marquardt parameter.
            // par is pass-by-reference
            lmpar(call_config, fjac, ipvt, diag, qtf, delta, par, wa1, wa2, wa3, wa4);

            // store the direction p and x + p. calculate the norm of p.
            for (int64_t j = 0; j < call_config->n_free_param; j++)
                wa1[j] *= -1.0;

            double alpha = one;
            for (int64_t j = 0; j < call_config->n_free_param; j++)
            {
                float lower = read_param(call_config, call_config->free_params[j]->lower);
                float upper = read_param(call_config, call_config->free_params[j]->upper);

                // Determine if any of the parameters are pegged at the limits
                bool lpegged = params[j] == lower;
                bool upegged = params[j] == upper;
                int dwa1 = fabs(wa1[j]) > DBL_EPSILON;

                if (lpegged && (wa1[j] < 0))
                    wa1[j] = 0;

                if (upegged && (wa1[j] > 0))
                    wa1[j] = 0;

                if (dwa1 && (params[j] + wa1[j]) < lower)
                    alpha = fmin(alpha, (lower - params[j]) / wa1[j]);

                if (dwa1 && (params[j] + wa1[j]) > upper)
                    alpha = fmin(alpha, (upper - params[j]) / wa1[j]);
            }

            for (int64_t j = 0; j < call_config->n_free_param; j++)
            {
                float lower = read_param(call_config, call_config->free_params[j]->lower);
                float upper = read_param(call_config, call_config->free_params[j]->upper);

                double sgnu, sgnl;
                double ulim1, llim1;

                wa1[j] *= alpha;
                wa2[j] = params[j] + wa1[j];

                // Adjust the output values.  If the step put us exactly on a boundary, make sure it is exact.
                sgnu = (upper >= 0) ? (+1) : (-1);
                sgnl = (lower >= 0) ? (+1) : (-1);
                ulim1 = upper * (1 - sgnu * DBL_EPSILON) - ((upper == 0) ? (DBL_EPSILON) : 0);
                llim1 = lower * (1 + sgnl * DBL_EPSILON) + ((lower == 0) ? (DBL_EPSILON) : 0);

                if (wa2[j] >= ulim1)
                    wa2[j] = upper;

                if (wa2[j] <= llim1)
                    wa2[j] = lower;
            }

            for (int64_t j = 0; j < call_config->n_free_param; j++)
                wa3[j] = diag[j] * wa1[j];

            pnorm = norm(call_config->n_free_param, wa3);

            // on the first iteration, adjust the initial step bound.
            if (call_config->iter == 1)
                delta = fmin(delta, pnorm);

            // evaluate the function at x + p and calculate its norm.
            memcpy(params_new, wa2, call_config->n_free_param * sizeof(double));
            memcpy(params_new + call_config->n_free_param, fixed, call_config->n_fixed_param * sizeof(double));
            if (!setup_params(params_new))
                return RESULT_ERR_USER_FUNC;

            if (!get_deviates(call_config, wa4, params_new, ind_vars))
                return RESULT_ERR_USER_FUNC;

            call_config->fnorm1 = norm(call_config->m, wa4);

            // Compute the scaled actual reduction.
            actred = -one;
            if ((p1 * call_config->fnorm1) < call_config->fnorm)
            {
                double temp = call_config->fnorm1 / call_config->fnorm;
                actred = one - temp * temp;
            }

            // Compute the scaled predicted reduction and the scaled directional derivative.
            for (int64_t j = 0, jj = 0; j < call_config->n_free_param; j++, jj += call_config->m)
            {
                wa3[j] = zero;
                double temp = wa1[ipvt[j]];
                for (int64_t i = 0, ij = jj; i <= j; i++, ij++)
                    wa3[i] += fjac[ij] * temp;
            }

            // Remember, alpha is the fraction of the full LM step actually taken
            double temp1 = norm(call_config->n_free_param, wa3) * alpha / call_config->fnorm;
            double temp2 = (sqrt(alpha * par) * pnorm) / call_config->fnorm;
            prered = temp1 * temp1 + (temp2 * temp2) / p5;
            dirder = -(temp1 * temp1 + temp2 * temp2);

            // compute the ratio of the actual to the predicted reduction.
            ratio = zero;
            if (prered != zero)
                ratio = actred / prered;

            // update the step bound.
            if (ratio <= p25)
            {
                double temp;
                if (actred >= zero)
                    temp = p5;
                else
                    temp = p5 * dirder / (dirder + p5 * actred);

                if ((p1 * call_config->fnorm1 >= call_config->fnorm) || (temp < p1))
                    temp = p1;

                delta = temp * fmin(delta, pnorm / p1);
                par /= temp;
            }
            else if ((par == zero) || (ratio >= p75))
            {
                delta = pnorm / p5;
                par *= p5;
            }

            // test for successful iteration.
            if (ratio >= p0001)
            {
                // successful iteration. update x, fvec, and their norms.
                for (int64_t j = 0; j < call_config->n_free_param; j++)
                {
                    params[j] = wa2[j];
                    wa2[j] = diag[j] * params[j];
                }

                memcpy(fvec, wa4, call_config->m * sizeof(double));

                xnorm = norm(call_config->n_free_param, wa2);

                call_config->fnorm = call_config->fnorm1;
                call_config->iter++;
            }

            bool ok_par = delta <= call_config->xtol * xnorm;
            bool ok_chi_sq = (fabs(actred) <= call_config->ftol) && (prered <= call_config->ftol) && (p5 * ratio <= one);

            if (ok_par && ok_chi_sq)
                return RESULT_OK_BOTH;

            if (ok_par)
                return RESULT_OK_PAR;

            if (ok_chi_sq)
                return RESULT_OK_CHI_SQ;

            if (gnorm <= DBL_EPSILON)
                return RESULT_GTOL;

            if (delta <= DBL_EPSILON * xnorm)
                return RESULT_XTOL;

            if ((fabs(actred) <= DBL_EPSILON) && (prered <= DBL_EPSILON) && (p5 * ratio <= one))
                return RESULT_FTOL;

            // tests for termination and stringent tolerances.
            if (call_config->iter >= call_config->maxiter)
                return RESULT_MAX_ITER; // Too many iterations
        } // end of the inner loop.
        while (ratio < p0001);
    } // end of the outer loop
}

__device__ float read_param(struct CallConfig *call_config, struct ParamArray *param_array)
{
    return param_array->values[param_array->out_offsets[call_config->i_thread]];
}

__device__ struct Point *read_dataset(struct CallConfig *call_config, int64_t i_point, struct Dataset *dataset, double *ind_vars)
{
    int64_t point_out_offset = dataset->points_out_offsets[call_config->i_thread];
    int64_t point_fit_offset = dataset->points_fit_offsets[i_point];
    struct Point *point = dataset->points + (point_out_offset + point_fit_offset);

    if (!isfinite(point->value))
        return point;

    for (int64_t i_ind_var = 0; i_ind_var < dataset->n_ind_vars; i_ind_var++)
    {
        struct IndVar *ind_var = dataset->ind_vars[i_ind_var];

        int64_t ind_var_out_offset = ind_var->out_offsets[call_config->i_thread];
        int64_t ind_var_fit_offset = ind_var->fit_offsets[i_point];

        ind_vars[i_ind_var] = ind_var->values[ind_var_out_offset + ind_var_fit_offset];
    }

    return point;
}

__device__ bool get_deviates(struct CallConfig *call_config, double *fvec, double *params, double *ind_vars)
{
    int64_t m = 0;
    for (int64_t i_dataset = 0; i_dataset < call_config->n_dataset; i_dataset++)
    {
        struct Dataset *dataset = call_config->datasets[i_dataset];

        for (int64_t i_point = 0; i_point < dataset->fit_size; i_point++)
        {
            struct Point *point = read_dataset(call_config, i_point, dataset, ind_vars);
            if (!isfinite(point->value))
                continue;

            double value;
            if (!call_func(&value, params, ind_vars, i_dataset))
                return false;

            fvec[m++] = (value - point->value) / point->uncertainty;
        }
    }

    return true;
}

__device__ bool fdjac(struct CallConfig *call_config, double *fjac, double *params, double *fvec, double *ind_vars, double *wa1, double *wa2, double *wa4)
{
    int64_t m;
    if (!setup_params(params))
        return false;

    bool center_needed = false;
    for (int64_t j = 0; j < call_config->n_free_param; j++)
        if (call_config->free_params[j]->side != SIDE_BOTH)
        {
            center_needed = true;
            break;
        }

    if (center_needed)
    {
        m = 0;
        for (int64_t i_dataset = 0; i_dataset < call_config->n_dataset; i_dataset++)
        {
            struct Dataset *dataset = call_config->datasets[i_dataset];

            for (int64_t i_point = 0; i_point < dataset->fit_size; i_point++)
            {
                struct Point *point = read_dataset(call_config, i_point, dataset, ind_vars);
                if (!isfinite(point->value))
                    continue;

                double value;
                if (!call_func(&value, params, ind_vars, i_dataset))
                    return false;

                wa4[m++] = value;
            }
        }
    }

    int64_t ij = 0;
    for (int64_t j = 0; j < call_config->n_free_param; j++)
    {
        FreeParam *free_param = call_config->free_params[j];
        double h, temp = fabs(params[j]);

        if (free_param->step > 0.0)
            h = free_param->step;
        else
            h = (free_param->relstep) * temp;

        if (h == zero)
            h = LDBL_EPSILON;

        switch (free_param->side)
        {
        case SIDE_NEG:
            h = -h;
            break;
        case SIDE_AUTO:
            if (params[j] + h > read_param(call_config, free_param->upper))
                h = -h;
            break;
        case SIDE_BOTH:
            memcpy(wa1, params, call_config->n_param * sizeof(double));
            wa1[j] -= h;
            if (!update_params(wa1, j))
                return false;

            m = 0;
            for (int64_t i_dataset = 0; i_dataset < call_config->n_dataset; i_dataset++)
            {
                struct Dataset *dataset = call_config->datasets[i_dataset];

                for (int64_t i_point = 0; i_point < dataset->fit_size; i_point++)
                {
                    struct Point *point = read_dataset(call_config, i_point, dataset, ind_vars);
                    if (!isfinite(point->value))
                        continue;

                    double value;
                    if (!call_func(&value, wa1, ind_vars, i_dataset))
                        return false;

                    wa2[m++] = value;
                }
            }

            break;
        }

        memcpy(wa1, params, call_config->n_param * sizeof(double));
        wa1[j] += h;
        if (!update_params(wa1, j))
            return false;

        m = 0;
        for (int64_t i_dataset = 0; i_dataset < call_config->n_dataset; i_dataset++)
        {
            struct Dataset *dataset = call_config->datasets[i_dataset];

            for (int64_t i_point = 0; i_point < dataset->fit_size; i_point++)
            {
                struct Point *point = read_dataset(call_config, i_point, dataset, ind_vars);
                if (!isfinite(point->value))
                    continue;

                double value;
                if (!call_func(&value, wa1, ind_vars, i_dataset))
                    return false;

                if (free_param->side == SIDE_BOTH)
                    fjac[ij++] = (value - wa2[m++]) / (2 * point->uncertainty * h);
                else
                    fjac[ij++] = (value - wa4[m++]) / (point->uncertainty * h);
            }
        }
    }

    return true;
}

__device__ void qrfac(struct CallConfig *call_config, double *a, int64_t *ipvt, double *rdiag, double *acnorm, double *wa)
{
    // compute the initial column norms and initialize several arrays.
    for (int64_t j = 0, ij = 0; j < call_config->n_free_param; j++, ij += call_config->m)
    {
        wa[j] = rdiag[j] = acnorm[j] = norm(call_config->m, a + ij);
        ipvt[j] = j;
    }

    // reduce a to r with householder transformations.
    for (int64_t j = 0; j < call_config->n_free_param; j++)
    {
        // bring the column of largest norm into the pivot position.
        int64_t kmax = j;
        for (int64_t k = j; k < call_config->n_free_param; k++)
            if (rdiag[k] > rdiag[kmax])
                kmax = k;

        if (kmax != j)
        {
            for (int64_t i = 0, ij = (call_config->m) * j, jj = (call_config->m) * kmax; i < call_config->m; i++, ij++, jj++)
            {
                // SWAP
                double temp = a[ij];
                a[ij] = a[jj];
                a[jj] = temp;
            }

            rdiag[kmax] = rdiag[j];
            wa[kmax] = wa[j];

            // SWAP
            int64_t k = ipvt[j];
            ipvt[j] = ipvt[kmax];
            ipvt[kmax] = k;
        }

        // compute the householder transformation to reduce the
        // j-th column of a to a multiple of the j-th unit vector.
        int64_t jj = j + (call_config->m) * j;
        double ajnorm = norm((call_config->m) - j, a + jj);
        if (ajnorm != zero)
        {
            if (a[jj] < zero)
                ajnorm = -ajnorm;

            for (int64_t i = j, ij = jj; i < call_config->m; i++, ij++)
                a[ij] /= ajnorm;

            a[jj] += one;
            // apply the transformation to the remaining columns and update the norms.
            int64_t jp1 = j + 1;
            if (jp1 < call_config->n_free_param)
                for (int64_t k = jp1; k < call_config->n_free_param; k++)
                {
                    double sum = zero;
                    for (int64_t i = j, ij = j + (call_config->m) * k, jj = j + (call_config->m) * j; i < call_config->m; i++, ij++, jj++)
                        sum += a[jj] * a[ij];

                    double temp = sum / a[j + (call_config->m) * j];
                    for (int64_t i = j, ij = j + (call_config->m) * k, jj = j + (call_config->m) * j; i < call_config->m; i++, ij++, jj++)
                        a[ij] -= temp * a[jj];

                    if (rdiag[k] != zero)
                    {
                        temp = a[j + (call_config->m) * k] / rdiag[k];
                        temp = fmax(zero, one - temp * temp);
                        rdiag[k] *= sqrt(temp);
                        temp = rdiag[k] / wa[k];

                        if ((p05 * temp * temp) <= DBL_EPSILON)
                        {
                            rdiag[k] = norm((call_config->m) - j - 1, a + jp1 + (call_config->m) * k);
                            wa[k] = rdiag[k];
                        }
                    }
                }
        }

        rdiag[j] = -ajnorm;
    }
}

__device__ void lmpar(struct CallConfig *call_config, double *r, int64_t *ipvt, double *diag, double *qtb, double delta, double &par, double *x, double *sdiag, double *wa3, double *wa4)
{
    double dxnorm, fp, gnorm, parc, parl, paru;

    // compute and store in x the gauss-newton direction. if the
    // jacobian is rank-deficient, obtain a least squares solution.
    int64_t nsing = call_config->n_free_param;
    for (int64_t j = 0, jj = 0; j < call_config->n_free_param; j++, jj += (call_config->m) + 1)
    {
        wa3[j] = qtb[j];
        if ((r[jj] == zero) && (nsing == call_config->n_free_param))
            nsing = j;

        if (nsing < call_config->n_free_param)
            wa3[j] = zero;
    }

    if (nsing >= 1)
        for (int64_t k = 0; k < nsing; k++)
        {
            int64_t j = nsing - k - 1;
            wa3[j] /= r[j + (call_config->m) * j];
            double temp = wa3[j];
            for (int64_t i = 0, ij = (call_config->m) * j; i < j; i++, ij++)
                wa3[i] -= r[ij] * temp;
        }

    for (int64_t j = 0; j < call_config->n_free_param; j++)
        x[ipvt[j]] = wa3[j];

    // initialize the iteration counter.
    // evaluate the function at the origin, and test
    // for acceptance of the gauss-newton direction.
    for (int64_t j = 0; j < call_config->n_free_param; j++)
        wa4[j] = diag[j] * x[j];

    dxnorm = norm(call_config->n_free_param, wa4);
    fp = dxnorm - delta;
    if (fp <= p1 * delta)
    {
        par = zero;
        return;
    }

    // if the jacobian is not rank deficient, the newton
    // step provides a lower bound, parl, for the zero of
    // the function. otherwise set this bound to zero.
    parl = zero;
    if (nsing >= call_config->n_free_param)
    {
        for (int64_t j = 0; j < call_config->n_free_param; j++)
        {
            int64_t l = ipvt[j];
            wa3[j] = diag[l] * (wa4[l] / dxnorm);
        }

        for (int64_t j = 0, jj = 0; j < call_config->n_free_param; j++, jj += call_config->m)
        {
            double sum = zero;
            for (int64_t i = 0, ij = jj; i < j; i++, ij++)
                sum += r[ij] * wa3[i];

            wa3[j] = (wa3[j] - sum) / r[j + (call_config->m) * j];
        }

        double temp = norm(call_config->n_free_param, wa3);
        parl = ((fp / delta) / temp) / temp;
    }

    // calculate an upper bound, paru, for the zero of the function.
    for (int64_t j = 0, jj = 0; j < call_config->n_free_param; j++, jj += call_config->m)
    {
        double sum = zero;
        for (int64_t i = 0, ij = jj; i <= j; i++, ij++)
            sum += r[ij] * qtb[i];

        wa3[j] = sum / diag[ipvt[j]];
    }

    gnorm = norm(call_config->n_free_param, wa3);
    paru = gnorm / delta;
    if (paru == zero)
        paru = DBL_MIN / fmin(delta, p1);

    // if the input par lies outside of the interval (parl, paru),
    // set par to the closer endpoint.
    par = fmax(par, parl);
    par = fmin(par, paru);
    if (par == zero)
        par = gnorm / dxnorm;

    // beginning of an iteration.
    for (int64_t iter = 0; true; iter++)
    {
        // evaluate the function at the current value of par.
        if (par == zero)
            par = fmax(DBL_MIN, p001 * paru);

        double temp = sqrt(par);
        for (int64_t j = 0; j < call_config->n_free_param; j++)
            wa3[j] = temp * diag[j];

        qrsolv(call_config, r, ipvt, wa3, qtb, x, sdiag, wa4);
        for (int64_t j = 0; j < call_config->n_free_param; j++)
            wa4[j] = diag[j] * x[j];

        dxnorm = norm(call_config->n_free_param, wa4);
        temp = fp;
        fp = dxnorm - delta;

        // if the function is small enough, accept the current value
        // of par. also test for the exceptional cases where parl
        // is zero or the number of iterations has reached 10.
        if ((fabs(fp) <= p1 * delta) || ((parl == zero) && (fp <= temp) && (temp < zero)) || (iter == 9))
            return;

        // compute the newton correction.
        for (int64_t j = 0; j < call_config->n_free_param; j++)
        {
            int64_t l = ipvt[j];
            wa3[j] = diag[l] * (wa4[l] / dxnorm);
        }

        for (int64_t j = 0, jj = 0; j < call_config->n_free_param; j++, jj += call_config->m)
        {
            wa3[j] /= sdiag[j];
            temp = wa3[j];
            int64_t jp1 = j + 1;
            if (jp1 < call_config->n_free_param)
                for (int64_t i = jp1, ij = jp1 + jj; i < call_config->n_free_param; i++, ij++)
                    wa3[i] -= r[ij] * temp;
        }

        temp = norm(call_config->n_free_param, wa3);
        parc = ((fp / delta) / temp) / temp;
        // depending on the sign of the function, update parl or paru.
        if (fp > zero)
            parl = fmax(parl, par);

        if (fp < zero)
            paru = fmin(paru, par);

        // compute an improved estimate for par.
        par = fmax(parl, par + parc);
    }
}

__device__ void qrsolv(struct CallConfig *call_config, double *r, int64_t *ipvt, double *diag, double *qtb, double *x, double *sdiag, double *wa)
{
    // copy r and (q transpose)*b to preserve input and initialize s.
    // in particular, save the diagonal elements of r in x.
    for (int64_t j = 0, kk = 0; j < call_config->n_free_param; j++, kk += (call_config->m) + 1)
    {
        for (int64_t i = j, ij = kk, ik = kk; i < call_config->n_free_param; i++, ij++, ik += call_config->m)
            r[ij] = r[ik];

        x[j] = r[kk];
        wa[j] = qtb[j];
    }

    // eliminate the diagonal matrix d using a givens rotation.
    for (int64_t j = 0; j < call_config->n_free_param; j++)
    {
        // prepare the row of d to be eliminated, locating the
        // diagonal element using p from the qr factorization.
        int64_t l = ipvt[j];
        if (diag[l] != zero)
        {
            for (int64_t k = j; k < call_config->n_free_param; k++)
                sdiag[k] = zero;

            sdiag[j] = diag[l];
            // the transformations to eliminate the row of d
            // modify only a single element of (q transpose)*b
            // beyond the first n, which is initially zero.
            double qtbpj = zero;
            for (int64_t k = j; k < call_config->n_free_param; k++)
            {
                double sinx, cosx;

                // determine a givens rotation which eliminates the
                // appropriate element in the current row of d.
                if (sdiag[k] == zero)
                    continue;

                int64_t kk = k + (call_config->m) * k;
                if (fabs(r[kk]) < fabs(sdiag[k]))
                {
                    double cotanx = r[kk] / sdiag[k];
                    sinx = p5 / sqrt(p25 + p25 * cotanx * cotanx);
                    cosx = sinx * cotanx;
                }
                else
                {
                    double tanx = sdiag[k] / r[kk];
                    cosx = p5 / sqrt(p25 + p25 * tanx * tanx);
                    sinx = cosx * tanx;
                }

                // compute the modified diagonal element of r and
                // the modified element of ((q transpose)*b, 0).
                r[kk] = cosx * r[kk] + sinx * sdiag[k];
                double temp = sinx * qtbpj + cosx * wa[k];
                qtbpj = cosx * qtbpj - sinx * wa[k];
                wa[k] = temp;
                // accumulate the tranformation in the row of s.
                if (call_config->n_free_param > k + 1)
                    for (int64_t i = k + 1, ik = kk + 1; i < call_config->n_free_param; i++, ik++)
                    {
                        double temp = sinx * sdiag[i] + cosx * r[ik];
                        sdiag[i] = cosx * sdiag[i] - sinx * r[ik];
                        r[ik] = temp;
                    }
            }
        }

        // store the diagonal element of s and restore
        // the corresponding diagonal element of r.
        int64_t kk = j + (call_config->m) * j;
        sdiag[j] = r[kk];
        r[kk] = x[j];
    }

    // solve the triangular system for z. if the system
    // is singular, then obtain a least squares solution.
    int64_t nsing = call_config->n_free_param;
    for (int64_t j = 0; j < call_config->n_free_param; j++)
    {
        if ((sdiag[j] == zero) && (nsing == call_config->n_free_param))
            nsing = j;

        if (nsing < call_config->n_free_param)
            wa[j] = zero;
    }

    for (int64_t k = 0; k < nsing; k++)
    {
        int64_t j = nsing - k - 1;
        double sum = zero;
        if (nsing > j + 1)
            for (int64_t i = j + 1, ij = j + 1 + (call_config->m) * j; i < nsing; i++, ij++)
                sum += r[ij] * wa[i];

        wa[j] = (wa[j] - sum) / sdiag[j];
    }

    // permute the components of z back to components of x.
    for (int64_t j = 0; j < call_config->n_free_param; j++)
        x[ipvt[j]] = wa[j];
}

__device__ void covar(struct CallConfig *call_config, double *r, int64_t *ipvt, double *wa)
{
    // form the inverse of r in the full upper triangle of r.
    double tolr = call_config->covtol * fabs(r[0]);

    int64_t l = -1;
    for (int64_t k = 0; k < call_config->n_free_param; k++)
    {
        int64_t kk = k * (call_config->m) + k;
        if (fabs(r[kk]) <= tolr)
            break;

        r[kk] = one / r[kk];
        for (int64_t j = 0; j < k; j++)
        {
            int64_t kj = k * (call_config->m) + j;
            double temp = r[kk] * r[kj];
            r[kj] = zero;

            int64_t k0 = k * (call_config->m);
            int64_t j0 = j * (call_config->m);
            for (int64_t i = 0; i <= j; i++)
                r[k0 + i] -= temp * r[j0 + i];
        }
        l = k;
    }

    // Form the full upper triangle of the inverse of
    // (r transpose)*r in the full upper triangle of r
    if (l >= 0)
        for (int64_t k = 0; k <= l; k++)
        {
            int64_t k0 = k * (call_config->m);

            for (int64_t j = 0; j < k; j++)
            {
                double temp = r[k * (call_config->m) + j];

                int64_t j0 = j * (call_config->m);
                for (int64_t i = 0; i <= j; i++)
                    r[j0 + i] += temp * r[k0 + i];
            }

            double temp = r[k0 + k];
            for (int64_t i = 0; i <= k; i++)
                r[k0 + i] *= temp;
        }

    // For the full lower triangle of the covariance
    // matrix in the strict lower triangle or and in wa
    for (int64_t j = 0; j < call_config->n_free_param; j++)
    {
        int64_t jj = ipvt[j];
        bool sing = (j > l);
        int64_t j0 = j * (call_config->m);
        int64_t jj0 = jj * (call_config->m);
        for (int64_t i = 0; i <= j; i++)
        {
            int64_t ji = j0 + i;

            if (sing)
                r[ji] = zero;

            int64_t ii = ipvt[i];
            if (ii > jj)
                r[jj0 + ii] = r[ji];

            if (ii < jj)
                r[ii * (call_config->m) + jj] = r[ji];
        }
        wa[jj] = r[j0 + j];
    }

    // Symmetrize the covariance matrix in r
    for (int64_t j = 0; j < call_config->n_free_param; j++)
    {
        int64_t j0 = j * (call_config->m);
        for (int64_t i = 0; i < j; i++)
            r[j0 + i] = r[i * (call_config->m) + j];

        r[j0 + j] = wa[j];
    }
}

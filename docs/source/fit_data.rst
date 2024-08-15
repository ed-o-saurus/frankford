..
 Copyright (C) 2024 Edward F. Behn, Jr.
..
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
..
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
..
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.

Fit Data
========

The fit data returned by calling the fitter is a ``numpy`` array. This shape is determined by the :py:class:`frankford.Dataset` objects passed.
Each element of the array represents one fit and consists of several fields.

Result
++++++

The ``result`` field is a ``np.int8`` whose value indicates the way that fit algorithm failed of succeeded.
Negative values indicate failure while positive values indicate success.
It can be interpreted by using it to construct a :py:class:`frankford.Result` object.

.. autoclass:: frankford.Result
    :members: ERR_UNKNOWN, ERR_DOF, ERR_USER_FUNC, OK_CHI_SQ, OK_PAR, OK_BOTH, OK_DIR, MAX_ITER, FTOL, XTOL, GTOL, __bool__

Note that if ``result`` indicates failure of the fit, ``chi_sq``, ``parameters``, ``uncertainties``, and ``covar`` are not correct.

Final :math:`\chi^2`
++++++++++++++++++++

The ``chi_sq`` field is a ``np.double`` whose value indicates the final minimized :math:`\chi^2` value of the fit.

Degrees of Freedom
++++++++++++++++++

The ``dof`` field is a ``np.int64`` whose value indicates the number of degrees of freedom of the fit.
It is equal to the number of points used in the fit minus the number of free parameters.

Number of Iterations
++++++++++++++++++++

The ``num_iter`` field is a ``np.int64`` whose value indicates the number of iterations of the Levenberg–Marquardt algorithm used.


Original :math:`\chi^2`
+++++++++++++++++++++++

The ``orig_chi_sq`` field is a ``np.double`` whose value indicates the original :math:`\chi^2` value of the data using the initial parameter values.

Parameters
++++++++++

The ``parameters`` field is a collection of sub-fields of type ``np.double`` storing the optimized parameter values.
Free parameters are set to the values that minimize :math:`\chi^2`.
Fixed parameters are set to the values provided.
Tied parameters are calculated from other parameters.

Uncertainties
+++++++++++++

The ``uncertainties`` field is a collection of sub-fields of type ``np.double`` storing the uncertainties of optimized free parameters.

Covariance Matrix
+++++++++++++++++

The ``covar`` field is a collection of sub-fields of type ``np.double`` storing elements of the covariance matrix of free parameters.
The names are a combination of the two free parameter names separated by a dollar sign.
For example, the covariance of ``alpha`` and ``bravo`` is ``alpha$bravo``.

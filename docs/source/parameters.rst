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

Parameters
==========

There are three type of parameters that may be used in a fit.

Free Parameters
+++++++++++++++

A free parameter is adjusted by the fitter to minimize :math:`\chi^2`. It is represented by the class :py:class:`frankford.FreeParameter`.

.. autoclass:: frankford.FreeParameter
    :members: __init__

When the fitter is called, a :py:class:`frankford.FreeParameterSetting` object must be passed in the ``dict`` ``parameter_settings`` for each free parameter.

.. autoclass:: frankford.FreeParameterSetting
    :members: __init__, step, relative_step, side

The values for ``init_values``, ``lower``, and ``upper`` must each be either a scalar or an array with the same shape as the output array
as determined by the datasets.

.. autoclass:: frankford.Side
    :members: AUTO, POS, NEG, BOTH

Fixed Parameters
++++++++++++++++

A fixed parameter is assigned a value that does not change. It is represented by the class :py:class:`frankford.FixedParameter`.

.. autoclass:: frankford.FixedParameter
    :members: __init__

When the fitter is called, a :py:class:`frankford.FixedParameterSetting` object must be passed in the ``dict`` ``parameter_settings`` for each fixed parameter.

.. autoclass:: frankford.FixedParameterSetting
    :members: __init__

Tied Parameters
+++++++++++++++

A tied parameter has a value that is a function of other parameters. It is represented by the class :py:class:`frankford.TiedParameter`.

.. autoclass:: frankford.TiedParameter
    :members: __init__

The functions must take only other parameters are arguments.
The function must return a ``double`` when all arguments are type ``double``.
Further, it must conform to the requirements laid out in the
`numba documentation <https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html>`_ for cuda device functions.

If the function raises a python error the result for the fit is :py:class:`frankford.Result.ERR_USER_FUNC`.
Returning a non-finite (:math:`\pm \infty` or not-a-number (NaN)) value has the same result.

While the function can take one or more other tied parameters as arguments, there may not be a circular dependency.
For example, if ``a`` is a tied parameter with an argument of ``b``,
``b`` is a tied parameter with an argument of ``c``, and
``c`` is a tied parameter with an argument of ``a``, an error will be raised.





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

Datasets
=========

When a :py:class:`frankford.Fitter` object is created the ``models`` parameter is a ``dict`` whose values are functions that take parameters and
independent variables as arguments. A model must return a ``double`` when all arguments are type ``double``.
Further, it must conform to the requirements laid out in the
`numba documentation <https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html>`_ for cuda device functions.

If a model raises a python error the result for the fit is :py:class:`frankford.Result.ERR_USER_FUNC`.
Returning a non-finite (:math:`\pm \infty` or not-a-number (NaN)) value has the same result.

The :py:class:`frankford.Dataset` class is used to pass data used in the fit to the :py:class:`frankford.Fitter` object when it is called.
One :py:class:`frankford.Dataset` is passed for each model in the ``models`` parameters passed to ``__init__``.
The keys in this object must be the same as the ``models`` ``dict`` used when the :py:class:`frankford.Fitter` was created. 

.. autoclass:: frankford.Dataset
    :members: __init__

The ``points`` array must be a numpy array with dtype ``frankford.point_dtype``. This dtype has two ``np.double`` fields.
``value`` stores the value to be fitted to and ``uncertainty`` stores the uncertainty of that value. 

The output shape of the data set is the shape of the ``points`` array with the axes in ``axis`` eliminated.
For example if the shape of the array is ``(2, 3, 4, 5)`` and ``axis`` is ``(0, 2)``, the output shape is ``(3, 5)``.
The output shape will be the shape of the output array. All data sets must have the same output shape.

Each independent variable array must have the same shape as the ``points`` array or the shape of the ``points`` array
with only the axes specified.
For example if the shape of the array is ``(2, 3, 4, 5)`` and ``axis`` is ``(0, 2)``, the each independent variable array must have
shape ``(2, 3, 4, 5)`` or ``(2, 4)``.

Points whose ``value`` is not-a-number (NaN) are ignored by the fitter.
The number of remaining points is reflected in the fit's degrees of freedom returned.
If there are more free parameters than remaining points, the fit is invalid and
the return value has a ``result`` of :py:class:`frankford.Result.ERR_DOF`.

The keys in the ``ind_vars`` dict must be strings that match arguments in the model function. 

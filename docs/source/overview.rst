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

Overview
========

This document assume that the user is familiar with the numpy package and the Levenberg-Marquardt Algorithm. 

Summary
+++++++

The user provides one or more functions (models) and corresponding datasets to be fitted by their model by
adjusting specified parameters to minimize the total :math:`\chi^2` across all datasets.

A dataset consists of a multi-dimensional array of dependant values along with their uncertainties and one or more sets of 
independent variables. Each model is a function of the independent variables and some or all of the parameters. Each sub-array 
along the specified axes is independently fit by adjusting parameters.

A fit is optimized if the results of the model functions most closely match the dependant values provided.
This produces an output array of fits. Each fit specifies the optimal parameter values and other information.

Fitter Class
++++++++++++

The :py:class:`frankford.Fitter` class is used in two steps:

* The initializer is passed information about the parameters to be used in the fits and the model(s) (function(s)) that will be fitted to the data.
  This method creates the GPU code and loads it into the current context.

* The fitter object is called with data about free parameters and datasets. This actually runs the code on the GPU and returns fit data.
  The fit data that is returned contains the optimal parameter values and other information.

.. autoclass:: frankford.Fitter
    :members: __init__, __call__, ftol, xtol, gtol, stepfactor, covtol, maxiter, douserscale

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

Introduction
============

The Frankford package allows a user to use the CUDA system to fit multiple datasets simultaneously using the
`Levenberg–Marquardt algorithm <https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_.

.. _installation:

Installation
++++++++++++

To use Frankford, first install it using pip:

.. code-block:: console

   $ pip install frankford

Frankford requires an Nvidia GPU with CUDA toolkit 12 or higher installed. The installation of CUDA is beyond the scope of this document.
For more information, refer to the installation portion of Numba's `documentation <https://numba.readthedocs.io/en/stable/user/installing.html>`_.

.. _copyright:

Copyright
+++++++++

Python code copyright |copy| 2024 Edward F. Behn, Jr.

The C code is based on `MP-fit <https://pages.physics.wisc.edu/~craigm/idl/cmpfit.html>`_ developed by Craig Markwardt.

Translated from MINPACK-1 in FORTRAN, Apr-Jul 1998, CM
Copyright |copy| 1997-2002, Craig Markwardt
This software is provided as is without any warranty whatsoever.
Permission to use, copy, modify, and distribute modified or
unmodified copies is granted, provided this copyright and disclaimer
are included unchanged.

.. _license:

License
+++++++

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https://www.gnu.org/licenses/.

Name
++++

The package is named for the Frankford Arsenal where the Levenberg–Marquardt algorithm was first developed.

Contact
+++++++

To report an bug or suggest a feature, submit an issue on `GitHub <https://github.com/ed-o-saurus/frankford/issues>`_.

.. |copy| unicode:: U+000A9 .. COPYRIGHT SIGN

=======
phasepy
=======

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gustavochm/phasepy/master

What is phasepy?
----------------
Phasepy is an open-source scientific python package for fluid phase equilibria and interfacial properties computation.
This package facilitates the calculation of liquid-vapor equilibrium, liquid-liquid equilibrium and liquid-liquid-vapor equilibrium as well as density profiles and interfacial tension.
Equilibrium calculations can be performed with cubic equations of state (EoS) with classic or advances mixing rules or with a discontinuous approach using a virial equation of state for the vapor phase and an activity coefficients model for the liquid phase. On the other hand, the interfacial description must be done with a continuous model, i.e. cubic EoS.

Besides equilibria and interfacial computations, with Phasepy it is possible to fit pure component parameters as well as interaction parameters for quadratic mixing rule (QMR) and NRTL, Wilson and Redlich Kister activity coefficient models.

Phasepy relies on NumPy, SciPy and Cython extension modules, when necessary.

Installation Prerequisites
--------------------------
- numpy
- scipy
- pandas
- C/C++ Compiler for Cython extension modules

Installation
------------

Get the latest version of phasepy from
https://pypi.python.org/pypi/phasepy/

If you have an installation of Python with pip, simple install it with:

    $ pip install phasepy

To get the git version, run:

    $ git clone https://github.com/gustavochm/phasepy


Documentation
-------------

phasepy's documentation is available on the web:

    https://phasepy.readthedocs.io/en/latest/


Getting Started
---------------

This library is designed to work with absolute temperature in Kelvin, pressure in bar and
volume in cm3/mol. In order to create a mixture pure components have to be defined:

.. code-block:: python

	>>> from phasepy import component, mixture
	>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948,
	>>>	 w = 0.344861, GC = {'H2O':1})
	>>> ethanol = component(name = 'ethanol', Tc = 514.0, Pc = 61.37, Zc = 0.241, Vc = 168.0,
	>>>	 w = 0.643558, GC = {'CH3':1, 'CH2':1,'OH(P)':1})
	>>> mix = mixture(ethanol, water)

If, for example, we need the bubble point of the a mixture of x = 0.5 of ethanol at 320K, we could use
the Peng Robinson EoS with the advanced mixing rule MHV and the Modified-UNIFAC model:

.. code-block:: python

	>>> mix.unifac()
	>>> from phasepy import preos
	>>> eos = pr(mix, 'mhv_unifac')
	>>> from phasepy.equilibrium import bubblePy
	>>> y_guess, P_guess = [0.2,0.8] , 1
	>>> bubblePy(y_guess, P_guess, x = [0.5, 0.5], T = 320, model = eos)

Similarly, liquid-liquid and liquid-liquid-vapor equilibrium can be solved, if were the case, with the same object, eos.

For more examples, please have a look at the Jupyter Notebook files
located in the *examples* folder of the sources or
`view examples in github <https://github.com/gustavochm/phasepy/tree/master/examples>`_.


Latest source code
------------------

The latest development version of phasepy's sources can be obtained at

    https://github.com/gustavochm/phasepy


Bug reports
-----------

To report bugs, please use the phasepy's Bug Tracker at:

    https://github.com/gustavochm/phasepy/issues


License information
-------------------

See ``LICENSE.txt`` for information on the terms & conditions for usage
of this software, and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the phasepy license, if it is convenient for you,
please cite phasepy if used in your work. Please also consider contributing
any changes you make back, and benefit the community.


Chaparro, G, Mejía, A. Phasepy: A Python based framework for fluid phase
equilibria and interfacial properties computation.
J Comput Chem. 2020; 1– 23. `https://doi.org/10.1002/jcc.26405 <https://doi.org/10.1002/jcc.26405>`_.

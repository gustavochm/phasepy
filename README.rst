======
phasepy
======

What is phasepy?
----------------
Phasepy is a open-source scientific python package for fluid phase equilibria computation.
This package facilitate the calculation of liquid-vapour equilibrium, liquid-liquid equilibrium
and liquid-liquid-vapour equilibrium. Equilibrium calculations can be perfomed with cubic equations
of state with clasic or advances mixing rules o with a discontinuous approach using a virial equations
of state for the vapour phase and a activity coefficient model for the liquid phase.

Besides computations it is also possible to fit phase equilibria data, functions to fit quadratic
mix rule, NRTL Wilson Redlich Kister parameters, are included.

Phasety relys on numpy, scipy and cython extensiion modules, when necessary.

Installation
------------

Get the latest version of thermo from
https://pypi.python.org/pypi/phasepy/

If you have an installation of Python with pip, simple install it with:

    $ pip install phasepy

To get the git version, run:

    $ git clone https://github.com/gustavochm/phasepy


Getting Started
---------------

The library is designed to work with absolute temperature in Kelvin, pressure in bar and
volume in cm3/mol. In order to create a mixture pure components have to be defined:
	
.. code-block:: python

	>>> from phasepy import component, mixture
	>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948,
	>>>	 w = 0.344861, GC = {'H2O':1})
	>>> ethanol = component(name = 'ethanol', Tc = 514.0, Pc = 61.37, Zc = 0.241, Vc = 168.0,
	>>>	 w = 0.643558, GC = {'CH3':1, 'CH2':1,'OH(P)':1})
	>>> mix = mixture(ethanol, water)

If, for example, we need the bubble point of the of x = 0.5 of ethanol at 320K, we could use
the Peng Robinson EoS with an advanced mix rule with UNIFAC model:
	
.. code-block:: python

	>>> mix.unifac()
	>>> from phasepy import pr
	>>> eos = pr(mix, 'mhv_unifac')
	>>> from phasepy.equilibrium import bubblePy
	>>> y_guess, P_guess = [0.2,0.8] , 1
	>>> bubblePy(y_guess, P_guess, x = [0.5, 0.5], T = 320, model = eos)

Similarly, liquid-liquid and liquid-liquid-vapour equilibrium can be solved, if were the case,
with the same object, eos.


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
please cite thermo if used in your work. Please also consider contributing
any changes you make back, and benefit the community.


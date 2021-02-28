=======
phasepy
=======

Introduction
------------

Phasepy is an open-source scientific Python package for calculation of
`physical properties of phases <https://en.wikipedia.org/wiki/Physical_property>`_ at
`thermodynamic equilibrium <https://en.wikipedia.org/wiki/Thermodynamic_equilibrium>`_.
Main application areas include computation of fluid phase equilibria
and interfacial properties.

Phasepy includes routines for calculation of vapor-liquid equilibrium (VLE),
liquid-liquid equilibrium (LLE) and vapor-liquid-liquid equilibrium
(VLLE). Phase equilibrium can be modelled either with *the continous
approach*, using a combination of a cubic equation of state (EoS,
e.g. Van der Waals, Peng-Robinson, Redlich-Kwong, or their
derivatives) model and a mixing rule (Quadratic, Modified Huron-Vidal
or Wong-Sandler) for all phases, or *the discontinuous approach* using
a virial equation for the vapor phase and an activity coefficient model
(NRTL, Wilson, Redlich-Kister or Dortmund Modified UNIFAC) for the
liquid phase(s).

Interfacial property estimation using the continuous phase equilibrium
approach allows calculation of density profiles and interfacial
tension using the Square Gradient Theory (SGT).

Phasepy supports fitting of model parameter values from experimental data.

Installation Prerequisites
--------------------------
- numpy
- scipy
- pandas
- openpyxl
- C/C++ Compiler for Cython extension modules

Installation
------------

Get the latest version of phasepy from
https://pypi.python.org/pypi/phasepy/

An easy installation option is to use Python pip:

    $ pip install phasepy

Alternatively, you can build phasepy yourself using latest source
files:

    $ git clone https://github.com/gustavochm/phasepy


Documentation
-------------

Phasepy's documentation is available on the web:

    https://phasepy.readthedocs.io/en/latest/


Getting Started
---------------

Base input variables include temperature [K], pressure [bar] and molar
volume [cm^3/mol]. Specification of a mixture starts with
specification of pure components:

.. code-block:: python

   >>> from phasepy import component, mixture
   >>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948,
                         w=0.344861, GC={'H2O':1})
   >>> ethanol = component(name='ethanol', Tc=514.0, Pc=61.37, Zc=0.241, Vc=168.0,
		           w=0.643558, GC={'CH3':1, 'CH2':1, 'OH(P)':1})
   >>> mix = mixture(ethanol, water)

Here is an example how to calculate the bubble point vapor composition
and pressure of saturated 50 mol-% ethanol - 50 mol-% water liquid
mixture at temperature 320 K using Peng Robinson EoS. In this example
the Modified Huron Vidal mixing rule utilizes the Dortmund Modified
UNIFAC activity coefficient model for the solution of the mixture EoS.

.. code-block:: python

   >>> mix.unifac()
   >>> from phasepy import preos
   >>> eos = preos(mix, 'mhv_unifac')
   >>> from phasepy.equilibrium import bubblePy
   >>> y_guess, P_guess = [0.2, 0.8], 1.0
   >>> bubblePy(y_guess, P_guess, X=[0.5, 0.5], T=320.0, model=eos)
   (array([0.70761727, 0.29238273]), 0.23248584919691206)

For more examples, please have a look at the Jupyter Notebook files
located in the *examples* folder of the sources or
`view examples in github <https://github.com/gustavochm/phasepy/tree/master/examples>`_.


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


Chaparro, G., Mejía, A. Phasepy: A Python based framework for fluid phase
equilibria and interfacial properties computation.
J Comput Chem. 2020; 1– 23. `https://doi.org/10.1002/jcc.26405 <https://doi.org/10.1002/jcc.26405>`_.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   phasepy
   phasepy.equilibrium
   phasepy.sgt
   phasepy.fit


Indices and Search
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

phasepy
=======

Phasepy aims to require minimum quantity of parameters needed to do phase equilibrium calculations. First it is required to create components and mixtures, and then combine them with a phase equilibrium model to create a final model object, which can be used to carry out fluid phase equilibrium calculations.

.. toctree::
   	./basic/phasepy.component
	./basic/phasepy.mixture

With the class component :class:`phasepy.component`, only pure component info is saved. Info includes critical temperature, pressure, volume, acentric factor, Antoine coefficients and group contribution info.
The class :class:`phasepy.mixture` saves pure component data, but also interactions parameters for the activity coeffient models.

.. code-block:: python

	>>> from phasepy import component
	>>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861,
                              Ant=[11.64785144, 3797.41566067, -46.77830444],
                              GC={'H2O':1})
	>>> ethanol = component(name='ethanol', Tc=514.0, Pc=61.37, Zc=0.241, Vc=168.0, w=0.643558,
                                Ant=[11.61809279, 3423.0259436, -56.48094263],
                                GC={'CH3':1, 'CH2':1, 'OH(P)':1})
	>>> water.psat(T=373.0) # vapor saturation pressure [bar]
	1.0072796747419537
	>>> ethanol.vlrackett(T=310.0) # liquid molar volume [cm3/mol]
	56.32856995891473

A mixture can be created from two components:
	
.. code-block:: python

	>>> from phasepy import mixture
	>>> mix = mixture(ethanol, water)
	>>> mix.names
	['ethanol', 'water']
	>>> mix.nc # number of components
	2
	>>> mix.psat(T=373.0) # vapor saturation pressures [bar]
	array([2.23333531, 1.00727967])
	>>> mix.vlrackett(T=310.0) # liquid molar volumes [cm3/mol]
	array([56.32856996, 16.46025809])


Phasepy includes two phase equilibrium models:

1. A discontinous (:math:`\gamma - \phi`) Virial - Activity Coefficient Method model where the vapor and liquid deviations are modeled with an virial expansion and an activity coefficient model, respectively, or
2. A continuous  (:math:`\phi - \phi`) Cubic Equation of State model, using the same equation of state for all the phases.

.. toctree::
	:maxdepth: 1

   	./models/phasepy.virialgama
	./models/phasepy.cubic

The coded models were tested to pass molar partial property test and Gibbs-Duhem consistency, in the case of activity coefficient model the following equations were tested:

.. math::
	\frac{G^E}{RT} - \sum_{i=1}^c x_i \ln \gamma_i = 0\\
	\sum_{i=1}^c x_i d\ln \gamma_i = 0

where, :math:`G^E` refers to the Gibbs excess energy, :math:`R` is the ideal gas constant, :math:`T` is the absolute temperature, and :math:`x_i` and :math:`\gamma_i` are the mole fraction and activity coefficient of component :math:`i`. And in the case of cubic EoS:

.. math::
	\ln \phi - \sum_{i=1}^c x_i \ln \hat{\phi_i}  = 0 \\
	\frac{d \ln \phi}{dP} - \frac{Z - 1}{P} = 0 \\ 	
	\sum_{i=1}^c x_i d \ln \hat{\phi_i} = 0

Here, :math:`\phi` is the fugacity coefficient of the mixture,  :math:`x_i` and :math:`\hat{\phi_i}` are the mole fraction and fugacity coefficient of component :math:`i`, :math:`P` refers to pressure and :math:`Z` to the compressibility factor.

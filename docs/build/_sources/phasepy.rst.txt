phasepy
=======

Phasepy is a object oriented programmed Python package . In order to help the final user to deal with the minimun quantity of parameters as possible, those are saved as attributes in objects. Objects for creating components and mixtures are available, then this object are used within a EoS to create a final object that is going to be used for fluid phase equilibrium calculations.

In order to start with phase equilibrium calculations it is necessary to provide pure component and mixture info. Phasepy is an object oriented python package that implements two basics classes for this purpose.

.. toctree::
   	./basic/phasepy.component
	./basic/phasepy.mixture

With the class component :class:`phasepy.component`, pure component info is saved, this includes
critical temperature, pressure, volume, acentric factor, antoine coefficients and group contribution
info. On the other hand, classs :class:`phasepy.mixture` saves pure component data and also allows to
incorporate interactions parameters for the available models as, NRTL, Wilson, Redlich Kister.

.. code-block:: python

	>>> from phasepy import component
	>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948, w = 0.344861,
        ...        Ant =  [  11.64785144, 3797.41566067,  -46.77830444],
        ...        GC = {'H2O':1})
	>>> ethanol = component(name = 'ethanol', Tc = 514.0, Pc = 61.37, Zc = 0.241, Vc = 168.0, w = 0.643558,
        ...        Ant = [  11.61809279, 3423.0259436 ,  -56.48094263],
        ...        GC = {'CH3':1, 'CH2':1,'OH(P)':1})
	>>> water.psat(T = 373)
	1.0072796747419537
	>>> water.vlrackett(T = 310)
	16.46025809309672
	>>> ethanol.psat(T = 373)
	2.233335305328437
	>>> ethanol.vlrackett(T = 310)
	56.32856995891473

Now, from two components a mixture can be created:
	
.. code-block:: python

	>>> from phasepy import mixture
	>>> mix = mixture(ethanol, water)
	>>> mix.names
	['ethanol', 'water']
	>>> mix.nc
	2
	>>> mix.psat(T = 373)
	array([2.23333531, 1.00727967])
	>>> mix.vlrackett(T = 310)
	array([56.32856996, 16.46025809])


The mixture object can be used within any of the available models in Phasepy. There are two options when choosing a model, a discontinous model where the vapor and liquid deviations are modeled with an virial expansion and a activity coefficient model, respectively. Aditionally, both phases can be modeled with a continous model, using the same equation of state for all the phases. The available options are described bellow:

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

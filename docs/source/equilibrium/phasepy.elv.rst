Bubble Point and Dew Point
==========================

Calculation of
`bubble point <https://en.wikipedia.org/wiki/Bubble_point>`_ and
`dew point <https://en.wikipedia.org/wiki/Dew_point>`_
for vapor-liquid systems apply simplifications of the Rachford-Rice
mass balance, as the liquid phase fraction (0% or 100%) is already
known. Default solution strategy applies first Accelerated Successive
Substitutions method to update phase compositions in an inner loop,
and a Quasi-Newton method to update pressure or temperature in an
outer loop. If convergence is not reached in 10 iterations, or if user
defines initial estimates to be good, the algorithm switches to a
Phase Envelope method solving the following system of equations:

.. math::

	f_i &= \ln K_i + \ln \hat{\phi}_i^v(\underline{y}, T, P) -\ln \hat{\phi}_i^l(\underline{x}, T, P) \quad i = 1,...,c \\
	f_{c+1} &= \sum_{i=1}^c (y_i-x_i) 

Bubble Point
------------

Bubble point is a fluid state where saturated liquid of known
composition and liquid fraction 100% is forming a differential size
bubble. The algorithm finds vapor phase composition and either
temperature or pressure of the bubble point.

The algorithm solves composition using a simplified Rachford-Rice
equation:

.. math::

	FO = \sum_{i=1}^c x_i (K_i-1) = \sum_{i=1}^c y_i -1 = 0 


>>> import numpy as np
>>> from phasepy import component, mixture, rkseos
>>> from phasepy.equilibrium import bubbleTy, bubblePy
>>> butanol = component(name='butanol', Tc=563.0, Pc=44.14, Zc=0.258, Vc=274.0, w=0.589462,
                        Ant=[10.20170373, 2939.88668723, -102.28265042])
>>> mtbe = component(name='mtbe', Tc=497.1, Pc=34.3, Zc=0.273, Vc=329.0, w=0.266059,
                     Ant=[9.16238246, 2541.97883529, -50.40534341])
>>> Kij = np.zeros([2,2])
>>> mix = mixture(mtbe, butanol) 
>>> mix.kij_cubic(Kij)
>>> eos = rkseos(mix, 'qmr')
>>> x = np.array([0.5, 0.5])
>>> P0, T0 = 1.0, 340.0
>>> y0 = np.array([0.8, 0.2])
>>> bubbleTy(y0, T0, x, 1.0, eos) # vapor fractions, temperature
(array([0.90411878, 0.09588122]), 343.5331023048577)
>>> bubblePy(y0, P0, x, 343.533, eos) # vapor fractions, pressure
(array([0.90411894, 0.09588106]), 0.9999969450754181)

.. automodule:: phasepy.equilibrium.bubble
    :members: bubbleTy, bubblePy
    :undoc-members:
    :show-inheritance:
    :noindex:


Dew Point
---------

Dew point is a fluid state where saturated vapor of known composition
and liquid fraction 0% is forming a differential size liquid
droplet. The algorithm finds liquid phase composition and either
temperature or pressure of the dew point.

The algorithm solves composition using a simplified Rachford-Rice
equation:

.. math:: 
	FO = 1 - \sum_{i=1}^c \frac{y_i}{K_i} = 1 - \sum_{i=1}^c x_i = 0 


>>> import numpy as np
>>> from phasepy import component, mixture, prsveos
>>> from phasepy.equilibrium import dewPx, dewTx
>>> ethanol = component(name='ethanol', Tc=514.0, Pc=61.37, Zc=0.241, Vc=168.0, w=0.643558,
...                ksv=[1.27092923, 0.0440421],
...                GC={'CH3':1, 'CH2':1, 'OH(P)':1})
>>> mtbe = component(name='mtbe', Tc=497.1, Pc=34.3, Zc=0.273, Vc=329.0, w=0.266059,
...                ksv=[0.76429651, 0.04242646],
...                GC={'CH3':3, 'CH3O':1, 'C':1})
>>> mix = mixture(mtbe, ethanol)
>>> C0 = np.array([0.02635196, -0.02855964, 0.01592515])
>>> C1 = np.array([312.575789, 50.1476555, 5.13981131])
>>> mix.rk(C0, C1)
>>> eos = prsveos(mix, mixrule = 'mhv_rk')
>>> y = np.array([0.5, 0.5])
>>> P0, T0 = 1.0, 340.0
>>> x0 = np.array([0.2, 0.8])
>>> dewPx(x0, P0, y, 340.0, eos) # liquid fractions, pressure
(array([0.20223477, 0.79776523]), 1.0478247383242278)
>>> dewTx(x0, T0, y, 1.047825, eos) # liquid fractions, temperature
(array([0.20223478, 0.79776522]), 340.0000061757033)


.. automodule:: phasepy.equilibrium.dew
    :members: dewPx, dewTx
    :undoc-members:
    :show-inheritance:
    :noindex:

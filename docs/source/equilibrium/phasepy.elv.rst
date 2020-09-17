Vapor liquid equilibrium
=========================

Two phase flash can be used to compute vapor liquid equilibria at fixed temperature and pressure. When dealing with saturated liquids or vapor four types of problems arises, solution methods and routines are related to an simplification of the Radford-Rice mass balance, as the phase fraction is already known.

Other option for these situations is to solve the following system of equations:

.. math::

	f_i &= \ln K_i + \ln \hat{\phi}_i^v(\underline{y}, T, P) -\ln \hat{\phi}_i^l(\underline{x}, T, P) \quad i = 1,...,c \\
	f_{c+1} &= \sum_{i=1}^c (y_i-x_i) 

bubble points
#############

In this case a saturated liquid of known composition is forming a differential size bubble. If pressure is specified, temperature must be found. Similarly when temperature is specified, equilibrium pressure has to be calculated.

Usual approach for solving this problem consist in a combined quasi-Newton for solving for temperature or pressure and successive sustituion for composition with the following simplifation of the Radford-Rice equation:. 

.. math::

	FO = \sum_{i=1}^c x_i (K_i-1) = \sum_{i=1}^c y_i -1 = 0 

In case of having a good initial value of the true equilibrium values the full multidimentional system of equations can be solved. 


In the following code block and example from this computation it is shown.


>>> from phasepy import component, mixture, rkseos
>>> from phasepy.equilibrium import bubbleTy
>>> butanol = component(name = 'butanol', Tc =563.0, Pc = 44.14, Zc = 0.258, Vc = 274.0, w = 0.589462,
...                   Ant = [  10.20170373, 2939.88668723, -102.28265042])
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                Ant = [   9.16238246, 2541.97883529,  -50.40534341])
>>> Kij = np.zeros([2,2])
>>> mix = mixture(mtbe, butanol) 
>>> mix.kij_cubic(Kij)
>>> eos = rkseos(mix, 'qmr')
>>> x = np.array([0.5,0.5])
>>> P = 1 #bar
>>> y0 = np.array([0.8,0.2])
>>>T0 = 340
bubbleTy( y0, T0, x, P, eos)
>>> # y, T
(array([0.90359832, 0.09640168]), 343.8666585139102)

.. automodule:: phasepy.equilibrium.bubble
    :members: bubbleTy
    :undoc-members:
    :show-inheritance:
    :noindex:



In the following case a saturated liquid of known composition and temperature is forming a differential size bubble. We need to find composition and pressure of equilibrium.


>>> x = np.array([0.5,0.5])
>>> T = 340 #K
>>> y0 = np.array([0.8,0.2])
>>> P0 = 1
>>> bubblePy(y0, P0, x, T, eos)
>>> # y , P
(array([0.90955224, 0.09044776]), 0.8988497535228545)

.. automodule:: phasepy.equilibrium.bubble
    :members: bubblePy
    :undoc-members:
    :show-inheritance:
    :noindex:


dew points
##########

In this case a saturated vapour of known composition and temperature is forming a differential size dew. We need to find composition and pressure of equilibrium.

Usual approach for solving this problem consist in a combined quasi-Newton for solving for Pressure and successive sustituion for composition with the following simplifation of the Radford-Rice equation: 

.. math:: 
	FO = 1 - \sum_{i=1}^c \frac{y_i}{K_i} = 1 - \sum_{i=1}^c x_i = 0 

In case of having a good initial value of the true equilibrium values a full multidimentional system can be solved. 

In the following code block and example from this computation it is shown for composition and equilibrium pressure.


>>> from phasepy import component, mixture, prsveos
>>> from phasepy.equilibrium import dewPx
>>> ethanol = component(name = 'ethanol', Tc = 514.0, Pc = 61.37, Zc = 0.241, Vc = 168.0, w = 0.643558,
...                ksv = [1.27092923, 0.0440421],
...                GC = {'CH3':1, 'CH2':1, 'OH(P)':1})
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                ksv = [0.76429651, 0.04242646],
...                GC = {'CH3':3, 'CH3O':1, 'C':1})
>>> mix = mixture(mtbe, ethanol)
>>> C0 = np.array([ 0.02635196, -0.02855964,  0.01592515])
>>> C1 = np.array([312.575789  ,  50.1476555 ,   5.13981131])
>>> mix.rk(C0, C1)
>>> eos = prsveos(mix, mixrule = 'mhv_rk')
>>> y = np.array([0.5,0.5])
>>> T = 340. #K
>>> x0 = np.array([0.2,0.8])
>>> P0 = 1.
>>> dewPx( x0, P0, y, T, eos)
>>> x, P
array([0.20224128, 0.79775872]), 1.0478247387561512

.. automodule:: phasepy.equilibrium.dew
    :members: dewPx
    :undoc-members:
    :show-inheritance:
    :noindex:


Similarly, the calculation can be done for equilibria comosition and temperature:

>>> y = np.array([0.5,0.5])
>>> P = 1 #K
>>> x0 = np.array([0.2,0.8])
>>> T0 = 340
>>> dewTx( x0, T0, y, P , eos)
>>> # x, T
array([0.19854812, 0.80145188]), 338.85030223879545


.. automodule:: phasepy.equilibrium.dew
    :members: dewTx
    :undoc-members:
    :show-inheritance:
    :noindex:

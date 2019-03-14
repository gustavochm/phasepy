bubblePy
========

In this case a saturated liquid of known composition and temperature is forming a differential size bubble. We need to find composition and pressure of equilibrium.

Usual approach for solving this problem consist in a combined quasi-Newton for solving for Pressure and successive sustituion for composition. In case of having a good initial value of the true equilibrium values a full multidimentional system can be solved. 

In the following code block and example from this computation it is shown.

>>> from phasepy import component, mixture, rkseos
>>> from phasepy.equilibrium import bubblePy
>>> butanol = component(name = 'butanol', Tc =563.0, Pc = 44.14, Zc = 0.258, Vc = 274.0, w = 0.589462,
...                   Ant = [  10.20170373, 2939.88668723, -102.28265042])
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                Ant = [   9.16238246, 2541.97883529,  -50.40534341])
>>> Kij = np.zeros([2,2])
>>> mix = mixture(mtbe, butanol) 
>>> mix.kij_cubic(Kij)
>>> elv = rkseos(mix, 'qmr')
>>> x = np.array([0.5,0.5])
>>> T = 340 #K
>>> y0 = np.array([0.8,0.2])
>>> P0 = 1
>>> bubblePy( y0, P0, x, T, elv)
>>> # y , P
(array([0.90955224, 0.09044776]), 0.8988497535228545)

.. automodule:: phasepy.equilibrium.bubble
    :members: bubblePy
    :undoc-members:
    :show-inheritance:


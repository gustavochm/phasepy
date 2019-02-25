bubbleTy
========

>>> from phasepy import component, mixture, rkseos
>>> from phasepy.equilibrium import bubbleTy
>>> butanol = component(name = 'butanol', Tc =563.0, Pc = 44.14, Zc = 0.258, Vc = 274.0, w = 0.589462,
...                   Ant = [  10.20170373, 2939.88668723, -102.28265042])
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                Ant = [   9.16238246, 2541.97883529,  -50.40534341])
>>> Kij = np.zeros([2,2])
>>> mix = mixture(mtbe, butanol) 
>>> mix.kij_cubic(Kij)
>>> elv = rkseos(mix, 'qmr')
>>> x = np.array([0.5,0.5])
>>> P = 1 #bar
>>> y0 = np.array([0.8,0.2])
>>>T0 = 340
bubbleTy( y0, T0, x, P, elv)
>>> # y, T
(array([0.90359832, 0.09640168]), 343.8666585139102)

.. automodule:: phasepy.equilibrium.bubble
    :members: bubbleTy
    :undoc-members:
    :show-inheritance:
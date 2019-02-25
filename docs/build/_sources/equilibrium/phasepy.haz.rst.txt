Three phase equilibrium
=======================

Binary mixtures
---------------

>>> from phasepy import component, mixture, virialgama, unifac
>>> from phasepy.equilibrium import hazb
>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948, w = 0.344861,
...                Ant =  [  11.64785144, 3797.41566067,  -46.77830444],
...                GC = {'H2O':1})
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                Ant = [   9.16238246, 2541.97883529,  -50.40534341], 
...                GC = {'CH3':3, 'CH3O':1, 'C':1})
>>> mix = mixture(water, mtbe)
>>> mix.unifac()
>>> model = virialgama(mix, actmodel = unifac)
>>> P = 1.01 #bar
>>> #initial guess
>>> T0 = 320 #K
>>> x0 = np.array([0.01,0.99])
>>> w0 = np.array([0.99,0.01])
>>> y0 = (x0 + w0)/2
>>> hazb(x0,w0,y0, T0, P, 'P', model)
>>> #X, W, Y, T
>>> array([0.17165664, 0.82834336]) , array([0.99256232, 0.00743768]), array([0.15177621, 0.84822379]),  array([327.6066936])


.. automodule:: phasepy.equilibrium.hazb
    :members: hazb
    :undoc-members:
    :show-inheritance:

Multicomponent mixtures
-----------------------

>>> from phasepy import component, mixture, virialgama, unifac
>>> from phasepy.equilibrium import haz
>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948, w = 0.344861,
...                Ant =  [  11.64785144, 3797.41566067,  -46.77830444],
...                GC = {'H2O':1})
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                Ant = [   9.16238246, 2541.97883529,  -50.40534341], 
...                GC = {'CH3':3, 'CH3O':1, 'C':1})
>>> ethanol = component(name = 'ethanol', Tc = 514.0, Pc = 61.37, Zc = 0.241, Vc = 168.0, w = 0.643558,
...                Ant = [  11.61809279, 3423.0259436 ,  -56.48094263],
...                GC = {'CH3':1, 'CH2':1,'OH(P)':1})
>>> mix = mixture(water, mtbe)
>>> mix.add_component(ethanol)
>>> mix.unifac()
>>> model = virialgama(mix, actmodel = unifac)
>>> P = 1.01 #bar
>>> T = 328.5
>>> #initial guess
>>> x0 = np.array([0.95, 0.025, 0.025]),
>>> w0 =  np.array([0.4, 0.5 , 0.1])
>>> y0 = np.array([0.15,0.8,0.05])
>>> haz(x0, w0, y0, T, P, model, full_output = True)
>>> #T: 328.5
>>> #P: 1.01
>>> #error_outer: 8.153105081394752e-11
>>> #error_inner: 1.7587926210834326e-10
>>> #iter: 2
>>> #beta: array([0.4189389 , 0.18145858, 0.39960252])
>>> #tetha: array([0., 0., 0.])
>>> #X: array([[0.94674077, 0.01222667, 0.04103255],
>>> #       [0.23284951, 0.67121796, 0.09593252],
>>> #       [0.15295429, 0.78764814, 0.05939758]])
>>> #v: [None, None, None]
>>> #states: ['L', 'L', 'V']

.. automodule:: phasepy.equilibrium.hazt
    :members: haz
    :undoc-members:
    :show-inheritance:
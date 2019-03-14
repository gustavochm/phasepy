Liquid Liquid Equilibrium
=========================
Two phase flash can be used for solving liquid liquid equilibrium, but it is important to consider stability of the phases. For that reason a algorithm that can compute stability and equilibrium simultaneously was implemented in this packages.

In the following code block and example of how to solve this problem it is shown.

>>> from phasepy import component, mixture, virialgama, unifac
>>> from phasepy.equilibrium import ell
>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948, w = 0.344861,
...                Ant =  [  11.64785144, 3797.41566067,  -46.77830444],
...                GC = {'H2O':1})
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                Ant = [   9.16238246, 2541.97883529,  -50.40534341], 
...                GC = {'CH3':3, 'CH3O':1, 'C':1})
>>> mixell = mixture(water, mtbe)
>>> mixell.unifac()
>>> mell = virialgama(mixell, actmodel = unifac)
>>> T = 320 #K
>>> P = 1.01 #bar
>>> Z  = np.array([0.5,0.5])
>>> #initial guess
>>> x0 = np.array([0.01,0.99])
>>> w0 = np.array([0.99,0.01])
>>> ell(x0, w0, Z, T, P, mell)
>>> #x, w, beta
array([0.15601096, 0.84398904]), array([0.99289324, 0.00710676]), 0.41103635397012755

.. automodule:: phasepy.equilibrium.ell
    :members: ell
    :undoc-members:
    :show-inheritance:
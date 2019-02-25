Stability
=========

Minimization of tpd function
----------------------------

>>> from phasepy import component, mixture, preos
>>> from phasepy.equilibrium import tpd_min
>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948, w = 0.344861,
...                 Ant =  [  11.64785144, 3797.41566067,  -46.77830444],
...                GC = {'H2O':1})
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                Ant = [   9.16238246, 2541.97883529,  -50.40534341], 
...                GC = {'CH3':3, 'CH3O':1, 'C':1})
>>> mix = mixture(water, mtbe)
>>> mix.unifac()
>>> model = preos(mix, 'mhv_unifac')
>>> T = 320
>>> P = 1.01
>>> z = np.array([0.5,0.5])
>>> #Search for trial phase
>>> w = np.array([0.01,0.99])
>>> tpd_min(w, z, T, P, model, 'L', 'L')
>>>#composition of minimum found and tpd value
array([0.3068438, 0.6931562]), -0.005923914972662647
>>> tpd_min(w, z, T, P, model, 'V', 'L')
>>> #composition of minimum found and tpd value
array([0.16434071, 0.83565929]), 0.24576563932346407


.. automodule:: phasepy.equilibrium.stability
    :members: tpd_min
    :undoc-members:
    :show-inheritance:


Findind all minimums
--------------------

>>> from phasepy.equilibrium import tpd_minimas
>>> T = 320
>>> P = 1.01
>>> z = np.array([0.5,0.5])
>>> #Search for trial phase
>>> w = np.array([0.01,0.99])
>>> nmin = 3
>>> tpd_minimas(nmin , z, T, P, model, 'L', 'L')
>>> #composition of minimuns found and tpd values
(array([0.99538258, 0.00461742]), array([0.30683414, 0.69316586]), array([0.99538258, 0.00461742])),
array([-0.33722905, -0.00592392, -0.33722905])
>>> tpd_minimas(nmin , z, T, P, model, 'V', 'L')
>>> #composition of minimuns found and tpd values
(array([0.1643422, 0.8356578]), array([0.1643422, 0.8356578]), array([0.1643422, 0.8356578])),
array([0.24576564, 0.24576564, 0.24576564]))

.. automodule:: phasepy.equilibrium.stability
    :members: tpd_minimas
    :undoc-members:
    :show-inheritance:

Liquid liquid equilibrium initiation
------------------------------------

>>> from phasepy.equilibrium import ell_init
>>> T = 320
>>> P = 1.01
>>> z = np.array([0.5,0.5])
>>> ell_init(z, T, P, model)
>>>#initial values for ell computation
array([0.99538258, 0.00461742]), array([0.30683414, 0.69316586])

.. automodule:: phasepy.equilibrium.stability
    :members: ell_init
    :undoc-members:
    :show-inheritance:
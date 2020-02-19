Stability
=========

Stability analysis plays a fundamental role during phase equilibria computation. Most of stability test are based on the fact that a consistent equilibrium must minimize the Gibbs free energy of the system at given temperature and pressure. Within this idea Michelsen proposed the tangent plane distance function which allows to test the relative stability of a mixture at given composition (:math:`z`), temperature (:math:`T`),  and pressure(:math:`P`).

.. math::

	tpd(w) =  \sum_{i=1}^c w_i \left[\ln w_i +  \ln \hat{\phi}_i(w) - \ln z_i - \ln \hat{\phi}_i(z) \right]

The tpd function is evaluated for a trial composition (:math:`w`) and if the tpd takes a negative value it implies that the energy of the system decreased with the formation of the new phase, i.e. the original phase was unstable. In order to test stability of a mixture the usual method is to find a minimum of the function a verify the sign of the tpd function at the minimum. Minimization recomendations for this purpose where given by Michelsen and they are included in Phasepy's implementation. 

Minimization of tpd function
----------------------------

As this is an iterative process, in order to find a minimum a initial guess of it has to be supplied. In the following code block te stability of an liquid mixture is tested against the formation of another liquid.

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
>>> tpd_min(w, z, T, P, model, stateW = 'L', stateZ = 'L')
>>>#composition of minimum found and tpd value
array([0.3068438, 0.6931562]), -0.005923914972662647

As the tpd value at the minimum is negative it means that the phase is unstable at will split into two liquids. Similarly the relative stability can be tested against a vapor phase, in which case is found that the original phase was more stable than the vapor.

>>> tpd_min(w, z, T, P, model, stateW =  'V', stateZ =  'L')
>>> #composition of minimum found and tpd value
array([0.16434071, 0.83565929]), 0.24576563932346407

.. automodule:: phasepy.equilibrium.stability
    :members: tpd_min
    :undoc-members:
    :show-inheritance:



Findind all minimums
--------------------

Sometimes might be usefull to find all the minimas of a given mixture, for that case phasepy will try to find them using different initial guesses until the number of requested minimas is found. In the next example three minimias where requested.

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

Similar as the first example, all the minimas in vapor phase can be found, in this case there only one minimum.

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

Using the same principles stated above, tpd function can be used to generate initial guesses for liquid liquid equilibra, the function ell_init allows to find two minimas of the mixture.

>>> from phasepy.equilibrium import ell_init
>>> T = 320
>>> P = 1.01
>>> z = np.array([0.5,0.5])
>>> ell_init(z, T, P, model)
>>> #initial values for ell computation
array([0.99538258, 0.00461742]), array([0.30683414, 0.69316586])


.. automodule:: phasepy.equilibrium.stability
    :members: ell_init
    :undoc-members:
    :show-inheritance:
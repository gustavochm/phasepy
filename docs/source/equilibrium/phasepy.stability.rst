.. _stability:

Phase Stability Analysis
========================

Stability tests give a relative estimate of how stable a phase (given
temperature, pressure and composition) is thermodynamically. A stable
phase decreases the Gibbs free energy of the system compared to a
reference phase (e.g. the overall composition of a mixture). To
evaluate the relative stability, phasepy applies the Tangent Plane
Distance (TPD) function introduced by Michelsen.

.. math::

	F_{TPD}(w) =  \sum_{i=1}^c w_i \left[\ln w_i +  \ln \hat{\phi}_i(w) - \ln z_i - \ln \hat{\phi}_i(z) \right]

First, the TPD function is minimized locally w.r.t. phase composition
:math:`w`, starting from an initial composition. If the TPD value at
the minimum is negative, it implies that the phase is less stable than
the reference. A positive value means more stable than the reference.

The function ``tpd_min()`` can be applied to find a phase composition
from given initial values, and the TPD value of the minimized
result. Negative TPD value for the first example (trial phase is
liquid) means that the resulting liquid phase composition is
unstable. Positive TPD value for the second example (trial phase is
vapor) means that the resulting vapor phase is more stable than the
reference phase. Therefore the second solution could be used e.g. as
an initial estimate for two-phase flash calculation.

>>> import numpy as np
>>> from phasepy import component, mixture, preos
>>> from phasepy.equilibrium import tpd_min
>>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861,
                      Ant=[11.64785144, 3797.41566067, -46.77830444],
                      GC={'H2O':1})
>>> mtbe = component(name='mtbe', Tc=497.1, Pc=34.3, Zc=0.273, Vc=329.0, w=0.266059,
                     Ant=[9.16238246, 2541.97883529, -50.40534341],
                     GC={'CH3':3, 'CH3O':1, 'C':1})
>>> mix = mixture(water, mtbe)
>>> mix.unifac()
>>> eos = preos(mix, 'mhv_unifac')
>>> w = np.array([0.01, 0.99])
>>> z = np.array([0.5, 0.5])
>>> T = 320.0
>>> P = 1.01
>>> tpd_min(w, z, T, P, eos, stateW='L', stateZ='L') # molar fractions and TPD value
(array([0.30683438, 0.69316562]), -0.005923915138229763)
>>> tpd_min(w, z, T, P, eos, stateW='V', stateZ='L') # molar fractions and TPD value
(array([0.16434188, 0.83565812]), 0.24576563932356765)


.. automodule:: phasepy.equilibrium.stability
    :members: tpd_min
    :undoc-members:
    :show-inheritance:
    :noindex:


Phasepy function ``tpd_minimas()`` can be used to try to find
several TPD minima. The function uses random initial compositions to
search for minima, so it can find the same minimum multiple times.

>>> from phasepy.equilibrium import tpd_minimas
>>> nmin = 3
>>> tpd_minimas(nmin, z, T, P, eos, 'L', 'L') # minima and TPD values (two unstable minima)
(array([0.99538258, 0.00461742]), array([0.30683414, 0.69316586]), array([0.99538258, 0.00461742])),
array([-0.33722905, -0.00592392, -0.33722905])
>>> tpd_minimas(nmin, z, T, P, eos, 'V', 'L') # minima and TPD values (one stable minimum)
(array([0.1643422, 0.8356578]), array([0.1643422, 0.8356578]), array([0.1643422, 0.8356578])),
array([0.24576564, 0.24576564, 0.24576564])

.. automodule:: phasepy.equilibrium.stability
    :members: tpd_minimas
    :undoc-members:
    :show-inheritance:
    :noindex:


Function ``lle_init()`` can be used to generate two phase compositions
e.g. for subsequent liquid liquid equilibrium calculations.
Note that the results are not guaranteed to be stable or even different.

>>> from phasepy.equilibrium import lle_init
>>> lle_init(z, T, P, eos)
array([0.99538258, 0.00461742]), array([0.30683414, 0.69316586])


.. automodule:: phasepy.equilibrium.stability
    :members: lle_init
    :undoc-members:
    :show-inheritance:
    :noindex:

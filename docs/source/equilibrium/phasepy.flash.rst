Two-Phase Flash
===============

Determination of phase composition in
`flash evaporation <https://en.wikipedia.org/wiki/Flash_evaporation>`_
is a classical phase equilibrium problem. In the simplest case covered
here, temperature, pressure and overall composition of a system are
known (PT flash). If the system is thermodynamically unstable it will
form two (or more) distinct phases. The flash algorithm first solves
the Rachford-Rice mass balance and then updates composition by the
Accelerated Successive Substitution method.

.. math::
	FO = \sum_{i=1}^c \left( x_i^\beta - x_i^\alpha \right) = \sum_{i=1}^c \frac{z_i (K_i-1)}{1+\psi (K_i-1)}

:math:`x` is molar fraction of a component in a phase,
:math:`z` is the overall molar fraction of component in the system,
:math:`K = x^\beta / x^\alpha` is the equilibrium constant and
:math:`\psi` is the phase fraction of phase :math:`\beta` in the system.
Subscript refers to component index and superscript refers to phase
index.

If convergence is not reached in three iterations, the algorithm
switches to a second order minimization of the Gibbs free energy of
the system:

.. math::
	min \, {G(\underline{F}^\alpha, \underline{F}^\beta)} = \sum_{i=1}^c (F_i^\alpha \ln \hat{f}_i^\alpha + F_i^\beta \ln \hat{f}_i^\beta)

:math:`F` is the molar amount of component in a phase and
:math:`\hat{f}` is the effective fugacity.

.. warning::

   ``flash()`` routine does not check for the stability of the numerical
   solution (see :ref:`stability`).

Vapor-Liquid Equilibrium
------------------------

This example shows solution of vapor-liquid equilibrium (VLE) using
the ``flash()`` function.

>>> import numpy as np
>>> from phasepy import component, mixture, preos
>>> from phasepy.equilibrium import flash
>>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861,
                      Ant=[11.64785144, 3797.41566067, -46.77830444],
                      GC={'H2O':1})
>>> ethanol = component(name='ethanol', Tc=514.0, Pc=61.37, Zc=0.241, Vc=168.0, w=0.643558,
                        Ant=[11.61809279, 3423.0259436, -56.48094263],
                        GC={'CH3':1, 'CH2':1, 'OH(P)':1})
>>> mix = mixture(water, ethanol)
>>> mix.unifac()
>>> eos = preos(mix, 'mhv_unifac')
>>> T = 360.0
>>> P = 1.01
>>> Z = np.array([0.8, 0.2])
>>> x0 = np.array([0.1, 0.9])
>>> y0 = np.array([0.2, 0.8])
>>> flash(x0, y0, 'LV', Z, T, P, eos) # phase compositions, vapor phase fraction
(array([0.8979481, 0.1020519]), array([0.53414948, 0.46585052]), 0.26923713078124695)


.. automodule:: phasepy.equilibrium.flash
    :members: flash
    :undoc-members:
    :show-inheritance:


Liquid-Liquid Equilibrium
-------------------------

For liquid-liquid equilibrium (LLE), it is important to consider
stability of the phases. ``lle()`` function takes into account both
stability and equilibrium simultaneously for the PT flash.

>>> import numpy as np
>>> from phasepy import component, mixture, virialgamma
>>> from phasepy.equilibrium import lle
>>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861,
                      Ant=[11.64785144, 3797.41566067, -46.77830444],
                      GC={'H2O':1})
>>> mtbe = component(name='mtbe', Tc=497.1, Pc=34.3, Zc=0.273, Vc=329.0, w=0.266059,
                     Ant=[9.16238246, 2541.97883529, -50.40534341],
                     GC={'CH3':3, 'CH3O':1, 'C':1})
>>> mix = mixture(water, mtbe)
>>> mix.unifac()
>>> eos = virialgamma(mix, actmodel = 'unifac')
>>> T = 320.0
>>> P = 1.01
>>> Z = np.array([0.5, 0.5])
>>> x0 = np.array([0.01, 0.99])
>>> w0 = np.array([0.99, 0.01])
>>> lle(x0, w0, Z, T, P, eos) # phase compositions, phase 2 fraction
(array([0.1560131, 0.8439869]), array([0.99289324, 0.00710676]), 0.4110348438873743)

.. automodule:: phasepy.equilibrium.ell
    :members: lle
    :undoc-members:
    :show-inheritance:

Liquid-liquid flash can be also solved without considering stability by
using the ``flash()`` function, but this is not recommended.

>>> from phasepy.equilibrium import flash
>>> flash(x0, w0, 'LL', Z, T, P, eos) # phase compositions, phase 2 fraction
(array([0.1560003, 0.8439997]), array([0.99289323, 0.00710677]), 0.41104385845638447)


Vapor-Liquid-Liquid Equilibrium
===============================

To avoid meta-stable solutions in vapor-liquid-liquid equilibrium
(VLLE) calculations, phasepy applies a modified Rachford-Rice mass
balance system of equations by Gupta et al, which allows to verify the
stability and equilibria of the phases simultaneously.

.. math::

	\sum_{i=1}^c \frac{z_i (K_{ik} \exp{\theta_k}-1)}{1+ \sum\limits^{\pi}_{\substack{j=1 \\ j \neq r}}{\psi_j (K_{ij}} \exp{\theta_j} -1)} = 0 \qquad k = 1,..., \pi,  k \neq r

:math:`\theta` is a stability variable. Positive value indicates
unstable phase and zero value a stable phase. If default solution
using Accelerated Successive Substitution and Newton's method does not
converge fast, the algorith will switch to minimization of the Gibbs
free energy of the system:

.. math::
	min \, {G} = \sum_{k=1}^\pi \sum_{i=1}^c F_{ik} \ln \hat{f}_{ik}


Multicomponent Flash
--------------------

Function ``vlle()`` solves VLLE for mixtures with two or more
components given overall molar fractions *Z*, temperature and
pressure.

>>> import numpy as np
>>> from phasepy import component, mixture, virialgamma
>>> from phasepy.equilibrium import vlle
>>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861,
                      Ant=[11.64785144, 3797.41566067, -46.77830444],
                      GC={'H2O':1})
>>> mtbe = component(name='mtbe', Tc=497.1, Pc=34.3, Zc=0.273, Vc=329.0, w=0.266059,
                     Ant=[9.16238246, 2541.97883529, -50.40534341],
                     GC={'CH3':3, 'CH3O':1, 'C':1})
>>> ethanol = component(name='ethanol', Tc=514.0, Pc=61.37, Zc=0.241, Vc=168.0, w=0.643558,
                        Ant=[11.61809279, 3423.0259436, -56.48094263],
                        GC={'CH3':1, 'CH2':1, 'OH(P)':1})
>>> mix = mixture(water, mtbe)
>>> mix.add_component(ethanol)
>>> mix.unifac()
>>> eos = virialgamma(mix, actmodel='unifac')
>>> x0 = np.array([0.95, 0.025, 0.025])
>>> w0 = np.array([0.4, 0.5, 0.1])
>>> y0 = np.array([0.15, 0.8, 0.05])
>>> Z = np.array([0.5, 0.44, 0.06])
>>> T = 328.5
>>> P = 1.01
>>> vlle(x0, w0, y0, Z, T, P, eos, full_output=True)
           T: 328.5
           P: 1.01
 error_outer: 3.985492841236682e-08
 error_inner: 3.4482008487377304e-10
        iter: 14
        beta: array([0.41457868, 0.22479531, 0.36062601])
       tetha: array([0., 0., 0.])
           X: array([[0.946738  , 0.01222701, 0.04103499],
       [0.23284911, 0.67121402, 0.09593687],
       [0.15295408, 0.78764474, 0.05940118]])
           v: [None, None, None]
      states: ['L', 'L', 'V']


.. automodule:: phasepy.equilibrium.hazt
    :members: vlle
    :undoc-members:
    :show-inheritance:


Binary Mixture Composition
--------------------------

The function ``vlleb()`` can solve a special case to find component
molar fractions in each of the three phases in a VLLE system
containing only two components. Given either temperature or pressure,
``vlleb()`` solves the other, along with the component molar fractions.
This system is fully defined (zero degrees of freedom) according to the
`Gibbs phase rule <https://en.wikipedia.org/wiki/Gibbs_phase_rule>`_.

>>> import numpy as np
>>> from phasepy import component, mixture, virialgamma
>>> from phasepy.equilibrium import vlleb
>>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861,
                      Ant=[11.64785144, 3797.41566067, -46.77830444],
                      GC={'H2O':1})
>>> mtbe = component(name='mtbe', Tc=497.1, Pc=34.3, Zc=0.273, Vc=329.0, w=0.266059,
                     Ant=[9.16238246, 2541.97883529, -50.40534341],
                     GC={'CH3':3, 'CH3O':1, 'C':1})
>>> mix = mixture(water, mtbe)
>>> mix.unifac()
>>> eos = virialgamma(mix, actmodel='unifac')
>>> x0 = np.array([0.01, 0.99])
>>> w0 = np.array([0.99, 0.01])
>>> y0 = (x0 + w0)/2
>>> T0 = 320.0
>>> P = 1.01
>>> vlleb(x0, w0, y0, T0, P, 'P', eos, full_output=True)
      T: array([327.60666698])
      P: 1.01
  error: 4.142157056965187e-12
   nfev: 17
      X: array([0.17165659, 0.82834341])
     vx: None
 statex: 'Liquid'
      W: array([0.99256232, 0.00743768])
     vw: None
 statew: 'Liquid'
      Y: array([0.15177615, 0.84822385])
     vy: None
 statey: 'Vapor'

.. automodule:: phasepy.equilibrium.hazb
    :members: vlleb
    :undoc-members:
    :show-inheritance:

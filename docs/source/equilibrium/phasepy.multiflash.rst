Multiphase Flash
================

Meta-stable solutions of the isofugacity method is an important concern whel dealing with more than two liquid phases. Stability verification during the equilibria computation must be performed. In phasepy liquid-liquid equilibria a vapor-liquid-liquid equilibra is solved similarly with an modified Radford-Rice mass balance system of equations that allows to verify the stability and equilibria of the phases simultaneously.

.. math::

	\sum_{i=1}^c \frac{z_i (K_{ik} \exp{\theta_k}-1)}{1+ \sum\limits^{\pi}_{\substack{j=1 \\ j \neq r}}{\psi_j (K_{ij}} \exp{\theta_j} -1)} = 0 \qquad k = 1,..., \pi,  k \neq r

This system of equations was proposed by Gupta et al, and it is a modified Radford-Rice mass balance which introduces stability variables :math:`\theta`. This allows to solve the mass balance for phase fraction and stability variables and then update composition similarly as a regular flash. The stability variable gives information about the phase, if it takes a positive value the phase is unstable, on the hand, if it is zero then the phase is stable. The algorithm of successive sustitution and Newton method can be slow in some cases, in that situation the function will attempt to minimize the Gibbs free energy of the system.

.. math::

	min \, {G} = \sum_{k=1}^\pi \sum_{i=1}^c F_{ik} \ln \hat{f}_{ik}


Binary mixtures
---------------

For degrees of freedom's restriction, a systems of equations has to be solved for three phase equilibrium of binary mixtures. In the following code block a example of how to do it it is shown.

>>> from phasepy import component, mixture, virialgamma
>>> from phasepy.equilibrium import vlleb
>>> mix = mixture(water, mtbe)
>>> mix.unifac()
>>> model = virialgamma(mix, actmodel = 'unifac')
>>> P = 1.01 #bar
>>> #initial guess
>>> T0 = 320 #K
>>> x0 = np.array([0.01,0.99])
>>> w0 = np.array([0.99,0.01])
>>> y0 = (x0 + w0)/2
>>> vlleb(x0,w0,y0, T0, P, 'P', model)
>>> #X, W, Y, T
>>> array([0.17165664, 0.82834336]) , array([0.99256232, 0.00743768]),
... array([0.15177621, 0.84822379]),  array([327.6066936])

.. automodule:: phasepy.equilibrium.hazb
    :members: vlleb
    :undoc-members:
    :show-inheritance:



Multicomponent mixtures
-----------------------

When working with multicomponent mixtures (3 or more) a multiflash has to be performed in order to compute three phase equilibrium. This algorithm ensures that a stable phases are computed.

>>> from phasepy import component, mixture, virialgamma
>>> from phasepy.equilibrium import vlle
>>> ethanol = component(name = 'ethanol', Tc = 514.0, Pc = 61.37, Zc = 0.241, Vc = 168.0, w = 0.643558,
...                Ant = [  11.61809279, 3423.0259436 ,  -56.48094263],
...                GC = {'CH3':1, 'CH2':1,'OH(P)':1})
>>> mix = mixture(water, mtbe)
>>> mix.add_component(ethanol)
>>> mix.unifac()
>>> model = virialgamma(mix, actmodel = 'unifac')
>>> P = 1.01 #bar
>>> T = 328.5
>>> Z = np.array([0.5, 0.44, 0.06])
>>> #initial guess
>>> x0 = np.array([0.95, 0.025, 0.025]),
>>> w0 =  np.array([0.4, 0.5 , 0.1])
>>> y0 = np.array([0.15,0.8,0.05])
>>> vlle(x0, w0, y0, Z, T, P, model, full_output = True)
... T: 328.5
... P: 1.01
... error_outer: 8.996084220393732e-11
... error_inner: 1.546532851106652e-10
... iter: 2
... beta: array([0.41457829, 0.22478609, 0.36063562])
... tetha: array([0., 0., 0.])
... X: array([[0.946738  , 0.01222701, 0.04103499],
...       [0.23285379, 0.67120789, 0.09593832],
...       [0.15295408, 0.78764474, 0.05940118]])
... v: [None, None, None]
... states: ['L', 'L', 'V']


.. automodule:: phasepy.equilibrium.hazt
    :members: vlle
    :undoc-members:
    :show-inheritance:

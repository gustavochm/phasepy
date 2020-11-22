phasepy.mixture
===============

:class:`phasepy.mixture` object stores both pure component and mixture
related information and interaction parameters needed for equilibria
and interfacial properties computation.
Two pure components are required to create a base mixture:

.. code-block:: python

        >>> import numpy as np
	>>> from phasepy import component, mixture
	>>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861,
	                      Ant=[11.64785144, 3797.41566067, -46.77830444],
                              GC={'H2O':1})
	>>> ethanol = component(name='ethanol', Tc=514.0, Pc=61.37, Zc=0.241, Vc=168.0, w=0.643558,
                                Ant=[11.61809279, 3423.0259436, -56.48094263],
                                GC={'CH3':1, 'CH2':1, 'OH(P)':1})
	>>> mix = mixture(ethanol, water)

Additional components can be added to the mixture with
:func:`phasepy.mixture.add_component`.

.. code-block:: python

	>>> mtbe = component(name='mtbe', Tc=497.1, Pc=34.3, Zc=0.273, Vc=329.0, w=0.266059,
	                     Ant=[9.16238246, 2541.97883529, -50.40534341],
	                     GC={'CH3':3, 'CH3O':1, 'C':1})
	>>> mix.add_component(mtbe)

Once all components have been added to the mixture, the interaction
parameters must be supplied using a function depending on which model
will be used:

For quadratic mixing rule (QMR) used in cubic EoS:

>>> kij = np.array([[0, k12, k13],
                    [k21, 0, k23],
                    [k31, k32, 0]])
>>> mix.kij_cubic(kij)

For NRTL model:

>>> alpha = np.array([[0, alpha12, alpha13],
                      [alpha21, 0, alpha23],
                      [alpha31, alpha32, 0]])
>>> g = np.array([[0, g12, g13],
                  [g21, 0, g23],
                  [g31, g32, 0]])
>>> g1 = np.array([[0, gT12, gT13],
                   [gT21, 0, gT23],
                   [gT31, gT32, 0]])
>>> mix.NRTL(alpha, g, g1) 

For Wilson model:

>>> A = np.array([[0, A12, A13],
                  [A21, 0, A23],
                  [A31, A32, 0]])
>>> mix.wilson(A) 

For Redlich Kister parameters are set by polynomial by pairs, the
order of the pairs must be the following:
1-2, 1-3, ..., 1-n, 2-3, ..., 2-n, etc.

>>> C0 = np.array([poly12], [poly13], [poly23]]
>>> C1 = np.array([polyT12], [polyT13], [polyT23]]
>>> mix.rk(C0, C1)

For Modified-UNIFAC model, Dortmund public database must be read in:

>>> mix.unifac()

.. warning:: User is required to supply the necessary parameters for methods



.. autoclass:: phasepy.mixture
    :members:

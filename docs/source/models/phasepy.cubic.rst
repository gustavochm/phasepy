Cubic Equation of State Model
=============================

This phase equilibrium model class applies `equation of state (EoS)
model <https://en.wikipedia.org/wiki/Equation_of_state>`_ for both
vapor and liquid phases. EoS formulation is explicit:

.. math::
	P = \frac{RT}{v-b} - \frac{a}{(v+c_1b)(v+c_2b)}

Phasepy includes following cubic EoS:

- Van der Waals (VdW)
- Peng Robinson (PR)
- Redlich Kwong (RK)
- Redlich Kwong Soave (RKS), a.k.a Soave Redlich Kwong (SRK)
- Peng Robinson Stryjek Vera (PRSV)

Both pure component EoS and mixture EoS are supported.

Pure Component EoS
------------------

Pure component example using Peng-Robinson EoS:

>>> from phasepy import preos, component
>>> ethanol = component(name='ethanol', Tc=514.0, Pc=61.37, Zc=0.241, Vc=168.0, w=0.643558,
                        Ant=[11.61809279, 3423.0259436, -56.48094263],
                        GC={'CH3':1, 'CH2':1, 'OH(P)':1})
>>> eos = preos(ethanol)
>>> eos.psat(T=350.0) # saturation pressure, liquid volume and vapor volume
(array([0.98800647]), array([66.75754804]), array([28799.31921623]))

Density can be computed given the aggregation state (``L`` for liquid,
``V`` for vapor):

>>> eos.density(T=350.0, P=1.0, state='L')
0.01497960198094922
>>> eos.density(T=350.0, P=1.0, state='V')
3.515440899573752e-05


Volume Translation
------------------

Volume translated (VT) versions of EoS are available for PR, RK, RKS
and PRSV models. These models include an additional component specific
volume translation parameter ``c``, which can be used to improve
liquid density predictions without changing phase equilibrium.
EoS property ``volume_translation`` must be ``True`` to enable VT.

>>> ethanol = component(name='ethanol', Tc=514.0, Pc=61.37, Zc=0.241, Vc=168.0, w=0.643558,
                        Ant=[11.61809279, 3423.0259436, -56.48094263],
                        GC={'CH3':1, 'CH2':1, 'OH(P)':1},
                        c=5.35490936)
>>> eos = preos(ethanol, volume_translation=True)
>>> eos.psat(T=350.0) # saturation pressure, liquid volume and vapor volume
(array([0.98800647]), array([61.40263868]), array([28793.96430687]))
>>> eos.density(T=350.0, P=1.0, state='L')
0.01628597159790686
>>> eos.density(T=350.0, P=1.0, state='V')
3.5161028012629526e-05


Mixture EoS
-----------

Mixture EoS utilize one-fluid mixing rules, using parameters for
hypothetical pure fluids, to predict the mixture behavior. The mixing
rules require interaction parameter values as input (zero values are
assumed if no values are specified).

Classic Quadratic Mixing Rule (QMR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
	a_m = \sum_{i=1}^c \sum_{j=1}^c x_ix_ja_{ij} \quad a_{ij} =
	\sqrt{a_ia_j}(1-k_{ij}) \quad b_m = \sum_{i=1}^c x_ib_i

Example of Peng-Robinson with QMR:

>>> from phasepy import preos
>>> mix = mixture(ethanol, water)
>>> Kij = np.array([[0, -0.11], [-0.11, 0]])
>>> mix.kij_cubic(Kij)
>>> eos = preos(mix, mixrule='qmr')


Modified Huron Vidal (MHV) Mixing Rule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MHV mixing rule is specified in combination with an activity
coefficient model to solve EoS.
In MHV model, the repulsive (covolume) parameter is calcuated same way
as in QMR

.. math::
	b_m = \sum_{i=1}^c x_ib_i

while attractive term is an implicit function

.. math::
	g^e_{EOS} = g^e_{model}


Example of Peng-Robinson with MHV and NRTL:

>>> alpha = np.array([[0.0, 0.5597628],
                      [0.5597628, 0.0]])
>>> g = np.array([[0.0, -57.6880881],
                  [668.682368, 0.0]])
>>> g1 = np.array([[0.0, 0.46909821],
                   [-0.37982045, 0.0]])
>>> mix.NRTL(alpha, g, g1)
>>> eos = preos(mix, mixrule='mhv_nrtl')

Example of Peng-Robinson with MHV and Modified-UNIFAC:

>>> mix.unifac()
>>> eos = preos(mix, mixrule='mhv_unifac')


Wong Sandler (WS) Mixing Rule
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

WS mixing rule is specified in combination with an activity
coefficient model to solve EoS.

Example of Peng-Robinson with WS and Redlich Kister:

>>> C0 = np.array([1.20945699, -0.62209997,  3.18919339])
>>> C1 = np.array([-13.271128, 101.837857, -1100.29221])
>>> mix.rk(C0, C1)
>>> eos = preos(mix, mixrule='ws_rk')


.. automodule:: phasepy.cubic.cubic
   :members:
   :undoc-members:
   :show-inheritance:


EoS classes
-----------

.. toctree::
	phasepy.cubicp
	phasepy.cubicm


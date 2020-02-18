Cubic Equation of State
=======================
Equation of State for modeling vapor and liquid phases. This EoS is explicit in pressure and has the following form:

.. math::
	P = \frac{RT}{v-b} - \frac{a}{(v+c_1b)(v+c_2b)}

Where :math:`b` and :math:`a` are molecular parameters.

Main Cubic EoS functions of phasepy packages are bases on the following classes:

.. toctree::
	phasepy.cubicp
	phasepy.cubicm

This way, once you create an cubic EoS object it will check if you are working with a mixture or a pure component. The former case you will need to provide interaction parameters to the mixrule of the EoS. In the following code blocks you can see how to add interaction parameters to a mixture
and then how to specify a mixrule.

For the case of classic quadratic mixrule:

.. math::
	a_m = \sum_{i=1}^c \sum_{j=1}^c x_ix_ja_{ij} \quad a_{ij} = 	\sqrt{a_ia_j}(1-k_{ij}) \quad b_m = \sum_{i=1}^c x_ib_i


>>> from phasepy import preos
>>> mix = mixture(ethanol, water)
>>> Kij = np.array([[0, -0.11], [-0.11, 0]])
>>> mix.kij_cubic(Kij)
>>> pr = preos(mix, mixrule = 'qmr')

If no correction :math:`k_{ij}` is set, phasepy will consider it as zero.

In case of Modified Huron Vidal (MHV) and Wong Sandler (WS) mixing rule, it is necessary to provide information from a activity coefficient model in order to compute mixtures parameters. Covolume is calcuated same way as QMR.

.. math::
	b_m = \sum_{i=1}^c x_ib_i

While attractive term is an implicit function:

.. math::
	g^e_{EOS} = g^e_{model}



With NRTL model:

>>> alpha = np.array([[0.       , 0.5597628],
...       [0.5597628, 0.       ]])
>>> g = np.array([[  0.       , -57.6880881],
...        [668.682368 ,   0.       ]])
>>> g1 = np.array([[ 0.        ,  0.46909821],
...       [-0.37982045,  0.        ]])
>>> #Adding activity model parameters
>>> mix.NRTL(alpha, g, g1)
>>> pr = preos(mix, mixrule = 'mhv_nrtl')

In case of Modified Huron Vidal with UNIFAC:

>>> mix.unifac() #reading UNIFAC database
>>> pr = preos(mix, mixrule = 'mhv_unifac')

In case of Modified Huron Vidal with Redlich Kister Expansion:

>>> C0 = np.array([ 1.20945699, -0.62209997,  3.18919339])
>>> C1 = np.array([  -13.271128,   101.837857, -1100.29221 ])
>>> #Parameters are calculated as C = C0 + C1/T
>>> mix.rk(C0, C1)
>>> pr = preos(mix, mixrule = 'mhv_rk')

Phasepy has included the most widely known cubic EoS, as: Van der Waals, Peng Robinson, Redlich Kwong, Redlich Kwong Soave and Peng Robinson Stryjec Vera.

Additionally, volume translated versions are available for Peng Robinson, Redlich Kwong, Redlich Kwong Soave and Peng Robinson Stryjec Vera. 

van der Waals EoS
-----------------
.. automodule:: phasepy.cubic.cubic
    :members: vdweos
    :undoc-members:
    :show-inheritance:

Peng Robinson EoS
-----------------
.. automodule:: phasepy.cubic.cubic
    :members: preos
    :undoc-members:
    :show-inheritance:

Peng Robinson Stryjec Vera EoS
------------------------------
.. automodule:: phasepy.cubic.cubic
    :members: prsveos
    :undoc-members:
    :show-inheritance:

Redlich Kwong EoS
-----------------
.. automodule:: phasepy.cubic.cubic
    :members: rkeos
    :undoc-members:
    :show-inheritance:

Redlich Kwong Soave EoS
-----------------------
.. automodule:: phasepy.cubic.cubic
    :members: rkseos
    :undoc-members:
    :show-inheritance:




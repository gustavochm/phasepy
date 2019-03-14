Activity coefficient models
===========================

NRTL 
----
Non-Random-Two-Liquids model is a a local composition model, widely used to describe liquid vapour, liquid liquid and liquid liquid vapour equilibria. 

.. math::
	g^e = \sum_{i=1}^c \frac{\sum_{j=1}^c \tau_{ji}G_{ji}x_j}	{\sum_{l=1}^c G_{li}x_l}



.. automodule:: phasepy.actmodels.nrtl
    :members: nrtl
    :undoc-members:
    :show-inheritance:

Wilson
------

Wilson model is a local composiiton model recommended for vapour liquid equilibria calculation. It doesn't produce liquid liquid
equilibrium.

.. math::
	g^e = \sum_{i=1}^c x_i \ln ( \sum_{j=1}^c x_j 	\Lambda_{ij})

.. automodule:: phasepy.actmodels.wilson
    :members: wilson
    :undoc-members:
    :show-inheritance:

Redlich Kister
--------------
Non teorical model. Fit a Gibbs excess energy to a polynomial. It is not recommended to use more than 5 term of the expansion.

.. math::
	g^e_{ij} = x_ix_j \sum_{k=0}^m C_k (x_i - x_j)^k

.. automodule:: phasepy.actmodels.redlichkister
    :members: rk
    :undoc-members:
    :show-inheritance:

UNIFAC
------
Group contribution model. It uses Dortmund public database for VLE.

.. math::
	\ln \gamma_i = \ln \gamma_i^{comb} + \ln \gamma_i^{res}

.. automodule:: phasepy.actmodels.unifac
    :members: unifac
    :undoc-members:
    :show-inheritance:


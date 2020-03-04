SGT for mixtures and :math:`\beta_{ij} = 0`
===========================================

When working with mixtures, SGT solution procedure depends wether the influece parameter matrix is singular or not. The geometric mean rule leads to a singular matrix when all :math:`\beta_{ij} = 0`. In those cases the boundary value problem (BVP) can not be solved and alternative methods has to be used. Some of the options are the reference component method, which is the most popular. For this method the following system of equations has to be solved:

.. math::
	\sqrt{c_r} \left[ \mu_j(\rho) -  \mu_j^0 \right] = \sqrt{c_j} \left[ \mu_r(\rho)  -  \mu_r^0 \right] \qquad j \neq r 

Where the subscript :math:`r` refers to the reference component and :math:`j` to the other components present in the mixture. Alought implementation of this method is direct it may not be suitable for mixtures with several stationary points in the interface. In those cases a path function is recommended, Cornellise's doctoral thesis proposed the following path function: 

.. math::
	(dh)^2 = \sum_i  c_{i} d\rho_i^2

Which increased monotonically from one phase to another. One of the disadvantages of this path function is that its final lenght is not known beforehand and an iterative procedure with nested for loops is needed. For some of these reasons, Liang proposed the following path function: 

.. math::
	h = \sum_i \sqrt{c_i} \rho_i

This path function has a known value when the equilibrium densities are available. Also the solution procedure allows to formulate a auxiliar variable :math:`\alpha = (\mu_i - \mu_i^0)/\sqrt{c_i}`. This variable gives information about whether the geometric mean rule is suitable for the mixture. 


The ``sgt_mix_beta0`` function allows to compute interfacial tension and density profiles using SGT and :math:`\beta_{ij} = 0`, its use is showed in the following code block:




.. automodule:: phasepy.sgt.sgt_beta0
    :members: sgt_mix_beta0
    :undoc-members:
    :show-inheritance:

Individual functions for each method can be accesed trought the :class:`phasepy.sgt.ten_beta0_reference` for reference component method, :class:`phasepy.sgt.ten_beta0_hk` for Cornellise path function,  :class:`phasepy.sgt.ten_beta0_sk` for Liang path function. 


phasepy.sgt
===========

The Square Gradient Theory (SGT) is the reference framework when studying interfacial properties between fluid phases in equilibrium. It was originally proposed by van der Waals and then reformulated by Cahn and Hilliard.  SGT proposes that the Helmholtz free energy density at the interface can be described by a homogeneous and a gradient contribution.

.. math::
	a = a_0 + \frac{1}{2} \sum_i \sum_j c_{ij} \frac{d\rho_i}{dz} \frac{d\rho_j}{dz} + \cdots

Here :math:`a` is the Helmholtz free energy density, :math:`a_0` is the Helmholtz free energy density bulk contribution, :math:`c_{ij}` is the cross influence parameter between component :math:`i` and :math:`j`  , :math:`\rho` is denisty vector and :math:`z` is the lenght coordinate. The cross influence parameter is usually computed using a geometric mean rule and a correction :math:`c_{ij} = (1-\beta_{ij})\sqrt{c_i c_j}`.

The density profiles between the bulk phases are mean to minimize the energy of the system. This results in the following Euler-Lagrange system: 

.. math::
	\sum_j c_{ij} \frac{d^2 \rho_j}{dz^2} = \mu_i - \mu_i^0 \qquad i = 1,...,c

.. math::
	\rho(z \rightarrow -\infty) = \rho^\alpha \qquad \rho(z \rightarrow \infty) = \rho^\beta


Here :math:`\mu` represent the chemical potential and the superscript indicates its value evaluated at the bulk phase. :math:`\alpha` and :math:`\beta` are the bulk phases index.

Once the density profiles were solved the interfacial tension, :math:`\sigma` between the phases can be computed as:

.. math::
	\sigma =  \int_{-\infty}^{\infty} \sum_i \sum_j c_{ij} \frac{d\rho_i}{dz} \frac{d\rho_j}{dz} dz 

Solution procedure for SGT strongly depends for if you are working with a pure component or a mixture. In the latter, the correction value of :math:`\beta_{ij}` plays a huge role in the solution procedure. 
These cases will be covered.

.. toctree::
	./sgt/sgt.tenpure
	./sgt/sgt.tenbeta0
	./sgt/sgt.tensgt

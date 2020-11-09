phasepy.equilibrium
===================


Phase equilibrium conditions are obtained from a differential entropy balance of a system. The following equationes must be solved:

.. math::
	T^\alpha = T^\beta = ... &= T^\pi\\
	P^\alpha = P^\beta = ... &= P^\pi\\
	\mu_i^\alpha = \mu_i^\beta = ... &= \mu_i^\pi \quad i = 	1,...,c

Where :math:`T`, :math:`P` and :math:`\mu` are the temperature,
pressure and chemical potencial, :math:`\alpha`, :math:`\beta` and
:math:`\pi` are the phases and :math:`i` is component index.

For the continuous (:math:`\phi-\phi`) phase equilibrium approach, equilibrium is
defined using fugacity coefficients :math:`\phi`:

.. math::
	x_i^\alpha\hat{\phi}_i^\alpha = x_i^\beta \hat{\phi}_i^\beta = ... = x_i^\pi \hat{\phi}_i^\pi \quad i = 1,...,c

For the discontinuous (:math:`\gamma-\phi`) phase equilibrium
approach, the equilibrium is defined with vapor fugacity coefficient
and liquid phase activity coefficients :math:`\gamma`:

.. math::
	x_i^\alpha\hat{\gamma}_i^\alpha f_i^0 = x_i^\beta \hat{\gamma}_i^\beta f_i^0 = ... = x_i^\pi\hat{\phi}_i^\pi P \quad i = 1,...,c

where :math:`f_i^0` is standard state fugacity.


.. toctree::
   :maxdepth: 1

   ./equilibrium/phasepy.stability
   ./equilibrium/phasepy.elv
   ./equilibrium/phasepy.flash
   ./equilibrium/phasepy.multiflash

phasepy.equilibrium
===================

Phase equilibrium conditions are obtained from a differential entropy balance of a system. The following equationes must be solved:

.. math::
	T^\alpha = T^\beta = ... &= T^\pi\\
	P^\alpha = P^\beta = ... &= P^\pi\\
	\mu_i^\alpha = \mu_i^\beta = ... &= \mu_i^\pi \quad i = 	1,...,c

Where :math:`T`, :math:`P` and :math:`\mu` are the temperature, pressure and chemical potencial. When working with EoS usually equilibrium is guaranted by fugacity coefficients:

.. math::
	x_i^\alpha\hat{\phi}_i^\alpha = x_i^\beta \hat{\phi}_i^\beta = ... = x_i^\pi \hat{\phi}_i^\pi \quad i = 1,...,c


Usual equilibrium calculations includes vapour liquid equilibrium (flash, bubble point, dew point), liquid liquid equilibrium (flash and stability test) and liquid liquid vapour equilibrium (multiflash and stability test). Those algorithms are described in the following sections:

.. toctree::
	:maxdepth: 1

	./equilibrium/phasepy.flash
	./equilibrium/phasepy.elv
	./equilibrium/phasepy.ell
	./equilibrium/phasepy.haz
	./equilibrium/phasepy.stability
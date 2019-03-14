Virial EoS
==========

Ideal gas
---------
Recommended when working at low pressure.

.. math::

	Z = \frac{Pv}{RT} = 1

.. automodule:: phasepy.actmodels.virial
    :members: ideal_gas
    :undoc-members:
    :show-inheritance:

Abbott - Van Ness
-----------------

Correlation for the first virial coefficient, `B`:

.. math::
	\frac{BP_c}{RT_c} = B^{(0)} + \omega B^{(1)}

Where :math:`B^{(0)}` and :math:`B^{(1)}` are obtained from:

.. math::
	B^{(0)} &= 0.083 - \frac{0.422}{T_r^{1.6}}\\
	B^{(1)} &= 0.139 + \frac{0.179}{T_r^{4.2}}



.. automodule:: phasepy.actmodels.virial
    :members: Abbott
    :undoc-members:
    :show-inheritance:

Tsonopoulos
-----------
Correlation for the first virial coefficient, `B`:

.. math::
	\frac{BP_c}{RT_c} = B^{(0)} + \omega B^{(1)}

Where :math:`B^{(0)}` and :math:`B^{(1)}` are obtained from:

.. math::
	B^{(0)} &= 0.1445 - \frac{0.33}{T_r} - \frac{0.1385}{T_r^	2} - \frac{0.0121}{T_r^3} - \frac{0.000607}{T_r^8} \\
	B^{(1)} &= 0.0637 + \frac{0.331}{T_r^2} - \frac{0.423}{T_r^	3} - \frac{0.008}{T_r^8} 

.. automodule:: phasepy.actmodels.virial
    :members: Tsonopoulos
    :undoc-members:
    :show-inheritance:
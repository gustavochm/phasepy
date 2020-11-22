phasepy.component
=================

:class:`phasepy.component` object stores pure component information needed for equilibria and interfacial properties computation.
A component can be created as follows:

.. code-block:: python

	>>> from phasepy import component
	>>> water = component(name='water', Tc=647.13, Pc=220.55, Zc=0.229, Vc=55.948, w=0.344861,
                              Ant=[11.64785144, 3797.41566067, -46.77830444],
                              GC={'H2O':1})

Besides storing pure component data, the class incorporates basics methods for e.g. saturation pressure evaluation using Antoine equation, and liquid volume estimation with Rackett equation.

.. code-block:: python

	>>> water.psat(T=373.0) # vapor saturation pressure [bar]
	1.0072796747419537
	>>> water.vlrackett(T=310.0) # liquid molar volume [cm3/mol]
	16.46025809309672


.. warning:: User is required to supply the necessary parameters for methods

.. autoclass:: phasepy.component
    :members:


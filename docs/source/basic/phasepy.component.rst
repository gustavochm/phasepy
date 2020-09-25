phasepy.component
=================

Phasepy :class:`phasepy.component` class constructor function creates an object that stores necessary pure component information needed for equilibria and interfacial properties computation. The following parameters are allowed.

- name : name of the component
- Tc : Critical temperature in Kelvin
- Pc : Critical pressure in bar
- Zc : critical compresibility factor
- Vc : Critical volume in cm :math:`^3`/mol
- w : Acentric Factor
- c : Volume translation parameter used in cubic EoS in cm :math:`^3`/mol
- cii : polynomial coefficient for influence parameter in SGT, final units must be in J mol/m :math:`^5`.
- ksv : parameters to evaluate alpha functionof PRSV EoS
- Ant : Antoine correlation parameters
- GC : Group contribution information used in Modified-UNIFAC activity coefficient model, a list of possible groups can be found in `here <http://www.ddbst.com/PublishedParametersUNIFACDO.html#ListOfMainGroups>`_.


A component can be created as follows:

.. code-block:: python

	>>> from phasepy import component
	>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948, w = 0.344861,
        ...        Ant =  [  11.64785144, 3797.41566067,  -46.77830444],
        ...        GC = {'H2O':1})

Besides storing pure component data, the created object incorporates basics methods for saturation pressure evaluation using Antoine equation, and liquid volume estimation with Rackett equation.

.. code-block:: python

	>>> water.psat(T = 373)
	1.0072796747419537
	>>> water.vlrackett(T = 310)
	16.46025809309672


.. warning:: User is required to supply the necessary parameters for methods

.. autoclass:: phasepy.component
    :members:


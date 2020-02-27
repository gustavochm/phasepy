SGT for pure components
=======================

When working with pure components, SGT implementation is direct as there is a continuous path from the vapor to the liquid phase. SGT can be reformulated with density as indepedent variable.

.. math::
	\sigma = \sqrt{2} \int_{\rho^\alpha}^{\rho_\beta} \sqrt{c_i \Delta \Omega} d\rho

Here, :math:`\Delta \Omega` represents the grand thermodynamic potential, obtained from:

.. math:: 
	\Delta \Omega = a_0 - \rho \mu^0 + P^0

Where :math:`P^0` is the equilibrium pressure.

In phasepy this integration is done using ortoghonal collocation, which reduces the number of nodes needed for a desired error. This calculation is done with the ``sgt_pure`` function and it requires the equilibrium densities, temperature and pressure as inputs. 

>>> #component creation
>>> water =  component(name = 'Water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948, w = 0.344861
... ksv = [ 0.87185176, -0.06621339], cii = [2.06553362e-26, 2.64204784e-23, 4.10320513e-21])
>>> #EoS object creation
>>> eos = prsveos(water)

First vapor-liquid equilibria has to be computed. This is done with the ``psat`` method from the EoS, which returns the pressure and densities at equilibrium. Then the interfacial can be computed as it is shown.

>>> T = 350 #K
>>> Psat, vl, vv = eos.psat(T)
>>> rhol = 1/vl
>>> rhov = 1/vv
>>> sgt_pure(rhov, rhol, T, Psat, eos, full_output = False)
>>> #Tension in mN/m
... 63.25083234

Optionally, ``full_output`` allows to get all the computation information as the density profile, interfacial lenght and grand thermodynamic potential.

>>> solution = sgt_pure(rhol, rhov, T, Psat, eos, full_output = True)
>>> solution.z #interfacial lenght array
>>> solution.rho #density array
>>> solution.tension #IFT computed value


.. automodule:: phasepy.sgt.sgtpuros
    :members: sgt_pure
    :undoc-members:
    :show-inheritance:


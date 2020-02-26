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


Optionally, ``full_output`` allows to get all the computation information as the density profile, interfacial lenght and grand thermodynamic potential.

.. automodule:: phasepy.sgt.sgtpuros
    :members: sgt_pure
    :undoc-members:
    :show-inheritance:


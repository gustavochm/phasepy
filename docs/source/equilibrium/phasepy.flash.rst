Two phase Flash
===============
This is most classical calculation of phase equilibria. Temperature, pressure and global composition of a systems are known. If the mixture is unstable it will split into two o more phases. When trying to compute two phase separation the flash algorithm can be used.  Usual approach to solve this problem is to solve the following mass balance and then update composition by sucessive sustitution.

.. math::
	FO = \sum_{i=1}^c \left( x_i^\beta - x_i^\alpha \right) = \sum_{i=1}^c \frac{z_i (K_i-1)}{1+\psi (K_i-1)}

Where, :math:`z`, is the global composition of component :math:`K =  x^\beta / x^\alpha` are the equilibrium constant and :math:`\psi` is the phase fraction of phase :math:`\beta`. Subscript refers to component index and superscript refers to phase index.
This method can be slow at high pressures, and it that cases, the algorithm changes to a second order minimization of the Gibbs free energy of the system: 

.. math::
	min \, {G(\underline{F}^\alpha, \underline{F}^\beta)} = \sum_{i=1}^c (F_i^\alpha \ln \hat{f}_i^\alpha + F_i^\beta \ln \hat{f}_i^\beta)

Where, :math:`F` is the mole number and :math:`\hat{f}` is the effective fugacity.

In the following code block examples for solving vapor liquid is  shown: 


>>> from phasepy import component, mixture, preos
>>> from phasepy.equilibrium import flash
>>> water = component(name = 'water', Tc = 647.13, Pc = 220.55, Zc = 0.229, Vc = 55.948, w = 0.344861,
...                Ant =  [  11.64785144, 3797.41566067,  -46.77830444],
...                GC = {'H2O':1})
>>> ethanol = component(name = 'ethanol', Tc = 514.0, Pc = 61.37, Zc = 0.241, Vc = 168.0, w = 0.643558,
...                Ant = [  11.61809279, 3423.0259436 ,  -56.48094263],
...                GC = {'CH3':1, 'CH2':1,'OH(P)':1})
>>> #Mixture for Liquid Vapor flash
>>> mixelv = mixture(water, ethanol)
>>> mixelv.unifac()
>>> melv = preos(mixelv, 'mhv_unifac')
>>> T = 360 #K
>>> P = 1.01 #bar
>>> Z  = np.array([0.8,0.2]) #Global composition
>>> #initial guess
>>> x0 = np.array([0.1,0.9])
>>> y0 = np.array([0.2,0.8])
>>> flash(x0, y0,  'LV', Z, T, P, melv )
>>> #x, y, beta
(array([0.89794808, 0.10205192]), array([0.53414948, 0.46585052]), 0.26923709132954915)

The same algorithm can be applied for liquid-liquid equilibria, as can be seen:

>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                Ant = [   9.16238246, 2541.97883529,  -50.40534341], 
...                GC = {'CH3':3, 'CH3O':1, 'C':1})
>>> #Mixture for Liquid Liquid flash
>>> mixell = mixture(water, mtbe)
>>> mixell.unifac()
>>> mell = preos(mixell, 'mhv_unifac')
>>> T = 320 #K
>>> P = 1.01 #bar
>>> Z  = np.array([0.5,0.5])
>>> #initial guess
>>> x0 = np.array([0.01,0.99])
>>> y0 = np.array([0.99,0.01])
>>> flash(x0, y0,  'LL', Z, T, P, mell)
>>> #x, w, beta
(array([0.15604124, 0.84395876]), array([0.99289065, 0.00710935]), 0.4110193245129084)

.. automodule:: phasepy.equilibrium.flash
    :members: flash
    :undoc-members:
    :show-inheritance:
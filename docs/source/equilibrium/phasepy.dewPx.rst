dewPx
=====

>>> from phasepy import component, mixture, prsveos
>>> from phasepy.equilibrium import dewPx
>>> ethanol = component(name = 'ethanol', Tc = 514.0, Pc = 61.37, Zc = 0.241, Vc = 168.0, w = 0.643558,
...                ksv = [1.27092923, 0.0440421],
...                GC = {'CH3':1, 'CH2':1, 'OH(P)':1})
>>> mtbe = component(name = 'mtbe', Tc = 497.1, Pc = 34.3, Zc = 0.273, Vc = 329.0, w = 0.266059,
...                ksv = [0.76429651, 0.04242646],
...                GC = {'CH3':3, 'CH3O':1, 'C':1})
>>> mix = mixture(mtbe, ethanol)
>>> C0 = np.array([ 0.02635196, -0.02855964,  0.01592515])
>>> C1 = np.array([312.575789  ,  50.1476555 ,   5.13981131])
>>> mix.rk(C0, C1)
>>> eos = prsveos(mix, mixrule = 'mhv_rk')
>>> y = np.array([0.5,0.5])
>>> T = 340. #K
>>> x0 = np.array([0.2,0.8])
>>> P0 = 1.
>>> dewPx( x0, P0, y, T, eos)
>>> x, P
array([0.20224128, 0.79775872]), 1.0478247387561512

.. automodule:: phasepy.equilibrium.dew
    :members: dewPx
    :undoc-members:
    :show-inheritance:
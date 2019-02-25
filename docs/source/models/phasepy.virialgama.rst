Virial - Activity coefficient Model
===================================

This class is focused from phase equilibrium using a virial correlation for the
vapour phase and a activity coefficient model for the liquid phase. For the gas phase,
the virial coefficient can be estimated using Abbott or Tsonopoulos correlations, also 
an ideal gas option is available. For liquid phase, NRTL, Wilson, Redlich - Kister and UNIFAC
model are available. 

.. toctree::
	phasepy.virial
	phasepy.actmodels

Before creating a model object, it is necessary to supply the interactions parameters of the activity
coefficient model. The object for the mixture of water and ethanol created in the past section, would be
the following way for the NRTL model.

>>> from phasepy import virialgama, Abbott, nrtl
>>> alpha = np.array([[0.       , 0.5597628],
...       [0.5597628, 0.       ]])
>>> g = np.array([[  0.       , -57.6880881],
...       [668.682368 ,   0.       ]])
>>> g1 = np.array([[ 0.        ,  0.46909821],
...       [-0.37982045,  0.        ]])
>>> #Adding activity model parameters
>>> mix.NRTL(alpha, g, g1)
>>> #Creating Model
>>> model = virialgama(mix, virialmodel = Abbott, actmodel = nrtl )
>>> #Model parameters are saved in actmodelp attribute
>>> parameters = model.actmodelp
>>>np.all(parameters[0] == alpha), np.all(parameters[1] == g), np.all(parameters[2] == g1)
(True, True, True)

If you would like to use fitted parameters of a Redlich Kister expansion you can do it as follows:

>>> from phasepy import rk
>>> C0 = np.array([ 1.20945699, -0.62209997,  3.18919339])
>>> C1 = np.array([  -13.271128,   101.837857, -1100.29221 ])
>>> #Parameters are calculated as C = C0 + C1/T
>>> mix.rk(C0, C1)
>>> #Creating model
>>> model = virialgama(mix, actmodel = rk )
>>> #Readimg parameters
>>> parameters = model.actmodelp
>>> np.all(parameters[0] == C0), np.all(parameters[1] == C1)
(True, True)
>>> parameters[2]
[(0, 1)] 
>>> #Combinatory created by pairs, in case of mixtures with more componentes
>>> #it would be [(0,1), (0, 2), ..., (0, nc-1), (1, 2), (1, 3), ...]

If we would like to use the group contribution model UNIFAC with an ideal gas, the same class is used.

>>> from phasepy import unifac, ideal_gas
#Reading UNIFAC database
>>> mix.unifac()
#Creating Model
>>> model = virialgama(mix, virialmodel = ideal_gas, actmodel = unifac)

.. automodule:: phasepy.actmodels.virialgama
    :members: virialgama
    :undoc-members:
    :show-inheritance:

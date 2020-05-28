Virial - Activity coefficient Model
===================================

This class is focused for phase equilibrium using a virial correlation for the
vapor phase and a activity coefficient model for the liquid phase. For the gas phase,
the virial coefficient can be estimated using Abbott or Tsonopoulos correlations, also
an ideal gas option is available. For liquid phase, NRTL, Wilson, Redlich - Kister and Modified-UNIFAC
model are available.

.. toctree::
	phasepy.virial
	phasepy.actmodels

Before creating a model object, it is necessary to supply the interactions parameters of the activity
coefficient model. The mixture object is designed to store these parameters, as for example for the NRTL model.


>>> from phasepy import virialgama
>>> mix = mixture(ethanol, water)
>>> alpha = np.array([[0.       , 0.5597628],
...       [0.5597628, 0.       ]])
>>> g = np.array([[  0.       , -57.6880881],
...       [668.682368 ,   0.       ]])
>>> g1 = np.array([[ 0.        ,  0.46909821],
...       [-0.37982045,  0.        ]])
>>> #Adding activity model parameters
>>> mix.NRTL(alpha, g, g1)
>>> #Creating Model
>>> model = virialgama(mix, virialmodel = 'Abbott', actmodel = 'nrtl' )
>>> #Model parameters are saved in actmodelp attribute
>>> parameters = model.actmodelp
>>> np.all(parameters[0] == alpha), np.all(parameters[1] == g), np.all(parameters[2] == g1)
(True, True, True)

If you would like to use fitted parameters of a Redlich Kister expansion you can do it as follows:

>>> C0 = np.array([ 1.20945699, -0.62209997,  3.18919339])
>>> C1 = np.array([  -13.271128,   101.837857, -1100.29221 ])
>>> #Parameters are calculated as C = C0 + C1/T
>>> mix.rk(C0, C1)
>>> #Creating model
>>> model = virialgama(mix, actmodel = 'rk' )
>>> #Readimg parameters
>>> parameters = model.actmodelp
>>> np.all(parameters[0] == C0), np.all(parameters[1] == C1)
(True, True)

If we would like to use the group contribution model UNIFAC with an ideal gas, the same class can be used.

>>> #Reading UNIFAC database
>>> mix.unifac()
#Creating Model
>>> model = virialgama(mix, virialmodel = 'ideal_gas', actmodel = 'unifac')

.. automodule:: phasepy.actmodels.virialgama
    :members: virialgama
    :undoc-members:
    :show-inheritance:

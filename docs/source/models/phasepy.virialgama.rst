Virial - Activity coefficient Model
===================================

This phase equilibrium model class uses a virial correlation for the
vapor phase and an activity coefficient model for the liquid phase.
For the gas phase, the virial coefficient can be estimated
using ideal gas law, Abbott or Tsonopoulos correlations.
For liquid phase, NRTL, Wilson, Redlich-Kister and Modified-UNIFAC
activity coefficient models are available.

.. toctree::
	phasepy.virial
	phasepy.actmodels

Before creating a phase equilibrium model object, it is necessary to
supply the interactions parameters of the activity coefficient model
to the mixture object, for example using the NRTL model:

>>> from phasepy import virialgamma
>>> mix = mixture(ethanol, water)
>>> alpha = np.array([[0.0, 0.5597628],
...                   [0.5597628, 0.0]])
>>> g = np.array([[0.0, -57.6880881],
...               [668.682368, 0.0]])
>>> g1 = np.array([[0.0, 0.46909821],
...                [-0.37982045, 0.0]])
>>> mix.NRTL(alpha, g, g1)
>>> model = virialgamma(mix, virialmodel='Abbott', actmodel='nrtl' )

Parameters for Redlich-Kister are specified as follows:

>>> C0 = np.array([1.20945699, -0.62209997, 3.18919339])
>>> C1 = np.array([-13.271128, 101.837857, -1100.29221])
>>> mix.rk(C0, C1)
>>> model = virialgamma(mix, actmodel='rk')

Modified-UNIFAC with an ideal gas model is set up simply:

>>> mix.unifac()
>>> model = virialgamma(mix, virialmodel='ideal_gas', actmodel='unifac')

.. automodule:: phasepy.actmodels.virialgama
    :members: virialgamma
    :show-inheritance:

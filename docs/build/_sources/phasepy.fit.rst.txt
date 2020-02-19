phasepy.fit
===========

In order to compute phase equilibria and interfacial properties it is necessary to count with pure component parameters as: Antoine correlation parameters, volume translataion constant, influence parameter for SGT, among others. Similarly, when working with mixtures interaction parameters of activity coefficients models or binary correction :math:`k_{ij}` for quadratic mixing rule are necessary to predict phase equilibria. Often those parameters can be found in literature, but for many educational and research purposes there might be necessary to fit them to experimental data.

Phasepy includes several functions that relies on equilibria routines included in the package and in SciPy optimization tools for fitting models parameters. These functions are explained in the following secctions for pure component and for mixtures.


.. toctree::
	./fit/fit.pure
	./fit/fit.mixtures

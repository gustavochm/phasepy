# Phasepy Changelog

## v0.0.52

* Initial support for perfomning solid-liquid and solid-liquid-liquid equilibria. `component` function accepts now the enthalpy and temperature of fusion. These are needed to compute the solid phase fugacity coefficient.
* Functions `sle` and `slle` to compute solid-liquid and solid-liquid-liquid equilibria. Both function solve a flash that checks for phase stability.

## v0.0.51

* Now you can create personalized cubic EoS by choosing its alpha function and c1 and c2 parameters. See the `cubiceos` function.


## v0.0.50

* Now you can create mixtures by adding pure components (`+`)
* Bug in Wong-Sandler mixing rule fixed
* Updated Dortmund-UNIFAC database `(obtained July, 2021) <http://www.ddbst.com/PublishedParametersUNIFACDO.html>`_.
* MHV-1 mixing rule added for cubic equations of state


## v0.0.49

* UNIQUAC model added
* First Changelog!

# Phasepy Changelog

## v0.0.55
* Changed `cumtrapz` to `cumulative_trapezoid` function in the `path_sk` solver for SGT. (due to scipy deprecation)


## v0.0.54
* Updated the function `multiflash_solid` function used for both `sle` and `slle` solvers. The updated version allows controlings thresholds for the values of phase fractions (beta) and phase stability variables (tetha). The updated version also modified the Gibbs minimization step, to first do some iterations without derivative information. The errors from the minimization step now are consistent with the ASS step (`error_inner` refers to the mass balance and `error_outer` refers to the phase equilibria error). The `full_output` option now returns the method used to compute equilibria.

## v0.0.53
* Changed `np.int` to `int` (due to deprecation of `np.int`)

## v0.0.52

* Initial support for perfomning solid-liquid and solid-liquid-liquid equilibria. `component` function accepts now the enthalpy and temperature of fusion. These are needed to compute the solid phase fugacity coefficient.
* Functions `sle` and `slle` to compute solid-liquid and solid-liquid-liquid equilibria. Both function solve a flash that checks for phase stability.
* Fix bug in van der Waals EoS for mixtures. The object didn't have implemented methods for computing properties using `eos.temperature_aux`. 
* Added option to have an unbalanced `kij` matrix for QMR.

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

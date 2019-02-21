"""
phasepy: package for fluid phase equilibria with Python
=======================================================

Contents
-------------------------------------------------------

Available EoS
-------------
 Virial Gas: ideal_gas, Tsonopoulos, Abbott
 Activity Coefficient Models : NRTL, Wilson, UNIFAC, Redlich-Kister
 Cubic EoS: VdW, PR, PRSV, RK, RKS
 Mixrules: QMR, MHV

Available equilibrium routines (phasepy.equilibrium)
    LVE : Liquid Vapour Equilibrium
    LLE : Liquid Liquid Equilibrium
    LLVE : Liquid - Liquid - Vapour - Equilibrium

Available fitting routines (phasepy.fit)
    fit_kij : fit kij for qmr mixrule
    fit_nrtl : fit nrtl parameters
    fit_wilson : fit Wilson parameters
    fit_rk : fit Redlich Kister parameters
    
Interfacial properties (phasepy.sgt):
    ten_pure: SGT for pure fluids.
    ten_beta0 : SGT for mixtures with beta = 0 
    (Reference component, Path functions,linear approximation, spot approximation)
    

"""
__all__ = [s for s in dir() if not s.startswith('_')]

from .mixtures import *
from .cubic.cubic import *
from .actmodels import *
from . import actmodels
from . import cubic
from . import equilibrium
from . import fit
from . import sgt
from .math import *
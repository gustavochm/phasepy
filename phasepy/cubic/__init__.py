"""
phasepy.cubic: cubic equation of state with Python
=======================================================

Cubic EoS
---------
vdw : van der Waals EoS
pr :  Peng-Robinson EoE
prsv : Peng-Robinson-Stryjec-Vera EoS
rk : Redlich-Kwong EoS
rsv : Redlich-Kwong-Soave EoS

Available mixrules
------------------

qmr : quadratic mixrule
mhv_nrtl : Modified Huron Vidal mixing rule with NRTL model
mhv_wilson : Modified Huron Vidal mixing rule with Wilson model
mhv_unifac : Modified Huron Vidal mixing rule with Wilson model
mhv_rk : Modified Huron Vidal mixing rule with Redlich-Kister model

ws_nrtl : Wong-Sandler mixing rule with NRTL model
ws_wilson :  Wong-Sandler mixing rule with Wilson model
ws_unifac :  Wong-Sandler mixing rule with Wilson model
ws_rk :  Wong-Sandler mixing rule with Redlich-Kister model
"""

from __future__ import division, print_function, absolute_import
from .cubic import *

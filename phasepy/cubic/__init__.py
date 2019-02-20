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

qmr : quadratic mixrule
mhv_nrtl : Modified Huron Vidal mixrule with NRTL model
mhv_wilson : Modified Huron Vidal mixrule with Wilson model
mhv_unifac : Modified Huron Vidal mixrule with Wilson model
mhv_rk : Modified Huron Vidal mixrule with Redlich-Kister model
"""


from .cubic import *
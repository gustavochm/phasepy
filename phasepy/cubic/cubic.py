from .vdwpure import vdwpure
from .vdwmix import vdwm
from .cubicpure import prpure, prsvpure, rkspure, rkpure
from .cubicmix import prmix, prsvmix, rksmix, rkmix

def vdw(mix_or_component):
    nc = mix_or_component.nc
    if nc == 1:
        eos = vdwpure(mix_or_component)
    else:
        eos = vdwm(mix_or_component)
    return eos      
            
def pr(mix_or_component, mixrule = 'qmr'):
    nc = mix_or_component.nc
    if nc == 1:
        eos = prpure(mix_or_component)
    else:
        eos = prmix(mix_or_component, mixrule)
    return eos
            
def prsv(mix_or_component, mixrule = 'qmr'):
    nc = mix_or_component.nc
    if nc == 1:
        eos = prsvpure(mix_or_component)
    else:
        eos = prsvmix(mix_or_component, mixrule)
    return eos

def rk(mix_or_component, mixrule = 'qmr'):
    nc = mix_or_component.nc
    if nc == 1:
        eos = rkpure(mix_or_component)
    else:
        eos = rkmix(mix_or_component, mixrule)
    return eos

def rks(mix_or_component, mixrule = 'qmr'):
    nc = mix_or_component.nc
    if nc == 1:
        rkspure(mix_or_component)
    else:
        eos = rksmix(mix_or_component, mixrule)
    return eos

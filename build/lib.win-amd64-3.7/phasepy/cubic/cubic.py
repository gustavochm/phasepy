

from __future__ import division, print_function, absolute_import
from .vdwpure import vdwpure
from .vdwmix import vdwm
from .cubicpure import prpure, prsvpure, rkspure, rkpure
from .cubicmix import prmix, prsvmix, rksmix, rkmix

from .vtcubicpure import vtprpure, vtprsvpure, vtrkspure, vtrkpure
from .vtcubicmix import  vtprmix, vtprsvmix, vtrksmix, vtrkmix

def vdweos(mix_or_component):
    '''
    van der Waals EoS
    
    Parameters
    ----------
    mix_or_component : object
        created with component or mixture, in case of mixture object has to
        have interactions parameters.
                        
    Returns
    -------   
    eos : object
        eos used for phase equilibrium calculations
    '''
    nc = mix_or_component.nc
    if nc == 1:
        eos = vdwpure(mix_or_component)
    else:
        eos = vdwm(mix_or_component)
    return eos      
            
def preos(mix_or_component, mixrule = 'qmr',volume_traslation = False):
    '''
    Peng Robinson EoS
    
    Parameters
    ----------
    mix_or_component : object
        created with component or mixture, in case of mixture object has to
        have interactions parameters.
    mixrule : str
        available opitions 'qmr', 'mhv_nrtl', 'mhv_unifac', 'mhv_rk',
        'mhv_wilson'
        
    Returns
    -------   
    eos : object
        eos used for phase equilibrium calculations
    '''
    nc = mix_or_component.nc
    if nc == 1:
        if volume_traslation:
            eos = vtprpure(mix_or_component)
        else:
            eos = prpure(mix_or_component)
    else:
        if volume_traslation:
            eos = vtprmix(mix_or_component)
        else:
            eos = prmix(mix_or_component)
    return eos
            
def prsveos(mix_or_component, mixrule = 'qmr' ,volume_traslation = False):
    '''
    Peng Robinson EoS
    
    Parameters
    ----------
    mix_or_component : object
        created with component or mixture, in case of mixture object has to
        have interactions parameters.
    mixrule : str
        available opitions 'qmr', 'mhv_nrtl', 'mhv_unifac', 'mhv_rk', 
        'mhv_wilson'
    
    Returns
    -------   
    eos : object
        eos used for phase equilibrium calculations
    '''
    nc = mix_or_component.nc

        
        
    if nc == 1:
        if volume_traslation:
            eos = vtprsvpure(mix_or_component)
        else:
            eos = prsvpure(mix_or_component)
    else:
        if volume_traslation:
            eos = vtprsvmix(mix_or_component)
        else:
            eos = prsvmix(mix_or_component)
    return eos

def rkeos(mix_or_component, mixrule = 'qmr',volume_traslation = False):
    '''
    Redlich Kwong EoS
    
    Parameters
    ----------
    mix_or_component : object
        created with component or mixture, in case of mixture object has to
        have interactions parameters.
    mixrule : str
        available opitions 'qmr', 'mhv_nrtl', 'mhv_unifac', 'mhv_rk',
        'mhv_wilson'
    
    Returns
    -------   
    eos : object
        eos used for phase equilibrium calculations
    '''
    nc = mix_or_component.nc
        
    if nc == 1:
        if volume_traslation:
            eos = vtrkpure(mix_or_component)
        else:
            eos = rkpure(mix_or_component)
    else:
        if volume_traslation:
            eos = vtrkmix(mix_or_component)
        else:
            eos = rkmix(mix_or_component)
    return eos

def rkseos(mix_or_component, mixrule = 'qmr',volume_traslation = False):
    '''
    Redlich Kwong Soave EoS
    
    Parameters
    ----------
    mix_or_component : object
        created with component or mixture, in case of mixture object has to
        have interactions parameters.
    mixrule : str
        available opitions 'qmr', 'mhv_nrtl', 'mhv_unifac', 'mhv_rk',
        'mhv_wilson'
    
    Returns
    -------   
    eos : object
        eos used for phase equilibrium calculations      
    '''
    nc = mix_or_component.nc
    if nc == 1:
        if volume_traslation:
            eos = vtrkspure(mix_or_component)
        else:
            eos = rkspure(mix_or_component)
    else:
        if volume_traslation:
            eos = vtrksmix(mix_or_component)
        else:
            eos = rksmix(mix_or_component)
    
    return eos

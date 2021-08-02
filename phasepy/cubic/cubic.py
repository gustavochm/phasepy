from __future__ import division, print_function, absolute_import
from .vdwpure import vdwpure
from .vdwmix import vdwm
from .cubicpure import prpure, prsvpure, rkspure, rkpure
from .cubicmix import prmix, prsvmix, rksmix, rkmix

from .vtcubicpure import vtprpure, vtprsvpure, vtrkspure, vtrkpure
from .vtcubicmix import vtprmix, vtprsvmix, vtrksmix, vtrkmix


def vdweos(mix_or_component):
    '''
    Returns Van der Waals EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`phasepy.mixture` or :class:`phasepy.component` object
    '''
    nc = mix_or_component.nc
    if nc == 1:
        eos = vdwpure(mix_or_component)
    else:
        eos = vdwm(mix_or_component)
    return eos


def preos(mix_or_component, mixrule='qmr', volume_translation=False):
    '''
    Returns Peng Robinson EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`phasepy.mixture` or :class:`phasepy.component` object
    mixrule : str
        Mixing rule specification. Available opitions include 'qmr',
        'mhv_nrtl', 'mhv_unifac', 'mhv_rk', 'mhv_wilson', 'mhv_uniquac',
        'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac', 'ws_uniquac',
        'mhv1_nrtl', 'mhv1_unifac', 'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac'.
    volume_translation: bool
        If True, the volume translated version of this EoS will be used.

    '''
    nc = mix_or_component.nc
    if nc == 1:
        if volume_translation:
            eos = vtprpure(mix_or_component)
        else:
            eos = prpure(mix_or_component)
    else:
        if volume_translation:
            eos = vtprmix(mix_or_component, mixrule)
        else:
            eos = prmix(mix_or_component, mixrule)
    return eos


def prsveos(mix_or_component, mixrule='qmr', volume_translation=False):
    '''
    Returns Peng Robinson Stryjek Vera EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`phasepy.mixture` or :class:`phasepy.component` object
    mixrule : str
        Mixing rule specification. Available opitions include 'qmr',
        'mhv_nrtl', 'mhv_unifac', 'mhv_rk', 'mhv_wilson', 'mhv_uniquac',
        'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac', 'ws_uniquac',
        'mhv1_nrtl', 'mhv1_unifac', 'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac'.
    volume_translation: bool
        If True, the volume translated version of this EoS will be used.
    '''
    nc = mix_or_component.nc

    if nc == 1:
        if volume_translation:
            eos = vtprsvpure(mix_or_component)
        else:
            eos = prsvpure(mix_or_component)
    else:
        if volume_translation:
            eos = vtprsvmix(mix_or_component, mixrule)
        else:
            eos = prsvmix(mix_or_component, mixrule)
    return eos


def rkeos(mix_or_component, mixrule='qmr', volume_translation=False):
    '''Returns Redlich Kwong EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`phasepy.mixture` or :class:`phasepy.component` object
    mixrule : str
        Mixing rule specification. Available opitions include 'qmr',
        'mhv_nrtl', 'mhv_unifac', 'mhv_rk', 'mhv_wilson', 'mhv_uniquac',
        'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac', 'ws_uniquac',
        'mhv1_nrtl', 'mhv1_unifac', 'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac'.
    volume_translation: bool
        If True, the volume translated version of this EoS will be used.
    '''
    nc = mix_or_component.nc

    if nc == 1:
        if volume_translation:
            eos = vtrkpure(mix_or_component)
        else:
            eos = rkpure(mix_or_component)
    else:
        if volume_translation:
            eos = vtrkmix(mix_or_component, mixrule)
        else:
            eos = rkmix(mix_or_component, mixrule)
    return eos


def rkseos(mix_or_component, mixrule='qmr', volume_translation=False):
    '''Returns Redlich Kwong Soave EoS object.

    Parameters
    ----------
    mix_or_component : object
        :class:`phasepy.mixture` or :class:`phasepy.component` object
    mixrule : str
        Mixing rule specification. Available opitions include 'qmr',
        'mhv_nrtl', 'mhv_unifac', 'mhv_rk', 'mhv_wilson', 'mhv_uniquac',
        'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac', 'ws_uniquac',
        'mhv1_nrtl', 'mhv1_unifac', 'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac'.
    volume_translation: bool
        If True, the volume translated version of this EoS will be used.
    '''
    nc = mix_or_component.nc

    if nc == 1:
        if volume_translation:
            eos = vtrkspure(mix_or_component)
        else:
            eos = rkspure(mix_or_component)
    else:
        if volume_translation:
            eos = vtrksmix(mix_or_component, mixrule)
        else:
            eos = rksmix(mix_or_component, mixrule)

    return eos

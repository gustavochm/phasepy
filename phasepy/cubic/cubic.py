from __future__ import division, print_function, absolute_import
import numpy as np
from .cubicmix import cubicm
from .cubicpure import cpure
from .alphas import alpha_soave
from .vtcubicpure import vtcpure
from .vtcubicmix import vtcubicm

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
        'mhv_original_unifac', 'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac',
        'ws_uniquac', 'ws_original_unifac, mhv1_nrtl', 'mhv1_unifac',
        'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac, 'mhv1_original_unifac'
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
        'mhv_original_unifac', 'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac',
        'ws_uniquac', 'ws_original_unifac, mhv1_nrtl', 'mhv1_unifac',
        'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac, 'mhv1_original_unifac'
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
        'mhv_original_unifac', 'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac',
        'ws_uniquac', 'ws_original_unifac, mhv1_nrtl', 'mhv1_unifac',
        'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac, 'mhv1_original_unifac'
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
        'mhv_original_unifac', 'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac',
        'ws_uniquac', 'ws_original_unifac, mhv1_nrtl', 'mhv1_unifac',
        'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac, 'mhv1_original_unifac'
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


# generic object for any cubic eos
c1pr = 1.-np.sqrt(2.)
c2pr = 1.+np.sqrt(2.)


def cubiceos(mix_or_component, c1=c1pr, c2=c2pr, alpha_eos=alpha_soave,
             mixrule='qmr', volume_translation=False):
    '''Returns cubic EoS object. with custom c1, c2, alpha function and
    mixing rule.

    Parameters
    ----------
    mix_or_component : object
        :class:`phasepy.mixture` or :class:`phasepy.component` object
    c1, c2 : float
        Constants of the cubic EoS
    alpha_eos : function
        Function that returns the alpha parameter of the cubic EoS.
        alpha_eos(T, alpha_params, Tc)
    mixrule : str
        Mixing rule specification. Available opitions include 'qmr',
        'mhv_nrtl', 'mhv_unifac', 'mhv_rk', 'mhv_wilson', 'mhv_uniquac',
        'mhv_original_unifac', 'ws_nrtl', 'ws_wilson', 'ws_rk', 'ws_unifac',
        'ws_uniquac', 'ws_original_unifac, mhv1_nrtl', 'mhv1_unifac',
        'mhv1_rk', 'mhv1_wilson', 'mhv1_uniquac, 'mhv1_original_unifac'
    volume_translation: bool
        If True, the volume translated version of this EoS will be used.
    '''
    # Getting oma and omb for the given c1 and c2 constants of the cubic EoS
    if c1 == 0. and c2 == 0.:
        raise ValueError('For c1 and c2 equal to zero use vdweos function')

    a0 = (2. + c1 + c2)**3
    a1 = -3 * (-5 + c1**2 - 5*c2 + c2**2 - c1 * (5 + 7*c2))
    a2 = 3*(2 + c1 + c2)
    a3 = -1
    poly = np.array([a0, a1, a2, a3])
    roots = np.roots(poly)
    roots = roots[np.imag(roots) == 0.]
    roots = np.real(roots)
    roots = roots[roots > 0.]
    omb = np.min(roots)
    oma = (1 + omb * (2.+c1+c2+(1.+c1+c2+c1**2.-c1*c2+c2**2.) * omb)) / 3

    nc = mix_or_component.nc
    if nc == 1:
        if volume_translation:
            eos = vtcpure(pure=mix_or_component, c1=c1, c2=c2, oma=oma, 
                          omb=omb, alpha_eos=alpha_eos)
        else:
            eos = cpure(pure=mix_or_component, c1=c1, c2=c2, oma=oma, omb=omb,
                        alpha_eos=alpha_eos)
    else:  # mixture
        if volume_translation:
            eos = vtcubicm(mix=mix_or_component, c1=c1, c2=c2, oma=oma,
                           omb=omb, alpha_eos=alpha_eos, mixrule=mixrule)
        else:
            eos = cubicm(mix=mix_or_component, c1=c1, c2=c2, oma=oma, omb=omb,
                         alpha_eos=alpha_eos, mixrule=mixrule)

    return eos

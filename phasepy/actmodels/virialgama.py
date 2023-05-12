from __future__ import division, print_function, absolute_import
import numpy as np
from .virial import Tsonopoulos, ideal_gas, Abbott, Virialmix
from .nrtl import nrtl, dnrtl, nrtlter, dnrtlter
from .nrtl import nrtl_aux, dnrtl_aux, nrtlter_aux, dnrtlter_aux
from .redlichkister import rkb, rk, drk
from .redlichkister import rkb_aux, rk_aux, drk_aux
from .wilson import wilson, dwilson
from .wilson import wilson_aux, dwilson_aux
from .unifac import unifac, dunifac
from .unifac import unifac_aux, dunifac_aux
from .original_unifac import unifac_original, dunifac_original
from .original_unifac import unifac_original_aux, dunifac_original_aux
from .uniquac import uniquac, duniquac
from .uniquac import uniquac_aux, duniquac_aux
from ..constants import R, r


class virialgamma():
    '''
    Returns a phase equilibrium model with mixture using a virial EOS
    to describe vapour phase, and an activity coefficient model for
    liquid phase.

    Parameters
    ----------
    mix : object
        mixture created with mixture class
    virialmodel : string
        function to compute virial coefficients, available options are
        'Tsonopoulos', 'Abbott' or 'ideal_gas'
    actmodel : string
        function to compute activity coefficients, available optiones are
        'nrtl', 'wilson', 'original_unifac', 'unifac', 'uniquac', 'rkb' or 'rk'

    Methods
    -------
    temperature_aux: computes temperature dependent parameters.
    logfugef: computes effective fugacity coefficients.
    dlogfugef: computes effective fugacity coefficients and its
        composition derivatives.
    lngama: computes activity coefficients.
    dlngama: computes activity coefficients and its
        composition derivatives.
    '''

    def __init__(self, mix, virialmodel='Tsonopoulos', actmodel='nrtl'):

        self.psat = mix.psat
        self.vl = mix.vlrackett
        self.mezcla = mix
        self.nc = mix.nc
        self.Tij, self.Pij, self.Zij, self.wij = Virialmix(mix)
        self.dHf = np.asarray(mix.dHf)
        self.Tf = np.asarray(mix.Tf)
        self.dHf_r = np.asarray(mix.dHf) / r

        if virialmodel == 'Tsonopoulos':
            self.virialmodel = Tsonopoulos
        elif virialmodel == 'Abbott':
            self.virialmodel = Abbott
        elif virialmodel == 'ideal_gas':
            self.virialmodel = ideal_gas
        else:
            raise Exception('Virial model not implemented')

        if actmodel == 'nrtl':
            if hasattr(mix, 'g') and hasattr(mix, 'alpha'):
                self.actmodel = nrtl
                self.dactmodel = dnrtl
                self.actmodel_aux = nrtl_aux
                self.dactmodel_aux = dnrtl_aux
                self.actmodelp = (mix.alpha, mix.g, mix.g1)
                self.secondorder = True

                def actm_temp(self, T):
                    alpha, g, g1 = self.actmodelp
                    tau = g/T + g1
                    G = np.exp(-alpha*tau)
                    aux = (tau, G)
                    return aux
                self.actm_temp = actm_temp.__get__(self)
            else:
                raise Exception('NRTL parameters needed')

        elif actmodel == 'nrtlt':
            bool1 = hasattr(mix, 'g')
            bool2 = hasattr(mix, 'alpha')
            bool3 = hasattr(mix, 'rkternary')
            if bool1 and bool2 and bool3:
                self.actmodel = nrtlter
                self.dactmodel = dnrtlter
                self.actmodel_aux = nrtlter_aux
                self.dactmodel_aux = dnrtlter_aux
                self.actmodelp = (mix.alpha, mix.g, mix.g1, mix.rkternary)
                self.secondorder = True

                def actm_temp(self, T):
                    alpha, g, g1, rkternary = self.actmodelp
                    tau = g/T + g1
                    G = np.exp(-alpha*tau)
                    aux = (tau, G, rkternary)
                    return aux
                self.actm_temp = actm_temp.__get__(self)
            else:
                raise Exception('NRTL/ternary parameters needed')

        elif actmodel == 'wilson':
            if hasattr(mix, 'Aij'):

                self.actmodelp = (mix.Aij, mix.vlrackett)
                self.actmodel = wilson
                self.dactmodel = dwilson
                self.actmodel_aux = wilson_aux
                self.dactmodel_aux = dwilson_aux
                self.secondorder = True

                def actm_temp(self, T):
                    Aij, vlrackett = self.actmodelp
                    vl = vlrackett(T)
                    M = np.divide.outer(vl, vl).T * np.exp(-Aij/T)
                    aux = (M, )
                    return aux
                self.actm_temp = actm_temp.__get__(self)
            else:
                raise Exception('Wilson parameters needed')
        elif actmodel == 'uniquac':
            bool1 = hasattr(mix, 'ri')
            bool2 = hasattr(mix, 'qi')
            bool3 = hasattr(mix, 'a0') and hasattr(mix, 'a1')
            if bool1 and bool2 and bool3:
                self.actmodelp = (mix.ri, mix.qi, mix.a0, mix.a1)
                self.actmodel = uniquac
                self.dactmodel = duniquac
                self.actmodel_aux = uniquac_aux
                self.dactmodel_aux = duniquac_aux
                self.secondorder = True

                def actm_temp(self, T):
                    ri, qi, a0, a1 = self.actmodelp
                    Aij = a0 + a1 * T
                    tau = np.exp(-Aij/T)
                    aux = (ri, qi, tau)
                    return aux
                self.actm_temp = actm_temp.__get__(self)
            else:
                raise Exception('UNIQUAC parameters needed')

        elif actmodel == 'rk':
            if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT'):
                self.actmodelp = (mix.rkp, mix.rkpT, mix.combinatory)
                self.actmodel = rk
                self.dactmodel = drk
                self.actmodel_aux = rk_aux
                self.dactmodel_aux = drk_aux
                self.secondorder = True

                def actm_temp(self, T):
                    C, C1, combinatory = self.actmodelp
                    G = C + C1 / T
                    aux = (G, combinatory)
                    return aux
                self.actm_temp = actm_temp.__get__(self)
            else:
                raise Exception('RK parameters needed')

        elif actmodel == 'rkb':
            if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT') and mix.nc == 2:
                self.actmodelp = (mix.rkp.flatten(), mix.rkpT.flatten())
                self.actmodel = rkb
                self.actmodel_aux = rkb_aux
                self.dactmodel = None
                self.secondorder = False

                def actm_temp(self, T):
                    C, C1 = self.actmodelp
                    G = C + C1 / T
                    aux = (G.flatten(), )
                    return aux
                self.actm_temp = actm_temp.__get__(self)
            else:
                raise Exception('RK parameters needed')

        elif actmodel == 'unifac':
            mix.unifac()
            if hasattr(mix, 'actmodelp'):
                self.actmodel = unifac
                self.dactmodel = dunifac
                self.actmodel_aux = unifac_aux
                self.dactmodel_aux = dunifac_aux
                self.actmodelp = mix.actmodelp
                self.secondorder = True

                def actm_temp(self, T):
                    qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2 = self.actmodelp
                    amn = a0 + a1 * T + a2 * T**2
                    psi = np.exp(-amn/T)
                    aux = (qi, ri, ri34, Vk, Qk, tethai, amn, psi)
                    return aux
                self.actm_temp = actm_temp.__get__(self)
            else:
                raise Exception('Unifac parameters needed')
        elif actmodel == 'original_unifac':
            mix.original_unifac()
            if hasattr(mix, 'actmodelp'):
                self.actmodel = unifac_original
                self.dactmodel = dunifac_original
                self.actmodel_aux = unifac_original_aux
                self.dactmodel_aux = dunifac_original_aux
                self.actmodelp = mix.actmodelp
                self.secondorder = True

                def actm_temp(self, T):
                    qi, ri, Vk, Qk, tethai, amn = self.actmodelp
                    psi = np.exp(-amn/T)
                    aux = (qi, ri, Vk, Qk, tethai, psi)
                    return aux
                self.actm_temp = actm_temp.__get__(self)
            else:
                raise Exception('Original-Unifac parameters needed')
        else:
            raise Exception('Activity Coefficient Model not implemented')

    def temperature_aux(self, T):
        RT = R*T
        Bij = self.virialmodel(T, self.Tij, self.Pij, self.wij)
        psat = self.psat(T)
        vl = self.vl(T)
        actmp = self.actm_temp(T)
        aux = (RT, Bij, psat, vl, actmp, T)
        return aux

    def logfugef_aux(self, X, temp_aux, P, state, v0=None):
        (RT, Bij, psat, vl, actmp, T) = temp_aux

        if state == 'L':
            Bi = np.diag(Bij)
            pointing = vl*(P-psat)/(RT)
            fugPsat = Bi*psat/(RT)
            act = self.actmodel_aux(X, *actmp)
            return act+np.log(psat/P)+pointing+fugPsat, v0

        elif state == 'V':
            Bx = Bij*X
            Bm = np.sum(Bx.T*X)
            Bp = 2*np.sum(Bx, axis=1) - Bm
            return Bp*P/(RT), v0

    def dlogfugef_aux(self, X, temp_aux, P, state, v0=None):
        RT, Bij, psat, vl, actmp, T = temp_aux

        if state == 'L':
            Bi = np.diag(Bij)
            pointing = vl*(P-psat)/RT
            fugPsat = Bi*psat/RT
            act, dact = self.dactmodel_aux(X, *actmp)
            logfug = act+np.log(psat/P)+pointing+fugPsat
            dxidnj = np.eye(self.nc) - X
            dlogfug = dact@dxidnj.T

        elif state == 'V':
            Bx = Bij*X
            Bm = np.sum(Bx.T*X)
            Bp = 2*np.sum(Bx, axis=1) - Bm
            logfug = Bp*P/RT
            dlogfug = (2*Bij - np.add.outer(Bp, Bp))*P/RT

        return logfug, dlogfug, v0

    def logfugef(self, X, T, P, state, v0=None):
        """
        Returns array of effective fugacity coefficients at given
        composition, temperature and pressure as first return value, and
        passes through argument v0 as second value.

        Parameters
        ----------

        X : array
            molar fractions
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase, or 'V' for vapour phase
        v0 : float, optional
            volume of phase

        Returns
        -------
        logfug: array_like
            effective fugacity coefficients
        v0 : float
            volume of phase, if calculated
        """
        Bij = self.virialmodel(T, self.Tij, self.Pij, self.wij)
        RT = R*T
        if state == 'L':
            Bi = np.diag(Bij)
            psat = self.psat(T)
            pointing = self.vl(T)*(P-psat)/(RT)
            fugPsat = Bi*psat/(RT)
            act = self.actmodel(X, T, *self.actmodelp)
            return act+np.log(psat/P)+pointing+fugPsat, v0
        elif state == 'V':
            Bx = Bij*X
            Bm = np.sum(Bx.T*X)
            Bp = 2*np.sum(Bx, axis=1) - Bm
            return Bp*P/(RT), v0

    def dlogfugef(self, X, T, P, state, v0=None):
        """
        Returns array of effective fugacity coefficients at given
        composition, temperature and pressure as first return value,
        array of partial fugacity coefficients as second return value,
        and passes through argument v0 as third value.

        Parameters
        ----------

        X : array
            molar fractions
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase, or 'V' for vapour phase
        v0 : float, optional
            volume of phase

        Returns
        -------
        logfug: array_like
            effective fugacity coefficients
        dlogfug: array_like
            derivatives of effective fugacity coefficients
        v0 : float
            volume of phase, if calculated
        """
        RT = R*T
        Bij = self.virialmodel(T, self.Tij, self.Pij, self.wij)
        if state == 'L':
            Bi = np.diag(Bij)
            psat = self.psat(T)
            pointing = self.vl(T)*(P-psat)/(R*T)
            fugPsat = Bi*psat/(R*T)
            act, dact = self.dactmodel(X, T, *self.actmodelp)
            logfug = act+np.log(psat/P)+pointing+fugPsat
            dxidnj = np.eye(self.nc) - X
            dlogfug = dact@dxidnj.T

        elif state == 'V':
            Bx = Bij*X
            Bm = np.sum(Bx.T*X)
            Bp = 2*np.sum(Bx, axis=1) - Bm
            logfug = Bp*P/(R*T)
            dlogfug = (2*Bij - np.add.outer(Bp, Bp))*P/(R*T)

        return logfug, dlogfug, v0

    def lngama(self, X, T):
        """
        Computes the natural logarithm of activy coefficients.

        Parameters
        ----------
        X : array
            molar fractions
        T : float
            absolute temperature [K]

        Returns
        -------
        lngama: array_like
            activity coefficients
        """
        gamas = self.actmodel(X, T, *self.actmodelp)
        return gamas

    def dlngama(self, X, T):
        """
        Computes the natural logarithm of activy coefficients and its
        composition derivatives matrix.

        Parameters
        ----------
        X : array
            molar fractions
        T : float
            absolute temperature [K]

        Returns
        -------
        lngama: array_like
            activity coefficient
        dlngama: array_like
            derivatives of the activity coefficients
        """
        gamas, dgama = self.dactmodel(X, T, *self.actmodelp)
        return gamas, dgama

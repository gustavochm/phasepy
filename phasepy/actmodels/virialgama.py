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
from ..constants import R


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
        'nrtl', 'wilson', 'unifac', 'rkb' or 'rk'
    '''

    def __init__(self, mix, virialmodel='Tsonopoulos', actmodel='nrtl'):

        self.psat = mix.psat
        self.vl = mix.vlrackett
        self.mezcla = mix
        self.nc = mix.nc
        self.Tij, self.Pij, self.Zij, self.wij = Virialmix(mix)

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
            bool3 = hasattr(mix, 'rkternario')
            if bool1 and bool2 and bool3:
                self.actmodel = nrtlter
                self.dactmodel = dnrtlter
                self.actmodel_aux = nrtlter_aux
                self.dactmodel_aux = dnrtlter_aux
                self.actmodelp = (mix.alpha, mix.g, mix.g1, mix.rkternario)
                self.secondorder = True

                def actm_temp(self, T):
                    alpha, g, g1, rkternario = self.actmodelp
                    tau = g/T + g1
                    G = np.exp(-alpha*tau)
                    aux = (tau, G, rkternario)
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
        else:
            raise Exception('Activity Coefficient Model not implemented')

    def temperature_aux(self, T):
        RT = R*T
        Bij = self.virialmodel(T, self.Tij, self.Pij, self.wij)
        psat = self.psat(T)
        vl = self.vl(T)
        actmp = self.actm_temp(T)
        aux = (RT, Bij, psat, vl, actmp)
        return aux

    def logfugef_aux(self, X, temp_aux, P, state, v0=None):
        (RT, Bij, psat, vl, actmp) = temp_aux

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
        RT, Bij, psat, vl, actmp = temp_aux

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
        """
        Bij = self.virialmodel(T, self.Tij, self.Pij, self.wij)
        if state == 'L':
            Bi = np.diag(Bij)
            psat = self.psat(T)
            pointing = self.vl(T)*(P-psat)/(R*T)
            fugPsat = Bi*psat/(R*T)
            act = self.actmodel(X, T, *self.actmodelp)
            return act+np.log(psat/P)+pointing+fugPsat, v0
        elif state == 'V':
            Bx = Bij*X
            Bm = np.sum(Bx.T*X)
            Bp = 2*np.sum(Bx, axis=1) - Bm
            return Bp*P/(R*T), v0

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

        """
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

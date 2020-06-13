from __future__ import division, print_function, absolute_import
import numpy as np
from .virial import Tsonopoulos, ideal_gas, Abbott, Virialmix
from .nrtl import nrtl, dnrtl, nrtlter, dnrtlter
from .redlichkister import rkb, rk, drk
from .wilson import wilson, dwilson
from .unifac import unifac, dunifac

R = 83.14  # bar cm3/mol K


class virialgamma:
    '''
    Creates a model with mixture using a virial eos to describe vapour phase
    and an activity coefficient model for liquid phase.

    Parameters
    ----------
    mix : object
        mixture created with mixture class
    virialmodel : string
        function to compute virial coefficients, available options are 'Tsonopoulos',
        'Abbott' or 'ideal_gas'
    actmodel : string
        function to compute activity coefficients, available optiones are 'nrtl',
        'wilson', 'unifac', 'rkb' or 'rk'

    Methods
    -------
    logfugef: computes effective fugacity coefficients

    '''

    def __init__(self, mix, virialmodel = 'Tsonopoulos', actmodel = 'nrtl'):

        self.psat = mix.psat
        self.vl = mix.vlrackett
        self.mezcla = mix
        self.nc = mix.nc
        self.Tij, self.Pij, self.Zij, self.wij = Virialmix(mix)

        if virialmodel ==  'Tsonopoulos':
            self.virialmodel = Tsonopoulos
        elif virialmodel ==  'Abbott':
            self.virialmodel = Abbott
        elif virialmodel == 'ideal_gas':
            self.virialmodel = ideal_gas
        else:
            raise Exception('Virial model not implemented')

        if actmodel == 'nrtl':
            if hasattr(mix, 'g') and hasattr(mix, 'alpha'):
                self.actmodel = nrtl
                self.dactmodel = dnrtl
                self.actmodelp = (mix.alpha, mix.g, mix.g1)
                self.secondorder = True
            else:
                raise Exception('NRTL parameters needed')

        elif actmodel == 'nrtlt':
            if hasattr(mix, 'g') and hasattr(mix, 'alpha') and hasattr(mix, 'rkternario'):
                self.actmodel = nrtlter
                self.dactmodel = dnrtlter
                self.actmodelp = (mix.alpha, mix.g, mix.g1, mix.rkternario)
                self.secondorder = True
            else:
                raise Exception('NRTL/ternary parameters needed')

        elif actmodel == 'wilson':
            if hasattr(mix, 'Aij'):
                #este se utiliza con mhv_wilson
                self.actmodelp = (mix.Aij, mix.vlrackett)
                self.actmodel = wilson
                self.dactmodel = dwilson
                self.secondorder = True
            else:
                raise Exception('Wilson parameters needed')

        elif actmodel == 'rk':
            if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT'):
                self.actmodelp = (mix.rkp, mix.rkpT, mix.combinatory)
                self.actmodel = rk
                self.dactmodel = drk
                self.secondorder = True
            else:
                raise Exception('RK parameters needed')

        elif actmodel == 'rkb':
            if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT') and mix.nc ==2:
                self.actmodelp = (mix.rkp, mix.rkpT)
                self.actmodel = rkb
                self.dactmodel = None
                self.secondorder = False
            else:
                raise Exception('RK parameters needed')

        elif actmodel == 'unifac':
            mix.unifac()
            if hasattr(mix, 'actmodelp'):
                self.actmodel = unifac
                self.dactmodel = dunifac
                self.actmodelp = mix.actmodelp
                self.secondorder = True
            else:
                raise Exception('Unifac parameters needed')
        else:
            raise Exception('Activity Coefficient Model not implemented')


    def logfugef(self, X, T, P, state, v0 = None):
        """
        logfugef(X, T, P, state)

        Method that computes the effective fugacity coefficients  at given
        composition, temperature and pressure.

        Parameters
        ----------

        X : array_like, mole fraction vector
        T : absolute temperature in K
        P : pressure in bar
        state : 'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        logfug: array_like
            effective fugacity coefficients
        v0 : float
            volume of phase, if calculated
        """
        Bij = self.virialmodel(T, self.Tij, self.Pij, self.wij)
        #Bi, Bp = virial(X, T, self.Tij, self.Pij, self.wij, self.virialmodel)
        if state == 'L':
            Bi = np.diag(Bij)
            psat = self.psat(T)
            pointing = self.vl(T)*(P-psat)/(R*T)
            fugPsat = Bi*psat/(R*T)
            act = self.actmodel(X, T, *self.actmodelp)
            return act+np.log(psat/P)+pointing+fugPsat, v0
        elif state == 'V':
            Bx = Bij*X
            #virial de mezcla
            Bm = np.sum(Bx.T*X)
            #virial parcial
            Bp = 2*np.sum(Bx, axis=1) - Bm
            return Bp*P/(R*T), v0

    def dlogfugef(self, X, T, P, state, v0 = None):
        """
        logfugef(X, T, P, state)

        Method that computes the effective fugacity coefficients  at given
        composition, temperature and pressure.

        Parameters
        ----------

        X : array_like, mole fraction vector
        T : absolute temperature in K
        P : pressure in bar
        state : 'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        logfug: array_like
            effective fugacity coefficients
        v0 : float
            volume of phase, if calculated
        """
        Bij = self.virialmodel(T, self.Tij, self.Pij, self.wij)
        #Bi, Bp = virial(X, T, self.Tij, self.Pij, self.wij, self.virialmodel)
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
            #virial de mezcla
            Bm = np.sum(Bx.T*X)
            #virial parcial
            Bp = 2*np.sum(Bx, axis=1) - Bm
            logfug = Bp*P/(R*T)
            dlogfug = (2*Bij - np.add.outer(Bp, Bp))*P/(R*T)

        return logfug, dlogfug, v0



from __future__ import division, print_function, absolute_import
import numpy as np
from .alphas import alpha_vdw
from .psatpure import psat
from ..constants import R

class vdwpure():
    '''
    Mixture VdW EoS Object

    This object have implemeted methods for phase equilibrium
    as for iterfacial properties calculations.

    Parameters
    ----------
    pure : object
        pure component created with component class

    Attributes
    ----------
    Tc: critical temperture in K
    Pc: critical pressure in bar
    w: acentric factor
    cii : influence factor for SGT
    nc : number of components of mixture

    Methods
    -------
    a_eos : computes the attractive term of cubic eos.
    Zmix : computes the roots of compresibility factor polynomial.
    density : computes density of mixture.
    logfugef : computes effective fugacity coefficients.
    logfugmix : computes mixture fugacity coeficcient;
    a0ad : computes adimentional Helmholtz density energy
    muad : computes adimentional chemical potential.
    dOm : computes adimentional Thermodynamic Grand Potential.
    ci :  computes influence parameters matrix for SGT.
    sgt_adim : computes adimentional factors for SGT.

    '''
    def __init__(self, pure):

        self.c1 = 0
        self.c2 = 0
        self.oma = 27/64
        self.omb = 1/8
        self.alpha_eos = alpha_vdw
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))


        self.Tc = np.array(pure.Tc, ndmin = 1)
        self.Pc = np.array(pure.Pc, ndmin = 1)
        self.w = np.array(pure.w, ndmin = 1)
        self.cii = np.array(pure.cii, ndmin = 1)
        self.b = self.omb*R*self.Tc/self.Pc


    def __call__(self, T, v):
        b = self.b
        a = self.a_eos(T)
        c1 = self.c1
        c2 = self.c2
        return R*T/(v - b) - a/((v+c1*b)*(v+c2*b))

    def a_eos(self,T):
        """
        a_eos(T)

        Method that computes atractive term of cubic eos at fixed T (in K)

        Parameters
        ----------

        T : float
            absolute temperature in K


        Returns
        -------
        a : float
            atractive term array
        """
        alpha = self.alpha_eos()
        return self.oma*(R*self.Tc)**2*alpha/self.Pc

    def psat(self, T, P0 = None):
        """
        psat(T, P0)

        Method that computes saturation pressure at fixed T

        Parameters
        ----------

        T : float
            absolute temperature in K
        P0 : float, optional
            initial value to find saturation pressure in bar

        Returns
        -------
        psat : float
            saturation pressure
        """
        return psat(T, self, P0)

    def _Zroot(self,A,B):
        a1 = (self.c1+self.c2-1)*B-1
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a3 = -B*(self.c1*self.c2*(B**2+B)+A)
        Zpol=[1, a1, a2, a3]
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots>B]
        return Zroots

    def density(self, T, P, state):
        """
        density(T, P, state)
        Method that computes the density of the mixture at T, P


        Parameters
        ----------

        T : float
            absolute temperature in K
        P : float
            pressure in bar
        state : string
            'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        density: float
            density in moll/cm3

        """
        A = self.a_eos(T) * P /(R*T)**2
        B = self.b* P / (R*T)

        if state == 'L':
            Z=min(self._Zroot(A,B))
        if state == 'V':
            Z=max(self._Zroot(A,B))

        return P/(R*T*Z)


    def _logfug_aux(self,Z, A, B):

        logfug=Z-1-np.log(Z-B)
        logfug -= A/Z

        return logfug

    def logfug(self, T, P, state):
        """
        logfug(T, P, state)

        Method that computes the fugacity coefficient at given
        composition, temperature and pressure.

        Parameters
        ----------
        T : float
            absolute temperature in K
        P : float
            pressure in bar
        state : string
            'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        logfug: float
            fugacity coefficient
        """


        A = self.a_eos(T) * P /(R*T)**2
        B = self.b* P / (R*T)

        if state == 'L':
            Z=min(self._Zroot(A,B))
        if state == 'V':
            Z=max(self._Zroot(A,B))

        logfug = self._logfug_aux(Z, A, B)

        return logfug


    def a0ad(self, ro,T):
        """
        a0ad(ro, T)

        Method that computes the adimenstional Helmholtz density energy at given
        density and temperature.

        Parameters
        ----------

        ro : float
            adimentional density vector
        T : float
            absolute adimentional temperature

        Returns
        -------
        a0ad: float
            adimenstional Helmholtz density energy
        """

        Pref = 1
        a0 = -T*ro*np.log(1-ro)
        a0 += -T*ro*np.log(Pref/(T*ro))
        a0 += -ro**2

        return a0

    def muad(self, ro, T):
        """
        muad(ro, T)

        Method that computes the adimenstional chemical potential at given
        density and temperature.

        Parameters
        ----------

        roa : float
            adimentional density vector
        T : float
            absolute adimentional temperature

        Returns
        -------
        muad: float
            chemical potential
        """
        Pref = 1.
        mu = -T*np.log(1-ro)
        mu += -T*np.log(Pref/(T*ro)) + T
        mu += T*ro/(1-ro)
        mu -= 2*ro

        return mu

    def dOm(self, roa, Tad, mu0, Psat):
        """
        dOm(roa, T, mu, Psat)

        Method that computes the adimenstional Thermodynamic Grand potential at given
        density and temperature.

        Parameters
        ----------

        roa : float
            adimentional density vector
        T : floar
            absolute adimentional temperature
        mu : float
            adimentional chemical potential at equilibrium
        Psat : float
            adimentional pressure at equilibrium

        Returns
        -------
        Out: float, Thermodynamic Grand potential
        """


        return self.a0ad(roa, Tad)- roa*mu0 + Psat

    def ci(self, T):
        '''
        ci(T)

        Method that evaluates the polynomial for the influence parameters used
        in the SGT theory for surface tension calculations.

        Parameters
        ----------
        T : float
            absolute temperature in K

        Returns
        -------
        ci: float
            influence parameters
        '''
        return np.polyval(self.cii, T)

    def sgt_adim(self, T):
        '''
        sgt_adim(T)

        Method that evaluates adimentional factor for temperature, pressure,
        density, tension and distance for interfacial properties computations with
        SGT.

        Parameters
        ----------
        T : float
            absolute temperature in K

        Returns
        -------
        Tfactor : float
            factor to obtain dimentionless temperature (K -> adim)
         Pfactor : float
             factor to obtain dimentionless pressure (bar -> adim)
        rofactor : float
            factor to obtain dimentionless density (mol/cm3 -> adim)
        tenfactor : float
            factor to obtain dimentionless surface tension (mN/m -> adim)
        zfactor : float
            factor to obtain dimentionless distance  (Amstrong -> adim)

        '''
        a = self.a_eos(T)
        b = self.b
        ci = self.ci(T)
        Tfactor = R*b/a
        Pfactor = b**2/a
        rofactor = b
        tenfactor = 1000*np.sqrt(a*ci)/b**2*(np.sqrt(101325/1.01325)*100**3)
        zfactor = np.sqrt(a/ci*10**5/100**6)*10**-10
        return Tfactor, Pfactor, rofactor, tenfactor, zfactor

from __future__ import division, print_function, absolute_import
import numpy as np
from .qmr import qmr
from .alphas import alpha_vdw
from ..constants import R, r


class vdwm():
    '''
    Mixture VdW EoS Object

    This object have implemeted methods for phase equilibrium
    as for iterfacial properties calculations.

    Parameters
    ----------
    mix : object
        mixture created with mixture class

    Attributes
    ----------
    Tc: critical temperture [K]
    Pc: critical pressure [bar]
    w: acentric factor
    cii : influence parameter for SGT polynomial [J m5 mol-2]
    nc : number of components of mixture
    Mw : molar weight [g mol-1]

    Methods
    -------
    a_eos : computes the attractive term of cubic eos.
    Zmix : computes the roots of compressibility factor polynomial.
    density : computes density of mixture.
    logfugef : computes effective fugacity coefficients.
    logfugmix : computes mixture fugacity coeficcient;
    a0ad : computes adimentional Helmholtz density energy
    muad : computes adimentional chemical potential.
    dOm : computes adimentional Thermodynamic Grand Potential.
    ci :  computes influence parameters matrix for SGT.
    sgt_adim : computes adimentional factors for SGT.
    beta_sgt : method that incorporates the beta correction for SGT.
    EntropyR : computes residual Entropy.
    EnthalpyR: computes residual Enthalpy.
    CvR : computes residual isochoric heat capacity.
    CpR : computes residual isobaric heat capacity.
    speed_sound : computes the speed of sound.

    '''

    def __init__(self, mix):

        self.c1 = 0
        self.c2 = 0
        self.oma = 27/64
        self.omb = 1/8
        self.alpha_eos = alpha_vdw
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))

        self.Tc = np.array(mix.Tc, ndmin=1)
        self.Pc = np.array(mix.Pc, ndmin=1)
        self.w = np.array(mix.w, ndmin=1)
        self.cii = np.array(mix.cii, ndmin=1)
        self.b = self.omb*R*self.Tc/self.Pc
        self.nc = mix.nc
        self.beta = np.zeros([self.nc, self.nc])
        self.secondorder = False
        self.secondordersgt = False
        self.Mw = np.array(mix.Mw, ndmin=1)

        self.mixrule = qmr
        if hasattr(mix, 'kij'):
            self.kij = mix.kij
        else:
            self.kij = np.zeros([mix.nc, mix.nc])
        self.mixruleparameter = (self.kij,)

        # fusion and melting point (needed for SLE and SLLE)
        self.dHf = np.asarray(mix.dHf)
        self.Tf = np.asarray(mix.Tf)
        self.dHf_r = np.asarray(mix.dHf) / r

    # EoS methods
    def a_eos(self, T):
        """
        a_eos(T)

        Method that computes atractive term of cubic eos at fixed T (in K)

        Parameters
        ----------

        T : float
            absolute temperature [K]

        Returns
        -------
        a : array_like
            atractive term array [bar cm6 mol-2]
        """
        alpha = self.alpha_eos()
        return self.oma*(R*self.Tc)**2*alpha/self.Pc

    def temperature_aux(self, T):
        RT = R*T
        ai = self.a_eos(T)
        temp_aux = (RT, T, ai, self.mixruleparameter)
        return temp_aux

    def pressure(self, X, v, T):
        """
        pressure(X, v, T)

        Method that computes the pressure at given composition X,
        volume (cm3/mol) and temperature T (in K)

        Parameters
        ----------
        X : array_like
            mole fraction vector
        v : float
            molar volume in [cm3/mol]
        T : float
            absolute temperature [K]

        Returns
        -------
        P : float
            pressure [bar]
        """
        a = self.a_eos(T)
        am, bm = self.mixrule(X, T, a, self.b, 0, *self.mixruleparameter)
        RT = R * T

        P = RT/(v - bm) - am / v**2
        return P

    def _Zroot(self, A, B):
        a1 = -1*B-1
        a2 = +A
        a3 = -B*A

        Zpol = np.hstack([1., a1, a2, a3])
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots > B]
        return Zroots

    # Auxiliar method that computes volume roots
    def _volume_solver(self, P, RT, D, B, state, v0):

        Dr = D*P/RT**2
        Br = B*P/RT
        if state == 'L':
            Z = np.min(self._Zroot(Dr, Br))
        elif state == 'V':
            Z = np.max(self._Zroot(Dr, Br))
        else:
            raise Exception('Valid states: L for liquids and V for vapor ')
        V = (RT*Z)/P
        return V

    def Zmix(self, X, T, P):
        '''
        Zmix (X, T, P)

        Method that computes the roots of the compressibility factor polynomial
        at given mole fractions (X), Temperature (T) and Pressure (P)

        Parameters
        ----------

        X : array_like
            mole fraction vector
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]

        Returns
        -------
        Z : array_like
            roots of Z polynomial
        '''
        a = self.a_eos(T)
        am, bm = self.mixrule(X, T, a, self.b, 0, *self.mixruleparameter)
        A = am*P/(R*T)**2
        B = bm*P/(R*T)
        return self._Zroot(A, B)

    def density(self, X, T, P, state):
        """
        Method that computes the molar concentration (molar density)
        of the mixture at given composition (X), temperature (T) and
        pressure (P)

        Parameters
        ----------

        X : array_like
            mole fraction vector
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        density: float
            Molar concentration of the mixture [mol/cm3]
        """
        if state == 'L':
            Z = min(self.Zmix(X, T, P))
        elif state == 'V':
            Z = max(self.Zmix(X, T, P))
        return P/(R*T*Z)

    def logfugef_aux(self, X, temp_aux, P, state, v0=None):

        RT, T, a, mixruleparameter = temp_aux

        am, ai, bm, bp = self.mixrule(X, T, a, self.b, 1,
                                      *mixruleparameter)

        if state == 'V':
            Z = max(self.Zmix(X, T, P))
        elif state == 'L':
            Z = min(self.Zmix(X, T, P))

        V = Z * RT / P

        B = (bm*P)/(RT)
        A = (am*P)/(RT)**2

        logfug = (Z-1)*(bp/bm)-np.log(Z-B)
        logfug -= A*(ai/am - bp/bm)/Z

        return logfug, V

    def logfugef(self, X, T, P, state, v0=None):
        """
        logfugef(X, T, P, state)

        Method that computes the effective fugacity coefficients  at given
        composition, temperature and pressure.

        Parameters
        ----------

        X : array_like,
            mole fraction vector
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0 : float, optional
            initial volume to iterate [cm3/mol]

        Returns
        -------
        logfug: array_like
            effective fugacity coefficients
        v : float
            volume of the mixture [cm3/mol]
        """
        temp_aux = self.temperature_aux(T)
        logfug, V = self.logfugef_aux(X, temp_aux, P, state, v0)
        return logfug, V

    def logfugmix_aux(self, X, temp_aux, P, state, v0=None):
        RT, T, a, mixruleparameter = temp_aux
        am, bm = self.mixrule(X, T, a, self.b, 0, *mixruleparameter)

        if state == 'V':
            Z = max(self.Zmix(X, T, P))
        elif state == 'L':
            Z = min(self.Zmix(X, T, P))

        V = Z * RT / P
        B = (bm*P)/(RT)
        A = (am*P)/(RT)**2

        logfug = Z-1-np.log(Z-B)
        logfug -= A/Z

        return logfug, V

    def logfugmix(self, X, T, P, state, v0=None):
        """
        logfugmix(X, T, P, state)

        Method that computes the mixture fugacity coefficient at given
        composition, temperature and pressure.

        Parameters
        ----------

        X : array_like
            mole fraction vector
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        lofgfug : array_like
            effective fugacity coefficients
        v : float
            volume of phase [cm3/mol]
        """
        temp_aux = self.temperature_aux(T)
        logfug, V = self.logfugmix_aux(X, temp_aux, P, state, v0)

        return logfug, V

    def a0ad_aux(self, rhoa, temp_aux):
        RT, T, ai, mixruleparameter = temp_aux
        bi = self.b
        a = ai[0]
        b = bi[0]
        ro = np.sum(rhoa)
        X = rhoa/ro

        am, bm = self.mixrule(X, T, ai, bi, 0, *mixruleparameter)
        Prefa = 1*b**2/a
        Tad = RT*b/a
        ama = am/a
        bma = bm/b

        a0 = np.sum(np.nan_to_num(Tad*rhoa*np.log(rhoa)))
        a0 += -Tad*ro*np.log(1-bma*ro)
        a0 += -Tad*ro*np.log(Prefa/Tad)
        a0 += -ama*ro**2

        return a0

    def a0ad(self, rhoa, T):
        """
        a0ad(roa, T)

        Method that computes the adimenstional Helmholtz density energy at
        given density and temperature.

        Parameters
        ----------

        rhoa : array_like
            adimentional density vector [rhoa = rho * b[0]]
        T : float
            absolute temperature [K]

        Returns
        -------
        a0ad: float
            adimenstional Helmholtz density energy
        """
        temp_aux = self.temperature_aux(T)
        a0 = self.a0ad_aux(rhoa, temp_aux)

        """
        ai = self.a_eos(T)
        bi = self.b
        a = ai[0]
        b = bi[0]
        ro = np.sum(rhoa)
        X = rhoa/ro

        am, bm = self.mixrule(X, T, ai, bi, 0, *self.mixruleparameter)
        Prefa = 1*b**2/a
        Tad = R*T*b/a
        ama = am/a
        bma = bm/b

        a0 = np.sum(np.nan_to_num(Tad*rhoa*np.log(rhoa)))
        a0 += -Tad*ro*np.log(1-bma*ro)
        a0 += -Tad*ro*np.log(Prefa/Tad)
        a0 += -ama*ro**2
        """
        return a0

    def muad_aux(self, rhoa, temp_aux):
        RT, T, ai, mixruleparameter = temp_aux
        bi = self.b

        ro = np.sum(rhoa)
        X = rhoa/ro

        am, aip, bm, bp = self.mixrule(X, T, ai, bi, 1, *mixruleparameter)

        a = ai[0]
        b = bi[0]

        ap = aip - am

        Prefa = 1.*b**2/a
        Tad = R*T*b/a
        apa = ap/a
        ama = am/a
        bma = bm/b
        bad = bp/b

        mui = -Tad*np.log(1-bma*ro)
        mui += -Tad*np.log(Prefa/(Tad*rhoa))+Tad
        mui += bad*Tad*ro/(1-bma*ro)
        mui -= ro*(apa+ama)
        return mui

    def muad(self, rhoa, T):
        """
        muad(roa, T)

        Method that computes the adimenstional chemical potential at given
        density and temperature.

        Parameters
        ----------

        rhoa : array_like
            adimentional density vector [rhoa = rho * b[0]]
        T : float
            absolute temperature [K]

        Returns
        -------
        muad : array_like
            adimentional chemical potential vector
        """
        temp_aux = self.temperature_aux(T)
        mui = self.muad_aux(rhoa, temp_aux)

        """
        ai = self.a_eos(T)
        bi = self.b

        ro = np.sum(rhoa)
        X = rhoa/ro

        am, aip, bm, bp = self.mixrule(X, T, ai, bi, 1, *self.mixruleparameter)

        a = ai[0]
        b = bi[0]

        ap = aip - am

        Prefa = 1.*b**2/a
        Tad = R*T*b/a
        apa = ap/a
        ama = am/a
        bma = bm/b
        bad = bp/b

        mui = -Tad*np.log(1-bma*ro)
        mui += -Tad*np.log(Prefa/(Tad*rhoa))+Tad
        mui += bad*Tad*ro/(1-bma*ro)
        mui -= ro*(apa+ama)
        """
        return mui

    def dOm_aux(self, rhoa, temp_aux, mu, Psat):
        a0ad = self.a0ad_aux(rhoa, temp_aux)
        dom = a0ad - np.sum(np.nan_to_num(rhoa*mu)) + Psat
        return dom

    def dOm(self, rhoa, T, mu, Psat):
        """
        dOm(roa, T, mu, Psat)

        Method that computes the adimenstional Thermodynamic Grand potential
        at given density and temperature.

        Parameters
        ----------

        rhoa : array_like
            adimentional density vector [rhoa = rho * b[0]]
        T : float
            absolute temperature [K]
        mu : array_like
            adimentional chemical potential at equilibrium [adim]
        Psat : float
            adimentional pressure at equilibrium [adim]

        Returns
        -------
        dom: float
            Thermodynamic Grand potential
        """

        temp_aux = self.temperature_aux(T)
        dom = self.dOm_aux(rhoa, temp_aux, mu, Psat)

        return dom

    def lnphi0(self, T, P):

        nc = self.nc
        a_puros = self.a_eos(T)
        Ai = a_puros*P/(R*T)**2
        Bi = self.b*P/(R*T)
        a1 = (self.c1+self.c2-1)*Bi-1
        a2 = self.c1*self.c2*Bi**2-(self.c1+self.c2)*(Bi**2+Bi)+Ai
        a3 = -Bi*(self.c1*self.c2*(Bi**2+Bi)+Ai)

        pols = np.array([a1, a2, a3])
        Zs = np.zeros([nc, 2])
        for i in range(nc):
            zroot = np.roots(np.hstack([1, pols[:, i]]))
            zroot = zroot[zroot > Bi[i]]
            Zs[i, :] = np.array([max(zroot), min(zroot)])

        lnphi = self.logfug(Zs.T, Ai, Bi)
        lnphi = np.amin(lnphi, axis=0)

        return lnphi

    def beta_sgt(self, beta):
        '''
        beta_sgt

        Method that allow asigning the beta correction for the influence
        parameter in Square Gradient Theory.

        Parameters
        ----------
        beta : array_like
            beta corrections for influence parameter
        '''

        nc = self.nc
        BETA = np.asarray(beta)
        shape = BETA.shape

        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(BETA, BETA.T)

        if isSquare and isSymmetric:
            self.beta = BETA
        else:
            raise Exception('beta matrix is not square or symmetric')

    def ci(self, T):
        '''
        ci(T)

        Method that evaluates the polynomials for the influence parameters used
        in the SGT theory for surface tension calculations.

        Parameters
        ----------
        T : float
            absolute temperature [K]

        Returns
        -------
        cij: array_like
            matrix of influence parameters with geometric mixing
            rule [J m5 mol-2]
        '''
        n = self.nc
        ci = np.zeros(n)
        for i in range(n):
            ci[i] = np.polyval(self.cii[i], T)
        self.cij = np.sqrt(np.outer(ci, ci))
        return self.cij

    def sgt_adim(self, T):
        '''
        sgt_adim(T)

        Method that evaluates adimentional factor for temperature, pressure,
        density, tension and distance for interfacial properties computations
        with SGT.

        Parameters
        ----------
        T : absolute temperature [K]

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
        a0 = self.a_eos(T)[0]
        b0 = self.b[0]
        ci = self.ci(T)[0, 0]
        Tfactor = R*b0/a0
        Pfactor = b0**2/a0
        rofactor = b0
        tenfactor = 1000*np.sqrt(a0*ci)/b0**2*(np.sqrt(101325/1.01325)*100**3)
        zfactor = np.sqrt(a0/ci*10**5/100**6)*10**-10
        return Tfactor, Pfactor, rofactor, tenfactor, zfactor

    def ares(self, V, T, D, B):

        V_B = V - B

        g = np.log(V_B/V)
        f = 1./(R*V)

        F = -g - (D/T)*f
        return F

    def EntropyR(self, X, T, P, state, v0=None, T_Step=0.1):
        """
        EntropyR(X, T, P, state, v0, T_step)

        Method that computes the residual entropy at given composition,
        temperature and pressure.

        Parameters
        ----------
        X : array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [cm3/mol]
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy

        Returns
        -------
        Sr : float
            residual entropy [J/mol K]

        """
        h = T_Step

        RT = R*T
        ai = self.a_eos(T)
        bi = self.b
        mixingrulep = self.mixruleparameter

        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, state, v0)
        Z = P*V/RT

        F = self.ares(V, T, D, B)

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        ai1 = self.a_eos(T1)
        ai2 = self.a_eos(T1)
        ai_1 = self.a_eos(T_1)
        ai_2 = self.a_eos(T_2)

        RT1 = R*T1
        RT2 = R*T2
        RT_1 = R*T_1
        RT_2 = R*T_2

        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep)
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep)
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep)
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep)

        F1 = self.ares(V, T1, D1, B1)
        F2 = self.ares(V, T2, D2, B2)
        F_1 = self.ares(V, T_1, D_1, B_1)
        F_2 = self.ares(V, T_2, D_2, B_2)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h

        Sr_TVN = -T*dFdT - F  # residual entropy (TVN) divided by R
        Sr_TPN = Sr_TVN + np.log(Z)  # residual entropy (TPN) divided by R
        Sr_TPN *= r  # J / mol K
        return Sr_TPN

    def EnthalpyR(self, X, T, P, state, v0=None, T_Step=0.1):
        """
        EnthalpyR(X, T, P, state, v0, T_step)

        Method that computes the residual enthalpy at given composition,
        temperature and pressure.

        Parameters
        ----------
        X : array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [cm3/mol]
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy

        Returns
        -------
        Hr : float
            residual enthalpy [J/mol]

        """
        h = T_Step

        RT = R*T
        ai = self.a_eos(T)
        bi = self.b
        mixingrulep = self.mixruleparameter

        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, state, v0)
        Z = P*V/RT

        F = self.ares(V, T, D, B)

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        ai1 = self.a_eos(T1)
        ai2 = self.a_eos(T1)
        ai_1 = self.a_eos(T_1)
        ai_2 = self.a_eos(T_2)

        RT1 = R*T1
        RT2 = R*T2
        RT_1 = R*T_1
        RT_2 = R*T_2

        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep)
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep)
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep)
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep)

        F1 = self.ares(V, T1, D1, B1)
        F2 = self.ares(V, T2, D2, B2)
        F_1 = self.ares(V, T_1, D_1, B_1)
        F_2 = self.ares(V, T_2, D_2, B_2)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h

        Sr_TVN = -T*dFdT - F  # residual entropy (TVN) divided by R
        Hr_TPN = F + Sr_TVN + Z - 1.  # residual entalphy divided by RT
        Hr_TPN *= (r*T)  # J / mol
        return Hr_TPN

    def CvR(self, X, T, P, state, v0=None, T_Step=0.1):
        """
        Cpr(X, T, P, state, v0, T_step)

        Method that computes the residual isochoric heat capacity at given
        composition, temperature and pressure.

        Parameters
        ----------
        X : array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [cm3/mol]
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy

        Returns
        -------
        Cv: float
            residual isochoric heat capacity [J/mol K]
        """
        h = T_Step

        RT = R*T
        ai = self.a_eos(T)
        bi = self.b
        mixingrulep = self.mixruleparameter

        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, state, v0)

        F = self.ares(V, T, D, B)

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        ai1 = self.a_eos(T1)
        ai2 = self.a_eos(T1)
        ai_1 = self.a_eos(T_1)
        ai_2 = self.a_eos(T_2)

        RT1 = R*T1
        RT2 = R*T2
        RT_1 = R*T_1
        RT_2 = R*T_2

        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep)
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep)
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep)
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep)

        F1 = self.ares(V, T1, D1, B1)
        F2 = self.ares(V, T2, D2, B2)
        F_1 = self.ares(V, T_1, D_1, B_1)
        F_2 = self.ares(V, T_2, D_2, B_2)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h
        d2FdT = (-F_2/12 + 4*F_1/3 - 5*F/2 + 4*F1/3 - F2/12)/h**2

        Cvr_TVN = -T**2*d2FdT - 2*T*dFdT  # residual isochoric heat capacity
        Cvr_TVN *= r
        return Cvr_TVN

    def CpR(self, X, T, P, state, v0=None, T_Step=0.1):
        """
        Cpr(X, T, P, state, v0, T_step)

        Method that computes the residual heat capacity at given composition,
        temperature and pressure.

        Parameters
        ----------
        X : array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [cm3/mol]
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy

        Returns
        -------
        Cp: float
            residual heat capacity [J/mol K]
        """
        h = T_Step

        RT = R*T
        ai = self.a_eos(T)
        bi = self.b
        mixingrulep = self.mixruleparameter

        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, state, v0)

        V_B = V - B

        g = np.log(V_B/V)
        f = 1./(R*V)

        F = -g - (D/T)*f

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        ai1 = self.a_eos(T1)
        ai2 = self.a_eos(T1)
        ai_1 = self.a_eos(T_1)
        ai_2 = self.a_eos(T_2)

        RT1 = R*T1
        RT2 = R*T2
        RT_1 = R*T_1
        RT_2 = R*T_2

        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep)
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep)
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep)
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep)

        F1 = self.ares(V, T1, D1, B1)
        F2 = self.ares(V, T2, D2, B2)
        F_1 = self.ares(V, T_1, D_1, B_1)
        F_2 = self.ares(V, T_2, D_2, B_2)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h
        d2FdT = (-F_2/12 + 4*F_1/3 - 5*F/2 + 4*F1/3 - F2/12)/h**2
        dDdT = (D_2/12 - 2*D_1/3 + 2*D1/3 - D2/12)/h

        dPdT = R/V_B - dDdT/V**2
        dPdV = -RT/V_B**2 + 2*D / V**3

        Cvr_TVN = -T**2*d2FdT - 2*T*dFdT  # residual isochoric heat capacity
        Cvr_TVN *= r

        # residual heat capacity
        Cpr = Cvr_TVN - r - (T*dPdT**2/dPdV) / 10
        return Cpr

    def speed_sound(self, X, T, P, state, v0=None, T_Step=0.1, CvId=3*r/2,
                    CpId=5*r/2):
        """
        speed_sound(X, T, P, state, v0, T_step, CvId, CpId)

        Method that computes the speed of sound at given temperature
        and pressure.

        This calculation requires that the molar weight [g/mol] of the fluid
        has been set in the component function.

        By default the ideal gas Cv and Cp are set to 3R/2 and 5R/2, the user
        can supply better values if available.

        Parameters
        ----------
        X : array_like
            molar fraction array
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase
        v0: float, optional
            initial guess for volume root [cm3/mol]
        T_step: float, optional
            Step to compute the numerical temperature derivates of Helmholtz
            free energy
        CvId: float, optional
            Ideal gas isochoric heat capacity, set to 3R/2 by default [J/mol K]
        CpId: float, optional
            Ideal gas heat capacity, set to 3R/2 by default [J/mol K]


        Returns
        -------
        w: float
            speed of sound [m/s]
        """
        h = T_Step

        RT = R*T
        ai = self.a_eos(T)
        bi = self.b
        mixingrulep = self.mixruleparameter

        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, state, v0)

        V_B = V - B

        g = np.log(V_B/V)
        f = 1./(R*V)

        F = -g - (D/T)*f

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        ai1 = self.a_eos(T1)
        ai2 = self.a_eos(T1)
        ai_1 = self.a_eos(T_1)
        ai_2 = self.a_eos(T_2)

        RT1 = R*T1
        RT2 = R*T2
        RT_1 = R*T_1
        RT_2 = R*T_2

        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep)
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep)
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep)
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep)

        F1 = self.ares(V, T1, D1, B1)
        F2 = self.ares(V, T2, D2, B2)
        F_1 = self.ares(V, T_1, D_1, B_1)
        F_2 = self.ares(V, T_2, D_2, B_2)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h
        d2FdT = (-F_2/12 + 4*F_1/3 - 5*F/2 + 4*F1/3 - F2/12)/h**2
        dDdT = (D_2/12 - 2*D_1/3 + 2*D1/3 - D2/12)/h

        dPdT = R/V_B - dDdT/V**2
        dPdV = -RT/V_B**2 + 2*D / V**3

        Cvr_TVN = -T**2*d2FdT - 2*T*dFdT  # residual isochoric heat capacity
        Cvr_TVN *= r

        # residual heat capacity
        Cpr = Cvr_TVN - r - (T*dPdT**2/dPdV) / 10

        # speed of sound calculation
        Cp = CpId + Cpr
        Cv = CvId + Cvr_TVN

        betas = - (Cv/Cp) / dPdV / V

        Mwx = np.dot(X, self.Mw)
        w2 = 100.*V/(betas * Mwx)
        w = np.sqrt(w2)
        return w

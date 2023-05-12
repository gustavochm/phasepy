from __future__ import division, print_function, absolute_import
import numpy as np
from .mixingrules import mixingrule_fcn
from .alphas import alpha_soave, alpha_sv, alpha_rk
from ..constants import R, r


class vtcubicm():
    '''
    Mixture Cubic EoS Object

    This object have implemeted methods for phase equilibrium
    as for interfacial properties calculations.

    Parameters
    ----------
    mix : object
        mixture created with mixture class
    c1, c2 : float
        constants of cubic EoS
    oma, omb : float
        constants of cubic EoS
    alpha_eos : function
        function that gives thermal funcionality  to attractive term of EoS
    mixrule : function
        computes mixture attactive and cohesive terms

    Attributes
    ----------
    Tc: array_like
        critical temperture [K]
    Pc: array_like
        critical pressure [bar]
    w: array_like
        acentric factor
    cii : array_like
        influence factor for SGT polynomials [J m5 mol-2]
    nc : int
        number of components of mixture
    Mw : array_like
        molar weight of the fluids [g mol-1]

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
    sgt_adim : computes adimentional factors for SGT.
    beta_sgt : method that incorporates the beta correction for SGT.
    EntropyR : computes residual Entropy.
    EnthalpyR: computes residual Enthalpy.
    CvR : computes residual isochoric heat capacity.
    CpR : computes residual isobaric heat capacity.
    speed_sound : computes the speed of sound.
    '''

    def __init__(self, mix, c1, c2, oma, omb, alpha_eos, mixrule):

        self.c1 = c1
        self.c2 = c2
        self.oma = oma
        self.omb = omb
        self.alpha_eos = alpha_eos
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))

        self.k = np.array(mix.alpha_params, ndmin=1)
        self.Mw = np.array(mix.Mw, ndmin=1)
        self.Tc = np.array(mix.Tc, ndmin=1)
        self.Pc = np.array(mix.Pc, ndmin=1)
        self.w = np.array(mix.w, ndmin=1)
        self.cii = np.array(mix.cii, ndmin=1)
        self.b = self.omb*R*self.Tc/self.Pc
        self.c = np.array(mix.c, ndmin=1)

        # fusion and melting point (needed for SLE and SLLE)
        self.dHf = np.asarray(mix.dHf)
        self.Tf = np.asarray(mix.Tf)
        self.dHf_r = np.asarray(mix.dHf) / r

        self.nc = mix.nc
        self.beta = np.zeros([self.nc, self.nc])
        mixingrule_fcn(self, mix, mixrule)

        # Matrix used for second order composition derivatives
        Ci = self.c
        self.MatrixCpC = np.add.outer(Ci, Ci)
        self.MatrixCC = np.outer(Ci, Ci)

    # Cubic EoS methods
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
        alpha = self.alpha_eos(T, self.k, self.Tc)
        a = self.oma*(R*self.Tc)**2*alpha/self.Pc
        return a

    def temperature_aux(self, T):
        RT = R*T
        ai = self.a_eos(T)
        mixingrulep = self.mixrule_temp(T)
        temp_aux = (RT, T, ai, mixingrulep)
        return temp_aux

    def _Zroot(self, A, B, C):

        a1 = (self.c1+self.c2-1)*B-1 + 3 * C
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a2 += 3*C**2 + 2*C*(-1 + B*(-1 + self.c1 + self.c2))
        a3 = A*(-B+C) + (-1-B+C)*(C+self.c1*B)*(C+self.c2*B)

        Zpol = np.hstack([1., a1, a2, a3])
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots > (B - C)]
        return Zroots

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
        RT, T, ai, mixingrulep = self.temperature_aux(T)
        # a = self.a_eos(T)
        c = self.c
        am, bm = self.mixrule(X, RT, ai, self.b, 0, *mixingrulep)
        cm = np.dot(X, c)
        A = am*P/RT**2
        B = bm*P/RT
        C = cm*P/RT

        return self._Zroot(A, B, C)

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
        RT, T, ai, mixingrulep = self.temperature_aux(T)

        am, bm = self.mixrule(X, RT, ai, self.b, 0, *mixingrulep)
        cm = np.dot(X, self.c)
        P = RT/(v+cm-bm)-am/((v+cm+self.c1*bm)*(v+cm+self.c2*bm))
        return P

    # Auxiliar method that computes volume roots
    def _volume_solver(self, P, RT, D, B, C, state):

        Dr = D*P/RT**2
        Br = B*P/RT
        Cr = C*P/RT

        if state == 'L':
            Z = np.min(self._Zroot(Dr, Br, Cr))
        elif state == 'V':
            Z = np.max(self._Zroot(Dr, Br, Cr))
        else:
            raise Exception('Valid states: L for liquids and V for vapor ')
        V = (RT*Z)/P

        return V

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
        RT, T, ai, mixingrulep = self.temperature_aux(T)

        bi = self.b
        c = self.c
        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        C = np.dot(X, c)

        V = self._volume_solver(P, RT, D, B, C, state)

        rho = 1. / V
        return rho

    def logfugef_aux(self, X, temp_aux, P, state, v0=None):

        RT, T, ai, mixingrulep = temp_aux

        c1 = self.c1
        c2 = self.c2

        bi = self.b
        Ci = self.c
        C = np.dot(X, Ci)

        D, Di, B, Bi = self.mixrule(X, RT, ai, bi, 1, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, C, state)

        Z = P*V/RT

        D_T = D/T
        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        gb = -1./VCB
        gc = -gb

        fv = -1./(R*VCc1B*VCc2B)
        fb = - (f + (V+C) * fv)/B
        fc = fv

        Fn = -g
        Fb = -gb - D_T * fb
        Fc = - gc - D_T * fc
        Fd = - f / T

        dF_dn = Fn + Fb * Bi + Fc * Ci + Fd * Di
        logfug = dF_dn - np.log(Z)

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
            'L' for liquid phase, 'V' for vapour phase
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

    def dlogfugef_aux(self, X, temp_aux, P, state, v0=None):

        RT, T, ai, mixingrulep = temp_aux
        c1 = self.c1
        c2 = self.c2

        bi = self.b
        Ci = self.c
        C = np.dot(X, Ci)
        Cij = 0.

        D, Di, Dij, B, Bi, Bij = self.mixrule(X, RT, ai, bi, 2, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, C, state)

        Z = P * V / RT

        D_T = D/T

        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B
        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        gb = -1./VCB
        gc = -gb
        gv = -1./V - gb

        fv = -1./(R*VCc1B*VCc2B)
        fb = - (f + (V+C) * fv)/B
        fc = fv

        gbv = gb**2
        gcv = - gbv
        gvv = 1./V**2 - gbv
        gbb = - gbv
        gcc = - gbv
        gbc = gbv

        fvv = (1./(VCc1B*VCc2B**2) + 1./(VCc1B**2*VCc2B))/R
        fcv = fvv
        fcc = fvv
        fbv = - (2*fv + (V+C) * fvv)/B
        fbb = - (2*fb + (V+C) * fbv)/B
        fbc = fbv

        Fn = - g
        Fb = - gb - D_T * fb
        Fd = - f / T
        Fc = - gc - D_T * fc

        Fnv = -gv
        Fnb = -gb
        Fnc = -gc

        Fbv = -gbv - D_T * fbv
        Fcv = -gcv - D_T * fcv
        Fvv = -gvv - D_T * fvv
        Fdv = -fv/T

        Fbd = -fb/T
        Fbb = -gbb - D_T * fbb
        Fbc = -gbc - D_T * fbc
        Fdc = -fc/T
        Fcc = -gcc - D_T * fcc

        dF_dn = Fn + Fb * Bi + Fc * Ci + Fd * Di
        logfug = dF_dn - np.log(Z)

        MatrixBD = np.outer(Bi, Di)
        MatrixBD += MatrixBD.T
        MatrixDC = np.outer(Di, Ci)
        MatrixDC += MatrixDC.T

        MatrixBpB = np.add.outer(Bi, Bi)
        MatrixBB = np.outer(Bi, Bi)
        MatrixBC = np.outer(Bi, Ci)
        MatrixBC += MatrixBC.T

        dF_dnij = Fnb * MatrixBpB + Fbd * MatrixBD
        dF_dnij += Fb * Bij + Fbb * MatrixBB + Fd * Dij
        dF_dnij += Fnc * self.MatrixCpC + Fbc * MatrixBC
        dF_dnij += Fdc * MatrixDC + Fc * Cij + Fcc * self.MatrixCC

        d2F_dv = Fvv
        dF_dndv = Fnv + Fbv * Bi + Fdv * Di + Fcv * Ci
        dP_dV = - RT * d2F_dv - RT / V**2
        dP_dn = - RT * dF_dndv + RT / V

        dlogfugef = dF_dnij + 1. + np.outer(dP_dn, dP_dn) / (R * T * dP_dV)

        return logfug, dlogfugef, V

    def dlogfugef(self, X, T, P, state, v0=None):
        """
        dlogfugef(X, T, P, state)

        Method that computes the effective fugacity coefficients and its
        composition derivatives at given composition, temperature and pressure.

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
        v0 : float, optional
            initial volume to iterate [cm3/mol]

        Returns
        -------
        logfug: array_like
            effective fugacity coefficients
        dlogfug: array_like
            composition derivatives of effective fugacity coefficients
        v : float
            volume of phase [cm3/mol]
        """

        temp_aux = self.temperature_aux(T)
        logfug, dlogfugef, V = self.dlogfugef_aux(X, temp_aux, P, state, v0)

        return logfug, dlogfugef, V

    def logfugmix_aux(self, X, temp_aux, P, state, v0=None):

        RT, T, ai, mixingrulep = temp_aux

        c1 = self.c1
        c2 = self.c2

        bi = self.b
        c = self.c
        cm = np.dot(X, c)

        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, cm, state)

        Z = P * V / RT
        am = D
        bm = B

        A = am * P / RT**2
        B = bm * P/RT
        C = cm * P/RT

        logfug = Z - 1 - np.log(Z+C-B)
        logfug -= (A/(c2-c1)/B)*np.log((Z+C+c2*B)/(Z+C+c1*B))

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

    def a0ad_aux(self, roa, temp_aux):

        RT, T, ai, mixingrulep = temp_aux

        c1 = self.c1
        c2 = self.c2

        bi = self.b
        ci = self.c
        a = ai[0]
        b = bi[0]
        ro = np.sum(roa)
        X = roa/ro

        C = np.dot(X, ci)
        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)

        V = b/ro
        D_T = D/T
        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        F = - g - D_T * f

        RT_V = R*T/V
        adfactor = b**2/a

        F = - g - D * f / T
        a0 = F  # A residual
        a0 += np.dot(X, np.nan_to_num(np.log(X)))
        a0 += np.log(RT_V)
        a0 *= RT_V
        a0 *= adfactor

        return a0

    def a0ad(self, roa, T):
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
        a0 = self.a0ad_aux(roa, temp_aux)

        return a0

    def muad_aux(self, rhoa, temp_aux):
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

        RT, T, ai, mixingrulep = temp_aux

        c1 = self.c1
        c2 = self.c2
        bi = self.b
        a = ai[0]
        b = bi[0]
        ro = np.sum(rhoa)
        X = rhoa/ro

        Ci = self.c
        C = np.dot(X, Ci)
        D, Di, B, Bi = self.mixrule(X, RT, ai, bi, 1, *mixingrulep)

        adfactor = b/a
        V = b/ro
        RT_V = RT/V

        D_T = D/T
        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        gb = -1./VCB
        gc = -gb

        fv = -1./(R*VCc1B*VCc2B)
        fb = - (f + (V+C) * fv)/B
        fc = fv

        Fn = -g
        Fb = -gb - D_T * fb
        Fc = - gc - D_T * fc
        Fd = - f / T

        dF_dn = Fn + Fb * Bi + Fc * Ci + Fd * Di
        mui = np.log(RT_V) + np.log(X) + 1.
        mui += dF_dn
        mui *= RT
        mui *= adfactor

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

        return mui

    def dmuad_aux(self, roa, temp_aux):
        RT, T, ai, mixingrulep = temp_aux

        c1 = self.c1
        c2 = self.c2
        bi = self.b
        a = ai[0]
        b = bi[0]
        ro = np.sum(roa)
        X = roa/ro

        Ci = self.c
        Cij = 0
        C = np.dot(X, Ci)
        D, Di, Dij, B, Bi, Bij = self.mixrule(X, RT, ai, bi, 2, *mixingrulep)

        adfactor = b/a
        V = b/ro
        RT = R*T
        RT_V = RT/V

        D_T = D/T
        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        gb = -1./VCB
        gc = -gb

        fv = -1./(R*VCc1B*VCc2B)
        fb = - (f + (V+C) * fv)/B
        fc = fv

        gbv = gb**2
        gbb = - gbv
        gcc = - gbv
        gbc = gbv

        fvv = (1./(VCc1B*VCc2B**2) + 1./(VCc1B**2*VCc2B))/R
        fcc = fvv
        fbv = - (2*fv + (V+C) * fvv)/B
        fbb = - (2*fb + (V+C) * fbv)/B
        fbc = fbv

        Fn = - g
        Fb = - gb - D_T * fb
        Fd = - f / T
        Fc = - gc - D_T * fc

        Fnb = -gb
        Fnc = -gc

        Fbd = -fb/T
        Fbb = -gbb - D_T * fbb
        Fbc = -gbc - D_T * fbc
        Fdc = -fc/T
        Fcc = -gcc - D_T * fcc

        MatrixBD = np.outer(Bi, Di)
        MatrixBD += MatrixBD.T
        MatrixDC = np.outer(Di, Ci)
        MatrixDC += MatrixDC.T

        MatrixBpB = np.add.outer(Bi, Bi)
        MatrixBB = np.outer(Bi, Bi)
        MatrixBC = np.outer(Bi, Ci)
        MatrixBC += MatrixBC.T

        dF_dnij = Fnb * MatrixBpB + Fbd * MatrixBD
        dF_dnij += Fb * Bij + Fbb * MatrixBB + Fd * Dij
        dF_dnij += Fnc * self.MatrixCpC + Fbc * MatrixBC
        dF_dnij += Fdc * MatrixDC + Fc * Cij + Fcc * self.MatrixCC

        dF_dn = Fn + Fb * Bi + Fc * Ci + Fd * Di

        mui = np.log(RT_V) + np.log(X) + 1.
        mui += dF_dn
        mui *= RT
        mui *= adfactor

        dx = (np.eye(self.nc) - X)/ro
        dmui = dF_dnij / ro
        dmui += 1./ro
        dmui += dx/X
        dmui *= RT
        dmui *= adfactor

        return mui, dmui

    def dmuad(self, rhoa, T):
        """
        muad(roa, T)

        Method that computes the adimenstional chemical potential and
        its derivatives at given density and temperature.

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
        muad : array_like
            adimentional derivatives of chemical potential vector
        """

        temp_aux = self.temperature_aux(T)
        mui, dmui = self.dmuad_aux(rhoa, temp_aux)

        return mui, dmui

    def dOm_aux(self, roa, temp_aux, mu, Psat):
        a0ad = self.a0ad_aux(roa, temp_aux)
        dom = a0ad - np.sum(np.nan_to_num(roa*mu)) + Psat
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

    def _lnphi0(self, T, P):

        nc = self.nc
        a_puros = self.a_eos(T)
        Ai = a_puros*P/(R*T)**2
        Bi = self.b*P/(R*T)
        Ci = self.c*P/(R*T)
        c1, c2 = self.c1, self.c2

        a1 = (c1+c2-1)*Bi-1 + 3 * Ci
        a2 = c1*c2*Bi**2-(c1+c2)*(Bi**2+Bi)+Ai
        a2 += 3*Ci**2 + 2*Ci*(-1 + Bi*(-1 + c1 + c2))
        a3 = Ai*(-Bi + Ci) + (-1-Bi+Ci)*(Ci+c1*Bi)*(Ci+c2*Bi)
        pols = np.array([a1, a2, a3])
        Zs = np.zeros([nc, 2])

        for i in range(nc):
            zroot = np.roots(np.hstack([1, pols[:, i]]))
            zroot = zroot[zroot > Bi[i]]
            Zs[i, :] = np.array([max(zroot), min(zroot)])

        logphi = Zs - 1 - np.log(Zs.T+Ci-Bi)
        logphi -= (Ai/(c2-c1)/Bi)*np.log((Zs.T+Ci+c2*Bi)/(Zs.T+Ci+c1*Bi))
        logphi = np.amin(logphi, axis=0)

        return logphi

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
            absolute temperature in K

        Returns
        -------
        cij: array_like
            matrix of influence parameters with geomtric mixing rule.
        '''

        n = self.nc
        ci = np.zeros(n)
        for i in range(n):
            ci[i] = np.polyval(self.cii[i], T)
        self.cij = np.sqrt(np.outer(ci, ci))*(1-self.beta)
        return self.cij

    def sgt_adim(self, T):
        '''
        sgt_adim(T)

        Method that evaluates adimentional factor for temperature, pressure,
        density, tension and distance for interfacial properties computations
        with SGT.

        Parameters
        ----------
        T : absolute temperature in K

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

    def ares(self, V, T, D, B, C):
        c1 = self.c1
        c2 = self.c2

        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

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

        temp_aux = self.temperature_aux(T)
        RT, T, ai, mixingrulep = temp_aux
        bi = self.b
        Ci = self.c
        C = np.dot(X, Ci)
        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, C, state)

        Z = P*V/RT

        F = self.ares(V, T, D, B, C)

        temp_aux1 = self.temperature_aux(T+h)
        RT1, T1, ai1, mixingrulep1 = temp_aux1
        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep1)

        temp_aux2 = self.temperature_aux(T+2*h)
        RT2, T2, ai2, mixingrulep2 = temp_aux2
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep2)

        temp_aux_1 = self.temperature_aux(T-h)
        RT_1, T_1, ai_1, mixingrulep_1 = temp_aux_1
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep_1)

        temp_aux_2 = self.temperature_aux(T-2*h)
        RT_2, T_2, ai_2, mixingrulep_2 = temp_aux_2
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep_2)

        F1 = self.ares(V, T1, D1, B1, C)
        F2 = self.ares(V, T2, D2, B2, C)
        F_1 = self.ares(V, T_1, D_1, B_1, C)
        F_2 = self.ares(V, T_2, D_2, B_2, C)

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

        temp_aux = self.temperature_aux(T)
        RT, T, ai, mixingrulep = temp_aux
        bi = self.b
        Ci = self.c
        C = np.dot(X, Ci)
        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, C, state)

        Z = P*V/RT

        F = self.ares(V, T, D, B, C)

        temp_aux1 = self.temperature_aux(T+h)
        RT1, T1, ai1, mixingrulep1 = temp_aux1
        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep1)

        temp_aux2 = self.temperature_aux(T+2*h)
        RT2, T2, ai2, mixingrulep2 = temp_aux2
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep2)

        temp_aux_1 = self.temperature_aux(T-h)
        RT_1, T_1, ai_1, mixingrulep_1 = temp_aux_1
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep_1)

        temp_aux_2 = self.temperature_aux(T-2*h)
        RT_2, T_2, ai_2, mixingrulep_2 = temp_aux_2
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep_2)

        F1 = self.ares(V, T1, D1, B1, C)
        F2 = self.ares(V, T2, D2, B2, C)
        F_1 = self.ares(V, T_1, D_1, B_1, C)
        F_2 = self.ares(V, T_2, D_2, B_2, C)

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

        temp_aux = self.temperature_aux(T)
        RT, T, ai, mixingrulep = temp_aux
        bi = self.b
        Ci = self.c
        C = np.dot(X, Ci)
        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, C, state)

        F = self.ares(V, T, D, B, C)

        temp_aux1 = self.temperature_aux(T+h)
        RT1, T1, ai1, mixingrulep1 = temp_aux1
        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep1)

        temp_aux2 = self.temperature_aux(T+2*h)
        RT2, T2, ai2, mixingrulep2 = temp_aux2
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep2)

        temp_aux_1 = self.temperature_aux(T-h)
        RT_1, T_1, ai_1, mixingrulep_1 = temp_aux_1
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep_1)

        temp_aux_2 = self.temperature_aux(T-2*h)
        RT_2, T_2, ai_2, mixingrulep_2 = temp_aux_2
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep_2)

        F1 = self.ares(V, T1, D1, B1, C)
        F2 = self.ares(V, T2, D2, B2, C)
        F_1 = self.ares(V, T_1, D_1, B_1, C)
        F_2 = self.ares(V, T_2, D_2, B_2, C)

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

        temp_aux = self.temperature_aux(T)
        RT, T, ai, mixingrulep = temp_aux
        bi = self.b
        Ci = self.c
        C = np.dot(X, Ci)
        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, C, state)

        c1 = self.c1
        c2 = self.c2
        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        F = -g - (D/T)*f

        temp_aux1 = self.temperature_aux(T+h)
        RT1, T1, ai1, mixingrulep1 = temp_aux1
        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep1)

        temp_aux2 = self.temperature_aux(T+2*h)
        RT2, T2, ai2, mixingrulep2 = temp_aux2
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep2)

        temp_aux_1 = self.temperature_aux(T-h)
        RT_1, T_1, ai_1, mixingrulep_1 = temp_aux_1
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep_1)

        temp_aux_2 = self.temperature_aux(T-2*h)
        RT_2, T_2, ai_2, mixingrulep_2 = temp_aux_2
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep_2)

        F1 = self.ares(V, T1, D1, B1, C)
        F2 = self.ares(V, T2, D2, B2, C)
        F_1 = self.ares(V, T_1, D_1, B_1, C)
        F_2 = self.ares(V, T_2, D_2, B_2, C)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h
        d2FdT = (-F_2/12 + 4*F_1/3 - 5*F/2 + 4*F1/3 - F2/12)/h**2
        dDdT = (D_2/12 - 2*D_1/3 + 2*D1/3 - D2/12)/h

        dPdT = R/VCB - dDdT/(VCc1B*VCc2B)
        dPdV = -RT/VCB**2 + D * (VCc1B+VCc2B)/(VCc1B*VCc2B)**2

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

        temp_aux = self.temperature_aux(T)
        RT, T, ai, mixingrulep = temp_aux
        bi = self.b
        Ci = self.c
        C = np.dot(X, Ci)
        D, B = self.mixrule(X, RT, ai, bi, 0, *mixingrulep)
        V = self._volume_solver(P, RT, D, B, C, state)

        c1 = self.c1
        c2 = self.c2
        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        F = -g - (D/T)*f

        temp_aux1 = self.temperature_aux(T+h)
        RT1, T1, ai1, mixingrulep1 = temp_aux1
        D1, B1 = self.mixrule(X, RT1, ai1, bi, 0, *mixingrulep1)

        temp_aux2 = self.temperature_aux(T+2*h)
        RT2, T2, ai2, mixingrulep2 = temp_aux2
        D2, B2 = self.mixrule(X, RT2, ai2, bi, 0, *mixingrulep2)

        temp_aux_1 = self.temperature_aux(T-h)
        RT_1, T_1, ai_1, mixingrulep_1 = temp_aux_1
        D_1, B_1 = self.mixrule(X, RT_1, ai_1, bi, 0, *mixingrulep_1)

        temp_aux_2 = self.temperature_aux(T-2*h)
        RT_2, T_2, ai_2, mixingrulep_2 = temp_aux_2
        D_2, B_2 = self.mixrule(X, RT_2, ai_2, bi, 0, *mixingrulep_2)

        F1 = self.ares(V, T1, D1, B1, C)
        F2 = self.ares(V, T2, D2, B2, C)
        F_1 = self.ares(V, T_1, D_1, B_1, C)
        F_2 = self.ares(V, T_2, D_2, B_2, C)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h
        d2FdT = (-F_2/12 + 4*F_1/3 - 5*F/2 + 4*F1/3 - F2/12)/h**2
        dDdT = (D_2/12 - 2*D_1/3 + 2*D1/3 - D2/12)/h

        dPdT = R/VCB - dDdT/(VCc1B*VCc2B)
        dPdV = -RT/VCB**2 + D * (VCc1B+VCc2B)/(VCc1B*VCc2B)**2

        Cvr_TVN = -T**2*d2FdT - 2*T*dFdT  # residual isochoric heat capacity
        Cvr_TVN *= r
        Cvr_TVN

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


# Peng Robinson EoS
c1pr = 1-np.sqrt(2)
c2pr = 1+np.sqrt(2)
omapr = 0.4572355289213825
ombpr = 0.07779607390388854


class vtprmix(vtcubicm):
    def __init__(self, mix, mixrule='qmr'):
        vtcubicm.__init__(self, mix, c1=c1pr, c2=c2pr, oma=omapr, omb=ombpr,
                          alpha_eos=alpha_soave, mixrule=mixrule)

        self.k = 0.37464 + 1.54226*self.w - 0.26992*self.w**2


# Peng Robinson SV EoS
class vtprsvmix(vtcubicm):
    def __init__(self, mix, mixrule='qmr'):
        vtcubicm.__init__(self, mix, c1=c1pr, c2=c2pr, oma=omapr, omb=ombpr,
                          alpha_eos=alpha_sv, mixrule=mixrule)
        if np.all(mix.ksv == 0):
            self.k = np.zeros([self.nc, 2])
            self.k[:, 0] = 0.378893+1.4897153*self.w-0.17131838*self.w**2
            self[:, 0] += 0.0196553*self.w**3
        else:
            self.k = np.array(mix.ksv)


# RKS - EoS
c1rk = 0
c2rk = 1
omark = 0.42748
ombrk = 0.08664


class vtrksmix(vtcubicm):
    def __init__(self, mix, mixrule='qmr'):
        vtcubicm.__init__(self, mix, c1=c1rk, c2=c2rk, oma=omark, omb=ombrk,
                          alpha_eos=alpha_soave, mixrule=mixrule)
        self.k = 0.47979 + 1.5476*self.w - 0.1925*self.w**2 + 0.025*self.w**3


# RK - EoS
class vtrkmix(vtcubicm):
    def __init__(self, mix, mixrule='qmr'):
        vtcubicm.__init__(self, mix, c1=c1rk, c2=c2rk, oma=omark, omb=ombrk,
                          alpha_eos=alpha_rk, mixrule=mixrule)

    def a_eos(self, T):
        alpha = self.alpha_eos(T, self.Tc)
        return self.oma*(R*self.Tc)**2*alpha/self.Pc

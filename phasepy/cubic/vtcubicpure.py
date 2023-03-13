from __future__ import division, print_function, absolute_import
import numpy as np
from .alphas import alpha_soave, alpha_sv, alpha_rk
from ..constants import R, r
from scipy.optimize import brentq, newton


def psat(T, cubic, P0=None):
    """
    Computes saturation pressure with cubic eos

    Parameters
    ----------
    T : float,
        Absolute temperature [K]
    cubic : object
          eos object
    Returns
    -------
    P : float
       saturation pressure [bar]
    """
    a = cubic.a_eos(T)
    b = cubic.b
    c1 = cubic.c1
    c2 = cubic.c2
    emin = cubic.emin
    c = cubic.c
    e = a/(b*R*T)

    if P0 is None:
        if e > emin:  # Zero pressure initiation
            U = (e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2
            if c1 == 0 and c2 == 0:
                S = -1-np.log(U-1)-e/U
            else:
                S = -1-np.log(U-1)-e*np.log((U+c1)/(U+c2))/(c1-c2)
            P = np.exp(S)*R*T/b  # bar

        else:  # Pmin Pmax initiation
            a1 = -R*T
            a2 = -2*b*R*T*(c1+c2)+2*a
            a3 = -R*T*b**2*(c1**2+4*c1*c2+c2**2)+a*b*(c1+c2-4)
            a4 = -R*T*2*b**3*c1*c2*(c1+c2)+2*a*b**2*(1-c1-c2)
            a5 = -R*T*b**4*c1*c2+a*b**3*(c1+c2)
            V = np.roots([a1, a2, a3, a4, a5])
            V = V[np.isreal(V)]
            V = V[V > b]
            P = cubic(T, V)
            P[P < 0] = 0.
            P = P.mean()
    else:
        P = P0
    itmax = 20
    RT = R * T
    for k in range(itmax):
        A = a*P/RT**2
        B = b*P/RT
        C = c*P/RT
        Z = cubic._Zroot(A, B, C)
        Zl = min(Z)
        Zv = max(Z)
        fugL = cubic._logfug_aux(Zl, A, B, C)
        fugV = cubic._logfug_aux(Zv, A, B, C)
        FO = fugV-fugL
        dFO = (Zv-Zl)/P
        dP = FO/dFO
        P -= dP
        if abs(dP) < 1e-8:
            break
    vl = Zl*RT/P
    vv = Zv*RT/P
    return P, vl, vv


def fobj_tsat(T, P, cubic):

    a = cubic.a_eos(T)
    b = cubic.b
    c = cubic.c
    RT = R*T
    A = a*P/(RT)**2
    B = b*P/(RT)
    C = c*P/RT
    Z = cubic._Zroot(A, B, C)
    Zl = min(Z)
    Zv = max(Z)
    fugL = cubic._logfug_aux(Zl, A, B, C)
    fugV = cubic._logfug_aux(Zv, A, B, C)
    FO = fugV-fugL

    return FO


def tsat(cubic, P, T0=None, Tbounds=None):
    """
    Computes saturation temperature with cubic eos

    Parameters
    ----------
    cubic: object
        cubic eos object
    P: float
        saturation pressure [bar]
    T0 : float, optional
         Temperature to start iterations [K]
    Tbounds : tuple, optional
            (Tmin, Tmax) Temperature interval to start iterations [K]

    Returns
    -------
    T : float
        saturation temperature [K]
    vl: float
        saturation liquid volume [cm3/mol]
    vv: float
        saturation vapor volume [cm3/mol]

    """
    bool1 = T0 is None
    bool2 = Tbounds is None

    if bool1 and bool2:
        raise Exception('You must provide either Tbounds or T0')

    if not bool1:
        sol = newton(fobj_tsat, x0=T0, args=(P, cubic),
                     full_output=False)
        Tsat = sol[0]
    elif not bool2:
        sol = brentq(fobj_tsat, Tbounds[0], Tbounds[1], args=(P, cubic),
                     full_output=False)
        Tsat = sol

    vl = 1./cubic.density(Tsat, P, 'L')
    vv = 1./cubic.density(Tsat, P, 'V')
    out = (Tsat, vl, vv)
    return out


class vtcpure():
    '''
    Pure component Cubic EoS Object

    This object have implemeted methods for phase equilibrium
    as for iterfacial properties calculations.

    Parameters
    ----------
    pure : object
        pure component created with component class
    c1, c2 : float
        constants of cubic EoS
    oma, omb : float
        constants of cubic EoS
    alpha_eos : function
        function that gives thermal funcionality  to attractive term of EoS

    Attributes
    ----------
    Tc: float
        critical temperture [K]
    Pc: float
        critical pressure [bar]
    w: float
        acentric factor
    cii : array_like
        influence factor for SGT polynomial [J m5 mol-2]
    Mw : float
        molar weight of the fluid [g mol-1]

    Methods
    -------
    a_eos : computes the attractive term of cubic eos.
    psat : computes saturation pressure.
    tsat : computes saturation temperature
    density : computes density of mixture.
    logfug : computes fugacity coefficient.
    a0ad : computes adimentional Helmholtz density energy
    muad : computes adimentional chemical potential.
    dOm : computes adimentional Thermodynamic Grand Potential.
    ci :  computes influence parameters matrix for SGT.
    sgt_adim : computes adimentional factors for SGT.
    EntropyR : computes residual Entropy.
    EnthalpyR: computes residual Enthalpy.
    CvR : computes residual isochoric heat capacity.
    CpR : computes residual isobaric heat capacity.
    speed_sound : computes the speed of sound.

    '''

    def __init__(self, pure, c1, c2, oma, omb, alpha_eos):

        self.c1 = c1
        self.c2 = c2
        self.oma = oma
        self.omb = omb
        self.alpha_eos = alpha_eos
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))

        
        self.Mw = pure.Mw
        self.Tc = np.array(pure.Tc, ndmin=1)  # Critical temperature in K
        self.Pc = np.array(pure.Pc, ndmin=1)  # Critical Pressure in bar
        self.w = np.array(pure.w, ndmin=1)
        self.cii = np.array(pure.cii, ndmin=1)
        self.b = self.omb*R*self.Tc/self.Pc
        self.c = np.array(pure.c, ndmin=1)
        self.k = np.array(pure.alpha_params, ndmin=1)

    def __call__(self, T, v):
        b = self.b
        a = self.a_eos(T)
        c1 = self.c1
        c2 = self.c2
        return R*T/(v - b) - a/((v+c1*b)*(v+c2*b))

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
        a : float
            atractive term array [bar cm6 mol-2]
        """
        alpha = self.alpha_eos(T, self.k, self.Tc)
        return self.oma*(R*self.Tc)**2*alpha/self.Pc

    def psat(self, T, P0=None):
        """
        psat(T, P0)

        Method that computes saturation pressure at given temperature

        Parameters
        ----------

        T : float
            absolute temperature [K]
        P0 : float, optional
            initial value to find saturation pressure [bar], None for automatic
            initiation

        Returns
        -------
        psat : float
            saturation pressure [bar]
        """
        p0, vl, vv = psat(T, self, P0)
        return p0, vl, vv

    def tsat(self, P, T0=None, Tbounds=None):
        """
        tsat(P, T0, Tbounds)

        Method that computes saturation temperature at given pressure

        Parameters
        ----------
        P : float
            pressure [bar]
        T0 : float, optional
             Temperature to start iterations [K]
        Tbounds : tuple, optional
                (Tmin, Tmax) Temperature interval to start iterations [K]

        Returns
        -------
        tsat : float
            saturation pressure [bar]
        vl: float
            saturation liquid volume [cm3/mol]
        vv: float
            saturation vapor volume [cm3/mol]
        """
        Tsat, vl, vv = tsat(self, P, T0, Tbounds)
        return Tsat, vl, vv

    def _Zroot(self, A, B, C):
        a1 = (self.c1+self.c2-1)*B-1 + 3 * C
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a2 += 3*C**2 + 2*C*(-1 + B*(-1 + self.c1 + self.c2))
        a3 = A*(-B+C)+(-1-B+C)*(C+self.c1*B)*(C+self.c2*B)
        Zpol = np.hstack([1., a1, a2, a3])
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots > (B - C)]
        return Zroots

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

    def density(self, T, P, state):
        """
        density(T, P, state)
        Method that computes the density of the mixture at given temperature
        and pressure.

        Parameters
        ----------

        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        density: float
            molar density [mol/cm3]

        """
        RT = R * T
        A = self.a_eos(T)*P/(RT)**2
        B = self.b*P/(RT)
        C = self.c*P/(RT)

        if state == 'L':
            Z = min(self._Zroot(A, B, C))
        if state == 'V':
            Z = max(self._Zroot(A, B, C))
        return P/(R*T*Z)

    def _logfug_aux(self, Z, A, B, C):
        c1 = self.c1
        c2 = self.c2
        logfug = Z-1-np.log(Z+C-B)
        logfug -= (A/(c2-c1)/B)*np.log((Z+C+c2*B)/(Z+C+c1*B))
        return logfug

    def logfug(self, T, P, state):
        """
        logfug(T, P, state)

        Method that computes the fugacity coefficient at given temperature
        and pressure.

        Parameters
        ----------
        T : float
            absolute temperature [K]
        P : float
            pressure [bar]
        state : string
            'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        logfug: float
            fugacity coefficient
        v : float
            volume of the fluid [cm3/mol]
        """

        RT = R * T
        A = self.a_eos(T)*P/(RT)**2
        B = self.b*P/(RT)
        C = self.c*P/(RT)

        if state == 'L':
            Z = min(self._Zroot(A, B, C))
        if state == 'V':
            Z = max(self._Zroot(A, B, C))

        logfug = self._logfug_aux(Z, A, B, C)

        return logfug

    def a0ad(self, ro, T):
        """
        a0ad(ro, T)

        Method that computes the adimenstional Helmholtz density energy at
        given density and temperature.

        Parameters
        ----------

        ro : float
            adimentional density vector [rho = rho * b]
        T : float
            absolute adimentional temperature [Adim]

        Returns
        -------
        a0ad: float
            adimenstional Helmholtz density energy [Adim]
        """
        c1 = self.c1
        c2 = self.c2
        cro = self.c * ro / self.b

        Pref = 1
        a0 = -T*ro*np.log(1-ro+cro)
        a0 += -T*ro*np.log(Pref/(T*ro))
        a0 += -ro*np.log((1+c2*ro + cro)/(1+c1*ro+cro))/((c2-c1))

        return a0

    def muad(self, ro, T):
        """
        muad(ro, T)

        Method that computes the adimenstional chemical potential at given
        density and temperature.

        Parameters
        ----------

        roa : float
            adimentional density vector [rho = rho * b]
        T : float
            absolute adimentional temperature [adim]

        Returns
        -------
        muad: float
            chemical potential [Adim]
        """
        c1 = self.c1
        c2 = self.c2
        cro = self.c * ro / self.b

        Pref = 1
        mu = - ro/((1+cro+c1*ro)*(1+cro+c2*ro))
        mu += T / (1-ro+cro)
        mu += np.log((1+cro+c2*ro)/(1+cro+c1*ro))/(c1-c2)
        mu -= T * np.log(1-ro+cro)
        mu -= T * np.log(Pref/T/ro)

        return mu

    def dOm(self, roa, Tad, mu, Psat):
        r"""
        dOm(roa, T, mu, Psat)

        Method that computes the adimenstional Thermodynamic Grand potential
        at given density and temperature.

        Parameters
        ----------

        roa : float
            adimentional density vector [rho = rho * b]
        T : floar
            absolute adimentional temperature [Adim]
        mu : float
            adimentional chemical potential at equilibrium [Adim]
        Psat : float
            adimentional pressure at equilibrium [Adim]

        Returns
        -------
        Out: float
            Thermodynamic Grand potential [Adim]
        """

        return self.a0ad(roa, Tad)-roa*mu+Psat

    def ci(self, T):
        '''
        ci(T)

        Method that evaluates the polynomial for the influence parameters used
        in the SGT theory for surface tension calculations.

        Parameters
        ----------
        T : float
            absolute temperature [K]

        Returns
        -------
        ci: float
            influence parameters [J m5 mol-2]
        '''

        return np.polyval(self.cii, T)

    def sgt_adim_fit(self, T):
        a = self.a_eos(T)
        b = self.b
        Tfactor = R*b/a
        Pfactor = b**2/a
        rofactor = b
        tenfactor = 1000*np.sqrt(a)/b**2*(np.sqrt(101325/1.01325)*100**3)
        return Tfactor, Pfactor, rofactor, tenfactor

    def sgt_adim(self, T):
        '''
        sgt_adim(T)

        Method that evaluates adimentional factor for temperature, pressure,
        density, tension and distance for interfacial properties computations
        with SGT.

        Parameters
        ----------
        T : float
        absolute temperature [K]

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

    def ares(self, V, T, D, B, C):
        c1 = self.c1
        c2 = self.c2

        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        F = - g - (D/T)*f
        return F

    def EntropyR(self, T, P, state, v0=None, T_Step=0.1):
        """
        EntropyR(T, P, state, v0, T_step)

        Method that computes the residual entropy at given temperature and
        pressure.

        Parameters
        ----------
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
        D = self.a_eos(T)
        B = self.b
        C = self.c
        RT = R*T

        V = self._volume_solver(P, RT, D, B, C, state)
        Z = P*V/RT

        F = self.ares(V, T, D, B, C)
        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B, C)
        F2 = self.ares(V, T2, D2, B, C)
        F_1 = self.ares(V, T_1, D_1, B, C)
        F_2 = self.ares(V, T_2, D_2, B, C)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h

        Sr_TVN = -T*dFdT - F  # residual entropy (TVN) divided by R
        Sr_TPN = Sr_TVN + np.log(Z)  # residual entropy (TPN) divided by R
        Sr_TPN *= r  # J / mol K
        return Sr_TPN

    def EnthalpyR(self, T, P, state, v0=None, T_Step=0.1):
        """
        EnthalpyR(T, P, state, v0, T_step)

        Method that computes the residual enthalpy at given temperature and
        pressure.

        Parameters
        ----------
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
        D = self.a_eos(T)
        B = self.b
        C = self.c
        RT = R*T

        V = self._volume_solver(P, RT, D, B, C, state)
        Z = P*V/RT

        F = self.ares(V, T, D, B, C)

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B, C)
        F2 = self.ares(V, T2, D2, B, C)
        F_1 = self.ares(V, T_1, D_1, B, C)
        F_2 = self.ares(V, T_2, D_2, B, C)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h

        Sr_TVN = -T*dFdT - F  # residual entropy (TVN) divided by R
        Hr_TPN = F + Sr_TVN + Z - 1.  # residual entalphy divided by RT
        Hr_TPN *= (r*T)  # J / mol
        return Hr_TPN

    def CvR(self, T, P, state, v0=None, T_Step=0.1):
        """
        Cpr(T, P, state, v0, T_step)

        Method that computes the residual isochoric heat capacity at given
        temperature and pressure.

        Parameters
        ----------
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
        D = self.a_eos(T)
        B = self.b
        C = self.c
        RT = R*T

        V = self._volume_solver(P, RT, D, B, C, state)

        F = self.ares(V, T, D, B, C)

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B, C)
        F2 = self.ares(V, T2, D2, B, C)
        F_1 = self.ares(V, T_1, D_1, B, C)
        F_2 = self.ares(V, T_2, D_2, B, C)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h
        d2FdT = (-F_2/12 + 4*F_1/3 - 5*F/2 + 4*F1/3 - F2/12)/h**2

        Cvr_TVN = -T**2*d2FdT - 2*T*dFdT  # residual isochoric heat capacity
        Cvr_TVN *= r
        return Cvr_TVN

    def CpR(self, T, P, state, v0=None, T_Step=0.1):
        """
        Cpr(T, P, state, v0, T_step)

        Method that computes the residual heat capacity at given temperature
        and pressure.

        Parameters
        ----------
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
        D = self.a_eos(T)
        B = self.b
        C = self.c
        RT = R*T

        V = self._volume_solver(P, RT, D, B, C, state)

        c1 = self.c1
        c2 = self.c2

        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        F = -g - (D/T)*f

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B, C)
        F2 = self.ares(V, T2, D2, B, C)
        F_1 = self.ares(V, T_1, D_1, B, C)
        F_2 = self.ares(V, T_2, D_2, B, C)

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

    def speed_sound(self, T, P, state, v0=None, T_Step=0.1, CvId=3*r/2,
                    CpId=5*r/2):
        """
        speed_sound(T, P, state, v0, T_step, CvId, CpId)

        Method that computes the speed of sound at given temperature
        and pressure.

        This calculation requires that the molar weight [g/mol] of the fluid
        has been set in the component function.

        By default the ideal gas Cv and Cp are set to 3R/2 and 5R/2, the user
        can supply better values if available.

        Parameters
        ----------
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
        D = self.a_eos(T)
        B = self.b
        C = self.c
        RT = R*T

        V = self._volume_solver(P, RT, D, B, C, state)

        c1 = self.c1
        c2 = self.c2

        VCc1B = V + C + c1 * B
        VCc2B = V + C + c2 * B
        VCB = V + C - B

        g = np.log(VCB / V)
        f = (1. / (R*B*(c1 - c2))) * np.log(VCc1B / VCc2B)

        F = -g - (D/T)*f

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B, C)
        F2 = self.ares(V, T2, D2, B, C)
        F_1 = self.ares(V, T_1, D_1, B, C)
        F_2 = self.ares(V, T_2, D_2, B, C)

        dFdT = (F_2/12 - 2*F_1/3 + 2*F1/3 - F2/12)/h
        d2FdT = (-F_2/12 + 4*F_1/3 - 5*F/2 + 4*F1/3 - F2/12)/h**2
        dDdT = (D_2/12 - 2*D_1/3 + 2*D1/3 - D2/12)/h

        dPdT = R/VCB - dDdT/(VCc1B*VCc2B)
        dPdV = -RT/VCB**2 + D * (VCc1B+VCc2B)/(VCc1B*VCc2B)**2

        Cvr_TVN = -T**2*d2FdT - 2*T*dFdT  # residual isochoric heat capacity
        Cvr_TVN *= r

        # residual heat capacity
        Cpr = Cvr_TVN - r - (T*dPdT**2/dPdV) / 10

        # speed of sound calculation
        Cp = CpId + Cpr
        Cv = CvId + Cvr_TVN

        betas = - (Cv/Cp) / dPdV / V

        w2 = 100.*V/(betas * self.Mw)
        w = np.sqrt(w2)
        return w


# Peng Robinson EoS
c1pr = 1-np.sqrt(2)
c2pr = 1+np.sqrt(2)
omapr = 0.4572355289213825
ombpr = 0.07779607390388854


class vtprpure(vtcpure):
    def __init__(self, pure):
        vtcpure.__init__(self, pure, c1=c1pr, c2=c2pr, oma=omapr, omb=ombpr,
                         alpha_eos=alpha_soave)
        self.k = 0.37464 + 1.54226*self.w - 0.26992*self.w**2


# PRSV EoS
class vtprsvpure(vtcpure):
    def __init__(self, pure):
        vtcpure.__init__(self, pure, c1=c1pr, c2=c2pr, oma=omapr, omb=ombpr,
                         alpha_eos=alpha_sv)
        if np.all(pure.ksv == 0):
            self.k = np.zeros(2)
            self.k[0] = 0.378893+1.4897153*self.w-0.17131838*self.w**2
            self.k[0] += 0.0196553*self.w**3
        else:
            self.k = np.array(pure.ksv, ndmin=1)


# RKS EoS
c1rk = 0
c2rk = 1
omark = 0.42748
ombrk = 0.08664


class vtrkspure(vtcpure):
    def __init__(self, pure):
        vtcpure.__init__(self, pure, c1=c1rk, c2=c2rk, oma=omark, omb=ombrk,
                         alpha_eos=alpha_soave)
        self.k = 0.47979 + 1.5476*self.w - 0.1925*self.w**2 + 0.025*self.w**3


# RK EoS
class vtrkpure(vtcpure):
    def __init__(self, pure):
        vtcpure.__init__(self, pure, c1=c1rk, c2=c2rk, oma=omark, omb=ombrk,
                         alpha_eos=alpha_rk)

    def a_eos(self, T):
        alpha = self.alpha_eos(T, self.Tc)
        return self.oma*(R*self.Tc)**2*alpha/self.Pc

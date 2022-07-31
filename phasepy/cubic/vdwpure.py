from __future__ import division, print_function, absolute_import
import numpy as np
from .alphas import alpha_vdw
from .psatpure import psat
from .tsatpure import tsat
from ..constants import R, r


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
    Zmix : computes the roots of compressibility factor polynomial.
    density : computes density of mixture.
    logfugef : computes effective fugacity coefficients.
    logfugmix : computes mixture fugacity coeficcient;
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

    def __init__(self, pure):

        self.c1 = 0
        self.c2 = 0
        self.oma = 27/64
        self.omb = 1/8
        self.alpha_eos = alpha_vdw
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))

        self.Tc = np.array(pure.Tc, ndmin=1)
        self.Pc = np.array(pure.Pc, ndmin=1)
        self.w = np.array(pure.w, ndmin=1)
        self.cii = np.array(pure.cii, ndmin=1)
        self.b = self.omb*R*self.Tc/self.Pc
        self.Mw = pure.Mw

    def __call__(self, T, v):
        b = self.b
        a = self.a_eos(T)
        c1 = self.c1
        c2 = self.c2
        return R*T/(v - b) - a/((v+c1*b)*(v+c2*b))

    def pressure(self, T, v):
        """
        pressure(v, T)

        Method that computes the pressure at given volume (cm3/mol)
        and temperature T (in K)

        Parameters
        ----------
        T : float
            absolute temperature [K]
        v : float
            molar volume [cm3/mol]

        Returns
        -------
        P : float
            pressure [bar]
        """
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
        alpha = self.alpha_eos()
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
        return psat(T, self, P0)

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

    def _Zroot(self, A, B):
        a1 = (self.c1+self.c2-1)*B-1
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a3 = -B*(self.c1*self.c2*(B**2+B)+A)
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
        A = self.a_eos(T)*P/(R*T)**2
        B = self.b*P/(R*T)

        if state == 'L':
            Z = min(self._Zroot(A, B))
        if state == 'V':
            Z = max(self._Zroot(A, B))

        return P/(R*T*Z)

    def _logfug_aux(self, Z, A, B):

        logfug = Z-1-np.log(Z-B)
        logfug -= A/Z

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
        RT = R*T
        A = self.a_eos(T)*P/(RT)**2
        B = self.b*P/(RT)

        if state == 'L':
            Z = min(self._Zroot(A, B))
        if state == 'V':
            Z = max(self._Zroot(A, B))

        logfug = self._logfug_aux(Z, A, B)
        v = Z*RT/P

        return logfug, v

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
            adimentional density vector [rho = rho * b]
        T : float
            absolute adimentional temperature [adim]

        Returns
        -------
        muad: float
            chemical potential [Adim]
        """
        Pref = 1.
        mu = -T*np.log(1-ro)
        mu += -T*np.log(Pref/(T*ro)) + T
        mu += T*ro/(1-ro)
        mu -= 2*ro

        return mu

    def dOm(self, roa, Tad, mu0, Psat):
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

        return self.a0ad(roa, Tad) - roa*mu0 + Psat

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

    def ares(self, V, T, D, B):

        V_B = V - B

        g = np.log(V_B/V)
        f = 1./(R*V)

        F = -g - (D/T)*f
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
        RT = R*T

        V = self._volume_solver(P, RT, D, B, state, v0)
        Z = P*V/RT

        F = self.ares(V, T, D, B)

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B)
        F2 = self.ares(V, T2, D2, B)
        F_1 = self.ares(V, T_1, D_1, B)
        F_2 = self.ares(V, T_2, D_2, B)

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
        RT = R*T

        V = self._volume_solver(P, RT, D, B, state, v0)
        Z = P*V/RT

        F = self.ares(V, T, D, B)

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B)
        F2 = self.ares(V, T2, D2, B)
        F_1 = self.ares(V, T_1, D_1, B)
        F_2 = self.ares(V, T_2, D_2, B)

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
        RT = R*T

        V = self._volume_solver(P, RT, D, B, state, v0)

        F = self.ares(V, T, D, B)

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B)
        F2 = self.ares(V, T2, D2, B)
        F_1 = self.ares(V, T_1, D_1, B)
        F_2 = self.ares(V, T_2, D_2, B)

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
        RT = R*T

        V = self._volume_solver(P, RT, D, B, state, v0)

        V_B = V - B

        g = np.log(V_B/V)
        f = 1./(R*V)

        F = -g - (D/T)*f

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B)
        F2 = self.ares(V, T2, D2, B)
        F_1 = self.ares(V, T_1, D_1, B)
        F_2 = self.ares(V, T_2, D_2, B)

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
        RT = R*T

        V = self._volume_solver(P, RT, D, B, state, v0)

        V_B = V - B

        g = np.log(V_B/V)
        f = 1./(R*V)

        F = -g - (D/T)*f

        T1 = T+h
        T2 = T+2*h
        T_1 = T-h
        T_2 = T-2*h

        D1 = self.a_eos(T1)
        D2 = self.a_eos(T2)
        D_1 = self.a_eos(T_1)
        D_2 = self.a_eos(T_2)

        F1 = self.ares(V, T1, D1, B)
        F2 = self.ares(V, T2, D2, B)
        F_1 = self.ares(V, T_1, D_1, B)
        F_2 = self.ares(V, T_2, D_2, B)

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

        w2 = 100.*V/(betas * self.Mw)
        w = np.sqrt(w2)
        return w

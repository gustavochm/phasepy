

from __future__ import division, print_function, absolute_import
import numpy as np
from .alphas import alpha_soave, alpha_sv, alpha_rk 
from ..constants import R

#ajuste parametros alpha de EOS 
def psat(T, cubic, P0 = None):
    """
    Computes saturation pressure with cubic eos
    
    Parameters
    ----------
    T : float,
        temperatura a la que se evalua la presion, en kelvin
    cubic : object
          objeto creado a partir de puro y ecuacion de estado
    Returns
    -------   
    P : float
       saturation pressure
    """
    a = cubic.a_eos(T)
    b = cubic.b
    c1 = cubic.c1
    c2 = cubic.c2
    emin = cubic.emin
    c = cubic.c
    e=a/(b*R*T)
    
    if P0 == None:
        if e> emin: #metodo iniciacion presion cero
            U = (e-c1-c2-np.sqrt((e-c1-c2)**2-4*(c1*c2+e)))/2;
            if c1 == 0 and c2 == 0:
                S = -1-np.log(U-1)-e/U
            else:
                S = -1-np.log(U-1)-e*np.log((U+c1)/(U+c2))/(c1-c2);
            P=np.exp(S)*R*T/b; #bar
            
        else: #metodo iniciacion presion media
            a1=-R*T
            a2=-2*b*R*T*(c1+c2)+2*a
            a3=-R*T*b**2*(c1**2+4*c1*c2+c2**2)+a*b*(c1+c2-4)
            a4=-R*T*2*b**3*c1*c2*(c1+c2)+2*a*b**2*(1-c1-c2)
            a5=-R*T*b**4*c1*c2+a*b**3*(c1+c2)
            V=np.roots([a1,a2,a3,a4,a5])
            V=V[np.isreal(V)]
            V=V[V>b]
            P = cubic(T,V)
            P[P<0] = 0.
            P = P.mean()
    else:
        P =  P0 
    itmax=20
    RT = R * T
    for k in range(itmax):
        A=a*P/RT**2
        B=b*P/RT
        C=c*P/RT
        Z = cubic._Zroot(A, B, C)
        Zl = min(Z)
        Zv = max(Z)
        fugL = cubic._logfug_aux(Zl, A, B, C)
        fugV = cubic._logfug_aux(Zv, A, B,C)
        FO=fugV-fugL
        dFO=(Zv-Zl)/P
        dP=FO/dFO
        P -= dP
        if abs(dP)<1e-8: break
    vl = Zl*RT/P
    vv = Zv*RT/P
    return P, vl, vv


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
    Tc: critical temperture in K
    Pc: critical pressure in bar
    w: acentric factor
    cii : influence factor for SGT
    
    Methods
    -------
    a_eos : computes the attractive term of cubic eos.
    psat : computes saturation pressure.
    density : computes density of mixture.
    logfug : computes fugacity coefficient.
    a0ad : computes adimentional Helmholtz density energy
    muad : computes adimentional chemical potential.
    dOm : computes adimentional Thermodynamic Grand Potential.
    ci :  computes influence parameters matrix for SGT.
    sgt_adim : computes adimentional factors for SGT.

    '''
    def __init__(self, pure,c1, c2, oma, omb, alpha_eos):
        
        self.c1 = c1
        self.c2 = c2
        self.oma = oma
        self.omb = omb
        self.alpha_eos = alpha_eos 
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))
        #parametros de la mezcla
        
        self.Tc = np.array(pure.Tc, ndmin = 1) #temperaturas criticas en K
        self.Pc = np.array(pure.Pc, ndmin = 1) # presiones criticas en bar
        self.w = np.array(pure.w, ndmin = 1)
        self.cii = np.array(pure.cii, ndmin = 1) 
        self.b = self.omb*R*self.Tc/self.Pc
        self.c = np.array(pure.c, ndmin = 1)
    

            
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
        alpha=self.alpha_eos(T, self.k, self.Tc)
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
        p0, vl, vv = psat(T, self, P0)
        return p0, vl, vv
    
    
    def _Zroot(self, A, B, C):
        a1 = (self.c1+self.c2-1)*B-1 + 3 * C
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a2 += 3*C**2 + 2*C*(-1 + B*(-1 + self.c1 + self.c2))
        a3 = A*(-B + C) + (-1-B+C)* (C +self.c1 * B)*(C+self.c2*B)
        Zpol=[1,a1,a2,a3]
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots> (B - C)]
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
        RT = R * T
        A = self.a_eos(T)*P/(RT)**2
        B = self.b*P/(RT)
        C = self.c*P/(RT)
        
        if state == 'L':
            Z=min(self._Zroot(A,B,C))
        if state == 'V':
            Z=max(self._Zroot(A,B,C))
        return P/(R*T*Z)
            
    def _logfug_aux(self,Z, A, B, C):
        c1 = self.c1
        c2 = self.c2
        logfug= Z - 1 - np.log(Z + C -B)
        logfug -= (A/(c2-c1)/B)*np.log((Z+C+c2*B)/(Z+C+c1*B))
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
        
        RT = R * T
        A = self.a_eos(T)*P/(RT)**2
        B = self.b*P/(RT)
        C = self.c*P/(RT)
        
        if state == 'L':
            Z=min(self._Zroot(A, B, C))
        if state == 'V':
            Z=max(self._Zroot(A,B, C))
        
        logfug = self._logfug_aux(Z, A, B, C)
        
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

        c1 = self.c1
        c2 = self.c2
        cro = self.c * ro / self.b

        Pref = 1
        a0 = -T*ro*np.log(1- ro + cro)
        a0 += -T*ro*np.log(Pref/(T*ro))
        a0 += -ro*np.log((1+c2*ro + cro)/(1+c1*ro+cro))/((c2-c1))
        
        return a0

    def muad(self, ro,T):
        
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
        
        c1 = self.c1
        c2 = self.c2
        cro = self.c * ro / self.b
        
        Pref = 1
        mu = - ro / ((1 + cro +c1*ro) * (1 + cro + c2*ro))
        mu += T / (1 - ro + cro)
        mu += np.log((1+ cro + c2*ro)/(1+cro+c1*ro)) / (c1 - c2)
        mu -= T * np.log(1 - ro + cro)
        mu -= T * np.log(Pref/T/ro)
        
        return mu

    def dOm(self, roa, Tad, mu, Psat):
        
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
        
        return self.a0ad(roa, Tad)- roa*mu + Psat   
    
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
     

# Peng Robinson EoS
c1pr = 1-np.sqrt(2)
c2pr = 1+np.sqrt(2)
omapr = 0.4572355289213825
ombpr = 0.07779607390388854
class vtprpure(vtcpure):
    def __init__(self, pure):
        vtcpure.__init__(self,pure,c1 = c1pr, c2 = c2pr,
              oma = omapr, omb = ombpr, alpha_eos = alpha_soave)
        self.k =  0.37464 + 1.54226*self.w - 0.26992*self.w**2

# PRSV EoS
class vtprsvpure(vtcpure):
    def __init__(self,pure):
        vtcpure.__init__(self, pure, c1 = c1pr, c2 = c2pr,
              oma = omapr, omb = ombpr, alpha_eos = alpha_sv)
        if np.all(pure.ksv == 0):
            self.k = np.zeros(2)
            self.k[0] = 0.378893+1.4897153*self.w-0.17131838*self.w**2+0.0196553*self.w**3
        else:
            self.k = np.array(pure.ksv, ndmin = 1)

             
# RKS EoS
c1rk = 0
c2rk = 1
omark = 0.42748
ombrk = 0.08664
class vtrkspure(vtcpure):
    def __init__(self,pure):
        vtcpure.__init__(self, pure, c1 = c1rk, c2 = c2rk,
              oma = omark, omb = ombrk, alpha_eos=alpha_soave)        
        self.k =  0.47979 + 1.5476*self.w - 0.1925*self.w**2 + 0.025*self.w**3
        
# RK EoS
class vtrkpure(vtcpure):
    def __init__(self,pure):
        vtcpure.__init__(self, pure, c1 = c1rk, c2 = c2rk,
              oma = omark, omb = ombrk, alpha_eos=alpha_rk)                
    def a_eos(self,T):
        alpha=self.alpha_eos(T, self.Tc)
        return self.oma*(R*self.Tc)**2*alpha/self.Pc


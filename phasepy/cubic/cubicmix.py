

from __future__ import division, print_function, absolute_import
import numpy as np
from.qmr import qmr
from .wongsandler import ws_nrtl, ws_wilson, ws_nrtlt, ws_unifac, ws_rk
from .mhv import mhv_nrtl, mhv_wilson, mhv_nrtlt, mhv_unifac, mhv_rk
from .alphas import alpha_soave, alpha_sv, alpha_rk
from ..constants import R

class cubicm():
    '''
    Mixture Cubic EoS Object
    
    This object have implemeted methods for phase equilibrium 
    as for iterfacial properties calculations.
    
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
        critical temperture in K
    Pc: array_like
        critical pressure in bar
    w: array_like
        acentric factor
    cii : array_like
        influence factor for SGT
    nc : int
        number of components of mixture
    
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
    
    def __init__(self, mix, c1, c2, oma, omb, alpha_eos, mixrule):
        
        self.c1 = c1
        self.c2 = c2
        self.oma = oma
        self.omb = omb
        self.alpha_eos = alpha_eos 
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))

        
        self.Tc = np.array(mix.Tc, ndmin = 1) 
        self.Pc = np.array(mix.Pc, ndmin = 1)
        self.w = np.array(mix.w, ndmin = 1)
        self.cii = np.array(mix.cii, ndmin = 1) 
        self.b = self.omb*R*self.Tc/self.Pc
        self.nc = mix.nc
        self.beta = np.zeros([self.nc, self.nc])
        
        if mixrule == 'qmr':
            self.mixrule = qmr  
            if hasattr(mix, 'kij'):
                self.kij = mix.kij
                self.mixruleparameter = (mix.kij,)
            else: 
                self.kij = np.zeros([self.nc, self.nc])
                self.mixruleparameter = (self.kij, )
                
        elif mixrule == 'mhv_nrtl':
            self.mixrule = mhv_nrtl  
            if hasattr(mix, 'g') and hasattr(mix, 'alpha'):
                #Este se utiliza con mhv_nrtl
                self.nrtl = (mix.alpha, mix.g, mix.g1)
                self.mixruleparameter = (self.c1,self.c2, 
                                         mix.alpha, mix.g, mix.g1)
            else: 
                raise Exception('NRTL parameters needed')
                
        elif mixrule == 'mhv_nrtlt':
            self.mixrule = mhv_nrtlt  
            if hasattr(mix, 'g') and hasattr(mix, 'alpha') and hasattr(mix, 'rkternario'):
                self.nrtlt = (mix.g, mix.alpha, mix.g1, mix.rkternario)
                self.mixruleparameter = (self.c1,self.c2,mix.g, 
                                         mix.alpha, mix.g1, mix.rkternario)
            else: 
                raise Exception('NRTL/ternary parameters needed')
                
        elif mixrule == 'mhv_wilson':
            self.mixrule = mhv_wilson  
            if hasattr(mix, 'Aij'):
                #este se utiliza con mhv_wilson
                self.wilson = (mix.Aij, mix.vlrackett)
                self.mixruleparameter = (self.c1,self.c2, mix.Aij, mix.vlrackett)
            else: 
                raise Exception('Wilson parameters needed')
                
        elif mixrule == 'mhv_unifac':
            self.mixrule = mhv_unifac  
            if hasattr(mix, 'actmodelp'):
                mix.unifac()
                self.unifac = mix.actmodelp
                self.mixruleparameter = (self.c1,self.c2, *mix.actmodelp)
            else: 
                raise Exception('Unifac parameters needed')
        
        elif mixrule == 'mhv_rk':
            self.mixrule = mhv_rk  
            if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT'):
                self.rk = (mix.rkp, mix.rkpT, mix.combinatoria)
                self.mixruleparameter = (self.c1,self.c2, mix.rkp, mix.rkpT,
                                         mix.combinatoria)
            else:
                raise Exception('RK parameters needed')
                
        elif mixrule == 'ws_nrtl':
            self.mixrule = ws_nrtl  
            if hasattr(mix, 'Kijws'):
                self.Kijws = mix.Kijws
            else:
                self.Kijws = np.zeros([self.nc, self.nc]) 

            if hasattr(mix, 'g') and hasattr(mix, 'alpha'):
                #Este se utiliza con mhv_nrtl
                c1, c2 = self.c1, self.c2
                C = np.log((1+c1)/(1+c2))/(c1-c2)
                self.nrtl = (mix.alpha, mix.g, mix.g1)
                self.mixruleparameter = (C, self.Kijws, 
                                         mix.alpha, mix.g, mix.g1)
            else: 
                raise Exception('NRTL parameters needed')
                
        elif mixrule == 'ws_wilson':
            self.mixrule = ws_wilson  
            if hasattr(mix, 'Kijws'):
                self.Kijws = mix.Kijws
            else:
                self.Kijws = np.zeros([self.nc, self.nc]) 
            if hasattr(mix, 'Aij'):
                #este se utiliza con mhv_wilson
                c1, c2 = self.c1, self.c2
                C = np.log((1+c1)/(1+c2))/(c1-c2)
                self.wilson = (mix.Aij, mix.vlrackett)
                self.mixruleparameter = (C, self.Kijws, mix.Aij, mix.vlrackett)
            else: 
                raise Exception('Wilson parameters needed')
                
        elif mixrule == 'ws_rk':
            self.mixrule = ws_rk
            if hasattr(mix, 'Kijws'):
                self.Kijws = mix.Kijws
            else:
                self.Kijws = np.zeros([self.nc, self.nc])
                
            if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT'):
                c1, c2 = self.c1, self.c2
                C = np.log((1+c1)/(1+c2))/(c1-c2)
                self.rk = (mix.rkp, mix.rkpT, mix.combinatoria)
                self.mixruleparameter = (C, self.Kijws, mix.rkp, mix.rkpT,
                                         mix.combinatoria)
            else:
                raise Exception('RK parameters needed')
                
        elif mixrule == 'ws_unifac':
            self.mixrule = ws_unifac  
            if hasattr(mix, 'Kijws'):
                self.Kijws = mix.Kijws
            else:
                self.Kijws = np.zeros([self.nc, self.nc])       

            c1, c2 = self.c1, self.c2
            C = np.log((1+c1)/(1+c2))/(c1-c2)
            mix.unifac()
            self.unifac = mix.actmodelp
            self.mixruleparameter = (C, self.Kijws,*self.unifac)
        else: 
            raise Exception('Mixrule not valid')
            
            
    #Cubic EoS methods    
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
        a : array_like
            atractive term array
        """
        alpha = self.alpha_eos(T,self.k,self.Tc)
        a  = self.oma*(R*self.Tc)**2*alpha/self.Pc
        return a
    
    def _Zroot(self,A,B):
        a1 = (self.c1+self.c2-1)*B-1
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a3 = -B*(self.c1*self.c2*(B**2+B)+A)
        Zpol=[1,a1,a2,a3]
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots>B]
        return Zroots
        
    def Zmix(self, X, T, P):
        '''
        Zmix (X, T, P)
        
        Method that computes the roots of the compresibility factor polynomial
        at given mole fractions (X), Temperature (T) and Pressure (P)
        
        Parameters
        ----------
        
        X : array_like
            mole fraction vector
        T : float
            absolute temperature in K
        P : float
            pressure in bar

        Returns
        -------        
        Z : array_like
            roots of Z polynomial
        '''
        a = self.a_eos(T)
        am, bm, ep, ap, bp = self.mixrule(X,T, a, self.b,*self.mixruleparameter)
        A = am*P/(R*T)**2
        B = bm*P/(R*T)
        return self._Zroot(A,B)

    def density(self, X, T, P, state):
        """ 
        density(X, T, P, state)
        Method that computes the density of the mixture at X, T, P

        
        Parameters
        ----------
        
        X : array_like
            mole fraction vector
        T : float
            absolute temperature in K
        P : float
            pressure in bar
        state : string
            'L' for liquid phase and 'V' for vapour phase

        Returns
        -------
        density: array_like
            density vector of the mixture in moll/cm3
        """
        if state == 'L':
            Z=min(self.Zmix(X,T,P))
        elif state == 'V':
            Z=max(self.Zmix(X,T,P))
        return X*P/(R*T*Z)
    
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
        b = self.b
        a = self.a_eos(T)
        am, bm, ep, ap, bp = self.mixrule(X, T, a, b, *self.mixruleparameter)
        if state == 'V':
            Z=max(self.Zmix(X,T,P))
        elif state == 'L':
            Z=min(self.Zmix(X,T,P))
        
        B=(bm*P)/(R*T)
        
        logfug=(Z-1)*(bp/bm) - np.log(Z-B)
        logfug -= (ep/(self.c2-self.c1))*np.log((Z+self.c2*B)/(Z+self.c1*B))
        return logfug, v0
        
    def logfugmix(self, X, T, P, state, v0 = None):
        
        """ 
        logfugmix(X, T, P, state)
        
        Method that computes the mixture fugacity coefficient at given
        composition, temperature and pressure. 

        Parameters
        ----------
        
        X : array_like
            mole fraction vector
        T : float
            absolute temperature in K
        P : float
            pressure in bar
        state : string
            'L' for liquid phase and 'V' for vapour phase
        
        Returns
        -------
        lofgfug : array_like
            effective fugacity coefficients
        """

        a = self.a_eos(T)
        am, bm, ep, ap, bp = self.mixrule(X,T,a,self.b,*self.mixruleparameter)
        if state == 'V':
            Z=max(self.Zmix(X,T,P))
        elif state == 'L':
            Z=min(self.Zmix(X,T,P))
        
        B=(bm*P)/(R*T)
        A=(am*P)/(R*T)**2
        
        logfug=Z - 1 - np.log(Z-B)
        logfug -= (A/(self.c2-self.c1)/B)*np.log((Z+self.c2*B)/(Z+self.c1*B))
        
        return logfug, v0
    
    def a0ad(self, roa, T):
        
        """ 
        a0ad(roa, T)
        
        Method that computes the adimenstional Helmholtz density energy at given
        density and temperature.

        Parameters
        ----------
        
        roa : array_like
            adimentional density vector
        T : float
            absolute temperature in K

        Returns
        -------        
        a0ad: float
            adimenstional Helmholtz density energy
        """
        
        c1 = self.c1
        c2 = self.c2
        ai = self.a_eos(T)
        bi = self.b
        a = ai[0]
        b = bi[0]
        ro = np.sum(roa)
        X = roa/ro
        
        am, bm, ep, ap, bp = self.mixrule(X, T, ai, bi, *self.mixruleparameter)
        Prefa=1*b**2/a
        Tad = R*T*b/a
        ama = am/a
        bma = bm/b
        
        a0 = np.sum(np.nan_to_num(Tad*roa*np.log(roa/ro)))
        a0 += -Tad*ro*np.log(1-bma*ro)
        a0 += -Tad*ro*np.log(Prefa/(Tad*ro))
        a0 += -ama*ro*np.log((1+c2*ro*bma)/(1+c1*ro*bma))/((c2-c1)*bma)
        
        return a0
    
    def muad(self, roa, T):
        
        """ 
        muad(roa, T)
        
        Method that computes the adimenstional chemical potential at given
        density and temperature.

        Parameters
        ----------
        
        roa : array_like
            adimentional density vector
        T : float
            absolute temperature in K

        Returns
        -------        
        muad : array_like
            adimentional chemical potential vector
        """
        
        c1 = self.c1
        c2 = self.c2
        ai = self.a_eos(T)
        bi = self.b
        a = ai[0]
        b = bi[0]
        ro = np.sum(roa)
        X = roa/ro
        
        am, bm, ep, ap, bp = self.mixrule(X,T, ai, bi,*self.mixruleparameter)
        Prefa=1*b**2/a
        Tad = R*T*b/a
        apa = ap/a
        ama = am/a
        bma = bm/b
        bad = bp/b
        
        mui = -Tad*np.log(1-bma*ro)
        mui += -Tad*np.log(Prefa/(Tad*roa))+Tad
        mui += bad*Tad*ro/(1-bma*ro)
        
        mui -= ro*(ama+apa) * np.log((1+c2*bma*ro)/(1+c1*bma*ro)) / ((c2-c1)*bma*ro)
        mui += bad*ama * np.log((1+c2*bma*ro)/(1+c1*bma*ro)) / ((c2-c1)*bma**2)
        mui -= bad*ama*ro / ((1+c2*bma*ro)*(1+c1*bma*ro)*bma)
        
        return mui
    
    
    def dOm(self, roa, T, mu, Psat):
        """ 
        dOm(roa, T, mu, Psat)
        
        Method that computes the adimenstional Thermodynamic Grand potential at given
        density and temperature.

        Parameters
        ----------
        
        roa : array_like
            adimentional density vector
        T : float
            absolute temperature in K
        mu : array_like
            adimentional chemical potential at equilibrium
        Psat : float
            adimentional pressure at equilibrium

        Returns
        -------       
        dom: float
            Thermodynamic Grand potential
        """
        dom = self.a0ad(roa, T) - np.sum(np.nan_to_num(roa*mu)) + Psat
        return dom
        
    def _lnphi0(self, T, P):
        
        nc = self.nc
        a_puros = self.a_eos(T)
        Ai = a_puros*P/(R*T)**2
        Bi = self.b*P/(R*T)
        pols = np.array([Bi-1,-3*Bi**2-2*Bi+Ai,(Bi**3+Bi**2-Ai*Bi)])
        Zs = np.zeros([nc,2])
        for i in range(nc):
            zroot = np.roots(np.hstack([1,pols[:,i]]))
            zroot = zroot[zroot>Bi[i]]
            Zs[i,:]=np.array([max(zroot),min(zroot)])
    
        lnphi = self.logfug(Zs.T,Ai,Bi)
        lnphi = np.amin(lnphi,axis=0)
    
        return lnphi
    
    def beta_sgt(self, beta):
        self.beta = beta
    
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

        n=self.nc
        ci=np.zeros(n)
        for i in range(n):
            ci[i]=np.polyval(self.cii[i],T)
        self.cij=np.sqrt(np.outer(ci,ci))*(1-self.beta)
        return self.cij
    
    def sgt_adim(self, T):
        '''
        sgt_adim(T)
        
        Method that evaluates adimentional factor for temperature, pressure, 
        density, tension and distance for interfacial properties computations with
        SGT.
        
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
        ci = self.ci(T)[0,0]
        Tfactor = R*b0/a0
        Pfactor = b0**2/a0
        rofactor = b0
        tenfactor = 1000*np.sqrt(a0*ci)/b0**2*(np.sqrt(101325/1.01325)*100**3) 
        zfactor = np.sqrt(a0/ci*10**5/100**6)*10**-10 
        return Tfactor, Pfactor, rofactor, tenfactor, zfactor
                 
# Peng Robinson EoS 
c1pr = 1-np.sqrt(2)
c2pr = 1+np.sqrt(2)
omapr = 0.4572355289213825
ombpr = 0.07779607390388854
class prmix(cubicm):   
    def __init__(self, mix, mixrule = 'qmr'):
        cubicm.__init__(self, mix,c1 = c1pr, c2 = c2pr,
              oma = omapr, omb = ombpr, alpha_eos = alpha_soave, mixrule = mixrule )
        
        self.k =  0.37464 + 1.54226*self.w - 0.26992*self.w**2
        
# Peng Robinson SV EoS 
class prsvmix(cubicm):   
    def __init__(self, mix, mixrule = 'qmr'):
        cubicm.__init__(self, mix, c1 = c1pr, c2 = c2pr,
              oma = omapr, omb = ombpr, alpha_eos = alpha_sv, mixrule = mixrule )
        if np.all(mix.ksv == 0):
            self.k = np.zeros([self.nc,2])
            self.k[:,0] = 0.378893+1.4897153*self.w-0.17131838*self.w**2+0.0196553*self.w**3
        else:
             self.k = np.array(mix.ksv) 

# RK - EoS
c1rk = 0
c2rk = 1
omark = 0.42748
ombrk = 0.08664
class rksmix(cubicm):   
    def __init__(self, mix, mixrule = 'qmr'):
        cubicm.__init__(self, mix, c1 = c1rk, c2 = c2rk,
              oma = omark, omb = ombrk, alpha_eos = alpha_soave, mixrule = mixrule)
        self.k =  0.47979 + 1.5476*self.w - 0.1925*self.w**2 + 0.025*self.w**3
      
#RKS- EoS        
class rkmix(cubicm):   
    def __init__(self, mix, mixrule = 'qmr'):
        cubicm.__init__(self, mix, c1 = c1rk, c2 = c2rk,
              oma = omark, omb = ombrk, alpha_eos = alpha_rk, mixrule = mixrule)
        def a_eos(self,T):
            alpha=self.alpha_eos(T, self.Tc)
            return self.oma*(R*self.Tc)**2*alpha/self.Pc


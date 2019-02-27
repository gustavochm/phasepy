import numpy as np
from .mixingrules import qmr
from .alphas import alpha_vdw
from ..constants import R

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
    
    
    def __init__(self, mix):
        
        self.c1 = 0
        self.c2 = 0
        self.oma = 27/64
        self.omb = 1/8
        self.alpha_eos = alpha_vdw 
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))

        
        self.Tc = np.array(mix.Tc, ndmin = 1) 
        self.Pc = np.array(mix.Pc, ndmin = 1) 
        self.w = np.array(mix.w, ndmin = 1)
        self.cii = np.array(mix.cii, ndmin = 1) 
        self.b = self.omb*R*self.Tc/self.Pc
        self.nc = mix.nc
        self.beta = np.zeros([self.nc, self.nc])
        
        self.mixrule = qmr  
        if hasattr(mix, 'kij'):
            self.kij = mix.kij
            self.mixruleparameter = (mix.kij,)
        else: 
            raise Exception('Matriz Kij no ingresada')
                
            
    #metodos cubica    
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
        alpha=self.alpha_eos()
        return self.oma*(R*self.Tc)**2*alpha/self.Pc
    
    def _Zroot(self,A,B):
        a1 = (self.c1+self.c2-1)*B-1
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a3 = -B*(self.c1*self.c2*(B**2+B)+A)
        Zpol=[1,a1,a2,a3]
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots>B]
        return Zroots
        
    def Zmix(self,X,T,P):
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
        am,bm,ep,ap = self.mixrule(X,T, a, self.b,*self.mixruleparameter)
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
        a = self.a_eos(T)
        am, bm, ep, ap = self.mixrule(X, T, a, self.b, *self.mixruleparameter)
        if state == 'V':
            Z=max(self.Zmix(X,T,P))
        elif state == 'L':
            Z=min(self.Zmix(X,T,P))
        
        B=(bm*P)/(R*T)
        
        logfug=(Z-1)*(self.b/bm)-np.log(Z-B)
        
        logfug -= B*ep/Z

        return logfug, v0
        
    def logfugmix(self, X, T, P, estado, v0 = None):
        
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
        am,bm,ep,ap = self.mixrule(X,T,a,self.b,*self.mixruleparameter)
        if estado == 'V':
            Z=max(self.Zmix(X,T,P))
        elif estado == 'L':
            Z=min(self.Zmix(X,T,P))
        
        B=(bm*P)/(R*T)
        A=(am*P)/(R*T)**2
        
        return self.logfug(Z,A,B), v0
    
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
        
        ai = self.a_eos(T)
        a = ai[0]
        ro = np.sum(roa)
        X = roa/ro
        
        am,bm,ep,ap = self.mixrule(X, T, ai, self.b, *self.mixruleparameter)
        Prefa=1*self.b[0]**2/a
        Tad = R*T*self.b[0]/a
        ama = am/a
        bma = bm/self.b[0]
        
        a0 = np.sum(np.nan_to_num(Tad*roa*np.log(roa/ro)))
        a0 += -Tad*ro*np.log(1-bma*ro)
        a0 += -Tad*ro*np.log(Prefa/(Tad*ro))
        a0 += -ama*ro**2

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
        
        
        ai = self.a_eos(T)
        a = ai[0]
        ro = np.sum(roa)
        X = roa/ro
        
        am,bm,ep,ap = self.mixrule(X,T, ai, self.b,*self.mixruleparameter)
        Prefa=1*self.b[0]**2/a
        Tad = R*T*self.b[0]/a
        apa = ap/a
        ama = am/a
        bma = bm/self.b[0]
        bad = self.b/self.b[0]
        
        mui = -Tad*np.log(1-bma*ro)
        mui += -Tad*np.log(Prefa/(Tad*roa))+Tad
        mui += bad*Tad*ro/(1-bma*ro)
        mui -= ro*(apa+ama)

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
        
        return self.a0ad(roa, T) - np.sum(np.nan_to_num(roa*mu)) + Psat
        
    def lnphi0(self, T, P):
        
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
        self.cij=np.sqrt(np.outer(ci,ci))
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
        tenfactor = 1000*np.sqrt(a0*ci)/b0**2*(np.sqrt(101325/1.01325)*100**3) #para dejarlo en nM/m
        zfactor = np.sqrt(a0/ci*10**5/100**6)*10**-10 #Para dejarlo en Amstrong
        return Tfactor, Pfactor, rofactor, tenfactor, zfactor
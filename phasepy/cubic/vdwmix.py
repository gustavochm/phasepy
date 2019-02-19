import numpy as np
from .mixingrules import qmr
from .alphas import alpha_vdw
from ..constants import R

class vdwm(object):
    
    def __init__(self, mix):
        
        self.c1 = 0
        self.c2 = 0
        self.oma = 27/64
        self.omb = 1/8
        self.alpha_eos = alpha_vdw 
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))
        #parametros de la mezcla
        
        self.Tc = np.array(mix.Tc, ndmin = 1) #temperaturas criticas en K
        self.Pc = np.array(mix.Pc, ndmin = 1) # presiones criticas en bar
        self.w = np.array(mix.w, ndmin = 1)
        self.cii = np.array(mix.cii, ndmin = 1) 
        self.b = self.omb*R*self.Tc/self.Pc
        self.nc = mix.nc
        
        self.mixrule = qmr  
        if hasattr(mix, 'kij'):
            self.kij = mix.kij
            self.mixruleparameter = (mix.kij,)
        else: 
            raise Exception('Matriz Kij no ingresada')
                
            
        #metodos cubica    
        def a_eos(self,T):
            """ 
            a_eos(T),
            Method that computes atractive term of cubic eos at fixed T (in K)
        
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
            a = self.a_eos(T)
            am,bm,ep,ap = self.mixrule(X,T, a, self.b,*self.mixruleparameter)
            A = am*P/(R*T)**2
            B = bm*P/(R*T)
            return self._Zroot(A,B)

        def density(self, X, T, P, state):
            """ 
            Method that computes the density of the mixture at X, T, P
    
            
            Inputs
            ----------
            
            x : array_like, mole fraction vector
            T : absolute temperature in K
            P : pressure in bar
            state : 'L' for liquid phase and 'V' for vapour phase
            
            Out: array_like, density vector of the mixture
            """
            if state == 'L':
                Z=min(self.Zmix(X,T,P))
            elif state == 'V':
                Z=max(self.Zmix(X,T,P))
            return X*P/(R*T*Z)
        
        def logfugef(self, X, T, P, state, v0 = None):

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
            
            #temperatura ingresada en K, densidad ingresada adimensional
            ai = self.a_eos(T)
            a = ai[0]
            ro = np.sum(roa)
            X = roa/ro
            
            am,bm,ep,ap = self.mixrule(X, T, ai, self.b, *self.mixruleparameter)
            Prefa=1*self.b[0]**2/a
            Tad = R*T*self.b[0]/a
            ama = am/a
            bma = bm/self.b[0]
            #bad = self.b/self.b[0]
            
            a0 = np.sum(np.nan_to_num(Tad*roa*np.log(roa/ro)))
            a0 += -Tad*ro*np.log(1-bma*ro)
            a0 += -Tad*ro*np.log(Prefa/(Tad*ro))
            a0 += -ama*ro**2

            return a0
        
        def muad(self, roa, T):
            
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
            #todos los terminos ingresados deben ser adimensionales, excepto temperatura en K
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
        
        
        def ci(self, T):

            n=self.nc
            ci=np.zeros(n)
            for i in range(n):
                ci[i]=np.polyval(self.cii[i],T)
            self.cij=np.sqrt(np.outer(ci,ci))
            return self.cij
        
        def sgt_adim(self, T):
             a0 = self.a_eos(T)[0]
             b0 = self.b[0]
             ci = self.ci(T)[0,0]
             Tfactor = R*b0/a0
             Pfactor = b0**2/a0
             rofactor = b0
             tenfactor = 1000*np.sqrt(a0*ci)/b0**2*(np.sqrt(101325/1.01325)*100**3) #para dejarlo en nM/m
             zfactor = np.sqrt(a0/ci*10**5/100**6)*10**-10 #Para dejarlo en Amstrong
             return Tfactor, Pfactor, rofactor, tenfactor, zfactor
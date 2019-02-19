import numpy as np
from .alphas import alpha_vdw
from .psatpure import psat 
from ..constants import R

class vdwpure(object):
    def __init__(self, puro):
        
        self.c1 = 0
        self.c2 = 0
        self.oma = 27/64
        self.omb = 1/8
        self.alpha_eos = alpha_vdw 
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))
        #parametros de la mezcla
        
        self.Tc = np.array(puro.Tc, ndmin = 1) #temperaturas criticas en K
        self.Pc = np.array(puro.Pc, ndmin = 1) # presiones criticas en bar
        self.w = np.array(puro.w, ndmin = 1)
        self.cii = np.array(puro.cii, ndmin = 1) 
        self.b = self.omb*R*self.Tc/self.Pc

            
    def __call__(self, T, v):
        b = self.b
        a = self.a_eos(T)
        c1 = self.c1
        c2 = self.c2
        return R*T/(v - b) - a/((v+c1*b)*(v+c2*b))
        
    def a_eos(self,T):
        alpha = self.alpha_eos()
        return self.oma*(R*self.Tc)**2*alpha/self.Pc

    def psat(self, T, P0 = None):
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
        
        A = self.a_eos(T) * P /(R*T)**2
        B = self.b* P / (R*T)
        
        if state == 'L':
            Z=min(self._Zroot(A,B))
        if state == 'V':
            Z=max(self._Zroot(A,B))
        
        logfug = self._logfug_aux(Z, A, B)
        
        return logfug   


    def a0ad(self, ro,T):
        #Calculo de energia de Helmohtlz adimensional, se ingresa densidad y temperatura adimensionales        
        Pref = 1 
        a0 = -T*ro*np.log(1-ro)
        a0 += -T*ro*np.log(Pref/(T*ro))
        a0 += -ro**2
        
        return a0

    def muad(self, ro, T):
        #Calculo de potencial quimico adimensional, se ingresa densidad y temperatura adimensionales
        
        Pref = 1 
        mu = -T*np.log(1-ro)
        mu += -T*np.log(Pref/(T*ro)) + T
        mu += T*ro/(1-ro)           
        mu -= 2*ro
        
        return mu

    def dOm(self, roa, Tad, mu0, Psat):
        #todos los terminos ingresados deben ser adimensionales
        return self.a0ad(roa, Tad)- roa*mu0 + Psat   
    
    def ci(self, T):
        return np.polyval(self.cii, T)
    
    def sgt_adim(self, T):
         a = self.a_eos(T)
         b = self.b[0]
         ci = self.ci(T)
         Tfactor = R*b/a
         Pfactor = b**2/a
         rofactor = b
         tenfactor = 1000*np.sqrt(a*ci)/b**2*(np.sqrt(101325/1.01325)*100**3) #para dejarlo en nM/m
         zfactor = np.sqrt(a/ci*10**5/100**6)*10**-10 #Para dejarlo en Amstrong
         return Tfactor, Pfactor, rofactor, tenfactor, zfactor
    

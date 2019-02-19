import numpy as np
from .alphas import alpha_soave, alpha_sv, alpha_rk
from .psatpure import psat 
from ..constants import R

class cpure(object):
    def __init__(self, puro,c1, c2, oma, omb, alpha_eos):
        
        self.c1 = c1
        self.c2 = c2
        self.oma = oma
        self.omb = omb
        self.alpha_eos = alpha_eos 
        self.emin = 2+self.c1+self.c2+2*np.sqrt((1+self.c1)*(1+self.c2))
        #parametros de la mezcla
        
        self.Tc = np.array(puro.Tc, ndmin = 1) #temperaturas criticas en K
        self.Pc = np.array(puro.Pc, ndmin = 1) # presiones criticas en bar
        self.w = np.array(puro.w, ndmin = 1)
        self.cii = np.array(puro.cii, ndmin = 1) 
        self.b = self.omb*R*self.Tc/self.Pc
        
        #k de VdW-S
        #self.k =  0.449 + 1.5928*self.w - 0.19463*self.w**2+0.025*self.w**3

            
    def __call__(self, T, v):
        b = self.b
        a = self.a_eos(T)
        c1 = self.c1
        c2 = self.c2
        return R*T/(v - b) - a/((v+c1*b)*(v+c2*b))
        
    def a_eos(self,T):
        alpha=self.alpha_eos(T, self.k, self.Tc)
        return self.oma*(R*self.Tc)**2*alpha/self.Pc

    def psat(self, T, P0 = None):
        return psat(T, self, P0)

    def _Zroot(self,A,B):
        a1 = (self.c1+self.c2-1)*B-1
        a2 = self.c1*self.c2*B**2-(self.c1+self.c2)*(B**2+B)+A
        a3 = -B*(self.c1*self.c2*(B**2+B)+A)
        Zpol=[1,a1,a2,a3]
        Zroots = np.roots(Zpol)
        Zroots = np.real(Zroots[np.imag(Zroots) == 0])
        Zroots = Zroots[Zroots>B]
        return Zroots

    def density(self,T,P,estado):

        A = self.a_eos(T)*P/(R*T)**2
        B = self.b*P/(R*T)
        
        if estado == 'L':
            Z=min(self._Zroot(A,B))
        if estado == 'V':
            Z=max(self._Zroot(A,B))
        return P/(R*T*Z)       
    
    def _logfug_aux(self,Z, A, B):
              
        logfug=Z-1-np.log(Z-B)
        logfug -= (A/(self.c2-self.c1)/B)*np.log((Z+self.c2*B)/(Z+self.c1*B))
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
        #Calculo de enrgia de Helmohtlz adimensional, se ingresa densidad y temperatura adimensionales        
        Pref = 1 
        a0 = -T*ro*np.log(1-ro)
        a0 += -T*ro*np.log(Pref/(T*ro))
        a0 += -ro*np.log((1+self.c2*ro)/(1+self.c1*ro))/((self.c2-self.c1))
        
        return a0

    def muad(self, ro,T):
        #Calculo de potencial quimico adimensional, se ingresa densidad y temperatura adimensionales
        
        Pref = 1 
        mu = -T*np.log(1-ro)
        mu += -T*np.log(Pref/(T*ro)) + T
        mu += T*ro/(1-ro)           
        mu += -np.log((1+self.c2*ro)/(1+self.c1*ro))/(self.c2-self.c1)
        mu += -ro/((1+self.c2*ro)*(1+self.c1*ro))  
        
        return mu

    def dOm(self, roa, Tad, mu, Psat):
        #todos los terminos ingresados deben ser adimensionales
        return self.a0ad(roa, Tad)- roa*mu + Psat   
    
    def ci(self, T):
        return np.polyval(self.cii, T)
    
    def sgt_adim_fit(self, T):
         a = self.a_eos(T)
         b = self.b
         Tfactor = R*b/a
         Pfactor = b**2/a
         rofactor = b
         tenfactor = 1000*np.sqrt(a)/b**2*(np.sqrt(101325/1.01325)*100**3) #para dejarlo en nM/m
         return Tfactor, Pfactor, rofactor, tenfactor
     
    def sgt_adim(self, T):
         a = self.a_eos(T)
         b = self.b
         ci = self.ci(T)
         Tfactor = R*b/a
         Pfactor = b**2/a
         rofactor = b
         tenfactor = 1000*np.sqrt(a*ci)/b**2*(np.sqrt(101325/1.01325)*100**3) #para dejarlo en nM/m
         zfactor = np.sqrt(a/ci*10**5/100**6)*10**-10 #Para dejarlo en Amstrong
         return Tfactor, Pfactor, rofactor, tenfactor, zfactor
     

# Ecuacion de estado Peng Robinson    
c1pr = 1-np.sqrt(2)
c2pr = 1+np.sqrt(2)
omapr = 0.4572355289213825
ombpr = 0.07779607390388854
class prpure(cpure):   
    def __init__(self, pure):
        cpure.__init__(self,pure,c1 = c1pr, c2 = c2pr,
              oma = omapr, omb = ombpr, alpha_eos = alpha_soave)
        self.k =  0.37464 + 1.54226*self.w - 0.26992*self.w**2


class prsvpure(cpure):   
    def __init__(self,pure):
        cpure.__init__(self, pure, c1 = c1pr, c2 = c2pr,
              oma = omapr, omb = ombpr, alpha_eos = alpha_sv)
        if np.all(pure.ksv == 0):
            self.k = np.zeros(2)
            self.k[0] = 0.378893+1.4897153*self.w-0.17131838*self.w**2+0.0196553*self.w**3
        else:
            self.k = np.array(pure.ksv, ndmin = 1) #parametros utilizado para evaluar la funcion alpha_eos

             
# Ecuacion de estado de RKS
c1rk = 0
c2rk = 1
omark = 0.42748
ombrk = 0.08664
class rkspure(cpure):   
    def __init__(self,pure):
        cpure.__init__(self, pure, c1 = c1rk, c2 = c2rk,
              oma = omark, omb = ombrk, alpha_eos=alpha_soave)        
        self.k =  0.47979 + 1.5476*self.w - 0.1925*self.w**2 + 0.025*self.w**3
        
# Ecuacion de estado de RK
c1rk = 0
c2rk = 1
omark = 0.42748
ombrk = 0.08664
class rkpure(cpure):   
    def __init__(self,pure):
        cpure.__init__(self, pure, c1 = c1rk, c2 = c2rk,
              oma = omark, omb = ombrk, alpha_eos=alpha_rk)                
        def a_eos(self,T):
            alpha=self.alpha_eos(T, self.Tc)
            return self.oma*(R*self.Tc)**2*alpha/self.Pc


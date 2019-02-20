import numpy as np
from collections import Counter 
from pandas import read_excel
import os
from copy import copy
from .constants import Na
from itertools import combinations


class component(object):
    
    def __init__(self,name='None',Tc = 0,Pc = 0, Zc = 0, Vc = 0, w = 0, cii = 0,
                 ksv = [0, 0], Ant = [0,0,0],  GC = None,
                 m = 0, sigma = 0 , e = 0, kapaAB = 0, eAB = 0, site = [0,0,0]): 
        '''
        class component
        Creates an object with pure component info
        
        Name (name)
        Critical Temperature (Tc)
        Critical Pressure (Pc)
        Critical compresibility (Zc)
        Critical Volume (Vc)
        Acentric factor (w)
        Influence coefficient SGT (cii)
        Parameters alpha PRSV eos (if fitted) (ksv)
        Antoine parameters (Ant)
        Group contribution info (GC)
        
        Methods
        ------
        psat
        tsat
        vlrackett
        ci
        '''
        self.name = name
        self.Tc = Tc #Critical Temperature in K
        self.Pc = Pc #Critical Pressure in bar
        self.Zc = Zc #Critical compresibility factor
        self.Vc = Vc #Critical volume in cm3/mol
        self.w = w #Acentric Factor
        self.Ant = Ant #Antoine coefficeint, base e = 2.71 coeficientes de antoine, list or array
        self.cii = cii #Influence factor SGT, list or array
        self.ksv = ksv #
        self.GC = GC # Dict, Group contribution info
        self.nc = 1
        
        #Parametros Saft
        self.m = m
        self.sigma = 1e-9*sigma*(Na**(1./3)) 
        self.e = e 
        self.kapaAB = kapaAB 
        self.eAB = eAB
        self.site = site 
        
        
    def psat(self,T):
        '''
        Method that computes saturation pressure at T using Ant eq.
        '''
        coef = self.Ant
        return np.exp(coef[0]-coef[1]/(T+coef[2]))
    
    
    def tsat(self, P):
        '''
        Method that computes the saturation temperature at P using Ant eq.
        '''
        coef =self.Ant
        T = - coef[2] + coef[1] / (coef[0] - np.log(P))
        
        return T
    
    def vlrackett(self,T):
        '''
        Method that computes the liquid volume using Rackett eq.
        '''
        Tr=T/self.Tc
        V=self.Vc*self.Zc**((1-Tr)**(2/7))
        return V
    
    def ci(self, T):
        '''
        Method that evaluates the polynomial for cii coeffient of SGT
        cii must be in J m^5 / mol and T in K.
        '''
        return np.polyval(self.cii, T)
    
class mixture(object):
    '''
    class mixture
    Creates an object that cointains info about a mixture.
    
    Methods
    -------
    add_component : adds a component to the mixture
    psat : computes saturation pressure of pures
    tsat: computes saturation temperature of pures
    vlrackett : computes liquid volume of pure
    copy: returns a copy of the object
    
    kij_cubic : add kij matrix for QMR mixrule
    NRTL : add energy interactions and aleatory factor for NRTL model
    wilson : add energy interactions for wilson model
    rk: polynomial parameters for RK G exc model
    unifac: read Dortmund data base for the mixture
    ci : computes cij matrix at T for SGT
    '''
    def __init__(self, component1, component2):
        
        self.names = [component1.name ,component2.name]
        self.Tc = [component1.Tc, component2.Tc]
        self.Pc = [component1.Pc, component2.Pc]
        self.Zc = [component1.Zc, component2.Zc]
        self.w = [component1.w, component2.w]
        self.Ant = [component1.Ant, component2.Ant]
        self.Vc = [component1.Vc, component2.Vc]
        self.cii = [component1.cii, component2.cii]
        self.ksv = [component1.ksv, component2.ksv]
        self.nc = 2
        self.GC = [component1.GC,  component2.GC]
        
        self.m = [component1.m, component2.m]
        self.sigma = [component1.sigma, component2.sigma]
        self.e = [component1.e, component2.e]
        self.kapaAB = [component1.kapaAB, component2.kapaAB]
        self.eAB = [component1.eAB, component2.eAB]
        self.site = [component1.site, component2.site]
        
    def add_component(self,component):
        self.names.append(component.name)
        self.Tc.append(component.Tc)
        self.Pc.append(component.Pc)
        self.Zc.append(component.Zc)
        self.Vc.append(component.Vc)
        self.w.append(component.w)
        self.Ant.append(component.Ant)
        self.cii.append(component.cii)
        self.ksv.append(component.ksv)
        self.GC.append(component.GC)
        
        self.m.append(component.m)
        self.sigma.append(component.sigma)
        self.e.append(component.e)
        self.kapaAB.append(component.kapaAB)
        self.eAB.append(component.eAB)
        self.site.append(component.site)
        
        self.nc += 1
        
    def psat(self,T):
        '''
        Method that computes saturation pressure at T using Ant eq.
        '''
        coef = np.vstack(self.Ant)
        return np.exp(coef[:,0]-coef[:,1]/(T+coef[:,2]))
    
    def tsat(self, P):
        '''
        Method that computes saturation temperature at P using Ant eq.
        '''
        coef=np.vstack(self.Ant)
        T = - coef[:,2] + coef[:,1] / (coef[:,0] - np.log(P))
        return T
    
    
    def vlrackett(self,T):
        '''
        Method that computes the liquid volume using Rackett eq.

        '''
        Tc = np.array(self.Tc)
        Vc = np.array(self.Vc)
        Zc = np.array(self.Zc)
        Tr=T/Tc
        V=Vc*Zc**((1-Tr)**(2/7))
        return V    
    
    def kij_saft(self, k):
        self.K = k
        
    def mixrule2_saft(self, K, LA, NU):
        self.K = K
        self.LA = LA
        self.NU = NU
        
    
    def kij_cubic(self,k):
        ''' 
        Method that add kij matrix for QMR mixrule. Matrix must be symmetrical
        and the main diagonal must be zero.
        '''
        self.kij = k
        
    def NRTL(self, alpha, g , g1 = None):
        '''
        Method that adds NRTL parameters to the mixture.
        Matrix g (in K), main diagonal must be zero.
        Matrix g1 (in 1/K), main diagonal must be zero.
        Matrix alpha: symmetrical and main diagonal must be zero.
        
        tau = ((g + g1*T)/T)
        '''
        #ingresar matriz de parametros g y de aleotoridad de modelo NRTL
        self.g = g
        self.alpha = alpha
        if g1 is None:
            g1 = np.zeros_like(g)
        self.g1 = g1
        self.actmodelp = (self.alpha, self.g, self.g1)
        
    def rkt(self, D):
        '''
        Method that adds a ternary polynomial modification to NRTL model
        '''
        self.rkternario = D        
        self.actmodelp = (self.g, self.alpha, self.g1, self.rkternario)
    
    def wilson(self,A):
        '''
        Method that adds wilson model parameters to the mixture
        Matrix A main diagonal must be zero. Values in K.
        '''
        self.Aij = A
        self.actmodelp = (self.Aij , self.vlrackett)
        
    def rkb(self, c, c1 = None):
        '''
        Method that adds binary Redlich Kister polynomial coefficients for
        excess Gibbs energy.
        Coefficients are calculated:
        G = c + c1/T
        '''
        self.rkbinario = c
        if c1 is None:
            c1 = np.zeros_like(c)
        self.rkbinarioT = c1
        self.actmodelp = (c, c1)
        
    def rk(self, c, c1 = None):
        '''
        Method that adds binary Redlich Kister polynomial coefficients for
        excess Gibbs energy.
        Coefficients are calculated:
        G = c + c1/T
        '''
        combinatoria = list(combinations(range(self.nc),2))
        c = np.atleast_2d(c)
        self.rkp = c
        if c1 is None:
            c1 = np.zeros_like(c)
        self.rkpT = c1
        self.combinatoria = np.array(combinatoria)
        self.actmodelp = (c, c1, combinatoria)
    
    
    def unifac(self):
        
        '''
        Method that read the Dortmund database for UNIFAC model
        After calling this function activity coefficient are ready
        to be calculated.
        '''
        
        #lectura de los parametros de UNIFAC
        database = os.path.join(os.path.dirname(__file__), 'database')
        database +=  '/dortmund.xlsx'
        qkrk = read_excel(database, 'RkQk', index_col = 'Especie')
        a0 = read_excel(database, 'A0', index_col = 'Grupo')
        a0.fillna(0, inplace = True)
        a1 = read_excel(database, 'A1', index_col = 'Grupo')
        a1.fillna(0, inplace = True)
        a2 = read_excel(database, 'A2', index_col = 'Grupo')
        a2.fillna(0, inplace = True)
        
        #Lectura de puros y creacion de informacion de grupos de mezcla
        puregc = self.GC
        mix = Counter()
        for i in puregc:
            mix += Counter(i)
            
        subgroups = list(mix.keys())
        
        #creacion de diccionarios por especie
        vk = []
        dics = []
        for i in puregc:
            d = dict.fromkeys(subgroups, 0)
            d.update(i)
            dics.append(d)
            vk.append(list(d.values()))
        Vk = np.array(vk)
        
        groups = qkrk.loc[subgroups, 'Grupo ID'].values
        
        a = a0.loc[groups, groups].values
        b = a1.loc[groups, groups].values
        c = a2.loc[groups, groups].values
        
        #lectura del volumen de los grupos presentes
        rq = qkrk.loc[subgroups, ['Rk', 'Qk']].values
        Qk = rq[:,1]
        
        ri, qi = (Vk@rq).T
        ri34 = ri**(0.75)
        
        Xmi = (Vk.T/Vk.sum(axis=1)).T
        t = Xmi*Qk
        tethai = (t.T/t.sum(axis=1)).T
        
        self.actmodelp = (qi, ri, ri34, Vk, Qk, tethai, a, b, c)
        
        
    def ci(self,T):
        
        '''
        Method that computes the matrix of cij interaction parameter for SGT at
        T.
        beta is a modification to the interaction parameters and must be added 
        as a symmetrical matrix with main diagonal set to zero.
        '''
        
        #ingresar beta como matriz
        n = len(self.cii)
        ci = np.zeros(n)
        for i in range(n):
            ci[i] = np.polyval(self.cii[i],T)
        self.cij = np.sqrt(np.outer(ci,ci))
        return self.cij
    
    def copy(self):
        return copy(self)

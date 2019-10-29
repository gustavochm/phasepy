from __future__ import division, print_function, absolute_import
import numpy as np
from collections import Counter 
from pandas import read_excel
import os
from copy import copy
from itertools import combinations
from .saft_forcefield import saft_forcefield

class component(object):
    '''
    Creates an object with pure component info
    
    Parameters
    ----------
    name : str
        Name of the component
    Tc : float
        Critical temperature
    Pc : float
        Critical Pressure
    Zc : float
        critical compresibility factor
    Vc : float
        critical volume
    w  : float
        acentric factor
    cii : list
        polynomial coefficient for influence parameter in SGT
    ksv : list
        parameter for alpha for PRSV EoS, if fitted
    Ant : list
        Antoine correlation parameters
    GC : dict
        Group contribution info

    
    Attributes
    ----------
    
    name : str
        Name of the component
    Tc : float
        Critical temperature
    Pc : float
        Critical Pressure
    Zc : float
        critical compresibility factor
    Vc : float
        critical volume
    w  : float
        acentric factor
    cii : list
        polynomial coefficient for influence parameter in SGT
    ksv : list
        parameter for alpha for PRSV EoS, if fitted
    Ant : list
        Antoine correlation parameters
    GC : dict
        Group contribution info
    
    Methods
    -------
    psat : computes saturation pressure with Antoine correlation
    tsat : compues saturation temperature with Antoine correlation
    vlrackett : computes liquid volume with Rackett correlation
    ci :  evaluates influence parameter polynomial
    '''
    
    def __init__(self,name='None',Tc = 0,Pc = 0, Zc = 0, Vc = 0, w = 0, cii = 0,
                 ksv = [0, 0], Ant = [0,0,0],  GC = None,
                 ms = 1, sigma = 0 , eps = 0, lambda_r = 12., lambda_a = 6.,
                eAB = 0., rcAB = 1., rdAB = 0.4, sites = [0,0,0]): 
        
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
        
        #Saft Parameters
        
        self.ms = ms
        self.sigma = sigma
        self.eps = eps 
        self.lambda_a = np.asarray(lambda_a)
        self.lambda_r = np.asarray(lambda_r)
        self.lambda_ar = self.lambda_r + self.lambda_a
        #Association Parameters
        self.eAB = eAB
        self.rcAB = rcAB * sigma
        self.rdAB = rdAB * sigma
        self.sites = sites
        
        
        
    def psat(self,T):
        """
        Method that computes saturation pressure at T using Ant eq. Expontential
        base is used.
    
        Parameters
        ----------
        T : float
            absolute temperature in K
    
        Returns
        -------
        Psat : foat
            Saturation pressure in bar
        """
        
        coef = self.Ant
        return np.exp(coef[0]-coef[1]/(T+coef[2]))
    
    
    def tsat(self, P):
        """ 
        Method that computes the saturation temperature at P using Ant eq.
        Expontential base is used.
    
        Parameters
        ----------
        Psat : foat
            Saturation pressure in bar
            
        Returns
        -------
        T : float
            absolute temperature in K

        """
     
        coef =self.Ant
        T = - coef[2] + coef[1] / (coef[0] - np.log(P))
        
        return T
    
    def vlrackett(self,T):
        """
        Method that computes the liquid volume using Rackett eq.
    
        Parameters
        ----------
        T : float
            absolute temperature in K
            
        Returns
        -------
        vl : float
            liquid volume in cm3/mol

        """
        Tr=T/self.Tc
        V=self.Vc*self.Zc**((1-Tr)**(2/7))
        return V
    
    def ci(self, T):
        """ 
        Method that evaluates the polynomial for cii coeffient of SGT
        cii must be in J m^5 / mol and T in K.
    
        Parameters
        ----------
        T : float
            absolute temperature in K
            
        Returns
        -------
        ci : float
            influence parameter at given temperature

        """

        return np.polyval(self.cii, T)
    
    def saftvrmie(self, ms, rhol07):
        lambda_r, lambda_a, ms, eps, sigma = saft_forcefield(ms, self.Tc, self.w, rhol07)
        self.lambda_a = np.asarray(lambda_a)
        self.lambda_r = np.asarray(lambda_r)
        self.lambda_ar = self.lambda_r + self.lambda_a
        self.ms = ms
        self.sigma = sigma
        self.eps = eps
    
class mixture(object):
    '''
    class mixture
    Creates an object that cointains info about a mixture.

    Parameters
    ----------
    component1 : object
        component created with component class
    component2 : object
        component created with component class
        
    Attributes
    ----------
    name : list
        Name of the component
    Tc : list
        Critical temperature
    Pc : list
        Critical Pressure
    Zc : list
        critical compresibility factor
    Vc : list
        critical volume
    w  : list
        acentric factor
    cii : list
        polynomial coefficient for influence parameter in SGT
    ksv : list
        parameter for alpha for PRSV EoS, if fitted
    Ant : list
        Antoine correlation parameters
    GC : list
        Group contribution info
    
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
        
        self.lr = [component1.lambda_r, component2.lambda_r]
        self.la = [component1.lambda_a, component2.lambda_a]
        self.sigma = [component1.sigma, component2.sigma]
        self.eps = [component1.eps, component2.eps]
        self.ms = [component1.ms, component2.ms]
        self.eAB = [component1.eAB, component2.eAB]
        self.rc = [component1.rcAB, component2.rcAB]
        self.rd = [component1.rdAB, component2.rdAB]
        self.sitesmix = [component1.sites, component2.sites]
        
    def add_component(self,component):
        """
        Method that add a component to the mixture
        """
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
        
        self.lr.append(component.lambda_r)
        self.la.append(component.lambda_a)
        self.sigma.append(component.sigma)
        self.eps.append(component.eps)
        self.ms.append(component.ms)
        self.eAB.append(component.eAB)
        self.rc.append(component.rcAB)
        self.rd.append(component.rdAB)
        self.sitesmix.append(component.sites)
        
        self.nc += 1
        
    def psat(self,T):
        """         
        Method that computes saturation pressure at T using Ant eq. Expontential
        base is used.
    
        Parameters
        ----------
        T : float
            absolute temperature in K
    
        Returns
        -------
        Psat : array_like
            Saturation pressure in bar
        """
        coef = np.vstack(self.Ant)
        return np.exp(coef[:,0]-coef[:,1]/(T+coef[:,2]))
    
    def tsat(self, P):
        """ 
        Method that computes the saturation temperature at P using Ant eq.
        Expontential base is used.
    
        Parameters
        ----------
        Psat : foat
            Saturation pressure in bar
            
        Returns
        -------
        T : array_like
            absolute temperature in K

        """
        coef=np.vstack(self.Ant)
        T = - coef[:,2] + coef[:,1] / (coef[:,0] - np.log(P))
        return T
    
    
    def vlrackett(self,T):
        """         
        Method that computes the liquid volume using Rackett eq.
    
        Parameters
        ----------
        T : float
            absolute temperature in K
            
        Returns
        -------
        vl : float
            liquid volume in cm3/mol

        """
        Tc = np.array(self.Tc)
        Vc = np.array(self.Vc)
        Zc = np.array(self.Zc)
        Tr=T/Tc
        V=Vc*Zc**((1-Tr)**(2/7))
        return V 
    
    def kij_saft(self, kij):
        self.kij_saft = kij
    
    def kij_ws(self, kij):
        self.Kijws = kij
    
    def kij_cubic(self,k):
        '''
        Method that add kij matrix for QMR mixrule. Matrix must be symmetrical
        and the main diagonal must be zero.
        
        Parameters
        ----------
        k: array like
            matrix of interactions parameters

        ''' 


        self.kij = k
        
    def NRTL(self, alpha, g , g1 = None):
        '''
        Method that adds NRTL parameters to the mixture
        
        Parameters
        ----------
        g: array like
            matrix of energy interactions in K
        g1: array_like
            matrix of energy interactions in 1/K
        alpha: array_like
            aleatory factor
            
        Note
        ----
        Parameters are evaluated as a function of temperature:
        tau = ((g + g1*T)/T)

        '''
        
        self.g = g
        self.alpha = alpha
        if g1 is None:
            g1 = np.zeros_like(g)
        self.g1 = g1
        self.actmodelp = (self.alpha, self.g, self.g1)
        
    def rkt(self, D):
        '''
        Method that adds a ternary polynomial modification to NRTL model
        
        Parameters
        ----------
        D: array_like
            ternary interaction parameters values

        '''
        self.rkternario = D        
        self.actmodelp = (self.alpha, self.g, self.g1, self.rkternario)
    
    def wilson(self,A):
        '''
        Method that adds wilson model parameters to the mixture
        Matrix A main diagonal must be zero. Values in K.
        
        Parameters
        ----------
        A: array_like
            interaction parameters values

        '''
        
        self.Aij = A
        self.actmodelp = (self.Aij , self.vlrackett)
        
    def rkb(self, c, c1 = None):
        '''
        Method that adds binary Redlich Kister polynomial coefficients for
        excess Gibbs energy.
        
        Parameters
        ----------
        c: array_like
            polynomial values adim
        c1: array_like, optional
            polynomial values in K
            
        Note
        ----
        Parameters are evaluated as a function of temperature:
        
        G = c + c1/T
        
        '''
        self.rkb = c
        if c1 is None:
            c1 = np.zeros_like(c)
        self.rkb = c1
        self.actmodelp = (c, c1)
        
    def rk(self, c, c1 = None):
        '''
        Method that adds binary Redlich Kister polynomial coefficients for
        excess Gibbs energy.
        
        Parameters
        ----------
        c: array_like
            polynomial values adim
        c1: array_like, optional
            polynomial values in K
            
        Note
        ----
        Parameters are evaluated as a function of temperature:
        
        G = c + c1/T
        
        '''
        nc = self.nc
        combinatory = np.array(list(combinations(range(nc),2)), dtype = np.int)
        self.combinatory = combinatory
        c = np.atleast_2d(c)
        self.rkp = c
        if c1 is None:
            c1 = np.zeros_like(c)
        self.rkpT = c1
        self.actmodelp = (c, c1, combinatory)
    
    
    def unifac(self):
        """         
        Method that read the Dortmund database for UNIFAC model
        After calling this function activity coefficient are ready
        to be calculated.

        """
        
        #UNIFAC database reading
        database = os.path.join(os.path.dirname(__file__), 'database')
        database +=  '/dortmund.xlsx'
        qkrk = read_excel(database, 'RkQk', index_col = 'Especie')
        a0 = read_excel(database, 'A0', index_col = 'Grupo')
        a0.fillna(0, inplace = True)
        a1 = read_excel(database, 'A1', index_col = 'Grupo')
        a1.fillna(0, inplace = True)
        a2 = read_excel(database, 'A2', index_col = 'Grupo')
        a2.fillna(0, inplace = True)
        

        #Reading pure component and mixture group contribution info
        puregc = self.GC
        mix = Counter()
        for i in puregc:
            mix += Counter(i)
            
        subgroups = list(mix.keys())
        

        #Dicts created for each component
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
        
        #Reading info of present groups
        rq = qkrk.loc[subgroups, ['Rk', 'Qk']].values
        Qk = rq[:,1]
        
        ri, qi = (Vk@rq).T
        ri34 = ri**(0.75)
        
        Xmi = (Vk.T/Vk.sum(axis=1)).T
        t = Xmi*Qk
        tethai = (t.T/t.sum(axis=1)).T
        
        self.actmodelp = (qi, ri, ri34, Vk, Qk, tethai, a, b, c)
        
        
    def ci(self,T):
        
        """ 
        Method that computes the matrix of cij interaction parameter for SGT at
        T.
        beta is a modification to the interaction parameters and must be added 
        as a symmetrical matrix with main diagonal set to zero.
    
        Parameters
        ----------
        T : float
            absolute temperature in K
            
        Returns
        -------
        ci : array_like
            influence parameter matrix at given temperature

        """
        
        n = len(self.cii)
        ci = np.zeros(n)
        for i in range(n):
            ci[i] = np.polyval(self.cii[i],T)
        self.cij = np.sqrt(np.outer(ci,ci))
        return self.cij
    
    def copy(self):
        """ 
        Method that return a copy of the mixture

            
        Returns
        -------
        mix : object
            returns a copy a of the mixture
        """
        
        return copy(self)

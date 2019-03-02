from __future__ import division, print_function, absolute_import
import numpy as np
from .virial import Tsonopoulos, virial, Virialmix
from .nrtl import nrtl 

R = 83.14 #bar cm3/mol K 
class virialgama:
    '''
    Creates a model with mixture using a virial eos to describe vapour phase
    and an activity coefficient model for liquid phase.
    
    Parameters
    ----------
    mix : object
        mixture created with mixture class
    virialmodel : function
        function to compute virial coefficients
    actmodel : function
        function to compute activity coefficients
    
    Methods
    -------
    logfugef: computes effective fugacity coefficients
    
    '''
    
    def __init__(self, mix, virialmodel = Tsonopoulos, actmodel = nrtl):
        
        self.psat = mix.psat
        self.vl = mix.vlrackett
        self.mezcla = mix
        self.virialmodel = virialmodel
        self.actmodel = actmodel 
        self.actmodelp = mix.actmodelp
        self.nc = mix.nc
        
        self.Tij, self.Pij, self.Zij, self.wij = Virialmix(mix)

    def logfugef(self, X,T,P,state, v0 = None):
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
        Bi,Bp = virial(X,T,self.Tij, self.Pij, self.wij, self.virialmodel)
        if state == 'L':
            psat = self.psat(T)
            pointing = self.vl(T)*(P-psat)/(R*T)
            fugPsat = Bi*psat/(R*T)
            act = self.actmodel(X,T,*self.actmodelp)
            return act+np.log(psat/P)+pointing+fugPsat, v0
        elif state == 'V':
            return Bp*P/(R*T), v0
    
            
            
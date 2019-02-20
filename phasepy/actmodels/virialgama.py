import numpy as np
from .virial import Tsonopoulos, virial
from .nrtl import nrtl 

R = 83.14 #bar cm3/mol K 
class virialgama(object):
    '''
    class
    Creates a model with mixture using a virial eos to describe vaopur phase
    and an activity coefficient model for liquid phase.
    
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
    
    def logfugef(self, X,T,P,estado, v0 = None):
        Bi,Bp = virial(X,T,self.mezcla,self.virialmodel)
        if estado == 'L':
            psat = self.psat(T)
            pointing = self.vl(T)*(P-psat)/(R*T)
            fugPsat = Bi*psat/(R*T)
            act = self.actmodel(X,T,*self.actmodelp)
            return act+np.log(psat/P)+pointing+fugPsat, v0
        elif estado == 'V':
            return Bp*P/(R*T), v0
    
            
            
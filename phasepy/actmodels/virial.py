import numpy as np
from ..constants import R

def Virialmix(mix):
    '''
    VirialMix creates the needed arrays to work with multicomponent
    virial eos
    
    Parameters
    ----------
    mix: object
        Created from two or meore components

    Returns
    -------
    Tij: array_like
        square array of critical temperatures
    Pij: array_like
        square array of critical pressures
    Zij: array_like
        square array of critical compresibility factor
    wij: array_like
        square array of acentric factor
        
    '''
    Tc = np.asarray(mix.Tc)
    Zc = np.asarray(mix.Zc)
    w = np.asarray(mix.w)
    Vc = np.asarray(mix.Vc)

    Vc3 = Vc**(1/3)
    vij = (np.add.outer(Vc3, Vc3)/2)**3
    kij = 1 - np.sqrt(np.outer(Vc, Vc))/vij
    Tij = np.sqrt(np.outer(Tc, Tc))*(1-kij)
    wij = np.add.outer(w, w)/2
    Zij = np.add.outer(Zc, Zc)/2
    Pij = Zij*R*Tij/vij
    
    return Tij,Pij,Zij,wij

def Tsonopoulos(T, Tij, Pij, wij):
    '''
    Computes the virial coefficient for a mixture at given 
    temperature with Tsonopoulos correlation
    
    Parameters
    ----------
    T : float
        absolute temperature in K
    Tij: array_like
        square array of critical temperatures
    Pij: array_like
        square array of critical pressures
    wij: array_like
        square array of acentric factor

    Returns
    -------
    Bij: array_like
        square array of virial coefficient        
    '''
    Tr=T/Tij
    B0=0.1145-0.330/Tr-0.1385/Tr**2-0.0121/Tr**3-0.000607/Tr**8
    B1=0.0637+0.331/Tr**2-0.423/Tr**3-0.008/Tr**8
    Bij=(B0+wij*B1)*R*Tij/Pij
    return Bij

def Abbott(T, Tij, Pij, wij):
    '''
    Computes the virial coefficient for a mixture at given 
    temperature with Abbott correlation
    
    Parameters
    ----------
    T : float
        absolute temperature in K
    Tij: array_like
        square array of critical temperatures
    Pij: array_like
        square array of critical pressures
    wij: array_like
        square array of acentric factor

    Returns
    -------
    Bij: array_like
        square array of virial coefficient        
    '''
    Tr=T/Tij
    B0=0.083-0.422/Tr**1.6
    B1=0.139-0.172/Tr**4.2
    Bij=(B0+wij*B1)*R*Tij/Pij
    return Bij

def ideal_gas(T, Tij, Pij, wij):
    '''
    Computes the virial coefficient for a mixture at given 
    temperature for an ideal gas (Bij = 0)
    
    Parameters
    ----------
    T : float
        absolute temperature in K
    Tij: array_like
        square array of critical temperatures
    Pij: array_like
        square array of critical pressures
    wij: array_like
        square array of acentric factor

    Returns
    -------
    Bij: array_like
        square array of virial coefficient        
    '''
    Bij = np.zeros_like(Tij)
    return Bij

def virial(x, T, Tij, Pij, wij, virialmodel):
    '''
    Computes the virial coefficient and partial virial coefficient for a mixture
    at given temperature and composition.
    
    Parameters
    ----------
    x: array_like
        fraction mole array
    T : float
        absolute temperature in K
    Tij: array_like
        square array of critical temperatures
    Pij: array_like
        square array of critical pressures
    wij: array_like
        square array of acentric 
    virialmodel : function
        Function that computes the virial coefficient.

    Returns
    -------
    Bi: array_like
        array of virial coefficient of pure component
    Bp : array_like
        array of partial virial coefficients
    '''
    
    Bij = virialmodel(T, Tij, Pij, wij)
    
    Bx = Bij*x
    #virial de mezcla
    Bm = np.sum(Bx.T*x)
    #virial parcial
    Bp = 2*np.sum(Bx, axis=1) - Bm
    
    return np.diag(Bij), Bp
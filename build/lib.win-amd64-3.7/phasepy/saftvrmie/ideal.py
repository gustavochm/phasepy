import numpy as np

h = 6.626070150e-34 # J s
me = 9.10938291e-31 #1/Kg

#Eq 68
def aideal(rho, beta):
    broglie_vol = h / np.sqrt(2*np.pi * me / beta)
    a = np.log(rho * broglie_vol**3 ) - 1
    return a

def daideal_drho(rho, beta):
    broglie_vol = h / np.sqrt(2*np.pi * me / beta)
    a = np.log(rho * broglie_vol**3 ) - 1
    #a = 0.
    da = 1./rho
    return np.hstack([a, da])

def d2aideal_drho(rho, beta):
    broglie_vol = h / np.sqrt(2*np.pi * me / beta)
    a = np.log(rho * broglie_vol**3 ) - 1
    #a= 0.
    da = 1./rho
    d2a = -1/rho**2
    return np.hstack([a, da, d2a])
import numpy as np
from .. import gauss

from .monomer_aux import Xi, dXi, d2Xi
from .monomer_aux import dkHS, d2kHS, d3kHS

from .a1sB_monomer import da1B_eval, d2a1B_eval, d3a1B_eval
from .a1sB_monomer import x0lambda_eval 

from .pertubaciones_eval import ahs, dahs_deta, d2ahs_deta
from .pertubaciones_eval import a1m
from .pertubaciones_eval import a2m,  da2m_deta, d2a2m_deta
from .pertubaciones_eval import da2m_new_deta, d2a2m_new_deta, d3a2m_new_deta
from .pertubaciones_eval import a3m, da3m_deta, d2a3m_deta

from .ideal import aideal, daideal_drho, d2aideal_drho
from .monomer import amono, damono_drho, d2amono_drho
from .chain import achain, dachain_drho, d2achain_drho

from .density_solver import  density_topliss, density_newton
from .psat_saft import psat

#Constants 
kb = 1.3806488e-23 # K/J
Na = 6.02214e23 
R = Na * kb       
    
        
def U_mie(r, c, eps, lambda_r, lambda_a):
    u = c * eps * (r**lambda_r - r**lambda_a)
    return u


#Second perturbation
phi16 = np.array([[7.5365557, -37.60463, 71.745953, -46.83552, -2.467982, -0.50272, 8.0956883],
[-359.44, 1825.6, -3168.0, 1884.2, -0.82376, -3.1935, 3.7090],
[1550.9, -5070.1, 6534.6, -3288.7, -2.7171, 2.0883, 0],
[-1.19932, 9.063632, -17.9482, 11.34027, 20.52142, -56.6377, 40.53683],
[-1911.28, 21390.175, -51320.7, 37064.54, 1103.742, -3264.61, 2556.181],
[9236.9, -129430., 357230., -315530., 1390.2, -4518.2, 4241.6]])

nfi = np.arange(0,7)
nfi_num = nfi[:4]
nfi_den = nfi[4:]

#Eq 20
def fi(alpha, i):
    phi = phi16[i-1]
    num = np.dot(phi[nfi_num], np.power(alpha, nfi_num))
    den = 1 + np.dot(phi[nfi_den], np.power(alpha, nfi_den - 3))
    return num/den
        
        
class saftvrmie_pure():
    
    def __init__(self, pure):
        
        self.ms = pure.ms
        self.sigma = pure.sigma
        self.eps = pure.eps 
        self.lambda_a = pure.lambda_a
        self.lambda_r = pure.lambda_r
        self.lambda_ar = self.lambda_r + self.lambda_a
        
        dif_c = self.lambda_r - self.lambda_a
        self.c = self.lambda_r / dif_c * (self.lambda_r / self.lambda_a) ** (self.lambda_a / dif_c)
        alpha = self.c*(1/(self.lambda_a - 3) - 1/(self.lambda_r - 3))
        self.alpha = alpha
        
        self.f1 = fi(alpha, 1)
        self.f2 = fi(alpha, 2)
        self.f3 = fi(alpha, 3)
        self.f4 = fi(alpha, 4)
        self.f5 = fi(alpha, 5)
        self.f6 = fi(alpha, 6)
        
        roots, weights = gauss(100)
        self.roots = roots
        self.weights = weights
        
        self.umie = U_mie(1./roots, self.c, self.eps, self.lambda_r, self.lambda_a)
        
        #For SGT Computations
        if pure.cii == 0.:
            cii = self.ms * (0.12008072630855947 + 2.2197907527439655 * alpha)
            cii *= np.sqrt(Na**2 * self.eps * self.sigma**5)
            cii **= 2
            self.cii = cii
        else:
            self.cii = np.asarray(pure.cii, ndmin = 1)
    
    def d(self,beta):
        integrer = np.exp(-beta * self.umie)
        d = self.sigma * (1. - np.dot(integrer, self.weights))
        return d
    
    def eta_sigma(self,rho):
        return self.ms * rho * np.pi * self.sigma**3 / 6

    def eta_bh(self, rho, d):
        deta_drho =  self.ms * np.pi * d**3 / 6
        eta = deta_drho * rho
        return eta, deta_drho
    
    def density(self, T, P, state, rho0 = None):
        if rho0 == None:
            rho = density_topliss(state, T, P, self)
        else:
            rho = density_newton(rho0, T, P, self)
        return rho
    
    def psat(self, T, P0 = None, v0 = [None, None]):
        P, vl, vv = psat(self, T, P0 , v0)
        return P, vl, vv
    
    def ares(self, rho, T):
        #Constants evaluated at given density and temperatura
        beta = 1 / (kb*T)
        dia = self.d(beta)
        tetha  = np.exp(beta*self.eps)-1
        eta, deta = self.eta_bh(rho, dia)
        nsigma = self.eta_sigma(rho)
        x0 = self.sigma/dia
        x03 = x0**3
        
        #parameters needed for evaluating the helmothlz contributions
        x0_a1, x0_a2, x0_a12, x0_a22  = x0lambda_eval(x0, self.lambda_a, self.lambda_r, self.lambda_ar) 
        a1sb_a1, a1sb_a2 = da1B_eval(x0, eta, self.lambda_a, self.lambda_r, self.lambda_ar, self.eps) #valor y derivada
        dkhs = dkHS(eta) #valor y derivada
        xi = Xi(x03, nsigma, self.f1, self.f2, self.f3) #solo valor
        da2m_new = da2m_new_deta(x0_a2, a1sb_a2,  dkhs, self.c, self.eps) #sol derivada

        #ideal conribution 
        #a_ideal = aideal(rho, beta)

        #monomer 
        ahs_eval = ahs(eta) #solo valor
        a1m_eval = a1m(x0_a1 , a1sb_a1, self.c) #valor y derivada
        a2m_eval = a2m(x0_a2, a1sb_a2[0], dkhs[0], xi, self.c, self.eps) #solo valor
        a3m_eval = a3m(x03, nsigma, self.eps, self.f4, self.f5, self.f6) #solo valor
        a_mono = amono(ahs_eval, a1m_eval[0], a2m_eval, a3m_eval, beta, self.ms)

        #chain contributions
        a_chain = achain(x0, eta, 
              x0_a12, a1sb_a1[0], a1m_eval[1], 
              x03, nsigma, self.alpha, tetha,
              x0_a22, a1sb_a2[0], da2m_new, dkhs[0],
              dia, deta, rho, beta,  self.eps, self.c, self.ms)

        #total helmolthz 
        a = a_mono + a_chain
        return a
    
    def dares_drho(self, rho, T):
        #Constants evaluated at given density and temperatura
        beta = 1 / (kb*T)
        dia = self.d(beta)
        tetha  = np.exp(beta*self.eps)-1
        eta, deta = self.eta_bh(rho, dia)
        nsigma = self.eta_sigma(rho)
        x0 = self.sigma/dia
        x03 = x0**3
        
        drho = np.array([1. , deta, deta**2])
        
        #parameters needed for evaluating the helmothlz contributions
        x0_a1, x0_a2, x0_a12, x0_a22  = x0lambda_eval(x0, self.lambda_a, self.lambda_r, self.lambda_ar) 
        a1sb_a1, a1sb_a2 = d2a1B_eval(x0, eta, self.lambda_a, self.lambda_r, self.lambda_ar, self.eps) #valor y 1 y 2 derivada
        dkhs = d2kHS(eta) #valor y  1y 2 derivada
        dxi = dXi(x03, nsigma, self.f1, self.f2, self.f3) #valor y derivada
        da2m_new = d2a2m_new_deta(x0_a2, a1sb_a2, dkhs, self.c, self.eps) #1 y 2da derivada
        
        #ideal conribution 
        #a_ideal = daideal_drho(rho, beta)

        
        #monomer 
        ahs_eval = dahs_deta(eta) #valor y derivada
        a1m_eval = a1m(x0_a1 , a1sb_a1, self.c) #valor y 1 y 2 derivada
        a2m_eval = da2m_deta(x0_a2, a1sb_a2[:2], dkhs[:2], dxi, self.c, self.eps) #valor y 1 derivada
        a3m_eval = da3m_deta(x03, nsigma, self.eps, self.f4, self.f5, self.f6) #valor y 1 derivada
        a_mono = damono_drho(ahs_eval, a1m_eval[:2], a2m_eval, a3m_eval, beta, drho[:2], self.ms)

        #chain contributions
        a_chain = dachain_drho(x0, eta, 
              x0_a12, a1sb_a1[:2], a1m_eval, 
              x03, nsigma, self.alpha, tetha,
              x0_a22, a1sb_a2[:2], da2m_new, dkhs[:2],
              dia, drho, rho, beta,  self.eps, self.c, self.ms)

        #total helmolthz 
        a = a_mono + a_chain
        return a
    
    def d2ares_drho(self, rho, T):
        #Constants evaluated at given density and temperatura
        beta = 1 / (kb*T)
        dia = self.d(beta)
        tetha  = np.exp(beta*self.eps)-1
        eta, deta = self.eta_bh(rho, dia)
        nsigma = self.eta_sigma(rho)
        x0 = self.sigma/dia
        x03 = x0**3
        
        drho = np.array([1. , deta, deta**2, deta**3])
        
        #parameters needed for evaluating the helmothlz contributions
        x0_a1, x0_a2, x0_a12, x0_a22  = x0lambda_eval(x0, self.lambda_a, self.lambda_r, self.lambda_ar) 
        a1sb_a1, a1sb_a2 = d3a1B_eval(x0, eta, self.lambda_a, self.lambda_r, self.lambda_ar, self.eps) #valor y 1 y 2 derivada
        dkhs = d3kHS(eta) #valor y  1,2, y 3 derivada
        dxi = d2Xi(x03, nsigma, self.f1, self.f2, self.f3) #valor y derivada
        da2m_new = d3a2m_new_deta(x0_a2, a1sb_a2,  dkhs, self.c, self.eps) #1 y 2da derivada
        
        #ideal conribution 
        #a_ideal = daideal_drho(rho, beta)
        
        #monomer 
        ahs_eval = d2ahs_deta(eta) #valor y derivada
        a1m_eval = a1m(x0_a1 , a1sb_a1, self.c) #valor y 1 y 2 derivada
        a2m_eval = d2a2m_deta(x0_a2, a1sb_a2[:3], dkhs[:3], dxi, self.c, self.eps) #valor y 1 derivada
        a3m_eval = d2a3m_deta(x03, nsigma,  self.eps, self.f4, self.f5, self.f6) #valor y 1 derivada
        a_mono = d2amono_drho(ahs_eval, a1m_eval[:3], a2m_eval, a3m_eval, beta, drho[:3], self.ms)
        
        #chain contributions
        a_chain = d2achain_drho(x0, eta, 
              x0_a12, a1sb_a1[:3], a1m_eval, 
              x03, nsigma, self.alpha, tetha,
              x0_a22, a1sb_a2[:3], da2m_new, dkhs[:3],
              dia, drho, rho, beta, self.eps, self.c, self.ms)
        #total helmolthz 
        a = a_mono + a_chain
        return a
    
    def afcn(self, rho, T):
        a = self.ares(rho, T)
        beta = 1 / (kb*T)
        a += aideal(rho, beta)
        a *= (Na/beta)
        return a
    
    def dafcn_drho(self, rho, T):
        a = self.dares_drho(rho, T)
        beta = 1 / (kb*T)
        a += daideal_drho(rho, beta)
        a *= (Na/beta)
        return a
    
    def d2afcn_drho(self, rho, T):
        a = self.d2ares_drho(rho, T)
        beta = 1 / (kb*T)
        a += d2aideal_drho(rho, beta)
        a *= (Na/beta)
        return a
    
    def dP_drho(self, rho, T):
        afcn, dafcn, d2afcn = self.d2afcn_drho(rho, T) 
        Psaft = rho**2 * dafcn / Na
        dPsaft = 2 * rho * dafcn + rho**2 * d2afcn
        dPsaft /= Na
        return Psaft, dPsaft
    
    

    def logfug(self, T, P, state, v0 = None):
        if v0 is None:
            rho = self.density(T, P, state, None)
        else:
            rho0 = Na/v0
            rho = self.density(T, P, state, rho0)
        v = Na/rho    
        ares = self.ares(rho, T)
        Z = P * v / (R * T)
        lnphi = ares  + (Z - 1.) - np.log(Z)
        return lnphi, v
    
    def sgt_adim(self, T):

        Tfactor = 1
        Pfactor = 1
        rofactor = 1
        tenfactor = np.sqrt(self.cii) * 1000 #To give tension in mN/m
        zfactor = 10**-10

        return Tfactor, Pfactor, rofactor, tenfactor, zfactor
    
    def a0ad(self, rho, T):
        
        rhomolecular = rho * Na
        a0 = self.afcn(rhomolecular, T)
        a0 *= rho
        
        return a0
    
    def muad(self, rho, T):
        
        rhomolecular = rho * Na
        afcn, dafcn = self.dafcn_drho(rhomolecular, T)
        mu = afcn + rhomolecular * dafcn
        
        return mu
    
    def dOm(self, roa, Tad, mu, Psat):

        a0 = self.a0ad(roa, Tad)
        GPT = a0 - roa*mu + Psat 
        
        return GPT  



    
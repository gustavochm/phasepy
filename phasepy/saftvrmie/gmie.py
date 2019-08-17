import numpy as np
from .gmie_aux import gdHS, dgdHS_drho, d2gdHS_drho
from .gmie_aux import g1sigma, dg1sigma_drho, d2g1sigma_drho
from .gmie_aux import g2sigma, dg2sigma_drho, d2g2sigma_drho


def gmie(x0, eta, 
      x0_a1, a1sb_a1, da1m_deta,
      x03, nsigma, alpha, tetha,
      x0_a2, a1sb_a2, da2m_new_deta, khs,
      d, deta_drho, rho, beta, eps, c, ms):
    
    be = beta * eps
    
    ghs = gdHS(x0, eta)
    g1 = g1sigma(x0_a1, a1sb_a1, da1m_deta,  eta, d, deta_drho, rho,eps, c, ms)
    g2 = g2sigma(x03, nsigma, alpha, tetha, 
                 x0_a2, a1sb_a2, da2m_new_deta, khs, eta,  d, deta_drho, rho,  eps, c, ms)
    
    gm = ghs*np.exp((be*g1+be**2*g2)/ghs)
    return gm

def lngmie(x0, eta, 
          x0_a1, a1sb_a1, da1m_deta,
          x03, nsigma, alpha, tetha,
          x0_a2, a1sb_a2, da2m_new_deta, khs,
          d, deta_drho, rho, beta, eps, c, ms):
    be = beta * eps
    ghs = gdHS(x0, eta)
    g1 = g1sigma(x0_a1, a1sb_a1, da1m_deta,  eta, d, deta_drho, rho, eps, c, ms)
    g2 = g2sigma(x03, nsigma, alpha, tetha, 
                 x0_a2, a1sb_a2, da2m_new_deta, khs, eta,  d, deta_drho, rho,  eps, c, ms)

    lng = np.log(ghs)  + (be*g1+be**2*g2)/ghs
    return lng

def dlngmie_drho(x0, eta, 
              x0_a1, a1sb_a1, da1m_deta,
              x03, nsigma, alpha, tetha,
              x0_a2, a1sb_a2, da2m_new_deta, khs,
              d, drho, rho, beta, eps, c, ms):
    be = beta * eps

    
    ghs, dghs = dgdHS_drho(x0, eta, drho)

    g1, dg1 = dg1sigma_drho(x0_a1, a1sb_a1, da1m_deta,  eta, d, drho, rho, eps, c, ms)
  
    g2, dg2 = dg2sigma_drho(x03, nsigma, alpha, tetha,
                            x0_a2, a1sb_a2, da2m_new_deta, khs, eta,  d, drho, rho,  eps, c, ms)
 
    lng = np.log(ghs)  + (be*g1+be**2*g2)/ghs
    
    dlng =  be * ghs *  (dg1 + be * dg2)
    dlng += dghs * (ghs - be * (g1 + be * g2))
    dlng /= ghs**2
    
    return np.hstack([lng, dlng])

def d2lngmie_drho(x0, eta, 
                  x0_a1, a1sb_a1, da1m_deta,
                  x03, nsigma, alpha, tetha,
                  x0_a2, a1sb_a2, da2m_new_deta, khs,
                  d, drho, rho, beta, eps, c, ms):
    
    be = beta * eps
    
    ghs, dghs, d2ghs = d2gdHS_drho(x0, eta, drho)
    g1, dg1, d2g1 = d2g1sigma_drho(x0_a1, a1sb_a1, da1m_deta, eta, d, drho, rho, eps, c, ms)
    g2, dg2, d2g2 = d2g2sigma_drho(x03, nsigma, alpha, tetha,
                                   x0_a2, a1sb_a2, da2m_new_deta, khs, eta,  d, drho, rho,  eps, c, ms)
    
    lng = np.log(ghs)  + (be*g1+be**2*g2)/ghs
    
    dlng =  be * ghs *  (dg1 + be * dg2)
    dlng += dghs * (ghs - be * (g1 + be * g2))
    dlng /= ghs**2
    
    d2lng =  2* be * dghs**2 *  (g1 + be * g2)
    d2lng += ghs**2* (d2ghs + be * (d2g1 + be * d2g2))
    d2lng += ghs * (-dghs *(2*be*(dg1 + be* dg2) + dghs) - be * (g1 + be *g2) * d2ghs)
    d2lng /= ghs**3
    
    return np.hstack([lng, dlng, d2lng])
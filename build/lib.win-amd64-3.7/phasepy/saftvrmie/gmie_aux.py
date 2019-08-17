import numpy as np

#Eq 47 a 50
def ki_chain(eta):
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta2**2
    eta_1 = (1 - eta)
    eta_13 = eta_1**3
    k0 = -np.log(eta_1) + (42*eta - 39*eta2 + 9*eta3 - 2*eta4)/(6*eta_13)
    k1 = (eta4 + 6*eta2 - 12*eta)/(2*eta_13)
    k2 = -3*eta2/(8*eta_1**2)
    k3 = (-eta4 + 3*eta2 + 3*eta)/(6*eta_13)
    return np.array([k0, k1, k2, k3])


#Eq 47 a 50
def dki_chain(eta):
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta2**2
    eta_1 = (1 - eta)
    eta_13 = eta_1**3
    k0 = -np.log(eta_1) + (42*eta - 39*eta2 + 9*eta3 - 2*eta4)/(6*eta_13)
    k1 = (eta4 + 6*eta2 - 12*eta)/(2*eta_13)
    k2 = -3*eta2/(8*eta_1**2)
    k3 = (-eta4 + 3*eta2 + 3*eta)/(6*eta_13)
    
    dk0 = (24 + eta*(-6+eta*(3-7*eta+eta**2)))/(eta-1)**4/3
    dk1 = - (12 +eta*(2+eta)*(6 -6*eta+eta2))/(eta-1)**4/2
    dk2 = 3*eta/(4*(-1+eta)**3)
    dk3 = (3+eta*(12+eta*(eta-3)*(eta-1)))/(6*(-1+eta)**4)
    return np.array([[k0, k1, k2, k3],
                     [dk0, dk1, dk2, dk3]])

#Eq 47 a 50
def d2ki_chain(eta):
    eta2 = eta**2
    eta3 = eta**3
    eta4 = eta2**2
    eta_1 = (1 - eta)
    eta_13 = eta_1**3
    k0 = -np.log(eta_1) + (42*eta - 39*eta2 + 9*eta3 - 2*eta4)/(6*eta_13)
    k1 = (eta4 + 6*eta2 - 12*eta)/(2*eta_13)
    k2 = -3*eta2/(8*eta_1**2)
    k3 = (-eta4 + 3*eta2 + 3*eta)/(6*eta_13)
    
    dk0 = (24 + eta*(-6+eta*(3-7*eta+eta**2)))/(eta-1)**4/3
    dk1 = - (12 +eta*(2+eta)*(6 -6*eta+eta2))/(eta-1)**4/2
    dk2 = 3*eta/(4*(-1+eta)**3)
    dk3 = (3+eta*(12+eta*(eta-3)*(eta-1)))/(6*(-1+eta)**4)
    
    d2k0 = (-30 + eta*(1+eta)*(4+eta))/(eta-1)**5
    d2k1 =  6*(5-2*eta*(eta-1))/(eta-1)**5
    d2k2 = -3*(1+2*eta)/(4*(eta-1)**4)
    d2k3 = (-4+eta*(eta-7))/((-1+eta)**5)
    return np.array([[k0, k1, k2, k3],
                     [dk0, dk1, dk2, dk3],
                     [d2k0, d2k1, d2k2, d2k3]]) 
    
    
#modificadas
#Eq 46
def gdHS(x0, eta):
    ks = ki_chain(eta)
    xs = np.array([1, x0, x0**2, x0**3])
    g = np.exp(np.dot(xs, ks))
    return g

def dgdHS_drho(x0, eta, deta_drho):
    dks = dki_chain(eta)
    xs = np.array([1., x0, x0**2, x0**3])
    dg = np.matmul(dks, xs)
    dg[0] = np.exp(dg[0])
    dg[1] *= dg[0]
    dg *= deta_drho[:2]
    return dg

def d2gdHS_drho(x0, eta, deta_drho):
    d2ks = d2ki_chain(eta)
    xs = np.array([1., x0, x0**2, x0**3])
    d2g = np.matmul(d2ks, xs)
    d2g[0] = np.exp(d2g[0])
    d2g[2] += d2g[1]**2
    d2g[2] *= d2g[0]
    d2g[1] *= d2g[0]
    d2g *= deta_drho[:3]
    return d2g

#Eq 64
def g1sigma(x0lambda, a1sb, da1m_deta,  eta, d, deta_drho, rho, eps, c, ms):
    
    ctes = 1/(2*np.pi*eps*ms*d**3)

    da1 = da1m_deta * deta_drho

    sum1 =  np.dot(a1sb, x0lambda)
    g1 = ctes * (3.*da1 - c*sum1/rho)
    
    return g1


def dg1sigma_drho(x0lambda, da1sb, d2a1m_deta,  eta, d, drho, rho, eps, c, ms):
    ctes = 1/(2*np.pi*eps*ms*d**3)
    '''
    a1, da1, d2a1 = d2a1m_deta * drho
    
    suma1 = np.matmul(da1sb,  np.hstack([lambda_a, lambda_r]) * x0lambda)
    suma1 *= drho[:2]
    sum1, dsum1 =  suma1
    
    g1 = ctes * (3.*da1 - c*sum1/rho)
    dg1 = ctes * (3.*d2a1 - c*dsum1/rho + c*sum1/rho**2)
    return g1, dg1
    '''
    
    suma1 = np.matmul(da1sb, x0lambda)
    suma1 *= drho[:2]
    suma1 *= c
    
    d2a1m_drho = d2a1m_deta * drho 
    dg1 = 3.*d2a1m_drho[1:] - suma1/rho
    dg1[1] += suma1[0]/rho**2 
    dg1 *= ctes
    
    return dg1

#drho = np.array([1, deta_drho, deta_drho**2,  deta_drho**3])

def d2g1sigma_drho(x0lambda, d2a1sb, d3a1m_deta,  eta, d, drho, rho, eps, c, ms):
    
    ctes = 1/(2*np.pi*eps*ms*d**3)
    '''
    a1, da1, d2a1, d3a1 = d3a1m_deta * drho
    
    suma1 = np.matmul(d2a1sb,  np.hstack([lambda_a, lambda_r]) * x0lambda)
    suma1 *= drho[:3]
    sum1, dsum1, d2sum1 = suma1
    
    
    
    g1 = ctes * (3.*da1 - c*sum1/rho)
    dg1 = ctes * (3.*d2a1 - c*dsum1/rho + c*sum1/rho**2)
    d2g1 = ctes * (3.*d3a1 - c*(2*sum1/rho**3 -2*dsum1/rho**2 + d2sum1/rho))
    g1, dg1, d2g1
    '''
    
    suma1 = np.matmul(d2a1sb,   x0lambda)
    suma1 *= drho[:3]
    suma1 *= c
    
    d3a1m_drho = d3a1m_deta * drho 
    d2g1 = 3*d3a1m_drho[1:] - suma1/rho
    d2g1[1] += suma1[0]/rho**2
    d2g1[2] += -2*suma1[0]/rho**3 + 2*suma1[1]/rho**2
    d2g1 *= ctes

    
    return d2g1


#Eq 65
def g2MCA(x0lambda, a1sb, da2m_new_deta, khs, eta,  d, deta_drho, rho, eps, c, ms):
    da2 = da2m_new_deta * deta_drho
    
    cte1 = (2*np.pi*eps**2*ms*d**3)
    cte2 = eps*c**2
    
    sum1 =  np.dot(a1sb,  x0lambda)

    g2 = (3.*da2 - cte2 * khs *sum1 / rho)/cte1
    return g2

def dg2MCA_drho(x0lambda, a1sb, d2a2m_new_deta, dKhs, eta,  d, drho, rho, eps, c, ms):
    
    dKhs *= drho[:2]
    khs, dkhs = dKhs

    d2a2 = d2a2m_new_deta * drho[1:]
    
    cte1 = (2*np.pi*eps**2*ms*d**3)
    cte2 = eps*c**2   
    
    suma1 = np.matmul(a1sb, x0lambda)
    suma1 *= drho[:2]
    suma1 *= cte2
    
    dg2 = 3*d2a2 - suma1 * khs/ rho
    dg2[1] +=  suma1[0] * khs/ rho**2 - suma1[0] * dkhs / rho
    dg2 /= cte1
    '''
    da2, d2a2 = d2a2m_new_deta * drho[1:]
    
    cte1 = (2*np.pi*eps**2*ms*d**3)
    cte2 = eps*c**2    
    
    suma1 = np.matmul(a1sb,  np.hstack([lambda_a, lambda_ar/2, lambda_r]) * x0lambda)
    suma1 *= drho[:2]
    sum1, dsum1 =  suma1

    g2 = (3.*da2 - cte2 * khs *sum1 / rho)/cte1
    dg2 = (3.*d2a2 + cte2 * (khs * sum1 /rho**2 - sum1 * dkhs / rho - khs * dsum1 /rho))/cte1
    return g2, dg2
    '''
    return dg2

def d2g2MCA_drho(x0lambda, a1sb, d3a2m_new_deta, d2Khs, eta,  d, drho, rho, eps, c, ms):
    
    d2Khs *= drho[:3]
    khs, dkhs, d2khs = d2Khs 

    d3a2 = d3a2m_new_deta * drho[1:]
    
    cte1 = (2*np.pi*eps**2*ms*d**3)
    cte2 = eps*c**2   
    
    suma1 = np.matmul(a1sb, x0lambda)
    suma1 *= drho[:3]
    suma1 *= cte2
    sum1, dsum1, d2sum1 =  suma1
    
    d2g2 = 3*d3a2 - suma1 * khs/ rho
    d2g2[1] +=  sum1 * khs/ rho**2 - sum1 * dkhs / rho
    d2g2[2] += -2*khs*sum1/rho**3 + 2*sum1*dkhs/rho**2 + 2*khs*dsum1/rho**2
    d2g2[2] += -2*dkhs*dsum1/rho - sum1*d2khs/rho
    d2g2 /= cte1
    
    '''
    da2, d2a2, d3a2  = d3a2m_new_deta * drho[1:]

    d2Khs *= drho[:3]
    khs, dkhs, d2khs = d2Khs 

    cte1 = (2*np.pi*eps**2*ms*d**3)
    cte2 = eps*c**2    
    
    suma1 = np.matmul(a1sb,  np.hstack([lambda_a, lambda_ar/2, lambda_r]) * x0lambda) 
    suma1 *= drho[:3]
    sum1, dsum1, d2sum1 =  suma1

    g2 = (3.*da2 - cte2 * khs *sum1 / rho)/cte1
    dg2 = (3.*d2a2 + cte2 * (khs * sum1 /rho**2 - sum1 * dkhs / rho - khs * dsum1 /rho))/cte1
    
    d2g2 = -2*khs*sum1/rho**3 + 2*sum1*dkhs/rho**2 + 2*khs*dsum1/rho**2
    d2g2 += -2*dkhs*dsum1/rho - sum1*d2khs/rho - khs *d2sum1 / rho
    d2g2 *= cte2
    d2g2 += 3.*d3a2
    d2g2 /= cte1 

    return g2, dg2, d2g2
    '''
    return d2g2

phi7 = np.array([10., 10., 0.57, -6.7, -8])

#Eq 63
def gammac(x0, nsigma, alpha, tetha):
    gc  = phi7[0]*(-np.tanh(phi7[1]*(phi7[2]-alpha))+1)
    gc *= nsigma*tetha*np.exp(phi7[3]*nsigma+ phi7[4]*nsigma**2)
    return gc

def dgammac_deta(x03, nsigma, alpha, tetha):

    cte = phi7[0]*(-np.tanh(phi7[1]*(phi7[2]-alpha))+1)*tetha
    g = cte * np.exp(phi7[3]*nsigma+ phi7[4]*nsigma**2)
    
    dg = np.full(2, g)
    dg[0] *=nsigma
    
    dg[1] *= (1. + nsigma*(phi7[3]+ 2*phi7[4]*nsigma)) 
    dg[1] *= x03 

    return dg

def d2gammac_deta(x03, nsigma, alpha, tetha):

    cte = phi7[0]*(-np.tanh(phi7[1]*(phi7[2]-alpha))+1)*tetha
    g = cte * np.exp(phi7[3]*nsigma+ phi7[4]*nsigma**2)
    
    dg = np.full(3, g)
    dg[0] *=nsigma
    
    dg[1] *= (1. + nsigma*(phi7[3]+ 2*phi7[4]*nsigma)) 
    dg[1] *= x03
    
    dg[2] *= (phi7[3]**2*nsigma + 2*phi7[4]*nsigma *(3+2*phi7[4]*nsigma**2) + phi7[3]*(2. + 4.*phi7[4]*nsigma**2))
    dg[2] *= x03**2
    

    return dg


def g2sigma(x03, nsigma, alpha, tetha, x0lambda, a1sb, da2m_new_deta, khs, eta,  d, deta_drho, rho,  eps, c, ms):
    gc = gammac(x03, nsigma, alpha, tetha)
    g2 = g2MCA(x0lambda, a1sb, da2m_new_deta, khs, eta,  d, deta_drho, rho,  eps, c, ms)
    return (1.+gc)*g2

def dg2sigma_drho(x03, nsigma, alpha, tetha, x0lambda, a1sb, d2a2m_new_deta, dKhs, eta,  d, drho, rho,  eps, c, ms):
    
    gc_eval = dgammac_deta(x03, nsigma, alpha, tetha)
    gc_eval *= drho[:2]
    gc, dgc = gc_eval
    '''
    g2mca, dg2mca = dg2MCA_drho(x0lambda, a1sb, d2a2m_new_deta, dKhs, eta, d, drho, rho)
    
    g2 = (1.+gc)*g2mca
    dg2 = dgc * g2mca + dg2mca * (1. + gc)
    return g2, dg2
    '''
    dg2mca = dg2MCA_drho(x0lambda, a1sb, d2a2m_new_deta, dKhs, eta, d, drho, rho,  eps, c, ms)
    dg2 = dg2mca * (1. + gc)
    dg2[1] += dgc * dg2mca[0]
    
    return dg2

def d2g2sigma_drho(x03, nsigma, alpha, tetha, x0lambda, a1sb, d3a2m_new_deta, d2Khs, eta,  d, drho, rho,  eps, c, ms):
    gc_eval = d2gammac_deta(x03, nsigma, alpha, tetha)
    gc_eval *= drho[:3]
    gc, dgc, d2gc = gc_eval

    '''
    g2mca, dg2mca, d2g2mca = d2g2MCA_drho(x0lambda, a1sb, d3a2m_new_deta, d2Khs, eta,  d, drho, rho)

    g2 = (1.+gc)*g2mca
    dg2 = dgc * g2mca + dg2mca * (1. + gc)
    d2g2 = d2gc * g2mca + 2*dgc*dg2mca + (1.+gc)*d2g2mca
    '''
    d2g2mca = d2g2MCA_drho(x0lambda, a1sb, d3a2m_new_deta, d2Khs, eta,  d, drho, rho,  eps, c, ms)
    d2g2 = d2g2mca * (1. + gc)
    d2g2[1] += dgc * d2g2mca[0]
    d2g2[2] += d2gc*d2g2mca[0] + 2.*dgc*d2g2mca[1]
    
    return d2g2
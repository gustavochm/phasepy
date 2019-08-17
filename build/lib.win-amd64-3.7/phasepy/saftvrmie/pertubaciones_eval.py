import numpy as np

#Perturbucacion 0 Eq 11
#Hard sphere
def ahs(eta):
    a = (4*eta - 3*eta**2)/(1-eta)**2
    return a

def dahs_deta(eta):
    a = (4*eta - 3*eta**2)/(1-eta)**2
    da = 2*(-2+eta)/(-1+eta)**3
    return np.array([a, da])

def d2ahs_deta(eta):
    a = (4*eta - 3*eta**2)/(1-eta)**2
    da = 2*(-2+eta)/(-1+eta)**3
    d2a = (10-4*eta)/(-1+eta)**4
    return np.array([a, da, d2a])


#Primera perturbacion Eq 34
def a1m(x0lambda , a1sb, c):
    #Se puede ocupar con las derivadas tambien ya que es la misma expresion.
    a1 = np.matmul(a1sb, x0lambda)
    #a1 = x0la * a1sb_a
    #a1 -= x0lr * a1sb_r
    a1 *= c
    return a1

#Segunda pertubacion Eq 36
def a2m(x0lambda, a1sb, khs, xi, c, eps):
        
    sum1 = np.dot(a1sb, x0lambda)

    a2 = khs*(1+xi)*eps*c**2*sum1/2.
    return a2

#Segunda pertubacion Eq 36
def da2m_deta(x0lambda, da1sb,  dKhs, dXi, c, eps):
    
    khs, dkhs = dKhs
    xi, dx1 = dXi
    x1 = 1.+ xi
    
    sum1, dsum1 = np.matmul(da1sb, x0lambda)

    a2 = khs*x1*eps*c**2*sum1/2.
    
    da2 = sum1*x1*dkhs + khs * x1 * dsum1 + khs * sum1 * dx1
    da2 *= (0.5*eps*c**2)
    return np.hstack([a2, da2])

def d2a2m_deta(x0lambda, d2a1sb,  d2Khs, d2Xi, c, eps):
    
    khs, dkhs, d2khs = d2Khs
    xi, dx1, d2x1 = d2Xi
    x1 = 1.+ xi
    
    sum1, dsum1, d2sum1 = np.matmul(d2a1sb, x0lambda)

    a2 = khs*x1*eps*c**2*sum1/2.
    
    da2 = sum1*x1*dkhs + khs * x1 * dsum1 + khs * sum1 * dx1
    da2 *= (0.5*eps*c**2)
    
    d2a2 = d2khs *sum1 * x1 + d2x1 * sum1 * khs + d2sum1 * khs * x1
    d2a2 += 2* dkhs * dsum1 *x1 
    d2a2 += 2* sum1 * dkhs * dx1
    d2a2 += 2* khs * dsum1 * dx1
    d2a2 *= (0.5*eps*c**2)
    return np.hstack([a2, da2, d2a2])



#Derivada segunda perturbacion
def da2m_new_deta(x0lambda, da1sb,  dKhs, c, eps):
    
    khs, dkhs = dKhs
    sum1, dsum1 = np.matmul(da1sb, x0lambda)
    da2 = 0.5*eps*c**2*(dkhs*sum1 + khs * dsum1)

    return da2

#Derivada segunda perturbacion
def d2a2m_new_deta(x0lambda, d2a1sb,  d2Khs, c, eps):
    
    khs, dkhs, d2khs = d2Khs
    sum1, dsum1, d2sum1 = np.matmul(d2a1sb, x0lambda)
    
    aux = 0.5*eps*c**2
    da2 = aux*(dkhs*sum1 + khs * dsum1)
    
    d2a2 = 2 * dkhs * dsum1 + sum1 * d2khs + d2sum1 * khs
    d2a2 *= aux
    
    return np.hstack([da2, d2a2])

#Derivada segunda perturbacion
def d3a2m_new_deta(x0lambda, d3a1sb,  d3Khs, c, eps):
    
    khs, dkhs, d2khs, d3khs = d3Khs
    sum1, dsum1, d2sum1, d3sum1 = np.matmul(d3a1sb, x0lambda)
    aux = 0.5*eps*c**2
    da2 = aux*(dkhs*sum1 + khs * dsum1)
    
    d2a2 = 2 * dkhs * dsum1 + sum1 * d2khs + d2sum1 * khs
    d2a2 *= aux

    d3a2 = 3 * dsum1 * d2khs + 3 *dkhs * d2sum1
    d3a2 += khs * d3sum1 + d3khs * sum1
    d3a2 *= aux
    return np.hstack([da2, d2a2, d3a2])


#Tercera perturbunacion Eq 19
def a3m(x03, nsigma, eps, f4, f5, f6):
    ter1 = -eps**3*f4*nsigma
    ter2 = np.exp(f5*nsigma + f6*nsigma**2)
    return ter1*ter2

def da3m_deta(x03, nsigma, eps, f4, f5, f6):

    
    aux = np.exp(f5*nsigma + f6*nsigma**2)
    a = -eps**3*f4*nsigma
    a *= aux
    
    da = -eps**3*f4*aux
    da *= (1. + nsigma * (f5 + 2*f6*nsigma))
    da *= x03
    return np.hstack([a, da])

def d2a3m_deta(x03, nsigma, eps, f4, f5, f6):

    aux = np.exp(f5*nsigma + f6*nsigma**2)
    a = -eps**3*f4*nsigma
    a *= aux
    
    da = -eps**3*f4*aux
    da *= (1 + nsigma * (f5 + 2*f6*nsigma))
    da *= x03
    
    d2a = -eps**3*f4*aux
    d2a *= (f5**2*nsigma + 2*f6*nsigma*(3 + 2*f6*nsigma**2) + f5*(2+4*f6*nsigma**2))
    d2a *= x03**2
    return np.hstack([a, da, d2a])



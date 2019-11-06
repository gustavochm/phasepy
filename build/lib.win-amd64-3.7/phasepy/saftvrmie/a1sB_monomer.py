import numpy as np
from .B_monomer import B, dB, d2B, d3B
from .a1s_monomer import a1s, da1s, d2a1s, d3a1s
'''
#Suma de a1s con B (utilizado para todas la primera y segunda perturbación)
def a1sB(x0, eta, lam, eps):
    b = B(x0, eta, lam, eps)
    a1 = a1s(eta, lam, eps)
    a1b = a1 + b
    return a1b


def da1sB(x0, eta, lam, eps):
    db = dB(x0, eta, lam, eps)
    da1 = da1s(eta, lam, eps)
    
    da1b = da1 + db
    
    return da1b


def d2a1sB(x0, eta, lam, eps):
    d2b = d2B(x0, eta, lam, eps)
    d2a1 = d2a1s(eta, lam, eps)
    
    d2a1b = d2a1 + d2b
    return d2a1b

def d3a1sB(x0, eta, lam, eps):
    d3b = d3B(x0, eta, lam, eps)
    d3a1 = d3a1s(eta, lam, eps)
    
    d3a1b = d3a1 + d3b
    return d3a1b
'''
############################
    
def a1sB(x0, eta, lam, cctes, eps):
    b = B(x0, eta, lam, eps)
    a1 = a1s(eta, lam, cctes, eps)
    a1b = a1 + b
    return a1b


def da1sB(x0, eta, lam, cctes, eps):
    db = dB(x0, eta, lam, eps)
    da1 = da1s(eta, lam,cctes, eps)
    
    da1b = da1 + db
    
    return da1b


def d2a1sB(x0, eta, lam, cctes, eps):
    d2b = d2B(x0, eta, lam, eps)
    d2a1 = d2a1s(eta, lam,cctes, eps)
    
    d2a1b = d2a1 + d2b
    return d2a1b

def d3a1sB(x0, eta, lam, cctes, eps):
    d3b = d3B(x0, eta, lam, eps)
    d3a1 = d3a1s(eta, lam, cctes, eps)
    
    d3a1b = d3a1 + d3b
    return d3a1b

###########################3
'''
#Function for creating arrays with a1s + B at all lambdas
def a1B_eval(x0, eta, lambda_a, lambda_r, lambda_ar, eps):
    a1sb_a = a1sB(x0, eta, lambda_a, eps)
    a1sb_r = a1sB(x0, eta, lambda_r, eps)
    a1sb_2a = a1sB(x0, eta, 2*lambda_a, eps)
    a1sb_2r = a1sB(x0, eta, 2*lambda_r, eps)
    a1sb_ar = a1sB(x0, eta, lambda_ar, eps)
    
    a1sb_a1 = np.hstack([a1sb_a, a1sb_r])
    a1sb_a2 = np.hstack([a1sb_2a, a1sb_ar, a1sb_2r])
    return a1sb_a1, a1sb_a2

#Function for creating arrays with a1s + B and it's first derivative
#at all lambdas
def da1B_eval(x0, eta, lambda_a, lambda_r, lambda_ar, eps):
    da1sb_a = da1sB(x0,eta, lambda_a, eps)
    da1sb_r = da1sB(x0, eta, lambda_r, eps)
    da1sb_2a = da1sB(x0, eta, 2*lambda_a, eps)
    da1sb_2r = da1sB(x0, eta, 2*lambda_r, eps)
    da1sb_ar = da1sB(x0, eta, lambda_ar, eps)
                                                 
    a1sb_a1 = np.column_stack([da1sb_a, da1sb_r])
    a1sb_a2 = np.column_stack([da1sb_2a, da1sb_ar, da1sb_2r])
    return a1sb_a1, a1sb_a2                                                       

#Function for creating arrays with a1s + B and it's first and second derivative
#at all lambdas
def d2a1B_eval(x0, eta, lambda_a, lambda_r, lambda_ar, eps):
    d2a1sb_a = d2a1sB(x0,eta, lambda_a, eps)
    d2a1sb_r = d2a1sB(x0, eta, lambda_r, eps)
    d2a1sb_2a = d2a1sB(x0, eta, 2*lambda_a, eps)
    d2a1sb_2r = d2a1sB(x0, eta, 2*lambda_r, eps)
    d2a1sb_ar = d2a1sB(x0, eta, lambda_ar, eps)
    
    a1sb_a1 = np.column_stack([d2a1sb_a, d2a1sb_r])
    a1sb_a2 = np.column_stack([d2a1sb_2a, d2a1sb_ar, d2a1sb_2r])
    
    return a1sb_a1, a1sb_a2  

#Function for creating arrays with a1s + B and it's first, second and third
# derivative at all lambdas
def d3a1B_eval(x0, eta, lambda_a, lambda_r, lambda_ar, eps):
    
    d3a1sb_a = d3a1sB(x0, eta, lambda_a, eps)
    d3a1sb_r = d3a1sB(x0, eta, lambda_r, eps)
    d3a1sb_2a = d3a1sB(x0, eta, 2*lambda_a, eps)
    d3a1sb_2r = d3a1sB(x0, eta, 2*lambda_r, eps)
    d3a1sb_ar = d3a1sB(x0, eta, lambda_ar, eps)
    
    a1sb_a1 = np.column_stack([d3a1sb_a, d3a1sb_r])
    a1sb_a2 = np.column_stack([d3a1sb_2a, d3a1sb_ar, d3a1sb_2r])
    
    return a1sb_a1, a1sb_a2  
 '''                               

#Function for creating arrays with a1s + B at all lambdas
def a1B_eval(x0, eta, lambda_a, lambda_r, lambda_ar, cctes, eps):
    
    cctes_la, cctes_lr, cctes_2la, cctes_2lr, cctes_lar = cctes
    
    a1sb_a = a1sB(x0, eta, lambda_a, cctes_la, eps)
    a1sb_r = a1sB(x0, eta, lambda_r, cctes_lr, eps)
    a1sb_2a = a1sB(x0, eta, 2*lambda_a, cctes_2la, eps)
    a1sb_2r = a1sB(x0, eta, 2*lambda_r, cctes_2lr, eps)
    a1sb_ar = a1sB(x0, eta, lambda_ar, cctes_lar,eps)
    
    a1sb_a1 = np.hstack([a1sb_a, a1sb_r])
    a1sb_a2 = np.hstack([a1sb_2a, a1sb_ar, a1sb_2r])
    return a1sb_a1, a1sb_a2

#Function for creating arrays with a1s + B and it's first derivative
#at all lambdas
def da1B_eval(x0, eta, lambda_a, lambda_r, lambda_ar, cctes, eps):
    
    cctes_la, cctes_lr, cctes_2la, cctes_2lr, cctes_lar = cctes
    
    da1sb_a = da1sB(x0,eta, lambda_a, cctes_la, eps)
    da1sb_r = da1sB(x0, eta, lambda_r, cctes_lr, eps)
    da1sb_2a = da1sB(x0, eta, 2*lambda_a, cctes_2la, eps)
    da1sb_2r = da1sB(x0, eta, 2*lambda_r, cctes_2lr, eps)
    da1sb_ar = da1sB(x0, eta, lambda_ar, cctes_lar, eps)
                                                 
    a1sb_a1 = np.column_stack([da1sb_a, da1sb_r])
    a1sb_a2 = np.column_stack([da1sb_2a, da1sb_ar, da1sb_2r])
    return a1sb_a1, a1sb_a2                                                       

#Function for creating arrays with a1s + B and it's first and second derivative
#at all lambdas
def d2a1B_eval(x0, eta, lambda_a, lambda_r, lambda_ar, cctes, eps):
    
    cctes_la, cctes_lr, cctes_2la, cctes_2lr, cctes_lar = cctes
    
    d2a1sb_a = d2a1sB(x0,eta, lambda_a, cctes_la, eps)
    d2a1sb_r = d2a1sB(x0, eta, lambda_r, cctes_lr, eps)
    d2a1sb_2a = d2a1sB(x0, eta, 2*lambda_a, cctes_2la, eps)
    d2a1sb_2r = d2a1sB(x0, eta, 2*lambda_r, cctes_2lr, eps)
    d2a1sb_ar = d2a1sB(x0, eta, lambda_ar, cctes_lar, eps)
    
    a1sb_a1 = np.column_stack([d2a1sb_a, d2a1sb_r])
    a1sb_a2 = np.column_stack([d2a1sb_2a, d2a1sb_ar, d2a1sb_2r])
    
    return a1sb_a1, a1sb_a2  

                                                                      

#Function for creating arrays with a1s + B and it's first, second and third
# derivative at all lambdas
def d3a1B_eval(x0, eta, lambda_a, lambda_r, lambda_ar, cctes, eps):
    
    cctes_la, cctes_lr, cctes_2la, cctes_2lr, cctes_lar = cctes
    
    d3a1sb_a = d3a1sB(x0, eta, lambda_a, cctes_la, eps)
    d3a1sb_r = d3a1sB(x0, eta, lambda_r, cctes_lr, eps)
    d3a1sb_2a = d3a1sB(x0, eta, 2*lambda_a, cctes_2la, eps)
    d3a1sb_2r = d3a1sB(x0, eta, 2*lambda_r, cctes_2lr, eps)
    d3a1sb_ar = d3a1sB(x0, eta, lambda_ar, cctes_lar, eps)
    
    a1sb_a1 = np.column_stack([d3a1sb_a, d3a1sb_r])
    a1sb_a2 = np.column_stack([d3a1sb_2a, d3a1sb_ar, d3a1sb_2r])
    
    return a1sb_a1, a1sb_a2  

###############3  
    
def x0lambda_eval(x0, lambda_a, lambda_r, lambda_ar):
    x0la = x0**lambda_a
    x0lr = x0**lambda_r
    x02la = x0**(2*lambda_a)
    x02lr = x0**(2*lambda_r)
    x0lar = x0**lambda_ar
    
    x0_a1 = np.hstack([x0la, -x0lr])
    x0_a2 = np.hstack([x02la, -2*x0lar, x02lr])
    
    x0_a12 = np.hstack([lambda_a * x0la, -lambda_r* x0lr])
    x0_a22 = np.hstack([lambda_a * x02la, -lambda_ar * x0lar, lambda_r * x02lr])
    return x0_a1, x0_a2, x0_a12, x0_a22    
                                                                                                        
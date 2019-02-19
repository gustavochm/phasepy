import numpy as np
from scipy.optimize import root
from ..math import lobatto

def fobj_saddle(ros, mu0, T, eos):
    mu = eos.muad(ros, T)
    return mu - mu0

def ten_linear(ro1, ro2, Tsat, Psat, model, n = 50):
    
    if (ro1 - ro2).sum() > 0:
        ro_aux = ro1.copy()
        ro1 = ro2.copy()
        ro2 = ro_aux
    
    #adimensionalizar variables 
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = ro1*rofactor
    ro2a = ro2*rofactor
    
    cij = model.ci(Tsat)
    cij /= cij[0,0]
    
    mu0 = model.muad(ro1a, Tsat)
    
    roots, weigths = lobatto(n)
    
    #perfiles lineales
    pend = (ro2a - ro1a)
    b = ro1a
    ro = (np.outer(roots, pend) + b).T

    dOm = np.zeros(n)
    for i in range(1, n-1):
        dOm[i] = model.dOm(ro[:,i], Tsat, mu0, Pad)
        
    integrer = np.nan_to_num(np.sqrt(2*dOm))
    integral = np.dot(integrer, weigths)

    u = ro2a - ro1a
    u /= np.linalg.norm(ro2a-ro1a)

    cijfactor = cij@u@u

    integral *= np.sqrt(cijfactor)
    ten = integral * tenfactor
    return ten
    
def ten_spot(ro1, ro2, Tsat, Psat, model, n = 100):
    
    if (ro1 - ro2).sum() > 0:
        ro_aux = ro1.copy()
        ro1 = ro2.copy()
        ro2 = ro_aux
    
    #adimensionalizar variables 
    Tfactor, Pfactor, rofactor, tenfactor, zfactor = model.sgt_adim(Tsat)
    Pad = Psat*Pfactor
    ro1a = ro1*rofactor
    ro2a = ro2*rofactor
    
    cij = model.ci(Tsat)
    cij /= cij[0,0]
    
    mu0 = model.muad(ro1a, Tsat)
    
    roots, weigths = lobatto(n)
    
    try:
        ros = (ro1a + ro2a)/2
        ros = root(fobj_saddle, ros, args =(mu0, Tsat, model), method = 'lm')
        if ros.success:
            ros = ros.x
            #segmento1
            #perfiles lineales
            pend = (ros - ro1a)
            b = ro1a
            ro1 = (np.outer(roots, pend) + b).T

            u1 = pend/np.linalg.norm(pend)
            cijfactor1 = cij@u1@u1

            #segmento2
            #perfiles lineales
            pend = (ro2a - ros)
            b = ros
            ro2 = (np.outer(roots, pend) + b).T

            u2 = pend/np.linalg.norm(pend)
            cijfactor2 = cij@u2@u2
            print(ros/rofactor)
            dOm1 = np.zeros(n)
            dOm2 = np.zeros(n)
            for i in range(1,n-1):
                dOm1[i] = model.dOm(ro1[:,i], Tsat, mu0, Pad)
                dOm2[i] = model.dOm(ro2[:,i], Tsat, mu0, Pad)

            integrer1 = np.nan_to_num(np.sqrt(2*dOm1))
            integral1 = np.dot(integrer1, weigths)/2
            ten1 = cijfactor1 * integral1

            integrer2 = np.nan_to_num(np.sqrt(2*dOm2))
            integral2 = np.dot(integrer2, weigths)/2 #el dos viene que se supone que ajusto a la mitad de z
            ten2 = cijfactor2 * integral2
            ten = ten1 + ten2
            ten *= tenfactor
        else:
            ten = ten_linear(ro1, ro2, Tsat, Psat, model, n)
    except:
        ten = ten_linear(ro1, ro2, Tsat, Psat, model, n)

    return ten
    
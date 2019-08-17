import numpy as np


def amono(ahs, a1m, a2m, a3m, beta, ms):
    am = ms * (ahs + beta * a1m + beta**2 * a2m + beta**3 * a3m)
    return am

def damono_drho(ahs, a1m, a2m, a3m, beta, drho, ms):
    #drho = np.array([1. , deta_drho])
    am = ms * drho *  (ahs + beta * a1m + beta**2 * a2m + beta**3 * a3m)
    return am

def d2amono_drho(ahs, a1m, a2m, a3m, beta, drho, ms):
    #drho = np.array([1. , deta_drho, deta_drho**2])
    am = ms * drho *  (ahs + beta * a1m + beta**2 * a2m + beta**3 * a3m)
    return am
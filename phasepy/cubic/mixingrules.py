from __future__ import division, print_function, absolute_import
import numpy as np
from .qmr import qmr
from .wongsandler import ws_nrtl, ws_wilson, ws_unifac, ws_rk
from .mhv import mhv_nrtl, mhv_wilson, mhv_nrtlt, mhv_unifac, mhv_rk


def mixingrule_fcn(self, mix, mixrule):
    if mixrule == 'qmr':

        self.mixrule = qmr
        self.secondorder = True
        self.secondordersgt = True

        if hasattr(mix, 'kij'):
            self.kij = mix.kij
            self.mixruleparameter = (mix.kij,)
        else:
            self.kij = np.zeros([self.nc, self.nc])
            self.mixruleparameter = (self.kij, )

        def mixrule_temp(self, T):
            mixrulep = (self.kij, )
            return mixrulep
        self.mixrule_temp = mixrule_temp.__get__(self)

    elif mixrule == 'mhv_nrtl':
        if hasattr(mix, 'g') and hasattr(mix, 'alpha'):
            self.nrtl = (mix.alpha, mix.g, mix.g1)
            self.mixruleparameter = (self.c1, self.c2,
                                     mix.alpha, mix.g, mix.g1)
            self.mixrule = mhv_nrtl
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                alpha, g, g1 = self.nrtl
                tau = g/T + g1
                G = np.exp(-alpha*tau)
                aux = (self.c1, self.c2, tau, G)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('NRTL parameters needed')

    elif mixrule == 'mhv_nrtlt':
        bool1 = hasattr(mix, 'g')
        bool2 = hasattr(mix, 'alpha')
        bool3 = hasattr(mix, 'rkternario')
        if bool1 and bool2 and bool3:
            self.nrtlt = (mix.alpha, mix.g, mix.g1, mix.rkternario)
            self.mixruleparameter = (self.c1, self.c2, mix.alpha, mix.g,
                                     mix.g1, mix.rkternario)
            self.mixrule = mhv_nrtlt
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                alpha, g, g1, rkternario = self.nrtlt
                tau = g/T + g1
                G = np.exp(-alpha*tau)
                aux = (self.c1, self.c2, tau, G, rkternario)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('NRTL/ternary parameters needed')

    elif mixrule == 'mhv_wilson':
        if hasattr(mix, 'Aij'):
            self.wilson = (mix.Aij, mix.vlrackett)
            self.mixruleparameter = (self.c1, self.c2, mix.Aij, mix.vlrackett)
            self.mixrule = mhv_wilson
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                Aij, vlrackett = self.wilson
                vl = vlrackett(T)
                M = np.divide.outer(vl, vl).T * np.exp(-Aij/T)
                aux = (self.c1, self.c2, M)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('Wilson parameters needed')

    elif mixrule == 'mhv_unifac':
        mix.unifac()
        if hasattr(mix, 'actmodelp'):
            self.unifac = mix.actmodelp
            self.mixruleparameter = (self.c1, self.c2, *mix.actmodelp)
            self.mixrule = mhv_unifac
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2 = self.unifac
                amn = a0 + a1 * T + a2 * T**2
                psi = np.exp(-amn/T)
                aux = (self.c1, self.c2, qi, ri, ri34, Vk, Qk, tethai,
                       amn, psi)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)

        else:
            raise Exception('Unifac parameters needed')

    elif mixrule == 'mhv_rk':

        self.mixrule = mhv_rk
        if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT'):
            self.rk = (mix.rkp, mix.rkpT, mix.combinatory)
            self.mixruleparameter = (self.c1, self.c2, mix.rkp, mix.rkpT,
                                     mix.combinatory)
            self.mixrule = mhv_rk
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                C, C1, combinatory = self.rk
                G = C + C1 / T
                aux = (self.c1, self.c2, G, combinatory)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('RK parameters needed')

    elif mixrule == 'ws_nrtl':
        if hasattr(mix, 'Kijws'):
            self.Kijws = mix.Kijws
        else:
            self.Kijws = np.zeros([self.nc, self.nc])

        if hasattr(mix, 'g') and hasattr(mix, 'alpha'):
            c1, c2 = self.c1, self.c2
            C = np.log((1+c1)/(1+c2))/(c1-c2)
            self.Cws = C
            self.nrtl = (mix.alpha, mix.g, mix.g1)
            self.mixruleparameter = (C, self.Kijws,
                                     mix.alpha, mix.g, mix.g1)
            self.mixrule = ws_nrtl
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                alpha, g, g1 = self.nrtl
                tau = g/T + g1
                G = np.exp(-alpha*tau)
                aux = (self.Cws, self.Kijws, tau, G)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('NRTL parameters needed')

    elif mixrule == 'ws_wilson':
        if hasattr(mix, 'Kijws'):
            self.Kijws = mix.Kijws
        else:
            self.Kijws = np.zeros([self.nc, self.nc])
        if hasattr(mix, 'Aij'):
            c1, c2 = self.c1, self.c2
            C = np.log((1+c1)/(1+c2))/(c1-c2)
            self.Cws = C
            self.wilson = (mix.Aij, mix.vlrackett)
            self.mixruleparameter = (C, self.Kijws, mix.Aij, mix.vlrackett)
            self.mixrule = ws_wilson
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                Aij, vlrackett = self.wilson
                vl = vlrackett(T)
                M = np.divide.outer(vl, vl).T * np.exp(-Aij/T)
                aux = (self.Cws, self.Kijws, M)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('Wilson parameters needed')

    elif mixrule == 'ws_rk':
        if hasattr(mix, 'Kijws'):
            self.Kijws = mix.Kijws
        else:
            self.Kijws = np.zeros([self.nc, self.nc])

        if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT'):
            c1, c2 = self.c1, self.c2
            C = np.log((1+c1)/(1+c2))/(c1-c2)
            self.Cws = C
            self.rk = (mix.rkp, mix.rkpT, mix.combinatory)
            self.mixruleparameter = (C, self.Kijws, mix.rkp, mix.rkpT,
                                     mix.combinatory)
            self.mixrule = ws_rk
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                C, C1, combinatory = self.rk
                G = C + C1 / T
                aux = (self.Cws, self.Kijws, G, combinatory)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('RK parameters needed')

    elif mixrule == 'ws_unifac':
        if hasattr(mix, 'Kijws'):
            self.Kijws = mix.Kijws
        else:
            self.Kijws = np.zeros([self.nc, self.nc])

        c1, c2 = self.c1, self.c2
        C = np.log((1+c1)/(1+c2))/(c1-c2)
        self.Cws = C
        mix.unifac()
        self.unifac = mix.actmodelp
        self.mixruleparameter = (C, self.Kijws, *self.unifac)
        self.mixrule = ws_unifac
        self.secondorder = True
        self.secondordersgt = True

        def mixrule_temp(self, T):
            qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2 = self.unifac
            amn = a0 + a1 * T + a2 * T**2
            psi = np.exp(-amn/T)
            aux = (self.Cws, self.Kijws, qi, ri, ri34, Vk, Qk, tethai,
                   amn, psi)
            return aux
        self.mixrule_temp = mixrule_temp.__get__(self)
    else:
        raise Exception('Mixrule not valid')

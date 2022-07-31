from __future__ import division, print_function, absolute_import
import numpy as np
from .qmr import qmr
from .wongsandler import ws_nrtl, ws_wilson, ws_unifac, ws_rk, ws_uniquac
from .wongsandler import ws_unifac_original
from .mhv import mhv_nrtl, mhv_wilson, mhv_nrtlt, mhv_unifac, mhv_rk
from .mhv import mhv_uniquac, mhv_unifac_original
from .mhv1 import mhv1_nrtl, mhv1_wilson, mhv1_nrtlt, mhv1_unifac, mhv1_rk
from .mhv1 import mhv1_uniquac, mhv1_unifac_original


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
    # MHV mixing rule
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
        bool3 = hasattr(mix, 'rkternary')
        if bool1 and bool2 and bool3:
            self.nrtlt = (mix.alpha, mix.g, mix.g1, mix.rkternary)
            self.mixruleparameter = (self.c1, self.c2, mix.alpha, mix.g,
                                     mix.g1, mix.rkternary)
            self.mixrule = mhv_nrtlt
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                alpha, g, g1, rkternary = self.nrtlt
                tau = g/T + g1
                G = np.exp(-alpha*tau)
                aux = (self.c1, self.c2, tau, G, rkternary)
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

    elif mixrule == 'mhv_original_unifac':
        mix.original_unifac()
        if hasattr(mix, 'actmodelp'):
            self.unifac = mix.actmodelp
            self.mixruleparameter = (self.c1, self.c2, *mix.actmodelp)
            self.mixrule = mhv_unifac_original
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                qi, ri, Vk, Qk, tethai, amn = self.unifac
                psi = np.exp(-amn/T)
                aux = (self.c1, self.c2, qi, ri, Vk, Qk, tethai, psi)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('Original-Unifac parameters needed')

    elif mixrule == 'mhv_uniquac':
        bool1 = hasattr(mix, 'ri')
        bool2 = hasattr(mix, 'qi')
        bool3 = hasattr(mix, 'a0') and hasattr(mix, 'a1')
        if bool1 and bool2 and bool3:
            self.uniquac = (mix.ri, mix.qi, mix.a0, mix.a1)
            self.mixruleparameter = (self.c1, self.c2, mix.ri, mix.qi,
                                     mix.a0, mix.a1)
            self.mixrule = mhv_uniquac
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                ri, qi, a0, a1 = self.uniquac
                Aij = a0 + a1 * T
                tau = np.exp(-Aij/T)
                aux = (self.c1, self.c2, ri, qi, tau)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('NRTL parameters needed')

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
    # Wong Sandler Mixing rule
    elif mixrule == 'ws_nrtl':
        if hasattr(mix, 'Kijws'):
            self.Kijws = mix.Kijws
        else:
            self.Kijws = np.zeros([self.nc, self.nc])

        if hasattr(mix, 'g') and hasattr(mix, 'alpha'):
            c1, c2 = self.c1, self.c2
            # C = np.log((1+c1)/(1+c2))/(c1-c2)
            C = - np.log((1+c1)/(1+c2))/(c1-c2)
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
            # C = np.log((1+c1)/(1+c2))/(c1-c2)
            C = - np.log((1+c1)/(1+c2))/(c1-c2)
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
            # C = np.log((1+c1)/(1+c2))/(c1-c2)
            C = - np.log((1+c1)/(1+c2))/(c1-c2)
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
        #C = np.log((1+c1)/(1+c2))/(c1-c2)
        C = - np.log((1+c1)/(1+c2))/(c1-c2)
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

    elif mixrule == 'ws_original_unifac':
        mix.original_unifac()
        if hasattr(mix, 'actmodelp'):
            if hasattr(mix, 'Kijws'):
                self.Kijws = mix.Kijws
            else:
                self.Kijws = np.zeros([self.nc, self.nc])

            c1, c2 = self.c1, self.c2
            #C = np.log((1+c1)/(1+c2))/(c1-c2)
            C = - np.log((1+c1)/(1+c2))/(c1-c2)
            self.Cws = C

            self.unifac = mix.actmodelp
            self.mixruleparameter = (C, self.Kijws, *self.unifac)
            self.mixrule = ws_unifac_original
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                qi, ri, Vk, Qk, tethai, amn = self.unifac
                psi = np.exp(-amn/T)
                aux = (self.Cws, self.Kijws, qi, ri, Vk, Qk, tethai, psi)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('Original-Unifac parameters needed')

    elif mixrule == 'ws_uniquac':
        if hasattr(mix, 'Kijws'):
            self.Kijws = mix.Kijws
        else:
            self.Kijws = np.zeros([self.nc, self.nc])

        bool1 = hasattr(mix, 'ri')
        bool2 = hasattr(mix, 'qi')
        bool3 = hasattr(mix, 'a0') and hasattr(mix, 'a1')
        if bool1 and bool2 and bool3:
            c1, c2 = self.c1, self.c2
            # C = np.log((1+c1)/(1+c2))/(c1-c2)
            C = - np.log((1+c1)/(1+c2))/(c1-c2)
            self.Cws = C
            self.uniquac = (mix.ri, mix.qi, mix.a0, mix.a1)
            self.mixruleparameter = (C, self.Kijws, mix.ri, mix.qi,
                                     mix.a0, mix.a1)
            self.mixrule = ws_uniquac
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                ri, qi, a0, a1 = self.uniquac
                Aij = a0 + a1 * T
                tau = np.exp(-Aij/T)
                aux = (self.Cws, self.Kijws, ri, qi, tau)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('UNIQUAC parameters needed')
    # MHV1 Mixing rule
    elif mixrule == 'mhv1_nrtl':
        if hasattr(mix, 'g') and hasattr(mix, 'alpha'):
            if self.c1 == 0. and self.c2 == 1.:
                q1 = -0.593
            elif self.c1 == 1. and self.c2 == 0.:
                q1 = -0.593
            elif self.c1 == 1-np.sqrt(2) and self.c2 == 1+np.sqrt(2):
                q1 = -0.53
            elif self.c2 == 1-np.sqrt(2) and self.c1 == 1+np.sqrt(2):
                q1 = -0.53
            else:
                raise Exception('Unkmown q1 value for MHV1 mixing-rule')
            self.q1 = q1
            self.nrtl = (mix.alpha, mix.g, mix.g1)
            self.mixruleparameter = (self.q1, mix.alpha, mix.g, mix.g1)
            self.mixrule = mhv1_nrtl
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                alpha, g, g1 = self.nrtl
                tau = g/T + g1
                G = np.exp(-alpha*tau)
                aux = (self.q1, tau, G)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('NRTL parameters needed')

    elif mixrule == 'mhv1_nrtlt':
        bool1 = hasattr(mix, 'g')
        bool2 = hasattr(mix, 'alpha')
        bool3 = hasattr(mix, 'rkternary')
        if bool1 and bool2 and bool3:
            if self.c1 == 0. and self.c2 == 1.:
                q1 = -0.593
            elif self.c1 == 1. and self.c2 == 0.:
                q1 = -0.593
            elif self.c1 == 1-np.sqrt(2) and self.c2 == 1+np.sqrt(2):
                q1 = -0.53
            elif self.c2 == 1-np.sqrt(2) and self.c1 == 1+np.sqrt(2):
                q1 = -0.53
            else:
                raise Exception('Unkmown q1 value for MHV1 mixing-rule')
            self.q1 = q1
            self.nrtlt = (mix.alpha, mix.g, mix.g1, mix.rkternary)
            self.mixruleparameter = (self.q1, mix.alpha, mix.g,
                                     mix.g1, mix.rkternary)
            self.mixrule = mhv1_nrtlt
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                alpha, g, g1, rkternary = self.nrtlt
                tau = g/T + g1
                G = np.exp(-alpha*tau)
                aux = (self.q1, tau, G, rkternary)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('NRTL/ternary parameters needed')

    elif mixrule == 'mhv1_wilson':
        if hasattr(mix, 'Aij'):
            if self.c1 == 0. and self.c2 == 1.:
                q1 = -0.593
            elif self.c1 == 1. and self.c2 == 0.:
                q1 = -0.593
            elif self.c1 == 1-np.sqrt(2) and self.c2 == 1+np.sqrt(2):
                q1 = -0.53
            elif self.c2 == 1-np.sqrt(2) and self.c1 == 1+np.sqrt(2):
                q1 = -0.53
            else:
                raise Exception('Unkmown q1 value for MHV1 mixing-rule')
            self.q1 = q1
            self.wilson = (mix.Aij, mix.vlrackett)
            self.mixruleparameter = (self.q1, mix.Aij, mix.vlrackett)
            self.mixrule = mhv1_wilson
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                Aij, vlrackett = self.wilson
                vl = vlrackett(T)
                M = np.divide.outer(vl, vl).T * np.exp(-Aij/T)
                aux = (self.q1, M)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('Wilson parameters needed')

    elif mixrule == 'mhv1_unifac':
        mix.unifac()
        if hasattr(mix, 'actmodelp'):
            if self.c1 == 0. and self.c2 == 1.:
                q1 = -0.593
            elif self.c1 == 1. and self.c2 == 0.:
                q1 = -0.593
            elif self.c1 == 1-np.sqrt(2) and self.c2 == 1+np.sqrt(2):
                q1 = -0.53
            elif self.c2 == 1-np.sqrt(2) and self.c1 == 1+np.sqrt(2):
                q1 = -0.53
            else:
                raise Exception('Unkmown q1 value for MHV1 mixing-rule')
            self.q1 = q1
            self.unifac = mix.actmodelp
            self.mixruleparameter = (self.q1, *mix.actmodelp)
            self.mixrule = mhv1_unifac
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                qi, ri, ri34, Vk, Qk, tethai, a0, a1, a2 = self.unifac
                amn = a0 + a1 * T + a2 * T**2
                psi = np.exp(-amn/T)
                aux = (self.q1, qi, ri, ri34, Vk, Qk, tethai,
                       amn, psi)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)

        else:
            raise Exception('Unifac parameters needed')

    elif mixrule == 'mhv1_original_unifac':
        mix.original_unifac()
        if hasattr(mix, 'actmodelp'):
            if self.c1 == 0. and self.c2 == 1.:
                q1 = -0.593
            elif self.c1 == 1. and self.c2 == 0.:
                q1 = -0.593
            elif self.c1 == 1-np.sqrt(2) and self.c2 == 1+np.sqrt(2):
                q1 = -0.53
            elif self.c2 == 1-np.sqrt(2) and self.c1 == 1+np.sqrt(2):
                q1 = -0.53
            else:
                raise Exception('Unkmown q1 value for MHV1 mixing-rule')

            self.q1 = q1
            self.unifac = mix.actmodelp
            self.mixruleparameter = (self.q1, *mix.actmodelp)
            self.mixrule = mhv1_unifac_original
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                qi, ri, Vk, Qk, tethai, amn = self.unifac
                psi = np.exp(-amn/T)
                aux = (self.q1, qi, ri, Vk, Qk, tethai, psi)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('Original-Unifac parameters needed')

    elif mixrule == 'mhv1_uniquac':
        bool1 = hasattr(mix, 'ri')
        bool2 = hasattr(mix, 'qi')
        bool3 = hasattr(mix, 'a0') and hasattr(mix, 'a1')
        if bool1 and bool2 and bool3:
            if self.c1 == 0. and self.c2 == 1.:
                q1 = -0.593
            elif self.c1 == 1. and self.c2 == 0.:
                q1 = -0.593
            elif self.c1 == 1-np.sqrt(2) and self.c2 == 1+np.sqrt(2):
                q1 = -0.53
            elif self.c2 == 1-np.sqrt(2) and self.c1 == 1+np.sqrt(2):
                q1 = -0.53
            else:
                raise Exception('Unkmown q1 value for MHV1 mixing-rule')
            self.q1 = q1
            self.uniquac = (mix.ri, mix.qi, mix.a0, mix.a1)
            self.mixruleparameter = (self.q1, mix.ri, mix.qi,
                                     mix.a0, mix.a1)
            self.mixrule = mhv1_uniquac
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                ri, qi, a0, a1 = self.uniquac
                Aij = a0 + a1 * T
                tau = np.exp(-Aij/T)
                aux = (self.q1, ri, qi, tau)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('NRTL parameters needed')

    elif mixrule == 'mhv1_rk':
        if hasattr(mix, 'rkp') and hasattr(mix, 'rkpT'):
            if self.c1 == 0. and self.c2 == 1.:
                q1 = -0.593
            elif self.c1 == 1. and self.c2 == 0.:
                q1 = -0.593
            elif self.c1 == 1-np.sqrt(2) and self.c2 == 1+np.sqrt(2):
                q1 = -0.53
            elif self.c2 == 1-np.sqrt(2) and self.c1 == 1+np.sqrt(2):
                q1 = -0.53
            else:
                raise Exception('Unkmown q1 value for MHV1 mixing-rule')
            self.q1 = q1
            self.rk = (mix.rkp, mix.rkpT, mix.combinatory)
            self.mixruleparameter = (self.q1, mix.rkp, mix.rkpT,
                                     mix.combinatory)
            self.mixrule = mhv1_rk
            self.secondorder = True
            self.secondordersgt = True

            def mixrule_temp(self, T):
                C, C1, combinatory = self.rk
                G = C + C1 / T
                aux = (self.q1, G, combinatory)
                return aux
            self.mixrule_temp = mixrule_temp.__get__(self)
        else:
            raise Exception('RK parameters needed')

    else:
        raise Exception('Mixrule not valid')

from __future__ import division, print_function, absolute_import
import numpy as np
from collections import Counter
from pandas import read_excel
import os
from copy import copy
from itertools import combinations
from .saft_forcefield import saft_forcefield
from .constants import kb, R


class component(object):
    '''
    Object class for storing pure component information.

    Parameters
    ----------
    name : str
        Name of the component
    Tc : float
        Critical temperature [K]
    Pc : float
        Critical pressure [bar]
    Zc : float
        Critical compressibility factor
    Vc : float
        Critical molar volume [:math:`\mathrm{cm^3/mol}`]
    w  : float
        Acentric factor
    c : float
        Volume translation parameter used in cubic EoS [:math:`\mathrm{cm^3/mol}`]
    cii : List[float]
        Polynomial coefficients for influence parameter used in SGT model
    ksv : List[float]
        Parameter for alpha for PRSV EoS
    Ant : List[float]
        Antoine correlation parameters
    GC : dict
        Group contribution information used in Modified-UNIFAC
        activity coefficient model. Group definitions can be found `here
        <http://www.ddbst.com/PublishedParametersUNIFACDO.html#ListOfMainGroups>`_.
    Mw : float
        molar weight of the fluid [g/mol]
    ri : float
        Component molecular volume for UNIQUAC model
    qi : float
        Component molecular surface for UNIQUAC model
    alpha_params : Any
        Parameters for alpha function used in cubic EoS
    dHf : float
        Enthalpy of fusion [J/mol]
    Tf : float
        Temperature of fusion [K]
    '''

    def __init__(self, name='None', Tc=0, Pc=0, Zc=0, Vc=0, w=0, c=0,
                 cii=0, ksv=[0, 0], Ant=[0, 0, 0],  GC=None, Mw=1.,
                 ri=0., qi=0., alpha_params=0.,
                 dHf=0., Tf=0.,
                 ms=1, sigma=0, eps=0, lambda_r=12.,
                 lambda_a=6., eAB=0., rcAB=1., rdAB=0.4, sites=[0, 0, 0]):

        self.name = name
        self.Tc = Tc  # Critical Temperature in K
        self.Pc = Pc  # Critical Pressure in bar
        self.Zc = Zc  # Critical compressibility factor
        self.Vc = Vc  # Critical volume in cm3/mol
        if Vc == 0 and Zc != 0:
            self.Vc = R*Zc*Tc/Pc
        elif Vc != 0 and Zc == 0:
            self.Zc = Pc*Vc/(R*Tc)
        self.w = w  # Acentric Factor
        self.Ant = Ant  # Antoine coefficeint, base e = 2.71
        self.cii = cii  # Influence factor SGT, list or array
        self.ksv = ksv
        self.c = c  # volume translation for cubic EoS
        self.GC = GC  # Dict, Group contribution info
        self.nc = 1
        self.Mw = Mw  # molar weight in g/mol
        self.ri = ri  # Component molecular volume for UNIQUAC model
        self.qi = qi  # Component molecular surface for UNIQUAC model
        self.alpha_params = alpha_params  # alpha parameters for cubic EoS
        self.dHf = dHf  # Enthalpy of fusion [J/mol]
        self.Tf = Tf  # Temperature of fusion [K]

        # Saft Parameters
        self.ms = ms
        self.sigma = sigma * 1e-10
        self.eps = eps * kb
        self.lambda_a = np.asarray(lambda_a)
        self.lambda_r = np.asarray(lambda_r)
        self.lambda_ar = self.lambda_r + self.lambda_a

        # Association Parameters
        self.eAB = eAB * kb
        self.rcAB = rcAB * self.sigma
        self.rdAB = rdAB * self.sigma
        self.sites = sites

    def __add__(self, component2):
        '''
        Methods to add two components and create a mixture with them.
        '''
        return mixture(self, component2)

    def psat(self, T):
        """
        Returns vapor saturation pressure [bar] at a given temperature
        using Antoine equation. Exponential base is :math:`e`.

        The following Antoine's equation is used:

        :math:`\ln (P /bar) = A - \frac{B}{T/K + C}`

        Parameters
        ----------
        T : float
            Absolute temperature [K]
        """

        coef = self.Ant
        return np.exp(coef[0]-coef[1]/(T+coef[2]))

    def tsat(self, P):
        """
        Returns vapor saturation temperature [K] at a given pressure using
        Antoine equation. Exponential base is :math:`e`.

        The following Antoine's equation is used:

        :math:`\ln (P /bar) = A - \frac{B}{T/K + C}`

        Parameters
        ----------
        P : float
            Saturation pressure [bar]
        """

        coef = self.Ant
        T = - coef[2] + coef[1] / (coef[0] - np.log(P))

        return T

    def vlrackett(self, T):
        """
        Returns liquid molar volume [:math:`\mathrm{cm^3/mol}`] at a given
        temperature using the Rackett equation.

        :math:`v = v_c Z_c^{(1 - T_r)^{2/7}}`

        Parameters
        ----------
        T : float
            Absolute temperature [K]
        """

        Tr = T/self.Tc
        V = self.Vc*self.Zc**((1-Tr)**(2/7))
        return V

    def ci(self, T):
        """
        Returns value of SGT model influence parameter
        [:math:`\mathrm{J m^5 / mol}`] at a given temperature.

        Parameters
        ----------
        T : float
            absolute temperature [K]
        """

        return np.polyval(self.cii, T)

    def saftvrmie(self, ms, rhol07):
        lambda_r, lambda_a, ms, eps, sigma = saft_forcefield(ms, self.Tc,
                                                             self.w, rhol07)
        self.lambda_a = np.asarray(lambda_a)
        self.lambda_r = np.asarray(lambda_r)
        self.lambda_ar = self.lambda_r + self.lambda_a
        self.ms = ms
        self.sigma = sigma
        self.eps = eps


class mixture(object):
    '''
    Object class for info about a mixture.

    Parameters
    ----------
    component1 : component
        First mixture component object
    component2 : component
        Second mixture component object

    Attributes
    ----------
    name : List[str]
        Names of the components
    Tc : List[float]
        Critical temperatures [K]
    Pc : List[float]
        Critical pressures [bar]
    Zc : List[float]
        critical compressibility factors
    Vc : List[float]
        Critical molar volumes [:math:`\mathrm{cm^3/mol}`]
    w  : List[float]
        Acentric factors
    c : List[float]
        Volume translation parameter used in cubic EoS [:math:`\mathrm{cm^3/mol}`]
    cii : List[list]
        Polynomial coefficients for influence parameter used in SGT model
    ksv : List[list]
        Parameters for alpha for PRSV EoS, if fitted
    Ant : List[list]
        Antoine correlation parameters
    GC : List[dict]
        Group contribution information used in Modified-UNIFAC
        activity coefficient model. Group definitions can be found `here
        <http://www.ddbst.com/PublishedParametersUNIFACDO.html#ListOfMainGroups>`_.
    Mw : list[dict]
        molar weights of the fluids in the mixture [g/mol]
    qi: list[dict]
        Component molecular surface used in UNIQUAC model
    ri: list[dict]
        Component molecular volume used in UNIQUAC model
    alpha_params: list[Any]
        parameters for custom cubic alpha functions
    dHf : list[float]
        Enthalpy of fusion [J/mol]
    Tf : list[float]
        Temperature of fusion [K]
    '''

    def __init__(self, component1, component2):
        self.names = [component1.name, component2.name]
        self.Tc = [component1.Tc, component2.Tc]
        self.Pc = [component1.Pc, component2.Pc]
        self.Zc = [component1.Zc, component2.Zc]
        self.w = [component1.w, component2.w]
        self.Ant = [component1.Ant, component2.Ant]
        self.Vc = [component1.Vc, component2.Vc]
        self.cii = [component1.cii, component2.cii]
        self.c = [component1.c, component2.c]
        self.ksv = [component1.ksv, component2.ksv]
        self.nc = 2
        self.GC = [component1.GC,  component2.GC]
        self.Mw = [component1.Mw,  component2.Mw]
        self.ri = [component1.ri, component2.ri]
        self.qi = [component1.qi, component2.qi]
        self.alpha_params = [component1.alpha_params, component2.alpha_params]
        self.dHf = [component1.dHf, component2.dHf]
        self.Tf = [component1.Tf, component2.Tf]

        self.lr = [component1.lambda_r, component2.lambda_r]
        self.la = [component1.lambda_a, component2.lambda_a]
        self.sigma = [component1.sigma, component2.sigma]
        self.eps = [component1.eps, component2.eps]
        self.ms = [component1.ms, component2.ms]
        self.eAB = [component1.eAB, component2.eAB]
        self.rc = [component1.rcAB, component2.rcAB]
        self.rd = [component1.rdAB, component2.rdAB]
        self.sitesmix = [component1.sites, component2.sites]

    def add_component(self, component):
        """
        Adds a component to the mixture
        """
        self.names.append(component.name)
        self.Tc.append(component.Tc)
        self.Pc.append(component.Pc)
        self.Zc.append(component.Zc)
        self.Vc.append(component.Vc)
        self.w.append(component.w)
        self.Ant.append(component.Ant)
        self.cii.append(component.cii)
        self.c.append(component.c)
        self.ksv.append(component.ksv)
        self.GC.append(component.GC)
        self.Mw.append(component.Mw)
        self.ri.append(component.ri)
        self.qi.append(component.qi)
        self.alpha_params.append(component.alpha_params)
        self.dHf.append(component.dHf)
        self.Tf.append(component.Tf)

        self.lr.append(component.lambda_r)
        self.la.append(component.lambda_a)
        self.sigma.append(component.sigma)
        self.eps.append(component.eps)
        self.ms.append(component.ms)
        self.eAB.append(component.eAB)
        self.rc.append(component.rcAB)
        self.rd.append(component.rdAB)
        self.sitesmix.append(component.sites)

        self.nc += 1

    def __add__(self, new_component):
        if isinstance(new_component, component):
            self.add_component(new_component)
        else:
            raise Exception('You can only add components objects to an existing mixture')
        return self

    def psat(self, T):
        """
        Returns array of vapour saturation pressures [bar] at a given
        temperature using Antoine equation. Exponential base is :math:`e`.

        The following Antoine's equation is used:

        :math:`\ln (P /bar) = A - \frac{B}{T/K + C}`

        Parameters
        ----------
        T : float
            Absolute temperature [K]

        Returns
        -------
        Psat : array_like
            saturation pressure of each component [bar]
        """

        coef = np.vstack(self.Ant)
        return np.exp(coef[:, 0]-coef[:, 1]/(T+coef[:, 2]))

    def tsat(self, P):
        """
        Returns array of vapour saturation temperatures [K] at a given pressure
        using Antoine equation. Exponential base is :math:`e`.

        The following Antoine's equation is used:

        :math:`\ln (P /bar) = A - \frac{B}{T/K + C}`

        Parameters
        ----------
        Psat : float
            Saturation pressure [bar]

        Returns
        -------
        Tsat : array_like
            saturation temperature of each component [K]
        """

        coef = np.vstack(self.Ant)
        T = - coef[:, 2] + coef[:, 1] / (coef[:, 0] - np.log(P))
        return T

    def vlrackett(self, T):
        """
        Returns array of liquid molar volumes [:math:`\mathrm{cm^3/mol}`] at a
        given temperature using the Rackett equation.

        :math:`v = v_c Z_c^{(1 - T_r)^{2/7}}`

        Parameters
        ----------
        T : float
            Absolute temperature [K]

        Returns
        -------
        vl : array_like
            liquid volume of each component [cm3 mol-1]
        """

        Tc = np.array(self.Tc)
        Vc = np.array(self.Vc)
        Zc = np.array(self.Zc)
        Tr = T/Tc
        V = Vc*Zc**((1-Tr)**(2/7))
        return V

    def kij_saft(self, kij):
        '''
        Adds kij binary interaction matrix for SAFT-VR-Mie to the
        mixture. Matrix must be symmetrical and the main diagonal must
        be zero.

        .. math::
           \epsilon_{ij} = (1-k_{ij}) \frac{\sqrt{\sigma_i^3 \sigma_j^3}}{\sigma_{ij}^3} \sqrt{\epsilon_i \epsilon_j}

        Parameters
        ----------
        kij: array_like
            Matrix of interaction parameters
        '''
        nc = self.nc
        KIJ = np.asarray(kij)
        shape = KIJ.shape

        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(KIJ, KIJ.T)

        if isSquare and isSymmetric:
            self.KIJsaft = kij
        else:
            raise Exception('kij matrix is not square or symmetric')

    def kij_ws(self, kij):
        '''
        Adds kij matrix coefficients for WS mixing rule to the
        mixture. Matrix must be symmetrical and the main diagonal must
        be zero.

        Parameters
        ----------
        kij: array_like
            Matrix of interaction parameters
        '''
        nc = self.nc
        KIJ = np.asarray(kij)
        shape = KIJ.shape

        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(KIJ, KIJ.T)

        if isSquare and isSymmetric:
            self.Kijws = kij
        else:
            raise Exception('kij matrix is not square or symmetric')

    def kij_cubic(self, kij, balanced=True):
        '''
        Adds kij matrix coefficients for QMR mixing rule to the
        mixture. Matrix must be symmetrical and the main diagonal must
        be zero.

        Parameters
        ----------
        kij: array_like
            Matrix of interaction parameters
        balanced: bool, optional
            If True, the kij matrix must be symmetrical
        '''
        nc = self.nc
        KIJ = np.asarray(kij)
        shape = KIJ.shape

        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(KIJ, KIJ.T)
        isDiagonalZero = np.all(np.diag(KIJ) == 0.)

        if isDiagonalZero:
            if isSquare and isSymmetric and balanced:
                self.kij = KIJ
            elif isSquare and not isSymmetric and not balanced:
                self.kij = KIJ
            else:
                raise Exception('kij matrix is not square or symmetric')
        else:
            raise Exception('kij matrix diagonal must be zero')


    def NRTL(self, alpha, g, g1=None):
        r'''
        Adds NRTL parameters to the mixture.

        Parameters
        ----------
        alpha: array
            Aleatory factor
        g: array
            Matrix of energy interactions [K]
        g1: array, optional
            Matrix of energy interactions [1/K]

        Note
        ----
        Parameters are evaluated as a function of temperature:
        :math:`\tau = g/T + g_1`
        '''
        nc = self.nc
        Alpha = np.asarray(alpha)
        shape = Alpha.shape

        isSquare = shape == (nc, nc)
        isSymmetric = np.allclose(Alpha, Alpha.T)

        if isSquare and isSymmetric:
            self.alpha = Alpha
        else:
            raise Exception('alpha matrix is not square or symmetric')

        self.g = g
        if g1 is None:
            g1 = np.zeros_like(g)
        self.g1 = g1
        self.actmodelp = (self.alpha, self.g, self.g1)

    def uniquac(self, a0, a1=None):
        r'''
        Adds UNIQUAC interaction energies to the mixture.

        Parameters
        ----------
        a0: array
            Matrix of energy interactions [K]
        a1: array, optional
            Matrix of energy interactions [Adim.]

        Note
        ----
        Parameters are evaluated as a function of temperature:
        :math:`a_{ij} = a_0 + a_1 T`
        '''
        ri = np.asarray(self.ri)
        qi = np.asarray(self.qi)

        self.ri = ri
        self.qi = qi

        bool1 = np.all(ri != 0)
        bool2 = np.all(qi != 0)
        if not bool1:
            raise Exception('A component molecular volume -ri- has not been set')
        if not bool2:
            raise Exception('A component molecular surface -qi- has not been set')

        a0 = np.asarray(a0)
        shape = a0.shape
        nc = self.nc
        isSquare = shape == (nc, nc)

        if isSquare:
            self.a0 = a0
        else:
            raise Exception('a0 matrix is not square (nc, nc)')

        if a1 is None:
            a1 = np.zeros_like(a0)
            self.a1 = a1
        else:
            a1 = np.asarray(a1)
            shape2 = a1.shape
            isSquare2 = shape2 == (nc, nc)
            if isSquare2:
                self.a1 = a1
            else:
                raise Exception('a1 matrix is not square (nc, nc)')

        self.actmodelp = (self.ri, self.qi, self.a0, self.a1)

    def rkt(self, D):
        '''
        Adds a ternary polynomial modification for NRTL model to the mixture.

        Parameters
        ----------
        D: array
            Ternary interaction parameter values
        '''

        self.rkternary = D
        self.actmodelp = (self.alpha, self.g, self.g1, self.rkternary)

    def wilson(self, A):
        '''
        Adds Wilson model coefficients to the mixture.
        Argument matrix main diagonal must be zero.

        Parameters
        ----------
        A: array
            Interaction parameter values [K]
        '''

        self.Aij = A
        self.actmodelp = (self.Aij, self.vlrackett)

    def rkb(self, c, c1=None):
        '''
        Adds binary Redlich Kister polynomial coefficients for
        excess Gibbs energy to the mixture.

        Parameters
        ----------
        c: array
            Polynomial values [Adim]
        c1: array, optional
            Polynomial values [K]

        Note
        ----
        Parameters are evaluated as a function of temperature:
        :math:`G = c + c_1/T`
        '''

        self.rkb = c
        if c1 is None:
            c1 = np.zeros_like(c)
        self.rkbT = c1
        self.actmodelp = (c, c1)

    def rk(self, c, c1=None):
        '''
        Adds Redlich Kister polynomial coefficients for
        excess Gibbs energy to the mixture.

        Parameters
        ----------
        c: array
            Polynomial values [Adim]
        c1: array, optional
            Polynomial values [K]

        Note
        ----
        Parameters are evaluated as a function of temperature:
        :math:`G = c + c_1/T`
        '''

        nc = self.nc
        combinatory = np.array(list(combinations(range(nc), 2)), dtype=int)
        self.combinatory = combinatory
        c = np.atleast_2d(c)
        self.rkp = c
        if c1 is None:
            c1 = np.zeros_like(c)
        c1 = np.atleast_2d(c1)
        self.rkpT = c1
        self.actmodelp = (c, c1, combinatory)

    def unifac(self):
        """
        Reads the Dortmund database for Modified-UNIFAC model
        to the mixture for calculation of activity coefficients.

         Group definitions can be found `here
        <http://www.ddbst.com/PublishedParametersUNIFACDO.html#ListOfMainGroups>`_.
        """

        # UNIFAC database reading
        database = os.path.join(os.path.dirname(__file__), 'database')
        database += '/dortmund-2021.xlsx'
        qkrk = read_excel(database, 'RkQk', index_col='subgroup', engine='openpyxl')
        a0 = read_excel(database, 'A0', index_col='GroupID', engine='openpyxl')
        a0.fillna(0, inplace=True)
        a1 = read_excel(database, 'A1', index_col='GroupID', engine='openpyxl')
        a1.fillna(0, inplace=True)
        a2 = read_excel(database, 'A2', index_col='GroupID', engine='openpyxl')
        a2.fillna(0, inplace=True)

        # Reading pure component and mixture group contribution info
        puregc = self.GC
        mix = Counter()
        for i in puregc:
            mix += Counter(i)

        subgroups = list(mix.keys())

        # Dicts created for each component
        vk = []
        dics = []
        for i in puregc:
            d = dict.fromkeys(subgroups, 0)
            d.update(i)
            dics.append(d)
            vk.append(list(d.values()))
        Vk = np.array(vk)

        groups = qkrk.loc[subgroups, 'MainGroupID'].values

        a = a0.loc[groups, groups].values
        b = a1.loc[groups, groups].values
        c = a2.loc[groups, groups].values

        # Reading info of present groups
        rq = qkrk.loc[subgroups, ['Rk', 'Qk']].values
        Qk = rq[:, 1]

        ri, qi = (Vk@rq).T
        ri34 = ri**(0.75)

        Xmi = (Vk.T/Vk.sum(axis=1)).T
        t = Xmi*Qk
        tethai = (t.T/t.sum(axis=1)).T

        self.actmodelp = (qi, ri, ri34, Vk, Qk, tethai, a, b, c)

    def original_unifac(self):
        """
        Reads database for Original-UNIFAC model
        to the mixture for calculation of activity coefficients.

        Group definitions can be found `here
        <http://www.ddbst.com/published-parameters-unifac.html#ListOfMainGroups>`_.
        """

        # UNIFAC database reading
        database = os.path.join(os.path.dirname(__file__), 'database')
        database += '/original-unifac.xlsx'
        qkrk = read_excel(database, 'RkQk', index_col='subgroup', engine='openpyxl')
        a0 = read_excel(database, 'A0', index_col='GroupID', engine='openpyxl')
        a0.fillna(0, inplace=True)

        # Reading pure component and mixture group contribution info
        puregc = self.GC
        mix = Counter()
        for i in puregc:
            mix += Counter(i)

        subgroups = list(mix.keys())

        # Dicts created for each component
        vk = []
        dics = []
        for i in puregc:
            d = dict.fromkeys(subgroups, 0)
            d.update(i)
            dics.append(d)
            vk.append(list(d.values()))
        Vk = np.array(vk)

        groups = qkrk.loc[subgroups, 'MainGroupID'].values

        amn = a0.loc[groups, groups].values

        # Reading info of present groups
        rq = qkrk.loc[subgroups, ['Rk', 'Qk']].values
        Qk = rq[:, 1]

        ri, qi = (Vk@rq).T

        Xmi = (Vk.T/Vk.sum(axis=1)).T
        t = Xmi*Qk
        tethai = (t.T/t.sum(axis=1)).T

        self.actmodelp = (qi, ri, Vk, Qk, tethai, amn)

    def ci(self, T):
        """
        Returns the matrix of cij interaction parameters for SGT model at
        a given temperature.

        Parameters
        ----------
        T : float
            Absolute temperature [K]
        """

        n = len(self.cii)
        ci = np.zeros(n)
        for i in range(n):
            ci[i] = np.polyval(self.cii[i], T)
        self.cij = np.sqrt(np.outer(ci, ci))
        return self.cij

    def copy(self):
        """
        Returns a copy of the mixture object
        """
        return copy(self)

"""
thermoPy.ajustes: ajuste parametros de ecuaciones de estado y modelos de 
                    coeficientes de activida con Python
=======================================================

Contenidos
-------------------------------------------------------

thermoPy.ajustes incluye ajuste calculo de equilibrios de fases fluidas,
utilizando como modelos ecuaciones viriales, ecuaciones 
cubicas de estado o modelos de coeficientes de actividad.


Los equilibrios que es posible ajustar son los siguientes:
    Equilibrio liquido - vapor 
    Equilibrio liquido - liquido


 

Funciones
---------
ajuste_ant : ajuste ecuacion de Anotine para presion de saturacion
fobj_alpha : funcion objetivo para ajustar parametros de funcion alpha de termino
             atractivo de ecuacion cubica de estado
ajuste_cii : ajusta parametros de ci utilizado en teoria del gradiente para prediccion de tension superficial

fobj_kij : funcion objetivo para optimizar parametro de interaccion binaria en una ecuacion cubica
fobj_nrtl : funcion objetivo parametros de modelo de actividad NRTL
fobj_wilson : funcion objetivo parametros de modelo de actividad de Wilson

fobj_ll_kij : funcion objetivo para ajustar parametro de interaccion binaria a ELL
fobj_ll_nrtl : funcion objetivo para ajustar modelo NRTL a ELL

fobj_nrtkt: funcion objetivo para ajustar aporte ternario a modelo NRTL
"""

__all__ = [s for s in dir() if not s.startswith('_')]

from .ajustesbinarios import *
from .ajustesternarios import *
from .ajustepsat import *
from .ajustecii import *
from .ajustemulticomponente import *
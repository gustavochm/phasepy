"""
thermoPy: paquete de equilibrios de fases con Python
=======================================================

Contenidos
-------------------------------------------------------

thermoPy incluye calculo de equilibrios de fases fluidas,
utilizando como modelos ecuaciones viriales, ecuaciones 
cubicas de estadoo o modelos de coeficientes de actividad.

Al utilizar ecuaciones cúbicas para mezclas multicomponentes,
es posible elegir entre reglado de mezclado clásica (qmr) y reglas
avanzadas al límite de presión cero (mhv)

Los equilibrios que es posible calcular son los siguientes:
    Equilibrio liquido - vapor 
    Equilibrio liquido - liquido
    Equilibrio liquido - liquido - vapor 
    
Este paquete también incluye un subpaquete para el cálculo de tensiones
interfasiales, mediante el uso de la Teoría del Gradiente de VdW.


Subpaquetes
--------------------------------------------------------
cubicas: Ecuaciones cúbicas, reglas de mezclado, funciones temino atractivo.

equilibrios: Incluye rutinas para calculo de puntos de rocio, burbuja, flash y ELL.

ajustes:  Incluye rutinas para la optimizacion de coeficientes binarios de interacción,
modelos de de coeficiente de actividad, y optimizacion de parametros de puros.

"""
__all__ = [s for s in dir() if not s.startswith('_')]

from .mixtures import *
from .cubic.cubic import *
from .actmodels import *
from . import equilibrium
from . import fit
from . import sgt
from .math import *
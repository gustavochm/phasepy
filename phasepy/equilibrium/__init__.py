"""
thermoPy.equilibrios: paquete de equilibrios de fases con Python
=======================================================

Contenidos
-------------------------------------------------------

thermoPy.quilibrios incluye calculo de equilibrios de fases fluidas,
utilizando como modelos ecuaciones viriales, ecuaciones 
cubicas de estado o modelos de coeficientes de actividad.

Al utilizar ecuaciones cúbicas para mezclas multicomponentes,
es posible elegir entre reglado de mezclado clásica (qmr) y reglas
avanzadas al límite de presión cero (mhv)

Los equilibrios que es posible calcular son los siguientes:
    Equilibrio liquido - vapor 
    Equilibrio liquido - liquido
    Equilibrio liquido - liquido - vapor 
    

Funciones
---------
bubbleTy : punto de burbuja P, x -> T, y
bubblePy : punto de burbuja T, x -> P, y
dewTx : punto de rocio P, y -> T, x
dewTy : punto de rocio T, y -> P, x
flash : flash isotermico isobarico z, T, P -> x,y,beta
ell : equilibrio liquido liquido z, T, P -> x, w, beta
ell_init : encuentra puntos de inicializacion de ell usando minimos de tpd 
multiflash : encuentra quilibrio de tres fases, 2 liquidos y 1 vapor
ellv_binary : equilibrio trifasico para mezcla binaria

tpd : funcion plano tangente de Michelsen adimensional
tpd_mim : encuentra el minimo de tpd respeto a punto inicial, incluye restriccoines variables
tpd_minimas : encuentra los minimos del tpd

psat : calculo de presion de saturacion de puro con ecuacion cubica

"""


__all__ = [s for s in dir() if not s.startswith('_')]

from .bubble import bubblePy, bubbleTy
from .dew import dewTx, dewTx
from .flash import flash
from .multiflash import multiflash
from .hazt import haz
from .hazb import hazb
from .stability import tpd_min, tpd_minimas, ell_init
from .ell import ell
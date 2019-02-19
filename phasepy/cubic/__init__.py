"""
thermoPy.cubicas: ecuaciones cubicas de estado con Python
=======================================================

Contenidos
-------------------------------------------------------

thermoPy.cubicas permite la creacion de ecuaciones cubicas de estado
con una programacion enfocada a objetos.
   

Funciones
---------
cubicapuros: clase base para la creacion de una cubica de un compuesto puro
prpuros : eos de PR para puro
prsvpuro: eos de PR con termino atractivo de Stryjec-Vera
rkpuros :  eos de RK

cubica : clase base para la creacion de una cubica para mezclas
preos : eos de PR para mezclas
prsveos :  eos de PR con termino atractivo de Stryjec-Vera para mezclas
rkeos : eos de RK para mezclas

alpha_soave : termino atractivo de Soave
alpha_sv : termino atractivo de Stryjec-Vera

qmr : regla de mezclado cuadratica
mhv : regla de mezclado avanzada con limite de presion cero
mhv_nrtl : regla de mezclado avanzada utilizando modelo NRTL
mhv_wilson : regla de mezclado avanzada utilizando modelo de Wilson 
"""

__all__ = [s for s in dir() if not s.startswith('_')]

from .cubic import *
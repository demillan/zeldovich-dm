# -*- coding: utf-8 -*-
"""
Implementación de Aproximación Zeldovich 

David E. Millán Calero 2018-11-26
"""

import numpy as np

# Parámetros iniciales
boxsize = 128
ngrid = 128
M = int(boxsize / ngrid)
l = np.arange(-M, M + 1)
m = np.arange(-M, M + 1)
n = np.arange(-M, M + 1)

# Leemos P(k) inicial
pkinit = np.loadtxt('PkTable.dat', skiprows = 5)

# Generamos la delta_k de cada uno de los M^3 modos

# Primero generamos las (2*M + 1)^3 posibles combinaciones de l, m, n
nlm_grid = np.array(np.meshgrid(l, m, n)).reshape(3, (2*M + 1)**3).T

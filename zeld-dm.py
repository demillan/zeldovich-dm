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

# Ahora calculamos theta_k y |delta_k|
tetha_k = 2*np.pi*np.random.uniform(0, 1, nlm_grid.shape[0])
norm_k = (2*np.pi)*(np.sum(nlm_grid**2, 1))/boxsize
norm_delta_k = np.sqrt(-np.interp(norm_k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1, nlm_grid.shape[0])))

delta_k = norm_delta_k * np.exp(tetha_k*1j)

# Calculamos desplazamiento
x = [1, 1, 1]
(1/(L**1.5))*np.sum((delta_k*1j)/(norm_k**2))*(np.exp(), 1)
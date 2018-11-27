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
lmn_grid = np.array(np.meshgrid(l, m, n)).reshape(3, (2*M + 1)**3).T
lmn_grid = np.delete(lmn_grid, int((2*M + 1)**3 / 2), axis = 0)

# Ahora calculamos theta_k y |delta_k|
tetha_k = 2*np.pi*np.random.uniform(0, 1, lmn_grid.shape[0])
norm_k = (2*np.pi)*(np.sum(lmn_grid**2, 1))/boxsize
norm_delta_k = np.sqrt(-np.interp(norm_k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1, lmn_grid.shape[0])))

delta_k = norm_delta_k * np.exp(tetha_k*1j)

# Calculamos desplazamiento
x = [1, 1, 1]

const1_L32 = 1/(boxsize**1.5)
const2pi_L = 2*np.pi/boxsize

N = 100000
r1 = (np.random.random(size=(N, 3)))*boxsize 

sumx = np.zeros(len(r1), np.complex)

for i in range(0, len(r1)):
    for l in range(-M, M+1):
        for m in range(-M, M+1):
            for n in range(-M, M+1):
                if (np.abs(l)+np.abs(m)+np.abs(n) != 0):
                    k2 = const2pi_L*(l**2 + m**2 + n**2)
                    norm_delta_klmn = np.sqrt(-np.interp(np.sqrt(k2), pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
                    tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
                    delta_klmn = norm_delta_klmn * np.exp(tetha_klmn*1j)
                    psum = const1_L32 * 1j * (delta_klmn / k2) * np.exp(1j * const2pi_L * np.dot([l, m, n], r1[i]))
                    sumx[i] += psum
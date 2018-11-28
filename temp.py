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
# lmn_grid = np.delete(lmn_grid, int((2*M + 1)**3 / 2), axis = 0)

# Ahora calculamos theta_k y |delta_k|

tetha_k = 2*np.pi*np.random.uniform(0, 1, lmn_grid.shape[0])
norm_k = (2*np.pi)*(np.sum(lmn_grid**2, 1))/boxsize
norm_delta_k = np.sqrt(-np.interp(norm_k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1, lmn_grid.shape[0])))

delta_k = norm_delta_k * np.exp(tetha_k*1j)

# Calculamos desplazamiento

const1_L32 = 1/(boxsize**1.5)
kmin = 2*np.pi/boxsize

#setup particles on a uniform grid
N = 256**3
x1 = np.linspace(0, boxsize, num=N)
x2 = np.linspace(0, boxsize, num=N)
x3 = np.linspace(0, boxsize, num=N)
x = np.vstack([x1,x2,x3])

sumx = np.zeros(N, np.complex)

for i in range(0, N):
    for j in range(0, len(lmn_grid)):
        psum = 0
        if ((lmn_grid[j][0] != 0) & (lmn_grid[j][1] != 0) & (lmn_grid[j][2] != 0)):
            k2 = kmin*(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
            norm_delta_klmn = np.sqrt(-np.interp(np.sqrt(k2), pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
            tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
            delta_klmn = norm_delta_klmn * np.exp(tetha_klmn*1j)
            psum = const1_L32 * 1j * (delta_klmn / k2) * np.exp(1j * kmin * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], x[:,i]))
        sumx[i] += psum


#%%

import numpy as np
from numpy import linalg as LA
from spatial_stats import getPk
import zeldovich as Z
import pylab as mp
from joblib import Parallel, delayed
import multiprocessing
from par_funcs import init_field_part

# Parámetros iniciales

num_cores = 10

redshift = 0
boxsize = 100
eps = 10 # resolution

const1_L32 = 1/(boxsize**1.5)
kmin = (2*np.pi)/boxsize
kmin2 = kmin**2

M = int(boxsize / eps)
l = np.arange(1, M + 1)
m = np.arange(-M, M + 1)
n = np.arange(-M, M + 1)

# Generamos la delta_k de cada uno de los M^3 modos
# Primero generamos las posibles combinaciones de l, m, n
lmn_grid = np.array(np.meshgrid(l, m, n)).reshape(3, (2*M + 1)**2*M).T

randnorm = np.random.random_sample(len(lmn_grid))
randtheta = np.random.random_sample(len(lmn_grid))

N3 = M**3

# Leemos P(k) inicial
pkinit = np.loadtxt('PkTable.dat', skiprows = 5)

#%%

# delta field generation
reps = 50
delta_init_all = np.zeros(shape = (N3, reps), dtype = np.float64)

for rep in range(0, reps):
    print("rep: ", rep)
    for j in range(0, len(lmn_grid)):
            k = kmin*np.sqrt(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
            norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(randnorm[j]))
            tetha_klmn = 2*np.pi*randtheta[j]
            delta_lmn = norm_delta * np.exp(tetha_klmn*1j)
            delta_part += 2*np.real(delta_lmn * np.exp(1j * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x[i], y[i], z[i]])))
    

#%%
j = 10
print(lmn_grid[j])

#%%
tetha_klmn = 2*np.pi*randtheta[j]
print(tetha_klmn)

#%%
k = kmin*np.sqrt(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
pk_interp = np.interp(k, pkinit[:,0], pkinit[:,1])
norm_delta = np.sqrt(-pk_interp*np.log(randnorm[j]))

print(k, pk_interp, norm_delta)

#%%
delta_lmn = norm_delta * np.exp(tetha_klmn*1j)
delta_lmn_norm2 = LA.norm(delta_lmn)**2
print(delta_lmn, delta_lmn_norm2)

#%%
reps = 100000
cum_norm = 0
for rep in range(0, reps):
    k = kmin*np.sqrt(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
    tetha_klmn = 2*np.pi*np.random.random()
    pk_interp = np.interp(k, pkinit[:,0], pkinit[:,1])
    norm_delta = np.sqrt(-pk_interp*np.log(np.random.random()))
    delta_lmn = norm_delta * np.exp(tetha_klmn*1j)
    delta_lmn_norm2 = LA.norm(delta_lmn)**2
    cum_norm += delta_lmn_norm2

print(pk_interp, cum_norm/reps)

#%%


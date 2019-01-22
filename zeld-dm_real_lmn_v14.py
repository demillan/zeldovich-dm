
#%%

import numpy as np
from numpy import linalg as LA
from spatial_stats import getPk
import pylab as mp

# omega_m = 0.307115
# omega_l = 0.692885
# omega_m=0.289796, omega_l=0.710204 # original
def growthfunc(a, omega_m=0.307115, omega_l=0.692885):
    da=a/10000.
    integral = 0.
    for i in range(10000):
        aa=(i+1)*da
        integral+=da/((aa*np.sqrt(omega_m/(aa**3)+omega_l))**3)
    return 5*omega_m/2*np.sqrt(omega_m/a**3+omega_l)*integral

# Parámetros iniciales

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

l2 = np.array(0)
m2 = np.arange(-M, M + 1)
n2 = np.arange(-M, M + 1)

# Generamos la delta_k de cada uno de los M^3 modos
# Primero generamos las posibles combinaciones de l, m, n
lmn_grid = np.array(np.meshgrid(l, m, n)).reshape(3, (2*M + 1)**2*M).T

# Añadimos término l = 0 y (l!=0 m!=0 n!=0)
l2 = [0]
m2 = np.append(np.arange(-M, 0), np.arange(1, M + 1))
n2 = np.append(np.arange(-M, 0), np.arange(1, M + 1))

lmn_grid2 = np.array(np.meshgrid(l2, m2, n2)).reshape(3, (2*M)**2*1).T
lmn_grid = np.append(lmn_grid, lmn_grid2, axis = 0)

N3 = M**3

# Leemos P(k) inicial
pkinit = np.loadtxt('PkTable.dat', skiprows = 5)

#%%

#setup particles on a uniform grid
sk = (M, M, M)
x1 = np.fromfunction(lambda x,y,z:x+0.5, sk).astype(np.float)
x2 = np.fromfunction(lambda x,y,z:y+0.5, sk).astype(np.float)
x3 = np.fromfunction(lambda x,y,z:z+0.5, sk).astype(np.float)
x1=eps*x1.flatten()
x2=eps*x2.flatten()
x3=eps*x3.flatten()

mp.scatter(x1, x2, s=0.1, marker = ".")

#%%

# delta_k generation
reps = 10
k_list = kmin*np.sqrt(lmn_grid[:,0]**2 + lmn_grid[:,1]**2 + lmn_grid[:,2]**2)
delta_k = np.zeros(shape = (len(lmn_grid), reps), dtype = np.complex)
delta_norm_k2 = np.zeros(shape = (len(lmn_grid), reps), dtype = np.float64)

for rep in range(0, reps):
    print("rep: ", rep)
    for j in range(0, len(lmn_grid)):
        k = kmin*np.sqrt(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
        norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(np.random.random()))
        tetha_klmn = 2*np.pi*np.random.random()
        delta_lmn = norm_delta * np.exp(tetha_klmn*1j)
        delta_k[j, rep] = norm_delta*np.exp(1j*tetha_klmn)
        delta_norm_k2[j, rep] = LA.norm(delta_k[j, rep])**2

#%%
mean_delta_norm_k2 = np.sum(delta_norm_k2, axis = 1) / reps

k_sort_ix = np.argsort(k_list, axis=0)

mp.figure(figsize=(10,10))
mp.loglog(pkinit[:,0], pkinit[:,1])
mp.loglog(k_list[k_sort_ix], mean_delta_norm_k2[k_sort_ix])
mp.ylim([10**-2, 10**5])
mp.xlabel('k [h/Mpc]')
mp.ylabel('P(k)')
mp.title('Power spectrum of the final density field')

#%%
mean_delta_k = np.sum(delta_k, axis = 1) / reps 
mean_delta_k[0]

#%%

mean_delta_k = np.sum(delta_k, axis = 1) / reps 

# delta field generation
reps = 1
delta_init_all = np.zeros(shape = (N3, reps), dtype = np.float64)

for rep in range(0, reps):
    print("rep: ", rep)
    delta_init = np.zeros(N3, dtype = np.float64)
    for i in range(0, N3):
        if (np.remainder(i, 100) == 0):
            print(i)
        for j in range(0, len(lmn_grid)):
            delta_init[i] += 2*np.real(delta_k[j, rep]* np.exp(1j * kmin * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[i], x2[i], x3[i]])))
    delta_init_all[:,rep] = delta_init * const1_L32

# delta_init_all = delta_init_all * reps
# delta_init_3d = delta_init_all.reshape(M, M, M)

#%%
# delta_init_0 = 0
# for j in range(0, len(lmn_grid)):
#     delta_init_0 += 2*np.real(delta_k[j, 0]* np.exp(1j * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[11], x2[11], x3[11]])))

#%%
from scipy.integrate import tplquad

# limits for x,y,z
l1 = np.float(0)
l2 = np.float(boxsize)

def inv_sf(x1,x2,x3,kmin,l,m,n):
    return delta_init_all*np.exp(-j*0.02*np.pi*np.dot([0,1,1], [x1,x2,x3]))

# delta_k_inv = tplquad(inv_sf, x, y, z, kmin, l, m, n, lambda x:   l1, lambda x:   l2,
#                                        lambda x,y: l1, lambda x,y: l2,
#                                        lambda x,y,z: l1, lambda x,y,z: l2)[0]

delta_k_inv = tplquad(inv_sf, lambda x1: l1, lambda x1: l2,
                              lambda x1,x2: l1, lambda x1,x2: l2,
                              lambda x1,x2,x3: l1, lambda x1,x2,x3: l2)[0]

#%%
delta_init_3d = np.mean(delta_init_all, axis = 1).reshape(M, M, M)
# delta_init_3d = delta_init_all[:,0].reshape(M, M, M)
k, pk = getPk(delta_init_3d, nkbins=40, boxsize=boxsize, deconvolve_cic=False, exp_smooth=0.0)

mp.loglog(pkinit[:,0], pkinit[:,1])
mp.loglog(k, pk)
mp.ylim([10**-2, 10**5])
mp.xlabel('k [h/Mpc]')
mp.ylabel('P(k)')
mp.title('Power spectrum of the final density field')
mp.figure()

#%%
delta_init_all[1:10,:]

#%%
np.mean(delta_init_all[1:10,:], axis = 0)

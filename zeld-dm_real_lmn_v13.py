
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

# Generamos la delta_k de cada uno de los M^3 modos
# Primero generamos las posibles combinaciones de l, m, n
lmn_grid = np.array(np.meshgrid(l, m, n)).reshape(3, (2*M + 1)**2*M).T

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

# delta field generation
reps = 100
delta_init_all = np.zeros(shape = (N3, reps), dtype = np.float64)
for rep in range(0, reps):
    print("rep: ", rep)
    delta_init = np.zeros(N3, dtype = np.float64)
    for i in range(0, N3):
        if (np.remainder(i, 100) == 0):
            print(i)
        for ll in range(1, M+1):
            for mm in range(-M, M+1):
                for nn in range(-M, M+1):
                    k = kmin*np.sqrt(ll**2 + mm**2 + nn**2)
                    norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
                    tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
                    delta_init[i] += 2*np.real(norm_delta*np.exp(1j*tetha_klmn)*np.exp(1j*kmin*np.dot([ll, mm, nn], [x1[i], x2[i], x3[i]])))
    delta_init_all[:,rep] = delta_init * const1_L32

# delta_init_all = delta_init_all * reps
# delta_init_3d = delta_init_all.reshape(M, M, M)

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

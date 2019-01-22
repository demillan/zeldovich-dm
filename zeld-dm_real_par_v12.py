
#%%

import numpy as np
from numpy import linalg as LA
from spatial_stats import getPk
import zeldovich as Z
import pylab as mp
from joblib import Parallel, delayed
import multiprocessing
from par_funcs import init_field_part

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

num_cores = 4

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

# randnorm = np.random.random_sample(len(lmn_grid))
# randtheta = np.random.random_sample(len(lmn_grid))

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
reps = 1
delta_init_all = np.zeros(shape = (N3, reps), dtype = np.float64)

for rep in range(0, reps):
    print("rep: ", rep)
    randnorm = np.random.random(len(lmn_grid))
    randtheta = np.random.random(len(lmn_grid))
    delta_init = Parallel(n_jobs=num_cores)(delayed(init_field_part)(kmin, lmn_grid, pkinit, x1, x2, x3, i, randnorm, randtheta) for i in range(0, N3))
    delta_init_all[:,rep] = np.array(delta_init) * const1_L32

#%%
fft = np.fft.rfftn(delta_init_all)

#%%
# Otro AZ
#setup particles on a uniform grid
ngrid = M
sk = (ngrid, ngrid, ngrid)
a0 = np.fromfunction(lambda x,y,z:x+0.5, sk).astype(np.float)
b0 = np.fromfunction(lambda x,y,z:y+0.5, sk).astype(np.float)
c0 = np.fromfunction(lambda x,y,z:z+0.5, sk).astype(np.float)
a0=eps*a0.flatten()
b0=eps*b0.flatten()
c0=eps*c0.flatten()

dens0 = np.zeros(shape = (M, M, M, reps), dtype = np.float64)
iseed = int(np.random.randint(20190109, size=1))
for rep in range(0, reps):
    dens0[:,:,:,rep] = Z.make_gauss_init(pkinit, boxsize=boxsize, ngrid=ngrid, seed=int(np.random.randint(iseed, size=1)), exactpk=True, smw=2.)

#%%
mp.pcolormesh(np.linspace(0, boxsize, M), np.linspace(0, boxsize, M), dens0[:,:,1,0], cmap='rainbow')
mp.title("Previous AZ implementation")
mp.colorbar()
mp.show()

#%%
delta_init_3d = delta_init_all[:,21].reshape(M, M, M)
k, pk = getPk(delta_init_3d, nkbins=20, boxsize=boxsize, deconvolve_cic=False, exp_smooth=0.0)

mp.loglog(pkinit[:,0], pkinit[:,1])
mp.loglog(k, pk)
mp.xlim([10**-2, 10**1])
mp.ylim([10**1, 10**5])
mp.xlabel('k [h/Mpc]')
mp.ylabel('P(k)')
mp.title('Power spectrum of the final density field')
mp.figure()

#%%
k0, pk0 = getPk(dens0[:,:,:,4], nkbins=20, boxsize=boxsize, deconvolve_cic=False, exp_smooth=0.0)

mp.loglog(pkinit[:,0], pkinit[:,1])
mp.loglog(k, pk)
mp.loglog(k0, pk0)
mp.xlim([10**-2, 10**1])
mp.ylim([10**1, 10**5])
mp.xlabel('k [h/Mpc]')
mp.ylabel('P(k)')
mp.title('Power spectrum of the final density field')
mp.figure()

#%%
cumk = np.zeros(13, dtype = np.float64)
cumpk = np.zeros(13, dtype = np.float64)
cumk0 = np.zeros(13, dtype = np.float64)
cumpk0 = np.zeros(13, dtype = np.float64)

for rep in range(0, reps):
    delta_init_3d = delta_init_all[:, rep].reshape(M, M, M)
    k, pk = getPk(delta_init_3d, nkbins=20, boxsize=boxsize, deconvolve_cic=False, exp_smooth=0.0)
    k0, pk0 = getPk(dens0[:,:,:,rep], nkbins=20, boxsize=boxsize, deconvolve_cic=False, exp_smooth=0.0)
    cumk += k
    cumpk += pk
    cumk0 += k0
    cumpk0 += pk0

meank = cumk / reps
meanpk = cumpk / reps
meank0 = cumk0 / reps
meanpk0 = cumpk0 / reps

mp.loglog(pkinit[:,0], pkinit[:,1])
mp.loglog(meank, meanpk)
mp.loglog(meank0, meanpk0)
mp.xlim([10**-2, 10**1])
mp.ylim([10**1, 10**5])
mp.xlabel('k [h/Mpc]')
mp.ylabel('P(k)')
mp.title('Mean power spectrum of the final density field')
mp.figure()

#%%
np.random.randint(20190110, size=1)

import numpy as np
from numpy import linalg as LA
from spatial_stats import getPk
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

# num_cores = multiprocessing.cpu_count()
num_cores = 10 

redshift = 0
boxsize = 256
eps = 2 # resolution

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

#setup particles on a uniform grid
sk = (M, M, M)
x1 = np.fromfunction(lambda x,y,z:x+0.5, sk).astype(np.float)
x2 = np.fromfunction(lambda x,y,z:y+0.5, sk).astype(np.float)
x3 = np.fromfunction(lambda x,y,z:z+0.5, sk).astype(np.float)
x1=eps*x1.flatten()
x2=eps*x2.flatten()
x3=eps*x3.flatten()

# delta field generation
reps = 100
delta_init_all = np.zeros(shape = (N3, reps), dtype = np.float64)

for rep in range(0, reps):
    print("rep: ", rep)
    randnorm = np.random.random(len(lmn_grid))
    randtheta = np.random.random(len(lmn_grid))
    delta_init = Parallel(n_jobs=num_cores)(delayed(init_field_part)(kmin, lmn_grid, pkinit, x1, x2, x3, i, randnorm, randtheta) for i in range(0, N3))
    delta_init_all[:,rep] = np.array(delta_init) * const1_L32


np.savetxt("delta_init_all.txt", delta_init_all)
np.save("delta_init_all", delta_init_all)
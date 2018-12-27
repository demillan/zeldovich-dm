import numpy as np
from numpy import linalg as LA
import pylab as mp
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from par_funcs import init_field, disp_func

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
num_cores = multiprocessing.cpu_count()

redshift = 0
boxsize = 100
eps = 5 # resolution

M = int(boxsize / eps)
l = np.arange(-M, M + 1)
m = np.arange(-M, M + 1)
n = np.arange(-M, M + 1)

# Leemos P(k) inicial
pkinit = np.loadtxt('PkTable.dat', skiprows = 5)

# Generamos la delta_k de cada uno de los M^3 modos
# Primero generamos las (2*M + 1)^3 posibles combinaciones de l, m, n
lmn_grid = np.array(np.meshgrid(l, m, n)).reshape(3, (2*M + 1)**3).T

# Calculamos desplazamiento

const1_L32 = 1/(boxsize**1.5)
kmin = (2*np.pi)/boxsize
kmin2 = kmin**2

#setup particles on a uniform grid
N3 = M**3

#setup particles on a uniform grid
sk = (M, M, M)
x1 = np.fromfunction(lambda x,y,z:x+0.5, sk).astype(np.float)
x2 = np.fromfunction(lambda x,y,z:y+0.5, sk).astype(np.float)
x3 = np.fromfunction(lambda x,y,z:z+0.5, sk).astype(np.float)
x1=eps*x1.flatten()
x2=eps*x2.flatten()
x3=eps*x3.flatten()

mp.scatter(x1, x2, s=0.1, marker = ".")
mp.show()

# delta field generation
reps = 100
tetha_klmn_sum = np.zeros(shape=(len(lmn_grid)), dtype = np.complex)
delta_sum = np.zeros(shape=(len(lmn_grid)), dtype = np.complex)

#k = kmin*np.sqrt(lmn_grid[:,0]**2 + lmn_grid[:,1]**2 + lmn_grid[:,2]**2)
#k2 = k*k
#norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
#tetha_klmn = 2*np.pi*np.random.uniform(0, 1, size = len(k))
#delta = norm_delta * np.exp(tetha_klmn*1j)

for i in tqdm(range(0, reps)):
    k = kmin*np.sqrt(lmn_grid[:,0]**2 + lmn_grid[:,1]**2 + lmn_grid[:,2]**2)
    k2 = k*k
    norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
    tetha_klmn = 2*np.pi*np.random.uniform(0, 1, size = len(k))
    delta = norm_delta * np.exp(tetha_klmn*1j)
    tetha_klmn_sum+=tetha_klmn
    delta_sum+=delta

norm_delta_klmn = tetha_klmn_sum/reps
delta_field_klmn = delta_sum/reps

dx = np.zeros(shape=(N3, 3), dtype = np.complex)
# lambdas = np.zeros(shape=(N3, 3), dtype = np.complex)
for i in tqdm(range(0, N3)):
    for j in range(0, len(lmn_grid)):
        if ((lmn_grid[j][0] == 0) & (lmn_grid[j][1] == 0) & (lmn_grid[j][2] == 0)):
            dx[i] += 0
        else:
            dsum = kmin * 1j * (delta_field_klmn[j] / k2[j]) * np.exp(1j * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[i], x2[i], x3[i]]))*np.array([lmn_grid[j][0].astype(np.float), lmn_grid[j][1].astype(np.float), lmn_grid[j][2].astype(np.float)])
            dx[i] += const1_L32 * dsum
            # tensor
            # T = np.zeros((3,3), dtype = np.complex)
            # for ii in range(0,3):
            #     for jj in range(0,3):
            #         T[ii,jj] = (kmin**2)*kmin * 1j * (delta_field_klmn[j] / k2) * np.exp(1j * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[i], x2[i], x3[i]]))*lmn_grid[j][ii]*lmn_grid[j][jj]
            # lambdas[i] = LA.eigvalsh(T)

#dx = np.zeros(shape=(N3, 3), dtype = np.complex)
#for i in tqdm(range(0, N3)):
#    res = Parallel(n_jobs=num_cores)(delayed(disp_func)(i, j, x1, x2, x3, kmin, delta_field_klmn, lmn_grid, k2, const1_L32) for j in range(0, len(lmn_grid)))
#    dx[i] += np.sum(res)
            
fx = dx[:,0]
fy = dx[:,1]
fz = dx[:,2]

#displacements, scaled by the growth function at the redshift we want
d1=growthfunc(1./(1+redshift))/growthfunc(1.)
xdisp=fx.real*d1
ydisp=fy.real*d1
zdisp=fz.real*d1

mp.hist(fx.real*d1, bins = 100)
mp.show()
mp.hist(fy.real*d1, bins = 100)
mp.show()
mp.hist(fz.real*d1, bins = 100)
mp.show()
    
#displace particles from the grid
x1+=xdisp
x2+=ydisp
x3+=zdisp
    
#periodic boundary conditions
x1[np.where(x1<0)]+=boxsize
x1[np.where(x1>boxsize)]-=boxsize
x2[np.where(x2<0)]+=boxsize
x2[np.where(x2>boxsize)]-=boxsize
x3[np.where(x3<0)]+=boxsize
x3[np.where(x3>boxsize)]-=boxsize

mp.scatter(x3, x2, s=0.1, marker = ".")
mp.show()

lambda1 = lambdas[:,0].real
lambda2 = lambdas[:,1].real
lambda3 = lambdas[:,2].real

deltanl = 1/((1-lambda1)*(1-lambda2)*(1-lambda3))

deltanl_final = deltanl.reshape(int(N), int(N), int(N))

mp.pcolormesh(np.linspace(0, boxsize, N), np.linspace(0, boxsize, N), np.arcsinh(deltanl_final[:,:,0]), cmap='rainbow')
mp.colorbar()
mp.xlabel('x [Mpc/h]')
mp.ylabel('y [Mpc/h]')
mp.show()


N = int(10)
p = np.random.random((N, 3))

p[..., None] * p[:, None, :]

x = np.column_stack((x1,x2,x3))

x[..., None] * lmn_grid[:, None, :]
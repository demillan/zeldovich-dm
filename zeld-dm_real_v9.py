import numpy as np
from numpy import linalg as LA
import pylab as mp
import zeldovich as Z
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
boxsize = 200
eps = 10 # resolution
f = 0

M = int(boxsize / eps)
l = np.arange(1, M + 1)
m = np.arange(-M, M + 1)
n = np.arange(-M, M + 1)

# Generamos la delta_k de cada uno de los M^3 modos
# Primero generamos las posibles combinaciones de l, m, n
lmn_grid = np.array(np.meshgrid(l, m, n)).reshape(3, (2*M + 1)**2*M).T

# Leemos P(k) inicial
pkinit = np.loadtxt('PkTable.dat', skiprows = 5)
mp.loglog(pkinit[:,0], pkinit[:,1])
mp.ylim([10**-2, 10**5])
mp.xlabel('k [h/Mpc]')
mp.ylabel('P(k)')
mp.show()

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

mp.scatter(a0, b0, s=0.1, marker = ".")
mp.show()

dens0 = Z.make_gauss_init(pkinit, boxsize=boxsize, ngrid=ngrid, seed=20181217, exactpk=True, smw=2.)
fx0, fy0, fz0 = Z.get_disp(dens0, boxsize=boxsize, ngrid=ngrid)
x0, y0, z0 = Z.get_pos(fx0, fy0, fz0*(1+f), redshift, boxsize=boxsize, ngrid=ngrid) 

mp.hist(fx0.flatten(), bins = 100, color = "red")
mp.show()
mp.hist(fy0.flatten(), bins = 100, color = "red")
mp.show()
mp.hist(fz0.flatten(), bins = 100, color = "red")
mp.show()

mp.scatter(x0, y0, s=0.1, marker = ".", color = "red")
mp.show()

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

deltax = np.zeros(N3, dtype = np.float64)

for i in tqdm(range(0, N3)):
    for j in range(0, len(lmn_grid)):
        k = kmin*np.sqrt(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
        norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
        tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
        delta_lmn = norm_delta * np.exp(tetha_klmn*1j)
        deltax[i] += 2*np.real(delta_lmn * np.exp(1j * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[i], x2[i], x3[i]])))

deltax = const1_L32*deltax

z = np.array(deltax).reshape(M, M, M)
#mp.imshow(z[:,:,0], extent=[0, boxsize, 0, boxsize], cmap='rainbow')
#mp.colorbar()
#mp.show()

mp.pcolormesh(np.linspace(0, boxsize, M), np.linspace(0, boxsize, M), z[:,:,1], cmap='rainbow')
mp.title("New AZ implementation")
mp.colorbar()
mp.show()

mp.pcolormesh(np.linspace(0, boxsize, M), np.linspace(0, boxsize, M), dens0[:,:,1], cmap='rainbow')
mp.title("Previous AZ implementation")
mp.colorbar()
mp.show()

#mp.scatter(x1, x3, c=deltax, cmap='rainbow')
#mp.colorbar()
#mp.show()

#norm_delta_klmn = tetha_klmn_sum/reps
#delta_field_klmn = delta_sum/reps

dx = np.zeros(shape=(N3, 3), dtype = np.float64)
# lambdas = np.zeros(shape=(N3, 3), dtype = np.complex)
for i in tqdm(range(0, N3)):
    for li in range(0, len(l)):
        for mni in range(0, len(mn_grid)):
            if ((mn_grid[mni][0] == 0) & (mn_grid[mni][1] == 0)):
                dx[i] += 0
            else:
                k = kmin*np.sqrt(l[li]**2 + mn_grid[mni][0]**2 + mn_grid[mni][1]**2)
                k2 = k*k
                norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
                tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
                delta_lmn = norm_delta * np.exp(tetha_klmn*1j)
                dx[i] += 2*(kmin/k2)*np.real(delta_lmn * np.exp(1j * np.dot([l[li], mn_grid[mni][0], mn_grid[mni][1]], [x1[i], x2[i], x3[i]])))*np.array([l[li].astype(np.float), mn_grid[mni][0].astype(np.float), mn_grid[mni][1].astype(np.float)])
            # tensor
            # T = np.zeros((3,3), dtype = np.complex)
            # for ii in range(0,3):
            #     for jj in range(0,3):
            #         T[ii,jj] = (kmin**2)*kmin * 1j * (delta_field_klmn[j] / k2) * np.exp(1j * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[i], x2[i], x3[i]]))*lmn_grid[j][ii]*lmn_grid[j][jj]
            # lambdas[i] = LA.eigvalsh(T)

dx = -const1_L32*dx
#dx = np.zeros(shape=(N3, 3), dtype = np.complex)
#for i in tqdm(range(0, N3)):
#    res = Parallel(n_jobs=num_cores)(delayed(disp_func)(i, j, x1, x2, x3, kmin, delta_field_klmn, lmn_grid, k2, const1_L32) for j in range(0, len(lmn_grid)))
#    dx[i] += np.sum(res)
            
fx = dx[:,0]
fy = dx[:,1]
fz = dx[:,2]

#displacements, scaled by the growth function at the redshift we want
d1=growthfunc(1./(1+redshift))/growthfunc(1.)
xdisp=fx*d1
ydisp=fy*d1
zdisp=fz*d1

mp.hist(xdisp, bins = 100)
mp.show()
mp.hist(ydisp, bins = 100)
mp.show()
mp.hist(zdisp, bins = 100)
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

mp.scatter(x1, x2, s=1, marker = ".")
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
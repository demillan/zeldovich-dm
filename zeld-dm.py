import numpy as np
import pylab as M

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
N = 32
N3 = 32**3
x1 = np.linspace(0, boxsize, N)
x2 = np.linspace(0, boxsize, N)
x3 = np.linspace(0, boxsize, N)

x1, x2, x3 = np.random.random((3, N))*boxsize

sk = (N, N, N)
    
x1 = np.fromfunction(lambda x,y,z:x, sk).astype(np.float)
x1[np.where(x1 > ngrid/2)] -= ngrid
x2 = np.fromfunction(lambda x,y,z:y, sk).astype(np.float)
x2[np.where(x2 > ngrid/2)] -= ngrid
x3 = np.fromfunction(lambda x,y,z:z, sk).astype(np.float)
x3[np.where(x3 > ngrid/2)] -= ngrid

M.scatter(x1, x2, s=1)
M.show()

dx = np.zeros(shape=(N3, 3), dtype = np.complex)

for i in range(0, N3):
    for j in range(0, len(lmn_grid)):
        psum = 0
        if ((lmn_grid[j][0] != 0) & (lmn_grid[j][1] != 0) & (lmn_grid[j][2] != 0)):
            k2 = kmin*(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
            norm_delta_klmn = np.sqrt(-np.interp(np.sqrt(k2), pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
            tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
            delta_klmn = norm_delta_klmn * np.exp(tetha_klmn*1j)
            dsum = kmin * const1_L32 * 1j * (delta_klmn / k2) * np.exp(1j * kmin * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1.flatten()[i], x2.flatten()[i], x3.flatten()[i]]))*np.array([lmn_grid[j][0].astype(np.float), lmn_grid[j][1].astype(np.float), lmn_grid[j][2].astype(np.float)])
        dx[i] += dsum

fx = dx[:,0]
fy = dx[:,1]
fz = dx[:,2]

#displacements, scaled by the growth function at the redshift we want
d1=growthfunc(1./(1+redshift))/growthfunc(1.)
xdisp=fx.real*d1
ydisp=fy.real*d1
zdisp=fz.real*d1
    
#assuming ngrid=nparticles, displace particles from the grid
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

M.scatter(x1, x2, s=0.5)
M.show()
import numpy as np
import pylab as mp
import zeldovich as Z

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
ngrid = 32
f=0.0

M = int(boxsize / ngrid)
l = np.arange(-M, M + 1)
m = np.arange(-M, M + 1)
n = np.arange(-M, M + 1)

# Leemos P(k) inicial
pkinit = np.loadtxt('PkTable.dat', skiprows = 5)

# Previous ZA implementation

# initial position
cell_len=np.float(boxsize)/np.float(128)
    
#setup particles on a uniform grid
sk = (ngrid,ngrid,ngrid)
a0 = np.fromfunction(lambda x,y,z:x+0.5, sk).astype(np.float)
b0 = np.fromfunction(lambda x,y,z:y+0.5, sk).astype(np.float)
c0 = np.fromfunction(lambda x,y,z:z+0.5, sk).astype(np.float)
a0=cell_len*a0.flatten()
b0=cell_len*b0.flatten()
c0=cell_len*c0.flatten()

mp.scatter(a0, b0, s=0.1, marker = ".")
mp.show()

dens0 = Z.make_gauss_init(pkinit, boxsize=boxsize, ngrid=ngrid, seed=20181203, exactpk=True, smw=2.)
fx0, fy0, fz0 = Z.get_disp(dens0, boxsize=boxsize, ngrid=ngrid)
x0, y0, z0 = Z.get_pos(fx0, fy0, fz0*(1+f), redshift, boxsize=boxsize, ngrid=ngrid) 

mp.scatter(x0, y0, s=0.1, marker = ".")
mp.show()

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
kmin = (2*np.pi)/boxsize

#setup particles on a uniform grid
N = 32
N3 = N**3
x1 = a0
x2 = b0
x3 = c0

mp.scatter(x1, x2, s=0.1, marker = ".")
mp.show()

dx = np.zeros(shape=(N3, 3), dtype = np.complex)

for i in range(0, N3):
    for j in range(0, len(lmn_grid)):
        psum = 0
        if ((lmn_grid[j][0] != 0) & (lmn_grid[j][1] != 0) & (lmn_grid[j][2] != 0)):
            k2 = (lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)*(kmin**2)
            norm_delta_klmn = np.sqrt(-np.interp(np.sqrt(k2), pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
            tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
            delta_klmn = norm_delta_klmn * np.exp(tetha_klmn*1j)
            dsum = kmin * const1_L32 * 1j * (delta_klmn / k2) * np.exp(1j * kmin * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[i], x2[i], x3[i]]))*np.array([lmn_grid[j][0].astype(np.float), lmn_grid[j][1].astype(np.float), lmn_grid[j][2].astype(np.float)])
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

mp.scatter(x1, x2, s=0.1, marker = ".")
mp.show()


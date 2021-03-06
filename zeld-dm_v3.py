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
boxsize = 32
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
cell_len=np.float(boxsize)/np.float(ngrid)
    
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

mp.hist(fx0.flatten(), bins = 100)
mp.show()
mp.hist(fy0.flatten(), bins = 100)
mp.show()
mp.hist(fz0.flatten(), bins = 100)
mp.show()

mp.scatter(x0, y0, s=0.1, marker = ".")
mp.show()

# Calculamos desplazamiento

const1_L32 = 1/(boxsize**1.5)
kmin = (2*np.pi)/boxsize
kmin2 = kmin**2

#setup particles on a uniform grid
N = ngrid
N3 = N**3
x1 = np.copy(a0)
x2 = np.copy(b0)
x3 = np.copy(c0)

mp.scatter(x1, x2, s=0.1, marker = ".")
mp.show()

dx = np.zeros(shape=(N3, 3), dtype = np.complex)

for i in range(0, N3):
    for l in range(-M, M+1):
        for m in range(-M, M+1):
            for n in range(-M, M+1):
                if ((l == 0) & (m == 0) & (n == 0)):
                    dx[i] += 0
                else:
                    k = kmin*np.sqrt(l**2 + m**2 + n**2)
                    k2 = k*k
                    norm_delta_klmn = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1]) * np.log(np.random.uniform(0, 1)))
                    tetha_klmn = 2 * np.pi * np.random.uniform(0, 1)
                    delta_klmn = norm_delta_klmn * np.exp(tetha_klmn * 1j)
                    # dsum = kmin * 1j * (delta_klmn / k2) * np.exp(1j * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[i], x2[i], x3[i]]))*np.array([lmn_grid[j][0].astype(np.float), lmn_grid[j][1].astype(np.float), lmn_grid[j][2].astype(np.float)])
                    # dsum = kmin * np.array([float(l), float(m), float(n)]) * 1j * (delta_klmn / k2) * np.exp(1j * np.dot([l, m, n], [x1[i], x2[i], x3[i]]))
                    dsum = kmin * np.array([float(l), float(m), float(n)]) * 1j * (delta_klmn / k2) * np.exp(1j * (l*x1[i] + m*x2[i] + n*x3[i]))
                    dx[i] += const1_L32 * dsum

fx = dx[:,0]
fy = dx[:,1]
fz = dx[:,2]

#displacements, scaled by the growth function at the redshift we want
d1=growthfunc(1./(1+redshift))/growthfunc(1.)
xdisp=fx.real*d1
ydisp=fy.real*d1
zdisp=fz.real*d1

mp.hist(fx.real*d1, bins = 100, color = "orange")
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

mp.scatter(x1, x2, s=0.1, marker = ".", color = "orange")
mp.show()

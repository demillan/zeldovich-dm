
#%%

import numpy as np
from numpy import linalg as LA
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
boxsize = 200
eps = 10 # resolution

const1_L32 = 1/(boxsize**1.5)
kmin = (2*np.pi)/boxsize
kmin2 = kmin**2

M = int(boxsize / eps)
l = np.arange(-M, M + 1)
m = np.arange(-M, M + 1)
n = np.arange(-M, M + 1)

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

sqrt2 = np.sqrt(2.)

# This module computes the power spectrum and correlation function of a given 3d density grid

# density is the configuration-space density grid (real-valued, shape = ngrid x ngrid x ngrid)
# nkbins is the number of bins to compute the power spectrum in. It computes it in log-spaced bins in the range (2*Pi/L to Pi*ngrid / L)
# deconvolve_cic if you want to deconvolve a cloud-in-cell window function
# exp_smooth is the size of an exponential smoothing window to deconvolve (if = 0.0, then it does nothing)
def getPk(density, nkbins=40, boxsize=512., deconvolve_cic=False, exp_smooth=0.0):
    #make sure the density has mean 0
    density=density-np.mean(density)
    ngrid=density.shape[0]


    #Fourier transform of density
    deltak=np.fft.rfftn(density)
    sk=deltak.shape

    #Square the density in Fourier space to get the 3D power, make sure k=0 mode is 0
    dk2=(deltak*np.conjugate(deltak)).astype(np.float)
    dk2[0,0,0]=0.0

    #set up k-grid
    kmin=2*np.pi/boxsize
    kny=np.pi*ngrid/boxsize

    a = np.fromfunction(lambda x,y,z:x, sk).astype(np.float)
    a[np.where(a > ngrid/2)] -= ngrid
    b = np.fromfunction(lambda x,y,z:y, sk).astype(np.float)
    b[np.where(b > ngrid/2)] -= ngrid
    c = np.fromfunction(lambda x,y,z:z, sk).astype(np.float)
    c[np.where(c > ngrid/2)] -= ngrid
    kgrid = kmin*np.sqrt(a**2+b**2+c**2).astype(np.float)
    if (deconvolve_cic):
        wx=np.sin(kmin*a*np.pi/(2*kny))**4/(kmin*a*np.pi/(2*kny))**4
        wx[np.where(a==0)]=1.0
        wy=np.sin(kmin*b*np.pi/(2*kny))**4/(kmin*b*np.pi/(2*kny))**4
        wy[np.where(b==0)]=1.0
        wz=np.sin(kmin*c*np.pi/(2*kny))**4/(kmin*c*np.pi/(2*kny))**4
        wz[np.where(c==0)]=1.0
        ww=wx*wy*wz
        dk2=dk2/ww

    if (exp_smooth!=0):
        filt=np.exp(-kgrid**2*exp_smooth**2)
        dk2=dk2/filt


    #Now we want to compute the 1-D power spectrum which involves averaging over shells in k-space

    #define the k-bins we want to compute the power spectrum in
    binedges=np.logspace(np.log10(kmin), np.log10(kny),nkbins)
    numinbin=np.zeros_like(binedges)
    pk=np.zeros_like(binedges)
    kmean=np.zeros_like(binedges)

    kgrid=kgrid.flatten()
    dk2 = dk2.flatten()
    index = np.argsort(kgrid)

    kgrid=kgrid[index]
    dk2=dk2[index]
    c0=0.*c.flatten()+1.
    c0[np.where(c.flatten()==0.)]-=0.5
    c0=c0[index]
    cuts = np.searchsorted(kgrid,binedges)


    for i in np.arange(0, nkbins-1):
        if (cuts[i+1]>cuts[i]):
            numinbin[i]=np.sum(c0[cuts[i]:cuts[i+1]])
            pk[i]=np.sum(c0[cuts[i]:cuts[i+1]]*dk2[cuts[i]:cuts[i+1]])
            kmean[i]=np.sum(c0[cuts[i]:cuts[i+1]]*kgrid[cuts[i]:cuts[i+1]])
    wn0=np.where(numinbin>0.)
    pk=pk[wn0]
    kmean=kmean[wn0]
    numinbin=numinbin[wn0]

    pk/=numinbin
    kmean/=numinbin

    pk*= boxsize**3/ngrid**6

    return kmean, pk

#%%
# delta field generation
reps = 10
delta_init = np.zeros(N3, dtype = np.float64)
delta_init_all = np.zeros(N3, dtype = np.float64)
for rep in range(0, reps):
    print("rep: ", rep)
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
    delta_init_all += (delta_init * const1_L32)

delta_init_all = delta_init_all / reps
delta_init_3d = delta_init_all.reshape(M, M, M)

#%%
k, pk = getPk(delta_init_3d, nkbins=20, boxsize=boxsize, deconvolve_cic=False, exp_smooth=0.0)

mp.loglog(pkinit[:,0], pkinit[:,1])
mp.loglog(k, pk)
mp.ylim([10**-2, 10**5])
mp.xlabel('k [h/Mpc]')
mp.ylabel('P(k)')
mp.title('Power spectrum of the final density field')
mp.figure()

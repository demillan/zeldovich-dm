import numpy as np

def init_field(j, kmin, lmn_grid, pkinit):
    k = kmin*np.sqrt(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
    k2 = k*k
    norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
    tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
    delta = norm_delta * np.exp(tetha_klmn*1j)
    return tetha_klmn, delta

import numpy as np

def init_field(j, kmin, lmn_grid, pkinit):
    k = kmin*np.sqrt(lmn_grid[j][0]**2 + lmn_grid[j][1]**2 + lmn_grid[j][2]**2)
    k2 = k*k
    norm_delta = np.sqrt(-np.interp(k, pkinit[:,0], pkinit[:,1])*np.log(np.random.uniform(0, 1)))
    tetha_klmn = 2*np.pi*np.random.uniform(0, 1)
    delta = norm_delta * np.exp(tetha_klmn*1j)
    return tetha_klmn, delta

def disp_func(i, j, x1, x2, x3, kmin, delta_field_klmn, lmn_grid, k2, const1_L32):
    if ((lmn_grid[j][0] == 0) & (lmn_grid[j][1] == 0) & (lmn_grid[j][2] == 0)):
        return [0j, 0j, 0j]
    else:
        dsum = kmin * 1j * (delta_field_klmn[j] / k2[j]) * np.exp(1j * np.dot([lmn_grid[j][0], lmn_grid[j][1], lmn_grid[j][2]], [x1[i], x2[i], x3[i]]))*np.array([lmn_grid[j][0].astype(np.float), lmn_grid[j][1].astype(np.float), lmn_grid[j][2].astype(np.float)])
        return const1_L32 * dsum
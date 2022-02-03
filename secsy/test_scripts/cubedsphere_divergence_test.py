""" Test numerical calculation of divergence in cubed sphere coords
"""
import numpy as np
import matplotlib.pyplot as plt
from secsy import CSgrid, CSprojection

# need this function to calculate spherical harmonics and deriatives below
def get_legendre(theta, keys):
    """ 
    Calculate Schmidt semi-normalized associated Legendre functions

    Calculations based on recursive algorithm found in "Spacecraft Attitude Determination and Control" by James Richard Wertz
    
    Parameters
    ----------
    theta : array
        Array of colatitudes in degrees
    keys: iterable
        list of spherical harmnoic degree and order, tuple (n, m) for each 
        term in the expansion

    Returns
    -------
    P : array 
        Array of Legendre functions, with shape (theta.size, len(keys)). 
    dP : array
        Array of dP/dtheta, with shape (theta.size, len(keys))
    """

    # get maximum N and maximum M:
    n, m = np.array([k for k in keys]).T
    nmax, mmax = np.max(n), np.max(m)

    theta = theta.flatten()[:, np.newaxis]

    P = {}
    dP = {}
    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    # Initialize Schmidt normalization
    S = {}
    S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            # compute Schmidt normalization
            if m == 0:
                S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
            else:
                S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    # now apply Schmidt normalization
    for n in range(1, nmax + 1):
        for m in range(0, min([n + 1, mmax + 1])):
            P[n, m]  *= S[n, m]
            dP[n, m] *= S[n, m]

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    return Pmat, dPmat    



d2r = np.pi / 180
RE = 6371.2

stencil_size = 1 # stencil to use for numerical differentiation

# DEFINE THE SPHERICAL HARMONIC FUNCTION
lat0, lon0 = 70., 150. # center of projection - and pole of the SH
orientation = 45. # orientation of projection in east/north
N, M = 10, 5 # SH degree and order of the spherical harmonic used for testing
h, g = np.pi, np.e

# Set up cubed sphere projection
projection = CSprojection((lon0, lat0), orientation)
grid = CSgrid(projection, 4000/RE, 4000/RE, 4/RE, 4/RE, R = 1.)

# coordinates of the spherical harmonic function (the local coords of CS projection)
ph = grid.local_lon.flatten().reshape((-1, 1))
th = 90 - grid.local_lat.flatten().reshape((-1, 1))

# calculate the SH function Y
P, dP = get_legendre(th, [(N, M)])
Y = P * (g * np.cos(M * ph * d2r) + h * np.sin(M * ph * d2r))

# calculate the Laplacian ("analytic") of Y on the CS grid:
delY = -N * (N + 1) * Y.reshape(grid.shape)

# calculate the eastward and northward components of grad(Y)
# in the *local* coordinate system:
dYdph_local = P * M * (-g * np.sin(M * ph * d2r) + h * np.cos(M * ph * d2r)) / np.sin(th * d2r) # eastward component
dYdth_local = dP    * ( g * np.cos(M * ph * d2r) + h * np.sin(M * ph * d2r)) # southward component
gradY_local = np.vstack((dYdph_local.flatten(), -dYdth_local.flatten())) # east, north (hence the sign change)

# Rotate the gradient components into the global coordinate system
R = grid.projection.local2geo_enu_rotation(ph.flatten(), 90 - th.flatten())
gradY_e, gradY_n = np.einsum('nij, nj->ni', R, gradY_local.T).T

# Calculate the gradient of Y numerically:
Le, Ln = grid.get_Le_Ln(S = stencil_size, return_sparse = True)
gradY_e_num = Le.dot(Y) # east
gradY_n_num = Ln.dot(Y) # north

# calculate the Laplacian of Y numerically:
D = grid.divergence(S = stencil_size, return_sparse = True)
delY_num = D.dot(np.hstack((gradY_e, gradY_n))).reshape(grid.shape)




# plot comparison
fig, axs = plt.subplots(ncols = 3, nrows = 3, figsize = (10, 10))
axs[0, 0].contourf(grid.xi, grid.eta, Y.reshape(grid.shape))
axs[0, 0].set_title('The Function')
axs[0, 0].contour(grid.xi, grid.eta, grid.lat, levels = np.r_[-85:86:5], linestyles = '--', colors = 'black', linewidths = .5)

axs[0, 1].contourf(grid.xi, grid.eta, gradY_e.reshape(grid.shape))
axs[0, 1].set_title('Eastward gradient, "analytical"')
axs[0, 2].contourf(grid.xi, grid.eta, gradY_n.reshape(grid.shape))
axs[0, 2].set_title('Northward gradient, "analytical"')

axs[1, 0].scatter(gradY_e.flatten(), gradY_e_num.flatten(), marker = '.', c = 'C0', label = 'east')
axs[1, 0].scatter(gradY_n.flatten(), gradY_n_num.flatten(), marker = '.', c = 'C1', label = 'north')
axs[1, 0].legend(frameon = False)
axs[1, 0].set_xlabel('analytical')
axs[1, 0].set_ylabel('numerical')

axs[1, 1].contourf(grid.xi, grid.eta, gradY_e_num.reshape(grid.shape))
axs[1, 1].set_title('Eastward gradient , numerical')
axs[1, 2].contourf(grid.xi, grid.eta, gradY_n_num.reshape(grid.shape))
axs[1, 2].set_title('Northward gradient, numerical)')



axs[2, 0].scatter(delY.flatten(), delY_num.flatten(), marker = '.', c = 'C0')
axs[2, 0].set_xlabel('analytical')
axs[2, 0].set_ylabel('numerical')


axs[2, 2].contourf(grid.xi, grid.eta, delY)
axs[2, 2].set_title('Laplacian, "analytical"')
axs[2, 1].contourf(grid.xi, grid.eta, delY_num)
axs[2, 1].set_title('Laplacian, numerical')

   
plt.ion()
plt.show()
plt.pause(0.001)



""" Example script to test and demonstrate spline interpolation technique """

import numpy as np
from secsy import cubedsphere
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt


# set up grid
position = (40, 70) # lon, lat
orientation = (1, 2) # east, north
L, W, Lres, Wres = 1000e3, 750e3, 20.e3, 20.e3 # dimensions and resolution of grid
grid = cubedsphere.CSgrid(cubedsphere.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)
xi, eta = grid.projection.geo2cube(np.ravel(grid.lon), np.ravel(grid.lat))

# Build K x K matrix Pmain where each row is like a 2D spline basis function defined
# on each grid cell. The basis function in the k'th row is a spline function that should 
# maximize near the center of the k'th cell, and be reduced away from the cell at a speed that 
# depends on the smoothing factor, and in a manner that depends on spline order.  
# It should be possible to add the basis functions together in different proportions to 
# retrieve any scalar field. Adding all basis functions together direclty should give 1s everywhere 
Pmain = np.empty((xi.size, grid.size))
splines = [] 
for i in range(grid.size):
    M = np.zeros(grid.size)
    M[i] = 1
    splines.append( RectBivariateSpline(grid.xi[0], grid.eta[:, 0], M.reshape(grid.shape).T, s = 1) )

    Pmain[:, i] = splines[i].ev(xi, eta)

# make ground truth
SH = grid.lat.flatten() * 0 + 10
SH = SH * np.exp(-(grid.lon.flatten() - 40)**2 / (2 * 1**2) )

# extract random sample to mimic dataset with limited coverage:
iii = np.arange(len(SH))#np.random.choice(np.arange(SH.size), size = 226, replace = False)
iii = np.random.choice(np.arange(SH.size), size = 40, replace = False)
SH_subset = np.log(SH[iii] ) # we fit log(x) since exp(fig(x)) will be positive even if fig(x) is negative
lon = grid.lon.flatten()[iii]
lat = grid.lat.flatten()[iii]
xi, eta = grid.projection.geo2cube(lon, lat)




# Build matrix P by evaluating the effect of unit vectors along 
# each dimension of grid. This defines the columns of the P matrix. 
P = np.empty((xi.size, grid.size))
for i in range(P.shape[1]):
    P[:, i] = splines[i].ev(xi, eta)


m = np.linalg.lstsq(P, SH_subset, rcond = 0)[0]
SH_reconstructed = np.exp(Pmain.dot(m)) # take exponential to get back field

fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize =(14, 7))
ax[0].contourf(grid.xi, grid.eta, SH.reshape(grid.shape), levels = np.linspace(0, 10, 12))
ax[0].set_title('Truth')
ax[1].contourf(grid.xi, grid.eta, SH_reconstructed.reshape(grid.shape), levels = np.linspace(0, 10, 12))
ax[1].scatter(xi, eta, c = np.exp(SH_subset), vmin = 0, vmax = 12, s = 30, edgecolors = 'black', linewidths = .8)
ax[1].set_title('Data points/\n retrieved field')
ax[2].contourf(grid.xi, grid.eta, splines[400].ev(grid.xi.flatten(), grid.eta.flatten()).reshape(grid.shape))
ax[2].scatter(grid.xi.flatten()[400], grid.eta.flatten()[400])
ax[2].set_title('basis function example')
plt.savefig('spline_interpolation_matrix_example.png')
plt.show()



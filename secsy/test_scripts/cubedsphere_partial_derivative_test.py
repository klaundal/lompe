""" 
Test numerical calculation of gradient components in cubed sphere coordinates
"""
import numpy as np
from secsy import CSgrid, CSprojection
import matplotlib.pyplot as plt
d2r = np.pi / 180

N, M = 4, 4 # SH degree and order of the spherical harmonic used for testing

### SET UP CUBED SPHERE GRID AND PROJECTION
position, orientation = (50, 75), 45
projection = CSprojection(position, orientation)
grid = CSgrid(projection, 3000, 3000, 5., 10., R = 1000)
shape = grid.lat.shape
ph = grid.local_lon * np.pi / 180
th = (90 - grid.local_lat) * np.pi / 180
Lxi, Leta = grid.get_Le_Ln(S = 1, return_dxi_deta = True, return_sparse = True)
Le, Ln = grid.get_Le_Ln(S = 1, return_sparse = True)

###### CARTESIAN FUNCTION ################
##########################################
z = np.sin(N * grid.xi) * np.cos(M*grid.eta) + (5*grid.xi)**5
dzdxi = N * np.cos(M*grid.eta) * np.cos(M * grid.xi) + 5 * (5 * grid.xi)**4 * 5
dzdet = -M*np.sin(N * grid.xi) * np.sin(N*grid.eta)


fig = plt.figure(figsize = (15,10))
axz  = fig.add_subplot(231)
ax_dzdxi_true  = fig.add_subplot(232)
ax_dzdxi_est   = fig.add_subplot(235)
ax_dzdet_true  = fig.add_subplot(233)
ax_dzdet_est   = fig.add_subplot(236)
ax_sc = fig.add_subplot(234)

axz.contourf(grid.xi, grid.eta, z)
_ = ax_dzdet_true.contourf(grid.xi, grid.eta, dzdet)
ax_dzdet_est.contourf(grid.xi, grid.eta, Leta.dot(z.flatten()).reshape(grid.shape))#, levels = _.levels)
_ = ax_dzdxi_true.contourf(grid.xi, grid.eta, dzdxi)
ax_dzdxi_est.contourf(grid.xi, grid.eta, Lxi.dot(z.flatten()).reshape(grid.shape))#, levels = _.levels)

ax_sc.scatter(Lxi.dot(z.flatten()), dzdxi, c = 'C0')
ax_sc.scatter(Leta.dot(z.flatten()), dzdet, c = 'C1')
ax_sc.plot(ax_sc.get_xlim(), ax_sc.get_xlim())


####### SPHERICAL FUNCTION ###############
## I'm defining the spherical function in local 
## coords, so the gradient calculation is also 
## done in local coords before rotating into global
## for comparison with output from the numerical
## differentiation. 
## (It looks weird but it's supposed to)
##########################################
P = np.cos(N * 2 * th) - 1
dP = -N * 2 * np.sin(N * 2 * th)

Y     = P  * (np.cos(M * ph) + np.sin(M * ph))
dYdth = dP * (np.cos(M * ph) + np.sin(M * ph))
dYdph = P * M * (np.cos(M * ph) - np.sin(M * ph))

gradY_e_local =  dYdph / np.sin(th) / grid.R
gradY_n_local = -dYdth              / grid.R
grad_local = np.vstack((gradY_e_local.flatten(), gradY_n_local.flatten()))

# function is defined in local coords, but we want the gradient in global. So rotate:
R = grid.projection.local2geo_enu_rotation(ph.flatten() / d2r, 90 - th.flatten() / d2r)

gradY_e, gradY_n = map(lambda x: x.reshape(grid.shape), np.einsum('nij, nj->ni', R, grad_local.T).T )
#gradY_e, gradY_n = gradY_e.reshape(grid.shape), gradY_n.reshape(grid.shape)


### PLOT THE COMPARISON
fig = plt.figure(figsize = (15,10))
axY  = fig.add_subplot(131)
ax_gradY_e_true  = fig.add_subplot(232)
ax_gradY_e_est   = fig.add_subplot(235)
ax_gradY_n_true  = fig.add_subplot(233)
ax_gradY_n_est   = fig.add_subplot(236)

axY.contourf(grid.xi, grid.eta, Y)



_ = ax_gradY_e_true.contourf(grid.xi, grid.eta, gradY_e)
ax_gradY_e_est .contourf(grid.xi, grid.eta, Le.dot(Y.flatten()).reshape(shape), levels = _.levels)
_ = ax_gradY_n_true.contourf(grid.xi, grid.eta, gradY_n)
ax_gradY_n_est .contourf(grid.xi, grid.eta, Ln.dot(Y.flatten()).reshape(shape), levels = _.levels)



for ax in [axY, ax_gradY_e_est, ax_gradY_e_true, ax_gradY_n_est, ax_gradY_n_true]:
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for lat in np.r_[-80:81:10]:
        x, y = projection.geo2cube(np.arange(361), np.ones(361) * lat)
        ax.plot(x, y, 'k:')
    for lon in np.r_[0:361:15]:
        x, y = projection.geo2cube(np.ones(180) * lon, np.linspace(-80, 80, 180))
        ax.plot(x, y, 'k:')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')


axY.set_title('Scalar field')
ax_gradY_e_true.set_title('Eastward gradient \nexact (top) and estimated (bottom)')
ax_gradY_n_true.set_title('Northward gradient \nexact (top) and estimated (bottom)')


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Ln.dot(Y.flatten()), gradY_n, c = 'C0')
ax.scatter(Le.dot(Y.flatten()), gradY_e, c = 'C1')
ax.plot(ax.get_xlim(), ax.get_xlim())

plt.ion()
plt.show()
plt.pause(0.001)

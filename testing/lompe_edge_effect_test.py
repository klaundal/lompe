""" 
Testing what happens if there is a uniform flow field across the grid
The SECS representation can not really model this very well, since the
amplitudes are measures of the divergence of the electric field, which
is zero for a uniform field. However, as this demo shows, the SECS poles
form sheets on the edges of the grid to represent a uniform flow across
the grid. That means (hopefully) that this is a problem that affects the 
edges of the grid to a much greater extent than the interior.

I will produce a set of plots where I rotate the velocity field and show
the resulting solution vector. 


KML, 2022-01-12
"""

import numpy as np
import lompe
from secsy import cubedsphere as cs
import matplotlib.pyplot as plt

point = (5.3, 60.4)  # position of grid (shouldn't matter)
orientation = (1, 0) # east, north 

p = cs.CSprojection(point, orientation)
grid = cs.CSgrid(p, 1000e3, 1000e3, 20e3, 20e3, R = 6371.2e3)

rotations = [0, 45, 90, 135, 180, 225, 270, 315]

# initialize Lompe model:
c = lambda x, y: np.ones_like(x) # conductance function - irrelevant here but must be defined
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (c, c), dipole = True)

# make original flow field, that will be rotated
ve0, vn0 = np.ones(model.grid_J.shape).flatten(), np.zeros(model.grid_J.shape).flatten()
lon, lat = model.grid_J.lon, model.grid_J.lat

# set up plot
fig, axes = plt.subplots(nrows = 2, ncols = 4)

# loop through plot axes and rotation angles:
for rot, ax in zip(rotations, np.ravel(axes)):

    # remove prior data (if any)
    model.clear_model()

    # rotate velocity field:
    a = rot * np.pi/180
    R = np.vstack(((np.cos(a), -np.sin(a)), (np.sin(a), np.cos(a))))
    ve, vn = R.dot(np.vstack((ve0, vn0)))

    # plot arrow in the center of the map to indicate direction of flow
    xi, eta, Axi, Aeta = p.vector_cube_projection(ve[0], vn[0], point[0], point[1] + 0.1)
    ax.arrow(xi[0], eta[0], Axi[0] * 10 * model.grid_E.dxi, Aeta[0] * 10 * model.grid_E.dxi, zorder = 6)

    # make Lompe data object and add to model object:
    vdata = lompe.Data(np.vstack((ve.flatten(), vn.flatten())), coordinates = np.vstack((lon.flatten(), lat.flatten())), datatype = 'convection')
    model.add_data(vdata)

    # run inversion (regularization is needed, but not much!)
    model.run_inversion(l1 = 1e-3, l2 = 0)

    # plot the model vector in cubed sphere projection:
    ax.pcolormesh(model.grid_E.xi_mesh, model.grid_E.eta_mesh, model.m.reshape(model.grid_E.shape), cmap = plt.cm.bwr, vmin = -5, vmax = 5)

    # make plots nice
    ax.set_axis_off()
    ax.set_aspect('equal')


plt.show()



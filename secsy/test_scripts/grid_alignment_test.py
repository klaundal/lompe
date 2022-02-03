""" script to test that the alignment of a projection works as intended, defined 
    by angle or by vector. The script makes a contour plot of the xi/eta coordinates
    of a grid on the same projection.
"""

import matplotlib.pyplot as plt
import numpy as np
from secsy import cubedsphere as cs


fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize = (14, 10))
axes = axes.flatten()

position = (5.32, 60.39)
mainp = cs.CSprojection(position, 0) # all plots will be displayed on this projection

# the four orientations that will be tested:
angles = [0, 90, 180, 270] # east north west south
vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)] # east north west south
labels = ['east', 'north', 'west', 'south']

for i in range(4):
    pa = cs.CSprojection(position, angles[i]) # projection defined with angle
    pv = cs.CSprojection(position, vectors[i]) # defined with vector

    grid_a = cs.CSgrid(pa, 1000e3, 500e3, 20e3, 20e3, R = 6371.2e3) # angle grid
    grid_v = cs.CSgrid(pv, 1000e3, 500e3, 20e3, 20e3, R = 6371.2e3) # vector grid

    xi_a, eta_a = mainp.geo2cube(grid_a.lon, grid_a.lat) # project grid on main projection
    xi_v, eta_v = mainp.geo2cube(grid_v.lon, grid_v.lat)

    axes[i*4    ].contourf(xi_a, eta_a, grid_a.xi , cmap = plt.cm.bwr, levels = 40) # xi angle
    axes[i*4 + 1].contourf(xi_a, eta_a, grid_a.eta, cmap = plt.cm.bwr, levels = 40) # eta angle
    axes[i*4 + 2].contourf(xi_v, eta_v, grid_v.xi , cmap = plt.cm.bwr, levels = 40) # xi vector
    axes[i*4 + 3].contourf(xi_v, eta_v, grid_v.eta, cmap = plt.cm.bwr, levels = 40) # eta vector

    axes[i*4    ].set_title(r'$\xi$, ' + labels[i] + ', angle')
    axes[i*4 + 1].set_title(r'$\eta$, ' + labels[i] + ', angle')
    axes[i*4 + 2].set_title(r'$\xi$, ' + labels[i] + ', vector')
    axes[i*4 + 3].set_title(r'$\eta$, ' + labels[i] + ', vector')


# find common limits for all plots
ximin  = np.min([ax.get_xlim()[0] for ax in axes]) 
ximax  = np.max([ax.get_xlim()[1] for ax in axes])
etamin = np.min([ax.get_ylim()[0] for ax in axes])
etamax = np.max([ax.get_ylim()[1] for ax in axes])

# fix up each panel
for ax in axes:
    ax.set_axis_off() # remove axes
    ax.set_aspect('equal') # equal aspect ratio

    for lat in np.r_[30:80:3]: # plot circles of latitude
        xi, eta = mainp.geo2cube(np.linspace(0, 360, 180), np.full(180, lat))
        ax.plot(xi, eta, 'k-', linewidth = .8)

    for lon in np.r_[-90:90:5]: # meridians:
        xi, eta = mainp.geo2cube(np.full(50, lon), np.linspace(30, 80, 50))
        ax.plot(xi, eta, 'k-', linewidth = .8)

    ax.set_xlim(ximin, ximax)
    ax.set_ylim(etamin, etamax)

plt.show()


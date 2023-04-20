import lompe
import numpy as np
import pandas as pd
from datetime import datetime
from lompe.utils.conductance import hardy_EUV
import apexpy
import matplotlib.pyplot as plt

# run an example model, copying code from demoscript
####################################################

event = '2012-04-05'
apex = apexpy.Apex(int(event[0:4]), refh = 110)

supermagfn = '../examples/sample_dataset/20120405_supermag.h5'
superdarnfn = '../examples/sample_dataset/20120405_superdarn_grdmap.h5'
iridiumfn = '../examples/sample_dataset/20120405_iridium.h5'

# set up grid
position = (-90, 65) # lon, lat
orientation = (-1, 2) # east, north
L, W, Lres, Wres = 4200e3, 7000e3, 100.e3, 100.e3 # dimensions and resolution of grid
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)

# load ampere, supermag, and superdarn data from 2012-05-05
ampere    = pd.read_hdf(iridiumfn)
supermag  = pd.read_hdf(supermagfn)
superdarn = pd.read_hdf(superdarnfn)

t0, t1 = datetime(2012, 4, 5, 12, 0), datetime(2012, 4, 5, 12, 5)

# crop AMPERE data and make lompe Data object:
amp = ampere[(ampere.time >= t0) & (ampere.time <= t1)]
B = np.vstack((amp.B_e.values, amp.B_n.values, amp.B_r.values))
coords = np.vstack((amp.lon.values, amp.lat.values, amp.r.values))
amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', scale = 200e-9)

# crop supermag and make lompe Data object
sm = supermag[t0:t1]
B = np.vstack((sm.Be.values, sm.Bn.values, sm.Bu.values))
coords = np.vstack((sm.lon.values, sm.lat.values))
sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', scale = 100e-9)

# crop superdarn and make lompe Data object
sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1)]
vlos = sd['vlos'].values
coords = np.vstack((sd['glon'].values, sd['glat'].values))
los  = np.vstack((sd['le'].values, sd['ln'].values))
sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', scale = 500)


Kp = 4 # for Hardy conductance model
SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t0, 'hall'    )
SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t0, 'pedersen')
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))

model.add_data(amp_data, sm_data, sd_data)
model.run_inversion(l1 = 1, l2 = 10)


# Calculating FAC method 1 (divergence of J)
fig, axes = plt.subplots(ncols = 3, figsize = (12, 5))
fac1 = model.FAC() * 1e6
axes[0].contourf(model.grid_J.xi, model.grid_J.eta, fac1.reshape(model.grid_J.shape), cmap = plt.cm.bwr, levels = np.linspace(-2.5, 2.5, 22))
axes[0].set_title('FACs calculated as div(J_h)')

# Calculating FAC method 2 (getting SECS CF amplitudes directly)
icf = model._B_cf_matrix(return_poles = True)
fac2 = -icf / np.diag(model.A) * 1e6
axes[1].contourf(model.grid_J.xi, model.grid_J.eta, fac1.reshape(model.grid_J.shape), cmap = plt.cm.bwr, levels = np.linspace(-2.5, 2.5, 22))
axes[1].set_title('FACs calculated with CF amplitudes')

axes[2].scatter(fac1, fac2, zorder = 1)
axes[2].set_title('Scatter plot one vs the other')

for ax in axes:
    ax.set_aspect('equal')

xlim = axes[2].get_xlim()
axes[2].plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], color = 'black', zorder = 0)

plt.show()

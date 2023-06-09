import numpy as np
import pandas as pd
from datetime import timedelta
from lompe.utils.conductance import hardy_EUV
import apexpy
import lompe
import matplotlib.pyplot as plt

#%%
plt.ioff()

#%%
conductance_functions = True

event = '2012-04-05'
#savepath = '/Users/amalie/Downloads/demofigs/'
savepath = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/temp_storage/demofigs_resolution_n_scale/'
apex = apexpy.Apex(int(event[0:4]), refh = 110)

supermagfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_resolution/lompe/examples/sample_dataset/20120405_supermag.h5'
superdarnfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_resolution/lompe/examples/sample_dataset/20120405_superdarn_grdmap.h5'
iridiumfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/computer/UiB/lompe_resolution/lompe/examples/sample_dataset/20120405_iridium.h5'

# set up grid
position = (-90, 65) # lon, lat
orientation = (-1, 2) # east, north
L, W, Lres, Wres = 4200e3, 7000e3, 100.e3, 100.e3 # dimensions and resolution of grid
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)

# load ampere, supermag, and superdarn data from 2012-05-05
ampere    = pd.read_hdf(iridiumfn)
supermag  = pd.read_hdf(supermagfn)
superdarn = pd.read_hdf(superdarnfn)

# these files contain entire day. Function to select from a smaller time interval is needed:
def prepare_data(t0, t1):
    """ get data from correct time period """
    # prepare ampere
    amp = ampere[(ampere.time >= t0) & (ampere.time <= t1)]
    B = np.vstack((amp.B_e.values, amp.B_n.values, amp.B_r.values))
    coords = np.vstack((amp.lon.values, amp.lat.values, amp.r.values))
    amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', scale = 200e-9, error=20e-9)
    #amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', scale = 100e-9, error=20e-9)
    #amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', scale = 200e-9)

    # prepare supermag
    sm = supermag[t0:t1]
    B = np.vstack((sm.Be.values, sm.Bn.values, sm.Bu.values))
    coords = np.vstack((sm.lon.values, sm.lat.values))
    sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', scale = 100e-9, error=1e-9)
    #sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', scale = 100e-9)

    # prepare superdarn
    sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1)]
    vlos = sd['vlos'].values
    coords = np.vstack((sd['glon'].values, sd['glat'].values))
    los  = np.vstack((sd['le'].values, sd['ln'].values))
    sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', scale = 500, error=100)
    #sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', scale = 500)
    
    return amp_data, sm_data, sd_data

#%% Create lompe model

# times during entire day
times = pd.date_range('2012-04-05 00:00', '2012-04-05 23:59', freq = '3Min')
DT = timedelta(seconds = 2 * 60) # will select data from +- DT


Kp = 4 # for Hardy conductance model
SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, times[0], 'hall'    )
SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, times[0], 'pedersen')
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))

#i = 0
t = times[1]
print(t)

SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset

amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)

model.add_data(amp_data, sm_data, sd_data)

# Solve inverse problem
gtg, ltl = model.run_inversion(l1 = 2, l2 = 0)

#%% Standard lompe plot

model.m *= 500

savefile = savepath + str(t).replace(' ','_').replace(':','')
fig = lompe.lompeplot(model, include_data=True, time=t, apex=apex, 
                      savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

#%% Plots related to the model resolution matrix

# Plot PSF
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_PSF'
fig = lompe.visualization.PSFplot(model, i=700, apex=apex, 
                                  savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

# Calculate spatial resolution
model.calc_resolution()

# Plot spatial resolution
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_resolution'
fig = lompe.visualization.resolutionplot(model, apex=apex, 
                                         savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

# Plot offset
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_L'
fig = lompe.visualization.resLplot(model, apex=apex, 
                                   savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

#%% Plots related to model covariance

model.Cmpost *= 500**2

# Plot the posterior model covariance
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_Cmpost'
fig = lompe.visualization.Cmplot(model, apex=apex, 
                                   savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

# Plot the posterior model covariance projected in to ground B
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_Cdpost_gmag'
fig = lompe.visualization.Cdplot(model, dtype='ground_mag', apex=apex, figsize=(12,18), fs=30,
                                   savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

# Plot the posterior model covariance projected into space B
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_Cdpost_smag'
fig = lompe.visualization.Cdplot(model, dtype='space_mag_full', apex=apex, figsize=(12,18), fs=30,
                                   savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

# Plot the posterior model covariance projected into Efield
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_Cdpost_efield'
fig = lompe.visualization.Cdplot(model, dtype='efield', apex=apex, figsize=(12,18), fs=25,
                                   savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

# Plot the posterior model covariance projected into convection
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_Cdpost_convection'
fig = lompe.visualization.Cdplot(model, dtype='convection', apex=apex, figsize=(12,18), fs=25,
                                   savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})

# Plot the posterior model covariance projected into FAC
savefile = savepath + str(t).replace(' ','_').replace(':','') + '_Cdpost_fac'
fig = lompe.visualization.Cdplot(model, dtype='fac', apex=apex, figsize=(12,18), fs=20,
                                   savekw={'fname':savefile, 'dpi':200, 'bbox_inches':'tight'})


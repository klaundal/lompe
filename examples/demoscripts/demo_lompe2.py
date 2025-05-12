import numpy as np
import pandas as pd
from datetime import timedelta
from lompe.utils.conductance import hardy_EUV
import apexpy
import lompe
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.ioff()

#%%

conductance_functions = True

event = '2012-04-05'
savepath = '/home/bing/Dropbox/work/temp_storage/Lompe_demo_figs/'
apex = apexpy.Apex(int(event[0:4]), refh = 110)

supermagfn = '/home/bing/Dropbox/work/code/repos/lompe/examples/sample_dataset/20120405_supermag.h5'
superdarnfn = '/home/bing/Dropbox/work/code/repos/lompe/examples/sample_dataset/20120405_superdarn_grdmap.h5'
iridiumfn = '/home/bing/Dropbox/work/code/repos/lompe/examples/sample_dataset/20120405_iridium.h5'

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
    amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', error = 30e-9, iweight=1.0)

    # prepare supermag
    sm = supermag[t0:t1]
    B = np.vstack((sm.Be.values, sm.Bn.values, sm.Bu.values))
    coords = np.vstack((sm.lon.values, sm.lat.values))
    sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', error = 10e-9, iweight=0.4)

    # prepare superdarn
    sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1)]
    vlos = sd['vlos'].values
    coords = np.vstack((sd['glon'].values, sd['glat'].values))
    los  = np.vstack((sd['le'].values, sd['ln'].values))
    sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', error = 50, iweight=1.0)
    
    return amp_data, sm_data, sd_data

# times during entire day
times = pd.date_range('2012-04-05 00:00', '2012-04-05 23:59', freq = '3Min')
DT = timedelta(seconds = 2 * 60) # will select data from +- DT

Kp = 4 # for Hardy conductance model

#%%
    
# loop through times and save
plt.ioff()
model = None
for t in tqdm(times[:100], total=len(times[:100])):
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    if model is None:
        model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))
    else:
        model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    model.add_data(amp_data, sm_data, sd_data)

    reg = lompe.Regularizer(model.gH, lreg=2)
    model.add_regularization(reg)

    model.solve_steady_state()
    
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')
plt.ion()

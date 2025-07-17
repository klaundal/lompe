#%% Import

import numpy as np
import pandas as pd
from datetime import timedelta
from lompe.utils.conductance import hardy_EUV
import apexpy
import lompe
import matplotlib.pyplot as plt
from tqdm import tqdm

#%% Define input and output paths

event = '2012-04-05'
savepath = '/home/bing/Dropbox/work/temp_storage/Lompe2_kf_demo_figs/'
savepath2 = '/home/bing/Dropbox/work/temp_storage/Lompe2_kf_demo_ind_figs/'
apex = apexpy.Apex(int(event[0:4]), refh = 110)

supermagfn = '/home/bing/Dropbox/work/code/repos/lompe/examples/sample_dataset/20120405_supermag.h5'
smsplinefn = '/home/bing/Dropbox/work/code/repos/lompe/examples/sample_dataset/20120405_supermag_spline.h5'
superdarnfn = '/home/bing/Dropbox/work/code/repos/lompe/examples/sample_dataset/20120405_superdarn_grdmap.h5'
iridiumfn = '/home/bing/Dropbox/work/code/repos/lompe/examples/sample_dataset/20120405_iridium.h5'

#%% Load data

# load ampere, supermag, and superdarn data from 2012-05-05
ampere    = pd.read_hdf(iridiumfn)
supermag  = pd.read_hdf(supermagfn)
smspline  = pd.read_hdf(smsplinefn)
superdarn = pd.read_hdf(superdarnfn)

#%% Define grid

# set up grid
position = (-90, 65) # lon, lat
orientation = (-1, 2) # east, north
L, W, Lres, Wres = 4200e3, 7000e3, 100.e3, 100.e3 # dimensions and resolution of grid
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = 6481.2e3)

#%% Constant used for conductance model

Kp = 5 # for Hardy conductance model

#%% Function for generating Lompe data object

# these files contain entire day. Function to select from a smaller time interval is needed:
def prepare_data_sm(t0, t1):
    """ get data from correct time period """
    # prepare supermag
    sm = supermag[t0:t1]
    B = np.vstack((sm.Be.values, sm.Bn.values, sm.Bu.values))
    coords = np.vstack((sm.lon.values, sm.lat.values))    
    return B*1e-9, coords

def prepare_data_ss(t0, t1):
    """ get data from correct time period """
    # prepare supermag
    sm = smspline[t0:t1]
    B = np.vstack((sm.Be.values, sm.Bn.values, sm.Bu.values))
    coords = np.vstack((sm.lon.values, sm.lat.values))    
    return B*1e-9, coords
    
def prepare_data_sd(t0, t1):
    """ get data from correct time period """
    # prepare superdarn
    sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1)]
    vlos = sd['vlos'].values
    coords = np.vstack((sd['glon'].values, sd['glat'].values))
    los  = np.vstack((sd['le'].values, sd['ln'].values))    
    return vlos, coords, los

def prepare_data_am(t0, t1):
    """ get data from correct time period """
    # prepare ampere
    amp = ampere[(ampere.time >= t0) & (ampere.time <= t1)]
    B = np.vstack((amp.B_e.values, amp.B_n.values, amp.B_r.values))
    coords = np.vstack((amp.lon.values, amp.lat.values, amp.r.values))
    return B*1e-9, coords

#%% Defining time windows for the model.
  
# times during entire day
times = pd.date_range(str(supermag.index[0]), str(supermag.index[-1]), freq = '3Min')
DT = timedelta(seconds = 2 * 60) # will select data from +- DT

times = times[:10]
ts = [(t-times[0]).seconds for t in times]

#%% Loop over all data and make steady-state model

sm_var, sm_coo            = [], []
ss_var, ss_coo            = [], []
am_var, am_coo            = [], []
sd_var, sd_coo, sd_los    = [], [], []

sm_t, sd_t, am_t, SHP_t = [], [], [], []
for t in tqdm(times, total=len(times)):
    
    # data
    d= prepare_data_sm(t-DT, t+DT)
    sm_var.append(d[0])
    sm_coo.append(d[1])
    d= prepare_data_ss(t-DT, t+DT)
    ss_var.append(d[0])
    ss_coo.append(d[1])
    d = prepare_data_am(t-DT, t+DT)
    am_var.append(d[0])
    am_coo.append(d[1])
    d = prepare_data_sd(t-DT, t+DT)
    sd_var.append(d[0])
    sd_coo.append(d[1])
    sd_los.append(d[2])
    
    # conductance
    SHP_t.append((lambda lon = grid.lon, lat = grid.lat, _Kp=Kp, _t=t: hardy_EUV(lon, lat, _Kp, _t, 'hall'    ),
                  lambda lon = grid.lon, lat = grid.lat, _Kp=Kp, _t=t: hardy_EUV(lon, lat, _Kp, _t, 'pedersen')))

del d, t

sm_t = lompe.TimeSeries(values=sm_var, coordinates=sm_coo, 
                        times=ts, datatype='ground_mag', error=10e-9, iweight=0.4)

ss_t = lompe.TimeSeries(values=ss_var, coordinates=ss_coo,
                        times=ts, datatype='dB_ground_mag', error=10e-9, iweight=0.4)

am_t = lompe.TimeSeries(values=am_var, coordinates=am_coo, 
                        times=ts, datatype='space_mag_fac', error=30e-9, iweight=1.0)

sd_t = lompe.TimeSeries(values=sd_var, coordinates=sd_coo, LOS=sd_los,
                        times=ts, datatype='convection', error=50, iweight=1.0)

del sm_var, sm_coo, am_var, am_coo, sd_var, sd_coo, sd_los

#%% Init model
    
model = lompe.Emodel(grid, Hall_Pedersen_conductance_t = SHP_t, times = ts)

#%% Estimating Q and getting init

model.reset_timeseries()
model.add_timeseries(sm_t, sd_t, am_t)

# Create/add Lompe regularization object
model.add_regularization(lompe.Regularizer(model.gH, lreg=2e0))
model.get_SS(times=ts)
model.estimate_Q(cond=1000)

#%% Kalman filter

model.reset_timeseries()
model.add_timeseries(sm_t, ss_t, sd_t, am_t)

model.reset_regularization()
model.add_regularization(lompe.Regularizer(model.gH, lreg=5e-1))

model.solve_kalman(times=ts)

#%%
    
# loop through times and save
plt.ioff()
for i, t in tqdm(enumerate(times), total=len(times)):

    # Initiate or clear Lompe model object
    model.clear_model(Hall_Pedersen_conductance = SHP_t[i]) # reset
    
    model.add_timeseries_subset(ts[i])
    
    # Run Lompe steady state solver
    model.m_CF = model.mss[:, i]
    
    # Plot results and save it
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    plt.close('all')
plt.ion()


#%%

# loop through times and save
plt.ioff()
for i, t in tqdm(enumerate(times), total=len(times)):

    # Initiate or clear Lompe model object
    model.clear_model(Hall_Pedersen_conductance = SHP_t[i]) # reset
    
    model.add_timeseries_subset(ts[i])
    
    # Run Lompe steady state solver
    model.m_CF = model.mss[:, i]
    model.m_DF = model.mcs[:, i]
    
    # Plot results and save it
    savefile = savepath2 + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200}, induction=True)
    plt.close('all')
plt.ion()

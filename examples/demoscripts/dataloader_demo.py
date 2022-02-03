'''Demo of how to use dataloader and make Lompe model.'''

import numpy as np
import pandas as pd
import datetime as dt
import lompe
from lompe.model.cmodel import Cmodel
import lompe.data_tools.dataloader as dataloader
from lompe.utils import conductance
import apexpy

RE = 6371.2e3

# choose event date 
event = '2012-04-05'

# select time of day and time interval you want to model
DT = dt.timedelta(seconds = 2 * 60) # will select data from time +- DT
hour = 21
minute = 50
time = dt.datetime(int(event[0:4]), int(event[5:7]), int(event[8:10]), hour, minute)

# make apex object for magnetic coordinates
a = apexpy.Apex(time.year)

# set up model grid
position = (-80, 82) # lon, lat of location
orientation = (-0.1, 1) # east, north
L, W, Lres, Wres = 10000e3, 6000e3, 100.e3, 100.e3 # dimensions and resolution of grid (L, Lres are along orientation vector)
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, Lres, Wres, R = (RE + 110e3))

# path to save processed files 
tempfile_path = '../sample_dataset/'

# path to raw files (from website)
basepath = tempfile_path + 'raw/'

# all raw files are saved in folder, create processed files and return path:
amperefn = dataloader.read_iridium(event, basepath=basepath, tempfile_path=tempfile_path)
superdarnfn = dataloader.read_sdarn(event, basepath=basepath, tempfile_path=tempfile_path)
supermagfn = dataloader.read_smag(event, basepath=basepath, tempfile_path=tempfile_path, file_name='20220201-09-27-supermag.netcdf')

# if you want to use the superMAG API, replace above line with the following
# supermagfn = dataloader.read_smag(event, file_path=tempfile_path, fromfile=False, userid='your_user_id')

# load data files (whole days of data)
ampere    = pd.read_hdf(amperefn)
superdarn = pd.read_hdf(superdarnfn)
supermag  = pd.read_hdf(supermagfn)

###### MAKE CONDUCTANCE OBJECTS

# 1) INCLUDE SSUSI IMAGES FOR AURORAL CONDUCTANCES
# Use Cmodel to estimate conductances with auroral conductances from SSUSI-image in basepath
# setting EUV=True will include solar EUV in Hall and Pedersen conductances (calibration = 'MoenBrekke1993')
# make conductance object for Lompe
cmod = Cmodel(grid, event, time, EUV=True, basepath=basepath, tempfile_path=tempfile_path)
SH = cmod.hall
SP = cmod.pedersen

# # 2) OR USE HARDY MODEL FOR AURORAL CONDUCTANCES
# Kp = 4      # for Hardy conductance model (2012-05-04)
# SH = lambda lon = grid.lon, lat = grid.lat: conductance.hardy_EUV(lon, lat, Kp, time, 'hall')
# SP = lambda lon = grid.lon, lat = grid.lat: conductance.hardy_EUV(lon, lat, Kp, time, 'pedersen')


###### PREPARE DATA FOR LOMPE
# ampere
amp = ampere[(ampere.time >= time - DT) & (ampere.time <= time + DT)].dropna()
amp_B = np.vstack((amp.B_e.values, amp.B_n.values, amp.B_r.values))
amp_coords = np.vstack((amp.lon.values, amp.lat.values, amp.r.values))

# superdarn
sd = superdarn.loc[(superdarn.index >= time - DT) & (superdarn.index <= time+ DT) & (superdarn.vlos < 2000)].dropna()
sd_vlos = sd['vlos'].values
sd_coords = np.vstack((sd['glon'].values, sd['glat'].values))
sd_los  = np.vstack((sd['le'].values, sd['ln'].values))

# supermag
smag = supermag[time - DT : time + DT].dropna()
smag_B = np.vstack((smag.Be.values, smag.Bn.values, smag.Bu.values)) # nT
smag_coords = np.vstack((smag.lon.values, smag.lat.values))

# make the data objects for Lompe
amp_data =  lompe.Data(amp_B * 1e-9 ,  amp_coords ,                datatype = 'space_mag_full', scale = 200e-9)
sd_data =   lompe.Data(sd_vlos      ,  sd_coords  ,  LOS = sd_los, datatype = 'convection'    , scale = 500)
sm_data =   lompe.Data(smag_B * 1e-9,  smag_coords,                datatype = 'ground_mag'    , scale = 100e-9)

##### MODELLING 
# initialize model, add datasets, and run inversion
model = lompe.Emodel(grid, (SH, SP))
model.add_data(amp_data, sm_data, sd_data)
# 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction
model.run_inversion(l1 = 1, l2 = 10)

# PLOT
fig = lompe.lompeplot(model, include_data = True, time = time, apex = a)
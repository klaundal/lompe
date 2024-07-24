import sys
sys.path.insert(0, '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/secsy/src')
sys.path.insert(0, '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/polplot/src')
sys.path.insert(0, '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/dipole/src')
sys.path.insert(0, '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/lompe')
import numpy as np
import pandas as pd
from datetime import timedelta
from lompe.utils.conductance import hardy_EUV
import apexpy
import matplotlib.pyplot as plt
import lompe

from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
from kneed import KneeLocator as KL
from kneefinder import KneeFinder as KF

#%%
plt.ioff()

conductance_functions = True

event = '2012-04-05'
savepath = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/projects/lompe_joule_reg/figures/'
apex = apexpy.Apex(int(event[0:4]), refh = 110)

supermagfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/lompe/examples/sample_dataset/20120405_supermag.h5'
superdarnfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/lompe/examples/sample_dataset/20120405_superdarn_grdmap.h5'
iridiumfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/lompe/examples/sample_dataset/20120405_iridium.h5'

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

# get figures from entire day and save somewhere

# times during entire day
times = pd.date_range('2012-04-05 00:00', '2012-04-05 23:59', freq = '3Min')
DT = timedelta(seconds = 2 * 60) # will select data from +- DT


Kp = 4 # for Hardy conductance model
SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, times[0], 'hall'    )
SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, times[0], 'pedersen')
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))


    
# loop through times and save
for t in times[1:]:
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)

    gtg, ltl = model.run_inversion(l1 = 2, l2 = 0)
    
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})


#%% 1 timestep

plt.ioff()

conductance_functions = True

event = '2012-04-05'
savepath = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/projects/lompe_joule_reg/figures/'
apex = apexpy.Apex(int(event[0:4]), refh = 110)

supermagfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/lompe/examples/sample_dataset/20120405_supermag.h5'
superdarnfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/lompe/examples/sample_dataset/20120405_superdarn_grdmap.h5'
iridiumfn = '/scratch/BCSS-DAG Dropbox/Michael Madelaire/work/code/lompe_induction_cheat/lompe/examples/sample_dataset/20120405_iridium.h5'

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
    
    ximin = grid.xi_mesh.min()
    ximax = grid.xi_mesh.max()
    etamin = grid.eta_mesh.min()
    etamax = grid.eta_mesh.max()
    
    # prepare ampere
    amp = ampere[(ampere.time >= t0) & (ampere.time <= t1)]
    B = np.vstack((amp.B_e.values, amp.B_n.values, amp.B_r.values))
    coords = np.vstack((amp.lon.values, amp.lat.values, amp.r.values))
    
    xi, eta = grid.projection.geo2cube(coords[0, :], coords[1, :])
    flag = (xi >= ximin) & (xi <= ximax) & (eta >= etamin) & (eta <= etamax)
    B = B[:, flag]
    coords = coords[:, flag]
    
    amp_data = lompe.Data(B * 1e-9, coords, datatype = 'space_mag_fac', error = 30e-9, iweight=1.0)

    # prepare supermag
    sm = supermag[t0:t1]
    B = np.vstack((sm.Be.values, sm.Bn.values, sm.Bu.values))
    coords = np.vstack((sm.lon.values, sm.lat.values))
    
    xi, eta = grid.projection.geo2cube(coords[0, :], coords[1, :])
    flag = (xi >= ximin) & (xi <= ximax) & (eta >= etamin) & (eta <= etamax)
    B = B[:, flag]
    coords = coords[:, flag]
    
    sm_data = lompe.Data(B * 1e-9, coords, datatype = 'ground_mag', error = 10e-9, iweight=0.4)

    # prepare superdarn
    sd = superdarn.loc[(superdarn.index >= t0) & (superdarn.index <= t1)]
    vlos = sd['vlos'].values
    coords = np.vstack((sd['glon'].values, sd['glat'].values))
    los  = np.vstack((sd['le'].values, sd['ln'].values))
    
    xi, eta = grid.projection.geo2cube(coords[0, :], coords[1, :])
    flag = (xi >= ximin) & (xi <= ximax) & (eta >= etamin) & (eta <= etamax)
    vlos = vlos[flag]
    los = los[:, flag]
    coords = coords[:, flag]
    
    sd_data = lompe.Data(vlos, coordinates = coords, LOS = los, datatype = 'convection', error = 50, iweight=1.0)
    
    return amp_data, sm_data, sd_data

# get figures from entire day and save somewhere

# times during entire day
times = pd.date_range('2012-04-05 00:00', '2012-04-05 23:59', freq = '3Min')
DT = timedelta(seconds = 2 * 60) # will select data from +- DT

Kp = 4 # for Hardy conductance model
SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, times[0], 'hall'    )
SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, times[0], 'pedersen')
model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))
    
# loop through times and save
for t in times[1:]:
    print(t)
    
    SH = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'hall'    )
    SP = lambda lon = grid.lon, lat = grid.lat: hardy_EUV(lon, lat, 5, t, 'pedersen')

    model.clear_model(Hall_Pedersen_conductance = (SH, SP)) # reset
    
    amp_data, sm_data, sd_data = prepare_data(t - DT, t + DT)
    
    model.add_data(amp_data, sm_data, sd_data)
    IRLS_iter = 50

    #gtg, ltl = model.run_inversion(l1 = 2, l2 = 0, lj=1e-2, joule_reg=True, IRLS_iter=IRLS_iter)
    #gtg, ltl = model.run_inversion(l1 = 2, l2 = 0, lj=10**np.linspace(-1, 7, 30), joule_reg=True, IRLS_iter=IRLS_iter, step=.5)
    #gtg, ltl = model.run_inversion(l1 = 2, l2 = 0, lj=10**4.5, joule_reg=True, IRLS_iter=IRLS_iter)
    #gtg, ltl = model.run_inversion(l1 = 2, l2 = 0, lj=10**np.linspace(-1, 7, 30), joule_reg=True, IRLS_iter=IRLS_iter, step=0.1)
    #gtg, ltl = model.run_inversion(l1 = 1, l2 = 0, lj=10**np.linspace(-4, 1, 30), joule_reg=True, IRLS_iter=IRLS_iter, step=1, threshold=1, IRLS_max=50, l1_redux=.1)
    gtg, ltl = model.run_inversion(l1 = 3e-1, l2 = 0, lj=10**np.linspace(-4, 1, 30), joule_reg=True, IRLS_iter=IRLS_iter, step=1, threshold=1, IRLS_max=50, l1_redux=.9, 
                                   E_reg=True, FAC_reg=False)
    #gtg, ltl = model.run_inversion(l1 = 2, l2 = 0, lj=10**(np.ones(2)*5), joule_reg=True, IRLS_iter=IRLS_iter)
    #gtg, ltl = model.run_inversion(l1 = 2, l2 = 0, lj=10**np.array([2, 0]), joule_reg=True, IRLS_iter=IRLS_iter)
    model.m = model.ms[-1]
    savefile = savepath + str(t).replace(' ','_').replace(':','')
    lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})
    
    
    SP = np.diag(model.pedersen_conductance(model.lon_J, model.lat_J))
    
    G_Ee, G_En = get_SECS_J_G_matrices(model.lat_J, model.lon_J, model.lat_E, model.lon_E,
                                   current_type = 'curl_free',
                                   RI = model.R,
                                   singularity_limit = model.secs_singularity_limit)
    
    
    jmax = 7e-3
    joule_heating = SP.dot(np.diag(G_Ee.dot(model.m)).dot(G_Ee).dot(model.m) + np.diag(G_En.dot(model.m)).dot(G_En).dot(model.m))
    plt.figure(figsize=(10, 10))
    plt.contourf(grid.xi, grid.eta, joule_heating.reshape(grid.shape), levels=np.linspace(0, jmax, 40), cmap='inferno')
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.title(np.round(np.sum(joule_heating),3))
    plt.savefig(savepath+'joule.png', bbox_inches='tight')
    plt.close('all')
    
    plt.figure(figsize=(15, 10))
    plt.plot(model.ms[0][400:500], color='tab:blue', alpha=0.7)
    plt.plot(model.ms[-1][400:500], color='tab:orange', alpha=0.7)
    for i in range(IRLS_iter-1):
        plt.plot(model.ms[i+1][400:500], linewidth=0.4, color='k', alpha=0.4)
    plt.savefig(savepath+'movie.png', bbox_inches='tight')
    plt.close('all')
    
    break

#%%

plt.figure()
plt.plot(grid.xi.flatten(), grid.eta.flatten(), '.')
plt.plot(grid.xi_mesh.flatten(), grid.eta_mesh.flatten(), '.')

xi, eta = grid.projection.geo2cube(amp_data.coords['lon'], amp_data.coords['lat'])
plt.plot(xi, eta, '.')

xi, eta = grid.projection.geo2cube(sm_data.coords['lon'], sm_data.coords['lat'])
plt.plot(xi, eta, '.')

xi, eta = grid.projection.geo2cube(sd_data.coords['lon'], sd_data.coords['lat'])
plt.plot(xi, eta, '.')

#%%

plt.figure(figsize=(15, 9))
plt.loglog(model.dnorms, model.mnorms, '.-')

kf = KF(np.log10(model.dnorms)[:-6], np.log10(model.mnorms)[:-6])
kf.find_knee()
opt_id = np.argmin(abs(np.log10(model.dnorms) - kf.knee[0]))
plt.loglog(model.dnorms[opt_id], model.mnorms[opt_id], '.', markersize=10, color='k')

#kneed = KL(np.log10(model.dnorms)[:-8], np.log10(model.mnorms)[:-8], curve='convex', direction='decreasing')
#opt_id = np.argmin(abs(np.log10(model.dnorms) - kneed.knee))
#plt.loglog(model.dnorms[opt_id], model.mnorms[opt_id], '.', markersize=10, color='tab:red')

#%%

it = 19
ratio = np.hstack((1/np.exp(np.linspace(0.01, 5, IRLS_iter-1)), 0))
model.m = model.ms[it]
print((ratio)[it-1]*100)

savefile = savepath + str(t).replace(' ','_').replace(':','')
lompe.lompeplot(model, include_data = True, time = t, apex = apex, savekw = {'fname': savefile, 'dpi' : 200})

jmax = 7e-3
joule_heating = SP.dot(np.diag(G_Ee.dot(model.m)).dot(G_Ee).dot(model.m) + np.diag(G_En.dot(model.m)).dot(G_En).dot(model.m))
plt.figure(figsize=(10, 10))
plt.contourf(grid.xi, grid.eta, joule_heating.reshape(grid.shape), levels=np.linspace(0, jmax, 40), cmap='inferno')
ax = plt.gca()
ax.set_aspect('equal')
ax.set_axis_off()
plt.title(np.round(np.sum(joule_heating),3))
plt.savefig(savepath+'joule.png', bbox_inches='tight')
plt.close('all')

mmax = np.max(abs(model.ms[0]))
plt.figure(figsize=(10, 10))
plt.contourf(grid.xi_mesh, grid.eta_mesh, model.m.reshape(grid.xi_mesh.shape), levels=np.linspace(-mmax, mmax, 40), cmap='bwr')
ax = plt.gca()
ax.set_aspect('equal')
ax.set_axis_off()
plt.title(np.round(np.sqrt(model.m.T.dot(model.m)),3))
plt.savefig(savepath+'SECS.png', bbox_inches='tight')
plt.close('all')


#%%
it = 8
opt_id = np.argmin(abs(10**np.linspace(-1, 6, 50) - model.ljs[it]))
plt.ion()
plt.figure()
plt.loglog(model.dnorms[it], model.mnorms[it], '.-')
plt.loglog(model.dnorms[it][opt_id], model.mnorms[it][opt_id], '.', markersize=10)

#%% sum of Joule heating
jhs = []
for i in range(len(model.ms)-1):
    model.m = model.ms[i+1]
    joule_heating = SP.dot(np.diag(G_Ee.dot(model.m)).dot(G_Ee).dot(model.m) + np.diag(G_En.dot(model.m)).dot(G_En).dot(model.m))
    jhs.append(np.round(np.sum(joule_heating),3))
               
plt.figure(figsize=(9, 6))
plt.semilogy(np.arange(1, len(jhs)+1), jhs, '.-', label='Joule stuff')
plt.semilogy(plt.gca().get_xlim(), [2, 2], color='k', label='Target')
plt.xlim([1, len(jhs)+1])
plt.xlabel('Iteration')
plt.ylabel('Integrated Joule heating')
plt.legend()

#%% Sum of squared Joule heating
jhs = []
for i in range(len(model.ms)-1):
    model.m = model.ms[i+1]
    joule_heating = SP.dot(np.diag(G_Ee.dot(model.m)).dot(G_Ee).dot(model.m) + np.diag(G_En.dot(model.m)).dot(G_En).dot(model.m))
    jhs.append(np.round(np.sum(joule_heating**2),3))
               
plt.figure(figsize=(9, 6))
plt.semilogy(np.arange(1, len(jhs)+1), jhs, '.-', label='Joule stuff')
plt.semilogy(plt.gca().get_xlim(), [0.006, 0.006], color='k', label='Target')
plt.xlim([1, len(jhs)+1])
plt.xlabel('Iteration')
plt.ylabel('Integrated squared Joule heating')
plt.legend()

#%% Regularization parameter

plt.figure(figsize=(9, 6))
plt.semilogy(np.arange(1, len(model.ljs)+1), model.ljs, '.-')
plt.xlim([1, len(model.ljs)+1])
plt.xlabel('Iteration')
plt.ylabel('Regularization parameter')

#%% Model variation

maxvar = np.log10(np.quantile(abs(np.diff(np.array(model.ms), axis=0) / np.array(model.ms)[:-1, :]) * 100, .95, axis=1))
minvar = np.log10(np.quantile(abs(np.diff(np.array(model.ms), axis=0) / np.array(model.ms)[:-1, :]) * 100, .05, axis=1))
medvar = np.log10(np.median(abs(np.diff(np.array(model.ms), axis=0) / np.array(model.ms)[:-1, :]) * 100, axis=1))

plt.figure(figsize=(9, 6))
plt.fill_between(np.arange(1, len(model.ljs)+1), minvar, maxvar, color='tab:blue', alpha=.5, label='90%')
plt.plot(np.arange(1, len(model.ljs)+1), medvar, color='tab:orange', label='median')
plt.xlim([1, len(model.ljs)+1])
plt.xlabel('Iteration')
plt.ylabel('Model paratmeter log10 % change')
plt.legend()

#%% L-curve

def calc_curvature(rnorm, mnorm):
    x_t = np.gradient(rnorm)
    y_t = np.gradient(mnorm)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)
    curvature = (xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
    return curvature

it = 1
fig, axs = plt.subplots(2, 1)
curv = calc_curvature(np.log10(model.dnorms[it]), np.log10(model.mnorms[it]))
opt_id = np.argmin(curv)
axs[0].loglog(model.dnorms[it], model.mnorms[it], '.-')
axs[0].loglog(model.dnorms[it][opt_id], model.mnorms[it][opt_id], '.', markersize=10, color='tab:red')

kneed = KL(model.dnorms[it], model.mnorms[it], curve='convex', direction='decreasing')
opt_id = np.argmin(abs(model.dnorms[it] - kneed.knee))
axs[0].loglog(model.dnorms[it][opt_id], model.mnorms[it][opt_id], '.', markersize=10, color='magenta')

kneed = KL(np.log10(model.dnorms[it]), np.log10(model.mnorms[it]), curve='convex', direction='decreasing')
opt_id = np.argmin(abs(np.log10(model.dnorms[it]) - kneed.knee))
axs[0].loglog(model.dnorms[it][opt_id], model.mnorms[it][opt_id], '.', markersize=16, color='tab:orange')

kf = KF(np.log10(model.dnorms[it]), np.log10(model.mnorms[it]))
kf.find_knee()
opt_id = np.argmin(abs(np.log10(model.dnorms[it]) - kf.knee[0]))
axs[0].loglog(model.dnorms[it][opt_id], model.mnorms[it][opt_id], '.', markersize=10, color='k')

opt_id = np.argmin(abs(np.linspace(-1, 7, 30) - np.log10(model.ljs[it])))
axs[0].loglog(model.dnorms[it][opt_id], model.mnorms[it][opt_id], '.', markersize=10, color='w')

axs[1].plot(np.linspace(-1, 6, 30), curv)





#Bsplines in time

#Get started using 04north america notebook
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
import lompe
import apexpy

###############################
# cubed sphere grid parameters:
position = (-90, 68)  # lon, lat for grid center
orientation = 0       # angle of grid x axis - anti-clockwise from east direction
L, W = 7000e3, 3800e3 # extents [m] of grid
dL, dW = 200e3, 200e3 # spatial resolution [m] of grid (originally 100x100 km)

# create grid object
grid = lompe.cs.CSgrid(lompe.cs.CSprojection(position, orientation), L, W, dL, dW, R = 6481.2e3)
nnn = (grid.shape[0]+1) * (grid.shape[1]+1)


# plot grid and coastlines
fig, ax = plt.subplots(figsize = (16, 10))
ax.set_axis_off()
for lon, lat in grid.get_grid_boundaries():
    xi, eta = grid.projection.geo2cube(lon, lat)
    ax.plot(xi, eta, color = 'grey', linewidth = .4)

xlim, ylim = ax.get_xlim(), ax.get_ylim()
for cl in grid.projection.get_projected_coastlines():
    ax.plot(cl[0], cl[1], color = 'C0')

ax.set_xlim(xlim)
ax.set_ylim(ylim)



################
event = '2012-04-05'
# file names and location
supermagfn = '../examples/sample_dataset/20120405_supermag.h5'
superdarnfn = '../examples/sample_dataset/20120405_superdarn_grdmap.h5'
iridiumfn = '../examples/sample_dataset/20120405_iridium.h5'

# load data
supermag  = pd.read_hdf(supermagfn)
superdarn = pd.read_hdf(superdarnfn)
iridium   = pd.read_hdf(iridiumfn)

###########################
def get_data_subsets(t0, t1):
    """ return subsets of data loaded above, between t0 and t1 """

    # Iridium data:
    irid = iridium[(iridium.time >= t0) & (iridium.time <= t1)]
    irid_B = np.vstack((irid.B_e.values, irid.B_n.values, irid.B_r.values))
    irid_coords = np.vstack((irid.lon.values, irid.lat.values, irid.r.values))

    # SuperMAG data:
    smag = supermag.loc[t0:t1, :]
    smag_B = np.vstack((smag.Be.values, smag.Bn.values, smag.Bu.values))
    smag_coords = np.vstack((smag.lon.values, smag.lat.values))

    # SuperDARN data:
    sd = superdarn.loc[t0:t1, :]
    vlos = sd['vlos'].values
    sd_coords = np.vstack((sd['glon'].values, sd['glat'].values))
    los  = np.vstack((sd['le'].values, sd['ln'].values))


    # Make the data objects. The scale keyword determines a weight for the dataset. Increase it to reduce weight
    iridium_data   = lompe.Data(irid_B * 1e-9, irid_coords,            datatype = 'space_mag_fac', scale = 200e-9)
    supermag_data  = lompe.Data(smag_B * 1e-9, smag_coords,            datatype = 'ground_mag'   , scale = 100e-9)
    superdarn_data = lompe.Data(vlos         , sd_coords  , LOS = los, datatype = 'convection'   , scale = 500 )

    return(iridium_data, supermag_data, superdarn_data)

################


# time to plot for
# T0 = dt.datetime(2012, 4, 5, 5, 12)
stime = dt.datetime(2012,4,5,0,0)
DT = dt.timedelta(seconds = 60 * 4) # length of time interval
periods = 15
times = pd.date_range(start=stime,end=stime+periods*DT,periods=periods+1)
# apex object for plotting in magnetic
apex = apexpy.Apex(stime, refh = 110)




##############
#Construct the BSpline functions
# import bspline
from scipy.interpolate import splev
from scipy.linalg import lstsq
knots = np.linspace(1,len(times)-1,len(times)//3)
# knots = np.arange(1,len(times)-1,4) #knots
order = 3
knots = np.array([knots[0]]*order  + list(knots) + [knots[-1]]*order )
# spline = bspline.Bspline(knots, order = order)
# d = ms[100::nnn] # value of secs node amplitude as function of time
# Prepare the model
# knots = spline.knot_vector
bases = []
for i in range(len(knots)):
    weights = np.zeros_like(knots)
    weights[i] = 1
    basis = splev(np.arange(len(times)), (knots, weights, order))
    bases.append(basis)
Gspline = np.array(bases).T


#Arrays to keep matrices for each timestep
Gs = np.empty((0, nnn*Gspline.shape[1]))
Gss = []
ds = np.empty( 0)
# ms = np.empty( 0)
ms = []
l1s = []
GTds = []

tc = 0 # time count, integer
for t in times:
    # making conductance tuples
    Kp = 4 # this is the input to the Hardy model
    SH = lambda lon = grid.lon, lat = grid.lat: lompe.conductance.hardy_EUV(lon, lat, Kp, t, 'hall')
    SP = lambda lon = grid.lon, lat = grid.lat: lompe.conductance.hardy_EUV(lon, lat, Kp, t, 'pedersen')

    # Create Emodel object. Pass grid and Hall/Pedersen conductance functions
    model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH, SP))

    # add datasets to model
    iridium_data, supermag_data, superdarn_data = get_data_subsets(t - DT/2, t + DT/2) # data from new model time
    model.add_data(iridium_data, supermag_data, superdarn_data)

    # Run inversion. l1 and l2 are regularization parameters that control the damping of
    # 1) model norm, and 2) gradient of SECS amplitudes (charges) in magnetic eastward direction
    l1 = 1
    model.run_inversion(l1 = l1, l2 = 0)

    #Store matrices
    G = model._G * model._w.reshape((-1,1)) #apply the lompe weights
    d = model._d * model._w #apply the lompe weights
    GTG = (model._G * model._w.reshape((-1,1))).T.dot(model._G)
    gtg_mag = np.median(np.diagonal(GTG))
    R = l1*gtg_mag
    l1s.append(R)
    GTd = model._G.T.dot(d)
    GTds.append(GTd)

    Gt= np.einsum('ij,k->ijk', G, Gspline[tc,:]).reshape(G.shape[0],G.shape[1]*Gspline.shape[1]) #The part of G from this timestep
    Gs = np.vstack((Gs, Gt ))
    ds = np.hstack((ds, d ))
    # ms = np.hstack((ms, model.m )) #Store result of individual inversion
    ms. append(model.m)
    tc = tc+1 #Advance time index

    print(t)




def get_lompe_m(spline_m, t, knots):
    '''
    spline_m : solution of the spline coeficients
    t : time to evaluate spline functions
    knots : time of spline knots, same units as t
    '''
    bases = [] #To hold contribution from B(t) of each basis function at time t
    for i in range(len(knots)):
        weights = np.zeros_like(knots)
        weights[i] = 1
        basis = splev(t, (knots, weights, order))
        bases.append(basis)
    splines = np.array(bases).T  #contribution from B(t) of each basis function at time t

    coefs = spline_m.reshape(model.m.shape[0],knots.shape[0]) #Array of the spline coefs, per secs  pole
    m_lompe = coefs.dot(splines)

    return m_lompe

####Solving
G = Gs
d = ds
w = np.ones(d.shape)
GTG = G.T.dot(G)
GTd = G.T.dot((d)[:, np.newaxis])

l1 = 1
gtg_mag = np.median(np.diagonal(GTG))
R = l1*gtg_mag * np.eye(GTG.shape[0])

m = lstsq(GTG+R, GTd)[0]


###########3
#Compare bspline solution to time-independent solution

from lompe.model.visualization import *
for i in range(len(times)):
    for ii in range(10):
        fig = plt.figure(figsize = (10,5))
        #Regular solve
        model.m = ms[i]
        ax = fig.add_subplot(121)
        xi, eta, z = model.grid_E.xi, model.grid_E.eta, model.m.reshape(model.grid_E.shape)
        ax.contourf(xi, eta, z, vmin=-10000,vmax=10000)
        plot_mlt(ax, model, times[i], apex, color = 'grey')
        plot_quiver(ax, model, 'convection')
        plot_potential(ax, model)
        # plot_datasets(ax, model, 'convection')
        ax.set_title('Time independent')
        format_ax(ax, model, apex = apex)


        # #Minimize difference from previous model
        # if i == 0:
        #     m0 = np.zeros(len(model.m))
        # else:
        #     m0 = ms[i-1]
        # model.m = ms[i] + (ms[i]/GTds[i]) * l1s[i] * m0
        # ax = fig.add_subplot(132)
        # plot_quiver(ax, model, 'convection')
        # plot_potential(ax, model)
        # # plot_datasets(ax, model, 'convection')
        # ax.set_title('Minimize difference from t-1')
        # format_ax(ax, model, apex = apex)

        #Bspline solve
        t = i + ii/10
        model.m = get_lompe_m(m, t, knots)
        ax = fig.add_subplot(122)
        xi, eta, z = model.grid_E.xi, model.grid_E.eta, model.m.reshape(model.grid_E.shape)
        ax.contourf(xi, eta, z, vmin=-10000,vmax=10000)
        plot_mlt(ax, model, times[i]+ DT*0.1*ii, apex, color = 'grey')
        # plot_quiver(ax, model, 'convection')
        # plot_potential(ax, model)
        # plot_datasets(ax, model, 'convection')
        ax.set_title('Bsplines')
        ax.text(-0.6,0.35,str(times[i]+ DT*0.1*ii))
        format_ax(ax, model, apex = apex)
        fig.savefig('./plots/t=%04.1f.png' % t)


#Make gif
import imageio
import glob
images = []
filenames = glob.glob('./plots/t=*')
filenames.sort()
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('./plots/bsplines.gif', images)

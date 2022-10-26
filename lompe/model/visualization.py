""" lompe visualization tools 

Lots of function to help plot the different lompe quantities
in a nice way. 

The default plotting tool is lompeplot. See documentation of that function
for more details. 

If you want more custom plots, there are many tools in this script that can 
be helpful. For example, the Polarplot class is good for making mlt/mlat 
plots.

"""

import matplotlib.pyplot as plt
import numpy as np
import apexpy
from lompe.dipole import Dipole
from scipy.interpolate import griddata
from matplotlib import rc
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PolyCollection, LineCollection
from lompe.polplot import Polarplot

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Verdana']})
rc('text', usetex=False)


###############################
# DEFINE SOME GLOBAL PARAMETERS

# Default arrow scales (all SI units):
QUIVERSCALES = {'ground_mag':       600 * 1e-9 , # ground magnetic field scale [T]
                'space_mag_fac':    600 * 1e-9 , # FAC magnetic field scale [T]
                'convection':       2000       , # convection velocity scale [m/s]
                'efield':           100  * 1e-3, # electric field scale [V/m]
                'electric_current': 1000 * 1e-3, # electric surface current density [A/m] Ohm's law 
                'secs_current':     1000 * 1e-3, # electric surface current density [A/m] SECS 
                'space_mag_full':   600 * 1e-9 } # FAC magnetic field scale [T]

# Default color scales (SI units):
COLORSCALES =  {'fac':        np.linspace(-1.95, 1.95, 40) * 1e-6 * 2,
                'ground_mag': np.linspace(-980, 980, 50) * 1e-9 / 3, # upward component
                'hall':       np.linspace(0, 20, 32), # mho
                'pedersen':   np.linspace(0, 20, 32)} # mho

# Default color map:
CMAP = plt.cm.magma

RE = 6371.2e3 # Earth radius in meters

# Number of arrows to plot along smallest dimension:
NN = 12 

# Dict of lompe.Model function names for calculating different parameters
funcs = {'efield':           'E', 
         'convection':       'v', 
         'ground_mag':       'B_ground',
         'electric_current': 'j',
         'space_mag_fac':    'B_space_FAC',
         'space_mag_full':   'B_space',
         'fac':              'FAC',
         'hall':             'hall_conductance',
         'pedersen':         'pedersen_conductance',
         'secs_current':     'get_B_SECS_currents'} # This attribute doesn't exist currently

# GLOBAL PARAMETERS DONE
########################


# HELPER FUNCTIONS
##################

def plot_coastlines(ax, model, resolution = '110m', **kwargs):
    """ plot coastlines on axis 
    
    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model object that contains the dataset
    """

    if 'color' not in kwargs.keys():
        kwargs['color'] = 'black'
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    if model.dipole:
        cd = Dipole(model.epoch)

    for cl in model.grid_J.projection.get_projected_coastlines(resolution = resolution):
        xi, eta = cl
        if model.dipole: 
            lon, lat = model.grid_J.projection.cube2geo(xi, eta) # retrieve geographic coords
            mlat, mlon = cd.geo2mag(lat, lon) # convert to magnetic
            xi, eta = model.grid_J.projection.geo2cube(mlon, mlat) # and back to xi, eta

        ax.plot(xi, eta, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def plot_mlt(ax, model, time, apex, mltlevels = np.r_[0:24:3], txtkwargs = None, **kwargs):
    """ plot mlt meridians 

    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model object that contains the dataset
    time: datetime
        needed to calculate mlt
    apex: apexpy.Apex
        needed to calculate mlt
    mltlevels: array
        mlts to show
    txtkwargs: dict, optional
        dictionary of keyword arguments to pass to text for
        writing MLT labels
    kwargs: passed to plot
    """

    if 'color' not in kwargs.keys():
        kwargs['color'] = 'black'

    default_txtkwargs = {'bbox':{'facecolor':'white','alpha':.5,'edgecolor':'none'}, 'ha':'center', 'va':'center'} 
    if txtkwargs == None:
        txtkwargs = default_txtkwargs
    else:
        default_txtkwargs.update(txtkwargs)
        txtkwargs = default_txtkwargs    

            
    if model.dipole:
        mlat, mlon = model.grid_J.lat, model.grid_J.lon
    else:
        mlat, mlon = apex.geo2apex(model.grid_J.lat, model.grid_J.lon, (model.R-RE)*1e-3) # to magnetic
    
    cd = Dipole(apex.year)
    mlt = cd.mlon2mlt(mlon, time)
    mlat = np.linspace(mlat.min(), mlat.max(), 50)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for mltlevel in mltlevels:
        mlon_ = cd.mlt2mlon(mltlevel, time)
        if model.dipole:
            lat, lon = mlat, np.ones(mlat.shape)*mlon_
        else:
            lat, lon, _ = apex.apex2geo(mlat, mlon_, 0)
        iii = model.grid_J.ingrid(lon, lat)
        if np.sum(iii) > 2:
            xi, eta = model.grid_J.projection.geo2cube(lon[iii], lat[iii]) # to xi, eta
            ax.plot(xi, eta, **kwargs)
            ax.text(xi[len(xi)//2], eta[len(xi)//2], str(np.int32(mltlevel)).zfill(2), **txtkwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

        
def format_ax(ax, model, apex = None, **kwargs):
    """ function to format axis for plotting lompe output

    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model to get grid from
    apex: apexpy.Apex object, optional
        If given, apex magnetic latitude contours will be plotted.
        If not (default), geograhpic latitude contours will be shown
    kwargs: optional
        passed to contour, which is used to plot latitude contours
    """ 
    
    if not model.dipole and (apex != None):
        lat, _ = apex.geo2apex(model.grid_J.lat, model.grid_J.lon, (model.R-RE)*1e-3)
    elif model.dipole:
        lat = model.grid_J.lat
    else:
        lat = model.grid_J.lat
    
    if 'levels' not in kwargs.keys():
        kwargs['levels'] = np.r_[-85:89:5]
    if 'colors' not in kwargs.keys():
        kwargs['colors'] = 'lightgrey'
    if 'linewidths' not in kwargs.keys():
        kwargs['linewidths'] = .5

    cs1 = ax.contour(model.grid_J.xi, model.grid_J.eta, lat, **kwargs)
    ax.clabel(cs1, inline=1, fontsize=10, fmt = '%1.0f$^\circ$')
    ax.set_axis_off()
    ax.set_aspect('equal')

    projection = model.grid_J.projection
    ax.format_coord = lambda xi, eta: 'lon = {:.2f}, lat = {:.2f}'.format(* tuple(projection.cube2geo(xi,eta))) 


# HELPER FUNCTIONS DONE
#######################


#####################
# PLOTTING FUNCTIONS 
def plot_quiver(ax, model, dtype, scale = None, **kwargs):
    """ quiver plot of dtype on uniform grid
    
    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model object that contains the dataset
    dtype: string
        type of dataset to plot
    scale: float/int, optional
        scale, in inches (default), to pass to quiver function
    kwargs: passed to quiver

    """
    dtype = dtype.lower()

    # set kwargs for quiver:
    if 'color' not in kwargs.keys():
        kwargs['color'] = 'black'

    if 'zorder' not in kwargs.keys():
        kwargs['zorder'] = 3

    if scale == None:
        kwargs['scale'] = QUIVERSCALES[dtype]
    else:
        kwargs['scale'] = scale

    if 'scale_units' not in kwargs.keys():
        kwargs['scale_units'] = 'inches'


    func = getattr(model, funcs[dtype])
    sh = np.array(model.grid_J.shape)

    # get function values on plotting grid:
    sh = sh // sh.min() * NN 
    ximin  = model.grid_J.xi .min() + model.grid_J.dxi  / 3
    ximax  = model.grid_J.xi .max() - model.grid_J.dxi  / 3
    etamin = model.grid_J.eta.min() + model.grid_J.deta / 3
    etamax = model.grid_J.eta.max() - model.grid_J.deta / 3
    xi, eta = np.meshgrid(np.linspace(ximin, ximax, sh[1]), np.linspace(etamin, etamax, sh[1]))
    lo, la = model.grid_J.projection.cube2geo(xi, eta)
    A = func(lon = lo, lat = la)
    if len(A) == 2:
        Ae, An = A
    if len(A) == 3:
        Ae, An, Au = A 
    x, y, Ax, Ay = model.grid_J.projection.vector_cube_projection(Ae, An, lo, la)
    
    return ax.quiver(x, y, Ax, Ay, **kwargs)


def plot_contour(ax, model, dtype, vertical = False, **kwargs):
    """ plot parameters as contour plot  
    
    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model object that contains the dataset
    dtype: string
        type of dataset to plot
    vertical: bool, optional
        set to True if you want to plot the vertical component of a 3D vector parameter.
        If False (default), the magnitude of the horizontal vector is shown
    kwargs: passed to contourf function

    """

    func = getattr(model, funcs[dtype])

    A = func()
    if len(A) not in [2, 3]: 
        z = A # treat as scalar field
    if len(A) == 2:
        Ae, An= A[0], A[1]
        z = np.sqrt(Ae**2 + An**2)
    if len(A) == 3:
        Ae, An, Au = A
        if not vertical:
            z = np.sqrt(Ae**2 + An**2)
        else:
            z = Au
    if 'cmap' not in kwargs.keys():
        if vertical or dtype == 'fac':
            kwargs['cmap'] = plt.cm.bwr
        else:
            kwargs['cmap'] = CMAP

    if 'levels' not in kwargs.keys():
        kwargs['levels'] = COLORSCALES[dtype]


    if 'zorder' not in kwargs.keys():
        kwargs['zorder'] = 0
    if 'extend' not in kwargs.keys():
        kwargs['extend'] = 'both'

    if z.size == model.grid_J.xi.size:
        xi, eta, z = model.grid_J.xi , model.grid_J.eta , z.reshape(model.grid_J.shape)
    else:
        xi, eta, z = model.grid_E.xi, model.grid_E.eta, z.reshape(model.grid_E.shape)

    return(ax.contourf(xi, eta, z, **kwargs))



def plot_datasets(ax, model, dtype = 'convection', scale = None, **kwargs):
    """ make quiverplot for given dataset

    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model object that contains the dataset
    dtype: string
        type of dataset to plot
    scale: float/int, optional
        scale, in inches (default), to pass to quiver function
    kwargs: passed to quiver
    """

    dtype = dtype.lower()

    if dtype not in model.data.keys():
        raise Exception('No such data type named {}'.format(dtype))

    if len(model.data[dtype]) == 0: 
        return None # no data to be plotted

    # set kwargs for quiver:
    if 'color' not in kwargs.keys():
        kwargs['color'] = 'C1'

    if 'zorder' not in kwargs.keys():
        kwargs['zorder'] = 2

    if scale == None:
        kwargs['scale'] = QUIVERSCALES[dtype]
    else:
        kwargs['scale'] = scale

    if 'scale_units' not in kwargs.keys():
        kwargs['scale_units'] = 'inches'


    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for dataset in model.data[dtype]:
        if (dtype == 'convection') & (dataset.los is not None):
            A = dataset.values
            Ae, An = A * dataset.los[0], A * dataset.los[1]
            lon, lat = dataset.coords['lon'], dataset.coords['lat']
            x, y, Ax, Ay = model.grid_J.projection.vector_cube_projection(Ae, An, lon, lat)
        else:
            if dataset.values.shape[0] == 2:
                Ae, An = dataset.values
            if dataset.values.shape[0] == 3:
                Ae, An, Au = dataset.values
            lon, lat = dataset.coords['lon'], dataset.coords['lat']
            x, y, Ax, Ay = model.grid_J.projection.vector_cube_projection(Ae, An, lon, lat)
        qs = ax.quiver(x, y, Ax, Ay, **kwargs)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    return(qs)



def plot_potential(ax, model, **kwargs):
    """ plot electric potential on axis 

    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model object that contains the dataset
    kwargs: passed to contour
        default contour interval is set to 5 kV. Change by specifying
        a different 'levels' value.
    """

    V = model.E_pot().reshape(model.grid_J.shape) * 1e-3
    V = V - V.min() - (V.max() - V.min())/2

    if 'levels' not in kwargs.keys():
        dV = 5 # contour level step size in kV
        kwargs['levels'] = np.r_[(V.min()//dV)*dV :(V.max()//dV)*dV + dV:dV]
    if 'colors' not in kwargs.keys():
        kwargs['colors'] = 'C0'
    if 'linewidths' not in kwargs.keys():
        kwargs['linewidths'] = 2

    return(ax.contour(model.grid_J.xi, model.grid_J.eta, V, **kwargs))


def polarplot(ax, model, apex, time, dV = None, **clkw):
    """ plot grid and coastlines on mlt/mlat grid

    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model object that contains the dataset
    apex: apexpy.Apex object
        needed for calculating magnetic coordinates
    time: datetime
        needed for calculation of magnetic local time
    dV: int, optional
        set to an integer that represents electric potential
        step size. If not given, electric potential will not be
        shown.
    clkw: dict, optional
        keywords for plotting coastlines passed to Polarplot.coastlines()
    """
    
    pax = Polarplot(ax, minlat = 50)
    cd = Dipole(apex.year)

    # coastlines
    if 'resolution' not in clkw.keys():
        resolution = '110m'
        kwargs = clkw.copy()
    else:
        resolution = clkw.pop('resolution')
        kwargs = clkw.copy()
    
    if 'color' not in kwargs.keys():
        kwargs['color'] = 'lightgrey'
    if 'linewidth' not in kwargs.keys():
        kwargs['linewidth'] = 2
        
    pax.coastlines(time = time, mag = apex, resolution=resolution, **kwargs)

    grid = model.grid_E
    xs = (grid.lon_mesh[0, :], grid.lon_mesh[-1, :], grid.lon_mesh[:, 0], grid.lon_mesh[:, -1])
    ys = (grid.lat_mesh[0, :], grid.lat_mesh[-1, :], grid.lat_mesh[:, 0], grid.lat_mesh[:, -1])
    for i, c in enumerate(zip(xs, ys)):
        lon, lat = c
        if not model.dipole:    
            lat, lon = apex.geo2apex(lat, lon, (model.R-RE)*1e-3)   # to magnetic apex
        mlt = cd.mlon2mlt(lon, time)
        pax.plot(lat, mlt, color = 'black', linewidth = 1.5 if i == 0 else .5, zorder = 2)

    if dV != None: # plot electric potential
        V = model.E_pot().reshape(model.grid_J.shape) * 1e-3
        V = V - V.min() - (V.max() - V.min())/2
        lat, lon = model.grid_J.lat, model.grid_J.lon
        if not model.dipole:
            lat, lon = apex.geo2apex(lat, lon, (model.R-RE)*1e-3)   # to magnetic apex
        mlt = cd.mlon2mlt(lon, time)

        levels = np.r_[(V.min()//dV)*dV :(V.max()//dV)*dV + dV:dV]

        pax.contour(lat, mlt, V, levels = levels, colors = 'C0', linewidths = 1, zorder = 3)


def plot_SECS_amplitudes(ax, model, curl_free = True, **kwargs):
    """ plot SECS amplitudes
    
    plot the amplitude of the magnetic field SECS poles in color, one per cell,
    divide by cell area. 

    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model object that contains the dataset
    curl_free: bool
        True for curl-free (default), False for divergence-free
    kwargs: passed to pcolormesh
    """

    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = plt.cm.bwr
    if 'zorder' not in kwargs.keys():
        kwargs['zorder'] = 0

    if 'levels' not in kwargs.keys():
        levels = COLORSCALES['fac']
    else:
        levels = kwargs['levels']
        kwargs.pop('levels')

    kwargs['norm'] = plt.matplotlib.colors.BoundaryNorm(levels, ncolors = kwargs['cmap'].N, clip = True)

    if curl_free:
        S = model.B_cf_matrix(return_poles = True) / np.diag(model.A)[:-1]
    else:
        S = model.B_df_matrix(return_poles = True) / np.diag(model.A)[:-1]
    S = S.reshape(model.grid_J.shape)
    ax.pcolormesh(model.grid_J.xi_mesh, model.grid_J.eta_mesh, S, **kwargs)
    


def lompeplot(model, figheight = 9, include_data = False, apex = None, time = None, 
                     savekw = None, clkw = {}, quiverscales = None, colorscales = None, 
                     debug = False, return_axes = False):
    """ produce a summary plot of lompe parameters. 

        The output is either a figure displayed on screen or, if savekw is given, a figure saved to disk

        parameters
        ----------
        model: lompe.model
            model that will be plotted
        figheight: float, optional
            figure height - the width is determined automatically based on aspect ratios,
            and (width, height) is the figsize given to matplotlib.pyplot.figure
        include_data: bool, optional
            set to True if you want to also plot data. False (default) if not
        apex: apexpy.Apex object, optional
            specify if you want magnetic coordinate grid instead of geographic
        time: datetime, optional
            specify if you want magnetic local time
        savekw: dictionary, optional
            keyword arguments passed to savefig. If None, the figure will be shown with plt.show()
        clkw: dictionary, optional
            keywords for Polarplot.coastlines(), used to show coastlines in polarplot. Ignored 
            if apex or time are not specified        
        quiverscales: dict, optional
            dictionary of scales (in inches) to use for quiver plots. keys must be valid datatype. 
            default values are used for datatypes that are not in list of keys
        colorscales: dict, optional
            dictionary of colorscales to use in contour plots. keys must be valid datatype. 
            default values are used for datatypes that are not in list of keys
        degbug: bool, optional
            set to True to show SECS currents and CF current amplitudes
        return_axes: bool, optional
            Set to True to return the matplotlib figure and axes objects.
            Default is False and will only return the matplotlib figure object

    """

    if quiverscales == None:
        quiverscales = QUIVERSCALES
    else:
        QUIVERSCALES.update(quiverscales)
        quiverscales = QUIVERSCALES

    if colorscales == None:
        colorscales = COLORSCALES
    else:
        COLORSCALES.update(colorscales)
        colorscales = COLORSCALES

    # Set up figures
    # --------------
    ar = model.grid_E.shape[1] / model.grid_E.shape[0] # aspect ratio
    figsize = ((3 * ar + 1)/2 * figheight * .8, figheight)

    fig = plt.figure(figsize = figsize)
    axes = np.vstack(([plt.subplot2grid((20, 4), ( 0, j), rowspan = 10) for j in range(3)],
                      [plt.subplot2grid((20, 4), (10, j), rowspan = 10) for j in range(3)]))
    for ax in axes.flatten():
        format_ax(ax, model, apex = apex)

    # Velocity
    # --------
    ax = axes[0, 0]
    plot_quiver(ax, model, 'convection')
    plot_potential(ax, model)
    if include_data:
        plot_datasets(ax, model, 'convection')
    ax.set_title('Convection velocity and electric potential')

    # Space magnetic field
    # --------------------
    ax = axes[0, 1]
    plot_quiver(  ax, model, 'space_mag_fac')
    plot_contour( ax, model, 'fac')
    if include_data:
        plot_datasets(ax, model, 'space_mag_fac')
        plot_datasets(ax, model, 'space_mag_full')
    ax.set_title('FAC and magnetic field')

    # Ground magnetic field
    # ---------------------
    ax = axes[0, 2]
    plot_quiver(  ax, model, 'ground_mag')
    plot_contour( ax, model, 'ground_mag', vertical = True)
    if include_data:
        plot_datasets(ax, model, 'ground_mag')
    ax.set_title('Ground magnetic field')

    # Hall conductance
    # ----------------
    ax = axes[1, 0]
    plot_contour(ax, model, 'hall')
    plot_coastlines(ax, model, color = 'grey')
    if time != None and apex != None:
        plot_mlt(ax, model, time, apex, color = 'grey')
    ax.set_title('Hall conductance')

    # Pedersen conductance
    # --------------------
    ax = axes[1, 1]
    plot_contour(ax, model, 'pedersen')
    plot_coastlines(ax, model, color = 'grey')
    if time != None and apex != None:
        plot_mlt(ax, model, time, apex, color = 'grey')
    ax.set_title('Pedersen conductance')

    # Current densities
    # -----------------
    ax = axes[1, 2]
    plot_quiver(ax, model, 'electric_current')
    ax.set_title('Electric currents')
    if debug:
        plot_SECS_amplitudes(ax, model, curl_free = True)
        plot_quiver(ax, model, 'SECS_current', color = 'C2')

    # Polarplot
    # ---------
    if time != None and apex != None:
        ax = plt.subplot2grid((20, 4), (0, 3), rowspan = 10) 
        polarplot(ax, model, apex, time, dV = 5, **clkw)

    # Make scales
    #------------
    cbarax1 = plt.subplot2grid((20, 40), (16, 32), rowspan = 1, colspan = 7)
    cbarax2 = plt.subplot2grid((20, 40), (12, 32), rowspan = 1, colspan = 7)

    arrowax = plt.subplot2grid((20, 40), (19, 31), rowspan = 1, colspan = 8)

    arrowax.set_axis_off()
    arrowax.quiver(.1, .5, 1, 0, scale = 2, scale_units = 'inches')
    arrowax.set_ylim(0, 1)
    arrowax.set_xlim(0, 20)
    arrowax.text(5, 1, '{:.0f} nT (ground), {:.0f} nT (space)\n{:.0f} mA/m, {:.0f} m/s'.format(quiverscales['ground_mag'] * 1e9 // 2, quiverscales['space_mag_full'] * 1e9 // 2, quiverscales['electric_current'] * 1e3 // 2, quiverscales['convection'] // 2 ), ha = 'left', va = 'top')

    if time != None:
        cbarax2.set_title(str(time) + ' UT', fontweight = 'bold')

    fac_levels = colorscales['fac']
    xx = np.vstack((fac_levels, fac_levels)) * 1e6
    yy = np.vstack((np.zeros_like(fac_levels), np.ones_like(fac_levels)))
    cbarax1.contourf(xx, yy, xx, cmap = plt.cm.bwr, levels = fac_levels * 1e6)
    cbarax1.set_xlabel('$\mu$A/m$^2$')
    cbarax1.set_yticks([])
    buax = plt.twiny(cbarax1)
    buax.set_xlim(colorscales['ground_mag'].min() *1e9, colorscales['ground_mag'].max() * 1e9)
    buax.set_xlabel('nT')

    conductance_levels = colorscales['hall']
    xx = np.vstack((conductance_levels, conductance_levels)) 
    yy = np.vstack((np.zeros_like(conductance_levels), np.ones_like(conductance_levels)))
    cbarax2.contourf(xx, yy, xx, levels = conductance_levels, cmap = CMAP)
    cbarax2.set_xlabel('mho')
    cbarax2.set_yticks([])


    # Finish
    # ------
    plt.subplots_adjust(top=0.91, bottom=0.065, left=0.01, right=0.99, hspace=0.1, wspace=0.02) 

    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()
    if return_axes==True:
        return fig, axes, arrowax, [cbarax1, cbarax2]
    else:
        return fig


def model_data_scatterplot(model, fig_parameters = {'figsize':(8, 8)}):
    """
    Make a scatter plot of lompe model predictions vs input data 

    parameters
    ----------
    model: lompe.Emodel object
        should contain datasets and a solution vector.
    fig_parameters: dict
        parameters passed to the plt.subplots function 

    returns
    -------
    ax: matplotlib AxesSubplot object
    """

    fig, ax = plt.subplots(**fig_parameters)

    # loop through the data objects and make scatter plots:
    counter = 0
    for dtype in model.data.keys(): # loop through data types
        for ds in model.data[dtype]: # loop through the datasets within each data type

            # skip data points that are outside biggrid:
            ds = ds.subset(model.biggrid.ingrid(ds.coords['lon'], ds.coords['lat']))

            if 'mag' in dtype:
                Gs = np.split(model.matrix_func[dtype](**ds.coords), 3, axis = 0)
                Bs = map(lambda G: G.dot(model.m), Gs)
                for B, d, sym in zip(Bs, ds.values, ['>', '^', 'o']):
                    ax.scatter(d/ds.scale, B/ds.scale, marker  = sym, c = 'C' + str(counter), alpha = .7)


            if (dtype in ['convection', 'efield']):
                Ge, Gn = model.matrix_func[dtype](**ds.coords)

                if ds.los is not None: # deal with line of sight data:
                    G = Ge * ds.los[0].reshape((-1, 1)) + Gn * ds.los[1].reshape((-1, 1))
                    ax.scatter(ds.values/ds.scale, G.dot(model.m)/ds.scale, c = 'C' + str(counter), marker = 'x', zorder = 4, alpha = .7)
                if ds.los is None:
                    Es = [Ge.dot(model.m), Gn.dot(model.m)]
                    for E, d, sym in zip(Es, ds.values, ['>', '^']):
                        ax.scatter(d/ds.scale, E/ds.scale, marker  = sym, c = 'C' + str(counter), zorder = 4, alpha = .7)

            counter += 1

    extent = np.max(np.abs(np.hstack((ax.get_xlim(), ax.get_ylim()))))
    ax.set_aspect('equal')

    ax.plot([-extent, extent], [-extent, extent], 'k-', zorder = 7)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)

    ax.plot([0, 0], [-1, 1], linestyle = '--', color = 'black', zorder = 7)
    ax.plot([-1, 1], [0, 0], linestyle = '--', color = 'black', zorder = 7)

    nranges = extent // 2 * 2 # number of unit lenghts for the scale

    ax.plot([extent*.9, extent*.9], [-extent + .1, - extent + .1 + nranges], 'k-', zorder = 7)
     # scale
    ax.plot([extent*.89, extent*.91], [-extent + .1]*2, 'k-', zorder = 7)
    ax.plot([extent*.89, extent*.91], [-extent + .1 + nranges]*2, 'k-', zorder = 7)

    ax.set_axis_off()


    # loop through the data objects and make labels:
    counter = 0
    for dtype in model.data.keys(): # loop through data types
        for ds in model.data[dtype]: # loop through the datasets within each data type

            if 'mag' in dtype:
                ax.text(extent * (.9 - counter * .05), -extent + .1 + nranges/2, str(int((ds.scale * nranges * 1e9))) + ' nT', c = 'C' + str(counter), va = 'center', ha = 'right', rotation = 90)

            if dtype == 'convection':
                ax.text(extent *  (.9 - counter * .05), -extent + .1 + nranges/2, str(int((ds.scale * nranges))) + ' m/s', c = 'C' + str(counter), va = 'center', ha = 'right', rotation = 90)

            if dtype == 'efield':
                ax.text(extent *  (.9 - counter * .05), -extent + .1 + nranges/2, str(int((ds.scale * nranges * 1e3))) + ' V/m', c = 'C' + str(counter))

            ax.text(-extent + .1, extent -.3 - counter * .25, ds.label.replace('_',' '), color = 'C' + str(counter), va = 'top', ha = 'left', size = 14)


            counter += 1



    # make legend for components:
    ax.scatter(extent+1, extent+1, marker = '>', c = 'black', label = 'east')
    ax.scatter(extent+1, extent+1, marker = '^', c = 'black', label = 'north')
    ax.scatter(extent+1, extent+1, marker = 'o', c = 'black', label = 'up')
    ax.scatter(extent+1, extent+1, marker = 'x', c = 'black', label = 'line-of-sight')
    ax.legend(frameon = False, loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))


    ax.set_title("Model (y) vs data (x)")

    ax.set_axis_off()

    return(ax)





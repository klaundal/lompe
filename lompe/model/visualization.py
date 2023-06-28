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
from dipole import Dipole
from scipy.interpolate import griddata
from matplotlib import rc
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PolyCollection, LineCollection
from polplot import Polarplot

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
    
    '''
    scale = 1
    if dtype == 'space_mag_fac':
        scale = 2e9
    elif dtype == 'electric_current':
        scale = 3e2
    elif dtype == 'ground_mag':
        scale = 5e9
                
    if len(A) == 3:
        Au *= scale
    Ae *= scale
    An *= scale
        
        
    if dtype != 'electric_current':
        if len(model.data[dtype]) != 0:
            if len(A) == 3:
                Au *= model.data[dtype][0].scale
            Ae *= model.data[dtype][0].scale
            An *= model.data[dtype][0].scale
    '''
    
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

    '''
    if dtype == 'ground_mag':
        z *= 3e2
    if dtype == 'fac':
        z *= 5e2
    '''
        
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
def plot_locations(ax, model, dtype='convection', **kwargs):
    dtype = dtype.lower()

    if dtype not in model.data.keys():
        raise Exception('No such data type named {}'.format(dtype))

    if len(model.data[dtype]) == 0: 
        return None # no data to be plotted
    # set kwargs for quiver:
    if 'color' not in kwargs.keys():
        kwargs['color'] = 'C1'

    if 'zorder' not in kwargs.keys():
        kwargs['zorder'] = 1
    
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    for dataset in model.data[dtype]:
        lon, lat = dataset.coords['lon'], dataset.coords['lat']
        x, y= model.grid_J.projection.geo2cube(lon, lat)
        scat = ax.scatter(x, y, **kwargs)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return(scat)

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

    '''
    V *= 3e2
    '''

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
    


def lompeplot(model, figheight = 9, include_data = False, show_data_location= False, apex = None, time = None, 
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
        show_data_location: bool, optional
            will scatter the locations of data. The default is False and the locations
            won't be plotted
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
    if show_data_location:
        plot_locations(ax, model, 'convection')
    if include_data:
        plot_datasets(ax, model, 'convection')
    ax.set_title('Convection velocity and electric potential')

    # Space magnetic field
    # --------------------
    ax = axes[0, 1]
    plot_quiver(  ax, model, 'space_mag_fac')
    plot_contour( ax, model, 'fac')
    if show_data_location:
        plot_locations(ax, model, 'space_mag_fac')
        plot_locations(ax, model, 'fac')

    if include_data:
        plot_datasets(ax, model, 'space_mag_fac')
        plot_datasets(ax, model, 'space_mag_full')
    ax.set_title('FAC and magnetic field')

    # Ground magnetic field
    # ---------------------
    ax = axes[0, 2]
    plot_quiver(  ax, model, 'ground_mag')
    plot_contour( ax, model, 'ground_mag', vertical = True)
    if show_data_location:
        plot_locations(ax, model, 'ground_mag')
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

def add_background(ax, xlim, ylim, color='k', alpha=0.8, zorder=-1):
    
    return ax

def mapPlot(ax, var, grid, model, mapDict=None, includeData=None, JBoundary=None, background=None):
    
    cc = ax.pcolormesh(grid.xi, grid.eta, var, **mapDict)
    
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if isinstance(background, dict):
        ax.fill_between(x=xlim, y1=[ylim[0]]*2, y2=[ylim[1]]*2, **background)
    
    # Plot grid edge
    if isinstance(JBoundary, dict):
        ximin, ximax = np.min(model.grid_J.xi), np.max(model.grid_J.xi)
        etamin, etamax = np.min(model.grid_J.eta), np.max(model.grid_J.eta)
        ax.plot([ximin, ximax], [etamin]*2, **JBoundary)
        ax.plot([ximin, ximax], [etamax]*2, **JBoundary)
        ax.plot([ximin]*2, [etamin, etamax], **JBoundary)
        ax.plot([ximax]*2, [etamin, etamax], **JBoundary)
    
    # Plot data
    if isinstance(includeData, dict):
        dtypes = includeData['dtypes']
        colors = includeData['colors']
        incData = includeData.copy()
        incData.pop('dtypes')
        incData.pop('colors')
        for dtype, dcol in zip(dtypes, colors):
            if len(model.data[dtype]) == 0:
                continue
            for dataset in model.data[dtype]:
                xi_d, eta_d = model.grid_E.projection.geo2cube(dataset.coords['lon'], dataset.coords['lat'])
                ax.plot(xi_d, eta_d, color=dcol, label=dtype, **incData)
    
    # Pretty
    ax.set_aspect('equal')
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    
    return ax, cc

def resolutionplot(model, apex=None, savekw = None, return_axes = False,
                   mapDict=None, background=None, JBoundary=None, includeData= None,
                   unit='km', figsize=(20,14), fs=20):
    """ Produce a plot of spatial resolution.

        NOTE: The spatial resolution is not enough by itself to determine the reliability of the model in a particular area.
        One should also use locerrorplot to assess whether localization error of the model parameter is also an issue.

        The output is either a figure displayed on screen or, if savekw is given, a figure saved to disk

        parameters
        ----------
        model: lompe.model
            model that will be plotted
        apex: apexpy.Apex object, optional
            specify if you want magnetic coordinate grid instead of geographic
        savekw: dictionary, optional
            keyword arguments passed to savefig. If None, the figure will be shown with plt.show()
        return_axes: bool, optional
            Set to True to return the matplotlib figure and axes objects.
            Default is False and will only return the matplotlib figure object
        mapDict: dict, optional
            Arguments for plotting map
        background: dict, optional
            Arguments for adding background when using mask
        JBoundary: dict, optional
            Arguments for showing the extent of grid_J
        includeData, dict, optional
            Arguments for illustration of data from inversion
        unit: string, optional
            Unit of spatial resolution shown on colorbar.
        figsize: tuple, optional
            Size of figure.
        fs: int, optional
            Fontsize of text in plot.

    """

    # Set up figures
    fig = plt.figure(figsize = figsize)
    axes = [plt.subplot2grid((32, 2), (0, 0), rowspan = 30),
            plt.subplot2grid((32, 2), (0, 1), rowspan = 30)]
    cax = plt.subplot2grid((32, 10), (31, 0), colspan = 7)
    
    # Format axes
    for ax in axes:
        format_ax(ax, model, apex = apex, colors='gray')
    
    # Color scale
    vmin = np.min(np.hstack((model.xiRes[model.xiResFlag > 0].flatten(), 
                             model.etaRes[model.etaResFlag > 0].flatten())))
    vmax = np.max(np.hstack((model.xiRes[model.xiResFlag > 0].flatten(), 
                             model.etaRes[model.etaResFlag > 0].flatten())))
    
    # Default settings
    if not isinstance(mapDict, dict):
        mapDict     = {'vmin':vmin, 'vmax':vmax, 'cmap':'Reds', 'zorder':0}
    
    if not isinstance(background, dict):
        background  = {'color':'k', 'alpha':0.8, 'zorder':-1}
    
    if not isinstance(JBoundary, dict):
        JBoundary   = {'linewidth':0.8, 'color':'tab:blue'}

    if not isinstance(includeData, dict):
        includeData = {'dtypes':['ground_mag', 'space_mag_fac', 'convection'],
                       'colors':['k', 'tab:green', 'tab:blue'],
                       'marker':'*', 'linestyle':'', 'alpha':0.8, 'zorder':1}
    
    var = np.ma.array(model.xiRes, mask=(model.xiResFlag+model.etaResFlag) < 2)
    _, cc = mapPlot(axes[0], var, model.grid_E, model, mapDict=mapDict, 
                includeData=includeData, JBoundary=JBoundary, background=background)
    
    var = np.ma.array(model.etaRes, mask=(model.xiResFlag+model.etaResFlag) < 2)
    _, cc = mapPlot(axes[1], var, model.grid_E, model, mapDict=mapDict, 
                includeData=includeData, JBoundary=JBoundary, background=background)
    
    # Add legend
    lgnd = ax.legend(loc=3, bbox_to_anchor=(0.4, -0.18), fontsize=fs)
    for lgndhandle in lgnd.legendHandles:
        lgndhandle._markersize = fs
    
    # Add colorbar
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticklabels(cax.get_xticklabels(), fontsize=fs)
    cax.set_xlabel(unit, fontsize=fs)
    
    # Add title
    axes[0].text(0.5, 1.02, 'Resolution in $\u03be$ [{}]'.format(unit), 
                 ha='center', va='bottom', fontsize=fs, transform=axes[0].transAxes)
    axes[1].text(0.5, 1.02, 'Resolution in $\u03b7$ [{}]'.format(unit), 
                 ha='center', va='bottom', fontsize=fs, transform=axes[1].transAxes)
    
    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()
    if return_axes:
        return fig, axes, cax
    else:
        return fig

def locerrorplot(model, apex=None, savekw=None, return_axes = False,
             mapDict=None, background=None, JBoundary=None, includeData= None,
             unit='km', figsize=(12,16), fs=20):
    """ Produce a plot of localization error. 
    
        Even if the output of resolutionplot looks good, one must be aware that where large values 
        appear in this plot, they indicate where the model output should not be trusted. The 
        values indicate the distance between the location of the model parameter and the center of 
        its PSF. See Equation (11) in Oldenborger et al (2009, doi: 10.1111/j.1365-246X.2008.04003.x),
        where this metric is referred to as the "localization error".

        The output is either a figure displayed on screen or, if savekw is given, a figure saved to disk

        parameters
        ----------
        model: lompe.model
            model that will be plotted
        apex: apexpy.Apex object, optional
            specify if you want magnetic coordinate grid instead of geographic
        savekw: dictionary, optional
            keyword arguments passed to savefig. If None, the figure will be shown with plt.show()
        return_axes: bool, optional
            Set to True to return the matplotlib figure and axes objects.
            Default is False and will only return the matplotlib figure object
        mapDict: dict, optional
            Arguments for plotting map
        background: dict, optional
            Arguments for adding background when using mask
        JBoundary: dict, optional
            Arguments for showing the extent of grid_J
        includeData, dict, optional
            Arguments for illustration of data from inversion
        unit: string, optional
            Unit of spatial resolution shown on colorbar.
        figsize: tuple, optional
            Size of figure.
        fs: int, optional
            Fontsize of text in plot.

    """

    # Set up figures
    fig = plt.figure(figsize = figsize)
    ax  = plt.subplot2grid((32, 1), (0, 0), rowspan=30)
    cax = plt.subplot2grid((32, 10), (31, 1), colspan = 8)
    
    format_ax(ax, model, apex = apex, colors='gray')
    
    # Colar scale
    vmin = np.min(model.resL[(model.xiResFlag + model.etaResFlag) == 2])
    vmax = np.max(model.resL[(model.xiResFlag + model.etaResFlag) == 2])
    
    # Default settings
    if not isinstance(mapDict, dict):
        mapDict     = {'vmin':vmin, 'vmax':vmax, 'cmap':'Reds', 'zorder':0}
    
    if not isinstance(background, dict):
        background  = {'color':'k', 'alpha':0.8, 'zorder':-1}
    
    if not isinstance(JBoundary, dict):
        JBoundary   = {'linewidth':0.8, 'color':'tab:blue'}

    if not isinstance(includeData, dict):
        includeData = {'dtypes':['ground_mag', 'space_mag_fac', 'convection'],
                       'colors':['k', 'tab:green', 'tab:blue'],
                       'marker':'*', 'linestyle':'', 'alpha':0.8, 'zorder':1}
    
    var = np.ma.array(model.resL, mask=(model.xiResFlag + model.etaResFlag) < 2)
    _, cc = mapPlot(ax, var, model.grid_E, model, mapDict=mapDict, 
                includeData=includeData, JBoundary=JBoundary, background=background)
    
    # Add colorbar
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticklabels(cax.get_xticklabels(), fontsize=fs)
    cax.set_xlabel(unit, fontsize=fs)
    
    # Legend
    lgnd = ax.legend(loc=8, bbox_to_anchor=(0.5, -0.25), fontsize=fs, ncols=2)
    for lgndhandle in lgnd.legendHandles:
        lgndhandle._markersize = fs
    
    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()
        
    if return_axes:
        return fig, ax
    else:
        return fig

def PSFplot(model, i, apex=None, savekw=None, return_axes = False,
            mapDict=None, background=None, JBoundary=None, includeData= None,
            figsize=(12,12), fs=20):
    """ produce a plot of PSFs.

        The output is either a figure displayed on screen or, if savekw is given, a figure saved to disk

        parameters
        ----------
        model: lompe.model
            model that will be plotted
        apex: apexpy.Apex object, optional
            specify if you want magnetic coordinate grid instead of geographic
        savekw: dictionary, optional
            keyword arguments passed to savefig. If None, the figure will be shown with plt.show()
        return_axes: bool, optional
            Set to True to return the matplotlib figure and axes objects.
            Default is False and will only return the matplotlib figure object
        mapDict: dict, optional
            Arguments for plotting map
        background: dict, optional
            Arguments for adding background when using mask
        JBoundary: dict, optional
            Arguments for showing the extent of grid_J
        includeData, dict, optional
            Arguments for illustration of data from inversion
        unit: string, optional
            Unit of spatial resolution shown on colorbar.
        figsize: tuple, optional
            Size of figure.
        fs: int, optional
            Fontsize of text in plot.

    """

    # Set up figures
    fig = plt.figure(figsize = figsize)
    ax = plt.gca()
    format_ax(ax, model, apex = apex, colors='gray')
    
    format_ax(ax, model, apex = apex, colors='gray')
    
    PSF = model.Rmatrix[:, i].reshape(model.grid_E.shape)
    
    # Colar scale
    vmax = np.max(PSF)
    
    # Default settings
    if not isinstance(mapDict, dict):
        mapDict     = {'vmin':-vmax, 'vmax':vmax, 'cmap':'bwr', 'zorder':0}
    
    if not isinstance(background, dict):
        background  = {'color':'k', 'alpha':0.8, 'zorder':-1}
    
    if not isinstance(JBoundary, dict):
        JBoundary   = {'linewidth':0.8, 'color':'tab:blue'}

    if not isinstance(includeData, dict):
        includeData = {'dtypes':['ground_mag', 'space_mag_fac', 'convection'],
                       'colors':['k', 'tab:green', 'tab:blue'],
                       'marker':'*', 'linestyle':'', 'alpha':0.8, 'zorder':1}
    
    _, cc = mapPlot(ax, PSF, model.grid_E, model, mapDict=mapDict, 
                includeData=includeData, JBoundary=JBoundary, background=background)
    
    # Plot id and max
    row = i//model.grid_E.shape[1]
    col = i%model.grid_E.shape[1]
    ax.plot(model.grid_E.xi[row, col], model.grid_E.eta[row, col], 
            '.', color='k', markersize=17, label='Impulse')
    
    ii = np.argmax(abs(PSF))
    rowPSF = ii//model.grid_E.shape[1]
    colPSF = ii%model.grid_E.shape[1]
    ax.plot(model.grid_E.xi[rowPSF, colPSF], model.grid_E.eta[rowPSF, colPSF], 
            '.', color='tab:green', markersize=14, label='max |PSF|')
        
    # Legend
    lgnd = ax.legend(loc=8, bbox_to_anchor=(0.5, -0.2), fontsize=fs, ncols=2)
    for lgndhandle in lgnd.legendHandles:
        lgndhandle._markersize = fs
    
    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()
        
    if return_axes:
        return fig, ax
    else:
        return fig

def Cmplot(model, apex=None, savekw = None, return_axes = False,
                   mapDict=None, background=None, JBoundary=None, includeData= None,
                   unit='kV', figsize=(12,16), fs=20):
    """ Produce a plot of the diagonals of the posterior model covariance (i.e., model parameter "uncertainty"). 

        The output is either a figure displayed on screen or, if savekw is given, a figure saved to disk

        parameters
        ----------
        model: lompe.model
            model that will be plotted
        apex: apexpy.Apex object, optional
            specify if you want magnetic coordinate grid instead of geographic
        savekw: dictionary, optional
            keyword arguments passed to savefig. If None, the figure will be shown with plt.show()
        return_axes: bool, optional
            Set to True to return the matplotlib figure and axes objects.
            Default is False and will only return the matplotlib figure object
        mapDict: dict, optional
            Arguments for plotting map
        background: dict, optional
            Arguments for adding background when using mask
        JBoundary: dict, optional
            Arguments for showing the extent of grid_J
        includeData, dict, optional
            Arguments for illustration of data from inversion
        unit: string, optional
            Unit of spatial resolution shown on colorbar (default kilovolts).
        figsize: tuple, optional
            Size of figure.
        fs: int, optional
            Fontsize of text in plot.

    """

    # Set up figures        
    fig = plt.figure(figsize = figsize)
    ax  = plt.subplot2grid((32, 1), (0, 0), rowspan=30)
    cax = plt.subplot2grid((32, 10), (31, 1), colspan = 8)
    
    format_ax(ax, model, apex = apex, colors='gray')
        
    var = np.sqrt(np.diag(model.Cmpost).reshape(model.grid_E.shape))*1e3
    
    # Colar scale
    vmin = np.min(var)
    vmax = np.max(var)
    
    # Default settings
    if not isinstance(mapDict, dict):
        mapDict     = {'vmin':vmin, 'vmax':vmax, 'cmap':'Reds', 'zorder':0}
    
    if not isinstance(background, dict):
        background  = {'color':'k', 'alpha':0.8, 'zorder':-1}
    
    if not isinstance(JBoundary, dict):
        JBoundary   = {'linewidth':0.8, 'color':'k'}

    if not isinstance(includeData, dict):
        includeData = {'dtypes':['ground_mag', 'space_mag_fac', 'convection'],
                       'colors':['tab:blue', 'k', 'tab:green'],
                       'marker':'*', 'linestyle':'', 'alpha':0.8, 'zorder':1}
    
    # Plot spatial resolution
    _, cc = mapPlot(ax, var, model.grid_E, model, mapDict=mapDict, 
                includeData=includeData, JBoundary=JBoundary, background=background)
        
    # Add colorbar
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticklabels(cax.get_xticklabels(), fontsize=fs)
    cax.set_xlabel(unit, fontsize=fs)
    
    # legend
    lgnd = ax.legend(loc=8, bbox_to_anchor=(0.5, -0.25), fontsize=fs, ncols=2)
    for lgndhandle in lgnd.legendHandles:
        lgndhandle._markersize = fs
            
    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()
        
    if return_axes:
        return fig, ax
    else:
        return fig

def Cdplot(model, dtype, apex=None, savekw = None, return_axes = False,
           mapDict=None, background=None, JBoundary=None, includeData= None,
           unit=None, figsize=(12,16), fs=20, manScale=1):
    
    """ Produce a plot of the diagonal of the posterior model covariance projected onto the data (i.e., "posterior data uncertainty").

        The output is either a figure displayed on screen or, if savekw is given, a figure saved to disk

        parameters
        ----------
        model: lompe.model
            model that will be plotted
        apex: apexpy.Apex object, optional
            specify if you want magnetic coordinate grid instead of geographic
        savekw: dictionary, optional
            keyword arguments passed to savefig. If None, the figure will be shown with plt.show()
        return_axes: bool, optional
            Set to True to return the matplotlib figure and axes objects.
            Default is False and will only return the matplotlib figure object
        mapDict: dict, optional
            Arguments for plotting map
        background: dict, optional
            Arguments for adding background when using mask
        JBoundary: dict, optional
            Arguments for showing the extent of grid_J
        includeData, dict, optional
            Arguments for illustration of data from inversion
        unit: string, optional
            Unit of spatial resolution shown on colorbar.
        figsize: tuple, optional
            Size of figure.
        fs: int, optional
            Fontsize of text in plot.

    """
    
    if 'mag' in dtype:
        grid = model.grid_E
        if dtype == 'ground_mag':
            coords = {'lon':grid.lon.flatten(), 'lat':grid.lat.flatten(), 'r':np.ones(grid.size)*6371.2e3}
        else:
            coords = {'lon':grid.lon.flatten(), 'lat':grid.lat.flatten(), 'r':np.ones(grid.size)*model.R}
        Gs = np.split(model.matrix_func[dtype](**coords), 3, axis = 0)
        scale = 1e9
        unit = 'nT'
        components = ['B$_{e}$', 'B$_{n}$', 'B$_{r}$']
    elif dtype in ['efield', 'convection']:
        grid = model.grid_J
        coords = {'lon':grid.lon.flatten(), 'lat':grid.lat.flatten()}
        Gs = model.matrix_func[dtype](**coords)
        if dtype == 'efield':
            scale = 1e3
            unit = 'mV/m'
            components = ['E$_{e}$', 'E$_{n}$']
        else:
            scale = 1e0
            unit = 'm/s'
            components = ['v$_{e}$', 'v$_{n}$']
    elif dtype == 'fac':
        grid = model.grid_J
        coords = {'lon':grid.lon.flatten(), 'lat':grid.lat.flatten()}
        Gs = [np.vstack(model.matrix_func[dtype](**coords))]
        scale = 1e0
        unit = 'A/m$^2$'
        components = ['']
    else:
        print('No match for dtype!')
        return
    
    # Start figure
    if  len(Gs) == 1:
        fig  = plt.figure(figsize = figsize)
        axes = [plt.subplot2grid((32, 1), (0, 0), rowspan=30)]
    elif len(Gs) == 2:        
        figsize = (2*figsize[0], figsize[1])
        fig  = plt.figure(figsize = figsize)
        axes = [plt.subplot2grid((32, 2), (0, 0), rowspan=30),
                plt.subplot2grid((32, 2), (0, 1), rowspan=30)]
    elif len(Gs) == 3:
        figsize = (3*figsize[0], figsize[1])
        fig  = plt.figure(figsize = figsize)
        axes = [plt.subplot2grid((32, 3), (0, 0), rowspan=30),
                plt.subplot2grid((32, 3), (0, 1), rowspan=30),
                plt.subplot2grid((32, 3), (0, 2), rowspan=30)]
    cax  = plt.subplot2grid((32, 10), (31, 2), colspan = 6)
    
    # Format axes
    for ax in axes:
        format_ax(ax, model, apex = apex, colors='gray')
    
    # Loop over all G matrices
    for ax, G, comp in zip(axes, Gs, components):
    
        # Project the posterior model covariance into data space
        Cdpost = np.sqrt(np.diag(G.dot(model.Cmpost).dot(G.T))).reshape(grid.shape)
        Cdpost *= manScale
        Cdpost *= scale
        
        # Default plot settings
        vmin = np.min(Cdpost)
        vmax = np.max(Cdpost)
        if not isinstance(mapDict, dict):
            mapDict     = {'vmin':vmin, 'vmax':vmax, 'cmap':'Reds', 'zorder':0}
    
        if not isinstance(background, dict):
            background  = {'color':'k', 'alpha':0.8, 'zorder':-1}
    
        if not isinstance(JBoundary, dict):
            JBoundary   = {'linewidth':0.8, 'color':'k'}

        if not isinstance(includeData, dict):
            includeData = {'dtypes':['ground_mag', 'space_mag_fac', 'convection'],
                           'colors':['tab:blue', 'k', 'tab:green'],
                           'marker':'*', 'linestyle':'', 'alpha':0.8, 'zorder':1}
    
        # Plot map
        _, cc = mapPlot(ax, Cdpost, grid, model, mapDict=mapDict, 
                    includeData=includeData, JBoundary=JBoundary, background=background)    
    
        # Add title
        if not dtype == 'fac':
            ax.text(0.5, 1.02, comp, ha='center', va='bottom', fontsize=fs, transform=ax.transAxes)
    
    # Add colorbar
    cbar = fig.colorbar(cc, cax=cax, orientation="horizontal")
    cax.set_xticklabels(cax.get_xticklabels(), fontsize=fs)
    cax.set_xlabel(unit, fontsize=fs)
            
    if savekw != None:
        plt.savefig(**savekw)
    else:
        plt.show()
        
    if return_axes:
        return fig, ax
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





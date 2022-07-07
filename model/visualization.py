""" lompe visualization tools 

Lots of function to help plot the different lompe quantities
in a nice way. 

The default plotting tool is lompeplot. See documentation of that function
for more details. 

If you want more custom plots, there are many tools in this script that can 
be helpful. For example, the Polarsubplot class is good for making mlt/mlat 
plots.

"""

import matplotlib.pyplot as plt
import numpy as np
import cartopy.io.shapereader as shpreader
import apexpy
from lompe.utils.sunlight import terminator
from scipy.interpolate import griddata
from matplotlib import rc
from matplotlib.patches import Polygon, Ellipse
from matplotlib.collections import PolyCollection, LineCollection


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
    for cl in model.grid_J.projection.get_projected_coastlines(resolution = resolution):
        ax.plot(cl[0], cl[1], **kwargs)
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

    mlat, mlon = apex.geo2apex(model.grid_J.lat, model.grid_J.lon, 110)
    mlt = apex.mlon2mlt(mlon, time)
    mlat = np.linspace(mlat.min(), mlat.max(), 50)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    for mltlevel in mltlevels:
        mlon_ = apex.mlt2mlon(mltlevel, time)
        glat, glon, error = apex.apex2geo(mlat, mlon_, 0)
        iii = model.grid_J.ingrid(glon, glat)
        if np.sum(iii) > 2:
            xi, eta = model.grid_J.projection.geo2cube(glon[iii], glat[iii])
            ax.plot(xi, eta, **kwargs)
            ax.text(xi[len(xi)//2], eta[len(xi)//2], str(np.int32(mltlevel)).zfill(2), **txtkwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)



def polarplot_coastline(lon, lat, pax, apex, time, **kwargs):
    """ plot coastline on mlat / mlt polar plot

    parameters
    ----------
    lon: array
        geographic longitude of coastlines [deg]
    lat: array
        geographic latitude of coastlines [deg]
    pax: Polarsubplot
        Polarsubplot object on which to plot
    apex: apexpy.Apex object
        for conversion to magnetic apex coords
    time: datetime
        for calculation of magnetic local time
    kwargs: dict, optional
        passed to plot
    """
    if 'color' not in kwargs.keys():
        kwargs['color'] = 'lightgrey'
    if 'linewidth' not in kwargs.keys():
        kwargs['linewidth'] = 2

    mlat, mlon = apex.geo2apex(lat, lon, 110)
    mlon[mlat < 50] = np.nan
    mlat[mlat < 50] = np.nan
    if np.sum(np.isfinite(mlat)) > 2:
        mlt = np.full_like(mlon, np.nan)
        mlt[np.isfinite(mlon)] = apex.mlon2mlt(mlon[np.isfinite(mlon)], time)
        pax.plot(mlat, mlt, **kwargs)

        
def format_ax(ax, model, apex = None, **kwargs):
    """ function to format axis for plotting lompe output

    parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        axis to plot on
    model: lompe.Model object
        model to get grid from
    apex: apexpy.Apex object, optional
        If given, magnetic latitude contours will be plotted.
        If not (default), geograhpic latitude contours will be shown
    kwargs: optional
        passed to contour, which is used to plot latitude contours
    """ 

    if apex != None:
        lat, lon = apex.geo2apex(model.grid_J.lat, model.grid_J.lon, 110)
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


def polarplot(ax, model, apex, time, dV = None, clkw = None):
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
    clkw: dict, optional
        keywords for shpreader.natural_earth
    dV: int, optional
        set to an integer that represents electric potential
        step size. If not given, electric potential will not be
        shown.
    """

    pax = Polarsubplot(ax, minlat = 50)

    if clkw == None:
        clkw = {'resolution':'110m', 'category':'physical', 'name':'coastline'}

    shpfilename = shpreader.natural_earth(**clkw)
    reader = shpreader.Reader(shpfilename)
    coastlines = reader.records()
    multilinestrings = []
    for coastline in coastlines:
        if coastline.geometry.geom_type == 'MultiLineString':
            multilinestrings.append(coastline.geometry)
            continue

        lon, lat = np.array(coastline.geometry.coords[:]).T 
        polarplot_coastline(lon, lat, pax, apex, time, zorder = 1)

    for mls in multilinestrings:
        for ls in mls:
            lon, lat = np.array(ls.coords[:]).T 
            polarplot_coastline(lon, lat, pax, apex, time, zorder = 1)

    grid = model.grid_E
    xs = (grid.lon_mesh[0, :], grid.lon_mesh[-1, :], grid.lon_mesh[:, 0], grid.lon_mesh[:, -1])
    ys = (grid.lat_mesh[0, :], grid.lat_mesh[-1, :], grid.lat_mesh[:, 0], grid.lat_mesh[:, -1])
    for i, c in enumerate(zip(xs, ys)):
        lon, lat = c
        mlat, mlon = apex.geo2apex(lat, lon, 110)
        mlt = apex.mlon2mlt(mlon, time)
        pax.plot(mlat, mlt, color = 'black', linewidth = 1.5 if i == 0 else .5, zorder = 2)


    if dV != None: # plot electric potential
        V = model.E_pot().reshape(model.grid_J.shape) * 1e-3
        V = V - V.min() - (V.max() - V.min())/2
        mlat, mlon = apex.geo2apex(model.grid_J.lat, model.grid_J.lon, 110)
        mlt = apex.mlon2mlt(mlon, time)

        levels = np.r_[(V.min()//dV)*dV :(V.max()//dV)*dV + dV:dV]

        pax.contour(mlat, mlt, V, levels = levels, colors = 'C0', linewidths = 1, zorder = 3)



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
                     savekw = None, clkw = None, quiverscales = None, colorscales = None, 
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
            keywords for shpreader.natural_earth, used to show coastlines in polarplot. Ignored 
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
        polarplot(ax, model, apex, time, dV = 5, clkw = clkw)

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
                    Es = [Ge.dot(m), Gn.dot(m)]
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

            ax.text(-extent + .1, extent -.3 - counter * .25, ds.label, color = 'C' + str(counter), va = 'top', ha = 'left', size = 14)


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






class Polarsubplot(object):
    def __init__(self, ax, minlat = 50, plotgrid = True, sector = 'all', **kwargs):
        """ pax = Polarsubplot(axis, minlat = 50, plotgrid = True, **kwargs)

            this is a class which handles plotting in polar coordinates, specifically
            an MLT/MLAT grid or similar
            
            parameters
            ----------
            ax: matplotlib.axes._subplots.AxesSubplot
                axis to make into polarsubplot
            minlat: int
                lowest latitude to plot
                Default: 50
            plotgrid: bool
                plot the MLT/MLAT grid
                Default: True
            sector: str
                Which sector of the hemisphere to show (e.g. 'dusk')
                Default: 'all'
            **kwargs are the plot parameters for the grid
            
            Example:
            --------
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            pax = Polarsubplot(ax)
            pax.MEMBERFUNCTION()
            plt.show()


            where memberfunctions include:
            plotgrid()                           - called by __init__
            plot(mlat, mlt, **kwargs)            - works like plt.plot
            write(mlat, mlt, text, **kwargs)     - works like plt.text
            scatter(mlat, mlt, **kwargs)         - works like plt.scatter
            writeMLTlabels(mlat = 48, **kwargs)  - writes MLT at given mlat - **kwargs to plt.text
            plotarrows(mlats, mlts, north, east) - works like plt.arrow (accepts **kwargs too)
            contour(mlat, mlt, f)                - works like plt.contour
            contourf(mlat, mlt, f)               - works like plt.contourf

        """
        self.minlat = minlat # the lower latitude boundary of the plot
        self.ax = ax
        self.ax.axis('equal')
        self.minlat = minlat

        self.sector = sector

        if 'linewidth' not in kwargs.keys():
            kwargs['linewidth'] = .5

        if 'color' not in kwargs.keys():
            kwargs['color'] = 'lightgrey'

        if 'linestyle' not in kwargs.keys():
            kwargs['linestyle'] = '--'


        if sector == 'all':
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-1.1, 1.1)
        if sector == 'dusk':
            self.ax.set_xlim(-1.1, 0.1)
            self.ax.set_ylim(-1.1, 1.1)
        if sector == 'dawn':
            self.ax.set_xlim(-0.1, 1.1)
            self.ax.set_ylim(-1.1, 1.1)
        if sector == 'night':
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-1.1, 0.1)
        if sector == 'day':
            self.ax.set_xlim(-1.1, 1.1)
            self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_axis_off()

        self.ax.format_coord = lambda x, y: 'mlt = {:.2f}, mlat = {:.2f}'.format(*tuple(self._XYtomltMlat(x, y)[::-1]))

        if plotgrid:
            self.plotgrid(**kwargs)

    def plot(self, mlat, mlt, **kwargs):
        """ plot curve based on mlat, mlt. Calls matplotlib.plot, so any keywords accepted by this is also accepted here """

        x, y = self._mltMlatToXY(mlt, mlat)
        return self.ax.plot(x, y, **kwargs)

    def write(self, mlat, mlt, text, **kwargs):
        """ write text on specified mlat, mlt. **kwargs go to matplotlib.pyplot.text"""
        x, y = self._mltMlatToXY(mlt, mlat)

        self.ax.text(x, y, text, **kwargs)

    def scatter(self, mlat, mlt, **kwargs):
        """ scatterplot on the polar grid. **kwargs go to matplotlib.pyplot.scatter """

        x, y = self._mltMlatToXY(mlt, mlat)
        c = self.ax.scatter(x, y, **kwargs)
        return c

    def plotgrid(self, **kwargs):
        """ plot mlt, mlat-grid on self.ax """

        if self.sector == 'all':
            self.ax.plot([-1, 1], [0 , 0], **kwargs)
            self.ax.plot([0 , 0], [-1, 1], **kwargs)
            angles = np.linspace(0, 2*np.pi, 360)
        if self.sector == 'dawn':
            self.ax.plot([0, 1], [0 , 0], **kwargs)
            self.ax.plot([0, 0], [-1, 1], **kwargs)
            angles = np.linspace(-np.pi/2, np.pi/2, 180)
        if self.sector == 'dusk':
            self.ax.plot([-1, 0], [0 , 0], **kwargs)
            self.ax.plot([0 , 0], [-1, 1], **kwargs)
            angles = np.linspace( np.pi/2, 3*np.pi/2, 180)
        if self.sector == 'night':
            self.ax.plot([-1, 1], [0 , 0], **kwargs)
            self.ax.plot([0 , 0], [-1, 0], **kwargs)
            angles = np.linspace(np.pi, 2*np.pi, 180)
        if self.sector == 'day':
            self.ax.plot([-1, 1], [0 , 0], **kwargs)
            self.ax.plot([0 , 0], [ 0, 1], **kwargs)
            angles = np.linspace(0, np.pi, 180)


        latgrid = (90 - np.r_[self.minlat:90:10])/(90. - self.minlat)


        for lat in latgrid:
            self.ax.plot(lat*np.cos(angles), lat*np.sin(angles), **kwargs)

    def writeMLTlabels(self, mlat = None, degrees = False, **kwargs):
        """ write MLT labels at given latitude (default 48)
            if degrees is true, the longitude will be written instead of hour (with 0 at midnight)
        """
        if mlat is None:
            mlat = self.minlat - 2

        if degrees:
            if self.sector in ['all', 'night', 'dawn', 'dusk']:
                self.write(mlat, 0,    '0$^\circ$', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs)
            if self.sector in ['all', 'night', 'dawn', 'day']:
                self.write(mlat, 6,   '90$^\circ$', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs)
            if self.sector in ['all', 'dusk', 'dawn', 'day']:
                self.write(mlat, 12, '180$^\circ$', verticalalignment = 'bottom', horizontalalignment = 'center', **kwargs)
            if self.sector in ['all', 'night', 'dusk', 'day']:
                self.write(mlat, 18, '-90$^\circ$', verticalalignment = 'center', horizontalalignment = 'right' , **kwargs)
        else:
            if self.sector in ['all', 'night', 'dawn', 'dusk']:
                self.write(mlat, 0, '00', verticalalignment = 'top'    , horizontalalignment = 'center', **kwargs)
            if self.sector in ['all', 'night', 'dawn', 'day']:
                self.write(mlat, 6, '06', verticalalignment = 'center' , horizontalalignment = 'left'  , **kwargs)
            if self.sector in ['all', 'dusk', 'dawn', 'day']:
                self.write(mlat, 12, '12', verticalalignment = 'bottom', horizontalalignment = 'center', **kwargs)
            if self.sector in ['all', 'night', 'dusk', 'day']:
                self.write(mlat, 18, '18', verticalalignment = 'center', horizontalalignment = 'right' , **kwargs)

    def plotpins(self, mlats, mlts, north, east, rotation = 0, SCALE = None, size = 10, unit = '', color = 'black', markercolor = 'black', marker = 'o', markersize = 20, **kwargs):
        """ like plotarrows, only it's not arrows but a dot with a line pointing in the arrow direction
            
            parameters
            ----------
            mlats, mlts: float
                MLAT/MLT coordinates of pin base
            nort, east: float
                length of the pin along the north, east direction
            kwargs go to ax.plot
            
            the markers at each pin can be modified by the following keywords, that go to ax.scatter:
            marker (default 'o')
            markersize (defult 20 - size in points^2)
            markercolor (default black)

        """

        mlts = mlts.flatten()
        mlats = mlats.flatten()
        north = north.flatten()
        east = east.flatten()
        R = np.array(([[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]))

        if SCALE is None:
            scale = 1.
        else:

            if unit is not None:
                self.ax.plot([0.9, 1], [0.95, 0.95], color = color, linestyle = '-', linewidth = 2)
                self.ax.text(0.9, 0.95, ('%.1f ' + unit) % SCALE, horizontalalignment = 'right', verticalalignment = 'center', size = size)

            #self.ax.set_xlim(-1.1, 1.1)
            #self.ax.set_ylim(-1.1, 1.1)
            scale = 0.1/SCALE

        segments = []
        for i in range(len(mlats)):#mlt, mlat in zip(mlts, mlats):#mlatenumerate(means.index):

            mlt = mlts[i]
            mlat = mlats[i]

            x, y = self._mltMlatToXY(mlt, mlat)
            dx, dy = R.dot(self._northEastToCartesian(north[i], east[i], mlt).reshape((2, 1))).flatten()

            segments.append([(x, y), (x + dx*scale, y + dy*scale)])

            #self.ax.plot([x, x + dx*scale], [y, y + dy*scale], color = color, **kwargs)
        self.ax.add_collection(LineCollection(segments, **kwargs))

        if markersize != 0:
            self.scatter(mlats, mlts, marker = marker, c = markercolor, s = markersize, edgecolors = markercolor)


    def contour(self, mlat, mlt, f, **kwargs):
        """ plot contour on grid, **kwargs are given to self.ax.contour. MLT in hours - no rotation
        """

        xea, yea = self._mltMlatToXY(mlt.flatten(), mlat.flatten())

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f.flatten(), (xx, yy))

        # ... and plot
        return self.ax.contour(xx, yy, gridf, **kwargs)


    def contourf(self, mlat, mlt, f, **kwargs):
        """ plot contour on grid, **kwargs are given to self.ax.contour. MLT in hours - no rotation
        """

        xea, yea = self._mltMlatToXY(mlt.flatten(), mlat.flatten())

        # convert to cartesian uniform grid
        xx, yy = np.meshgrid(np.linspace(-1, 1, 150), np.linspace(-1, 1, 150))
        points = np.vstack( tuple((xea, yea)) ).T
        gridf = griddata(points, f.flatten(), (xx, yy))

        # ... and plot
        return self.ax.contourf(xx, yy, gridf, **kwargs)

    def fill(self, mlat, mlt, **kwargs):
        """ Fill polygon defined in mlat/mlt, **kwargs are given to self.ax.contour. MLT in hours - no rotation
        """

        xx, yy = self._mltMlatToXY(mlt.flatten(), mlat.flatten())


        # plot
        return self.ax.fill(xx, yy, **kwargs)        


    def plot_terminator(self, position, sza = 90, north = True, shadecolor = None, terminatorcolor = 'black', terminatorlinewidth = 1, shadelinewidth = 0, **kwargs):
        """ shade the area antisunward of the terminator
            
            parameters
            ----------
            position: either a scalar or a datetime object
                if scalar: interpreted as the signed magnetic colatitude of the terminator, positive on dayside, negative on night side
                if datetime: terminator is calculated, and converted to magnetic apex coordinates (refh = 0, height = 0)
            sza: int
                sza to locate terminator, used if position is datetime
            north: bool 
                True if northern hemisphere, south if not (only matters if position is datetime)
            shadecolor: str
                color of the shaded area - default None
            terminatorcolor: str
                color of the terminator
                Default: 'black'
            terminatorlinewidth: int
                width of the terminator contour
                Default: 1
            shadelinewidth: int
                width of the contour surrounding the shaded area
                Default: 0 (invisible)
            **kwargs are passed to Polygon


            Example:
            --------
            to only plot the terminator (no shade):
            plot_terminator(position, color = 'white') <- sets the shade to white (or something different if the plot background is different)


            useful extensions:
            - height dependence...
        """

        if np.isscalar(position): # set the terminator as a horizontal bar
            if position >= 0: # dayside
                position = np.min([90 - self.minlat, position])
                x0, y0 = self._mltMlatToXY(12, 90 - np.abs(position))
            else: #nightside
                x0, y0 = self._mltMlatToXY(24, 90 - np.abs(position))

            xr = np.sqrt(1 - y0**2)
            xl = -xr
            lat, left_mlt  = self._XYtomltMlat(xl, y0)
            lat, right_mlt = self._XYtomltMlat(xr, y0)
            if position > -(90 - self.minlat):
                right_mlt += 24

            x = np.array([xl, xr])
            y = np.array([y0, y0])

        else: # calculate the terminator trajectory
            a = apexpy.Apex(date = position)

            t_glat, t_glon = terminator(position, sza = sza, resolution = 3600)
            t_mlat, t_mlon = a.geo2apex(t_glat, t_glon, 0)
            t_mlt          = a.mlon2mlt(t_mlon, position)

            # limit contour to correct hemisphere:
            iii = (t_mlat >= self.minlat) if north else (t_mlat <= -self.minlat)
            if len(iii) == 0:
                return 0 # terminator is outside plot
            t_mlat = t_mlat[iii]
            t_mlt = t_mlt[iii]

            x, y = self._mltMlatToXY(t_mlt, t_mlat)

            # find the points which are closest to minlat, and use these as edgepoints for the rest of the contour:
            xmin = np.argmin(x)
            xmax = np.argmax(x)
            left_mlt = t_mlt[xmin]
            right_mlt = t_mlt[xmax]
            if right_mlt < left_mlt:
                right_mlt += 24

        mlat_b = np.full(100, self.minlat)
        mlt_b  = np.linspace(left_mlt, right_mlt, 100)
        xb, yb = self._mltMlatToXY(mlt_b, mlat_b)

        # sort x and y to be in ascending order
        iii = np.argsort(x)
        x = x[iii[::-1]]
        y = y[iii[::-1]]

        if terminatorcolor is not None:
            self.ax.plot(x, y, color = terminatorcolor, linewidth = terminatorlinewidth)
        if shadecolor is not None:    
            kwargs['color'] = shadecolor
            kwargs['linewidth'] = shadelinewidth
            shade = Polygon(np.vstack((np.hstack((x, xb)), np.hstack((y, yb)))).T, closed = True, **kwargs)
            self.ax.add_patch(shade)



    def filled_cells(self, mlat, mlt, mlatres, mltres, data, resolution = 10, crange = None, levels = None, bgcolor = None, verbose = False, **kwargs):
        """ specify a set of cells in MLAT/MLT, along with a data array,
            and make a color plot of the cells
        """

        mlat, mlt, mlatres, mltres, data = map(np.ravel, [mlat, mlt, mlatres, mltres, data])

        if verbose:
            print(mlt.shape, mlat.shape, mlatres.shape, mltres.shape)

        la = np.vstack(((mlt - 6) / 12. * np.pi + i * mltres / (resolution - 1.) / 12. * np.pi for i in range(resolution))).T
        if verbose:
            print (la.shape)
        ua = la[:, ::-1]

        vertslo = np.dstack(((90 - mlat          )[:, np.newaxis] / (90. - self.minlat) * np.cos(la),
                             (90 - mlat          )[:, np.newaxis] / (90. - self.minlat) * np.sin(la)))
        vertshi = np.dstack(((90 - mlat - mlatres)[:, np.newaxis] / (90. - self.minlat) * np.cos(ua),
                             (90 - mlat - mlatres)[:, np.newaxis] / (90. - self.minlat) * np.sin(ua)))
        verts = np.concatenate((vertslo, vertshi), axis = 1)

        if verbose:
            print( verts.shape, vertslo.shape, vertshi.shape)


        if 'cmap' in kwargs.keys():
            cmap = kwargs['cmap']
            kwargs.pop('cmap')
        else:
            cmap = plt.cm.viridis

        if levels is not None: 
            # set up a function that maps data values to color levels:
            nlevels = len(levels)
            lmin, lmax = levels.min(), levels.max()
            self.colornorm = lambda x: plt.Normalize(lmin, lmax)(np.floor((x - lmin) / (lmax - lmin) * (nlevels - 1)) / (nlevels - 1) * (lmax - lmin) + lmin)
            coll = PolyCollection(verts, facecolors = cmap(self.colornorm(data.flatten())), **kwargs)

        else:
            coll = PolyCollection(verts, array = data, cmap = cmap, **kwargs)
            if crange is not None:
                coll.set_clim(crange[0], crange[1])



        if bgcolor != None:
            radius = 2*((90 - self.minlat)/ (90 - self.minlat))
            bg = Ellipse([0, 0], radius, radius, zorder = 0, facecolor = bgcolor)
            self.ax.add_artist(bg)

        self.ax.add_collection(coll)


    def _mltMlatToXY(self, mlt, mlat):
        """ convert mlt, mlat to Cartesian """

        mlt = np.asarray(mlt)
        mlat = np.asarray(mlat)
        r = (90. - np.abs(mlat))/(90. - self.minlat)
        a = (mlt - 6.)/12.*np.pi

        return r*np.cos(a), r*np.sin(a)

    def _XYtomltMlat(self, x, y):
        """ convert x, y to mlt, mlat, where x**2 + y**2 = 1 corresponds to self.minlat """
        x, y = np.array(x, ndmin = 1), np.array(y, ndmin = 1) # conver to array to allow item assignment

        lat = 90 - np.sqrt(x**2 + y**2)*(90. - self.minlat)
        mlt = np.arctan2(y, x)*12/np.pi + 6
        mlt[mlt < 0] += 24
        mlt[mlt > 24] -= 24

        return lat.squeeze()[()], mlt.squeeze()[()]


    def _northEastToCartesian(self, north, east, mlt):
        """ convert east north vector components to Cartesian components """

        a = (mlt - 6)/12*np.pi # convert MLT to angle with x axis (pointing from pole towards dawn)

        x1 = np.array([-north*np.cos(a), -north*np.sin(a)]) # arrow direction towards origin (northward)
        x2 = np.array([-east*np.sin(a),  east*np.cos(a)])   # arrow direction eastward

        return x1 + x2



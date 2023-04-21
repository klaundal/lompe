""" Saving functionality """
import numpy as np
from scipy.interpolate import interp2d
class ArgumentError(Exception):
     pass
# dictionary for finding the functions associated with each save string
funcs = {'efield':           'E', 
         'convection':       'v', 
         'ground_mag':       'B_ground',
         'electric_current': 'j',
         'space_mag_fac':    'B_space_FAC',
         'space_mag_full':   'B_space',
         'fac':              'FAC',
         'hall':             'hall_conductance',
         'pedersen':         'pedersen_conductance',
         'secs_current':     'get_SECS_currents'}
# vectors that will be returned from each save strings corresponding function
vectors = {'efield': ['_E', '_N'],
           'convection': ['_E', '_N'],
           'ground_mag': ['_E', '_N', '_U'],
           'electric_current':['_E', '_N'],
           'space_mag_fac': ['_E', '_N', '_U'],
           'space_mag_full': ['_E', '_N', '_U'],
           'fac': [''],
           'hall': [''],
           'pedersen': [''],
           'secs_current': ['_E', '_N']}


def save_model(model, save='all', 
         time=0, file_name= False, append=True, suppress_print=False, load_kwargs={}, **kwargs):
    """
    For saving the model and/or the lompe output

    Parameters
    ----------
    model : lompe.Emodel
        lompe model object.
    save : list, optional
        string or list informing what shall be in the a xarray dataset. 
        possible stand alone strings or strings in list: 'all', 'all model', 'all output', 'model', 
        'data locations', 'efield', 'convection', 'ground_mag', 'electric_current', 'space_mag_fac',
        'space_mag_full', 'fac', 'hall', 'pedersen', 'secs_current'
        
        The default is 'all'

         result of each string:
        → 'all' will save all model information (read 'all model') and all lompe outputs (read 'all output')
        
        → 'all model' will save model amplitudes, conductance (allowing the recreation
           of the lompe Emodel object) and data locations (allowing the creation of a DummyData object that has reduced
           functionality compared to lompe Data object)
        → 'all output' will save: ['efield','convection', 'ground_mag', 'electric_current', 
                                             'space_mag_fac', 'space_mag_full','fac', 'hall','pedersen','secs_current']
            (read each item)
        → 'efield' will save the electric field using Emodel.E
        → 'convection' will save the convection using Emodel.v
        → 'ground_mag' will save the ground magnetic field using Emodel.B_ground
        → 'electric_current' will save the ionospheric currents using Emodel.j
        → 'space_mag_fac' wil save the magnetic field resulting from the 
            field aligned currents using Emodel.B_space_FAC
        → 'space_mag_fall' will save the full space magnetic field using Emodel.B_space
        → 'fac' will save the field aligned currents using Emodel.FAC
        → 'hall' will save the hall conductance using Emodel.hall_conductance,
            this will always be activated if the 'model' is saved
        → 'pedersen' will save the pedersen conductance using Emodel.pedersen_conductance.
            this will always be saved if the 'model' is saved
        → 'secs_current' will save the horizontal ionospheric currents using
            secs pole amplitudes using Emodel.get_SECS_currents
        
        read the doc strings of each relevant function for more information
        
    time : int/float/datetime/timedelta, optional
        a quantity that indentifies the time of the dataset. 
        The default is 0 and will be changed when append is True to 1+ the maximum of the existing file. 
        It is recommened to choose a value when working with multiple times.
    file_name : str/bool, optional
        A string containing the path and name of the xarray file if you wish to save using this function. 
        The default is False and no save will be made.
    append : bool, optional
        if filename is provided the current dataset will be added on to the existing dataset if it exists. 
        The default is True and the dataset will be added to the existing.
    suppress_print : bool, optional
        used to deactivate printing in the function. The default is False, printing will be made.
    load_kwargs : dict, optional
        key argmuments used in the load xarray functionality when file_name is provided. The default is {}.
    **kwargs : dict
        key arguments to be passed to the functions for calculating the lompe output. Will likely fail if 
        key argmuments differ for each function.

    Raises
    ------
    ArgumentError
        An error for when there is a problem with one of the provided arguments.

    Returns
    -------
    Dataset : xarray.Dataset
        An xarray dataset containing the requested information and information for the cubed sphere grids 
        that will allow them to be recreated.

    """
    to_save=[]
    if isinstance(save, (str, np.str_)):
        save= list([save])
    save= [s.lower() for s in save]
    if 'all' in save:
        to_save=['model', 'data_locations', 'efield', 'convection', 'ground_mag',
                 'electric_current', 'space_mag_fac', 'space_mag_full',
                 'fac', 'hall','pedersen','secs_current']
        save.remove('all')
    else:
        if 'all model' in save:
            to_save.extend(['model', 'data_locations'])
            save.remove('all model')
        if 'all output' in save:
            to_save.extend(['efield', 'convection', 'ground_mag',
                     'electric_current', 'space_mag_fac', 'space_mag_full',
                     'fac', 'hall','pedersen','secs_current'])
            save.remove('all output')
    to_save.extend(save)
    if not suppress_print:
        print(f'saving................. : {to_save}')
    from xarray import Dataset
    import warnings
    import json
    warnings.simplefilter('ignore', (RuntimeWarning))
    data_vars1= {}
    save_model= 'model' in to_save
    data_locs= 'data_locations' in to_save
    if save_model:
        to_save.remove('model')
        to_save+= ['hall', 'pedersen']
    if data_locs:
        to_save.remove('data_locations')
    to_save= np.unique(to_save)
    for dtype in to_save:
        if 'conductance' in funcs[dtype]:
            data= np.array(getattr(model, funcs[dtype])(model.grid_E.lon, model.grid_E.lat, **kwargs))
        else:
            data= np.array(getattr(model, funcs[dtype])(**kwargs), ndmin=3)
        
        try:
            data= data.reshape(-1, *model.grid_J.shape)
        except:
            data= data.reshape(-1, *model.grid_E.shape)
        vecs= vectors[dtype]
        if 'mag' in dtype:
            unit= 'Tesla'
        elif 'current' in dtype :
            unit= 'Amps/Meter'
        elif dtype=='efield':
            unit= 'Volts/Meter'
        elif dtype=='fac':
            unit= 'Amps/(Meter^2)'
        elif dtype=='convection':
            unit= 'Meter/Second'
        elif 'conductance' in funcs[dtype]:
            unit= 'siemens'
        else:
            raise ArgumentError(f'to save string not known: {dtype}\nknown strings are: {["efield", "convection", "ground_mag","electric_current", "space_mag_fac", "space_mag_full","fac", "hall","pedersen","secs_current"]}')
        if data[0].size == model.grid_E.xi.size:
            grid= model.grid_E
            data_vars1.update({f'{dtype}{vec}': (['time', 'eta', 'xi'], [data[i].reshape(grid.shape)], 
                                                  {'units':unit}) for i, vec in enumerate(vecs)})

        else:
            grid = model.grid_J
            data_vars1.update({f'{dtype}{vec}': (['time', 'eta', 'xi'], 
                                                  [(interp2d(model.grid_J.xi, model.grid_J.eta, 
                                                            data[i].reshape(grid.shape))(model.grid_E.xi[0, :], 
                                                                                         model.grid_E.eta[:, 0])).reshape(model.grid_E.shape)],
                                                                                         {'units':unit,
                                                                                          'description':'has been linearly interpolated from default grid: "grid_E" to other grid "grid_J" using scipy.interpolate.interp2d'}) \
                               for i, vec in enumerate(vecs)})

    if save_model:
        data_vars1.update({'model_vector':(['time', 'eta', 'xi'], [model.m.reshape(model.grid_E.shape)],
                                  {'units':'Coulomb/Meter',
                                   'description':'model vector (the m attribute of the lompe model object) must be flattened before using as model attribute'})})                                                                  
    if data_locs: data_vars1.update(data_locs_to_dict(model))

    coords1= {'xi': model.grid_E.xi[0, :], 'eta': model.grid_E.eta[:, 0],
             'lon':(['xi', 'eta'], model.grid_E.lon.T),
             'lat':(['xi', 'eta'], model.grid_E.lat.T),
             'time':[time]}
    ds1=Dataset(data_vars=data_vars1, coords=coords1)
    ds1.attrs['Data_locs']= json.dumps(data_locs)
    ds1.attrs['Epoch']= model.epoch
    ds1.attrs['Dipole']=json.dumps(model.dipole)
    ds1.attrs['grid_info_E']=json.dumps(model.grid_E.to_dictionary())
    ds1.attrs['grid_info_J']=json.dumps(model.grid_J.to_dictionary())
    
    if file_name:
        import os
        from xarray import load_dataset, concat
        if os.path.isfile(file_name):
            if append:
                ds=load_dataset(file_name, **load_kwargs)
                if time==0:
                    ds1['time']= np.array([ds.time.max()+1])
                ds1= concat([ds, ds1], dim='time')
        ds1.to_netcdf(file_name)
            
    return ds1


def interp(grid, conductance, lon, lat):
    """
    Interpolation wrapper to allow hall and pedersen conductance loaded from 
    file to be used in the same way as before in the model object. Caution the
    methods will be not be exactly the same as this relies on linear interpolation
    using scipy.griddaata. If lon and lat are on the same grid the original
    conductance values are returned

    Parameters
    ----------
    grid : cubedsphere.CSgrid
        Cubed sphere grid on which the conductance values are evaluated on.
    conductance : 
        Conductance values to be interpolated.
    lon : numpy.ndarray
        numpy array containing the longitude values where the conductance is to be
        evaluated at.
    lat : numpy.ndarray
        numpy array containing the latitude values where the conductance is to be
        evaluated at.

    Raises
    ------
    ArgumentError
        An error for when there is a problem with one of the provided arguments.
    Returns
    -------
    conductance : numpy.ndarray
        conductance on the desired grid.

    """
    from scipy.interpolate import griddata
    if lon.size==grid.xi.size:
        if lon.ndim==1:
            return conductance.flatten()
        else:
            return conductance
    return griddata((grid.xi.flatten(), grid.eta.flatten()), conductance.flatten(), grid.projection.geo2cube(lon, lat))


class DummyData(object):
    def __init__(self, coordinates, data_type):
        """
        Generating a dummy data object that can function similar to the lompe Data object using data location values
        read from file. It has limited functionality and will raise an attribute 
        error when using functions that require the full data object

        Parameters
        ----------
        coordinates : dictionary
            a dictionary containing the longitude and lattitude of the data points must be formatted like:
                {'lon':longitude_values, 'lat':latitude_values}.
        data_type : str
            a string denoting the data_type functions the same as data_type argument 
            in lompe.Data object refer to doc string for additional information.
        Raises
        ------
        AtrributeError
            Has a message that the missing attribute may be due to the reduced functionality
            of the DummyData object compared to the lompe Data object.
        Returns
        -------
        None.

        """
        self.coords = coordinates
        self.data_type= data_type
        
        # Altering the getattr function to allow message to be added
        self.__getattributeoriginal__= self.__getattribute__
        self.__getattribute__= self.__getattributeCheck__
    def __getattr__(self, attribute):
        """
        Altering the getattr function 

        Parameters
        ----------
        attribute : str
            str of attribute name.

        Returns
        -------
            default getattr return
        """
        return self.__getattributeCheck__(attribute)
    def __getattributeCheck__(self, attribute):
        """
        

        Parameters
        ----------
        attribute : str
            str of attribute name.
            
        Raises
        ------
        AtrributeError
            Has a message that the missing attribute may be due to the reduced functionality
            of the DummyData object compared to the lompe Data object.

        Returns
        -------
            default getattr return

        """
        try: 
            return self.__getattributeoriginal__(attribute)
        except AttributeError:
            raise AttributeError(f'Attribute does not exist likely because data has been loaded from file so only coordinates are provided. Attribute used: {attribute}')





def data_locs_to_dict(model):
    """
    Converting the data locations and labels from the lompe Data object
    into a dictionary that is set up to be placed in a xarray Dataset.
    Primary use is in the save function.

    Parameters
    ----------
    model : lompe.Emodel
        Lompe model object to have data location and labels to be saved.

    Returns
    -------
    data_vars : dictonary
        dictionary that is to be used in xarray dataset, only dimension being used is time.
        locations are converted to bytes due to possible differences in array shapes and sizes for
        different times.

    """
    data_vars= {}
    for dtype in model.data.keys():
        coords=[]
        dtypes= []
        labels= []
        for ds in model.data[dtype]:
            coords.append([ds.coords[key] for key in ['lon', 'lat']])
            dtypes.append(dtype)
            labels.append(ds.label)
        if len(coords):
            coords= np.array(coords)
            data_vars.update({dtype+'_input_locations': (['time'],
                                      [np.array([np.concatenate([c for c in coords[:,0]]), np.concatenate([c for c in coords[:,1]])]).tobytes()],
                              {'labels': '\t'.join([f'{label} length: {len(c)}' for label, c in zip(labels, coords[:,0])])})})
        
            
    return data_vars
def load_model(file, time='first'):
    """
    

    Parameters
    ----------
    file : str/xarray.Dataset
        string with path and filename of xarray Dataset or an xarray Dataset to generate a model object from .
    time : int/float/datetime/timedelta, optional
        time to be loaded must match type of time array in dataset. 
        The default is 'first' which load the dataset occuring at the
        lowest value of time in the dataset.

    Raises
    ------
    ArgumentError
        raise if the file argument is not a valid type. i.e not an a dataset or string.

    Returns
    -------
    model : lompe.Emodel
        A lompe model object loaded from the values in the Dataset.

    """
    import xarray as xr
    from lompe.model import Emodel
    from secsy import cubedsphere as cs
    from functools import partial
    import json
    
    if isinstance(file, (str, np.str_)):    
        ds= xr.load_dataset(file)
    elif isinstance(file, (xr.Dataset)):
        ds= file
    else:
        raise ArgumentError(f'input not understood please provide either string or dataset, you provided: {type(file)}')
    if 'hall' not in ds or 'pedersen' not in ds or 'model_vector' not in ds:
        raise ArgumentError('data set does not include: '+''.join([param for param in  ['hall', 'pedersen', 'model_vector'] if param not in ds])+\
                            ' they are required to recreate the model object please include these parameters when using the save function')
    if time=='first':
        time= ds.time.values.min()
    ds= ds.sel(time=time)
    grid= cs.from_dictionary(json.loads(ds.attrs['grid_info_E']))
    model=Emodel(cs.from_dictionary(json.loads(ds.attrs['grid_info_J'])),[partial(interp, grid, ds['hall'].values), 
                                        partial(interp, grid, ds['pedersen'].values)], epoch= float(ds.Epoch),
                 dipole= json.loads(ds.Dipole))
    model.m= ds.model_vector.values.flatten()
    if json.loads(ds.attrs['Data_locs']):
        for dtype in ['efield', 'convection', 'ground_mag', 'space_mag_full', 'space_mag_fac', 'fac']:
            if dtype+'_input_locations' in ds:
                lon, lat= np.frombuffer(ds[dtype+'_input_locations'].values).reshape(2, -1)
                model.data[dtype].append(DummyData({'lon':lon, 'lat':lat}, dtype))
                
    return model
def load_grid(file):
    """
    

    Parameters
    ----------
    file : str/xarray.Dataset
        string with path and filename of xarray Dataset or an xarray Dataset to generate a model object from .

    Raises
    ------
    ArgumentError
        raise if the file argument is not a valid type. i.e not an a dataset or string.

    Returns
    -------
    grid_J : cubespehre.CSgrid
        A cubedsphere grid object loaded from the attributes in the Dataset.
    grid_E : cubespehre.CSgrid
        A cubedsphere grid object loaded from the attributes in the Dataset.

    """
    import xarray as xr
    from lompe.model import Emodel
    from secsy import cubedsphere as cs
    from functools import partial
    import json
    
    if isinstance(file, (str, np.str_)):    
        ds= xr.load_dataset(file)
    elif isinstance(file, (xr.Dataset)):
        ds= file
    else:
        raise ArgumentError(f'input not understood please provide either string or dataset, you provided: {type(file)}')
    if 'grid_info_J' not in ds.attrs or 'grid_info_E' not in ds.attrs:
        raise ArgumentError('No grid information store in dataset attributes')
    return cs.from_dictionary(json.loads(ds.attrs['grid_info_J'])), cs.from_dictionary(json.loads(ds.attrs['grid_info_E']))
""" helper functions - kept here to avoid clutter in model script """
import numpy as np
from functools import wraps
from itertools import combinations
import inspect
import warnings

RE = 6371.2e3


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def get_default_args(func):
    """ return dictionary of default arguments to func """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def extrapolation_check(func):
    """ checks if the coordinates provided are inside or outside the model
        vector grid
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        argnames = inspect.getfullargspec(func)[0]
        
        model = args[0]
        argdict = get_default_args(func) # start by collecting defaults
        l = len(args)
        if l > 1:
            named_args = dict(zip(argnames[1:l], args[1:]))
            argdict.update(named_args) # add positional arguments
        argdict.update(kwargs)
        if argdict['lat'] is None:
            return func(model, **argdict)

        # creates warning if there are points outside the grid
        if not (func.__name__.startswith('_') and func.__name__.endswith('matrix')) and not np.all(model.grid_E.ingrid(argdict['lon'], argdict['lat'])):
            warnings.warn('Some points of evaluation are outside the grid and are therefore poorly informed', UserWarning)
        
        params = ['r', 'lat', 'lon'] if 'r' in argdict.keys() else ['lat', 'lon']
        shape = np.broadcast(*(argdict[key] for key in params)).shape



        return func(model, **argdict)
    return wrapper

def check_input(func):
    """ checks that the inputs to get_matrix functions in Model object
        are consistent. The purpose of putting these tests here is to 
        make the get_matrix functions more readable

        The function checks that all required parameters are passed and 
        that they have the correct shape.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):

        argnames = inspect.getfullargspec(func)[0]
        if argnames[0] != 'self':
            raise Exception("check_input: 'self' missing from function {}".format(func.__name__))
        
        model = args[0]
        if type(model).__name__ not in ['Cmodel', 'Emodel', 'Cmodel2']:
            raise Exception('Function {} is not part of model class'.format(func.__name__)) 

        # build dictionary of arguments that are passed to function:
        argdict = get_default_args(func) # start by collecting defaults

        l = len(args)
        if l > 1:
            named_args = dict(zip(argnames[1:l], args[1:]))
            argdict.update(named_args) # add positional arguments

        argdict.update(kwargs) # add keyword arguments

        # list of missing arguments:
        missing = [name for name in argnames[1:] if name not in argdict.keys()]
        if len(missing) > 0: # raise excption if anything is missing
            raise Exception('check_input: parameters {} missing from call to {}'.format(missing, func.__name__)) 

        # set default values
        if argdict['lat'] is None:
            argdict['lat'] = model.lat_E if ('B' in func.__name__ or 'get_SECS_currents' in func.__name__) else model.lat_J
        if argdict['lon'] is None:
            argdict['lon'] = model.lon_E if ('B' in func.__name__ or 'get_SECS_currents' in func.__name__) else model.lon_J
        if 'r' in argdict.keys() and argdict['r'] is None:       
            argdict['r'] = 2 * model.R - RE if 'cf' in func.__name__ else RE
        # list of parameters passed to func
        params = ['r', 'lat', 'lon'] if 'r' in argdict.keys() else ['lat', 'lon']

        # ensure that they are arrays
        for key in params:
            argdict[key] = np.array(argdict[key], ndmin = 1)

        # ensure that shapes are consistent:
        if not all([is_broadcastable(argdict[keys[0]].shape, argdict[keys[1]].shape) for keys in combinations(params, 2)]):
            raise Exception('Not all parameters {} passed to {} are broadcastable'.format(params, func.__name__))

        shape = np.broadcast(*(argdict[key] for key in params)).shape

        # flatten parameters
        for key in params:
            argdict[key] = np.ravel(np.array(argdict[key]))

        # ensure that coordinates make sense
        if 'r' in params:
            if np.any(argdict['r'] < model.R * .5):
                raise Exception('radii passed to {} are too low to make sense'.format(func.__name__))
        xi, eta = model.grid_E.projection.geo2cube(argdict['lon'], argdict['lat'])

        # all done - return with shape or not
        if 'return_shape' in argdict.keys() and argdict['return_shape']:
            output = func(model, **argdict)
            if isinstance(output, tuple): # more than one variable returned by func
                return output + (shape,)
            else: # only one variable returned:
                return output, shape
        else:
            return func(model, **argdict)


    return wrapper




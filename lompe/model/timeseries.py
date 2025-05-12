#%% Import

from .data import Data

#%%

class TimeSeries(object):
    def __init__(self, values, coordinates = None, LOS = None, components = 'all', datatype = 'none', label = None, scale = None, iweight = None, error = 0):

        nt = len(values)        

        if not isinstance(coordinates, list):
            coordinates = [coordinates]*nt

        if not isinstance(LOS, list):
            LOS = [LOS]*nt

        if coordinates is None:
            coordinates = [None]*nt
        
        if LOS is None:
            LOS = [None]*nt
        
        if iweight is None:
            iweight []

        for val, coo in zip()
        Data(val, coo, )
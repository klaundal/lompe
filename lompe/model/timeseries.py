#%% Import

import numpy as np
from .data import Data
from typing import Union, Optional, Literal

#%%

class TimeSeries(object):
    def __init__(self,
                 values: Union[np.ndarray, list[np.ndarray]],
                 coordinates: Union[np.ndarray, list[np.ndarray]],
                 times: Optional[Union[np.ndarray, list[int], list[float], int, float]] = None, 
                 LOS: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
                 components: Literal['all', '1', '2', '3', '12', '13', '23'] = 'all',
                 datatype: str = 'none',
                 label: Optional[str] = None,
                 iweight: Optional[Union[float, int, list[int], list[float]]] = None,
                 error: Optional[Union[float, int, list[int], list[float]]] = 0
                 ):
              
        self.label = datatype if label is None else label        
        self.datatype = datatype
        
        values, coordinates, nt = TimeSeries.check_vc(values, coordinates)
        
        self.nt = nt
        
        if times is None:
            times = list(np.arange(self.nt))
        if isinstance(times, int) or isinstance(times, float):
            times = list(np.arange(0, self.nt*times, times))
        if len(times) != self.nt:
            raise ValueError('times and values do not have the same length')
        self.times = times
        
        if LOS is not None:
            LOS, ntL = TimeSeries.check_n(LOS, 'LOS')
            if ntL is None:
                LOS = [LOS]*self.nt
                ntL = self.nt
            if ntL != self.nt:
                raise ValueError('Mismatch of temporal dimension between values/coordinates and LOS.')
            del ntL        
            # Check that number of entries is the same for all timesteps
            bad = [coordinates[i].shape[1] != LOS[i].shape[1] for i in range(self.nt)]
            if any(bad):
                raise ValueError(f'{np.sum(bad)} instances in values/coordinates and LOS do not have the same length.')
        else:
            LOS = [LOS]*self.nt
        
        if isinstance(iweight, list):
            if len(iweight) != self.nt:
                raise ValueError('If iweight is list is has to have same length as values.')
        else:
            iweight = [iweight]*self.nt
        
        if isinstance(error, list):
            if len(error) != self.nt:
                raise ValueError('If error is list is has to have same length as values.')
        else:
            error = [error]*self.nt
        
        self.data = [None]*self.nt
        for i in range(self.nt):
            self.data[i] = Data(values=values[i], coordinates=coordinates[i], 
                                LOS=LOS[i], components=components,
                                datatype=self.datatype, label=self.label,
                                iweight=iweight[i], error=error[i])

    def get_t_subset(self, 
                     t: Union[int, float], 
                     method: Optional[Literal['nearest']] = 'nearest'):
        # TODO: Add more methods
        if method != 'nearest':
            raise ValueError('Only method=nearest is implemented.')
        
        if (t < np.min(self.times)) or (t > np.max(self.times)):
            print(f'Timeseries (label={self.label} and datatype={self.datatype}) has temporal range of {self.times.min()}-{self.times.max()}. However, evaluation at t={t} requested. Using nearest time step.')
        if t < np.min(self.times):
            tid = 0
        if t > np.max(self.times):
            tid = self.nt-1
        
        tid = np.argmin(abs(np.array(self.times) - t))
        
        return self.data[tid]

#%% Sanity check
# Most checks are done in the Data class.
    @staticmethod
    def check_vc(values: Union[np.ndarray, list[np.ndarray]], 
                 coordinates: Union[np.ndarray, list[np.ndarray]]):
        # values sanity check
        values, ntv = TimeSeries.check_n(values, 'values')
        
        # coordinates sanity check
        coordinates, ntc = TimeSeries.check_n(coordinates, 'coordinates')
        
        # Check temporal length of values
        if ntv is None:
            raise ValueError('Only one timestep provided. Use Data and not TimeSeries')
        
        # check temporal length of cooridinates
        if ntc is None:
            print('coordinates is 2D and will be repeated for all time steps.')
            coordinates = [coordinates]*ntv
            ntc = ntv
                
        # Same temporal length
        if ntv != ntc:
            raise ValueError('Mismatch of temporal dimension between values and coordinates.')
        nt = ntv
        
        # Check that number of entries is the same for all timesteps
        if len(values[0].shape) == 1:
            bad = [values[i].size != coordinates[i].shape[1] for i in range(nt)]
        else:
            bad = [values[i].shape[1] != coordinates[i].shape[1] for i in range(nt)]
        if any(bad):
            raise ValueError(f'{np.sum(bad)} instances in values and coordinates do not have the same length.')
        
        return values, coordinates, nt

    @staticmethod
    def check_n(var: Union[np.ndarray, list[np.ndarray]], 
                name: Optional[str] = 'NA'):
        if isinstance(var, list):
            ntn = len(var)
            #if any([len(var[i].shape) != 2 for i in range(ntn)]): # Not true if only 1 comp is provided or if SD Vlos is used.
            #    raise ValueError(f'The individual arrays in {name} have to be 2D.')
        
        if isinstance(var, np.ndarray):
            dim = len(var.shape)
            if dim not in (2, 3):
                raise ValueError(f'If {name} is np.ndarray it has to be 2D or 3D.')
            if (dim == 3):
                ntn = var.shape[0]
                var = [var[i] for i in range(ntn)]
            else:
                ntn = None
            
        return var, ntn

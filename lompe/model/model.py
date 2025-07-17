""" Model class """
import numpy as np
from .design import Design
from .gridhandler import GridHandler
from .solver import Solver
from .evaluator import Evaluator
from .regularizer import Regularizer
from .timeseries import TimeSeries
from .kalmanfilter import KalmanFilter
from typing import Union, Optional, Tuple, Callable
from tqdm import tqdm

import scipy
from copy import deepcopy as dcopy

#%%

RE = 6371.2e3 # Earth radius in meters

#%%

class Emodel(object):
    def __init__(self, grid,
                       Hall_Pedersen_conductance: Optional[Union[Tuple[Callable, Callable]]] = None,
                       Hall_Pedersen_conductance_t: Optional[list[Tuple[Callable, Callable]]] = None,
                       times: Optional[Union[list[int], list[float], int, float]] = None,
                       epoch: Optional[Union[int, float]] = 2015., # epoch, decimal year, used for IGRF dependent calculations
                       dipole: Optional[bool] = False, # set to True to use dipole field and dipole coords
                       perfect_conductor_radius: Optional[float] = None,
                       ew_regularization_limit: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
                       perimeter_width: Optional[int] = 10,
                       data_density_weight: Optional[bool] =True):
        """
        Electric field model

        Example
        -------
        grid = cs.CSgrid(*gridparams)

        model = lompe.Emodel(grid, (Hall_function, Pedersen_function))

        model.add_data(my_Efield_dataset, datatype = 'Efield')
        model.add_data(my_ground_B_dataset, datatype = 'ground_mag')
        model.run_inversion()

        lompeplot(model, include_data = True) # plot all parameters


        Parameters
        ----------
        grid: CSgrid
            cubed sphere grid
        Hall_Pedersen_conductance: tuple of functions
            provide a tuple of functions of lon, lat that returns
            Hall and Pedersen conductances, respectively.
        epoch: float, optional
            Decimal year, used in calculations of IGRF magnetic field and in
            calculation of magnetic coordinates. Set to 2015. by default
        dipole: bool or float, optional
            Set to True to use dipole magnetic field instead of IGRF. If True, all
            coords are assumed to be dipole coordinates. Useful for idealized calculations.
            Default is False
        perfect_conductor_radius: float, optional
            An option for different handling of ground induced currents. This keyword can be used to specify
            the radius (< grid.R) of a spherical shell that is a perfect conductor, at which induced currents 
            in the ground cancel Br from space currents (Juusola et al. 2016 doi:10.1002/2016JA022961). If 
            set to None (default), ground delta B will be modeled exclusively in terms of space currents
        ew_regularization_limit: tuple, optional
            Specify a tuple of two latitudes between which the east-west regularization term is
            reduced to zero towards the magnetic pole. The motivation for this is that east-west 
            regularization is not appropriate in the polar cap, and it might be better to turn it off there
        """
        # Check is time-dependent modeling is expected
        if Hall_Pedersen_conductance is None and Hall_Pedersen_conductance_t is None:
            raise ValueError('Hall_Pedersen_conductance and Hall_Pedersen_conductance_t cannot both be None.')
        if Hall_Pedersen_conductance is not None and Hall_Pedersen_conductance_t is not None:
            raise ValueError('Hall_Pedersen_conductance and Hall_Pedersen_conductance_t cannot both be defined.')        
        if Hall_Pedersen_conductance_t is None:
            self.time_dependent_modeling = False
        else:
            self.time_dependent_modeling = True
        
        # Various global parameters
        self.perfect_conductor_radius = perfect_conductor_radius
        self.dipole = dipole
        self.epoch = epoch
        self.use_gpu = False
        self.perimeter_width=perimeter_width
        
        # Calculate spatial weights
        self.data_density_weight = data_density_weight
        
        # Steady-state conductance functions
        self.hall_conductance = None
        self.pedersen_conductance = None
        
        # Time-dependent conductance functions
        self.ntc = None
        self.timesc = None
        self.hall_conductance_t = None
        self.pedersen_conductance_t = None
        
        # various Lompe classes
        self._builder = None
        self._solver = None
        self._ev = None
        self.gH = GridHandler(grid)

        # Regularization
        self.reg = None
        
        # Design matrices
        self._G_CF = None
        self._G_DF = None
        
        # Data
        self.data = {'efield':[], 'convection':[], 'ground_mag':[], 'db_ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}
        self.timeseries = {'efield':[], 'convection':[], 'ground_mag':[], 'db_ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}
        self._d = None
        
        # Data weights
        self._w = None
        self._sw = None
        self._iweight = None
        self._Cdinv = None
        self.rescale_iweights = True
        
        # Kalman
        self.nta = None
        self.A = None
        
        # Epot and Eind model parameters
        self.m_CF = None
        self.m_DF = None
        self.m_CF_t = None
        self.m_DF_t = None
        self.timesm = None # Array of time were the model will be evaluated
        self.m_CF_SS = None
        
        self.n_CF = self.gH.size_E
        self.n_DF = self.gH.size_E
        
        # Posterior coviariance matrix
        # TODO: Introduce Cmpost_CF and Cmpost_DF
        self.Cmpost = None
        
        # Define conductance functions and initiate Desgin
        self.clear_model(Hall_Pedersen_conductance = Hall_Pedersen_conductance,
                         Hall_Pedersen_conductance_t = Hall_Pedersen_conductance_t,
                         times = times,
                         perimeter_width=self.perimeter_width)
      
#%% Utils

    def clear_model(self, 
                    Hall_Pedersen_conductance: Optional[Tuple[Callable, Callable]] = None, 
                    Hall_Pedersen_conductance_t: Optional[list[Tuple[Callable, Callable]]] = None,
                    times: Optional[Union[list[int], list[float], int, float]] = None,
                    perimeter_width: Optional[int] = None,
                    reset_reg: bool = True):
        """ Reset data and model vectors

        parameters
        ----------
        Hall_Pedersen_conductance: tuple, optional
            provide a tuple of functions of lat, lon that returns
            Hall and Pedersen conductances, respectively. If not provided,
            the previous conductance model is kept
        """
        self.m_CF = None # clear electric field model parameters
        self.m_DF = None

        # dictionary of lists to store datasets in
        self.reset_data()

        # Hall and Pedersen conductance functions:
        if Hall_Pedersen_conductance is not None:
            _h, _p = Hall_Pedersen_conductance            
            self.hall_conductance     = lambda lon = self.gH.grid_J.lon, lat = self.gH.grid_J.lat: _h(lon, lat)
            self.pedersen_conductance = lambda lon = self.gH.grid_J.lon, lat = self.gH.grid_J.lat: _p(lon, lat)
        
        # Multiple Hall and Pedersen conductance functions for time-dependent modeling:
        if Hall_Pedersen_conductance_t is not None:
            self.ntc = len(Hall_Pedersen_conductance_t)
            self.hall_conductance_t = [None]*self.ntc
            self.pedersen_conductance_t = [None]*self.ntc
            for i, (h, p) in enumerate(Hall_Pedersen_conductance_t):
                self.hall_conductance_t[i]     = lambda lon = self.gH.grid_J.lon, lat = self.gH.grid_J.lat, _h=h: _h(lon, lat)
                self.pedersen_conductance_t[i] = lambda lon = self.gH.grid_J.lon, lat = self.gH.grid_J.lat, _p=p: _p(lon, lat)
        
            # Only do anything with conductance time if t conductance is changed
            if times is None:
                self.timesc = list(np.arange(self.ntc))
            elif isinstance(times, list):
                self.timesc = times
            else:
                self.timesc = np.arange(0, self.ntc*times, times)            

        self.reset_builder()
        self.reset_solver()
        if reset_reg:
            self.reset_regularization()
        self.reset_processed_data()
        self.reset_G()
        self.reset_evaluator()
        
        if perimeter_width is not None:
            self.change_parimeter_width(perimeter_width)

    @property
    def matrix_func_CF(self):
        return {
            'db_ground_mag':    self.builder._Br2Bg,
            'ground_mag':       self.builder._B_df_matrix_CF,
            'space_mag_full':   self.builder._B_cf_df_matrix_CF,
            'space_mag_fac':    self.builder._B_cf_matrix_CF,
            'efield':           self.builder._E_matrix_CF,
            'convection':       self.builder._v_matrix_CF,
            'fac':              self.builder.FAC_matrix_CF
            }

    @property
    def matrix_func_DF(self):
        return {            
            'ground_mag':       self.builder._B_df_matrix_DF,
            'space_mag_full':   self.builder._B_cf_df_matrix_DF,
            'space_mag_fac':    self.builder._B_cf_matrix_DF,
            'efield':           self.builder._E_matrix_DF,
            'convection':       self.builder._v_matrix_DF,
            'fac':              self.builder.FAC_matrix_DF
            }

#%% Data

    def add_data(self, *datasets):
        
        for dataset in datasets:
            if not dataset.isvalid:
                raise Exception('invalid dataset')

            dtype = dataset.datatype.lower()

            if dtype in self.data.keys():
                self.data[dtype].append(dataset)
            else:
                print('You passed {}, which is not in {} - ignored'.format(dtype, list(self.data.keys())))
            
    @property
    def d(self):
        if self._d is None:
            self.process_data()
        return self._d

    @property
    def w(self):
        if self._w is None:
            self.process_data()
        return self._w

    @property
    def sw(self):
        if self._sw is None:
            self.process_data()
        return self._sw

    @property
    def iweight(self):
        if self._iweight is None:
            self.process_data()
        return self._iweight

    @property
    def Cdinv(self):
        if self._Cdinv is None:
            self._Cdinv = self.process_data()
        return self._Cdinv

    def process_data(self):
        
        if not any(self.data.values()):
            raise ValueError('No data has been added...')
        
        self.reset_processed_data()
        
        self._d         = np.empty( 0) 
        self._w         = np.empty( 0)
        self._sw        = np.empty( 0)
        self._iweight   = np.empty( 0)
        self._Cdinv     = np.empty( 0)
        
        if self.gH.grid_d is None:
            self.gH.create_data_grid(self.perimeter_width)
        
        iweights = []
        for dtype in self.data.keys(): # loop through data types
            for ds in self.data[dtype]: # loop through the datasets within each data type
                iweights.append(ds.iweight)
        
        if np.max(iweights) != 1 and self.rescale_iweights:
            print('The provided iweights were re-scaled so max(iweights)=1')
            iweights = np.array(iweights)/np.max(iweights)
        
        ii = 0
        for dtype in self.data.keys(): # loop through data types
            for ds in self.data[dtype]: # loop through the datasets within each data type
                # skip data points that are outside biggrid:
                ds = ds.subset(self.gH.grid_d.ingrid(ds.coords['lon'], ds.coords['lat']))
                if ds.coords['lat'].size > 1: #If there is data inside biggrid                                    
                    # calculate weights based on data density:
                    if self.data_density_weight:
                        bincount = self.gH.grid_d.count(ds.coords['lon'], ds.coords['lat'])
                        i, j = self.gH.grid_d.bin_index(ds.coords['lon'], ds.coords['lat'])
                        spatial_weight = 1. / np.maximum(bincount[i, j], 1)
                        spatial_weight[i == -1] = 1
                        if ds.values.ndim == 2: # stack weights for each component in dataset.values:
                            spatial_weight = np.tile(spatial_weight, ds.values.shape[0])
                    else:
                        spatial_weight = np.ones(ds.values.size)

                    dimensions = np.array(ds.values, ndmin = 2).shape[0]
                    if np.array(ds.error, ndmin=2).shape[0]==1:
                        error = np.tile(ds.error, dimensions)
                    else: #error is different for different components
                        error = ds.error.flatten()

                    self._d = np.hstack((self._d, np.hstack(ds.values)))
                    self._sw = np.hstack((self._sw, spatial_weight))
                    self._iweight = np.hstack((self._iweight, np.ones(ds.values.size)*iweights[ii]))
                    self._Cdinv = np.hstack((self._Cdinv, 1/(error**2)))
                    ii += 1
        self._w = self._sw * self._iweight * self._Cdinv # Combined error

    def change_parimeter_width(self, perimeter_width=10):
        if self.perimeter_width == perimeter_width:
            print('Perimeter was not changed due to identical values')
        else:
            self.perimeter_width = perimeter_width
            self.gH.grid_d = None
            self.reset_processed_data()
    
    def reset_data(self):
        self.data = {'efield':[], 'convection':[], 'ground_mag':[], 'db_ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}
    
    def reset_processed_data(self):
        self._d = None
        self._sw = None
        self._iweight = None
        self._Cdinv = None
        self._w = None

#%% TimeSeries

    def add_timeseries(self, 
                       *tseriess: TimeSeries):
        for tseries in tseriess:
            for dataset in tseries.data:
                if not dataset.isvalid:
                    raise Exception('invalid dataset')

            dtype = tseries.datatype.lower()

            if dtype in self.timeseries.keys():
                self.timeseries[dtype].append(tseries)
            else:
                print('You passed {}, which is not in {} - ignored'.format(dtype, list(self.timeseries.keys())))

    def add_timeseries_subset(self, 
                      t: Union[int, float],
                      reset: bool = True):
        
        if reset: # These will also be reset if self.clear_model() is used
            self.reset_data()
            self.reset_processed_data()
        
        for dtype in self.timeseries.keys(): # loop through data types            
            for timeseries in self.timeseries[dtype]: # loop through the datasets within each data type
                # Fetch and add desired time-steps
                self.add_data(timeseries.get_t_subset(t))

    def reset_timeseries(self):
        self.timeseries = {'efield':[], 'convection':[], 'ground_mag':[], 'db_ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}

#%% Forward problem

    @property
    def G_CF(self):
        if self._G_CF is None:
            self._G_CF = self.compute_design(comp='CF')
        return self._G_CF
    
    @property
    def G_DF(self):
        if self._G_DF is None:
            self._G_DF = self.compute_design(comp='DF')
        return self._G_DF
    
    def compute_design(self, comp='CF'):

        if not any(self.data.values()):
            raise ValueError('No data has been added...')
        
        self.reset_builder()
        G = np.empty((0, self.gH.size_E))
        
        if self.gH.grid_d is None:
            self.gH.create_data_grid(self.perimeter_width)
        
        for dtype in self.data.keys(): # loop through data types
            if comp == 'CF':
                builder_func = self.matrix_func_CF.get(dtype)
            else:
                builder_func = self.matrix_func_DF.get(dtype)
            
            for ds in self.data[dtype]: # loop through the datasets within each data type
                # skip data points that are outside biggrid:
                ds = ds.subset(self.gH.grid_d.ingrid(ds.coords['lon'], ds.coords['lat']))
                if ds.coords['lat'].size > 1: #If there is data inside biggrid                
                    if 'mag' in dtype:
                        Gs = np.split(builder_func(**ds.coords), 3, axis = 0)
                        _G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])
                    if dtype in ['efield', 'convection']:
                        Gs = builder_func(**ds.coords)
                        _G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])
                    if dtype == 'fac':
                        _G = np.vstack(builder_func(**ds.coords))
                        
                    if (dtype in ['convection', 'efield']) & (ds.los is not None): # deal with line of sight data:
                        Ge, Gn = np.split(_G, 2, axis = 0)
                        _G = Ge * ds.los[0].reshape((-1, 1)) + Gn * ds.los[1].reshape((-1, 1))

                    G = np.vstack((G, _G))
        return G

    def reset_G(self):
        self.reset_G_CF()
        self.reset_G_DF()

    def reset_G_CF(self):
        self._G_CF = None

    def reset_G_DF(self):
        self._G_DF = None

    @property
    def builder(self):
        if self._builder is None:
            self._builder = Design(self.gH, self.hall_conductance, self.pedersen_conductance, 
                                  self.perfect_conductor_radius, self.dipole, self.epoch)
        return self._builder

    def reset_builder(self):
        self._builder = None

#%% Regularization

    def add_regularization(self, 
                           *reg: Regularizer, 
                           append: bool = False):
        if append and self.reg is not None:
            self.reg += reg
        else:
            self.reg = reg

    def reset_regularization(self):
        self.reg = None

#%% Inverse problem

    def solve_steady_state(self, **kwargs):        
        self.use_gpu = kwargs.pop('use_gpu', False)
        posterior = kwargs.pop('posterior', False)
        self.solver.solve_inverse_problem(posterior=posterior, **kwargs)
        self.m_CF = self.solver.m
        if posterior:
            self.Cmpost = self.solver.Cmpost
            
    @property
    def solver(self):
        if self.G_CF is None or self.d is None:
            raise ValueError('Solver cannot be defined without data')
        
        if self._solver is None:
            self._solver = Solver(G=self.G_CF, d=self.d, w=self.w, 
                                 reg=self.reg, use_gpu=self.use_gpu)
        return self._solver
    
    def reset_solver(self):
        self._solver = None

#%% Inverse problem - Steady-state - Timeseries

    def solve_multiple_steady_state(self, times, **kwargs):
        # We need some time steps if data is not given on equal time steps
        self.timesm = times        
        self.ntm = len(self.timesm)
        self.m_CF_t = [None]*self.ntm
        
        if kwargs.get('posterior') is True:
            self.P_CF_t = [None]*self.ntm
        
        desc = kwargs.pop('desc', 'SS solver')
        for i, t in tqdm(enumerate(self.timesm), total=self.ntm, desc=desc):

            # Fetch conductance first as many things (including data) will be reset.
            # Define conductance for t
            tid = np.argmin(abs(np.array(self.timesc) - t))
            self.clear_model(Hall_Pedersen_conductance = (self.hall_conductance_t[tid], self.pedersen_conductance_t[tid]),
                             reset_reg = False)
            
            # Get data at t
            self.add_timeseries_subset(t)
            
            # Run solver
            self.solve_steady_state(**kwargs)
            
            self.m_CF_t[i] = self.m_CF
            if kwargs.get('posterior') is True:
                self.P_CF_t[i] = self.Cmpost
        
        self.m_CF = None
        self.Cmpost = None

#%% Inverse problem - Kalman

    def add_dynamic_model_func(self, 
                               A: Union[np.ndarray, list[np.ndarray]]):
        
        if isinstance(A, list) and len(A[0].shape) == 2:
            self.nta = len(A)
            self.A = A
        elif len(A.shape) == 3:
            self.nta = A.shape[0]
            self.A = [A[i] for i in range(self.nta)]
        elif len(A.shape) == 2:
            self.A = A
        else:
            raise ValueError('A has to be: 1) list of 2D arrays. 2) 2D np.ndarray. 3) 3D np.ndarray')

    def get_SS(self, 
               times: Union[list[int], list[float]], 
               **kwargs):
        
        posterior = kwargs.pop('posterior', True)
        if not posterior:
            print('Setting posterior setting to True')
            posterior = True
        kwargs['posterior'] = posterior
        
        self.solve_multiple_steady_state(times, **kwargs)
        
        self.m_CF_SS = self.m_CF_t        
        self.m_CF_t = None
        
        self.P_CF_SS = self.P_CF_t
        self.P_CF_t = None

    def estimate_Q(self, 
                   m_CF_SS: Optional[Union[list[np.ndarray], np.ndarray]] = None,
                   cond: Optional[int] = None):
        
        if self.m_CF_SS is None and m_CF_SS is None:
            raise ValueError('Run get_SS() or provide m_CF_SS.')
        ## TODO: Check shape etc of m_CF_SS
        if m_CF_SS is not None:
            self.m_CF_SS = m_CF_SS
            
        if self.A is None:
            Ai = np.hstack((np.eye(self.n_CF)*2, -np.eye(self.n_CF)))
        
        rmstp = []
        for i in range(2, len(self.m_CF_SS)):
            if isinstance(self.A, list):
                Ai = self.A[i]
            elif self.A is not None:
                Ai = self.A
            
            rmstp.append(self.m_CF_SS[i] - Ai.dot(np.hstack((self.m_CF_SS[i-1], self.m_CF_SS[i-2]))))
        
        rmstp = np.vstack(rmstp).T
        self.Q_est = rmstp.dot(rmstp.T) / rmstp.shape[1]
        if cond is not None:
            self.Q_est = recond(self.Q_est, cond=cond, inflate=True)
        
    
    def solve_kalman(self,
                     A:     Optional[Union[list[np.ndarray], np.ndarray]]   = None,
                     times: Optional[Union[list[int], list[float]]]         = None,
                     Q_est: Optional[Union[list[np.ndarray], np.ndarray]]   = None,
                     mst:   Optional[np.ndarray]                            = None,
                     mstn:  Optional[np.ndarray]                            = None,
                     Pst:   Optional[np.ndarray]                            = None,
                     Pstn:  Optional[np.ndarray]                            = None,
                     **kwargs):
                
        self.rescale_iweights = False
        
        # Set A if provided
        if A is not None:
            self.add_dynamic_model_func(A)
        
        # Set Q_est if provided
        if Q_est is None and self.Q_est is None:
            self.Q_est = np.zeros((self.n_CF, self.n_CF))
        elif Q_est is not None:
            self.Q_est = Q_est
        
        # Initial guess
        if mst is None:
            mst = self.m_CF_SS[0]
        
        if mstn is None:
            mstn = self.m_CF_SS[0]

        if Pst is None:
            Pst = self.P_CF_SS[0]
        
        if Pstn is None:
            Pstn = self.P_CF_SS[0]

        # Allocate space        
        
        self.mss = np.zeros((self.n_CF, self.ntm))
        self.Pss = np.zeros((self.n_CF, self.n_CF, self.ntm))
        
        self.mcs = np.zeros((self.n_CF, self.ntm))
        self.Pcs = np.zeros((self.n_CF, self.n_CF, self.ntm))
        
        self.KF_weights = [] # Weight of each filter

        # Matrix mapping m_DF to dBr/dt
        Bc_ = self.builder.dBrdt_matrix(dt=1) # dt set to 1 s, should be scaled in loop

        # Start filter
        desc = kwargs.pop('desc', 'Kalman Filter')
        for i, t in tqdm(enumerate(self.timesm), total=self.ntm, desc=desc):
            
            ################### Prepare iteration ###################
            # Fetch conductance. Many things (including data) will be reset.
            tid = np.argmin(abs(np.array(self.timesc) - t))
            self.clear_model(Hall_Pedersen_conductance = (self.hall_conductance_t[tid], self.pedersen_conductance_t[tid]),
                             reset_reg = False)
            
            # Calculate dt for scaling of Bc_
            if i < self.ntm-1:
                dt = self.timesm[i+1]-self.timesm[i]                
            
            # Scale Bc : mapping m_DF to dBr (dt*Br/dt)
            Bc = Bc_ * dt
            
            # Mapping of m_CF to Br at r=RI
            Bs = self.builder._B_df_matrix_CF(r=RE)
            _, _, Bs = np.split(Bs, 3)
            Bs = recond(Bs, cond=1e3, inflate=True)
            
            # Mapping dm_CF to m_DF
            Bcd = scipy.linalg.lstsq(Bc.T.dot(Bc), Bc.T.dot(Bs))[0]            
            
            ################### Predict ###################
            # Initiate Kalman filter
            if isinstance(self.A, list):
                A1, A2 = self.A[i][:, :self.n_CF], self.A[i][:, self.n_CF:]
            else:
                if self.A is None:
                    A1, A2 = None, None
                else:                    
                    A1, A2 = self.A[:, :self.n_CF], self.A[:, self.n_CF:]
            
            kf = KalmanFilter(H=None, Q=self.Q_est, R=None, Pt=Pst, Ptn=Pstn, 
                              xt=mst, xtn=mstn, A1=A1, A2=A2, reg=self.reg)
            
            # predict step is equal for all datatypes
            kf.predict()
            
            ################### Update ###################
            # Update filter for each dataset.
            kf_m, kf_P = [], []
            for dtype in self.timeseries.keys(): # loop through data types            
                if len(self.timeseries[dtype]) == 0:
                    continue # Skip empty
                for timeseries in self.timeseries[dtype]: # loop through the datasets within each data type                    
                
                    # Fetch and add desired time-steps
                    self.reset_data()
                    self.reset_processed_data()                    
                    self.add_data(timeseries.get_t_subset(t))
                    
                    # Calculate design matrices
                    self.reset_G()
                    
                    if dtype == 'db_ground_mag':
                        ## TODO: Does not take difference in Bs into account.
                        Hcs = self.G_CF.dot(Bs)
                        H = 0 + Hcs
                    else:                    
                        Hcs = self.G_DF.dot(Bcd)
                        H = self.G_CF + Hcs
                    
                    # Correct data
                    d_p = self.d + Hcs.dot(mst)
                    del Hcs
                    
                    # Run update
                    try:
                        kf.update_MC(z=d_p, H=H, R=np.diag(1/self.w))
                    except:
                        raise ValueError(f'{dtype} failed inversion.')
                    
                    # Save update
                    kf_m.append(kf.x)
                    kf_P.append(kf.P)
            
            ################### Fuse modules ###################
            weights = np.array([1/np.trace(P) for P in kf_P])
            weights /= weights.sum()  # Normalize weights
            self.KF_weights.append(weights) # Save for curious people
    
            mstp = sum([w*m for w, m in zip(weights, kf_m)])            
            P_inv_sum = sum([w*lstsq_inv(P) for w, P in zip(weights, kf_P)])
            Pstp = lstsq_inv(P_inv_sum)
            del kf_m, kf_P, P_inv_sum, weights
    
            ################### Extract ###################

            # Calculate Cstp
            mctp = Bcd.dot(mstp - mst)
    
            Pctp = Bcd.dot(Pstp + Pst).dot(Bcd.T)
    
            ################### Save ###################
            self.mss[:, i] = mstp
            self.Pss[:, :, i] = Pstp
            
            self.mcs[:, i] = mctp
            self.Pcs[:, :, i] = Pctp
    
            mstn = dcopy(mst)
            mst = dcopy(mstp)
    
            Pstn = dcopy(Pst)
            Pst = dcopy(Pstp)
    
        self.rescale_iweights = True
        
#%% Evaluator

    @property
    def ev(self):
        if self._ev is None:
            self._ev = Evaluator(self)
        return self._ev
    
    def reset_evaluator(self):
        self._ev = None

#%%

def lstsq_inv(var, var2=None, reg=0):
    if var2 is None:
        var2 = np.eye(var.shape[0])
    return scipy.linalg.lstsq(var + reg*np.median(np.diag(var))*np.eye(var.shape[0]), var2, lapack_driver='gelsy')[0]

def recond(X, cond=1e2, inflate=False, inv=False):
        U, s, Vh = scipy.linalg.svd(X)
        scale = np.sum(s)
        T = s[0] / cond
        s[s<=T] = T
        
        if inflate:
            scale /= np.sum(s)    
        else:
            scale = 1
        
        if inv:
            S = np.zeros((Vh.shape[0], U.shape[1]))
            np.fill_diagonal(S, 1/s)
            return Vh.T.dot(S).dot(U.T) / scale
        else:
            S = np.zeros((U.shape[1], Vh.shape[0]))
            np.fill_diagonal(S, s)
            return U.dot(S).dot(Vh) * scale

#%% Old
'''





    def run_inversion(self, l1 = 0, l2 = 0, l3 = 0, FAC_reg=False,
                      data_density_weight = True, perimeter_width = 10, 
                      save_matrices=False, use_gpu=False,
                      **kwargs):
        """ Calculate model vector

        Uses all the data that has been added to solve full system of
        equations for electric field model vector.

        Parameters
        ----------
        l1 : float or tuple
            Damping parameter for model norm. If FAC_reg=True l1 can be a tuple 
            where the first and second entry target the FAC and E norm, 
            respectively. If it is just a float the E norm will be ignored.
        l2 : float
            Damping parameter for variation in the magnetic eastward direction.
            Functionality similar to l1 in regards to FAC_reg.
        l3 : float
            Damping parameter for variation in the magnetic northward direction
            Functionality similar to l1 in regards to FAC_reg.
        FAC_reg : boolean
            Activates FAC based regularization if True (default is False). Read
            l1 description for details on mixed FAC and E 2-norm regularization.
        data_density_weight : bool, optional
            Set to True to apply weights that are inversely proportional
            to data density. 
        perimeter_width: int, optional
            The number of grid cells with which the grid area will be expanded
            when choosing the data to be included in the inversion. Default is 10,
            which means that a 10 cell wide perimeter around the model inner
            grid will be included. 
        save_matrices : bool, optional
            Set to True to save G, d, and w in the lompe model object.

        **kwargs : dict
            key arguments to be passed to the scipy.linalg.lstsq (e.g., 'cond', 'lapack_driver').
            
        """

        # initialize G matrices
        if save_matrices:
            self._G = np.empty((0, self.grid_E.size))
            self._d = np.empty( 0)
            self._w = np.empty( 0)

        # make expanded grid for calculation of data density:
        self.biggrid = cs.CSgrid(self.grid_J.projection,
                                 self.grid_J.L + 2 * perimeter_width * self.grid_J.Lres, self.grid_J.W + 2 * perimeter_width * self.grid_J.Wres,
                                 self.grid_J.Lres, self.grid_J.Wres,
                                 R = self.R )
        
        GTGs = []
        GTds = []

        iweights = []
        for dtype in self.data.keys(): # loop through data types
            for ds in self.data[dtype]: # loop through the datasets within each data type
                iweights.append(ds.iweight)
        
        if np.max(iweights) != 1:
            print('The provided iweights were re-scaled so max(iweights)=1')
            iweights = np.array(iweights)/np.max(iweights)
        
        ii = 0
        for dtype in self.data.keys(): # loop through data types
            for ds in self.data[dtype]: # loop through the datasets within each data type
                # skip data points that are outside biggrid:
                ds = ds.subset(self.biggrid.ingrid(ds.coords['lon'], ds.coords['lat']))
                if ds.coords['lat'].size > 1: #If there is data inside biggrid                
                    if 'mag' in dtype:
                        Gs = np.split(self.matrix_func[dtype](**ds.coords), 3, axis = 0)
                        G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])
                    if dtype in ['efield', 'convection']:
                        Gs = self.matrix_func[dtype](**ds.coords)
                        G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])
                    if dtype == 'fac':
                        G = np.vstack(self.matrix_func[dtype](**ds.coords))

                    if (dtype in ['convection', 'efield']) & (ds.los is not None): # deal with line of sight data:
                        Ge, Gn = np.split(G, 2, axis = 0)
                        G = Ge * ds.los[0].reshape((-1, 1)) + Gn * ds.los[1].reshape((-1, 1))

                    # calculate weights based on data density:
                    if data_density_weight:
                        bincount = self.biggrid.count(ds.coords['lon'], ds.coords['lat'])
                        i, j = self.biggrid.bin_index(ds.coords['lon'], ds.coords['lat'])
                        spatial_weight = 1. / np.maximum(bincount[i, j], 1)
                        spatial_weight[i == -1] = 1
                        if ds.values.ndim == 2: # stack weights for each component in dataset.values:
                            spatial_weight = np.tile(spatial_weight, ds.values.shape[0])
                    else:
                        spatial_weight = np.ones(ds.values.size)

                    dimensions = np.array(ds.values, ndmin = 2).shape[0]
                    if np.array(ds.error, ndmin=2).shape[0]==1:
                        error = np.tile(ds.error, dimensions)
                    else: #error is different for different components
                        error = ds.error.flatten()
                                        
                    w_i = spatial_weight * 1/(error**2) * iweights[ii]
                    if iweights[ii] != 1:
                        print('{}: Measurement uncertainty effectively changed from {} to {}'.format(dtype, np.median(error), np.median(error)/np.sqrt(iweights[ii])))

                    if save_matrices:
                        self._G = np.vstack((self._G, G))
                        self._d = np.hstack((self._d, np.hstack(ds.values)))
                        self._w = np.hstack((self._w, w_i))

                    GTG_i = G.T.dot(np.diag(w_i)).dot(G)
                    GTd_i = G.T.dot(np.diag(w_i)).dot(np.hstack(ds.values))
                    
                    GTGs.append(GTG_i)
                    GTds.append(GTd_i)
                    ii += 1         

        self.GTG = np.sum(np.array(GTGs), axis=0)
        self.GTd = np.sum(np.array(GTds), axis=0)

        # Reguarlization
        if not FAC_reg and (isinstance(l1, tuple) or isinstance(l2, tuple) or isinstance(l3, tuple)):
            raise ValueError('l1, l2, and l3 can only be tuple if FAC_reg=True')
        
        def reg_E(self, l1, l2, l3):
            """Calculate the roughening matrix for E (normal) regularization"""
            LTL = 0
            if l1 > 0:
                LTL_l1 = np.eye(self.GTG.shape[0])
                LTL += l1 * LTL_l1 / np.median(LTL_l1.diagonal())
            if l2 > 0:
                LTL += l2 * self.LTLe / np.median(self.LTLe.diagonal())
            if l3 > 0:
                LTL += l3 * self.LTLn / np.median(self.LTLn.diagonal())
            return LTL
        
        def reg_FAC(self, l1, l2, l3):
            """Calculate the roughening matrix for FAC regularization"""
            G_FAC = self.FAC_matrix()
            LTL = 0
            if l1 > 0:
                LTL_l1 = G_FAC.T.dot(G_FAC)
                LTL += l1 * LTL_l1 / np.median(LTL_l1.diagonal())
            if l2 > 0:
                G_FAC_e = self.Le_J.dot(G_FAC)
                LTL_l2 = G_FAC_e.T.dot(G_FAC_e)
                LTL += l2 * LTL_l2 / np.median(LTL_l2.diagonal())
            if l3 > 0:
                G_FAC_n = self.Ln_J.dot(G_FAC)
                LTL_l3 = G_FAC_n.T.dot(G_FAC_n)
                LTL += l3 * LTL_l3 / np.median(LTL_l3.diagonal())
            return LTL
        
        def ensure_tuple(value):
            """Ensure the value is a tuple of length 2."""
            if isinstance(value, tuple):
                if len(value) != 2:
                    raise ValueError(f"Tuple {value} must have length 2.")
                return value
            return (value, 0)
        
        if not FAC_reg:
            LTL = reg_E(self, l1, l2, l3)
        elif FAC_reg and  any(isinstance(x, tuple) for x in (l1, l2, l3)):
            l1 = ensure_tuple(l1)
            l2 = ensure_tuple(l2)
            l3 = ensure_tuple(l3)
            LTL = reg_FAC(self, l1[0], l2[0], l3[0])
            LTL += reg_E(self, l1[1], l2[1], l3[1])
        elif FAC_reg:
            LTL = reg_FAC(self, l1, l2, l3)
        else:
            LTL = 0
        
        gtg_mag = np.median(np.diagonal(self.GTG))
        GG = self.GTG + LTL*gtg_mag
            
        if 'rcond' in kwargs.keys():
            warnings.warn("'rcond' keyword (and use of np.linalg.lstsq) is deprecated! Use kw 'cond' (for scipy.linalg.lstsq) instead")
            kwargs['cond'] = kwargs['rcond']
        if 'cond' not in kwargs.keys():
            kwargs['cond'] = None
        
        if 'lapack_driver' not in kwargs.keys():
            kwargs['lapack_driver'] = 'gelsd'

        if gpu_avail and use_gpu:            
            self.Cmpost = cp.asnumpy(cp.linalg.solve(cp.array(GG), cp.array(np.eye(GG.shape[0]))))
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device().synchronize()
            gc.collect()
        else:
            self.Cmpost = scipy.linalg.lstsq(GG, np.eye(GG.shape[0]), **kwargs)[0]

        self.Rmatrix = self.Cmpost.dot(self.GTG)
        self.m = self.Cmpost.dot(self.GTd)

        return (self.GTG, self.GTd)


    def get_G_CF(self, data_density_weight = True, perimeter_width = 10):
        """ Calculate design matrix and data covariance
        
        Parameters
        ----------
        data_density_weight : bool, optional
            Set to True to apply weights that are inversely proportional
            to data density. 
        perimeter_width: int, optional
            The number of grid cells with which the grid area will be expanded
            when choosing the data to be included in the inversion. Default is 10,
            which means that a 10 cell wide perimeter around the model inner
            grid will be included. 
        """

        # initialize G matrices
        self.G_CF   = np.empty((0, self.grid_E.size))
        self._d     = np.empty( 0)
        self._w     = np.empty( 0)
        self._Cd    = np.empty( 0)

        # make expanded grid for calculation of data density:
        self.biggrid = cs.CSgrid(self.grid_J.projection,
                                 self.grid_J.L + 2 * perimeter_width * self.grid_J.Lres, self.grid_J.W + 2 * perimeter_width * self.grid_J.Wres,
                                 self.grid_J.Lres, self.grid_J.Wres,
                                 R = self.R )

        iweights = []
        for dtype in self.data.keys(): # loop through data types
            for ds in self.data[dtype]: # loop through the datasets within each data type
                iweights.append(ds.iweight)
        
        if np.max(iweights) != 1:
            print('The provided iweights were re-scaled so max(iweights)=1')
            iweights = np.array(iweights)/np.max(iweights)
        
        ii = 0
        for dtype in self.data.keys(): # loop through data types
            for ds in self.data[dtype]: # loop through the datasets within each data type
                # skip data points that are outside biggrid:
                ds = ds.subset(self.biggrid.ingrid(ds.coords['lon'], ds.coords['lat']))
                if ds.coords['lat'].size > 1: #If there is data inside biggrid                
                    if 'mag' in dtype:
                        Gs = np.split(self.matrix_func[dtype](**ds.coords), 3, axis = 0)
                        G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])
                    if dtype in ['efield', 'convection']:
                        Gs = self.matrix_func[dtype](**ds.coords)
                        G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])
                    if dtype == 'fac':
                        G = np.vstack(self.matrix_func[dtype](**ds.coords))

                    if (dtype in ['convection', 'efield']) & (ds.los is not None): # deal with line of sight data:
                        Ge, Gn = np.split(G, 2, axis = 0)
                        G = Ge * ds.los[0].reshape((-1, 1)) + Gn * ds.los[1].reshape((-1, 1))

                    # calculate weights based on data density:
                    if data_density_weight:
                        bincount = self.biggrid.count(ds.coords['lon'], ds.coords['lat'])
                        i, j = self.biggrid.bin_index(ds.coords['lon'], ds.coords['lat'])
                        spatial_weight = 1. / np.maximum(bincount[i, j], 1)
                        spatial_weight[i == -1] = 1
                        if ds.values.ndim == 2: # stack weights for each component in dataset.values:
                            spatial_weight = np.tile(spatial_weight, ds.values.shape[0])
                    else:
                        spatial_weight = np.ones(ds.values.size)

                    dimensions = np.array(ds.values, ndmin = 2).shape[0]
                    if np.array(ds.error, ndmin=2).shape[0]==1:
                        error = np.tile(ds.error, dimensions)
                    else: #error is different for different components
                        error = ds.error.flatten()
                                        
                    w_i = spatial_weight * 1/(error**2) * iweights[ii]
                    if iweights[ii] != 1:
                        print('{}: Measurement uncertainty effectively changed from {} to {}'.format(dtype, np.median(error), np.median(error)/np.sqrt(iweights[ii])))

                    self.G_CF   = np.vstack((self.G_CF, G))
                    self._d     = np.hstack((self._d, np.hstack(ds.values)))
                    self._w     = np.hstack((self._w, w_i))
                    self._Cd    = np.hstack((self._Cd, error**2))

                    ii += 1    

    def get_G_DF(self, data_density_weight = True, perimeter_width = 10):
        """ Calculate design matrix and data covariance
        
        Parameters
        ----------
        data_density_weight : bool, optional
            Set to True to apply weights that are inversely proportional
            to data density. 
        perimeter_width: int, optional
            The number of grid cells with which the grid area will be expanded
            when choosing the data to be included in the inversion. Default is 10,
            which means that a 10 cell wide perimeter around the model inner
            grid will be included. 
        """

        # initialize G matrices
        self.G_DF   = np.empty((0, self.grid_E.size))
        self._d     = np.empty( 0)
        self._w     = np.empty( 0)
        self._Cd    = np.empty( 0)

        # make expanded grid for calculation of data density:
        self.biggrid = cs.CSgrid(self.grid_J.projection,
                                 self.grid_J.L + 2 * perimeter_width * self.grid_J.Lres, self.grid_J.W + 2 * perimeter_width * self.grid_J.Wres,
                                 self.grid_J.Lres, self.grid_J.Wres,
                                 R = self.R )

        iweights = []
        for dtype in self.data.keys(): # loop through data types
            for ds in self.data[dtype]: # loop through the datasets within each data type
                iweights.append(ds.iweight)
        
        if np.max(iweights) != 1:
            print('The provided iweights were re-scaled so max(iweights)=1')
            iweights = np.array(iweights)/np.max(iweights)
        
        ii = 0
        for dtype in self.data.keys(): # loop through data types
            for ds in self.data[dtype]: # loop through the datasets within each data type
                # skip data points that are outside biggrid:
                ds = ds.subset(self.biggrid.ingrid(ds.coords['lon'], ds.coords['lat']))
                if ds.coords['lat'].size > 1: #If there is data inside biggrid                
                    if 'mag' in dtype:
                        Gs = np.split(self.matrix_func_DF[dtype](**ds.coords), 3, axis = 0)
                        G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])
                    if dtype in ['efield', 'convection']:
                        Gs = self.matrix_func_DF[dtype](**ds.coords)
                        G = np.vstack([G_ for i, G_ in enumerate(Gs) if i in ds.components])
                    if dtype == 'fac':
                        G = np.vstack(self.matrix_func_DF[dtype](**ds.coords))

                    if (dtype in ['convection', 'efield']) & (ds.los is not None): # deal with line of sight data:
                        Ge, Gn = np.split(G, 2, axis = 0)
                        G = Ge * ds.los[0].reshape((-1, 1)) + Gn * ds.los[1].reshape((-1, 1))

                    # calculate weights based on data density:
                    if data_density_weight:
                        bincount = self.biggrid.count(ds.coords['lon'], ds.coords['lat'])
                        i, j = self.biggrid.bin_index(ds.coords['lon'], ds.coords['lat'])
                        spatial_weight = 1. / np.maximum(bincount[i, j], 1)
                        spatial_weight[i == -1] = 1
                        if ds.values.ndim == 2: # stack weights for each component in dataset.values:
                            spatial_weight = np.tile(spatial_weight, ds.values.shape[0])
                    else:
                        spatial_weight = np.ones(ds.values.size)

                    dimensions = np.array(ds.values, ndmin = 2).shape[0]
                    if np.array(ds.error, ndmin=2).shape[0]==1:
                        error = np.tile(ds.error, dimensions)
                    else: #error is different for different components
                        error = ds.error.flatten()
                                        
                    w_i = spatial_weight * 1/(error**2) * iweights[ii]
                    if iweights[ii] != 1:
                        print('{}: Measurement uncertainty effectively changed from {} to {}'.format(dtype, np.median(error), np.median(error)/np.sqrt(iweights[ii])))

                    self.G_DF   = np.vstack((self.G_DF, G))
                    self._d     = np.hstack((self._d, np.hstack(ds.values)))
                    self._w     = np.hstack((self._w, w_i))
                    self._Cd    = np.hstack((self._Cd, error**2))

                    ii += 1
        
'''
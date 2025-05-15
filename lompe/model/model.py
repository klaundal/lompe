""" Model class """
import numpy as np
from .design import Design
from .gridhandler import GridHandler
from .solver import Solver
from .evaluator import Evaluator
from .regularizer import Regularizer
from .timeseries import TimeSeries
from typing import Union, Optional, Tuple, Callable
from tqdm import tqdm

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
        self.data = {'efield':[], 'convection':[], 'ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}
        self.timeseries = {'efield':[], 'convection':[], 'ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}
        self._d = None
        
        # Data weights
        self._w = None
        self._sw = None
        self._iweight = None
        self._Cdinv = None
        
        # Kalman
        self.nta = None
        self.A = None
        
        # Epot and Eind model parameters
        self.m_CF = None
        self.m_DF = None
        self.m_CF_t = None
        self.m_DF_t = None
        self.timesm = None # Array of time were the model will be evaluated
        
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
        
        if np.max(iweights) != 1:
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
        self.data = {'efield':[], 'convection':[], 'ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}
    
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
                print('You passed {}, which is not in {} - ignored'.format(dtype, list(self.tseries.keys())))

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
        self.timeseries = {'efield':[], 'convection':[], 'ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}

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
        
        self.m_CF = None        

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
        
        self.solve_multiple_steady_state(times, **kwargs)
        self.m_CF_SS = self.m_CF_t
        self.m_CF_t = None

    def estimate_Q(self,
                   times: Optional[Union[list[int], list[float]]] = None,
                   **kwargs):
        
        if self.A is None:
            raise ValueError('No dynamic model function added!')
        else:
            if times is not None and isinstance(self.A, list):
                if len(times) != len(self.A):
                    raise ValueError('A and time have to have the same length')
                        
        if self.m_CF_SS is None:
            if times is None:
                raise ValueError('Either run get_SS() or define times.')
            self.get_SS(times, **kwargs)
            
        rmstp = []
        for i in range(2, self.ntm):
            if isinstance(self.A, list):
                Ai = self.A[i]
            else:
                Ai = self.A
            
            rmstp.append(self.m_CF_SS[i] - Ai.dot(np.hstack((self.m_CF_SS[i-1], self.m_CF_SS[i-2]))))
        
        rmstp = np.vstack(rmstp).T
        self.Q_est = rmstp.dot(rmstp.T) / rmstp.shape[1]

    def something():
        # Area of the mesh grid cells
        A = grid_E.A.flatten()

        # Curl relation of the DF SECS
        Q = np.eye(grid_E.size) - A.dot(np.full((grid_E.size, grid_E.size), 1 / (4 * np.pi * grid_E.R**2)))

        G = -dt * np.diag(1/A).dot(Q)
        
        Bc = dcopy(G)
        
        ## Br from potential E
        SH, SP = get_c(grid_big.lat, grid_big.lon, t)
        SH = SH.reshape(-1, 1)
        SP = SP.reshape(-1, 1)
        Bs = calc_G_r_CF_s2b(SH, SP)
        Bs = recond(Bs, cond=1e3, inflate=True) ## 1
        
        Bcd = scipy.linalg.lstsq(Bc.T.dot(Bc), Bc.T.dot(Bs))[0]

    def solve_kalman(self,
                     A=None,
                     times=None,
                     Q_est=None,
                     mst=None,
                     mstn=None,
                     Pst=None,
                     Pstn=None,
                     **kwargs):
        # Check if the dynamic model function is defined
        if self.A is None and A is None:
            raise ValueError('The dynamic model function needs to be defined')
        
        if A is not None:
            self.add_dynamic_model_func(A)
        
        if Q_est is None:
            self.estiamte_Q(times, **kwargs)

        self.n_CF = self.gH.size_E
        
        # Allocate space
        mss = np.zeros((self.n_CF, self.ntm))
        Pss = np.zeros((self.n_CF, self.n_CF, self.ntm))

        mcs = np.zeros((self.n_CF, self.ntm))
        Pcs = np.zeros((self.n_CF, self.n_CF, self.ntm))

        if mst is None:
            self.mst = self.m_CF_SS[0]
        else:
            self.mst = mst # Init
        
        if mst is None:
            self.mstn = self.m_CF_SS[0]
        else:
            self.mstn = mstn

        self.Pst = Pst
        self.Pstn = Pst

        weights_all = []
        for i, t in tqdm(enumerate(self.timesm), total=self.ntm, desc='Kalman Filter'):
            
            # Fetch conductance first as many things (including data) will be reset.
            # Define conductance for t
            tid = np.argmin(abs(np.array(self.timesc) - t))
            self.clear_model(Hall_Pedersen_conductance = (self.hall_conductance_t[tid], self.pedersen_conductance_t[tid]),
                             reset_reg = False)
            
            ################### Predict ###################
            # Initiate Kalman filter, predict step is the same for all
            kf = self.KF()
            if isinstance(self.A, list):
                kf.predict(self.A[i], self.mst, self.mstn, self.Pst, self.Pstn)
            else:
                kf.predict(self.A, self.mst, self.mstn, self.Pst, self.Pstn)
            
            ################### Update ###################
            # Update filter for each dataset.
            kf_m, kf_P = [], []
            for dtype in self.timeseries.keys(): # loop through data types            
                for timeseries in self.timeseries[dtype]: # loop through the datasets within each data type
                    # Fetch and add desired time-steps
                    self.clear_data()
                    self.clear_processed_data()
                    self.add_data(timeseries.get_t_subset(t))
                    
                    self.G_CF
                    self.G_DF
                    
                    Hcs = self.G_DF.dot(Bcd)
                    H = self.G_CF + Hcs
                    d_p = self.d + Hcs.self.mst
                    del Hcs
                    
                    kf.update(H, self.Q_est, np.diag(1/self.w), d_p)
                    kf_m.append(kf.m)
                    kf_P.append(kf.P)
            
            ################### Fuse modules ###################
            weights = np.array([1/np.trace(P) for P in kf_P])
            weights /= weights.sum()  # Normalize weights
            weights_all.append(weights) # Save for curious people
    
            mstp = np.sum([w*m for w, m in zip(weights, kf_m)])            
            P_inv_sum = np.sum([w*np.linalg.pinv(P) for w, P in zip(weights, kf_P)])
            Pstp = np.linalg.pinv(P_inv_sum)
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
    
            self.mstn = dcopy(self.mst)
            self.mst = dcopy(self.mstp)
    
            self.Pstn = dcopy(self.Pst)
            self.Pst = dcopy(self.Pstp)
        






    def solve_kalman(self):
        
        # Allocate space
        mss = np.zeros((n_CF, nt))
        Pss = np.zeros((n_CF, n_CF, nt))

        mcs = np.zeros((n_CF, nt))
        Pcs = np.zeros((n_CF, n_CF, nt))

        mst = dcopy(m0_CF)
        mstn = dcopy(m0_CF)

        Pst = dcopy(C0_CF)
        Pstn = dcopy(C0_CF)

        lreg=2e0

        weights_all = []
        model = None
        for i, start in tqdm(enumerate(times), total=nt, desc='Kalman Filter'):
            # Define end of window
            end = start + window_size_almost
    
            # Define time in seconds from start for spline model
            t = (start - t0).seconds

            # Make conductance functions
            def SH(lon, lat):
                H, P = get_c(lat, lon, t)
                return H
            def SP(lon, lat):
                H, P = get_c(lat, lon, t)
                return P
    
            # Get forward matrices
            ## delta Br from induction E
            Bc = dcopy(G)
    
            ## Br from potential E
            SH, SP = get_c(grid_big.lat, grid_big.lon, t)
            SH = SH.reshape(-1, 1)
            SP = SP.reshape(-1, 1)
            Bs = calc_G_r_CF_s2b(SH, SP)
            Bs = recond(Bs, cond=1e3, inflate=True) ## 1
        
            ################### Update - prep ###################
    
            Bcd = scipy.linalg.lstsq(Bc.T.dot(Bc), Bc.T.dot(Bs))[0]
    
            filters = []
            ## Data B
            # Extract subset of data
            df_smi = df_sm.loc[start:end]

            # Add measurements - Supermag
            f = grid.ingrid(df_smi['lon'], df_smi['lat'])
            coords = np.vstack((df_smi['lon'][f], df_smi['lat'][f], np.ones(np.sum(f))*6371.2e3))
            B = np.vstack((df_smi['Be'][f], df_smi['Bn'][f], df_smi['Bu'][f]))*1e-9
            error = 10*1e-9
            data = lompe.Data(B, coords, datatype='ground_mag', iweight= 1, error=error)
    
            if model is None:
                model = lompe.Emodel(grid, Hall_Pedersen_conductance = (SH_fun, SP_fun))
            else:
                model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
    
            model.add_data(data)
            model.get_G_CF()
            model.get_G_DF()
            d = model._d
            R_inv = model._w
            H = model.G_CF + 0
            R = np.linalg.inv(np.diag(R_inv))
            Hs = model.G_CF + 0
            Hc = model.G_DF + 0
            Hcs = Hc.dot(Bcd)
            H = Hs + Hcs
            d_p = d + Hcs.dot(mst)
    
            kf1 = KF.KalmanFilter(H, Q_est, R, Pst, Pstn, mst, mstn)
            kf1.predict()
            #kf1.update_MC(d_p, reg=1e0)
            kf1.update_MC(d_p, reg=lreg)
            filters.append(kf1)
        
            ## Data dB
            # Extract subset of data
            df_sm_dbi = df_sm_db.loc[start:end]
    
            # Add measurements - Supermag
            f = grid.ingrid(df_sm_dbi['lon'], df_sm_dbi['lat'])
            coords = np.vstack((df_sm_dbi['lon'][f], df_sm_dbi['lat'][f], np.ones(np.sum(f))*6371.2e3))
            B = np.vstack((df_sm_dbi['dBe_pred'][f], df_sm_dbi['dBn_pred'][f], df_sm_dbi['dBu_pred'][f]))*1e-9
            error = 10*1e-9
            data = lompe.Data(B, coords, datatype='ground_mag', iweight= 1, error=error)
    
            model.clear_model(Hall_Pedersen_conductance = (SH_fun, SP_fun)) # reset
            
            model.add_data(data)
            model.get_G_CF()
            model.get_G_DF()
            d = model._d
            R_inv = model._w
            R = np.linalg.inv(np.diag(R_inv))
            _, _, Gu = get_SECS_B_G_matrices(grid_E.lat.flatten(), grid_E.lon.flatten(), np.ones(grid_E.size)*model.R,
                                             grid.lat.flatten(), grid.lon.flatten())
            Grd = lstsq_inv(Gu.T.dot(Gu), var2=Gu.T)
            Ge, Gn, Gu = get_SECS_B_G_matrices(data.coords['lat'], data.coords['lon'], np.ones(data.coords['lat'].size)*6371000,
                                               grid.lat.flatten(), grid.lon.flatten())
            Gg = np.vstack((Ge, Gn, Gu))
            
            Hcs = Gg.dot(Grd).dot(Bs)
            Hcs /= dt # Go from T/s to T/dt
            d_p = d + Hcs.dot(mst)
    
            kf2 = KF.KalmanFilter(Hcs, Q_est, R, Pst, Pstn, mst, mstn)
            kf2.predict()
            #kf2.update_MC(d_p, reg=1e0)
            kf2.update_MC(d_p, reg=lreg)
            filters.append(kf2)
    
            ################### Fuse modules ###################
            uncertainties = np.zeros(len(filters))
            for j, kf in enumerate(filters):
                uncertainties[j] = np.trace(kf.P)

            weights = 1 / uncertainties
            weights /= weights.sum()  # Normalize weights
            weights_all.append(weights)
    
            mstp = np.zeros(n_CF)
            P_inv_sum = np.zeros((n_CF, n_CF)) 
            for w, kf in zip(weights, filters):
                mstp += w*kf.x
                P_inv_sum += w*lstsq_inv(kf.P)
    
            Pstp = np.linalg.inv(P_inv_sum)    
    
            ################### Extract ###################

            # Calculate Cstp
            mctp = Bcd.dot(mstp - mst)
    
            Pctp = Bcd.dot(Pstp + Pst).dot(Bcd.T)
    
            ################### Save ###################
            mss[:, i] = mstp
            Pss[:, :, i] = Pstp
            
            mcs[:, i] = mctp
            Pcs[:, :, i] = Pctp
    
            mstn = dcopy(mst)
            mst = dcopy(mstp)
    
            Pstn = dcopy(Pst)
            Pst = dcopy(Pstp)
        

#%% Evaluator

    @property
    def ev(self):
        if self._ev is None:
            self._ev = Evaluator(self)
        return self._ev
    
    def reset_evaluator(self):
        self._ev = None

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
""" Model class """
import numpy as np
from .design import Design
from .gridhandler import GridHandler
from .solver import Solver
from .evaluator import Evaluator

#%%

RE = 6371.2e3 # Earth radius in meters

#%%

class Emodel(object):
    def __init__(self, grid,
                       Hall_Pedersen_conductance,
                       epoch = 2015., # epoch, decimal year, used for IGRF dependent calculations
                       dipole = False, # set to True to use dipole field and dipole coords
                       perfect_conductor_radius = None,
                       ew_regularization_limit = None,
                       perimeter_width=10,
                       data_density_weight=True):
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
        # options
        self.perfect_conductor_radius = perfect_conductor_radius
        self.dipole = dipole
        self.epoch = epoch
        self.use_gpu = False
        self.perimeter_width=perimeter_width
        self.data_density_weight = data_density_weight
        
        self._builder = None
        self._solver = None
        self._ev = None

        self.reg = None
                
        self._G_CF = None
        self._G_DF = None
        
        self.data = {'efield':[], 'convection':[], 'ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}
        self._d = None
        
        self._w = None
        self._sw = None
        self._iweight = None
        self._Cdinv = None
        
        self.m_CF = None
        self.m_DF = None
        
        self.Cmpost = None
        
        # Initiate grid handler
        self.gH = GridHandler(grid)
        
        # Define conductance functions and initiate Desgin
        self.clear_model(Hall_Pedersen_conductance = Hall_Pedersen_conductance)
      
#%% Utils

    def clear_model(self, Hall_Pedersen_conductance = None, perimeter_width=None):
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
        self.data = {'efield':[], 'convection':[], 'ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}

        # Hall and Pedersen conductance - either inversion or functions:
        if Hall_Pedersen_conductance != None:

            _h, _p = Hall_Pedersen_conductance
            
            self.hall_conductance     = lambda lon = self.gH.grid_J.lon, lat = self.gH.grid_J.lat: _h(lon, lat)
            self.pedersen_conductance = lambda lon = self.gH.grid_J.lon, lat = self.gH.grid_J.lat: _p(lon, lat)

        self.reset_builder()
        self.reset_solver()
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
        self._w = self._sw * self._iweight * self._Cdinv # Combined error

    def change_parimeter_width(self, perimeter_width=10):
        if self.perimeter_width == perimeter_width:
            print('Perimeter was not changed due to identical values')
        else:
            self.perimeter_width = perimeter_width
            self.gH.grid_d = None
            self.reset_processed_data()
                
    def reset_processed_data(self):
        self._d = None
        self._sw = None
        self._iweight = None
        self._Cdinv = None
        self._w = None

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

    def add_regularization(self, *reg, append=False):
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
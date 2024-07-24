""" Model class """
import apexpy
import numpy as np
from scipy.interpolate import RectBivariateSpline, griddata
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
from secsy import cubedsphere as cs
from ppigrf import igrf
from lompe.utils.time import yearfrac_to_datetime
from dipole import Dipole
from .varcheck import check_input, extrapolation_check
import scipy
import warnings
from kneed import KneeLocator
from kneefinder import KneeFinder as KF

RE = 6371.2e3 # Earth radius in meters

class Emodel(object):
    def __init__(self, grid,
                       Hall_Pedersen_conductance,
                       epoch = 2015., # epoch, decimal year, used for IGRF dependent calculations
                       dipole = False, # set to True to use dipole field and dipole coords
                       perfect_conductor_radius = None,
                       ew_regularization_limit = None
                ):
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

        # function that tunes the east west regularization
        if ew_regularization_limit is None:
            lat_w = lambda lat: lat*0 + 1.
        else:
            try:
                a, b = ew_regularization_limit
            except:
                raise Exception('ew_regularization_limit should have two and only two values')
            lat_w = lambda lat: np.where(lat < a, 1, np.where(lat > b, 0, (b - lat) / (b - a)))


        # set up inner and outer grids:
        self.grid_J = grid # inner
        self.R = self.grid_J.R
        xi_e  = np.hstack((self.grid_J.xi_mesh[0]    , self.grid_J.xi_mesh [0 , - 1] + self.grid_J.dxi )) - self.grid_J.dxi /2 
        eta_e = np.hstack((self.grid_J.eta_mesh[:, 0], self.grid_J.eta_mesh[-1,   0] + self.grid_J.deta)) - self.grid_J.deta/2 
        self.grid_E = cs.CSgrid(cs.CSprojection(self.grid_J.projection.position, self.grid_J.projection.orientation),
                               self.grid_J.L + self.grid_J.Lres, self.grid_J.W + self.grid_J.Wres, self.grid_J.Lres, self.grid_J.Wres, 
                               edges = (xi_e, eta_e), R = self.R) # outer

        self.lat_J, self.lon_J = np.ravel( self.grid_J.lat ), np.ravel( self.grid_J.lon )
        self.lat_E, self.lon_E = np.ravel( self.grid_E.lat ), np.ravel( self.grid_E.lon )

        # set SECS singularity limit so it covers the cell:
        self.secs_singularity_limit = np.min([self.grid_J.Wres, self.grid_J.Lres])/2

        # dictionary of functions that belong to different datasets:
        self.matrix_func = {'ground_mag':self._B_df_matrix,
                            'convection':self._v_matrix,
                            'efield':self._E_matrix,
                            'space_mag_fac':self._B_cf_matrix,
                            'space_mag_full':self._B_cf_df_matrix,
                            'fac':self.FAC_matrix}

        self.clear_model(Hall_Pedersen_conductance = Hall_Pedersen_conductance)

        # calculate main field values for all grid points
        refh = (self.R - RE) * 1e-3 # apex reference height [km] - also used for IGRF altitude
        if self.dipole:
            Bn, Bu = Dipole(self.epoch).B(self.lat_E, self.grid_E.R * 1e-3)
            Be = np.zeros_like(Bn)
        else: # use IGRF
            Be, Bn, Bu = igrf(self.lon_E, self.lat_E, refh, yearfrac_to_datetime([self.epoch]))
        Be, Bn, Bu = Be * 1e-9, Bn * 1e-9, Bu * 1e-9 # nT -> T
        self.B0 = np.sqrt(Be**2 + Bn**2 + Bu**2).reshape((1, -1))
        self.Bu = Bu.reshape((1, -1))

        if not np.allclose(np.sign(self.Bu), np.sign(self.Bu.flatten()[0])):
            raise Exception('your grid covers two magnetic hemispheres. It should not')
        self.hemisphere = -np.sign(self.Bu.flatten()[0]) # 1 for north, -1 for south

        # calculate gradient and divergence matrices for inner grid:
        self.De, self.Dn = self.grid_J.get_Le_Ln()
        self.Ddiv = self.grid_J.divergence()

        # Matrices to json.loadsuate electric field on inner grid:
        self.Ee, self.En = self._E_matrix()
        self.Ee, self.En = self.Ee, self.En

        # Matrices to evaluate velocity field on inner grid:
        self.Ve, self.Vn = self._v_matrix()

        # cell area matrix:
        dxi, deta, A = self.grid_J.projection.differentials(self.grid_J.xi , self.grid_J.eta,
                                                            self.grid_J.dxi, self.grid_J.deta, R = self.R)
        self.A = np.diag(np.ravel(A))

        # curl/divergence distribution matrix Q:
        self.Q = np.eye(self.grid_J.size) - self.A.dot(np.full((self.grid_J.size, self.grid_J.size), 1 / (4 * np.pi * self.R**2)))

        # inverse of QA
        self.QiA = np.linalg.pinv(self.Q, hermitian = True).dot(self.A)

        # matrix L that calculates derivative in magnetic eastward direction on grid_E:
        De2, Dn2 = self.grid_E.get_Le_Ln()
        if self.dipole: # L matrix gives gradient in eastward direction
            self.Le = De2 * lat_w(self.hemisphere * self.grid_E.lat.flatten()).reshape((-1, 1))
            self.LTLe = self.Le.T.dot(self.Le)
            self.Ln = Dn2
            self.LTLn = self.Ln.T.dot(self.Ln)
        else: # L matrix gives gradient in QD eastward direction
            apx = apexpy.Apex(epoch, refh = refh)
            mlat, mlon = apx.geo2apex(self.grid_E.lat.flatten(), self.grid_E.lon.flatten(), refh)
            f1, f2 = apx.basevectors_qd(self.grid_E.lat.flatten(), self.grid_E.lon.flatten(), refh)
            f1 = f1/np.linalg.norm(f1, axis = 0)
            self.Le = De2 * f1[0].reshape((-1, 1)) + Dn2 * f1[1].reshape((-1, 1))
            self.Le = self.Le * lat_w(self.hemisphere * mlat).reshape((-1, 1))
            self.LTLe = self.Le.T.dot(self.Le)
            f2 = f2/np.linalg.norm(f2, axis = 0)
            self.Ln = De2 * f2[0].reshape((-1, 1)) + Dn2 * f2[1].reshape((-1, 1))
            self.LTLn = self.Ln.T.dot(self.Ln)
            
        # matrix L that calculates derivative in magnetic eastward direction on grid_J:
        # Hopefully this can be written in a smarter way!!
        De2, Dn2 = self.grid_J.get_Le_Ln()
        if self.dipole: # L matrix gives gradient in eastward direction
            self.Le_J = De2 * lat_w(self.hemisphere * self.grid_J.lat.flatten()).reshape((-1, 1))
            self.LTLe_J = self.Le_J.T.dot(self.Le_J)
            self.Ln_J = Dn2
            self.LTLn_J = self.Ln_J.T.dot(self.Ln_J)
        else: # L matrix gives gradient in QD eastward direction
            apx = apexpy.Apex(epoch, refh = refh)
            mlat, mlon = apx.geo2apex(self.grid_J.lat.flatten(), self.grid_J.lon.flatten(), refh)
            f1, f2 = apx.basevectors_qd(self.grid_J.lat.flatten(), self.grid_J.lon.flatten(), refh)
            f1 = f1/np.linalg.norm(f1, axis = 0)
            self.Le_J = De2 * f1[0].reshape((-1, 1)) + Dn2 * f1[1].reshape((-1, 1))
            self.Le_J = self.Le_J * lat_w(self.hemisphere * mlat).reshape((-1, 1))
            self.LTLe_J = self.Le_J.T.dot(self.Le_J)
            f2 = f2/np.linalg.norm(f2, axis = 0)
            self.Ln_J = De2 * f2[0].reshape((-1, 1)) + Dn2 * f2[1].reshape((-1, 1))
            self.LTLn_J = self.Ln_J.T.dot(self.Ln_J)


    def clear_model(self, Hall_Pedersen_conductance = None):
        """ Reset data and model vectors

        parameters
        ----------
        Hall_Pedersen_conductance: tuple, optional
            provide a tuple of functions of lat, lon that returns
            Hall and Pedersen conductances, respectively. If not provided,
            the previous conductance model is kept
        """
        self.m = None # clear electric field model parameters

        # dictionary of lists to store datasets in
        self.data = {'efield':[], 'convection':[], 'ground_mag':[], 'space_mag_full':[], 'space_mag_fac':[], 'fac':[]}

        # Hall and Pedersen conductance - either inversion or functions:
        if Hall_Pedersen_conductance != None:

            _h, _p = Hall_Pedersen_conductance
            self.hall_conductance     = lambda lon = self.grid_J.lon, lat = self.grid_J.lat: _h(lon, lat)
            self.pedersen_conductance = lambda lon = self.grid_J.lon, lat = self.grid_J.lat: _p(lon, lat)

    def save(self, time=0, parameters_to_save='all', **kwargs):
        """
        For saving the model and/or the lompe output. Calls lompe.utils.save_load_utiles.save_model
    
        Parameters
        ----------
        model : lompe.Emodel
            lompe model object.
        parameters_to_save : list, optional
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
        **kwargs : dict
            key arguments to be passed to the lompe.utils.save_load_utiles.save_model function.
            (read lompe.utils.save_load_utiles.save_model doc string for more information)
            
            Some possible kwargs (copied from lompe.utils.save_load_utiles.save_model doc string)
            file_name : str/bool, optional
                A string containing the path and name of the xarray file if you wish to save using this function. 
                The default is False and no save will be made.
            append : bool, optional
                If filename is provided the current dataset will be added on to the existing dataset if it exists. 
                The default is True and the dataset will be added to the existing.
    
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
        from lompe.utils import save_model
        return save_model(self, time=time, save=parameters_to_save, **kwargs)

    def reg_E(self, l1, l2, l3, E_reg=False):
        """Calculate the roughening matrix for E (normal) regularization"""
        LTL = 0
        if E_reg == False:            
            if l1 > 0:
                LTL_l1 = np.eye(self.GTG.shape[0])
                LTL += l1 * LTL_l1 / np.median(LTL_l1.diagonal())
            if l2 > 0:
                LTL += l2 * self.LTLe / np.median(self.LTLe.diagonal())
            if l3 > 0:
                LTL += l3 * self.LTLn / np.median(self.LTLn.diagonal())
        else:
            G_Ee, G_En = self._E_matrix()            
            if l1 > 0:
                #LTL_l1 = np.vstack((G_Ee, G_En)).T.dot(np.vstack((G_Ee, G_En)))
                LTL_l1 = (G_Ee+G_En).T.dot((G_Ee+G_En))
                #LTL_l1 = G_Ee.T.dot(G_Ee) + G_En.T.dot(G_En)
                LTL += l1 * LTL_l1 / np.median(LTL_l1.diagonal())
            if l2 > 0:
                G_Ee_e = self.Le_J.dot(G_Ee)
                G_En_e = self.Le_J.dot(G_En)
                LTL_l2 = G_Ee_e.T.dot(G_Ee_e) + G_En_e.T.dot(G_En_e)
                LTL += l2 * LTL_l2 / np.median(LTL_l2.diagonal())
            if l3 > 0:
                G_Ee_n = self.Ln_J.dot(G_Ee)
                G_En_n = self.Ln_J.dot(G_En)
                LTL_l3 = G_Ee_n.T.dot(G_Ee_n) + G_En_n.T.dot(G_En_n)
                LTL += l3 * LTL_l3 / np.median(LTL_l3.diagonal())            
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
    
    def ensure_tuple(self, value):
        """Ensure the value is a tuple of length 2."""
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f"Tuple {value} must have length 2.")
            return value
        return (value, 0)

    def joule_inversion_thing_16(self, l1=1e0, lj = 10**np.linspace(0, 1, 10), gtg_mag=0, step=1, threshold=1, IRLS_max=50, l1_redux=.5, LTL_E=0, LTL_FAC=0, E_reg=False, FAC_reg=False, joule_reg=True):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        if joule_reg:
            SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
            Q = np.diag(SP)**2
        
            G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                               current_type = 'curl_free',
                                               RI = self.R,
                                               singularity_limit = self.secs_singularity_limit)
        else:
            G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                               current_type = 'curl_free',
                                               RI = self.R,
                                               singularity_limit = self.secs_singularity_limit)
        
        LTL = l1_redux * LTL_E
        if FAC_reg:
            LTL += LTL_FAC
        
        LTL = gtg_mag * LTL
        
        self.dnorms = np.zeros(len(lj))
        self.mnorms = np.zeros(len(lj))
        self.ms = [0]*(len(lj)+1)
        self.ms[0] = self.m + 0
        self.counts = np.zeros(len(lj))
        
        for i, lj_c in enumerate(lj):
            
            percent_change = 100
            counter = 0
            if i == 0:
                m_old = self.m + 0
            elif i == 1:
                step /= 2
            
            while (percent_change > threshold) and (counter < IRLS_max):
                
                ve = G_Ee @ m_old
                vn = G_En @ m_old
                
                if joule_reg:
                    Qee = Q*ve*ve
                    Qnn = Q*vn*vn
                    Qen = Q*ve*vn
                    Qne = Qen
                
                    Jac =  4*(Qee*ve).T.dot(G_Ee)
                    Jac += 4*(Qnn*vn).T.dot(G_En)
                    Jac += 2*(Qee*vn).T.dot(G_En)
                    Jac += 2*(Qnn*ve).T.dot(G_Ee)
                
                    Hes =  12*G_Ee.T @ np.diag(Qee) @ G_Ee
                    Hes += 12*G_En.T @ np.diag(Qnn) @ G_En
                    Hes +=  2*G_En.T @ np.diag(Qee) @ G_En
                    Hes +=  4*G_Ee.T @ np.diag(Qen) @ G_En
                    Hes +=  2*G_Ee.T @ np.diag(Qnn) @ G_Ee
                    Hes +=  4*G_En.T @ np.diag(Qne) @ G_Ee
                
                else:
                    Jac = 2*ve.T.dot(G_Ee) + 2*vn.T.dot(G_En) + ve.T.dot(G_En) + vn.T.dot(G_Ee)                    
                    Hes = 2*G_Ee.T.dot(G_Ee) + 2*G_En.T.dot(G_En) + G_Ee.T.dot(G_En) + G_En.T.dot(G_Ee)
                
                reg_mag = np.median(abs(np.diag(Hes)))
                
                Jac = Jac / reg_mag * gtg_mag
                Hes = Hes / reg_mag * gtg_mag
        
                denom = 2*self.GTG + LTL + lj_c * Hes
                num =  2*self.GTG.dot(m_old) + lj_c * Hes.dot(m_old)
                num += step * (2*(self.GTd - self.GTG.dot(m_old)) - lj_c * Jac)
                
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                percent_change = np.max(abs((m_c - m_old) / m_old))*100
                norm_change = np.sum(abs(m_c - m_old)) / np.sum(abs(m_old)) * 100
                counter += 1
                m_old = m_c
                
                print('reg {}/{} : IRLS {} : dm {} : nc {}'.format(i+1, len(lj), counter, np.round(percent_change, 2), np.round(norm_change, 5)))
        
            self.ms[i+1] = m_c
            self.dnorms[i] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
            self.mnorms[i] = np.sqrt(m_c.T.dot(LTL + Hes).dot(m_c))
            self.counts[i] = counter
        
        return

    def joule_inversion_thing_15(self, l1=1e0, lj = 10**np.linspace(0, 1, 10), gtg_mag=0, step=1, threshold=1, IRLS_max=50, l1_redux=.5, LTL_E=0, LTL_FAC=0, E_reg=False, FAC_reg=False):
        """Carry out IRLS on Taylor expanded Joule heating"""
                
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        Q = np.diag(SP)**2
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        LTL = l1_redux * LTL_E
        if FAC_reg:
            LTL += LTL_FAC
        
        LTL = gtg_mag * LTL
        
        self.dnorms = np.zeros(len(lj))
        self.mnorms = np.zeros(len(lj))
        self.ms = [0]*(len(lj)+1)
        self.ms[0] = self.m + 0
        self.counts = np.zeros(len(lj))
        
        for i, lj_c in enumerate(lj):
            
            percent_change = 100
            counter = 0
            m_old = self.m + 0
            
            while (percent_change > threshold) and (counter < IRLS_max):
                
                ve = G_Ee @ m_old
                vn = G_En @ m_old
                Qee = Q*ve*ve
                Qnn = Q*vn*vn
                Qen = Q*ve*vn
                Qne = Qen
                
                Jac =  4*(Qee*ve).T.dot(G_Ee)
                Jac += 4*(Qnn*vn).T.dot(G_En)
                Jac += 2*(Qee*vn).T.dot(G_En)
                Jac += 2*(Qnn*ve).T.dot(G_Ee)
                
                Hes =  12*G_Ee.T @ np.diag(Qee) @ G_Ee
                Hes += 12*G_En.T @ np.diag(Qnn) @ G_En
                Hes +=  2*G_En.T @ np.diag(Qee) @ G_En
                Hes +=  4*G_Ee.T @ np.diag(Qen) @ G_En
                Hes +=  2*G_Ee.T @ np.diag(Qnn) @ G_Ee
                Hes +=  4*G_En.T @ np.diag(Qne) @ G_Ee
                
                reg_mag = np.median(abs(np.diag(Hes)))
                
                Jac = Jac / reg_mag * gtg_mag
                Hes = Hes / reg_mag * gtg_mag
        
                denom = 2*self.GTG + LTL + lj_c * Hes
                num =  2*self.GTG.dot(m_old) + lj_c * Hes.dot(m_old)
                num += step * (2*(self.GTd - self.GTG.dot(m_old)) - lj_c * Jac)
                
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                percent_change = np.max(abs((m_c - m_old) / m_old))*100
                norm_change = np.sum(abs(m_c - m_old)) / np.sum(abs(m_old)) * 100
                counter += 1
                m_old = m_c
                
                print('reg {}/{} : IRLS {} : dm {} : nc {}'.format(i+1, len(lj), counter, np.round(percent_change, 2), np.round(norm_change, 5)))
        
            self.ms[i+1] = m_c
            self.dnorms[i] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
            self.mnorms[i] = np.sqrt(m_c.T.dot(LTL + Hes).dot(m_c))
            self.counts[i] = counter
        
        return

    def joule_inversion_thing_14(self, l1=1e0, lj = 10**np.linspace(0, 1, 10), gtg_mag=0, step=1, threshold=1, IRLS_max=50, l1_redux=.5, LTL_E=0, LTL_FAC=0, E_reg=False, FAC_reg=False):
        """Carry out IRLS on Taylor expanded Joule heating"""
                
        #SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        #Q = np.diag(SP)**2
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        LTL = l1_redux * LTL_E
#        if E_reg:
#            LTL += 
        if FAC_reg:
            LTL += LTL_FAC
        
        LTL = gtg_mag * LTL
        
        #reg_l1 = self.reg_E(l1=l1_redux*l1, l2=0, l3=0, E_reg=E_reg)
        #reg_l1 = gtg_mag*reg_l1
        
        self.dnorms = np.zeros(len(lj))
        self.mnorms = np.zeros(len(lj))
        self.ms = [0]*(len(lj)+1)
        self.ms[0] = self.m + 0
        self.counts = np.zeros(len(lj))
        
        for i, lj_c in enumerate(lj):
            
            percent_change = 100
            counter = 0
            m_old = self.m + 0
            
            while (percent_change > threshold) and (counter < IRLS_max):
                
                ve = G_Ee @ m_old
                vn = G_En @ m_old
                
                Jac = 2*ve.T.dot(G_Ee) + 2*vn.T.dot(G_En) + ve.T.dot(G_En) + vn.T.dot(G_Ee)
                
                Hes = 2*G_Ee.T.dot(G_Ee) + 2*G_En.T.dot(G_En) + G_Ee.T.dot(G_En) + G_En.T.dot(G_Ee)
                
                reg_mag = np.median(abs(np.diag(Hes)))
                
                Jac = Jac / reg_mag * gtg_mag
                Hes = Hes / reg_mag * gtg_mag
        
                denom = 2*self.GTG + LTL + lj_c * Hes
                num =  2*self.GTG.dot(m_old) + lj_c * Hes.dot(m_old)
                num += step * (2*(self.GTd - self.GTG.dot(m_old)) - lj_c * Jac)
                
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                percent_change = np.max(abs((m_c - m_old) / m_old))*100
                norm_change = np.sum(abs(m_c - m_old)) / np.sum(abs(m_old)) * 100
                counter += 1
                m_old = m_c
                
                print('reg {}/{} : IRLS {} : dm {} : nc {}'.format(i+1, len(lj), counter, np.round(percent_change, 2), np.round(norm_change, 5)))
        
            self.ms[i+1] = m_c
            self.dnorms[i] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
            self.mnorms[i] = np.sqrt(m_c.T.dot(LTL + Hes).dot(m_c))
            self.counts[i] = counter
        
        return

    def joule_inversion_thing_13(self, l1=1e0, lj = 10**np.linspace(0, 1, 10), gtg_mag=0, step=1, threshold=1, IRLS_max=50, l1_redux=.5, E_reg=False):
        """Carry out IRLS on Taylor expanded Joule heating"""
                
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        Q = np.diag(SP)**2
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        reg_l1 = self.reg_E(l1=l1_redux*l1, l2=0, l3=0, E_reg=E_reg)
        reg_l1 = gtg_mag*reg_l1
        
        self.dnorms = np.zeros(len(lj))
        self.mnorms = np.zeros(len(lj))
        self.ms = [0]*(len(lj)+1)
        self.ms[0] = self.m + 0
        self.counts = np.zeros(len(lj))
        
        for i, lj_c in enumerate(lj):
            
            percent_change = 100
            counter = 0
            m_old = self.m + 0
            
            while (percent_change > threshold) and (counter < IRLS_max):
                
                ve = G_Ee @ m_old
                vn = G_En @ m_old
                
                Jac = 2*ve.T.dot(G_Ee) + 2*vn.T.dot(G_En) + ve.T.dot(G_En) + vn.T.dot(G_Ee)
                
                Hes = 2*G_Ee.T.dot(G_Ee) + 2*G_En.T.dot(G_En) + G_Ee.T.dot(G_En) + G_En.T.dot(G_Ee)
                
                reg_mag = np.median(abs(np.diag(Hes)))
                
                Jac = Jac / reg_mag * gtg_mag
                Hes = Hes / reg_mag * gtg_mag
        
                denom = 2*self.GTG + reg_l1 + lj_c * Hes
                num =  2*self.GTG.dot(m_old) + lj_c * Hes.dot(m_old)
                num += step * (2*(self.GTd - self.GTG.dot(m_old)) - lj_c * Jac)
                
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                percent_change = np.max(abs((m_c - m_old) / m_old))*100
                norm_change = np.sum(abs(m_c - m_old)) / np.sum(abs(m_old)) * 100
                counter += 1
                m_old = m_c
                
                print('reg {}/{} : IRLS {} : dm {} : nc {}'.format(i+1, len(lj), counter, np.round(percent_change, 2), np.round(norm_change, 5)))
        
            self.ms[i+1] = m_c
            self.dnorms[i] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
            self.mnorms[i] = np.sqrt(m_c.T.dot(Hes).dot(m_c))
            self.counts[i] = counter
        
        return

    def joule_inversion_thing_12(self, l1=1e0, lj = 10**np.linspace(0, 1, 10), gtg_mag=0, step=1, threshold=1, IRLS_max=50, l1_redux=.5, E_reg=False):
        """Carry out IRLS on Taylor expanded Joule heating"""
                
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        Q = np.diag(SP)**2
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        reg_l1 = self.reg_E(l1=l1_redux*l1, l2=0, l3=0, E_reg=E_reg)
        reg_l1 = gtg_mag*reg_l1
        
        self.dnorms = np.zeros(len(lj))
        self.mnorms = np.zeros(len(lj))
        self.ms = [0]*(len(lj)+1)
        self.ms[0] = self.m + 0
        self.counts = np.zeros(len(lj))
        
        for i, lj_c in enumerate(lj):
            
            percent_change = 100
            counter = 0
            m_old = self.m + 0
            
            while (percent_change > threshold) and (counter < IRLS_max):
                
                ve = G_Ee @ m_old
                vn = G_En @ m_old
                Qee = Q*ve*ve
                Qnn = Q*vn*vn
                Qen = Q*ve*vn
                Qne = Qen
                
                Jac =  4*(Qee*ve).T.dot(G_Ee)
                Jac += 4*(Qnn*vn).T.dot(G_En)
                Jac += 2*(Qee*vn).T.dot(G_En)
                Jac += 2*(Qnn*ve).T.dot(G_Ee)
                
                Hes =  12*G_Ee.T @ np.diag(Qee) @ G_Ee
                Hes += 12*G_En.T @ np.diag(Qnn) @ G_En
                Hes +=  2*G_En.T @ np.diag(Qee) @ G_En
                Hes +=  4*G_Ee.T @ np.diag(Qen) @ G_En
                Hes +=  2*G_Ee.T @ np.diag(Qnn) @ G_Ee
                Hes +=  4*G_En.T @ np.diag(Qne) @ G_Ee
                
                reg_mag = np.median(abs(np.diag(Hes)))
                
                Jac = Jac / reg_mag * gtg_mag
                Hes = Hes / reg_mag * gtg_mag
        
                denom = 2*self.GTG + reg_l1 + lj_c * Hes
                num =  2*self.GTG.dot(m_old) + lj_c * Hes.dot(m_old)
                num += step * (2*(self.GTd - self.GTG.dot(m_old)) - lj_c * Jac)
                
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                percent_change = np.max(abs((m_c - m_old) / m_old))*100
                norm_change = np.sum(abs(m_c - m_old)) / np.sum(abs(m_old)) * 100
                counter += 1
                m_old = m_c
                
                print('reg {}/{} : IRLS {} : dm {} : nc {}'.format(i+1, len(lj), counter, np.round(percent_change, 2), np.round(norm_change, 5)))
        
            self.ms[i+1] = m_c
            self.dnorms[i] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
            self.mnorms[i] = np.sqrt(m_c.T.dot(Hes).dot(m_c))
            self.counts[i] = counter
        
        return

    def joule_inversion_thing_11(self, IRLS_iter=10, l1=1e0, lj = 10**np.linspace(0, 1, 10), gtg_mag=0, step = 1):
        """Carry out IRLS on Taylor expanded Joule heating"""
                
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        Q = np.diag(SP)**2
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        self.ljs = []
        self.dnorms = []
        self.mnorms = []
        
        for i in range(IRLS_iter):
            
            print(i+1,'/',IRLS_iter)
            
            ve = G_Ee @ self.ms[i]
            vn = G_En @ self.ms[i]
            Qee = Q*ve*ve
            Qnn = Q*vn*vn
            Qen = Q*ve*vn
            Qne = Qen
            
            Jac =  4*(Qee*ve).T.dot(G_Ee)
            Jac += 4*(Qnn*vn).T.dot(G_En)
            Jac += 2*(Qee*vn).T.dot(G_En)
            Jac += 2*(Qnn*ve).T.dot(G_Ee)
            
            Hes =  12*G_Ee.T @ np.diag(Qee) @ G_Ee
            Hes += 12*G_En.T @ np.diag(Qnn) @ G_En
            Hes +=  2*G_En.T @ np.diag(Qee) @ G_En
            Hes +=  4*G_Ee.T @ np.diag(Qen) @ G_En
            Hes +=  2*G_Ee.T @ np.diag(Qnn) @ G_Ee
            Hes +=  4*G_En.T @ np.diag(Qne) @ G_Ee
            
            reg_mag = np.median(abs(np.diag(Hes)))
            
            Jac = Jac / reg_mag * gtg_mag
            Hes = Hes / reg_mag * gtg_mag
            
            dnorm = np.zeros(lj.size)
            mnorm = np.zeros(lj.size)
            models = []
            
            if i == 0:
                jh = 1
                jh2 = 1
            
            for j, lj_c in enumerate(lj):
                print(i+1,'/',IRLS_iter, ' : ', j+1, ' / ', lj.size, ' : ', np.round(jh, 2), ' : ', jh2, ' : ', step)
                
                denom = 2*self.GTG + 0.1*gtg_mag*l1*np.eye(len(self.m)) + lj_c * Hes
                num =  2*self.GTG.dot(self.ms[i]) + lj_c * Hes.dot(self.ms[i])
                num += step * (2*(self.GTd - self.GTG.dot(self.ms[i])) - lj_c * Jac)
                
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                dnorm[j] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
                mnorm[j] = np.sqrt(m_c.T.dot(Hes).dot(m_c))
                models.append(m_c)
                
            kf = KF(np.log10(dnorm), np.log10(mnorm))
            kf.find_knee()
            opt_id = np.argmin(abs(dnorm - 10**kf.knee[0]))
            
            lj_c = lj[opt_id]
            m_c = models[opt_id]
            
            self.ms.append(m_c)
            self.ljs.append(lj_c)
            self.dnorms.append(dnorm)
            self.mnorms.append(mnorm)

            jh = SP.dot(np.diag(G_Ee.dot(m_c)).dot(G_Ee).dot(m_c) + np.diag(G_En.dot(m_c)).dot(G_En).dot(m_c))
            jh2 = np.sum(jh**2)
            jh = np.sum(jh)

        return

    def joule_inversion_thing_10(self, IRLS_iter=10, lj=1e0, gtg_mag=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        Q = np.diag(SP)**2
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        self.ljs = []
        self.dnorms = []
        self.mnorms = []
        self.gcvs = []
        
        for i in range(IRLS_iter):
            
            ve = G_Ee @ self.ms[i]
            vn = G_En @ self.ms[i]
            Qee = Q*ve*ve
            Qnn = Q*vn*vn
            Qen = Q*ve*vn
            Qne = Qen
            
            Jac =  4*(Qee*ve).T.dot(G_Ee)
            Jac += 4*(Qnn*vn).T.dot(G_En)
            Jac += 2*(Qee*vn).T.dot(G_En)
            Jac += 2*(Qnn*ve).T.dot(G_Ee)
            
            Hes =  12*G_Ee.T @ np.diag(Qee) @ G_Ee
            Hes += 12*G_En.T @ np.diag(Qnn) @ G_En
            Hes +=  2*G_En.T @ np.diag(Qee) @ G_En
            Hes +=  4*G_Ee.T @ np.diag(Qen) @ G_En
            Hes +=  2*G_Ee.T @ np.diag(Qnn) @ G_Ee
            Hes +=  4*G_En.T @ np.diag(Qne) @ G_Ee
            
            #reg_mag = np.median(abs(np.diag(Hes)))
            
            if i == 0:
                step = .5
                jh = 1
                jh2 = 1
            
            print(i+1,'/',IRLS_iter, ' : ', np.round(jh, 2), ' : ', np.round(jh2, 5), ' : ', step)
            
            denom = 2*self.GTG + lj * Hes + 2e-1*gtg_mag*np.eye(len(self.m))
            num =  2*self.GTG.dot(self.ms[i]) + lj * Hes.dot(self.ms[i])
            num += step * (2*(self.GTd - self.GTG.dot(self.ms[i])) - lj * Jac)
            
            m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
            
            self.ms.append(m_c)

            jh = SP.dot(np.diag(G_Ee.dot(m_c)).dot(G_Ee).dot(m_c) + np.diag(G_En.dot(m_c)).dot(G_En).dot(m_c))
            jh2 = np.sum(jh**2)
            jh = np.sum(jh)
            '''
            if jh < 12:
                step = .01
            elif jh < 25:
                step = .05
            elif jh < 50:
                step = .1
            else:
                step = .5
            '''
            '''
            if (jh < 12) and (step >= .05):
                step = .05
            elif (jh < 25) and (step >= .2):
                step = .2
            '''
            if (jh < 5) and (step >= .1):
                step = .1

        return

    def joule_inversion_thing_9(self, IRLS_iter=10, lj = 10**np.linspace(0, 1, 10), gtg_mag=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        def calc_curvature(rnorm, mnorm):
            x_t = np.gradient(rnorm)
            y_t = np.gradient(mnorm)
            xx_t = np.gradient(x_t)
            yy_t = np.gradient(y_t)
            curvature = (xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
            return curvature
        
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        Q = np.diag(SP)**2
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        self.ljs = []
        self.dnorms = []
        self.mnorms = []
        self.gcvs = []
        
        for i in range(IRLS_iter):
            
            print(i+1,'/',IRLS_iter)
            
            ve = G_Ee @ self.ms[i]
            vn = G_En @ self.ms[i]
            Qee = Q*ve*ve
            Qnn = Q*vn*vn
            Qen = Q*ve*vn
            Qne = Qen
            
            Jac =  4*(Qee*ve).T.dot(G_Ee)
            Jac += 4*(Qnn*vn).T.dot(G_En)
            Jac += 2*(Qee*vn).T.dot(G_En)
            Jac += 2*(Qnn*ve).T.dot(G_Ee)
            
            Hes =  12*G_Ee.T @ np.diag(Qee) @ G_Ee
            Hes += 12*G_En.T @ np.diag(Qnn) @ G_En
            Hes +=  2*G_En.T @ np.diag(Qee) @ G_En
            Hes +=  4*G_Ee.T @ np.diag(Qen) @ G_En
            Hes +=  2*G_Ee.T @ np.diag(Qnn) @ G_Ee
            Hes +=  4*G_En.T @ np.diag(Qne) @ G_Ee
            
            #reg_mag = np.median(abs(np.diag(Hes)))
            
            dnorm = np.zeros(lj.size)
            mnorm = np.zeros(lj.size)
            gcv = np.zeros(lj.size)
            models = []
            
            if i == 0:
                step = .5
                jh = 1
                jh2 = 1
            
            for j, lj_c in enumerate(lj):
                print(i+1,'/',IRLS_iter, ' : ', j+1, ' / ', lj.size, ' : ', np.round(jh, 2), ' : ', jh2, ' : ', step)
                
                denom = 2*self.GTG + lj_c * Hes
                num =  2*self.GTG.dot(self.ms[i]) + lj_c * Hes.dot(self.ms[i])
                num += step * (2*(self.GTd - self.GTG.dot(self.ms[i])) - lj_c * Jac)
                
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                dnorm[j] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
                mnorm[j] = np.sqrt(m_c.T.dot(Hes).dot(m_c))
                models.append(m_c)
                
            
            kneed = KneeLocator(np.log10(dnorm), np.log10(mnorm), curve='convex', direction='decreasing')
            opt_id = np.argmin(abs(np.log10(dnorm) - kneed.knee))
            '''
            kf = KF(np.log10(dnorm), np.log10(mnorm))
            kf.find_knee()
            opt_id = np.argmin(abs(np.log10(dnorm) - kf.knee[0]))
            '''
            '''
            curv = calc_curvature(np.log10(dnorm), np.log10(mnorm))
            opt_id = np.argmin(curv)
            '''
            
            lj_c = lj[opt_id]
            m_c = models[opt_id]
            
            self.ms.append(m_c)
            self.ljs.append(lj_c)
            self.dnorms.append(dnorm)
            self.mnorms.append(mnorm)
            self.gcvs.append(gcv)

            jh = SP.dot(np.diag(G_Ee.dot(m_c)).dot(G_Ee).dot(m_c) + np.diag(G_En.dot(m_c)).dot(G_En).dot(m_c))
            jh2 = np.sum(jh**2)
            jh = np.sum(jh)
            '''
            if jh < 12:
                step = .01
            elif jh < 25:
                step = .05
            elif jh < 50:
                step = .1
            else:
                step = .5
            '''
            '''
            if (jh < 12) and (step >= .05):
                step = .05
            elif (jh < 25) and (step >= .2):
                step = .2
            '''
            if (jh < 5) and (step >= .1):
                step = .1

        return

    def joule_inversion_thing_8(self, IRLS_iter=10, lj = 10**np.linspace(0, 1, 10), gtg_mag=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        self.ljs = []
        self.dnorms = []
        self.mnorms = []
        self.gcvs = []
        
        for i in range(IRLS_iter):
            
            print(i+1,'/',IRLS_iter)
            
            v = G_Ee @ self.ms[i]
            Hes = 12*G_Ee.T @ np.diag((SP**2) @ (v**2)) @ G_Ee
            Jac = 4*G_Ee.T @ np.diag((SP**2) @ (v**3))
            
            v = G_En @ self.ms[i]            
            Hes += 12*G_En.T @ np.diag((SP**2) @ (v**2)) @ G_En
            Jac += 4*G_En.T @ np.diag((SP**2) @ (v**3))
            
            Jac = np.sum(Jac, axis=1)
            
            reg_mag = np.median(abs(np.diag(Hes)))
            
            dnorm = np.zeros(lj.size)
            mnorm = np.zeros(lj.size)
            gcv = np.zeros(lj.size)
            models = []
            
            step = 1
            
            for j, lj_c in enumerate(lj):
                print(i+1,'/',IRLS_iter, ' : ', j+1, ' / ', lj.size)
                
                denom = 2*self.GTG + lj_c * Hes
                num =  2*self.GTG.dot(self.ms[i]) + lj_c * Hes.dot(self.ms[i])
                num += step * (2*(self.GTd - self.GTG.dot(self.ms[i])) - lj_c * Jac)
                
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                dnorm[j] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
                mnorm[j] = np.sqrt(m_c.T.dot(Hes).dot(m_c))
                models.append(m_c)
                
            kf = KF(np.log10(dnorm), np.log10(mnorm))
            kf.find_knee()
            opt_id = np.argmin(abs(np.log10(dnorm) - kf.knee[0]))
            
            lj_c = lj[opt_id]
            m_c = models[opt_id]
            
            self.ms.append(m_c)
            self.ljs.append(lj_c)
            self.dnorms.append(dnorm)
            self.mnorms.append(mnorm)
            self.gcvs.append(gcv)
            
            jh = np.sum(SP.dot(np.diag(G_Ee.dot(m_c)).dot(G_Ee).dot(m_c) + np.diag(G_En.dot(m_c)).dot(G_En).dot(m_c)))
            if jh < 50:
                step = .5
            elif jh < 25:
                step = .2
            elif jh < 12:
                step = .1
            else:
                step = 1
            
        return

    def joule_inversion_thing_7(self, IRLS_iter=10, lj = 10**np.linspace(0, 1, 10), gtg_mag=0, step=1):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        self.ljs = []
        self.dnorms = []
        self.mnorms = []
        self.gcvs = []
        
        def calc_curvature(rnorm, mnorm):
            x_t = np.gradient(rnorm)
            y_t = np.gradient(mnorm)
            xx_t = np.gradient(x_t)
            yy_t = np.gradient(y_t)
            curvature = (xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
            return curvature
        
        for i in range(IRLS_iter):
            
            print(i+1,'/',IRLS_iter)
            
            v = G_Ee @ self.ms[i]
            Hes = 12*G_Ee.T @ np.diag((SP**2) @ (v**2)) @ G_Ee
            Jac = 4*G_Ee.T @ np.diag((SP**2) @ (v**3))
            
            v = G_En @ self.ms[i]            
            Hes += 12*G_En.T @ np.diag((SP**2) @ (v**2)) @ G_En
            Jac += 4*G_En.T @ np.diag((SP**2) @ (v**3))
            
            Jac = np.sum(Jac, axis=1)
            
            reg_mag = np.median(abs(np.diag(Hes)))
            
            #Jac = gtg_mag * Jac / reg_mag
            #Hes = gtg_mag * Hes / reg_mag
            
            dnorm = np.zeros(lj.size)
            mnorm = np.zeros(lj.size)
            gcv = np.zeros(lj.size)
            for j, lj_c in enumerate(lj):
                print(i+1,'/',IRLS_iter, ' : ', j+1, ' / ', lj.size)
                
                denom = 2*self.GTG + lj_c * Hes
                num =  2*self.GTG.dot(self.ms[i]) + lj_c * Hes.dot(self.ms[i])
                num += step * (2*(self.GTd - self.GTG.dot(self.ms[i])) - lj_c * Jac)
                
                #num = 2*self.GTd + lj_c * Hes.dot(self.ms[i]) - lj_c * Jac
                m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
                
                dnorm[j] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
                mnorm[j] = np.sqrt(m_c.T.dot(Hes).dot(m_c))
                                
                #gcv[j] = len(self._d) * (self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)) / np.trace(1 - self._G @ scipy.linalg.lstsq(2*self.GTG + lj_c * Hes, 2*self._G.T, lapack_driver='gelsy', check_finite=False)[0])
                
            
            kf = KF(np.log10(dnorm), np.log10(mnorm))
            kf.find_knee()
            opt_id = np.argmin(abs(np.log10(dnorm) - kf.knee[0]))
            
            #opt_id = np.argmin(gcv)
            #curv = calc_curvature(np.log10(dnorm), np.log10(mnorm))
            #opt_id = np.argmin(curv)
            #opt_id = np.argmin(abs(dnorm - KneeLocator(dnorm, mnorm, curve='convex', direction='decreasing').knee))
            #opt_id = 0
            lj_c = lj[opt_id]
            #m_c = scipy.linalg.lstsq(2*self.GTG + lj_c * Hes, 2*self.GTd + lj_c * Hes.dot(self.ms[i]) - lj_c * Jac, lapack_driver='gelsy', check_finite=False)[0]
            
            denom = 2*self.GTG + lj_c * Hes
            num =  2*self.GTG.dot(self.ms[i]) + lj_c * Hes.dot(self.ms[i])
            num += step * (2*(self.GTd - self.GTG.dot(self.ms[i])) - lj_c * Jac)            
            #num = 2*self.GTd + lj_c * Hes.dot(self.ms[i]) - lj_c * Jac
            m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
            
            self.ms.append(m_c)
            self.ljs.append(lj_c)
            self.dnorms.append(dnorm)
            self.mnorms.append(mnorm)
            self.gcvs.append(gcv)
    
        return


    def joule_inversion_thing_6(self, IRLS_iter=10, lj = 10**np.linspace(0, 1, 10), gtg_mag=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        #SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        SP = np.diag(np.hstack((self.pedersen_conductance(self.lon_J, self.lat_J), 
                                self.pedersen_conductance(self.lon_J, self.lat_J))))
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        self.ljs = []
        self.dnorms = []
        self.mnorms = []
                
        for i in range(IRLS_iter):
            
            print(i+1,'/',IRLS_iter)
            
            #Jac = 2 * (G_Ee.T.dot(np.diag(G_Ee.dot(self.ms[i])).T).dot(SP).T.dot(SP).dot(np.diag(G_Ee.dot(self.ms[i]))).dot(G_Ee).dot(self.ms[i]) +
            #           G_En.T.dot(np.diag(G_En.dot(self.ms[i])).T).dot(SP).T.dot(SP).dot(np.diag(G_En.dot(self.ms[i]))).dot(G_En).dot(self.ms[i]))
            
            #Hes = 2 * (G_Ee.T @ np.diag(G_Ee @ self.ms[i]).T @ SP.T @ SP @ np.diag(G_Ee @ self.ms[i]) @ G_Ee + 
            #           G_En.T @ np.diag(G_En @ self.ms[i]).T @ SP.T @ SP @ np.diag(G_En @ self.ms[i]) @ G_En)
            
            Hes = 2 * np.vstack((G_Ee, G_En)).T.dot(SP).dot(np.vstack((G_Ee, G_En)))
            
            #Hes = 2 * (G_Ee.T.dot(np.diag(G_Ee.dot(self.ms[i])).T).dot(SP).T.dot(SP).dot(np.diag(G_Ee.dot(self.ms[i]))).dot(G_Ee) +
            #           G_En.T.dot(np.diag(G_En.dot(self.ms[i])).T).dot(SP).T.dot(SP).dot(np.diag(G_En.dot(self.ms[i]))).dot(G_En))
            #Hes = gtg_mag * Hes / np.median(abs(np.diag(Hes)))
            
            dnorm = np.zeros(lj.size)
            mnorm = np.zeros(lj.size)
            for j, lj_c in enumerate(lj):
                print(i+1,'/',IRLS_iter, ' : ', j+1, ' / ', lj.size)
                #m_c = scipy.linalg.lstsq(self.GTG + lj_c * Hes, self.GTd, lapack_driver='gelsy', check_finite=False)[0]
                m_c = scipy.linalg.lstsq(self.GTG + lj_c * Hes, self.GTd)[0]
                
                dnorm[j] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
                mnorm[j] = np.sqrt(m_c.T.dot(Hes).dot(m_c))
            
            #opt_id = np.argmin(abs(dnorm - KneeLocator(dnorm, mnorm, curve='convex', direction='decreasing').knee))
            opt_id = 0
            lj_c = lj[opt_id]
            #m_c = scipy.linalg.lstsq(self.GTG + lj_c * Hes, self.GTd, lapack_driver='gelsy', check_finite=False)[0]
            m_c = scipy.linalg.lstsq(self.GTG + lj_c * Hes, self.GTd)[0]
            
            self.ms.append(m_c)
            self.ljs.append(lj_c)
            self.dnorms.append(dnorm)
            self.mnorms.append(mnorm)
    
        return

    def joule_inversion_thing_5(self, IRLS_iter=10, lj = 10**np.linspace(0, 1, 10), gtg_mag=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        '''
        def compute_curvature(dnorm, mnorm):
            # Ensure the inputs are numpy arrays
            dnorm = np.array(dnorm)
            mnorm = np.array(mnorm)
            
            # Compute first derivatives
            dnorm_prime = np.gradient(dnorm)
            mnorm_prime = np.gradient(mnorm)
            
            # Compute second derivatives
            dnorm_double_prime = np.gradient(dnorm_prime)
            mnorm_double_prime = np.gradient(mnorm_prime)
            
            # Compute the curvature using the formula
            numerator = np.abs(dnorm_prime * mnorm_double_prime - mnorm_prime * dnorm_double_prime)
            denominator = (dnorm_prime**2 + mnorm_prime**2)**1.5
            curvature = numerator / denominator
            
            return curvature
        '''
        
        self.ms = [self.m]
        self.ljs = []
        self.dnorms = []
        self.mnorms = []
        #self.curvs = []
                
        for i in range(IRLS_iter):
            
            print(i+1,'/',IRLS_iter)
            
            D = SP.dot(np.diag(G_Ee.dot(self.ms[i]))).dot(G_Ee)
            H = SP.dot(np.diag(G_En.dot(self.ms[i]))).dot(G_En)
            LTL_J = D.T.dot(D) + D.T.dot(H) + H.T.dot(D) + H.T.dot(H)
            LTL_J = gtg_mag * LTL_J / np.median(np.diag(LTL_J))
            dnorm = np.zeros(lj.size)
            mnorm = np.zeros(lj.size)
            for j, lj_c in enumerate(lj):
                print(i+1,'/',IRLS_iter, ' : ', j+1, ' / ', lj.size)
                denom = self.GTG + 4*lj_c*LTL_J
                num = self.GTd + 2*lj_c*LTL_J.dot(self.ms[i])
                m_c = scipy.linalg.lstsq(denom, num)[0]
                
                dnorm[j] = np.sqrt((self._d - self._G.dot(m_c)).T.dot(np.diag(self._w)).dot(self._d - self._G.dot(m_c)))
                mnorm[j] = np.sqrt(m_c.T.dot(LTL_J).dot(m_c))
            
            
            #self.curvs.append(compute_curvature(dnorm, mnorm))
            #lj_c = lj[np.argmax(self.curvs[i])]
            opt_id = np.argmin(abs(dnorm - KneeLocator(dnorm, mnorm, curve='convex', direction='decreasing').knee))
            lj_c = lj[opt_id]
            denom = self.GTG + 4*lj_c*LTL_J
            num = self.GTd + 2*lj_c*LTL_J.dot(self.ms[i])
            #m_c = scipy.linalg.lstsq(denom, num)[0]
            m_c = scipy.linalg.lstsq(denom, num, lapack_driver='gelsy', check_finite=False)[0]
            #m_c = scipy.linalg.solve(denom, num)
            
            self.ms.append(m_c)
            self.ljs.append(lj_c)
            self.dnorms.append(dnorm)
            self.mnorms.append(mnorm)
    
        return

    def joule_inversion_thing_4(self, IRLS_iter=10, l1=0, l2=0, l3=0, lj = 0, gtg_mag=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        self.ljs = []
        LTL = gtg_mag * np.eye(self.GTG.shape[0])
        jh = np.sum(SP.dot(np.diag(G_Ee.dot(self.m)).dot(G_Ee).dot(self.m) + np.diag(G_En.dot(self.m)).dot(G_En).dot(self.m)))
        
        #ratio = 1 - np.linspace(0.01, 1, IRLS_iter)
        ratio = 1 - np.hstack((1/np.exp(np.linspace(0.01, 5, IRLS_iter-1)), 0))
        ratio = np.hstack((1/np.exp(np.linspace(0.01, 5, IRLS_iter-1)), 0))
        
        for i in range(IRLS_iter):
            
            print(i+1,'/',IRLS_iter)
            
            l1_new = l1*ratio[i]
            jh_c = .95*jh
            m_c = self.ms[i]+0
            if i == 0:
                lj_c = lj+0
            sign = 0
            count = 1
            
            while abs((1 - jh_c/jh)*100) > 5:
                
                if sign != 0:                    
                    if abs((1 - jh_c/jh)*100) < 10:
                        step = 0.02
                    elif abs((1 - jh_c/jh)*100) < 50:
                        step = 0.06
                    else:
                        step = 0.1
                    
                    lj_c += sign*lj_c*step
                    
                D = SP.dot(np.diag(G_Ee.dot(self.ms[i]))).dot(G_Ee)
                H = SP.dot(np.diag(G_En.dot(self.ms[i]))).dot(G_En)
                LTL_J = D.T.dot(D) + D.T.dot(H) + H.T.dot(D) + H.T.dot(H)
                LTL_J = gtg_mag * LTL_J / np.median(np.diag(LTL_J))
                denom = self.GTG + l1_new * LTL + 4*lj_c*LTL_J
                num = self.GTd + 2*lj_c*LTL_J.dot(self.ms[i])
                m_c = scipy.linalg.lstsq(denom, num)[0]
                jh_c = np.sum(SP.dot(np.diag(G_Ee.dot(m_c)).dot(G_Ee).dot(m_c) + np.diag(G_En.dot(m_c)).dot(G_En).dot(m_c)))
                
                print(i+1,'/',IRLS_iter,' : ',count,' : ', np.round(abs((1 - jh_c/jh)*100), 0), ' : ', np.round(jh_c, 2), ' / ', np.round(jh, 2), ' : ', np.round(lj_c, 4), ' : ', np.round(l1_new, 4))
                
                if sign == 0:
                    if jh_c < jh:
                        sign = -1
                    else:
                        sign = 1
                        
                count += 1
            
            self.ms.append(m_c)
            self.ljs.append(lj_c)
    
        return

    def joule_inversion_thing_3(self, IRLS_iter=10, l1=0, l2=0, l3=0, lj = 0, gtg_mag=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        
        for i in range(IRLS_iter):
            print(i+1,'/',IRLS_iter)
            
            l1_new = l1*(1 - np.linspace(0.01, 1, IRLS_iter)[i])
            lj_new = lj*(np.linspace(0.01, 1, IRLS_iter)[i])
            
            LTL = gtg_mag * np.eye(self.GTG.shape[0])
            
            D = SP.dot(np.diag(G_Ee.dot(self.ms[i]))).dot(G_Ee)
            H = SP.dot(np.diag(G_En.dot(self.ms[i]))).dot(G_En)
            
            LTL_J = D.T.dot(D) + D.T.dot(H) + H.T.dot(D) + H.T.dot(H)
            LTL_J = gtg_mag * LTL_J / np.median(np.diag(LTL_J))
            
            denom = self.GTG + l1_new * LTL + 4*lj_new*LTL_J
            num = self.GTd + 2*lj_new*LTL_J.dot(self.ms[i])
            
            self.ms.append(scipy.linalg.lstsq(denom, num)[0])
    
        return

    def joule_inversion_thing_2(self, IRLS_iter=10, lj = 0, LTL=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        SP = np.diag(self.pedersen_conductance(self.lon_J, self.lat_J))
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        
        self.ms = [self.m]
        
        for i in range(IRLS_iter):
            print(i+1,'/',IRLS_iter)
            
            D = SP.dot(np.diag(G_Ee.dot(self.ms[i]))).dot(G_Ee)
            H = SP.dot(np.diag(G_En.dot(self.ms[i]))).dot(G_En)
            
            LTL_J = D.T.dot(D) + D.T.dot(H) + H.T.dot(D) + H.T.dot(H)
            denom = self.GTG + LTL + 4*lj*LTL_J
            num = self.GTd + 2*lj*LTL_J.dot(self.ms[i])
            
            self.ms.append(scipy.linalg.lstsq(denom, num)[0])
    
        return
    
    def joule_inversion_thing(self, IRLS_iter=10, lj = 0, LTL=0):
        """Carry out IRLS on Taylor expanded Joule heating"""
        
        G_Ee, G_En = get_SECS_J_G_matrices(self.lat_J, self.lon_J, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)
        G_E = np.vstack((G_Ee, G_En))
        
        SP = self.pedersen_conductance(self.lon_J, self.lat_J)
        
        G_Je = SP[:, np.newaxis] * G_Ee
        G_Jn = SP[:, np.newaxis] * G_En
        G_J = np.vstack((G_Je, G_Jn))    
        del G_Ee, G_En, G_Je, G_Jn
        
        self.ms = [self.m]
        
        for i in range(IRLS_iter):
            print(i+1,'/',IRLS_iter)
            #A = self.ms[i].T.dot(G_J.T).dot(G_E)
            #print(A.shape)
            #print(self.ms[i].shape)
            #num = self.GTd + 4*lj*A.T.dot(A).dot(self.ms[i])
            #denom = self.GTG + 4*lj*A.T.dot(A)
            num = self.GTd - lj*G_J.T.dot(G_E).dot(self.ms[i])
            denom = self.GTG + LTL
            
            self.ms.append(scipy.linalg.lstsq(denom, num)[0])
    
        return
        
    def run_inversion(self, l1 = 0, l2 = 0, l3 = 0, lj = 0, IRLS_iter=10, E_reg=False, FAC_reg=False, joule_reg=False, qN=False, step=1, IRLS_max=50, threshold=1, l1_redux=0.5,
                      data_density_weight = True, perimeter_width = 10,
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

        **kwargs : dict
            key arguments to be passed to the scipy.linalg.lstsq (e.g., 'cond', 'lapack_driver').
            
        """

        # initialize G matrices
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

        # Reguarlization - start
        if not FAC_reg and (isinstance(l1, tuple) or isinstance(l2, tuple) or isinstance(l3, tuple)):
            raise ValueError('l1, l2, and l3 can only be tuple if FAC_reg=True')
        
        LTL_E   = 0
        LTL_FAC = 0
        if not FAC_reg:
            LTL_E = self.reg_E(l1, l2, l3, E_reg=E_reg)
        
        if FAC_reg and  any(isinstance(x, tuple) for x in (l1, l2, l3)):
            l1 = self.ensure_tuple(l1)
            l2 = self.ensure_tuple(l2)
            l3 = self.ensure_tuple(l3)
            LTL_FAC = self.reg_FAC(l1[0], l2[0], l3[0])
            LTL_E   = self.reg_E(l1[1], l2[1], l3[1], E_reg=E_reg)
        else:
            LTL_FAC = self.reg_FAC(l1, l2, l3)
        
        LTL = LTL_E + LTL_FAC
        
        # Reguarlization - end
        
        gtg_mag = np.median(np.diagonal(self.GTG))
        GG = self.GTG + LTL*gtg_mag

        if 'rcond' in kwargs.keys():
            warnings.warn("'rcond' keyword (and use of np.linalg.lstsq) is deprecated! Use kw 'cond' (for scipy.linalg.lstsq) instead")
            kwargs['cond'] = kwargs['rcond']
        if 'cond' not in kwargs.keys():
            kwargs['cond'] = None
        
        if 'lapack_driver' not in kwargs.keys():
            kwargs['lapack_driver'] = 'gelsd'

        self.Cmpost = scipy.linalg.lstsq(GG, np.eye(GG.shape[0]), **kwargs)[0]
        self.Rmatrix = self.Cmpost.dot(self.GTG)
        self.m = self.Cmpost.dot(self.GTd)

        if qN:
            #self.joule_inversion_thing_2(IRLS_iter=IRLS_iter, lj=lj, LTL=gtg_mag*LTL)
            #self.joule_inversion_thing_3(IRLS_iter=IRLS_iter, l1=l1, l2=l2, l3=l3, lj=lj, gtg_mag=gtg_mag)
            #self.joule_inversion_thing_4(IRLS_iter=IRLS_iter, l1=l1, l2=l2, l3=l3, lj=lj, gtg_mag=gtg_mag)
            #self.joule_inversion_thing_5(IRLS_iter=IRLS_iter, lj=lj, gtg_mag=gtg_mag)
            #self.joule_inversion_thing_6(IRLS_iter=IRLS_iter, lj=lj, gtg_mag=gtg_mag)
            #self.joule_inversion_thing_7(IRLS_iter=IRLS_iter, lj=lj, gtg_mag=gtg_mag, step=step)
            #self.joule_inversion_thing_8(IRLS_iter=IRLS_iter, lj=lj, gtg_mag=gtg_mag)
            #self.joule_inversion_thing_9(IRLS_iter=IRLS_iter, lj=lj, gtg_mag=gtg_mag)
            #self.joule_inversion_thing_10(IRLS_iter=IRLS_iter, lj=lj, gtg_mag=gtg_mag)
            #self.joule_inversion_thing_11(IRLS_iter=IRLS_iter, l1=l1, lj=lj, gtg_mag=gtg_mag, step=step)
            #self.joule_inversion_thing_12(l1=l1, lj=lj, gtg_mag=gtg_mag, step=step, threshold=threshold, IRLS_max=IRLS_max, l1_redux=l1_redux, E_reg=E_reg)
            #self.joule_inversion_thing_13(l1=l1, lj=lj, gtg_mag=gtg_mag, step=step, threshold=threshold, IRLS_max=IRLS_max, l1_redux=l1_redux, E_reg=E_reg)
            #self.joule_inversion_thing_14(l1=l1, lj=lj, gtg_mag=gtg_mag, step=step, threshold=threshold, IRLS_max=IRLS_max, l1_redux=l1_redux, LTL_E=LTL_E, LTL_FAC=LTL_FAC, E_reg=E_reg, FAC_reg=FAC_reg)
            #self.joule_inversion_thing_15(l1=l1, lj=lj, gtg_mag=gtg_mag, step=step, threshold=threshold, IRLS_max=IRLS_max, l1_redux=l1_redux, LTL_E=LTL_E, LTL_FAC=LTL_FAC, E_reg=E_reg, FAC_reg=FAC_reg)
            self.joule_inversion_thing_16(l1=l1, lj=lj, gtg_mag=gtg_mag, step=step, threshold=threshold, IRLS_max=IRLS_max, l1_redux=l1_redux, LTL_E=LTL_E, LTL_FAC=LTL_FAC, E_reg=E_reg, FAC_reg=FAC_reg, joule_reg=joule_reg)

        return (self.GTG, self.GTd)        

    def calc_resolution(self, innerGrid=True):
        
        '''
        Calculate spatial resolution following Madelaire et al. [2023]
        '''
        
        # Get res in km
        colatxi = 90 - self.grid_E.lat
        lonxi = self.grid_E.lon
        d2r = np.pi/180

        xxi = self.R*1e-3 * np.sin(colatxi*d2r) * np.cos(lonxi*d2r)
        yxi = self.R*1e-3 * np.sin(colatxi*d2r) * np.sin(lonxi*d2r)
        zxi = self.R*1e-3 * np.cos(colatxi*d2r)

        euclidxi = np.median(np.sqrt(np.diff(xxi, axis=1)**2 + np.diff(yxi, axis=1)**2 + np.diff(zxi,axis=1)**2))
        euclideta = np.median(np.sqrt(np.diff(xxi, axis=0)**2 + np.diff(yxi, axis=0)**2 + np.diff(zxi,axis=0)**2))
        
        # Left right function
        def left_right(PSF_i, fraq=0.5):
        
            i_max = np.argmax(PSF_i)    
            PSF_max = PSF_i[i_max]
            
            j = 0
            i_left = 0
            left_edge = True
            while (i_max - j) >= 0:
                if PSF_i[i_max - j] < fraq*PSF_max:
                
                    dPSF = PSF_i[i_max - j + 1] - PSF_i[i_max - j]
                    dx = (fraq*PSF_max - PSF_i[i_max - j]) / dPSF
                    i_left = i_max - j + dx
                
                    left_edge = False
                
                    break
                else:
                    j += 1

            j = 0
            i_right = len(PSF_i) - 1
            right_edge = True
            while (i_max + j) < len(PSF_i):
                if PSF_i[i_max + j] < fraq*PSF_max:
                
                    dPSF = PSF_i[i_max + j] - PSF_i[i_max + j - 1]
                    dx = (fraq*PSF_max - PSF_i[i_max + j - 1]) / dPSF
                    i_right = i_max + j - 1 + dx 
                
                    right_edge = False
                
                    break
                else:
                    j += 1
        
            flag = True
            if left_edge and right_edge:
                print('I think something is wrong')
                flag = False
            elif left_edge:
                i_left = i_max - (i_right - i_max)
                flag = False
            elif right_edge:
                i_right = i_max + (i_max - i_left)
                flag = False
        
            return i_left, i_right, i_max, flag
        
        # Allocate space
        xiRes = np.zeros(self.grid_E.shape)
        etaRes = np.zeros(self.grid_E.shape)
        xiResFlag = np.zeros(self.grid_E.shape)
        etaResFlag = np.zeros(self.grid_E.shape)
        resL = np.zeros(self.grid_E.shape)
        
        # Loop over all PSFs
        for i in range(xiRes.size):
                        
            row = i//xiRes.shape[1]
            col = i%xiRes.shape[1]
            
            PSF = abs(self.Rmatrix[:, i]).reshape(self.grid_E.shape)
            
            ii = np.argmax(PSF)
            rowPSF = ii//self.grid_E.shape[1]
            colPSF = ii%self.grid_E.shape[1]
            
            dxi = abs(colPSF - col) * euclidxi
            deta = abs(rowPSF - row) * euclideta
            
            resL[row, col] = np.sqrt(dxi**2 + deta**2)
            
            PSF_xi = np.sum(PSF, axis=0)
            if innerGrid:
                PSF_xi[0] = 0.99*np.max(PSF_xi[1:-1])
                PSF_xi[-1] = 0.99*np.max(PSF_xi[1:-1])
            i_left, i_right, i_max, flag = left_right(PSF_xi)
            xiRes[row, col] = euclidxi * (i_right - i_left)
            xiResFlag[row, col] = flag
            
            PSF_eta = np.sum(PSF, axis=1)
            if innerGrid:
                PSF_eta[0] = 0.99*np.max(PSF_eta[1:-1])
                PSF_eta[-1] = 0.99*np.max(PSF_eta[1:-1])
            i_left, i_right, i_max, flag = left_right(PSF_eta)
            etaRes[row, col] = euclideta * (i_right - i_left)
            etaResFlag[row, col] = flag
        
        if innerGrid:
            xiResFlag[:, [0, -1]] = 0
            xiResFlag[[0, -1], :] = 0
            etaResFlag[:, [0, -1]] = 0            
            etaResFlag[[0, -1], :] = 0
        
        self.xiRes = xiRes
        self.etaRes = etaRes
        self.xiResFlag = xiResFlag
        self.etaResFlag = etaResFlag
        self.resL = resL

    def add_data(self, *datasets):
        """
        Add object of type lompe.Data to the model.
        If the function call is successful, the data will be used
        in next call to self.run_inversion()

        Parameters
        ----------
        datasets : lompe.Data
            one or more lompe datasets

        Note
        ----
        To remove data from model object without erasing all other
        matrices, call self.clear_model()

        """

        for dataset in datasets:
            if not dataset.isvalid:
                raise Exception('invalid dataset')

            dtype = dataset.datatype.lower()


            if dtype in self.data.keys():
                self.data[dtype].append(dataset)
            else:
                print('You passed {}, which is not in {} - ignored'.format(dtype, list(self.data.keys())))


    # ELECTRIC FIELD
    @check_input
    def _E_matrix(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates electric field measurements to
        model vector.

        Not intended to be called by user in standard use case
        """

        Ee, En = get_SECS_J_G_matrices(lat, lon, self.lat_E, self.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)

        return Ee, En

    @extrapolation_check
    def E(self, lon = None, lat = None):
        """
        Calculate electric field vector components

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon

        Returns
        -------
        Ee : array
            Eastward components of the electric field [V/m]. Same shape as lon / lat
        En : array
            Northward components of the electric field [V/m]. Same shape as lon / lat
        """

        if self.m is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        Ee, En, shape = self._E_matrix(lon, lat, return_shape = True)
        return Ee.dot(self.m).reshape(shape), En.dot(self.m).reshape(shape)


    @check_input
    def E_pot(self, lon = None, lat = None):
        """
        Calculate electric potential

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon

        Returns
        -------
        V : array
            Electric potential [V] at lon, lat. The shape is NOT preserved (output is flattened).
            Note that only the gradient in potential is constrained, so any constant can
            be added to the output.
        """

        if self.m is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        G = get_SECS_J_G_matrices(lat, lon, self.lat_E, self.lon_E,
                                  current_type = 'potential',
                                  RI = self.R,
                                  singularity_limit = self.secs_singularity_limit)

        return G.dot(self.m)


    # CONVECTION VELOCITY
    @check_input
    def _v_matrix(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates convection measurements to
        model vector.

        Not intended to be called by user in standard use case
        """

        Ee, En = self._E_matrix(lon, lat)
        Ve, Vn = En * self.Bu / self.B0**2, -Ee * self.Bu / self.B0**2
        # TODO: take into account horizontal components in B

        return Ve, Vn

    @extrapolation_check
    def v(self, lon = None, lat = None):
        """
        Calculate velocity vector components

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon

        Returns
        -------
        ve : array
            Eastward components of the convection velocity [m/s]. Same shape as lon / lat
        vn : array
            Northward components of the convection velocity [m/s]. Same shape as lon / lat
        """

        if self.m is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        Ve, Vn, shape = self._v_matrix(lon, lat, return_shape = True)
        return Ve.dot(self.m).reshape(shape), Vn.dot(self.m).reshape(shape)



    # MAGNETIC FIELDS
    @check_input
    def _B_df_matrix(self, lon = None, lat = None, r = None, return_shape = False, return_poles = False):
        """
        Calculate matrix that relates divergence-free magnetic field values to
        model vector.

        Call this function with return_poles = True to get the DF SECS amplitudes

        Not intended to be called by user in standard use case
        """

        He, Hn, Hu = get_SECS_B_G_matrices(lat, lon, r, self.lat_J, self.lon_J,
                                           current_type = 'divergence_free',
                                           RI = self.R,
                                           singularity_limit = self.secs_singularity_limit,
                                           induction_nullification_radius = self.perfect_conductor_radius)

        H = np.vstack((He, Hn, Hu))

        Ee, En = self.Ee, self.En # electric field design matrices
        E = np.vstack((Ee, En))

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # combine:
        HQiA = H.dot(self.QiA)
        c = - self.Dn.dot(SP) * Ee + self.De.dot(SP) * En \
            - self.Dn.dot(SH) * En * self.hemisphere \
            - self.De.dot(SH) * Ee * self.hemisphere \
            - SH * self.Ddiv.dot(E) * self.hemisphere

        if return_poles:
            return self.QiA.dot(c).dot(self.m)
        else:
            return HQiA.dot(c)


    @check_input
    def _B_cf_matrix(self, lon = None, lat = None, r = None, return_shape = False, return_poles = False):
        """
        Calculate matrix that relates magnetic field of curl-free currents to
        model vector.

        Call this function with return_poles = True to get the CF SECS amplitudes

        Not intended to be called by user in standard use case
        """

        He, Hn, Hu = get_SECS_B_G_matrices(lat, lon, r, self.lat_J, self.lon_J,
                                           current_type = 'curl_free',
                                           RI = self.R,
                                           singularity_limit = self.secs_singularity_limit)


        H = np.vstack((He, Hn, Hu))

        Ee, En = self.Ee, self.En # electric field design matrices
        E = np.vstack((Ee, En))

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # combine:
        HQiA = H.dot(self.QiA)
        d = - self.Dn.dot(SH) * Ee * self.hemisphere \
            + self.De.dot(SH) * En * self.hemisphere \
            + self.De.dot(SP) * Ee + self.Dn.dot(SP) * En \
            + SP * self.Ddiv.dot(E)

        if return_poles: # return SECS poles
            return self.QiA.dot(d).dot(self.m)
        else:
            return HQiA.dot(d)

    @check_input
    def _B_cf_df_matrix(self, lon = None, lat = None, r = None, return_shape = False):
        """
        Calculate matrix that relates magnetic fields of both curl-free and 
        divergence-free currents to model vector.

        Not intended to be called by user in standard use case
        """

        BBB_df = self._B_df_matrix(lon, lat, r)
        BBB_cf = self._B_cf_matrix(lon, lat, r)
        return BBB_df + BBB_cf


    @extrapolation_check
    def B_ground(self, lon = None, lat = None, r = None):
        """
        Calculate ground magnetic field perturbation vectors

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of *outer* grid points,
            (see self.grid_E). Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of *outer* grid points,
            (see self.grid_E). Must have same shape as lon
        r : array, optional
            Radius [m] of the evaluation points, default is Earth radius. Must have a shape that is
            consistent with lon and lat. Broadcasting rules apply


        Returns
        -------
        Be : array
            Eastward components of the ground magnetic field perturbation [T]. Same shape as lon / lat
        Bn : array
            Northward components of the ground magnetic field perturbation [T]. Same shape as lon / lat
        Bu : array
            Upward components of the ground magnetic field perturbation [T]. Same shape as lon / lat

        """

        if self.m is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        BBB, shape = self._B_df_matrix(lon, lat, r, return_shape = True)
        Be, Bn, Bu = np.split(np.ravel(BBB.dot(self.m)), 3)

        return Be.reshape(shape), Bn.reshape(shape), Bu.reshape(shape)

    @extrapolation_check
    def B_space(self, lon = None, lat = None, r = None, include_df = True):
        """
        Calculate space magnetic field perturbation vectors

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of *outer* grid points,
            (see self.grid_E). Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of *outer* grid points,
            (see self.grid_E). Must have same shape as lon
        r : array, optional
            Radius [m] of the evaluation points, default is the radius corresponding to twice the
            height of the ionosphere. Must have a shape that is consistent with lon and lat.
            Broadcasting rules apply

        Returns
        -------
        Be : array
            Eastward components of the space magnetic field perturbation [T]. Same shape as lon / lat
        Bn : array
            Northward components of the space magnetic field perturbation [T]. Same shape as lon / lat
        Bu : array
            Upward components of the space magnetic field perturbation [T]. Same shape as lon / lat
        """

        if self.m is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        # handle default r:
        if r is None: r = self.R * 2 - RE

        BBB, shape = self._B_cf_matrix(lon, lat, r, return_shape = True)
        Be, Bn, Bu = np.split(np.ravel(BBB.dot(self.m)), 3)

        if include_df:
            BBB = self._B_df_matrix(lon, lat, r, return_shape = False)
            Be_df, Bn_df, Bu_df = np.split(np.ravel(BBB.dot(self.m)), 3)
            Be, Bn, Bu = Be + Be_df, Bn + Bn_df, Bu + Bu_df

        return Be.reshape(shape), Bn.reshape(shape), Bu.reshape(shape)

    @extrapolation_check
    def B_space_FAC(self, lon = None, lat = None, r = None):
        """
        Calculate the space magnetic field perturbation vectors that
        correspond to field-aligned currents. Ignoring the effect of
        divergence-free currents.

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of *outer* grid points,
            (see self.grid_E). Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of *outer* grid points,
            (see self.grid_E). Must have same shape as lon
        r : array, optional
            Radius [m] of the evaluation points, default is the radius corresponding to twice the
            height of the ionosphere. Must have a shape that is consistent with lon and lat.
            Broadcasting rules apply

        Returns
        -------
        Be : array
            Eastward components of the space magnetic field perturbation [T]. Same shape as lon / lat
        Bn : array
            Northward components of the space magnetic field perturbation [T]. Same shape as lon / lat
        Bu : array
            Upward components of the space magnetic field perturbation [T]. Same shape as lon / lat
        """

        return self.B_space(lon = lon, lat = lat, r = r, include_df = False)

    @extrapolation_check
    def FAC_matrix(self, lon = None, lat = None):
        """
        Calculate matrix that relates FAC densities to electric field model
        parameters. The output matrix is intended to use with FACs defined
        each grid cell. It will have shape K_J x K_J, where K_J is the number of
        interior grid cells.

        Intended for "regional M-I coupling": Specify conductance and FACs
        and get get back everything else.
        """

        Ee, En = self.Ee, self.En # electric field design matrices

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # current matrices (N x M)
        JE = SP * Ee + SH * En * self.hemisphere
        JN = SP * En - SH * Ee * self.hemisphere
        J  = np.vstack((JE, JN))

        return -self.Ddiv.dot(J)


    # CURRENTS
    @check_input
    def j(self, lon = None, lat = None):
        """
        Calculate the horizontal ionospheric surface current density

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon

        Returns
        -------
        je : array
            Eastward components of the horizontal surface current density [A/m]. Same shape as lon / lat
        jn : array
            Northward components of the horizontal surface current density [A/m]. Same shape as lon / lat

        See also
        --------
        get_SECS_currents : Calculate current based on SECS, not Ohm's law (should be consistent!)
        """

        shape = np.broadcast(lon, lat).shape

        # get conductances
        SH = self.hall_conductance(    lon, lat)
        SP = self.pedersen_conductance(lon, lat)

        # electric field:
        Ee, En = self.E(lon, lat)

        je = Ee * SP + SH * En * self.hemisphere
        jn = En * SP - SH * Ee * self.hemisphere

        return je.reshape(shape), jn.reshape(shape)


    @check_input
    def FAC(self, lon = None, lat = None):
        """
        Calculate the upward volume current density. The calculation is performed by
        estimating the divergence of the Ohm's law currents.

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of exterior grid points (grid_E).
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of exterior grid points (grid_E).
            Must have same shape as lon

        Returns
        -------
        ju : array
            Upward current density [A/m^2]

        Note
        ----
        The FACs are calculated on grid using numerical differentiation, and then interpolated
        to the requested coordinates using griddata.

        """

        shape = np.broadcast(lon, lat).shape

        # get conductances on grid
        SH = self.hall_conductance(    self.grid_J.lon.flatten(), self.grid_J.lat.flatten())
        SP = self.pedersen_conductance(self.grid_J.lon.flatten(), self.grid_J.lat.flatten())

        # electric field on grid:
        Ee, En = self.E(self.grid_J.lon.flatten(), self.grid_J.lat.flatten())
        Ee, En = Ee, En

        # currents on grid
        je = Ee * SP + SH * En * self.hemisphere
        jn = En * SP - SH * Ee * self.hemisphere

        # upward current on grid is negative divergence:
        ju_ = -self.Ddiv.dot(np.hstack((je, jn)))

        # interpolate to desired coords if necessary
        xi, eta = self.grid_J.projection.geo2cube(lon, lat) # cs coords
        try: # if the input grid is equal grid_J, skip interpolation
            if np.all(np.isclose(xi - self.grid_J.xi.flatten(), 0)) & \
               np.all(np.isclose(eta - self.grid_J.eta.flatten(), 0)):
                return ju_.reshape(shape)
        except:
            pass

        gridcoords = np.vstack((self.grid_J.xi.flatten(), self.grid_J.eta.flatten())).T
        ju = griddata(gridcoords, ju_, np.vstack((xi, eta)).T)

        # return
        return ju.reshape(shape)


    @check_input
    def get_SECS_currents(self, lon = None, lat = None):
        """
        Calculate the horizontal ionospheric surface current density,
        using the SECS pole amplitudes instead of Ohm's law. Should be
        consistent with the output of self.j(), and could be used to
        check that the model representation is ok. Deviations could be
        due to inaccuracies in the finite difference evaluations, and
        would suggest improving the grid resolution (or the finite
        difference code...)

        Requires the model vector to be defined.

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon

        Returns
        -------
        je : array
            Eastward components of the horizontal surface current density [A/m]. Same shape as lon / lat
        jn : array
            Northward components of the horizontal surface current density [A/m]. Same shape as lon / lat
        """

        shape = np.broadcast(lon, lat).shape

        S_cf =  self._B_cf_matrix(return_poles = True)
        S_df =  self._B_df_matrix(return_poles = True)

        Be_cf, Bn_cf = get_SECS_J_G_matrices(lat, lon, self.lat_J, self.lon_J,
                                             current_type = 'curl_free',
                                             RI = self.R,
                                             singularity_limit = self.secs_singularity_limit)

        Be_df, Bn_df = get_SECS_J_G_matrices(lat, lon, self.lat_J, self.lon_J,
                                             current_type = 'divergence_free',
                                             RI = self.R,
                                             singularity_limit = self.secs_singularity_limit)

        return Be_cf.dot(S_cf) + Be_df.dot(S_df), Bn_cf.dot(S_cf) + Bn_df.dot(S_df)


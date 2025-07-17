#%% Import

import numpy as np
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
from .varcheck import check_input, extrapolation_check
from scipy.interpolate import griddata

#%%

RE = 6371.2e3

#%%
class Evaluator(object):
    def __init__(self, model):

        self.m_CF = model.m_CF
        self.m_DF = model.m_DF
        self.gH = model.gH
        self.builder = model.builder

#%%

    @property
    def func(self):
        return {
            'ground_mag':       self.B_ground,
            'space_mag_full':   self.B_space,
            'space_mag_fac':    self.B_space_FAC,
            'efield':           self.E,
            'convection':       self.v,
            'fac':              self.FAC,
            'joule':            self.joule,
            'current':          self.J,
            'potential':        self.Phi,
            'stream':           self.W,
            'hall':             self.builder.hall_conductance,
            'pedersen':         self.builder.pedersen_conductance,
            'dbrdt':            self.dBrdt
            }

#%% Electric field
    #@extrapolation_check
    def E(self, lon = None, lat = None, comp='CF'):
        def _get_variable(method):
            return method(lon, lat)
        
        if comp == 'CF':
            result = _get_variable(self.E_CF)
        elif comp == 'DF':
            result = _get_variable(self.E_DF)
        else:
            cf = _get_variable(self.E_CF)
            df = _get_variable(self.E_DF)
            result = tuple(c + d for c, d in zip(cf, df))
        return result    
    
    #@extrapolation_check
    def E_CF(self, lon = None, lat = None):
        if self.m_CF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        Ee, En, shape = self.builder._E_matrix_CF(lon, lat, return_shape = True)
        return Ee.dot(self.m_CF).reshape(shape), En.dot(self.m_CF).reshape(shape)

    #@extrapolation_check
    def E_DF(self, lon = None, lat = None):
        if self.m_DF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        Ee, En, shape = self.builder._E_matrix_DF(lon, lat, return_shape = True)
        return Ee.dot(self.m_DF).reshape(shape), En.dot(self.m_DF).reshape(shape)

#%% Scalar potentials
    #@check_input
    def Phi(self, lon = None, lat = None):
        if self.m_CF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')
        G, shape = self.builder._Phi_matrix(lon, lat, return_shape=True)
        return G.dot(self.m_CF).reshape(shape)

    #@check_input
    def W(self, lon = None, lat = None):
        if self.m_DF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')
        G, shape = self.builder._W_matrix(lon, lat, return_shape=True)
        return G.dot(self.m_DF).reshape(shape)

#%% Convection velocity
    #@extrapolation_check    
    def v(self, lon = None, lat = None, comp='CF'):
        def _get_variable(method):
            return method(lon, lat)
        
        if comp == 'CF':
            result = _get_variable(self.v_CF)
        elif comp == 'DF':
            result = _get_variable(self.v_DF)
        else:
            cf = _get_variable(self.v_CF)
            df = _get_variable(self.v_DF)
            result = tuple(c + d for c, d in zip(cf, df))
        return result
        
    #@extrapolation_check
    def v_CF(self, lon = None, lat = None):
        if self.m_CF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        Ve, Vn, shape = self.builder._v_matrix_CF(lon, lat, return_shape = True)
        return Ve.dot(self.m_CF).reshape(shape), Vn.dot(self.m_CF).reshape(shape)

    #@extrapolation_check
    def v_DF(self, lon = None, lat = None):
        if self.m_DF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        Ve, Vn, shape = self.builder._v_matrix_DF(lon, lat, return_shape = True)
        return Ve.dot(self.m_DF).reshape(shape), Vn.dot(self.m_DF).reshape(shape)

#%% Ground magnetic field
    #@extrapolation_check
    def B_ground(self, lon = None, lat = None, r = None, comp='CF'):
        def _get_variable(method):
            return method(lon, lat, r)
        
        if comp == 'CF':
            result = _get_variable(self.B_ground_CF)
        elif comp == 'DF':
            result = _get_variable(self.B_ground_DF)
        else:
            cf = _get_variable(self.B_ground_CF)
            df = _get_variable(self.B_ground_DF)
            result = tuple(c + d for c, d in zip(cf, df))
        return result
        
    #@extrapolation_check
    def B_ground_CF(self, lon = None, lat = None, r = None):
        if self.m_CF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        BBB, shape = self.builder._B_df_matrix_CF(lon, lat, r, return_shape = True)
        Be, Bn, Bu = np.split(np.ravel(BBB.dot(self.m_CF)), 3)

        return Be.reshape(shape), Bn.reshape(shape), Bu.reshape(shape)

    #@extrapolation_check
    def B_ground_DF(self, lon = None, lat = None, r = None):
        if self.m_DF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        BBB, shape = self.builder._B_df_matrix_DF(lon, lat, r, return_shape = True)
        Be, Bn, Bu = np.split(np.ravel(BBB.dot(self.m_DF)), 3)

        return Be.reshape(shape), Bn.reshape(shape), Bu.reshape(shape)

#%% dBrdt

    def dBrdt(self):
        return self.builder.dBrdt_matrix().dot(self.m_DF).reshape(self.gH.shape_E)

#%% Space magnetic field

    #@extrapolation_check
    def B_space(self, lon = None, lat = None, r = None, comp='CF'):
        def _get_variable(method):
            return method(lon, lat, r)
        
        if comp == 'CF':
            result = _get_variable(self.B_space_CF)
        elif comp == 'DF':
            result = _get_variable(self.B_space_DF)
        else:
            cf = _get_variable(self.B_space_CF)
            df = _get_variable(self.B_space_DF)
            result = tuple(c + d for c, d in zip(cf, df))
        return result

    #@extrapolation_check
    def B_space_CF(self, lon = None, lat = None, r = None, include_df = True):
        if self.m_CF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        # handle default r:
        if r is None: r = self.gH.R * 2 - RE

        BBB, shape = self.builder._B_cf_matrix_CF(lon, lat, r, return_shape = True)
        Be, Bn, Bu = np.split(np.ravel(BBB.dot(self.m_CF)), 3)

        if include_df:
            BBB = self.builder._B_df_matrix_CF(lon, lat, r, return_shape = False)
            Be_df, Bn_df, Bu_df = np.split(np.ravel(BBB.dot(self.m_CF)), 3)
            Be, Bn, Bu = Be + Be_df, Bn + Bn_df, Bu + Bu_df

        return Be.reshape(shape), Bn.reshape(shape), Bu.reshape(shape)

    #@extrapolation_check
    def B_space_DF(self, lon = None, lat = None, r = None, include_df = True):
        if self.m_DF is None:
            raise Exception('Model vector not defined yet. Add data and call run_inversion()')

        # handle default r:
        if r is None: r = self.gH.R * 2 - RE

        BBB, shape = self.builder._B_cf_matrix_DF(lon, lat, r, return_shape = True)
        Be, Bn, Bu = np.split(np.ravel(BBB.dot(self.m_DF)), 3)

        if include_df:
            BBB = self.builder._B_df_matrix_DF(lon, lat, r, return_shape = False)
            Be_df, Bn_df, Bu_df = np.split(np.ravel(BBB.dot(self.m_DF)), 3)
            Be, Bn, Bu = Be + Be_df, Bn + Bn_df, Bu + Bu_df

        return Be.reshape(shape), Bn.reshape(shape), Bu.reshape(shape)

#%% Space magnetic field (FAC)
    #@extrapolation_check
    def B_space_FAC(self, lon = None, lat = None, r = None, comp='CF'):
        def _get_variable(method):
            return method(lon, lat, r)
        
        if comp == 'CF':
            result = _get_variable(self.B_space_FAC_CF)
        elif comp == 'DF':
            result = _get_variable(self.B_space_FAC_DF)
        else:
            cf = _get_variable(self.B_space_FAC_CF)
            df = _get_variable(self.B_space_FAC_DF)
            result = tuple(c + d for c, d in zip(cf, df))
        return result

    #@extrapolation_check
    def B_space_FAC_CF(self, lon = None, lat = None, r = None):
        return self.B_space_CF(lon = lon, lat = lat, r = r, include_df = False)

    #@extrapolation_check
    def B_space_FAC_DF(self, lon = None, lat = None, r = None):
        return self.B_space_DF(lon = lon, lat = lat, r = r, include_df = False)

#%% Electric currents
    #@check_input
    def J(self, lon = None, lat = None, decomp=False, comp='CF'):
        def _get_variable(method):
            return method(lon, lat, decomp=decomp)        
        
        if comp == 'CF':
            result = _get_variable(self.J_CF)
        elif comp == 'DF':
            result = _get_variable(self.J_DF)
        else:
            cf = _get_variable(self.J_CF)
            df = _get_variable(self.J_DF)
            result = tuple(c + d for c, d in zip(cf, df))            
        return result

    #@check_input
    def J_CF(self, lon = None, lat = None, decomp=False):
        # get conductances
        if lon is None:
            SH = self.builder.hall_conductance(    )
            SP = self.builder.pedersen_conductance()
        else:
            SH = self.builder.hall_conductance(    lon, lat)
            SP = self.builder.pedersen_conductance(lon, lat)

        # electric field:
        Ee, En = self.E_CF(lon, lat)
        
        jeH =   SH * En * self.gH.hemisphere
        jnH = - SH * Ee * self.gH.hemisphere        
        jeP = Ee * SP
        jnP = En * SP
        
        if not decomp:
            return jeH+jeP, jnH+jnP
        else: 
            return jeH, jnH, jeP, jnP

    # CURRENTS
    #@check_input
    def J_DF(self, lon = None, lat = None, decomp=False):
        # get conductances
        if lon is None:
            SH = self.builder.hall_conductance(    )
            SP = self.builder.pedersen_conductance()
        else:
            SH = self.builder.hall_conductance(    lon, lat)
            SP = self.builder.pedersen_conductance(lon, lat)

        # electric field:
        Ee, En = self.E_DF(lon, lat)

        jeH =   SH * En * self.gH.hemisphere
        jnH = - SH * Ee * self.gH.hemisphere
        jeP = Ee * SP
        jnP = En * SP
        if not decomp:
            return jeH+jeP, jnH+jnP
        else: 
            return jeH, jnH, jeP, jnP

#%% FAC
    #@check_input
    def FAC(self, lon = None, lat = None, comp='CF'):
        def _get_variable(method):
            return method(lon, lat)
        
        if comp == 'CF':
            result = _get_variable(self.FAC_CF)
        elif comp == 'DF':
            result = _get_variable(self.FAC_DF)
        else:
            cf = _get_variable(self.FAC_CF)
            df = _get_variable(self.FAC_DF)
            result = cf + df
        return result

    @check_input
    def FAC_CF(self, lon = None, lat = None):
        shape = np.broadcast(lon, lat).shape

        # get conductances on grid
        SH = self.builder.hall_conductance(    self.gH.lon_J.flatten(), self.gH.lat_J.flatten())
        SP = self.builder.pedersen_conductance(self.gH.lon_J.flatten(), self.gH.lat_J.flatten())

        # electric field on grid:
        Ee, En = self.E_CF(self.gH.lon_J.flatten(), self.gH.lat_J.flatten())
        Ee, En = Ee, En

        # currents on grid
        je = Ee * SP + SH * En * self.gH.hemisphere
        jn = En * SP - SH * Ee * self.gH.hemisphere

        # upward current on grid is negative divergence:
        ju_ = -self.gH.Ddiv_J.dot(np.hstack((je, jn)))

        # interpolate to desired coords if necessary
        xi, eta = self.gH.grid_J.projection.geo2cube(lon, lat) # cs coords
        try: # if the input grid is equal grid_J, skip interpolation
            if np.all(np.isclose(xi - self.gH.xi_J.flatten(), 0)) & \
               np.all(np.isclose(eta - self.gH.eta_J.flatten(), 0)):
                return ju_.reshape(shape)
        except:
            pass

        gridcoords = np.vstack((self.gH.xi_J.flatten(), self.gH.eta_J.flatten())).T
        ju = griddata(gridcoords, ju_, np.vstack((xi, eta)).T)

        # return
        return ju.reshape(shape)

    @check_input
    def FAC_DF(self, lon = None, lat = None):
        shape = np.broadcast(lon, lat).shape

        # get conductances on grid
        SH = self.builder.hall_conductance(    self.gH.lon_J.flatten(), self.gH.lat_J.flatten())
        SP = self.builder.pedersen_conductance(self.gH.lon_J.flatten(), self.gH.lat_J.flatten())

        # electric field on grid:
        Ee, En = self.E_DF(self.gH.lon_J.flatten(), self.gH.lat_J.flatten())
        Ee, En = Ee, En

        # currents on grid
        je = Ee * SP + SH * En * self.gH.hemisphere
        jn = En * SP - SH * Ee * self.gH.hemisphere

        # upward current on grid is negative divergence:
        ju_ = -self.gH.Ddiv_J.dot(np.hstack((je, jn)))

        # interpolate to desired coords if necessary
        xi, eta = self.gH.grid_J.projection.geo2cube(lon, lat) # cs coords
        try: # if the input grid is equal grid_J, skip interpolation
            if np.all(np.isclose(xi - self.gH.xi_J.flatten(), 0)) & \
               np.all(np.isclose(eta - self.gH.eta_J.flatten(), 0)):
                return ju_.reshape(shape)
        except:
            pass

        gridcoords = np.vstack((self.gH.xi_J.flatten(), self.gH.eta_J.flatten())).T
        ju = griddata(gridcoords, ju_, np.vstack((xi, eta)).T)

        # return
        return ju.reshape(shape)

#%% Joule heating
    @check_input    
    def joule(self, lon = None, lat = None, comp=None):
        
        joule_CF = None
        joule_x  = None
        joule_DF = None
        
        SP = self.builder.pedersen_conductance(lon, lat)
        Ee_CF, En_CF = self.E(lon, lat, comp='CF')
        joule_CF = SP * (Ee_CF**2 + En_CF**2)
        if self.m_DF is not None:
            Ee_DF, En_DF = self.E(lon, lat, comp='DF')
            joule_x = 2 * SP * (Ee_CF*Ee_DF + En_CF*En_DF)
            joule_DF = SP * (Ee_DF**2 + En_DF**2)

        if comp == 'both':
            return joule_CF + joule_x + joule_DF
        elif comp == 'decomp':
            return joule_CF, joule_x, joule_DF
        elif comp == 'CF':
            return joule_CF
        elif comp == 'DF':
            return joule_x + joule_DF
        else:
            raise ValueError('Invalid comp provided')
        

#%%
    #@check_input
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

        S_cf =  self.builder._B_cf_matrix(return_poles = True).dot(self.m_CF)
        S_df =  self.builder._B_df_matrix(return_poles = True).dot(self.m_CF)

        Be_cf, Bn_cf = get_SECS_J_G_matrices(lat, lon, self.gH.lat_J, self.gH.lon_J,
                                             current_type = 'curl_free',
                                             RI = self.gH.R,
                                             singularity_limit = self.builder.secs_singularity_limit)

        Be_df, Bn_df = get_SECS_J_G_matrices(lat, lon, self.gH.lat_J, self.gH.lon_J,
                                             current_type = 'divergence_free',
                                             RI = self.gH.R,
                                             singularity_limit = self.builder.secs_singularity_limit)

        return Be_cf.dot(S_cf) + Be_df.dot(S_df), Bn_cf.dot(S_cf) + Bn_df.dot(S_df)
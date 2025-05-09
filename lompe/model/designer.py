""" 
Lompe design matrix builder class

The data input to the Lompe inversion should be given as lompe.Data objects. The Data
class is defined here. 

"""
import numpy as np


class Builder(object):
    def __init__(self, grid):
        """ 
        Parameters
        ----------
        values: array
            array of values in SI units - see specific data type for details
        """

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

    @check_input
    def _E_matrix_DF(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates electric field measurements to
        model vector.

        Not intended to be called by user in standard use case
        """

        Ee, En = get_SECS_J_G_matrices(lat, lon, self.lat_E, self.lon_E,
                                       current_type = 'divergence_free',
                                       RI = self.R,
                                       singularity_limit = self.secs_singularity_limit)

        return Ee, En

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

    @check_input
    def _v_matrix_DF(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates convection measurements to
        model vector.

        Not intended to be called by user in standard use case
        """

        Ee, En = self._E_matrix_DF(lon, lat)
        Ve, Vn = En * self.Bu / self.B0**2, -Ee * self.Bu / self.B0**2
        # TODO: take into account horizontal components in B

        return Ve, Vn

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
    def _B_df_matrix_DF(self, lon = None, lat = None, r = None, return_shape = False, return_poles = False):
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

        Ee_DF, En_DF = self.Ee_DF, self.En_DF # electric field design matrices

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # combine:
        HQiA = H.dot(self.QiA)
        stheta = np.sin(self.grid_E.lat.flatten()/180*np.pi)
        c = - self.Dn.dot(SP) * Ee_DF + self.De.dot(SP) * En_DF \
            + SP/(self.R*stheta) * (self.Dn.dot(Ee_DF * stheta) - self.De.dot(En_DF)) \
            - self.Dn.dot(SH) * En_DF * self.hemisphere \
            - self.De.dot(SH) * Ee_DF * self.hemisphere

        if return_poles:
            return self.QiA.dot(c).dot(self.m_ind)
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
    def _B_cf_matrix_DF(self, lon = None, lat = None, r = None, return_shape = False, return_poles = False):
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

        Ee_DF, En_DF = self.Ee_DF, self.En_DF # electric field design matrices
        Ecrossb = np.vstack((-En_DF, Ee_DF))

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # combine:
        HQiA = H.dot(self.QiA)
        d = - SH * self.Ddiv.dot(Ecrossb) \
            - self.Dn.dot(SH) * Ee_DF * self.hemisphere \
            + self.De.dot(SH) * En_DF * self.hemisphere \
            + self.De.dot(SP) * Ee_DF + self.Dn.dot(SP) * En_DF

        if return_poles: # return SECS poles
            return self.QiA.dot(d).dot(self.m_ind)
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

    @check_input
    def _B_cf_df_matrix_DF(self, lon = None, lat = None, r = None, return_shape = False):
        """
        Calculate matrix that relates magnetic fields of both curl-free and 
        divergence-free currents to model vector.

        Not intended to be called by user in standard use case
        """

        BBB_df = self._B_df_matrix_DF(lon, lat, r)
        BBB_cf = self._B_cf_matrix_DF(lon, lat, r)
        return BBB_df + BBB_cf

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

    @extrapolation_check
    def FAC_matrix_DF(self, lon = None, lat = None):
        """
        Calculate matrix that relates FAC densities to electric field model
        parameters. The output matrix is intended to use with FACs defined
        each grid cell. It will have shape K_J x K_J, where K_J is the number of
        interior grid cells.

        Intended for "regional M-I coupling": Specify conductance and FACs
        and get get back everything else.
        """

        Ee_DF, En_DF = self.Ee_DF, self.En_DF # electric field design matrices

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # current matrices (N x M)
        JE = SP * Ee_DF + SH * En_DF * self.hemisphere
        JN = SP * En_DF - SH * Ee_DF * self.hemisphere
        J  = np.vstack((JE, JN))

        return -self.Ddiv.dot(J)

""" 
Lompe design matrix builder class

The data input to the Lompe inversion should be given as lompe.Data objects. The Data
class is defined here. 

"""
import numpy as np
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
from dipole import Dipole
from ppigrf import igrf
from .varcheck import check_input, extrapolation_check
from lompe.utils.time import yearfrac_to_datetime

#%%

RE = 6371.2e3

#%%
class Design(object):
    def __init__(self, gH, hall_conductance, pedersen_conductance, perfect_conductor_radius=None, dipole=False, epoch=2015):
        """ 
        Parameters
        ----------
        values: array
            array of values in SI units - see specific data type for details
        """

        self.gH = gH

        self.hall_conductance = hall_conductance
        self.pedersen_conductance = pedersen_conductance

        # set SECS singularity limit so it covers the cell:
        self.secs_singularity_limit = np.min([self.gH.Wres, self.gH.Lres])/2
    
        self.perfect_conductor_radius = perfect_conductor_radius
        self.dipole = dipole
        self.epoch = epoch
    
        self._QiA = None
    
        self.get_main_field()
    
        self._Ee_CF, self._En_CF = None, None
        self._Ee_DF, self._En_DF = None, None

    def get_main_field(self):
        # Calculate main field values for all grid points
        refh = (self.gH.R - RE) * 1e-3 # apex reference height [km] - also used for IGRF altitude
        if self.dipole:
            Bn, Bu = Dipole(self.epoch).B(self.gH.lat_E, self.gH.R * 1e-3)
            Be = np.zeros_like(Bn)
        else: # use IGRF
            Be, Bn, Bu = igrf(self.gH.lon_E, self.gH.lat_E, refh, yearfrac_to_datetime([self.epoch]))
        Be, Bn, Bu = Be * 1e-9, Bn * 1e-9, Bu * 1e-9 # nT -> T
        self.B0 = np.sqrt(Be**2 + Bn**2 + Bu**2).reshape((1, -1))
        self.Bu = Bu.reshape((1, -1))
        if not np.allclose(np.sign(self.Bu), np.sign(self.Bu.flatten()[0])):
            raise Exception('your grid covers two magnetic hemispheres. It should not')
        self.gH.hemisphere = -np.sign(self.Bu.flatten()[0]) # 1 for north, -1 for south

    @property
    def QiA(self):
        if self._QiA is None:
            # cell area matrix:
            dxi, deta, A = self.gH.grid_J.projection.differentials(self.gH.xi_J , self.gH.eta_J,
                                                                   self.gH.grid_J.dxi, self.gH.grid_J.deta, R = self.gH.R)
            A = np.diag(np.ravel(A))
            # curl/divergence distribution matrix Q:
            Q = np.eye(self.gH.size_J) - A.dot(np.full((self.gH.size_J, self.gH.size_J), 1 / (4 * np.pi * self.gH.R**2)))

            # inverse of QA
            self._QiA = np.linalg.pinv(Q, hermitian = True).dot(A)
        return self._QiA

    @property
    def Ee_CF(self):
        if self._Ee_CF is None:
            self._Ee_CF, self._En_CF = self._E_matrix_CF()
        return self._Ee_CF

    @property
    def En_CF(self):
        if self._En_CF is None:
            self._Ee_CF, self._En_CF = self._E_matrix_CF()
        return self._En_CF

    @property
    def Ee_DF(self):
        if self._Ee_DF is None:
            self._Ee_DF, self._En_DF = self._E_matrix_DF()
        return self._Ee_CF

    @property
    def En_DF(self):
        if self._En_DF is None:
            self._Ee_DF, self._En_DF = self._E_matrix_DF()
        return self._En_DF

    # Scalar potentials
    @check_input
    def _Phi_matrix(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates the electric potential to the CF SECS.
        """
        G = get_SECS_J_G_matrices(lat, lon, self.gH.lat_E, self.gH.lon_E,
                                  current_type = 'potential',
                                  RI = self.gH.R, singularity_limit = self.secs_singularity_limit)
        return G

    @check_input
    def _W_matrix(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates the electric stream function to the DF SECS.
        """
        G = get_SECS_J_G_matrices(lat, lon, self.gH.lat_E, self.gH.lon_E,
                                  current_type = 'stream',
                                  RI = self.gH.R, singularity_limit = self.secs_singularity_limit)
        return G


    # ELECTRIC FIELD
    @check_input
    def _E_matrix_CF(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates electric field to the CF SECS.
        """

        Ee, En = get_SECS_J_G_matrices(lat, lon, self.gH.lat_E, self.gH.lon_E,
                                       current_type = 'curl_free',
                                       RI = self.gH.R,
                                       singularity_limit = self.secs_singularity_limit)

        return Ee, En

    @check_input
    def _E_matrix_DF(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates electric field to the DF SECS.
        """

        Ee, En = get_SECS_J_G_matrices(lat, lon, self.gH.lat_E, self.gH.lon_E,
                                       current_type = 'divergence_free',
                                       RI = self.gH.R,
                                       singularity_limit = self.secs_singularity_limit)

        return Ee, En

    # CONVECTION VELOCITY
    @check_input
    def _v_matrix_CF(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates convection to the CF SECS.
        """

        Ee, En = self._E_matrix_CF(lon, lat)
        Ve, Vn = En * self.Bu / self.B0**2, -Ee * self.Bu / self.B0**2
        # TODO: take into account horizontal components in B

        return Ve, Vn

    @check_input
    def _v_matrix_DF(self, lon = None, lat = None, return_shape = False):
        """
        Calculate matrix that relates convection to the DF SECS.
        """

        Ee, En = self._E_matrix_DF(lon, lat)
        Ve, Vn = En * self.Bu / self.B0**2, -Ee * self.Bu / self.B0**2
        # TODO: take into account horizontal components in B

        return Ve, Vn

    # MAGNETIC FIELDS
    @check_input
    def _B_df_matrix_CF(self, lon = None, lat = None, r = None, return_shape = False, return_poles = False):
        """
        Calculate matrix that relates divergence-free magnetic field to the CF SECS.
        
        If return_poles = True, then return amplitudes of DF SECS eq current
        """

        He, Hn, Hu = get_SECS_B_G_matrices(lat, lon, r, self.gH.lat_J, self.gH.lon_J,
                                           current_type = 'divergence_free',
                                           RI = self.gH.R,
                                           singularity_limit = self.secs_singularity_limit,
                                           induction_nullification_radius = self.perfect_conductor_radius)

        H = np.vstack((He, Hn, Hu))

        Ee, En = self.Ee_CF, self.En_CF # electric field design matrices
        E = np.vstack((Ee, En))

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # combine:
        HQiA = H.dot(self.QiA)
        c = - self.gH.Dn_J.dot(SP) * Ee + self.gH.De_J.dot(SP) * En \
            - self.gH.Dn_J.dot(SH) * En * self.gH.hemisphere \
            - self.gH.De_J.dot(SH) * Ee * self.gH.hemisphere \
            - SH * self.gH.Ddiv_J.dot(E) * self.gH.hemisphere

        if return_poles:
            return self.QiA.dot(c)
        else:
            return HQiA.dot(c)

    @check_input        
    def _B_df_matrix_DF(self, lon = None, lat = None, r = None, return_shape = False, return_poles = False):
        """
        Calculate matrix that relates divergence-free magnetic field to the DF SECS.
        
        If return_poles = True, then return amplitudes of DF SECS eq current
        """

        He, Hn, Hu = get_SECS_B_G_matrices(lat, lon, r, self.gH.lat_J, self.gH.lon_J,
                                           current_type = 'divergence_free',
                                           RI = self.gH.R,
                                           singularity_limit = self.secs_singularity_limit,
                                           induction_nullification_radius = self.perfect_conductor_radius)

        H = np.vstack((He, Hn, Hu))

        Ee, En = self.Ee_DF, self.En_DF # electric field design matrices

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # combine:
        HQiA = H.dot(self.QiA)
        stheta = np.sin(self.gH.lat_E.flatten()/180*np.pi)
        c = - self.gH.Dn.dot(SP) * Ee + self.gH.De.dot(SP) * En \
            + SP/(self.gH.R*stheta) * (self.gH.Dn.dot(Ee * stheta) - self.gH.De.dot(En)) \
            - self.gH.Dn.dot(SH) * En * self.gH.hemisphere \
            - self.gH.De.dot(SH) * Ee * self.gH.hemisphere

        if return_poles:
            return self.QiA.dot(c)
        else:
            return HQiA.dot(c)        

    @check_input
    def _B_cf_matrix_CF(self, lon = None, lat = None, r = None, return_shape = False, return_poles = False):
        """
        Calculate matrix that relates magnetic field of curl-free currents to
        model vector.

        Call this function with return_poles = True to get the CF SECS amplitudes

        Not intended to be called by user in standard use case
        """

        He, Hn, Hu = get_SECS_B_G_matrices(lat, lon, r, self.gH.lat_J, self.gH.lon_J,
                                           current_type = 'curl_free',
                                           RI = self.gH.R,
                                           singularity_limit = self.secs_singularity_limit)


        H = np.vstack((He, Hn, Hu))

        Ee, En = self.Ee_CF, self.En_CF # electric field design matrices
        E = np.vstack((Ee, En))

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # combine:
        HQiA = H.dot(self.QiA)
        d = - self.gH.Dn_J.dot(SH) * Ee * self.gH.hemisphere \
            + self.gH.De_J.dot(SH) * En * self.gH.hemisphere \
            + self.gH.De_J.dot(SP) * Ee + self.gH.Dn_J.dot(SP) * En \
            + SP * self.gH.Ddiv_J.dot(E)

        if return_poles: # return SECS poles
            return self.QiA.dot(d)
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

        He, Hn, Hu = get_SECS_B_G_matrices(lat, lon, r, self.gH.lat_J, self.gH.lon_J,
                                           current_type = 'curl_free',
                                           RI = self.gH.R,
                                           singularity_limit = self.secs_singularity_limit)


        H = np.vstack((He, Hn, Hu))

        Ee, En = self.Ee_DF, self.En_DF # electric field design matrices
        Exb = np.vstack((-En, Ee))

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # combine:
        HQiA = H.dot(self.QiA)
        d = - SH * self.gH.Ddiv_J.dot(Exb) \
            - self.gH.Dn_J.dot(SH) * Ee * self.gH.hemisphere \
            + self.gH.De_J.dot(SH) * En * self.gH.hemisphere \
            + self.gH.De_J.dot(SP) * Ee + self.gH.Dn_J.dot(SP) * En

        if return_poles: # return SECS poles
            return self.QiA.dot(d)
        else:
            return HQiA.dot(d)

    @check_input
    def _B_cf_df_matrix_CF(self, lon = None, lat = None, r = None, return_shape = False):
        """
        Calculate matrix that relates magnetic fields of both curl-free and 
        divergence-free currents to model vector.

        Not intended to be called by user in standard use case
        """

        BBB_df = self._B_df_matrix_CF(lon, lat, r)
        BBB_cf = self._B_cf_matrix_CF(lon, lat, r)
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
    def FAC_matrix_CF(self, lon = None, lat = None):
        """
        Calculate matrix that relates FAC densities to electric field model
        parameters. The output matrix is intended to use with FACs defined
        each grid cell. It will have shape K_J x K_J, where K_J is the number of
        interior grid cells.

        Intended for "regional M-I coupling": Specify conductance and FACs
        and get get back everything else.
        """

        Ee, En = self.Ee_CF, self.En_CF # electric field design matrices

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # current matrices (N x M)
        JE = SP * Ee + SH * En * self.gH.hemisphere
        JN = SP * En - SH * Ee * self.gH.hemisphere
        J  = np.vstack((JE, JN))

        return -self.gH.Ddiv_J.dot(J)

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

        Ee, En = self.Ee_DF, self.En_DF # electric field design matrices

        # column vectors of conductance:
        SH = np.ravel(self.hall_conductance()    ).reshape((-1, 1))
        SP = np.ravel(self.pedersen_conductance()).reshape((-1, 1))

        # current matrices (N x M)
        JE = SP * Ee + SH * En * self.gH.hemisphere
        JN = SP * En - SH * Ee * self.gH.hemisphere
        J  = np.vstack((JE, JN))

        return -self.gH.Ddiv_J.dot(J)

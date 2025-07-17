""" 
Lompe grid handler class

"""
import numpy as np
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
from secsy import cubedsphere as cs

#%%

class GridHandler(object):
    def __init__(self, grid):

        # Store grid characteristics 
        self.position = grid.projection.position
        self.orientation = grid.projection.orientation
        self.Wres, self.Lres = grid.Wres, grid.Lres
        self.W, self.L = grid.W, grid.L
        self.R = grid.R
        
        self.hemisphere = None # 1 for north, -1 for south
        
        # Inner grid
        self.grid_J = grid    
        
        # Outer grid
        xi_e  = np.hstack((self.grid_J.xi_mesh[0]    , self.grid_J.xi_mesh [0 , - 1] + self.grid_J.dxi )) - self.grid_J.dxi /2 
        eta_e = np.hstack((self.grid_J.eta_mesh[:, 0], self.grid_J.eta_mesh[-1,   0] + self.grid_J.deta)) - self.grid_J.deta/2     
        self.grid_E = cs.CSgrid(cs.CSprojection(self.position, self.orientation),
                                self.grid_J.L + self.Lres, self.grid_J.W + self.Wres, self.Lres, self.Wres, 
                                edges = (xi_e, eta_e), R = self.R) # outer    
        del xi_e, eta_e
        
        # s2b2s grid
        xi_e  = np.hstack((self.grid_E.xi_mesh[0]    , self.grid_E.xi_mesh [0 , - 1] + self.grid_E.dxi )) - self.grid_E.dxi /2
        eta_e = np.hstack((self.grid_E.eta_mesh[:, 0], self.grid_E.eta_mesh[-1,   0] + self.grid_E.deta)) - self.grid_E.deta/2 
        self.grid_sbs = cs.CSgrid(cs.CSprojection(self.position, self.orientation),
                                  self.grid_E.L + self.Lres, self.grid_E.W + self.Wres, self.Lres, self.Wres, 
                                  edges = (xi_e, eta_e), R = self.R) # outer
        del xi_e, eta_e
        
        # data grid
        self.grid_d = None
        
        # Gradient and divergence matrices for inner and outer grid:
        self._De_J, self._Dn_J, self._Ddiv_J = None, None, None
        self._De_E, self._Dn_E, self._Ddiv_E = None, None, None

        self._A_J, self._A_E = None, None

    def __repr__(self):
        return f"<GridHandler J-grid: {self.grid_J.shape}, E-grid: {self.grid_E.shape}, pos: {self.position}>"
    
    def create_data_grid(self, perimeter_width):
        self.grid_d = cs.CSgrid(cs.CSprojection(self.position, self.orientation),
                                self.L + 2 * perimeter_width * self.Lres, 
                                self.W + 2 * perimeter_width * self.Wres,
                                self.Lres, self.Wres, R = self.R)
    
    def clear_Ds(self):
        self._De_J, self._Dn_J, self._Ddiv_J = None, None, None
        self._De_E, self._Dn_E, self._Ddiv_E = None, None, None
    
    @property
    def A_J(self):
        if self._A_J is None:
            _, _, A = self.grid_J.projection.differentials(self.xi_J , self.eta_J,
                                                           self.dxi_J, self.deta_J, 
                                                           R = self.R)
            self._A_J = A
        return self._A_J

    @property
    def A_E(self):
        if self._A_E is None:
            _, _, A = self.grid_E.projection.differentials(self.xi_E , self.eta_E,
                                                           self.dxi_E, self.deta_E, 
                                                           R = self.R)
            self._A_E = A
        return self._A_E
    
    @property
    def shape_J(self):
        return self.grid_J.shape
        
    @property
    def size_J(self):
        return self.grid_J.size
    
    @property
    def shape_E(self):
        return self.grid_E.shape
        
    @property
    def size_E(self):
        return self.grid_E.size
    
    @property
    def lat_J(self):
        return self.grid_J.lat
    
    @property
    def lon_J(self):
        return self.grid_J.lon
        
    @property
    def lat_E(self):
        return self.grid_E.lat
    
    @property
    def lon_E(self):
        return self.grid_E.lon
        
    @property
    def xi_J(self):
        return self.grid_J.xi
    
    @property
    def eta_J(self):
        return self.grid_J.eta
        
    @property
    def xi_E(self):
        return self.grid_E.xi
    
    @property
    def eta_E(self):
        return self.grid_E.eta

    @property
    def latm_J(self):
        return self.grid_J.lat_mesh
    
    @property
    def lonm_J(self):
        return self.grid_J.lon_mesh
        
    @property
    def latm_E(self):
        return self.grid_E.lat_mesh
    
    @property
    def lonm_E(self):
        return self.grid_E.lon_mesh
        
    @property
    def xim_J(self):
        return self.grid_J.xi_mesh
    
    @property
    def etam_J(self):
        return self.grid_J.eta_mesh
        
    @property
    def xim_E(self):
        return self.grid_E.xi_mesh
    
    @property
    def etam_E(self):
        return self.grid_E.eta_mesh

    @property
    def dxi_J(self):
        return self.grid_J.dxi
    
    @property
    def deta_J(self):
        return self.grid_J.deta
        
    @property
    def dxi_E(self):
        return self.grid_E.dxi
    
    @property
    def deta_E(self):
        return self.grid_E.deta
        
    @property
    def De_J(self):
        if self._De_J is None:
            self._De_J, self._Dn_J = self.grid_J.get_Le_Ln()
        return self._De_J
    
    @property
    def Dn_J(self):
        if self._Dn_J is None:
            self._De_J, self._Dn_J = self.grid_J.get_Le_Ln()
        return self._Dn_J
    
    @property
    def Ddiv_J(self):
        if self._Ddiv_J is None:
            self._Ddiv_J = self.grid_J.divergence()
        return self._Ddiv_J
    
    @property
    def De_E(self):
        if self._De_E is None:
            self._De_E, self._Dn_E = self.grid_E.get_Le_Ln()
        return self._De_E
    
    @property
    def Dn_E(self):
        if self._Dn_E is None:
            self._De_E, self._Dn_E = self.grid_E.get_Le_Ln()
        return self._Dn_E
    
    @property
    def Ddiv_E(self):
        if self._Ddiv_E is None:
            self._Ddiv_E = self.grid_E.divergence()
        return self._Ddiv_E
        

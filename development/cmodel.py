""" Model class """
import apexpy
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
# from lompe.secsy import cubedsphere as cs
from .varcheck import check_input
from scipy.interpolate import RectBivariateSpline
from lompe.utils import sunlight, conductance
import xarray as xr
import pandas as pd
from lompe.data_tools import dataloader
from scipy.ndimage import median_filter


RE = 6371.2e3


class Cmodel(object):
    def __init__(self, grid,
                       epoch = 2015., # epoch, decimal year, used for IGRF dependent calculations
                       refh = 110., # apex reference height in km - also used for IGRF altitude
                       spline_smoothing = 0, # Used for conductance inversion
                       dipole = False
                ):
        """
        Conductance model

        Example
        -------
        grid = lompe.cs.CSgrid(*gridparams)

        model = lompe.Cmodel(grid)

        model.add_data(my_hall_conductance_dataset, datatype = 'Hall'))
        model.add_data(my_pedersen_conductance_dataset, datatype = 'Pedersen'))
        model.run_inversion()


        Parameters
        ----------
        grid : CSgrid
            cubed sphere grid - defined using the secsy module
        epoch : float, optional
            decimal year, used for IGRF dependent calculations
            Default: 2015.
        refh : float, optional
            apex reference height in km - also used for IGRF altitude
            Default: 110.
        spline_smoothing : int, optional
            spline smoothing - passed to RectBivariateSpline
            Only relevant if Hall_Pedersen_conductance is None
            Default: 0 
        dipole : bool, optional
            default (False) means that variations in QD eastward direction
            are damped. Set to True to damp in coordinate eastward direction
            Default: False
        """

        assert np.isclose(int(spline_smoothing),spline_smoothing),f"'{spline_smoothing}' is not a valid choice for spline_smoothing (which should be an integer)"

        self.spline_smoothing = spline_smoothing

        # set grid:
        self.grid = grid

        self.lat , self.lon  = np.ravel( self.grid.lat  ), np.ravel( self.grid.lon  )

        # Matrix to evaluate Hall and Pedersen conductance on inner grid:
        # Based on spline interpolation - so start by making spline functions
        M, x_, y_ = np.eye(self.grid.size)[0], self.grid.xi[0], self.grid.eta[:, 0]
        self.splines = [RectBivariateSpline(x_, y_, np.roll(M, i).reshape(grid.shape).T, s = self.spline_smoothing, kx=3, ky=3)
                        for i in range(len(M))]
        self.P = self.conductance_matrix()


        De, Dn = self.grid.get_Le_Ln()
        if dipole: # L gives variation in eastward direction:
            self.L = De
            self.LTL = self.L.T.dot(self.L)
        else: # L gives variation in QD eastward direction:
            apx = apexpy.Apex(epoch, refh = refh)
            f1, f2 = apx.basevectors_qd(self.grid.lat.flatten(), self.grid.lon.flatten(), refh)
            f1 = f1/np.linalg.norm(f1, axis = 0)
            self.L = De * f1[0].reshape((-1, 1)) + Dn * f1[1].reshape((-1, 1))
            self.LTL = self.L.T.dot(self.L)
        self.clear_model()


    def clear_model(self):

        self.m = {'hall':None, 'pedersen':None} # clear conductance model vectors

        self.data = {'hall':[], 'pedersen':[]}


    def run_inversion(self, l1 = 0, #Damping parameter for model norm
                            l2 = 0 #Damping parameter for variation in the magnetic eastward direction
                        ):
        
        #Solving using sparse matrices has been commented out. Need to look into why it was so slow. 
        K = self.grid.size
        # self._G = {'hall':sparse.csc_matrix((0, K)), 'pedersen':sparse.csc_matrix((0, K))}
        self._G = {'hall':np.empty((0, K)), 'pedersen':np.empty((0, K))}
        self._d = {'hall':np.empty( 0),     'pedersen':np.empty( 0)    }
        self._w = {'hall':np.empty( 0),     'pedersen':np.empty( 0)    }

        for dtype in ['hall', 'pedersen']:
            for dataset in self.data[dtype]:
                P = self.conductance_matrix(**dataset.coords)
                # self._G[dtype] = sparse.vstack((self._G[dtype], P))
                self._G[dtype] = np.vstack((self._G[dtype], P))
                #self._d[dtype] = np.hstack((self._d[dtype], np.log(dataset.values)))
                self._d[dtype] = np.hstack((self._d[dtype], dataset.values))
                self._w[dtype] = np.hstack((self._w[dtype], dataset.weights))

        # Gw_hall = self._G['hall'].multiply(self._w['hall'][:,np.newaxis])
        Gw_hall = np.multiply(self._G['hall'].T, self._w['hall'][:,np.newaxis].T).T
        GTGw_hall = Gw_hall.T @ self._G['hall']
        GTdw_hall = Gw_hall.T @ self._d['hall']
        # gtg_mag = np.median(GTGw_hall.diagonal())
        gtg_mag = np.median(np.diagonal(GTGw_hall))
        ltl_mag = np.median(self.LTL.diagonal())
        alpha_hall = np.sqrt(l1 * gtg_mag) #Could implement Michaels GCV score scheme here
        # R_hall = sparse.diags(np.ones(GTGw_hall.shape[0])) * alpha_hall**2
        R_hall = np.eye(GTGw_hall.shape[0]) * alpha_hall**2
        R_LTL = l2 * gtg_mag / ltl_mag * self.LTL
        # self.m['hall'] = spsolve(GTGw_hall+R_hall+R_LTL, GTdw_hall)
        self.m['hall'] = np.linalg.lstsq(GTGw_hall+R_hall+R_LTL, GTdw_hall, rcond = None)[0]

        Gw_pedersen = np.multiply(self._G['pedersen'].T, self._w['pedersen'][:,np.newaxis].T).T
        # Gw_pedersen = self._G['pedersen'].multiply(self._w['pedersen'][:,np.newaxis])
        GTGw_pedersen = Gw_pedersen.T @ self._G['pedersen']
        GTdw_pedersen = Gw_pedersen.T @ self._d['pedersen']
        # gtg_mag = np.median(GTGw_pedersen.diagonal())
        gtg_mag = np.median(np.diagonal(GTGw_pedersen))
        alpha_pedersen = np.sqrt(l1 * gtg_mag) #Could implement Michaels GCV score scheme here
        # R_pedersen = sparse.diags(np.ones(GTGw_pedersen.shape[0])) * alpha_pedersen**2
        R_pedersen = np.eye(GTGw_pedersen.shape[0]) * alpha_pedersen**2
        # self.m['pedersen'] = spsolve(GTGw_pedersen+R_pedersen+R_LTL, GTdw_pedersen)
        self.m['pedersen'] = np.linalg.lstsq(GTGw_pedersen+R_pedersen+R_LTL, GTdw_pedersen, rcond = None)[0]


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



    # CONDUCTANCES
    @check_input
    def conductance_matrix(self, lon = None, lat = None, return_shape = False):
        """ returns design matrix if return_values is False

        Parameters
        ----------
        type: str, optional
            'hall' or 'pedersen', default 'hall'
        """
        xi, eta = self.grid.projection.geo2cube(lon, lat)

        # Build matrix P by evaluating the effect of unit vectors along
        # each dimension of grid. This defines the columns of the P matrix.
        P = np.empty((xi.size, self.grid.size))
        for i in range(self.grid.size):
            M = np.zeros(self.grid.size)
            M[i] = 1
            P[:, i] = self.splines[i].ev(xi, eta)

        return P


    def conductance(self, lon = None, lat = None, type = 'hall'):
        """
        Calculate conductances

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon
        type : string, optional
            set to 'hall' (default) to calculate Hall conductance, or 'pedersen' to calculate
            Pedersen conductance

        Returns
        -------
        Sigma : array
            Conductance [mho], with same shape as lon / lat

        """
        if type not in self.m.keys():
            raise Exception('type must be "hall" or "pedersen"')

        P, shape = self.conductance_matrix(lon, lat, return_shape = True)

        #return np.exp(P.dot(self.m[type]).reshape(shape))
        return P.dot(self.m[type]).reshape(shape)


    def hall(self, lon = None, lat = None):
        """
        Calculate Hall conductance

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon
        type : string, optional
            set to 'hall' (default) to calculate Hall conductance, or 'pedersen' to calculate
            Pedersen conductance

        Returns
        -------
        Sigma : array
            Conductance [mho], with same shape as lon / lat

        """
        return self.conductance(lon, lat, type = 'hall')


    def pedersen(self, lon = None, lat = None):
        """
        Calculate Pedersen conductance

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon
        type : string, optional
            set to 'hall' (default) to calculate Hall conductance, or 'pedersen' to calculate
            Pedersen conductance

        Returns
        -------
        Sigma : array
            Conductance [mho], with same shape as lon / lat

        """
        return self.conductance(lon, lat, type = 'pedersen')



class Cmodel2(object):
    def __init__(self, grid,
                       epoch = 2015., # epoch, decimal year, used for IGRF dependent calculations
                       refh = 110., # apex reference height in km - also used for IGRF altitude
                       dipole = False,
                       S = 1
                ):
        """
        Conductance model

        Example
        -------
        grid = lompe.cs.CSgrid(*gridparams)

        model = lompe.Cmodel(grid)

        model.add_data(my_hall_conductance_dataset, datatype = 'Hall'))
        model.add_data(my_pedersen_conductance_dataset, datatype = 'Pedersen'))
        model.run_inversion()


        Parameters
        ----------
        grid: CSgrid
            cubed sphere grid - defined using the secsy module
        epoch : float, optional
            decimal year, used for IGRF dependent calculations
            Default: 2015.
        refh : float, optional
            apex reference height in km - also used for IGRF altitude
            Default: 110.
        dipole : bool, optional
            default (False) means that variations in QD eastward direction
            are damped. Set to True to damp in coordinate eastward direction
            Default: False
        S : int, optional
            

        """

        # set grid:
        self.grid = grid

        self.lat, self.lon = np.ravel(self.grid.lat), np.ravel(self.grid.lon)

        # matrix L that calculates derivative in magnetic eastward direction on grid2:
        De, Dn = self.grid.get_Le_Ln(S = S)
        Dx, Dy = self.grid.get_Le_Ln(return_dxi_deta = True, S = S)
        Dd     = self.grid.divergence(S = S)
        self.DL = Dd.dot(sparse.vstack((De, Dn)))
        if dipole:
            self.LTL = De.T.dot(De) + Dn.T.dot(Dn)# self.DL.T.dot(self.DL)
        else:
            apx = apexpy.Apex(epoch, refh = refh)
            f1, f2 = apx.basevectors_qd(self.grid.lat.flatten(), self.grid.lon.flatten(), refh)
            f1 = f1 / np.linalg.norm(f1, axis = 0)
            f2 = f2 / np.linalg.norm(f2, axis = 0)
            self.F1 = De.multiply( f1[0].reshape((-1, 1)) ) + Dn.multiply( f1[1].reshape((-1, 1)) )
            self.LTL_F1 = self.F1.T.dot(self.F1)
            self.F2 = De.multiply( f2[0].reshape((-1, 1)) ) + Dn.multiply( f2[1].reshape((-1, 1)) )
            self.LTL_F2 = self.F2.T.dot(self.F2)
            self.LTL = self.LTL_F1 #+ 1e-2 * self.DL#self.LTL_F2 
        #self.LTL = self.LTL_F1 + (Dx.T.dot(Dx) + Dy.T.dot(Dy)).multiply(.1 / np.max(Dx) * np.max(De))
        #self.LTL = Dx.T.dot(Dx) + Dy.T.dot(Dy)
        #self.LTL = self.DTD


        self.clear_model()


    def clear_model(self):

        self.m = {'hall':None, 'pedersen':None} # clear conductance model vectors

        self.data = {'hall':[], 'pedersen':[]}


    def run_inversion(self, alpha = 0 #Damping parameter for variation in the magnetic eastward direction
                        ):

        K = self.grid.size
        self._G = {'hall':sparse.csc_matrix((0, K)), 'pedersen':sparse.csc_matrix((0, K))}
        self._d = {'hall':np.empty( 0),     'pedersen':np.empty( 0)    }
        self._w = {'hall':np.empty( 0),     'pedersen':np.empty( 0)    }

        for dtype in ['hall', 'pedersen']:
            for dataset in self.data[dtype]:
                P = self.conductance_matrix(**dataset.coords)
                self._G[dtype] = sparse.vstack((self._G[dtype], P))
                #self._d[dtype] = np.hstack((self._d[dtype], np.log(dataset.values)))
                self._d[dtype] = np.hstack((self._d[dtype], dataset.values))
                self._w[dtype] = np.hstack((self._w[dtype], dataset.weights))

        Gw_hall = self._G['hall'].multiply(self._w['hall'][:,np.newaxis])
        GTGw_hall = Gw_hall.T @ self._G['hall']
        GTdw_hall = Gw_hall.T @ self._d['hall']
        #alpha_hall = np.sqrt(l1 * gtg_mag) #Could implement Michaels GCV score scheme here
        #R_hall = np.eye(GTGw_hall.shape[0]) * alpha_hall**2
        R_LTL = alpha * self.LTL

        print(type(GTGw_hall), type(R_LTL), type(GTdw_hall), GTGw_hall.shape, R_LTL.shape, GTdw_hall.shape)

        self.m['hall'] = spsolve(GTGw_hall +R_LTL, GTdw_hall)

        print('hm')

        Gw_pedersen = self._G['pedersen'].multiply(self._w['pedersen'][:,np.newaxis])
        GTGw_pedersen = Gw_pedersen.T @ self._G['pedersen']
        gtg_mag = np.median(GTGw_pedersen.diagonal())
        GTdw_pedersen = Gw_pedersen.T @ self._d['pedersen']
        #alpha_pedersen = np.sqrt(l1 * gtg_mag) #Could implement Michaels GCV score scheme here
        #R_pedersen = np.eye(GTGw_pedersen.shape[0]) * alpha_pedersen**2
        self.m['pedersen'] = spsolve(GTGw_pedersen + R_LTL, GTdw_pedersen)


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
                print('You passed {}, which is not in {} - ignored'.format(dtype, list(data.keys())))



    # CONDUCTANCES
    @check_input
    def conductance_matrix(self, lon = None, lat = None, return_shape = False):
        """ returns design matrix if return_values is False

        Parameters
        ----------
        type: str, optional
            'hall' or 'pedersen', default 'hall'
        """

        # build design matrix for conductance:
        i, j = self.grid.bin_index(lon, lat)
        row = ((i != -1) & (j != -1)).nonzero()[0]
        i, j = i[row], j[row]
        iii = np.ravel_multi_index((i, j), self.grid.shape)

        P = sparse.csc_matrix((np.ones_like(iii), (row, iii)), shape = (len(lon), self.grid.size))

        return P


    def conductance(self, lon = None, lat = None, type = 'hall'):
        """
        Calculate conductances

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon
        type : string, optional
            set to 'hall' (default) to calculate Hall conductance, or 'pedersen' to calculate
            Pedersen conductance

        Returns
        -------
        Sigma : array
            Conductance [mho], with same shape as lon / lat

        """
        if type not in self.m.keys():
            raise Exception('type must be "hall" or "pedersen"')

        P, shape = self.conductance_matrix(lon, lat, return_shape = True)

        #return np.exp(P.dot(self.m[type]).reshape(shape))
        return P.dot(self.m[type]).reshape(shape)


    def hall(self, lon = None, lat = None):
        """
        Calculate Hall conductance

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon
        type : string, optional
            set to 'hall' (default) to calculate Hall conductance, or 'pedersen' to calculate
            Pedersen conductance

        Returns
        -------
        Sigma : array
            Conductance [mho], with same shape as lon / lat

        """
        return self.conductance(lon, lat, type = 'hall')


    def pedersen(self, lon = None, lat = None):
        """
        Calculate Pedersen conductance

        Parameters
        ----------
        lon : array, optional
            Longitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lat
        lat : array, optional
            Latitudes [degrees] of the evaluation points, default is center of interior grid points.
            Must have same shape as lon
        type : string, optional
            set to 'hall' (default) to calculate Hall conductance, or 'pedersen' to calculate
            Pedersen conductance

        Returns
        -------
        Sigma : array
            Conductance [mho], with same shape as lon / lat

        """
        return self.conductance(lon, lat, type = 'pedersen')
    
class Cmodel3(object):
    def __init__(self, grid, #CS grid
                       event, # str on format 'YYY-MM-DD'
                       stime, #time used to find closest ssusi image
                       param = 'lbhs', #which parameter to use for estimating conductance
                       factorH = 0.05, #the crude conversion factor from vaules of param to mho unit for Hall conductance
                       factorP = 0.05, #the crude conversion factor from vaules of param to mho unit for Pedersen conductance
                       sat = 'F18', #str, Which ssusi satellite to use
                       basepath = './', #where the ssusi data is located
                       tempfile_path = './', #where to save temp files
                       how = 'mean', #str, 'mean' or 'median' for how computing the binned average
                       spline_smoothing=0, # smoothing factor in rectbivariatepline representation 
                       kx = 3, #spline degree
                       ky = 3, #spline degree
                       EUV = False, #Whether to add EUV conductance
                       f107 = 70, #F10.7 solar radio flux used to quantify EUV irradiation
                       euvtime = None, # time to use for EUV conductance. If None, EUV conductance will be estimated from stime
                       filtersize = 2, #size of median filter to apply to the binned average conductance
                ):
        """
        Conductance model

        Example
        
        
        """
            
        self.grid = grid
        self.param = param
        self.factorH = factorH
        self.factorP = factorP
        self.sat = sat
        self.event = event
        self.stime = stime
        self.basepath = basepath
        self.tempfile_path = tempfile_path
        self.how = how
        binned_hall, binned_pedersen = self.binned_conductance(self.event, self.stime, basepath = self.basepath, tempfile_path=self.tempfile_path, sat=self.sat, how=self.how, param=self.param)                
        self.binned_hall = median_filter(binned_hall, (filtersize,filtersize))
        self.binned_pedersen = median_filter(binned_pedersen, (filtersize,filtersize))
        self.spline_smoothing = spline_smoothing
        self.kx = kx
        self.ky = ky
        self.EUV = EUV
        self.f107 = f107
        self.euvtime = euvtime
        
        
    
    def hall(self, lon, lat):
        '''
        lat: input latitude (in degrees) to evaluate for conductance
        lon: input longitude (in degrees) to evaluate for conductance
        '''
        
        im = self.binned_hall
        x = self.grid.eta[:,0]
        y = self.grid.xi[0,:]
        f = RectBivariateSpline(x, y, im, kx=self.kx, ky=self.ky, s=self.spline_smoothing)
        eval_xi, eval_eta = self.grid.projection.geo2cube(lon, lat)
        if len(eval_xi.shape) == 2:
            eval_xi = eval_xi[0,:]
            eval_eta = eval_eta[:,0]
            gridded = f(eval_eta, eval_xi)
        else:
            gridded = f(eval_eta, eval_xi, grid=False)
    
        nans = np.isnan(gridded)
        gridded[nans] = 0
        
        if self.EUV:
            if self.euvtime is None:
                euvtime = self.stime
            else:
                euvtime = self.euvtime
            sza = sunlight.sza(lat, lon, euvtime)
            EUV = conductance.EUV_conductance(sza, self.f107, 'h')
            if len(lon.shape) == 2:
                EUV = EUV.reshape(lon.shape)
            gridded = gridded + EUV

        return gridded
    
    def pedersen(self, lon, lat):
        '''
        lat: input latitude (in degrees) to evaluate for conductance
        lon: input longitude (in degrees) to evaluate for conductance
        '''
        
        
        im = self.binned_pedersen
        x = self.grid.eta[:,0]
        y = self.grid.xi[0,:]
        f = RectBivariateSpline(x, y, im, kx=self.kx, ky=self.ky, s=self.spline_smoothing)
        eval_xi, eval_eta = self.grid.projection.geo2cube(lon, lat)
        if len(eval_xi.shape) == 2:
            eval_xi = eval_xi[0,:]
            eval_eta = eval_eta[:,0]
            gridded = f(eval_eta, eval_xi)
        else:
            gridded = f(eval_eta, eval_xi, grid=False)
    
        nans = np.isnan(gridded)
        gridded[nans] = 0
        
        if self.EUV:
            if self.euvtime is None:
                euvtime = self.stime
            else:
                euvtime = self.euvtime
            sza = sunlight.sza(lat, lon, euvtime)
            EUV = conductance.EUV_conductance(sza, self.f107, 'p')
            if len(lon.shape) == 2:
                EUV = EUV.reshape(lon.shape)                
            gridded = gridded + EUV
        
        return gridded
    
    def binned_conductance(self, event, stime, param = 'SH', factor = 1., basepath = './', 
                           tempfile_path = './data/', sat = 'F17', how = 'median'):
        '''
        Gets auroral conductance from SSUSI images. Note that location of SSUSI netcdf-files must be specified.
    
        Parameters
        ----------
        grid : Cubed sphere grid object
            DESCRIPTION.
        event : str
            on format 'YYYY-MM-DD'.
        stime : datetime object
            DESCRIPTION.
        param : str, optional
            Which ssusi data parameter to use for conductance. The default is 'SH'.
        factor : float, optional
            Scaling parameter for conductance (if using e.g lbh brightness). The default is 1.
        basepath : str, optional
            Location of APL SSUSI data. The default is './'.
        tempfile_path : str, optional
            Path where ssusi data for the day given in event will be stored. The default is './data/'.
        sat : str, optional
            Which satellite to use. The default is 'F17'
    
        Returns
        -------
        binned_param : 2d array
            Mean conductance values on the same grid as the input CS grid (grid.shape).
    
        '''
        
        #Load ssusi data
        a = apexpy.Apex(stime)
        ssusi = xr.load_dataset(dataloader.read_ssusi(event, basepath=basepath,tempfile_path=tempfile_path))
        use = ssusi.satellite == sat
        ssusi = ssusi.sel(date=use)
        ssusi_i = ssusi.sel(date=stime, method='nearest') #closest in time ssusi image to use
        glat_ssusi, glon_ssusi = a.convert(ssusi_i.mlat, ssusi_i.mlt,'mlt','geo',height=110, 
                                           datetime=pd.to_datetime(ssusi_i['date'].values))
        
        #Make median binned conductance image
        use = np.isfinite(ssusi_i[param]) & self.grid.ingrid(glon_ssusi, glat_ssusi, ext_factor=1)
        index_i, index_j = self.grid.bin_index(glon_ssusi[use], glat_ssusi[use]) #i,j index in grid of each conductance observation
        _index = self.grid._index(index_i, index_j) #correspoinding index in 1D format
        df = pd.DataFrame({'_index':_index, 'index_i':index_i, 'index_j':index_j, 'param':ssusi_i[param].values[use], 'SH':ssusi_i['SH'].values[use], 'SP':ssusi_i['SP'].values[use]})
        
        #Do conversion to conductance
        if param not in ['SH', 'SP']:
            E0 = 1 # Estimate of the
            counts_per_erg = 472. # 306 for lbhl. Need to be estimated from the image and region in question
            h, bb = np.histogram(df.param, bins=30,range=[-400,500])
            mode  = bb[np.argmax(h)]
            df.loc[:,'param'] = df.loc[:,'param'] - 0.7*mode
            df.loc[df['param'] < 0, 'param'] = 0
            je = df['param'] / counts_per_erg
            df.loc[:,'SP'] = (40. * E0 * np.sqrt(je)) / (16. + E0**2)
            df.loc[:,'SH'] = 0.45 * E0**0.85 * df.SP
                        
        if how == 'median':
            _hall = df.groupby('_index').SH.median()
            _pedersen = df.groupby('_index').SP.median()
        elif how == 'mean':
            _hall = df.groupby('_index').SH.mean()
            _pedersen = df.groupby('_index').SP.mean()
        else:
            print('"how" must be mean or median')
            print(1/0)
        # binned_count = df.groupby('_index').param.count()
        binned_i, binned_j = self.grid._index2d(_hall.index)
        binned_hall = np.zeros(self.grid.shape)# * np.nan
        binned_hall[binned_i, binned_j] = _hall.values
        nans = binned_hall == 0
        binned_hall[nans] = np.median(binned_hall)
        binned_i, binned_j = self.grid._index2d(_pedersen.index)
        binned_pedersen = np.zeros(self.grid.shape)# * np.nan
        binned_pedersen[binned_i, binned_j] = _pedersen.values
        nans = binned_pedersen == 0
        binned_pedersen[nans] = np.median(binned_pedersen)        
                
        return (binned_hall, binned_pedersen)
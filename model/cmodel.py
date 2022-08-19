""" Conductance model class """

import apexpy
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import median_filter
from lompe.utils import sunlight, conductance
import xarray as xr
import pandas as pd
from lompe.data_tools import dataloader


class Cmodel(object):

    def __init__(self, grid, event, stime, param = 'lbhs', factorH = 0.05, factorP = 0.05, 
                       sat = 'F18', basepath = './', tempfile_path = './', how = 'mean',
                       spline_smoothing = 0, kx = 3, ky = 3, EUV = True, F107 = 70, 
                       euvtime = None, filtersize = 2):
        """
        Conductance model, based on combination of precipitation characteristics from SSUSI
        images and empirical conductance model of EUV induced conductance.

        Example
        -------
        grid = lompe.cs.CSgrid(*gridparams)
        from lompe.model.cmodel import Cmodel
        cmodel = Cmodel(grid, '2014-12-15', dt.datetime(2015, 12, 15, 1, 40), basepath = './', tempfile_path = './')

        Parameters
        ----------
        grid : CSgrid
            cubed sphere grid. Does not have to be the same grid as is used with Lompe model. 
            This is the grid that the SSUSI auroral will be binned averaged on.
        event : str
            event date on format 'YYYY-MM-DD'
        stime : datetime object
            time used to find closest SSUSI image
        param : str, optional
            which parameter to use for estimating conductance
            Default: 'lbhs'
        factorH, float, optional
            the crude conversion factor from vaules of param to mho unit for Hall conductance
            Default: 0.05
        factorP, float, optional
            the crude conversion factor from vaules of param to mho unit for Pedersen conductance
            Default: 0.05
        sat : str, optional
            Which DMSP satellite to use
            Default: 'F18'
        basepath : str, optional
            where the SSUSI data is located, is passed to dataloader.read_ssusi()
            Default: './'
        tempfile_path : str, optional
            where to save temp files, is passed to dataloader.read_ssusi()
            Default: './'
        how : str, optional
            'mean' or 'median', for how to compute the binned average
            Default: 'mean'
        spline_smoothing : int, optional
            smoothing factor in rectbivariatepline representation
            Default: 0
        kx : int, optional
            Spline degree in x direction
            Default: 3
        ky : int, optional
            Spline degree in y direction
            Default: 3
        EUV : bool, optional 
            Whether to add EUV (solar) conductance
            Default: True
        F107 : int, optional
            F10.7 solar radio flux used to quantify EUV irradiation. Unit: sfu
            Default: 70
        euvtime : datetime object, optional
            time to use for EUV conductance. If None, EUV conductance will be estimated from stime
            Default: None
        filtersize : int, optional
            size of median filter to apply to the binned average conductance. Unit is number of grid cells.
            Default: 2
        """

        #Potulate class with input
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
        self.spline_smoothing = spline_smoothing
        self.kx = kx
        self.ky = ky
        self.EUV = EUV
        self.f107 = F107
        self.euvtime = euvtime

        # do binned average on the specified grid
        binned_hall, binned_pedersen = self.binned_conductance()

        # apply median filter on the binned average SSUSI image
        self.binned_hall = median_filter(binned_hall, (filtersize, filtersize))
        self.binned_pedersen = median_filter(binned_pedersen, (filtersize, filtersize))

    def binned_conductance(self):
        '''
        Compute binned averaged auroral conductance from SSUSI images on provided grid.
        Input parameters are passed from __init__().

        Returns
        -------
        (binned_hall, binned_pedersen) : tuple of two 2D array
            Binned average conductance values on the same grid as the input CS grid.

        '''

        # load SSUSI data
        a = apexpy.Apex(self.stime)
        ssusi = xr.load_dataset(dataloader.read_ssusi(self.event, basepath=self.basepath, tempfile_path=self.tempfile_path))
        use = ssusi.satellite == self.sat
        ssusi = ssusi.sel(date = use)
        ssusi_i = ssusi.sel(date = self.stime, method = 'nearest') # closest in time ssusi image to use
        glat_ssusi, glon_ssusi = a.convert(ssusi_i.mlat, ssusi_i.mlt, 'mlt', 'geo', height = 110,
                                           datetime = pd.to_datetime(ssusi_i['date'].values))

        # make median binned conductance image
        use = np.isfinite(ssusi_i[self.param]) & self.grid.ingrid(glon_ssusi, glat_ssusi, ext_factor = 1)
        index_i, index_j = self.grid.bin_index(glon_ssusi[use], glat_ssusi[use]) # i,j index in grid of each conductance observation
        _index = self.grid._index(index_i, index_j)                              # correspoinding index in 1D format
        df = pd.DataFrame({'_index':_index, 'index_i':index_i, 'index_j':index_j, 'param':ssusi_i[self.param].values[use], 'SH':ssusi_i['SH'].values[use], 'SP':ssusi_i['SP'].values[use]})

        # do conversion to conductance
        if self.param not in ['SH', 'SP']:
            # the following two parameters must be tuned to the specific event if
            # not param is 'SH' or 'SP':
            E0 = 1 # Estimate of the characteristic energy, to be applied in Robinson formulae.
            counts_per_erg = 472. # 306 for lbhl. Need to be estimated from the image and region in question

            # do statistics to estimate noise level
            h, bb = np.histogram(df.param, bins = 30, range = [-400,500])
            mode  = bb[np.argmax(h)]
            df.loc[:,'param'] = df.loc[:,'param'] - 0.7*mode # 0.7 is quite arbitrarily chosen, but serves to avoid subtracting too much
            df.loc[df['param'] < 0, 'param'] = 0
            je = df['param'] / counts_per_erg # energy flux in mW/m2
            
            # applying Robinson formulae
            df.loc[:,'SP'] = (40. * E0 * np.sqrt(je)) / (16. + E0**2) # Pedersen conductance
            df.loc[:,'SH'] = 0.45 * E0**0.85 * df.SP # Hall conductance

        if self.how == 'median':
            _hall = df.groupby('_index').SH.median()
            _pedersen = df.groupby('_index').SP.median()
        elif self.how == 'mean':
            _hall = df.groupby('_index').SH.mean()
            _pedersen = df.groupby('_index').SP.mean()
        else:
            raise RuntimeError('"how" must be mean or median')

        # binned_count = df.groupby('_index').param.count()
        
        # make binned average arrays, and populate cells without SSUSI pixels with a background value
        binned_i, binned_j = self.grid._index2d(_hall.index)
        binned_hall = np.zeros(self.grid.shape)
        binned_hall[binned_i, binned_j] = _hall.values
        nans = binned_hall == 0
        binned_hall[nans] = np.median(binned_hall)
        binned_i, binned_j = self.grid._index2d(_pedersen.index)
        binned_pedersen = np.zeros(self.grid.shape)
        binned_pedersen[binned_i, binned_j] = _pedersen.values
        nans = binned_pedersen == 0
        binned_pedersen[nans] = np.median(binned_pedersen)

        return (binned_hall, binned_pedersen)

    def hall(self, lon, lat):
        """
        Calculate Hall conductance

        Parameters
        ----------
        lon : array
            Longitudes [degrees] of the evaluation points. For Lompe, this will
            be the center of the interior grid
        lat : array
            Latitudes [degrees] of the evaluation points. For Lompe, this will
            be the center of the interior grid

        Returns
        -------
        Sigma : array
            Conductance [mho], with same shape as lon / lat

        """

        im = self.binned_hall
        x = self.grid.eta[:,0]
        y = self.grid.xi[0,:]
        f = RectBivariateSpline(x, y, im, kx = self.kx, ky = self.ky, s = self.spline_smoothing)
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
            gridded = np.sqrt(gridded**2 + EUV**2)

        return gridded

    def pedersen(self, lon, lat):
        """
        Calculate Pedersen conductance

        Parameters
        ----------
        lon : array
            Longitudes [degrees] of the evaluation points. For Lompe, this will
            be the center of the interior grid
        lat : array
            Latitudes [degrees] of the evaluation points. For Lompe, this will
            be the center of the interior grid

        Returns
        -------
        Sigma : array
            Conductance [mho], with same shape as lon / lat

        """

        im = self.binned_pedersen
        x = self.grid.eta[:,0]
        y = self.grid.xi[0,:]
        f = RectBivariateSpline(x, y, im, kx = self.kx, ky = self.ky, s = self.spline_smoothing)
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
            
            gridded = np.sqrt(gridded**2 + EUV**2)

        return gridded

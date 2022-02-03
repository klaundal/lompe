""" Code for working with cubed sphere projection in in a limited region. 
    A cubed sphere grid is a grid that is defined via the projection of a circumscribed 
    cube onto a sphere. The great advantage of this grid is that it avoids any pole 
    problem, and that there is not a large variation in spatial resolution across the 
    grid. The disadvantage is that it is non-orthogonal, which means that differential 
    operators change. The purpose of this script is to take care of that problem.

    This code only implements a grid on (part of) one side of the cube. The purpose
    is to use it for regional data analyses such as SECS, and potentially simple
    modelling. The code uses the equations for the north pole side of the cube

    The grid and associated math is completely based on:
    C. Ronchi, R. Iacono, P.S. Paolucci, The “Cubed Sphere”: A New Method for the 
    Solution of Partial Differential Equations in Spherical Geometry, Journal of 
    Computational Physics, Volume 124, Issue 1, 1996, Pages 93-114, 
    https://doi.org/10.1006/jcph.1996.0047.

    KML, May 2020
    Updates:
    - June 2021: Made differentiation matrix sparse + arbitrary stencil
    - October 2021: Fixed issue with xi and eta not going in expected direction
"""

import numpy as np
from ..secsy import spherical, diffutils
import cartopy.io.shapereader as shpreader
from scipy import sparse
d2r = np.pi / 180

class CSprojection(object):
    def __init__(self, position, orientation):
        """ Set up cubed sphere projection

        The CSprojection is set up by 
        1) rotating to a local coordinate system in which 'position' 
        is at the pole, and 'orientation' defines the x axis (prime meridian)
        2) applying the Ronchi et al. conversions to xi, eta coords on the 
        local coordinates

        Parameters
        ----------
        position: array (lon, lat)
            coordinate at which the cube surface should be 
            tangential to the sphere - the center of the projection.
            Pair of values for longitude and latitude [deg]
        orientation: scalar or 2-element array-like
            orientation of the cube surface.
            if scalar: angle in degrees, that defines the the xi axis: orientation = 0 / 180  
            implies a xi axis in the east-west direction, positive towards east / west. 
            orientation = 90 / 270 impliex a xi axis towards north / south. 
            if 2-element array-like: The elements denote the eastward and northward components
            of a vector that is aligned with the xi axis. 
        """

        self.position = np.array(position)
        self.orientation = np.array(orientation)

        if self.orientation.size == 2: # interpreted as a east, north component:
            self.orientation = self.orientation / np.linalg.norm(self.orientation)
        else: # interpreted as scalar
            assert self.orientation.size == 1, 'orientation must be either scalar or have 2 elements'
            self.orientation = np.array([np.cos(orientation * d2r), np.sin(orientation * d2r)])
        v = np.array([self.orientation[0], self.orientation[1], 0]).reshape((1, 3))

        self.lon0, self.lat0 = position

        # the z axis of local coordinat system described in geocentric coords:
        self.z = np.array([np.cos(self.lat0 * d2r) * np.cos(self.lon0 * d2r), 
                           np.cos(self.lat0 * d2r) * np.sin(self.lon0 * d2r),
                           np.sin(self.lat0 * d2r)])

        # the x axis is the orientation described in ECEF coords:
        self.y = spherical.enu_to_ecef(v, np.array(self.lon0), np.array(self.lat0)).flatten()
        
        # the y axis completes the system:
        self.x = np.cross(self.y, self.z)
 
        # define rotation matrices for rotations between local and geocentric:
        self.R_geo2local = np.vstack((self.x, self.y, self.z)) # rotation matrix from GEO to rotated coords (ECEF)
        self.R_local2geo = self.R_geo2local.T  # inverse


    def geo2cube(self, lon, lat):
        """ convert from geocentric coordinates to cube coords (xi, eta) 
        
        Input parameters must have same shape. Output will have same shape.
        Points that are outside the cube surface will be nans   

        Parameters
        ----------
        lon: array
            geocentric longitude(s) [deg] to convert to cube coords
        lat: array:
            geocentric latitude(s) [deg] to convert to cube coords.

        Returns
        -------
        xi: array
            xi, as defined in Ronchi et al, after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]
        eta: array
            eta, as defined in Ronchi et al., after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]

        """

        lon, lat = np.array(lon), np.array(lat)
        shape = lon.shape
        lon, lat = lon.flatten(), lat.flatten()

        # first convert to local spherical coordinate system (ROT):
        lon, lat = self.geo2local(lon, lat)

        theta, phi = (90 - lat) * d2r, lon * d2r
        X =  np.tan(theta) * np.sin(phi)
        Y = -np.tan(theta) * np.cos(phi)

        xi, eta = np.arctan(X), np.arctan(Y)

        # mask elements outside cube surface by nans:
        ii = theta > np.pi/4
        xi [ii] = np.nan
        eta[ii] = np.nan

        return xi.reshape(shape), eta.reshape(shape)


    def cube2geo(self, xi, eta):
        """ Convert from cube coordinates (xi, eta) to geocentric (lon, lat)

        Input parameters must have same shape. Output will have same shape.
        Points that are outside the cube surface will be nans   

        Parameters
        ----------
        lon: array
            geocentric longitude(s) [deg] to convert to cube coords
        lat: array:
            geocentric latitude(s) [deg] to convert to cube coords.

        Returns
        -------
        xi: array
            xi, as defined in Ronchi et al., after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]
        eta: array
            eta, as defined in Ronchi et al., after lon, lat have been
            converted to local coordinates. Unit is radians [-pi/4, pi/4]


        """
        xi, eta = np.array(xi), np.array(eta)
        shape = xi.shape
        xi, eta = xi.flatten(), eta.flatten()

        X = np.tan(xi)
        Y = np.tan(eta)
        phi = -np.arctan2(X , Y)
        theta = np.arctan(X / np.sin(phi))

        lon, lat = self.local2geo(phi / d2r, 90 - theta / d2r)

        return lon.reshape(shape), lat.reshape(shape)


    def geo2local(self, lon, lat, reverse = False):
        """ Convert from geocentric coordinates to local coordinates 

        lon and lat must have the same shape. Shapes are preserved in output.

        Parameters
        ----------
        lon: array-like
            array of longitudes [deg]
        lat: array-like
            array of latitudes [deg]
        reverse: bool, optional
            set to False (default) if you want to rate from geocentric to local, 
            set to True if you want the opposite rotation

        Returns
        -------
        lon: array-like
            array of longitudes [deg] in new coordinate system
        lat: array-like
            array of latitudes [deg] in new coordinate system
        """
        assert lat.shape == lon.shape
        shape = lat.shape

        # set up ECEF position vectors, and rotate using rotation matrices
        lat, lon = np.array(lat).flatten() * d2r, np.array(lon).flatten() * d2r
        r = np.vstack((np.cos(lat) * np.cos(lon), 
                       np.cos(lat) * np.sin(lon),
                       np.sin(lat)))
        if reverse:
            r_ = self.R_local2geo.dot(r)
        else:
            r_ = self.R_geo2local.dot(r)

        # calcualte spherical coords:
        newlat = np.arcsin (r_[2]) / d2r
        newlon = np.arctan2(r_[1], r_[0]) / d2r

        return (newlon.reshape(shape), newlat.reshape(shape))


    def local2geo(self, lon, lat, reverse = False):
        """ Convert from local coordinates to geocentric coordinates 

        lon and lat must have the same shape. Shapes are preserved in output

        Parameters
        ----------
        lon: array-like
            array of longitudes [deg]
        lat: array-like
            array of latitudes [deg]
        reverse: bool, optional
            set to False (default) if you want to rate from local to geocentric, 
            set to True if you want the opposite rotation

        Returns
        -------
        lon: array-like
            array of longitudes [deg] in new coordinate system
        lat: array-like
            array of latitudes [deg] in new coordinate system

        Note
        ----
        See self.geo2local for implementation
        """
        if reverse:
            return self.geo2local(lon, lat)
        else:
            return self.geo2local(lon, lat, reverse = True)


    def local2geo_enu_rotation(self, lon, lat):
        """ Calculate rotation matrices that transform local ENU to geocentric ENU

        Parameters
        ----------
        lon: array-like
            array of longitudes (local coords) for which rotation matrices should be calculated
        lat: array-like
            array of latitudes (local coords) for which rotation matrices should be calculated

        Returns
        -------
        R_localenu2geoenu: array
            Rotation matrices that rotate ENU vectors in local coordinates to ENU vectors
            in geocentric coordinates. Shape is (N, 2, 2). To get the opposite rotation, 
            use the transpose by swapping the last two axes of the array. The rotation 
            matrices are (2, 2), and should be applied on (east, north) components. The 
            upward component is the same in the two coordinate systems. 
            N is the size of lon and lat (they will be flattened)
        """

        th = (90 - np.array(lat).flatten()) * d2r
        ph = np.array(lon).flatten() * d2r

        # from ENU to ECEF:
        e_R = np.vstack((-np.sin(ph)             ,               np.cos(ph), np.zeros_like(ph))).T # (N, 3)
        n_R = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)       )).T # (N, 3)
        u_R = np.vstack(( np.sin(th) * np.cos(ph),  np.sin(th) * np.sin(ph), np.cos(th)       )).T # (N, 3)

        R_enulocal2eceflocal = np.stack((e_R, n_R, u_R), axis = 2) # (N, 3, 3) with e n u in columns

        # from local to geocentric:
        lon_G, lat_G = self.local2geo(lon, lat)
        th = (90 - lat_G) * d2r
        ph = lon_G * d2r

        e_G = np.vstack((-np.sin(ph)             ,               np.cos(ph), np.zeros_like(ph))).T # (N, 3)
        n_G = np.vstack((-np.cos(th) * np.cos(ph), -np.cos(th) * np.sin(ph), np.sin(th)       )).T # (N, 3)
        u_G = np.vstack(( np.sin(th) * np.cos(ph),  np.sin(th) * np.sin(ph), np.cos(th)       )).T # (N, 3)

        R_ecefgeo2enugeo = np.stack((e_G, n_G, u_G), axis = 1) # (N, 3, 3) with e n u in rows

        # Combine:
        R_enulocal2ecefgeo = np.einsum('ij , njk -> nik', self.R_local2geo, R_enulocal2eceflocal)
        R_enulocal2enugeo  = np.einsum('nij, njk -> nik', R_ecefgeo2enugeo, R_enulocal2ecefgeo)

        # the result should describe a 2D rotation matrix:
        assert np.all( np.isclose(R_enulocal2enugeo[:, 2, 2], 1, atol = 1e-7 ))
        assert np.all( np.isclose(R_enulocal2enugeo[:, 2, np.array([0, 1])], 0, atol = 1e-7 ))
        assert np.all( np.isclose(R_enulocal2enugeo[:, np.array([0, 1]), 2], 0, atol = 1e-7 ))
        return R_enulocal2enugeo[:, :2, :2] # (N, 2, 2)


    def vector_cube_projection(self, east, north, lon, lat, return_xi_eta = True):
        """ Calculate vector components projected on cube
        
        Perfor vector rotation from geographic system to cube
        system, using self.local2geo_enu_rotation and equation
        (14) of Ronchi et al. 

        Parameters
        ----------
        east: array-like
            Array of N eastward (geo) components
        north: array-like
            Array of N northward (geo) components
        lon: array-like
            Array of N longitudes that represent vector positions
        lat: array-like
            Array of N latitudes that represent vector positions
        return_xi_eta: bool, optional
            set to False to return only the vector components. If True
            (default), returning the xi, eta coordinates corresponding 
            to (lon, lat) as well. 

        Returns
        -------
        xi: array-like  (if return_xi_eta is True)
            N element array of xi coordinates
        eta: array-like
            N element array of eta coordinates
        Axi: array-like (if return_xi_eta is True)
            N element array of vector components in xi direction
        Aeta: array-like
            N element array of vector components in eta direction

        """

        east, north, lon, lat = tuple(map(lambda x: np.array(x).flatten(), 
                                          [east, north, lon, lat]))
        Ageo = np.vstack((east, north)).T

        # rotation from geo to local:
        local_lon, local_lat = self.geo2local(lon, lat)
        R_enu_global2local = self.local2geo_enu_rotation(local_lon, local_lat)
        Alocal = np.einsum('nji, nj->ni', R_enu_global2local, Ageo).T

        # rearrange to south, east instead of east, north:
        Alocal = np.vstack((-Alocal[1], Alocal[0])).T

        # calculate the parameters used in transformation matrix:
        xi, eta = self.geo2cube(lon, lat)
        X   = np.tan(-xi)
        Y   = np.tan(-eta)
        delta = 1 + X**2 + Y**2
        C = np.sqrt(1 + X**2)
        D = np.sqrt(1 + Y**2)
        dd = np.sqrt(delta - 1)

        # calculate transformation matrix elements:
        R = np.empty((east.size, 2, 2))
        R[:, 0, 0] = -D * X / dd 
        R[:, 0, 1] =  D * Y / dd / np.sqrt(delta)
        R[:, 1, 0] = -C * Y / dd
        R[:, 1, 1] = -C * X / dd / np.sqrt(delta)

        # rotate and return
        Acube = np.einsum('nij, nj->ni', R, Alocal).T

        # components in xi and eta directions:
        Axi, Aeta = Acube[0], Acube[1]
        if return_xi_eta:
            return xi, eta, Axi, Aeta
        else:
            return Axi, Aeta




    def get_projected_coastlines(self, **kwargs):
        """ generate coastlines in projected coordinates """

        if 'resolution' not in kwargs.keys():
            kwargs['resolution'] = '50m'
        if 'category' not in kwargs.keys():
            kwargs['category'] = 'physical'
        if 'name' not in kwargs.keys():
            kwargs['name'] = 'coastline'

        shpfilename = shpreader.natural_earth(**kwargs)
        reader = shpreader.Reader(shpfilename)
        coastlines = reader.records()
        multilinestrings = []
        for coastline in coastlines:
            if coastline.geometry.geom_type == 'MultiLineString':
                multilinestrings.append(coastline.geometry)
                continue
            lon, lat = np.array(coastline.geometry.coords[:]).T 
            yield self.geo2cube(lon, lat)

        for mls in multilinestrings:
            for ls in mls:
                lon, lat = np.array(ls.coords[:]).T 
                yield self.geo2cube(lon, lat)


    def differentials(self, xi, eta, dxi, deta, R = 1):
        """ calculate magnitudes of line and surface elements 

        Implementation of equations 18-20 of Ronchi et al. 

        Broadcasting rules apply, so that output will have the shape of
        the combination of input parameters:
        dS.shape will be equal to (xi * eta * dxi * deta).shape

        xi, eta, dxi, deta must all be given in radians. dlxi and dleta
        will be given in units of R, and dS in units of R squared (default
        is radian and steradian)

        Parameters
        ----------
        xi: array-like
            xi coordinate(s) of surface element(s)
        eta: array-like
            eta coordinate(s) of surface element(s)
        dxi: array-like
            dimension(s) of surface element(s) in xi direction
        deta: array-like
            dimension(s) of surface element(s) in eta direction
        R: float, optional
            radius of the sphere - default is 1

        Returns
        -------
        dlxi: array-like
            Length of line element(s), in radians or in unit of R,
            along xi direction
        dleta: array-like
            Length of line element(s), in radians or in unit of R,
            along eta direction
        dS: array-like
            Area(s) of surface element(s), in steradians or in 
            the unit of R squared
        """

        X = np.tan(xi)
        Y = np.tan(eta)
        delta = 1 + X**2 + Y**2
        C = np.sqrt(1 + X**2)
        D = np.sqrt(1 + Y**2)

        dlxi  = R * D * dxi  / (delta * np.cos( xi)**2)
        dleta = R * C * deta / (delta * np.cos(eta)**2)

        dS = R**2 * deta * dxi / (delta**(3./2) * np.cos(xi)**2 * np.cos(eta)**2)

        return dlxi, dleta, dS






class CSgrid(object):
    def __init__(self, projection, L, W, Lres, Wres, edges = None, wshift = 0, R = 6371.2):
        """ set up grid for cubed sphere projection 
        
        Create a regular grid in xi,eta-coordinates. The grid will cover a 
        region of the cube surface that is L by W, where L is the dimension along
        the projection.orientation vector. The center of the grid is located at
        projection.position. 

        Parameters
        ----------
        projection: CSprojection
            CSprojection
        L: float
            Dimension of grid along CSprojection.orientation, i.e. the "length"
            of the grid. Dimension corresponds to the dimension of R at the 
            cube-sphere intersection point
        W: float
            Dimension of grid perpendicular CSprojection.orientation, i.e. the 
            "width" of the grid. Dimension corresponds to the dimension of R at 
            the cube-sphere intersection point 
        Lres: float or int
            If float, Lres denotes the size of grid cells in L direction, with 
            dimension same as R (at cube-sphere intersection point)
            if int, Lres denotes the number of grid cells in the Lres direction
        Wres: float or int
            If Lres is float, Wres denotes the size of grid cells in W direction, with 
            dimension same as R (at cube-sphere intersection point). If Lres is int, 
            Wres denotes the number of grid cells in the Wres direction
        wshift: float, optional
            Distance, in units of R, by which to move the grid in the xi-direction, 
            or W direction. Positive numbers will move the center right (towards
            positive xi)
        edges: tuple, optional
            if you want to force the grid in xi/eta space to certain values, provide
            them in this tuple. 
        R: float (optional)
            Radius of the sphere. Default is 6371.2 (~Earth's radius in km) - if you
            use this to model the ionosphere, it is probably a good idea to add ~110 km

        """
        self.projection = projection
        self.R = R
        self.wshift = wshift

        # dimensions::
        self.L = L
        self.W = W
        self.Lres = Lres
        self.Wres = Wres

        # make xi and eta arrays for the grid cell boundaries:
        if edges == None:
            if isinstance(Lres, int):
                xi_edge  = np.linspace(-np.arctan(L/R)/2, np.arctan(L/R)/2, Wres + 1) - wshift/self.R
                eta_edge = np.linspace(-np.arctan(W/R)/2, np.arctan(W/R)/2, Lres + 1) - wshift/self.R
            else:
                xi_edge  = np.r_[-np.arctan(L/R)/2:np.arctan(L/R)/2:np.arctan(Lres/(R))] - wshift/self.R
                eta_edge = np.r_[-np.arctan(W/R)/2:np.arctan(W/R)/2:np.arctan(Wres/(R))] - wshift/self.R
        else:
            xi_edge, eta_edge = edges

        # outer grid limits in xi and eta coords:
        self.xi_min, self.xi_max = xi_edge.min(), xi_edge.max()
        self.eta_min, self.eta_max = eta_edge.min(), eta_edge.max()

        # number of grid cells in L (eta) and W (xi) directions:
        self.NL, self.NW = len(eta_edge) - 1, len(xi_edge) - 1

        # size of grid cells in xi, eta coordinates:
        self.dxi  = xi_edge [1] - xi_edge [0]
        self.deta = eta_edge[1] - eta_edge[0]
        
        # xi, eta coordinates of cell corners:
        self.xi_mesh, self.eta_mesh = np.meshgrid(xi_edge, eta_edge, indexing = 'xy')

        # lon, lat coordiantes of cell corners:
        self.lon_mesh, self.lat_mesh = self.projection.cube2geo(self.xi_mesh, self.eta_mesh)

        # xi, eta coordinates of grid points (cell centers):
        self.xi  = self.xi_mesh [0:-1, 0:-1] + self.dxi  / 2
        self.eta = self.eta_mesh[0:-1, 0:-1] + self.deta / 2

        # geocentric lon, lat [deg] of grid points:
        self.lon, self.lat = self.projection.cube2geo(self.xi, self.eta)
        self.local_lon, self.local_lat = self.projection.geo2local(self.lon, self.lat)

        # geocentric lon, colat [rad] of grid points:
        self.phi, self.theta = self.lon * d2r, (90 - self.lat) * d2r

        # cubed square parameters for grid points (cell centers)
        self.X = np.tan(self.xi)
        self.Y = np.tan(self.eta)
        self.delta = 1 + self.X**2 + self.Y**2
        self.C = np.sqrt(1 + self.X**2)
        self.D = np.sqrt(1 + self.Y**2)

        # set size and shape
        self.size = self.lat.size
        self.shape = self.lat.shape

        # calcualte cell area
        self.A = self.projection.differentials(self.xi , self.eta, self.dxi, self.deta, R = self.R)[2]



    def __repr__(self):
        """ string representation """

        th0, th1 = 2 * self.xi.max() / d2r, 2 * self.eta.max() / d2r
        orientation = self.projection.orientation.flatten()[:2] # east, north components
        lon, lat = self.projection.lon0, self.projection.lat0,

        return( ('{} x {} cubed sphere grid\n'.format(self.shape[0], self.shape[1]) +
                 'Centered at lon, lat = {:.1f}, {:.1f}\n'.format(lon, lat) +
                 'Orientation: {:.2f} east, {:.2f} north, \n'.format(orientation[0], orientation[1]) +
                 'Extent: ~{:.1f} x {:.1f} degrees central angle'.format(th0, th1) ))



    def _index(self, i, j):
        """ 
        Calculate the 1D index that corresponds to the grid index i, j

        Parameters
        ----------
        i: array-like (int)
            row index(es)
        j: array-like (int)
            columns index(es)

        Returns
        -------
        1D array of ints which denote the index(es) of i, j in a flattened version
        of a 2D array of shape (self.NL, self.NW)
        """
        i = np.array(i) % self.NL # wrap negative indices to other end
        j = np.array(j) % self.NW
        
        try:
            return np.ravel_multi_index((i, j), (self.NL, self.NW)).flatten()
        except:
            print('invalid index?', i, j, self.NL, self.NW)
            
            
    def _index2d(self, index1d):
        '''
        Calculate 2d indices from the input 1D index.
        Inverse of _index() function.

        Added 2021-11-02 by JPR

        Parammeters
        -----------
        index1d: array-like (int) of length N of 1d indices to be represented
            by the 2D ij indices

        Returns
        -------
        Two 1D arrays, first containing the i indices, second the j indices
        Same length (N) as input parameter.

        '''
        i = index1d // self.shape[1]
        j = index1d % self.shape[1]

        return i, j            


    def count(self, lon, lat, **kwargs):
        """ 
        Count number of points in each grid cell

        Parameters
        ----------
        lon : array
            array of longitudes [degrees]. Must have same size as lat
        lat : array
            array of latitudes [degrees]. Must have same size as lon
        kwargs : dict, optional
            passed to numpy.histogram2d. Use this if you want density, 
            normed, or weighted histograms for example. 


        Returns
        -------
        count : array
            array with count of how many of the coordinates defined
            by lon, lat are in each grid cell. Same shape as self.lat
            and self.lon
        """

        lon, lat = lon.flatten(), lat.flatten()
        xi, eta = self.projection.geo2cube(lon, lat)

        xi_edges, eta_edges = self.xi_mesh[0, :], self.eta_mesh[:, 0]
        count, xi_, eta_ = np.histogram2d(xi, eta, (xi_edges, eta_edges), **kwargs)

        return(count.T) # transpose because xi should be horizontal and eta vertical


    def bin_index(self, lon, lat):
        """
        Find the bin index (i, j) for each pair (lon, lat)

        Parameters
        ----------
        lon : array
            array of longitudes [degrees]. Must have same size as lat
        lat : array
            array of latitudes [degrees]. Must have same size as lon

        Returns
        -------
        i : array
            index array for each point (lon, lat) along axis 0 (eta direction)
            N-dimensional array where N is equal to lon.size and lat.size
        j : array
            index array for each point (lon, lat) along axis 1 (xi direction)
            N-dimensional array where N is equal to lon.size and lat.size


        Note
        ----
        Points that are outside the grid will be given index -1
        """

        lon, lat = lon.flatten(), lat.flatten()
        xi, eta = self.projection.geo2cube(lon, lat)

        xi_edges, eta_edges = self.xi_mesh[0, :], self.eta_mesh[:, 0]

        i = np.digitize(eta, eta_edges) - 1
        j = np.digitize(xi , xi_edges ) - 1

        iii = ~self.ingrid(lon, lat) # points not in grid
        i[iii] = -1
        j[iii] = -1

        return(i, j)


    def ingrid(self, lon, lat, ext_factor = 1.):
        """ 
        Determine if lon, lat are inside grid boundaries or not.

        Parameters
        ----------
        lon: array
            array of longitudes [degrees] - must have same shape as lat
        lat: array
            array of latitudes [degrees] - must have same shape as lon
        ext_factor: float, optional
            Set ext_factor to a number > 1 to extend self.L and self.W
            by the given factor to include include points that are
            outside the grid        

        Returns
        -------
        array of bools with shape of lon and lat
        """

        lat, lon = np.array(lat), np.array(lon)
        if lon.shape != lat.shape:
            raise Exception('CSgrid.ingrid: lon and lat must have same shape')
        shape = lon.shape
        lon, lat = lon.flatten(), lat.flatten()

        xi, eta = self.projection.geo2cube(lon, lat)
        ximin , ximax  = self.xi_mesh.min()  * ext_factor, self.xi_mesh.max()  * ext_factor
        etamin, etamax = self.eta_mesh.min() * ext_factor, self.eta_mesh.max() * ext_factor

        return ((xi < ximax) & (xi > ximin) & (eta < etamax) & (eta > etamin)).reshape(shape)


    def get_grid_boundaries(self, geocentric = True):
        """ 
        Get grid boundaries for plotting 
            
        Yields tuples of (lon, lat) arrays that outline
        the grid cell boundaries. 

        Example:
        --------
        for c in obj.get_grid_boundaries():
            lon, lat = c
            plot(lon, lat, 'k-', transform = ccrs.Geocentric())
        """
        if geocentric:
            x, y = self.lon_mesh, self.lat_mesh
        else:
            x, y = self.xi_mesh , self.eta_mesh

        for i in range(self.NL + self.NW + 2):
            if i < self.NL + 1:
                yield (x[i, :], y[i, :])
            else:
                i = i - self.NL - 1
                yield (x[:, i], y[:, i])


    def get_Le_Ln(self, S = 1, return_dxi_deta = False, return_sparse = False):
        """ 
        Calculate the matrix that produces the derivative in the 
        eastward and northward directions of a scalar field 
        defined on self

        set return_dxi_deta to True to return the matrices that 
        differentiate in cubed sphere coordinates instead of geo

        Parameters:
        -----------
        S: int, optional
            Stencil size. Default is 1, in which case derivatives will be calculated
            with a 3-point stencil. With S = 2, a 5-point stencil will be used. etc.
        return_dxi_deta: bool, optional
            Set to True if you want matrices that differentiate in the xi / eta 
            directions instead of east /  north
        return_sparse: bool, optional
            Set to True if you want scipy.sparse matrices instead of dense numpy arrays
        """


        dxi = self.dxi
        det = self.deta
        N = self.NL
        M = self.NW

        D_xi = {'rows':[], 'cols':[], 'elements':[]}
        D_et = {'rows':[], 'cols':[], 'elements':[]}

        # index arrays (0 to N, M)
        i_arr = np.arange(N)
        j_arr = np.arange(M)

        # meshgrid versions:
        ii, jj = np.meshgrid(i_arr, j_arr, indexing = 'xy')

        # inner grid points:
        points = np.r_[-S:S+1:1]
        coefficients = diffutils.stencil(points, order = 1)
        i_dx, j_dx = ii  [:, S:-S], jj  [:, S:-S]
        i_dy, j_dy = ii.T[:, S:-S], jj.T[:, S:-S]

        for ll in range(len(points)):
            D_et['rows']    .append(self._index(i_dx, j_dx             ))
            D_et['cols']    .append(self._index(i_dx + points[ll], j_dx))
            D_et['elements'].append(np.full(i_dx.size, coefficients[ll] / det))

            D_xi['rows']    .append(self._index(i_dy, j_dy             ))
            D_xi['cols']    .append(self._index(i_dy, j_dy + points[ll]))
            D_xi['elements'].append(np.full(i_dy.size, coefficients[ll] / dxi))

        # boundaries
        for kk in np.arange(0, S)[::-1]:

            # LEFT
            points = np.r_[-kk:S+1:1] 
            coefficients = diffutils.stencil(points, order = 1)
            i_dx, j_dx = ii  [:, kk], jj  [:, kk]
            i_dy, j_dy = ii.T[:, kk], jj.T[:, kk]

            for ll in range(len(points)):
                D_et['rows']    .append(self._index(i_dx, j_dx             ))
                D_et['cols']    .append(self._index(i_dx + points[ll], j_dx))
                D_et['elements'].append(np.full(i_dx.size, coefficients[ll] / det))

                D_xi['rows']    .append(self._index(i_dy, j_dy             ))
                D_xi['cols']    .append(self._index(i_dy, j_dy + points[ll]))
                D_xi['elements'].append(np.full(i_dy.size, coefficients[ll] / dxi))

            # RIGHT
            points = np.r_[-S:kk+1:1] 
            coefficients = diffutils.stencil(points, order = 1)
            i_dx, j_dx = ii  [:, -(kk + 1)], jj  [:, -(kk + 1)]
            i_dy, j_dy = ii.T[:, -(kk + 1)], jj.T[:, -(kk + 1)]

            for ll in range(len(points)):
                D_et['rows']    .append(self._index(i_dx, j_dx             ))
                D_et['cols']    .append(self._index(i_dx + points[ll], j_dx))
                D_et['elements'].append(np.full(i_dx.size, coefficients[ll] / det))

                D_xi['rows']    .append(self._index(i_dy, j_dy             ))
                D_xi['cols']    .append(self._index(i_dy, j_dy + points[ll]))
                D_xi['elements'].append(np.full(i_dy.size, coefficients[ll] / dxi))


        D_xi = {key:np.hstack(D_xi[key]) for key in D_xi.keys()}
        D_et = {key:np.hstack(D_et[key]) for key in D_et.keys()}

        D_xi = sparse.csc_matrix((D_xi['elements'], (D_xi['rows'], D_xi['cols'])), shape = (N * M, N * M))
        D_et = sparse.csc_matrix((D_et['elements'], (D_et['rows'], D_et['cols'])), shape = (N * M, N * M))

        if return_dxi_deta:
            if return_sparse:
                return D_xi, D_et
            else:
                return np.array(D_xi.todense()), np.array(D_et.todense())

        # convert to gradient compnents
        X = self.X.flatten().reshape((1, -1))
        Y = self.Y.flatten().reshape((1, -1))
        D = self.D.flatten().reshape((1, -1))
        C = self.C.flatten().reshape((1, -1))
        d = self.delta.flatten().reshape((1, -1))

        I = sparse.eye(self.size)

        # equation 21 of Ronchi et al.
        L_xi = (D_xi.multiply(D        ) + D_et.multiply(X * Y / D)) / self.R
        L_et = (D_xi.multiply(X * Y / C) + D_et.multiply(    C    )) / self.R
        dd = np.sqrt(d - 1)

        # conversion from xi/eta to geocentric east/west is accomplished through the
        # matrix in equation 14 of Ronchi et al. 
        # The elements of this matrix are:
        a00 =  D * X / dd 
        a01 = -D * Y / dd / np.sqrt(d) 
        a10 =  C * Y / dd 
        a11 =  C * X / dd / np.sqrt(d)        

        # The a matrix converts from local theta/phi to xi/eta. The elements of
        # the inverse are:
        det = a00*a11 - a01*a10
        b00 =  a11 /det 
        b01 = -a01 /det 
        b10 = -a10 /det 
        b11 =  a00 /det 

        # matrix that converts from xi/eta to local east/north
        Be_ = sparse.hstack((I.multiply(b00), I.multiply(b01)))
        Bn_ = sparse.hstack((I.multiply(b10), I.multiply(b11)))

        # Make rotation matrix from local east/north to geocentric east/south:
        R_l2g = self.projection.local2geo_enu_rotation(self.local_lon.flatten(), self.local_lat.flatten())
        r10 =  -R_l2g[:, 0, 0].reshape((1, -1))
        r11 =  -R_l2g[:, 0, 1].reshape((1, -1))
        r00 =   R_l2g[:, 1, 0].reshape((1, -1))
        r01 =   R_l2g[:, 1, 1].reshape((1, -1))
        Re = sparse.hstack((I.multiply(r00), I.multiply(r01)))
        Rn = sparse.hstack((I.multiply(r10), I.multiply(r11)))
        # where I switched the order of the rows and multiplied first row by -1
        # so that R acts on (south/east) instead of (east/north). 

        # combine all three operations: Differentiation of xi/eta, conversion to local, conversion to global
        L = sparse.vstack((Re, Rn)).dot(sparse.vstack((Be_, Bn_))).dot(sparse.vstack((L_xi, L_et)))

        # and return the upper and lower parts of L:
        Le, Ln = L[:self.size], L[self.size:]
        if return_sparse:
            return Le, Ln
        else:
            return np.array(Le.todense()), np.array(Ln.todense())


    def divergence(self, S = 1, return_sparse = False):
        """ 
        Calculate the matrix that produces the divergence of a vector field

        The returned 2N x N matrix operates on a 1D array that represents a 
        vector field. The array must be of length 2N, where N is the number 
        of grid cells. The first N elements are the eastward components and 
        the last N are the northward components. 

        Note - this code is based on equations (12) and (23) of Ronchi. The 
        'matrification' is explained in my regional data analysis document;
        it is not super easy to understand it from the code alone. 

        Parameters:
        -----------
        S: int, optional
            Stencil size. Default is 1, in which case derivatives will be calculated
            with a 3-point stencil. With S = 2, a 5-point stencil will be used. etc.
        return_sparse: bool, optional
            Set to True if you want scipy.sparse matrices instead of dense numpy arrays
        """


        # 1) construct matrix that operates on [[Vxi], [Veta]] to produce
        #    the divergence of teh vector field V according to equation (23)
        #    of Ronchi et al. 
        # 2) construct matrix that converts from east/north to xi/eta 
        #    in local coords
        # 3) construct matrix that rotates from global to local coords
        # 4) combine all three matrices and return


        # matrices that calculate differentials
        L_xi, L_eta = self.get_Le_Ln(S = S, return_dxi_deta = True, return_sparse = True)

        # define some parameteres that are needed 
        d   = self.delta.flatten().reshape((-1, 1))
        X   = self.X.flatten().reshape(    (-1, 1))
        Y   = self.Y.flatten().reshape(    (-1, 1))
        D   = self.D.flatten().reshape(    (-1, 1))
        C   = self.C.flatten().reshape(    (-1, 1))
        xi  = self.xi.flatten().reshape(   (-1, 1))
        eta = self.eta.flatten().reshape(  (-1, 1))
        R = self.R

        I = sparse.eye(xi.size)

        q1 = d / (R * D * C**2)
        q2 = -np.tan(xi ) / (R * D * C**2 * np.cos(xi )**2)
        p1 = d / (R * C * D**2)
        p2 = -np.tan(eta) / (R * C * D**2 * np.cos(eta)**2)

        # matrix that caculates the divergence with xi/eta components:
        L = sparse.hstack((L_xi.multiply(q1) + I.multiply(q2), L_eta.multiply(p1) + I.multiply(p2)))

        dd = np.sqrt(d - 1)
        aa = -D * Y / dd / np.sqrt(d)
        bb = -D * X / dd
        cc =  C * X / dd / np.sqrt(d)
        dd = -C * Y / dd

        # matrix that rotates from east/north to xi/eta:
        R = sparse.vstack((sparse.hstack((I.multiply(aa), I.multiply(bb))), 
                           sparse.hstack((I.multiply(cc), I.multiply(dd))))) 

        # Combine this with the rotation matrix from geocentric east/north to local east/north:
        R_l2g = self.projection.local2geo_enu_rotation(self.local_lon.flatten(), self.local_lat.flatten())
        R_g2l = np.swapaxes(R_l2g, 1, 2) # transpose to get rotation from geo 2 local

        r00 =  R_g2l[:, 0, 0].reshape((1, -1))
        r01 =  R_g2l[:, 0, 1].reshape((1, -1))
        r10 =  R_g2l[:, 1, 0].reshape((1, -1))
        r11 =  R_g2l[:, 1, 1].reshape((1, -1))

        RR = sparse.vstack((sparse.hstack((I.multiply(r00), I.multiply(r01))),
                            sparse.hstack((I.multiply(r10), I.multiply(r11)))))

        # combine the matrices so we get divergence of east/north
        D = L.dot(R.dot(RR) )
        return D if return_sparse else np.array(D.todense())


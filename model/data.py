""" 
Lompe Data class

The data input to the Lompe inversion should be given as lompe.Data objects. The Data
class is defined here. 

"""
import numpy as np

class Data(object):
    def __init__(self, values, coordinates = None, LOS = None, components = 'all', datatype = 'none', scale = None, error = 0):
        """ 
        Initialize Data object that can be passed to the Emodel.add_data function. 

        All data should be given in SI units.

        The data should be given as arrays with shape (M, N), where M is the number of dimensions
        and N is the number of data points. For example, for 3D vector valued data, M is 3, and
        the rows correspond to the east, north, and up components of the measurements, in that order.
        See documentation on specific data types for details

        The coordinates should be given as arrays with shape (M, N) where M is the number of dimensions
        and N is the number of data points. For example, ground magnetometer data should be provided
        with a (2, N) coordinate array with N values for the longitude and latitude, in degrees in the
        two rows. The order of coordinates is: longitude [degrees], latitude [degrees], radius [m]. See
        documentation on specific data types for details. 

        You must specify the data type. Acceptable types are 
        'ground_mag': Magnetic field perturbations on ground. Unless the components keyword is used, values 
        should be given as (3, N) arrays, with eastward, northward and upward components of the magnetic 
        field in the three rows, in Tesla. The coordinates should be given as (2, N) arrays with the magnetometers' 
        longitudes and latitudes in the two rows. The radius is assumed to be Earth radius. An error (measurement 
        uncertainty) can be given as an N-element array, or as a scalar if the uncertainty is the same for 
        all data points in the dataset. An alternative way of specifying 'ground_mag', if you do not have 
        full 3D measurements, is to provide it as (M, N) values, where M <3, and the rows correspond 
        to the directions that are measured. Specify which directions using the components keyword 
        (see documentation for that keyword for details)

        'space_mag_fac': Magnetic field perturbations in space associated with field-aligned currents.
        Unless the components keyword is used, values should be given as (3, N) arrays, with eastward, 
        northward and upward components of the magnetic field in the three rows, in Tesla. Note that the
        upward component is not used for this parameter, since field-lines are assumed to be radial and 
        FACs therefore have no vertical field (it must still be given). The coordinates should be given 
        as (3, N) arrays with the longitudes, latitudes, and radii of the measurements in the three rows. 

        'space_mag_full': Magnetic field perturbations in space associated with field-aligned currents 
        and horizontal divergence-free currents below the satellite. This is useful for low-flying satellites
        with accurate magnetometers (e.g., Swarm, CHAMP). The format is the same as for 'space_mag_fac'

        'convection': Ionospheric convection velocity perpendicular to the magnetic field, mapped to the
        ionospheric radius. The values should be given as (2, N) arrays, where the two rows correspond to
        eastward and northward components in m/s. The coordinates should be given as (2, N) arrays where the rows
        are longnitude and latitude in degrees. For line-of-sight measurements, the values parameters should be an N
        element array with velocities in the line-of-sight direction. The line-of-sight direction must be specified
        as a (2, N) array using the LOS keyword. The (2, N) LOS parameter should contain the eastward and northward
        components of the line-of-sight vector in the two rows. 

        'Efield': Ionospheric convection electric field, perpendicular to B and mapped to the ionospheric
        radius. The values should be given in [V/m], with the same format as for 'convection'. The LOS keyword
        can be used for this parameter also, if only one component of the electric field is known. 

        'fac': Field-aligned electric current density in A/m^2. It must be provided as a K_J*K_J element array, 
        where the elements correspond to the field-aligned current density at the Lompe inner grid points, in the
        order that they will have after a flatten/ravel operation. Values passed to coordinates will be ignored.
        This parameter is only meant to be used with large-scale datasets or simulation output that can be 
        interpolated to the Lompe model grid. This is different from all the other datatypes used in Lompe.


        Note
        ----
        One purpose of this class is to collect all data sanity checks in one place, and to make sure that 
        the data which is passed to Lompe has the correct shape, valid values etc. We're not quite there yet,
        so be careful! :)


        Parameters
        ----------
        values: array
            array of values in SI units - see specific data type for details
        coordinates: array
            array of  coordinates - see specific data types for details
        datatype: string
            datatype should indicate which type of data it is. They can be
            'ground_mag'     - ground magnetic field perturbation (no main field) data
            'space_mag_full' - space magnetic field perturbation with both FAC and
                               divergence-free current signal
            'space_mag_fac'  - space magnetic field perturbation with only FAC signal
            'convection'     - F region plasma convection data - mapped to R
            'Efield'         - electric field - mapped to R
            'Pedersen'       - Pedersen conductance
            'Hall'           - Hall conductance
        LOS: array, optional
            If the data is line-of-sight (los), indicate the line of sight using a (2, N) 
            array of east, north directions for the N unit vectors pointing in the los
            directions. By default, data is assumed to not be line of sight. Note that 
            LOS is only supported for Efield and convection, which are 2D data types. 
        components: int(s), optional
            indicate which components are included in the dataset. If 'all' (default),
            all components are included. If only one component is included, set to 
            0, 1, or 2 to specify which one: 0 is east, 1 is north, and 2 is up. If 
            two components are included, set to a list of ints (e.g. [0, 2] for east
            and up). NOTE: If LOS is set, this keyword is ignored
        scale: float, optional
            set to a typical scale for the data, in SI units. For example, convection could be
            typically 100 [m/s], and magnetic field 100e-9 [T]. If not set, a default value is
            used for each dataset.
        error: array of same length as values, or int, optional
            Measurement error. Used to calculate the data covariance matrix. Use SI units

        """

        self.isvalid = False
        datatype = datatype.lower()

        if datatype not in ['ground_mag', 'space_mag_full', 'space_mag_fac', 'convection', 'efield', 'fac', 'pedersen', 'hall']:
            print('datatype not recognized')
            return(None)

        scales = {'ground_mag':100e-9, 'space_mag_full':200e-9, 'space_mag_fac':200e-9, 'convection':100, 'efield':10e-3, 'fac':1e-6, 'pedersen':1, 'hall':1}

        if scale is None:
            self.scale = scales[datatype.lower()]
        else:
            self.scale = scale

        
        self.datatype = datatype
        self.values = values
        if coordinates is not None:
            if datatype.lower() == 'fac':
                print('Warning: FAC data must be defined on the whole Emodel.grid_J, but this is not checked.')
            if coordinates.shape[0] == 2:
                self.coords = {'lon':coordinates[0], 'lat':coordinates[1]}
            if coordinates.shape[0] == 3:
                self.coords = {'lon':coordinates[0], 'lat':coordinates[1], 'r':coordinates[2]}
        else:
            self.coords = {}
            assert datatype.lower() == 'fac'
        self.isvalid = True
        if np.ndim(self.values) == 2:
            self.N = self.values.shape[1] # number of data points
        if np.ndim(self.values) == 1:
            self.N = self.values.size


        if (LOS is not None) & (datatype in ['convection', 'efield']):
            self.los = LOS # should be (2, N) east, north components of line of sight vectors
            self.components = [0,1] #2021-10-29: jreistad added this to work with how components is used in model.py. Could avoid this slightly non-inututuve value by modifying model.py instead.
        else:
            self.los = None
            if components == 'all':
                self.components = [0, 1, 2]
            else: # components is specified as an int or a list of ints
                self.components = np.sort(np.array(components).flatten())
                assert np.all([i in [0, 1, 2] for i in self.components]), 'component(s) must be in [0, 1, 2]'


        # make data error:
        if np.array(error).size == 1:
            self.error = np.full(self.N, error)
        else:
            self.error = error

        # check that number of data points and coordinates match:
        if self.coords['lat'].size != np.array(self.values, ndmin = 2).shape[1]:
            raise Exception('not the same number of coordinates and data points')

        # remove nans from the dataset:
        iii = np.isfinite(self.values)
        if iii.ndim > 1:
            iii = np.all(iii, axis = 0) 

        self.subset(iii)


    def subset(self, indices):
        """
        modify the dataset so that it only contains data points given by the passed indices.
        The indices are 1D, so if the dataset is vector-valued, indices refer to the column
        index; it selects entire vectors, you can not select only specific components

        """

        self.values  = np.array(self.values , ndmin = 2)[:, indices].squeeze()
        self.error = self.error[indices]

        for key in self.coords.keys():
            self.coords[key] = self.coords[key][indices]

        if self.los is not None:
            self.los = self.los[:, indices]

        # update number of data points:
        if np.ndim(self.values) == 2:
            self.N = self.values.shape[1] 
        if np.ndim(self.values) == 1:
            self.N = self.values.size

        return self




    def __str__(self):
        return(self.datatype + ': ' + str(self.values))

    def __repr__(self):
        return(str(self))

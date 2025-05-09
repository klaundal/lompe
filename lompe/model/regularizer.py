#%% Import 

import numpy as np
import apexpy

#%% Regularizer

quantities = ['model', 'FAC']
derivatives = ['ns', 'ew']

class Regularizer(object):
    def __init__(self, gH, builder=None, lreg=0, LTL=None, quantity='model', 
                 scale=True, derivative=None, dipole=None, epoch=2015, refh=110, 
                 lat_lim=None):
        
        self.lreg = lreg # Regularization parameter
        self.quantity = quantity # Quantity ti be regularized
        self.derivative = derivative # Derivative of regularized quantity
        self.LTL = LTL # Roughening matrix
        self.scale = scale # Enable regularization scaling
        self.LTL_scale = 1 # Scale
        if dipole is None: # Enable magnetic derivatives
            self.mag = False
        else:
            self.mag = True
        self.epoch = epoch # Epoch for magnetic derivatives [year]
        self.refh = refh # Apex height for magnetic derivatives [km]
        self.lat_lim = lat_lim
                
        # Check if the requested quantity exists
        if self.LTL is None and self.quantity not in quantities:
            raise ValueError(f'Quantity not recognized: {quantity}')
        
        # Check if the requested derivative exists
        if self.LTL is None and self.derivative is not None and self.derivative not in derivatives:
            raise ValueError(f'Derivative not recognized: {derivative}')
        
        # Check if builder is present if necessary
        if self.LTL is None and quantity != 'model' and builder is None:
            raise ValueError('No builder provided')
        
        # Check that LTL has known shape
        if self.LTL is not None and self.LTL.shape[0] is not self.LTL.shape[1]:
            raise ValueError('LTL has to be square')
        if self.LTL is not None and self.LTL.shape[0] is not gH.size_E:
            raise ValueError('LTL does not have a known size')
        
        # Start by getting the right quantity
        if self.LTL is None:
            if self.quantity == 'model':
                L = np.eye(gH.size_E)
            if self.quantity == 'FAC':
                L = builder.FAC_matrix_CF()
        
        # Then determine derivative - mag=False
        if self.LTL is None and self.derivative is not None and not self.mag:
            if L.shape[1] is gH.size_J:
                lat, lon = gH.lat_J.flatten(), gH.lon_J.flatten()
                if derivative == 'ns':
                    self.D = gH.Dn_J
                else:
                    self.D = gH.De_J
                    
            if L.shape[1] is gH.size_E:
                lat, lon = gH.lat_E.flatten(), gH.lon_E.flatten()
                if derivative == 'ns':
                    self.D = gH.Dn_E
                else:
                    self.D = gH.De_E    

        # Then determine derivative - mag=True
        if self.LTL is None and self.derivative is not None and self.mag:
            if L.shape[1] is gH.size_J:
                lat, lon = gH.lat_J.flatten(), gH.lon_J.flatten()
                De, Dn = gH.De_J, gH.Dn_J
            else:
                lat, lon = gH.lat_E.flatten(), gH.lon_E.flatten()
                De, Dn = gH.De_E, gH.Dn_E

            apx = apexpy.Apex(epoch=self.epoch, refh=self.refh)
            mlat, mlon = apx.geo2apex(lat, lon, self.refh)
            f1, f2 = apx.basevectors_qd(lat, lon, self.refh)
            f1 = f1/np.linalg.norm(f1, axis = 0)
            f2 = f2/np.linalg.norm(f2, axis = 0)

            if derivative == 'ns':
                self.D = De2 * f2[0].reshape((-1, 1)) + Dn2 * f2[1].reshape((-1, 1))
            else:
                self.D = De2 * f1[0].reshape((-1, 1)) + Dn2 * f1[1].reshape((-1, 1))

        # Apply latitude limit if requested
        if self.LTL is None and self.lat_lim is not None:
            try:
                a, b = ew_regularization_limit
            except:
                raise Exception('ew_regularization_limit should have two and only two values')
            self.lat_w = lambda lat: np.where(lat < a, 1, np.where(lat > b, 0, (b - lat) / (b - a)))
            
            if self.mag:
                self.D = self.D * lat_w(self.hemisphere * mlat).reshape((-1, 1))
            else:
                self.D = self.D * lat_w(self.hemisphere *  lat).reshape((-1, 1))

        # Finally, combine
        if self.LTL is None:
            if self.derivative is None:
                self.LTL = L.T.dot(L)
            else:
                self.LTL = L.T.dot(self.D.T).dot(self.D).dot(L)
        
        # If scaling is requested
        self.LTL_scale = 1
        if scale:
            self.LTL_scale = np.median(np.diag(self.LTL))
        

#%% Import

import numpy as np
import scipy

#%% Fun

def lstsq_inv(var, var2=None, reg=0):
    if var2 is None:
        var2 = np.eye(var.shape[0])
    return scipy.linalg.lstsq(var + reg*np.median(np.diag(var))*np.eye(var.shape[0]), var2, lapack_driver='gelsy')[0]

class KalmanFilter:
    def __init__(self, H, Q, R, Pt, Ptn, xt, xtn, norm_limit_factor=1.5, A1=None, A2=None):
        """
        A: State transition matrix
        H: Measurement matrix
        Q: Process noise covariance
        R: Measurement noise covariance
        P: State covariance matrix
        x: Initial state estimate
        """
        #self.A = A
        self.A1 = A1
        self.A2 = A2
        self.H = H
        self.Q = Q
        self.R = R
        self.P = 0
        self.Pt = Pt
        self.Ptn = Ptn
        self.x = 0
        self.xt = xt
        self.xtn = xtn
        self.norm_limit_factor = norm_limit_factor
    
    def predict(self):
        """Predict the next state."""
        self.x = 2*self.xt - self.xtn
        self.P = 4*self.Pt + self.Ptn + self.Q

    def predict_complex(self):
        """Predict the next state."""
        self.x = self.A1.dot(self.xt) + self.A2.dot(self.xtn)
        self.P = self.A1.dot(self.Pt).dot(self.A1.T) + self.A2.dot(self.Ptn).dot(self.A2.T) + self.Q
                
    def update_MC(self, z, reg=0):
        
        R_inv = lstsq_inv(self.R)
        P_inv = lstsq_inv(self.P)
        GTG = self.H.T.dot(R_inv).dot(self.H) + P_inv
        GTG += reg*np.median(np.diag(GTG))*np.eye(GTG.shape[0])        
        self.P = lstsq_inv(GTG)
        
        GTd = self.H.T.dot(R_inv).dot(z) + P_inv.dot(self.x)
        
        self.x = self.P.dot(GTd)
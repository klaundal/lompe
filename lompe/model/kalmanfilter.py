#%% Import

import numpy as np
import scipy
from .solver import Solver

#%% Fun

def lstsq_inv(var, var2=None, reg=0):
    if var2 is None:
        var2 = np.eye(var.shape[0])
    return scipy.linalg.lstsq(var + reg*np.median(np.diag(var))*np.eye(var.shape[0]), var2, lapack_driver='gelsy')[0]

class KalmanFilter:
    def __init__(self, H, Q, R, Pt, Ptn, xt, xtn, A1=None, A2=None, reg=None):
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
        self.P = None
        self.Pt = Pt
        self.Ptn = Ptn
        self.bP = None
        self.x = None
        self.xt = xt
        self.xtn = xtn
        self.bx = None
        self.reg = reg
    
    def predict(self):
        if self.A1 is None or self.A2 is None:
            self.predict_simple()
        else:
            self.predict_complex()
    
    def predict_simple(self):
        """Predict the next state."""
        self.bx = 2*self.xt - self.xtn
        self.bP = 4*self.Pt + self.Ptn + self.Q

    def predict_complex(self):
        """Predict the next state."""
        self.bx = self.A1.dot(self.xt) + self.A2.dot(self.xtn)
        self.bP = self.A1.dot(self.Pt).dot(self.A1.T) + self.A2.dot(self.Ptn).dot(self.A2.T) + self.Q
                
    def update_MC(self, z, H=None, R=None, reg=None):
        
        if H is not None:
            self.H = H
        if self.H is None:
            raise ValueError('H has to be provided')
        
        if R is not None:
            self.R = R
        if self.R is None:
            raise ValueError('H has to be provided')
        
        if reg is not None:
            self.reg = reg
        
        R_inv = lstsq_inv(self.R)
        bP_inv = lstsq_inv(self.bP)

        GTG = self.H.T.dot(R_inv).dot(self.H) + bP_inv
        GTd = self.H.T.dot(R_inv).dot(z) + bP_inv.dot(self.bx)
        
        slv = Solver(GTG=GTG, GTd=GTd, reg=self.reg)
        slv.solve_inverse_problem(posterior=True)

        self.P = slv.Cmpost
        self.x = slv.m
        
        #GTG += reg*np.median(np.diag(GTG))*np.eye(GTG.shape[0])        
        #self.P = lstsq_inv(GTG)
        
        #GTd = self.H.T.dot(R_inv).dot(z) + P_inv.dot(self.x)
        
        #self.x = self.P.dot(GTd)
#%% Import 

import numpy as np
from scipy.linalg import lstsq

#%% Check is GPU depedencies and hardware is available

try:
    import cupy as cp
    import gc
    gpu_avail = True
except:
    gpu_avail = False

#%%

class Solver(object):
    def __init__(self, G, d, w=None, reg=None, use_gpu=False, **kwargs):
        
        # Store relevant inputs
        self.G = G
        self.d = d
        self.w = w
        self.use_gpu = use_gpu
        self.kwargs = kwargs
        
        # Initiate relevant variables
        self._GT = None
        self._GTd = None
        self._GTG = None
        self._GTG_scale = None
        
        self.reg = reg
        self._LTL = None
        
        self._Cmpost = None
        self._Rm = None
        self.m = None
        
    @property
    def GT(self):
        if self._GT is None:
            if self.w is None:
                self._GT = self.G.T
            elif len(self.w.shape) == 1:
                self._GT = self.G.T.dot(np.diag(self.w))
            else:
                self._GT = self.G.T.dot(self.w)
        return self._GT
                
    @property
    def GTd(self):
        if self._GTd is None:
            self._GTd = self.GT.dot(self.d)
        return self._GTd

    @property
    def GTG(self):
        if self._GTG is None:
            self._GTG = self.GT.dot(self.G)
        return self._GTG
    
    @property
    def GTG_scale(self):
        if self._GTG_scale is None:
            self._GTG_scale = np.median(np.diag(self.GTG))
        return self._GTG_scale

    @property
    def LTL(self):
        if self._LTL is None:
            self._LTL = 0
            if self.reg is not None:
                for regi in self.reg:
                    if regi.scale:
                        self._LTL += regi.lreg * regi.LTL / regi.LTL_scale * self.GTG_scale
                    else:
                        self._LTL += regi.lreg * regi.LTL
        return self._LTL
            
    @property
    def Cmpost(self):
        if self._Cmpost is None:
            if gpu_avail and self.use_gpu:
                self._Cmpost = cp.asnumpy(cp.linalg.solve(cp.array(self.GTG), cp.array(np.eye(self.GTG.shape[0]))))
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.Device().synchronize()
                gc.collect()
            else:
                self._Cmpost = lstsq(self.GTG + self.LTL, np.eye(self.GTG.shape[0]), **self.kwargs)[0]
        return self._Cmpost
    
    @property
    def Rm(self):
        if self._Rm is None:
            self._Rm = self.Cmpost.dot(self.GTG)
        return self._Rm
    
    def solve_inverse_problem(self, posterior=False, **kwargs):
        self.use_gpu = kwargs.pop('use_gpu', self.use_gpu)        
        if posterior:
            self.m = self.Cmpost.dot(self.GTd)
        else:
            self.m = lstsq(self.GTG + self.LTL, self.GTd, **kwargs)[0]
                


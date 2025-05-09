
    def calc_resolution(self, innerGrid=True):
        
        '''
        Calculate spatial resolution following Madelaire et al. [2023]
        '''
        
        # Get res in km
        colatxi = 90 - self.grid_E.lat
        lonxi = self.grid_E.lon
        d2r = np.pi/180

        xxi = self.R*1e-3 * np.sin(colatxi*d2r) * np.cos(lonxi*d2r)
        yxi = self.R*1e-3 * np.sin(colatxi*d2r) * np.sin(lonxi*d2r)
        zxi = self.R*1e-3 * np.cos(colatxi*d2r)

        euclidxi = np.median(np.sqrt(np.diff(xxi, axis=1)**2 + np.diff(yxi, axis=1)**2 + np.diff(zxi,axis=1)**2))
        euclideta = np.median(np.sqrt(np.diff(xxi, axis=0)**2 + np.diff(yxi, axis=0)**2 + np.diff(zxi,axis=0)**2))
        
        # Left right function
        def left_right(PSF_i, fraq=0.5):
        
            i_max = np.argmax(PSF_i)    
            PSF_max = PSF_i[i_max]
            
            j = 0
            i_left = 0
            left_edge = True
            while (i_max - j) >= 0:
                if PSF_i[i_max - j] < fraq*PSF_max:
                
                    dPSF = PSF_i[i_max - j + 1] - PSF_i[i_max - j]
                    dx = (fraq*PSF_max - PSF_i[i_max - j]) / dPSF
                    i_left = i_max - j + dx
                
                    left_edge = False
                
                    break
                else:
                    j += 1

            j = 0
            i_right = len(PSF_i) - 1
            right_edge = True
            while (i_max + j) < len(PSF_i):
                if PSF_i[i_max + j] < fraq*PSF_max:
                
                    dPSF = PSF_i[i_max + j] - PSF_i[i_max + j - 1]
                    dx = (fraq*PSF_max - PSF_i[i_max + j - 1]) / dPSF
                    i_right = i_max + j - 1 + dx 
                
                    right_edge = False
                
                    break
                else:
                    j += 1
        
            flag = True
            if left_edge and right_edge:
                print('I think something is wrong')
                flag = False
            elif left_edge:
                i_left = i_max - (i_right - i_max)
                flag = False
            elif right_edge:
                i_right = i_max + (i_max - i_left)
                flag = False
        
            return i_left, i_right, i_max, flag
        
        # Allocate space
        xiRes = np.zeros(self.grid_E.shape)
        etaRes = np.zeros(self.grid_E.shape)
        xiResFlag = np.zeros(self.grid_E.shape)
        etaResFlag = np.zeros(self.grid_E.shape)
        resL = np.zeros(self.grid_E.shape)
        
        # Loop over all PSFs
        for i in range(xiRes.size):
                        
            row = i//xiRes.shape[1]
            col = i%xiRes.shape[1]
            
            PSF = abs(self.Rmatrix[:, i]).reshape(self.grid_E.shape)
            
            ii = np.argmax(PSF)
            rowPSF = ii//self.grid_E.shape[1]
            colPSF = ii%self.grid_E.shape[1]
            
            dxi = abs(colPSF - col) * euclidxi
            deta = abs(rowPSF - row) * euclideta
            
            resL[row, col] = np.sqrt(dxi**2 + deta**2)
            
            PSF_xi = np.sum(PSF, axis=0)
            if innerGrid:
                PSF_xi[0] = 0.99*np.max(PSF_xi[1:-1])
                PSF_xi[-1] = 0.99*np.max(PSF_xi[1:-1])
            i_left, i_right, i_max, flag = left_right(PSF_xi)
            xiRes[row, col] = euclidxi * (i_right - i_left)
            xiResFlag[row, col] = flag
            
            PSF_eta = np.sum(PSF, axis=1)
            if innerGrid:
                PSF_eta[0] = 0.99*np.max(PSF_eta[1:-1])
                PSF_eta[-1] = 0.99*np.max(PSF_eta[1:-1])
            i_left, i_right, i_max, flag = left_right(PSF_eta)
            etaRes[row, col] = euclideta * (i_right - i_left)
            etaResFlag[row, col] = flag
        
        if innerGrid:
            xiResFlag[:, [0, -1]] = 0
            xiResFlag[[0, -1], :] = 0
            etaResFlag[:, [0, -1]] = 0            
            etaResFlag[[0, -1], :] = 0
        
        self.xiRes = xiRes
        self.etaRes = etaRes
        self.xiResFlag = xiResFlag
        self.etaResFlag = etaResFlag
        self.resL = resL
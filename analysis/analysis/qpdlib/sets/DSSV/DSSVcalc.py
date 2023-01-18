import sys, os
import  matplotlib
matplotlib.use('Agg')
import pylab as py
matplotlib.rc('text',usetex=True)

import numpy as np
import copy
import analysis.qpdlib.sets.DSSV.dssvlib as dssvlib

from scipy.interpolate import griddata
from scipy.integrate import cumtrapz

cwd = 'analysis/qpdlib/sets/DSSV'

#to get mean and errors on PDF (or other derived quantities)
#mean = PDF[0]
#std  = np.zeros(len(X))
#for i in range(1,20)
#    std += (PDF[i] - PDF[-i])**2
#std = np.sqrt(std)/2.0

class DSSV:

    def __init__(self):

        self.L = int((len(os.listdir('analysis/qpdlib/sets/DSSV/Grids')) - 1)/2)

    def xfxQ2(self,flav,X,Q2):

        PDF = np.zeros((2*self.L + 1, len(X)))
        for i in range(-self.L,self.L+1):
            dssvlib.dssvini(i)
            for j in range(len(X)):
                uv,dv,ub,db,s,g = dssvlib.dssvfit(X[j],Q2)
                if   flav==1:  pdf = dv + db 
                elif flav==2:  pdf = uv + ub 
                elif flav==3:  pdf = s
                elif flav==-1: pdf = db
                elif flav==-2: pdf = ub
                elif flav==-3: pdf = s
                elif flav==21: pdf = g
                else:
                    print('flav %s not available'%flav)
                    sys.exit()
                PDF[i][j] = pdf

        return PDF



  
        
        
        
        
        
        
        
        
        
        
        
        

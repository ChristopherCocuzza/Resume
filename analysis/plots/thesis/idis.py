#!/usr/bin/env python
import sys, os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import copy
import pandas as pd
import scipy as sp
from scipy.interpolate import griddata

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

## matplotlib
matplotlib.rcParams['text.latex.preview']=True
import pylab as py
from matplotlib.ticker import MultipleLocator

import kmeanconf as kc

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack fitlib
from fitlib.resman import RESMAN

## from fitpack analysis
from analysis.corelib import core
from analysis.corelib import classifier

cwd = 'plots/thesis'

def get_xbins(data,kind):
    query = {}
    if kind == 'p':
        query[26] = data.query('X > 4.80e-3 and X <= 5.70e-3') # x = 0.005 (i = 26)
        query[25] = data.query('X > 7.20e-3 and X <= 9.30e-3') # x = 0.008 (i = 25)
        query[24] = data.query('X > 1.15e-2 and X <= 1.40e-2') # x = 0.013 (i = 24)
        query[23] = data.query('X > 1.65e-2 and X <= 1.90e-2') # x = 0.018 (i = 23)
        query[22] = data.query('X > 2.30e-2 and X <= 2.90e-2') # x = 0.026 (i = 22)
        query[21] = data.query('X > 3.40e-2 and X <= 3.80e-2') # x = 0.036 (i = 21)
        query[20] = data.query('X > 4.65e-2 and X <= 5.40e-2') # x = 0.050 (i = 20)
        query[19] = data.query('X > 6.50e-2 and X <= 7.30e-2') # x = 0.070 (i = 19)
        query[18] = data.query('X > 8.50e-2 and X <= 9.20e-2') # x = 0.090 (i = 18)
        query[17] = data.query('X > 9.80e-2 and X <= 10.3e-2') # x = 0.100 (i = 17)
        query[16] = data.query('X > 10.8e-2 and X <= 11.3e-2') # x = 0.110 (i = 16)
        query[15] = data.query('X > 13.6e-2 and X <= 14.6e-2') # x = 0.140 (i = 15)
        query[14] = data.query('X > 17.1e-2 and X <= 18.7e-2') # x = 0.180 (i = 14)
        query[13] = data.query('X > 21.7e-2 and X <= 23.7e-2') # x = 0.230 (i = 13)
        query[12] = data.query('X > 26.0e-2 and X <= 29.0e-2') # x = 0.280 (i = 12)
        query[11] = data.query('X > 33.0e-2 and X <= 36.0e-2') # x = 0.350 (i = 11)
        query[10] = data.query('X > 42.0e-2 and X <= 45.0e-2') # x = 0.430 (i = 10)
        query[9]  = data.query('X > 45.0e-2 and X <= 48.0e-2') # x = 0.460 (i = 9 )
        query[8]  = data.query('X > 48.0e-2 and X <= 52.0e-2') # x = 0.500 (i = 8 )
        query[7]  = data.query('X > 52.0e-2 and X <= 55.0e-2') # x = 0.530 (i = 7 )
        query[6]  = data.query('X > 55.0e-2 and X <= 58.0e-2') # x = 0.560 (i = 6 )
        query[5]  = data.query('X > 58.0e-2 and X <= 60.0e-2') # x = 0.590 (i = 5 )
        query[4]  = data.query('X > 60.0e-2 and X <= 63.0e-2') # x = 0.610 (i = 4 )
        query[3]  = data.query('X > 63.0e-2 and X <= 66.0e-2') # x = 0.650 (i = 3 )
        query[2]  = data.query('X > 73.0e-2 and X <= 76.0e-2') # x = 0.750 (i = 2 )
        query[1]  = data.query('X > 84.0e-2 and X <= 86.0e-2') # x = 0.850 (i = 1 )
        query[0]  = data.query('X > 86.0e-2 and X <= 90.0e-2') # x = 0.880 (i = 0 )

    if kind == 'd':
        query[18] = data.query('X > 6.50e-2 and X <= 7.30e-2') # x = 0.070 (i = 18)
        query[17] = data.query('X > 8.50e-2 and X <= 9.20e-2') # x = 0.090 (i = 17)
        query[16] = data.query('X > 9.80e-2 and X <= 10.3e-2') # x = 0.100 (i = 16)
        query[15] = data.query('X > 13.6e-2 and X <= 14.6e-2') # x = 0.140 (i = 15)
        query[14] = data.query('X > 17.1e-2 and X <= 18.7e-2') # x = 0.180 (i = 14)
        query[13] = data.query('X > 21.7e-2 and X <= 23.7e-2') # x = 0.230 (i = 13)
        query[12] = data.query('X > 26.0e-2 and X <= 29.0e-2') # x = 0.280 (i = 12)
        query[11] = data.query('X > 33.0e-2 and X <= 36.0e-2') # x = 0.350 (i = 11)
        query[10] = data.query('X > 42.0e-2 and X <= 45.0e-2') # x = 0.430 (i = 10)
        query[9]  = data.query('X > 45.0e-2 and X <= 48.0e-2') # x = 0.460 (i = 9 )
        query[8]  = data.query('X > 48.0e-2 and X <= 52.0e-2') # x = 0.500 (i = 8 )
        query[7]  = data.query('X > 52.0e-2 and X <= 55.0e-2') # x = 0.530 (i = 7 )
        query[6]  = data.query('X > 55.0e-2 and X <= 58.0e-2') # x = 0.560 (i = 6 )
        query[5]  = data.query('X > 58.0e-2 and X <= 60.0e-2') # x = 0.590 (i = 5 )
        query[4]  = data.query('X > 60.0e-2 and X <= 63.0e-2') # x = 0.610 (i = 4 )
        query[3]  = data.query('X > 63.0e-2 and X <= 66.0e-2') # x = 0.650 (i = 3 )
        query[2]  = data.query('X > 73.0e-2 and X <= 76.0e-2') # x = 0.750 (i = 2 )
        query[1]  = data.query('X > 84.0e-2 and X <= 86.0e-2') # x = 0.850 (i = 1 )
        query[0]  = data.query('X > 86.0e-2 and X <= 90.0e-2') # x = 0.880 (i = 0 )

    if kind == 'd/p':
        query[17] = data.query('X > 4.80e-3 and X <= 5.70e-3') # x = 0.005 (i = 17)
        query[16] = data.query('X > 7.20e-3 and X <= 9.30e-3') # x = 0.008 (i = 16)
        query[15] = data.query('X > 1.15e-2 and X <= 1.40e-2') # x = 0.013 (i = 15)
        query[14] = data.query('X > 1.65e-2 and X <= 1.90e-2') # x = 0.018 (i = 14)
        query[13] = data.query('X > 2.30e-2 and X <= 2.90e-2') # x = 0.025 (i = 13)
        query[12] = data.query('X > 2.90e-2 and X <= 3.80e-2') # x = 0.035 (i = 12)
        query[11] = data.query('X > 4.60e-2 and X <= 5.40e-2') # x = 0.050 (i = 11) 
        query[10] = data.query('X > 6.50e-2 and X <= 7.30e-2') # x = 0.070 (i = 10)
        query[9]  = data.query('X > 8.50e-2 and X <= 9.20e-2') # x = 0.090 (i = 9 )
        query[8]  = data.query('X > 10.7e-2 and X <= 11.3e-2') # x = 0.110 (i = 8 )
        query[7]  = data.query('X > 13.6e-2 and X <= 14.6e-2') # x = 0.140 (i = 7 )
        query[6]  = data.query('X > 14.6e-2 and X <= 18.7e-2') # x = 0.180 (i = 6 )
        query[5]  = data.query('X > 18.7e-2 and X <= 23.7e-2') # x = 0.230 (i = 5 )
        query[4]  = data.query('X > 23.7e-2 and X <= 29.0e-2') # x = 0.280 (i = 4 )
        query[3]  = data.query('X > 29.0e-2 and X <= 36.0e-2') # x = 0.350 (i = 3 )
        query[2]  = data.query('X > 36.0e-2 and X <= 47.6e-2') # x = 0.450 (i = 2 )
        query[1]  = data.query('X > 47.6e-2 and X <= 57.5e-2') # x = 0.550 (i = 1 )
        query[0]  = data.query('X > 57.5e-2 and X <= 70.5e-2') # x = 0.680 (i = 0 )

    if kind == 'HERA NC':
        query[24] = data.query('X < 3.50e-5')                 # x = 2.47e-05,2.928e-05,3.088e-05                                                            (i = 24)
        query[23] = data.query('X > 3.50e-5 and X <= 5.5e-5')  # x = 3.66e-05,4.06e-05,4.09e-05,4.323e-05,4.60e-05,5e-05,5.124e-05,5.22e-05,5.31e-05         (i = 23)
        query[22] = data.query('X > 5.50e-5 and X <= 9.5e-5')  # x = 5.92e-05,6.176e-05,6.83e-05,7.32e-05,7.54e-05,8.e-05,8.029e-05,8.18e-05,8.55e-05        (i = 22)
        query[21] = data.query('X > 9.50e-5 and X <= 1.5e-4')  # x = 9.515e-05,9.86e-05,1.0499e-04,1.118e-04,1.2443e-04,1.29e-04,1.30e-04,1.39e-04,1.392e-04 (i = 21)
        query[20] = data.query('X > 1.50e-4 and X <= 2.5e-4')  # x = 1.61e-04,1.741e-04,1.821e-04,2.0e-04,2.09e-04,2.276e-04,2.37e-04                        (i = 20)
        query[19] = data.query('X > 2.50e-4 and X <= 3.5e-4')  # x = 2.68e-04,2.9e-04,3.14e-04,3.2e-04,3.28e-04,3.45e-04                                     (i = 19)
        query[18] = data.query('X > 3.50e-4 and X <= 5.5e-4')  # x = 3.55e-04,3.88e-04,4.1e-04,4.603e-04,5e-04,5.31e-04                                      (i = 18)
        query[17] = data.query('X > 5.50e-4 and X <= 6.6e-4')  # x = 5.90e-04,5.92e-04,6.16e-04,6.341e-04,6.57e-04                                           (i = 17)
        query[16] = data.query('X > 7.50e-4 and X <= 8.5e-4')  # x = 7.0e-04,8.0e-04                                                                         (i = 16)
        query[15] = data.query('X > 8.50e-4 and X <= 0.98e-3') # x = 8.6e-04,9.2e-04,9.22e-04,9.4e-04                                                        (i = 15)
        query[14] = data.query('X > 0.98e-3 and X <= 1.9e-3')  # x = 0.001,0.0011,0.00124,0.0013,0.0015, 0.0016, 0.00172,0.00188                             (i = 14)
        query[13] = data.query('X > 0.0019  and X <= 0.0030')  # x = 0.002,0.00212,0.0025,0.0026,0.0027                                                      (i = 13)
        query[12] = data.query('X > 0.0030  and X <= 0.0040')  # x = 00.0032,0.033,0.0039                                                                    (i = 12)
        query[11] = data.query('X > 4.80e-3 and X <= 5.6e-3')  # x = 0.0050,0.053, 0.0066                                                                    (i = 11)
        query[10] = data.query('X > 7.90e-3 and X <= 8.6e-3')  # x = 0.0080,0.0085                                                                           (i = 10)
        query[9]  = data.query('X > 1.00e-2 and X <= 1.5e-2')  # x = 1.05,  0.0130, 0.014                                                                    (i = 9 )
        query[8]  = data.query('X > 1.90e-2 and X <= 2.2e-2')  # x = 0.020, 0.0219                                                                           (i = 8 )
        query[7]  = data.query('X > 3.00e-2 and X <= 3.4e-2')  # x = 0.032                                                                                   (i = 7 )
        query[6]  = data.query('X > 4.80e-2 and X <= 5.6e-2')  # x = 0.050, 0.0470                                                                           (i = 6 )
        query[5]  = data.query('X > 7.90e-2 and X <= 8.8e-2')  # x = 0.080, 0.0875                                                                           (i = 5 )
        query[4]  = data.query('X > 12.0e-2 and X <= 14e-2')   # x = 0.13                                                                                    (i = 4 )
        query[3]  = data.query('X > 17.0e-2 and X <= 19e-2')   # x = 0.18                                                                                    (i = 3 )
        query[2]  = data.query('X > 23.0e-2 and X <= 26e-2')   # x = 0.25                                                                                    (i = 2 )
        query[1]  = data.query('X > 38.0e-2 and X <= 42.0e-2') # x = 0.40                                                                                    (i = 1 )
        query[0]  = data.query('X > 62.0e-2 and X <= 66e-2')   # x = 0.65                                                                                    (i = 0 )

    if kind == 'HERA CC':
        query[7]  = data.query('X > 0.0079 and X <= 0.0081') # x = 0.008 (i = 7) 
        query[6]  = data.query('X > 0.010  and X <= 0.015')  # x = 0.013 (i = 6) 
        query[5]  = data.query('X > 0.030  and X <= 0.034')  # x = 0.032 (i = 5) 
        query[4]  = data.query('X > 0.079  and X <= 0.081')  # x = 0.080 (i = 4) 
        query[3]  = data.query('X > 0.120  and X <= 0.14')   # x = 0.13  (i = 3) 
        query[2]  = data.query('X > 0.230  and X <= 0.27')   # x = 0.25  (i = 2)
        query[1]  = data.query('X > 0.380  and X <= 0.42')   # x = 0.40  (i = 1)
        query[0]  = data.query('X > 0.630  and X <= 0.66')   # x = 0.65  (i = 0)

    if kind == 'HERA other':
        query[29] = data.query('X < 3.50e-5 ')                # x = 3.27e-05                                                                               (i = 29)
        query[28] = data.query('X > 3.50e-5 and X <= 9.5e-5')  # x = 4.09e-05,5e-05,5.73e-05,8e-05,8.18e-05                                                 (i = 28)
        query[27] = data.query('X > 9.50e-5 and X <= 2.5e-4')  # x = 9.86e-05,1.3e-04,1.39e-04,1.61e-04,2e-04,2.46e-04                                      (i = 27)
        query[26] = data.query('X > 2.50e-4 and X <= 3.5e-4')  # x = 2.68e-04,3.2e-04,3.28e-04,3.35e-04                                                     (i = 26)
        query[25] = data.query('X > 3.50e-4 and X <= 5.5e-4')  # x = 4.1e-04,5e-04                                                                          (i = 25)
        query[24] = data.query('X > 5.50e-4 and X <= 6.6e-4')  # x = 5.74e-04                                                                               (i = 24)
        query[23] = data.query('X > 6.90e-4 and X <= 8.5e-4')  # x = 8e-04                                                                                  (i = 23)
        query[22] = data.query('X > 8.50e-4 and X <= 0.98e-3') # x =  8.8e-04,9.1e-04,9.206e-04,9.344e-04,  9.545e-04                                       (i = 22)
        query[21] = data.query('X > 0.98e-3 and X <= 0.0014')  # x = 1.3e-03                                                                                (i = 21)
        query[20] = data.query('X > 1.3e-3  and X <= 1.9e-03') # 1 . 3660e-03,1.392e-03,1.397e-03,1.409-03,1.46e-03,1.479e-03,1.578e-03,1.585e-03,1.591e-03 (i = 20)
        query[19] = data.query('X > 0.0019  and X <= 0.0029')  # x = 2e-03,2.12e-03                                                                         (i = 19)
        query[18] = data.query('X > 0.0029  and X <= 0.0040')  # x = 3.2e-03                                                                                (i = 18)
        query[17] = data.query('X > 4.0e-3  and X <= 5.55e-3') # x = 5e-03                                                                                  (i = 17)
        query[16] = data.query('X > 5.55e-3 and X <= 7.95e-3') # x = 5.6e-03, 5.727e-03, 5.754e-03, 5.8e-03, 5.9e-03, 5.918e-03, 6e-03, 6.1e-03,6.2e-03,6.4e-03,6.6e-03,6.9e-03,7.3e-03,7.398e-03,7.4e-03,7.6e-03,7.9e-03                                                                                                                                (i = 16)
        query[15] = data.query('X > 7.95e-3 and X <= 8.9e-3')  # x = 8e-03,8.5e-03                                                                          (i = 15)
        query[14] = data.query('X > 9.00e-3 and X <= 9.9e-3')  # x = 9.1e-03,9.3e-03,9.864e-03                                                              (i = 14)
        query[13] = data.query('X > 9.90e-3 and X <= 1.25e-2') # x = 1e-02,1.04e-02,1.05e-02,1.09e-02,1.16e-02,1.21e-02                                     (i = 13)
        query[12] = data.query('X > 1.25e-2 and X <= 1.5e-2')  # x = 1.3e-02,1.4e-02                                                                        (i = 12)
        query[11] = data.query('X > 1.50e-2 and X <= 1.7e-2')  # x = 1.51e-02,1.52e-02,1.61e-02,1.660e-02                                                   (i = 11)
        query[10] = data.query('X > 1.70e-2 and X <= 1.99e-2') # x = 1.71e-02,1.85e-02,1.97e-02                                                             (i = 10)
        query[9] =  data.query('X > 1.99e-2 and X <= 2.2e-2')  # x = 2e-02                                                                                  (i = 9)
        query[8] =  data.query('X > 2.30e-2 and X <= 2.7e-2')  # x = 2.420e-02,2.61e-02                                                                     (i = 8)
        query[7] =  data.query('X > 3.00e-2 and X <= 3.4e-2')  # x = 3.20e-02                                                                               (i = 7)
        query[6] =  data.query('X > 4.80e-2 and X <= 5.6e-2')  # x = 0.050                                                                                  (i = 6)
        query[5] =  data.query('X > 7.90e-2 and X <= 8.8e-2')  # x = 0.080                                                                                  (i = 5)
        query[4] =  data.query('X > 12.0e-2 and X <= 14e-2')   # x = 0.13                                                                                   (i = 4)
        query[3] =  data.query('X > 17.0e-2 and X <= 19e-2')   # x = 0.18                                                                                   (i = 3)
        query[2] =  data.query('X > 23.0e-2 and X <= 26e-2')   # x = 0.25                                                                                   (i = 2)
        query[1] =  data.query('X > 30.0e-2 and X <= 45e-2')   # x = 0.4                                                                                    (i = 1)
        query[0] =  data.query('X > 60.0e-2 and X <= 70e-2')   # x = 0.65                                                                                   (i = 0)


    return query

def get_Q2bins(data,kind):
    query = {}
    if kind == 'BONuS':
        query[5]  = data.query('Q2 > 3.7')              # Q2 = 4.0 (i = 5)
        query[4]  = data.query('Q2 > 3.2 and Q2 <= 3.7') # Q2 = 3.4 (i = 4)
        query[3]  = data.query('Q2 > 2.7 and Q2 <= 3.2') # Q2 = 2.9 (i = 3)
        query[2]  = data.query('Q2 > 2.3 and Q2 <= 2.7') # Q2 = 2.4 (i = 2)
        query[1]  = data.query('Q2 > 1.9 and Q2 <= 2.3') # Q2 = 2.0 (i = 1)
        query[0]  = data.query('Q2 > 1.5 and Q2 <= 1.9') # Q2 = 1.7 (i = 0)
    if kind == 'MARATHON' or kind == 'JLab':
        query[0]  = data.query('Q2 > 1.5 and Q2 <= 20') # Q2 = 1.7 (i = 0)

    return query

def get_thetabins(data,kind):
    query = {}
    if kind == 'JLab':
        query[5]  = data.query('theta > 65 and theta <= 75') # theta = 70 (i = 5)
        query[4]  = data.query('theta > 57 and theta <= 65') # theta = 60 (i = 4)
        query[3]  = data.query('theta > 50 and theta <= 57') # theta = 55 (i = 3)
        query[2]  = data.query('theta > 43 and theta <= 50') # theta = 45 (i = 2)
        query[1]  = data.query('theta > 40 and theta <= 43') # theta = 41 (i = 1)
        query[0]  = data.query('theta > 35 and theta <= 40') # theta = 38 (i = 0)

    return query

def get_plot(query,cluster_i=0):

    #--generate dictionary with everything needed for plot

    plot = {_:{} for _ in ['theory','X','Q2','value','alpha','std']}
    for key in query:
        theory = query[key]['thy-%d' % cluster_i]
        std    = query[key]['dthy-%d' % cluster_i]
        X      = query[key]['X']
        Q2     = query[key]['Q2']
        value  = query[key]['value']
        alpha  = query[key]['alpha']
        #--sort by ascending Q2
        zx = sorted(zip(Q2,X))
        zt = sorted(zip(Q2,theory))
        zv = sorted(zip(Q2,value))
        za = sorted(zip(Q2,alpha))
        zs = sorted(zip(Q2,std))
        plot['X'][key]      = np.array([zx[i][1] for i in range(len(zx))])
        plot['theory'][key] = np.array([zt[i][1] for i in range(len(zt))])
        plot['value'][key]  = np.array([zv[i][1] for i in range(len(zv))])
        plot['alpha'][key]  = np.array([za[i][1] for i in range(len(za))])
        plot['std'][key]    = np.array([zs[i][1] for i in range(len(zs))])
        plot['Q2'][key]     = np.array(sorted(Q2))

    return plot

def get_theory(PLOT,nbins,loop=True,funcQ2=True,funcX=False):

    #--interpolate theoretical values across Q2 or X

    theory = {}

    if funcQ2: svar = 'Q2'
    if funcX:  svar = 'X'

    theory = {_:{} for _ in [svar,'value','std']}
    for key in range(nbins):
        var = []
        thy = []
        std = []
        
        #--if plotting for multiple experiments, loop over and combine
        if loop:
            for exp in PLOT:
                var. extend(PLOT[exp][svar][key])
                thy.extend(PLOT[exp]['theory'][key])
                std.extend(PLOT[exp]['std'][key])
        else:
            var.extend(PLOT[svar][key])
            thy.extend(PLOT['theory'][key])
            std.extend(PLOT['std'][key])

        #--if nothing in bin, skip
        if len(var) == 0: continue

        vmin = np.min(var)
        vmax = np.max(var)
        theory[svar][key]  = np.geomspace(vmin,vmax,100)

        #--if more than one value, interpolate between them
        if len(var) > 1:
            theory['value'][key] = griddata(np.array(var),np.array(thy),theory[svar][key],method='linear')
            theory['std'][key]   = griddata(np.array(var),np.array(std),theory[svar][key],method='linear')
        else:
            theory['value'][key] = np.ones(100)*thy 
            theory['std'][key]   = np.ones(100)*std


    return theory

def get_details(exp):

    #--get details for plotting

    if exp=='NMC' :          color, marker, ms = 'firebrick', '*', 8
    if exp=='SLAC' :         color, marker, ms = 'darkgreen', '^', 6
    if exp=='BCDMS':         color, marker, ms = 'blue'     , 'o', 6

    if exp=='HERA 10026':    color, marker, ms = 'black'    , '.', 8 
    if exp=='HERA 10030':    color, marker, ms = 'firebrick', 'D', 8 
    if exp=='HERA 10031':    color, marker, ms = 'blue'     , '^', 6 
    if exp=='HERA 10032':    color, marker, ms = 'darkgreen', 's', 6 

    if exp=='HERA 10027':    color, marker, ms = 'darkgreen', 's', 6 
    if exp=='HERA 10028':    color, marker, ms = 'firebrick', '.', 8 
    if exp=='HERA 10029':    color, marker, ms = 'blue',      '^', 8 

    if exp=='JLab d':        color, marker, ms = 'firebrick', '*', 8 
    if exp=='JLab p':        color, marker, ms = 'darkgreen', '^', 8 
    if exp=='BONuS' :        color, marker, ms = 'blue'     , 'o', 6 

    if exp=='MARATHON dp':   color, marker, ms = 'darkgreen', 'o', 6 
    if exp=='MARATHON ht':   color, marker, ms = 'darkgreen', '^', 5 
 
    if exp=='JLab hd':       color, marker, ms = 'firebrick', 'o', 5
    return color,marker,ms

def plot_proton(wdir, data, kc, istep):

    nrows, ncols = 2, 2
    py.figure(figsize = (ncols * 12.0, nrows * 14.0))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)
    ax21 = py.subplot(nrows, ncols, 3)
    ax22 = py.subplot(nrows, ncols, 4)

    #################################
    #--Plot F2p from BCDMS, SLAC, NMC
    #################################

    nbins = 27

    nmc   = data[10020]  ## NMC p
    slac  = data[10010]  ## SLAC p
    bcdms = data[10016]  ## BCDMS p

    DATA = {}
    DATA['NMC']   = pd.DataFrame(nmc)
    DATA['SLAC']  = pd.DataFrame(slac)
    DATA['BCDMS'] = pd.DataFrame(bcdms)

    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_xbins(DATA[exp],'p')
            PLOT[exp] = get_plot(query)

        theory = get_theory(PLOT,nbins)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            Q2    = PLOT[exp]['Q2'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax11.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

    #--plot theory interpolated between data points
    for key in theory['value']:
        Q2   = theory['Q2'][key]
        mean = theory['value'][key]*2.0**float(key)
        std  = theory['std'][key]  *2.0**float(key)
        down = mean - std
        up   = mean + std
        thy_plot ,= ax11.plot(Q2,mean,linestyle='solid',color='black')
        thy_band  = ax11.fill_between(Q2,down,up,color='gold',alpha=1.0)

    #--plot labels
    ax11.text(2.4,   3.0e7,  r'$x=0.005\, (i=24)$', fontsize = 22)
    ax11.text(3.7,   1.2e7,  r'$x=0.008$',          fontsize = 22)
    ax11.text(5.8,   6.0e6,  r'$x=0.013$',          fontsize = 22)
    ax11.text(7.5,   3.0e6,  r'$x=0.018$',          fontsize = 22)
    ax11.text(12.0,  1.5e6,  r'$x=0.026$',          fontsize = 22)
    ax11.text(15.0,  8.0e5,  r'$x=0.036$',          fontsize = 22)
    ax11.text(21.0,  4.0e5,  r'$x=0.05$',           fontsize = 22)
    ax11.text(28.0,  2.0e5,  r'$x=0.07$',           fontsize = 22)
    ax11.text(28.0,  1.0e5,  r'$x=0.09$',           fontsize = 22)
    ax11.text(41.0,  5.0e4,  r'$x=0.10$',           fontsize = 22)
    ax11.text(38.0,  2.5e4,  r'$x=0.11$',           fontsize = 22)
    ax11.text(62.0,  1.0e4,  r'$x=0.14$',           fontsize = 22)
    ax11.text(70.0,  5.0e3,  r'$x=0.18$',           fontsize = 22)
    ax11.text(93.0,  2.5e3,  r'$x=0.23$',           fontsize = 22)
    ax11.text(125.0, 1.1e3,  r'$x=0.28$',           fontsize = 22)
    ax11.text(150.0, 4.0e2,  r'$x=0.35$',           fontsize = 22)
    ax11.text(200.0, 1.0e2,  r'$x=0.43$',           fontsize = 22)
    ax11.text(65.0,  60.0,   r'$x=0.46$',           fontsize = 22)
    ax11.text(15.0,  30.0,   r'$x=0.50$',           fontsize = 22)
    ax11.text(250.0, 8.0,    r'$x=0.53$',           fontsize = 22)
    ax11.text(22.0,  5.0,    r'$x=0.56$',           fontsize = 22)
    ax11.text(23.0,  2.5,    r'$x=0.59$',           fontsize = 22)
    ax11.text(24.0,  1.0,    r'$x=0.61$',           fontsize = 22)
    ax11.text(250.0, 0.2,    r'$x=0.65$',           fontsize = 22)
    ax11.text(250.0, 0.03,   r'$x=0.75$',           fontsize = 22)
    ax11.text(30.0,  0.008,  r'$x=0.85$',           fontsize = 22)
    ax11.text(30.0 , 0.002,  r'$x=0.88\, (i=0)$',   fontsize = 22)

    ax11.semilogy()
    ax11.semilogx()
    ax11.set_xlim(1.5, 5e2)
    ax11.set_ylim(0.0005, 1e8)

    ax11.tick_params(axis = 'both', labelsize = 30)

    ax11.yaxis.set_tick_params(which = 'major', length = 10)
    ax11.set_yticks([0.001,0.01, 0.1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8])
    ax11.set_yticklabels([r'$10^{-3}$',r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$',r'$10^8$'])
    locmin = matplotlib.ticker.LogLocator(base = 10.0, subs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks = 12)
    ax11.yaxis.set_minor_locator(locmin)
    ax11.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax11.yaxis.set_tick_params(which = 'minor', length = 5)

    ax11.xaxis.set_tick_params(which = 'major', length = 10)
    ax11.xaxis.set_tick_params(which = 'minor', length = 5)
    ax11.set_xticks([2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2])
    xtick_labels = [2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2]
    ax11.set_xticklabels(['$%0.0f$' % x for x in xtick_labels])

    ax11.text(0.05,0.05,r'\boldmath$F_2^p$',      transform = ax11.transAxes, size = 60)
    ax11.text(0.15,0.05,r'$(\times\, 2^{\, i})$', transform = ax11.transAxes, size = 40)

    ax11.set_xlabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    ax11.yaxis.set_ticks_position('both')
    ax11.xaxis.set_ticks_position('both')
    ax11.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax11.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    handles = [hand['BCDMS'],hand['NMC'],hand['SLAC'],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{BCDMS}}'
    label2  = r'\textbf{\textrm{NMC}}'
    label3  = r'\textbf{\textrm{SLAC}}'
    label4  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2,label3,label4]
    ax11.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #################################
    #--Plot F2p from Hall C
    #################################

    jlab_p = data[10003] ## JLab p 

    E  = 5.5 #--Elab
    Mp = 0.939

    theta = lambda X,Q2,M: np.arcsin(np.sqrt(Q2/(4*E**2 - 2*E*Q2/M/X)))*2*180/np.pi

    data[10003]['theta'] = theta(data[10003]['X'],data[10003]['Q2'],Mp)

    nbins = 6 

    DATA = {}
    DATA['JLab p'] = pd.DataFrame(jlab_p) 

    theory = {}
    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_thetabins(DATA[exp],'JLab')
            PLOT[exp] = get_plot(query)

            theory[exp] = get_theory(PLOT[exp],nbins,loop=False)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            Q2    = PLOT[exp]['Q2'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax12.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            Q2   = theory[exp]['Q2'][key]
            mean = theory[exp]['value'][key]*2.0**float(key)
            std  = theory[exp]['std'][key]  *2.0**float(key)
            down = mean - std
            up   = mean + std
            thy_plot ,= ax12.plot(Q2,mean,linestyle='solid',color='black')
            thy_band  = ax12.fill_between(Q2,down,up,color='gold', alpha = 1.0)


    ax12.text(6.55,0.90,   r'$\theta=70^{\circ} \, (i=5)$',   fontsize = 30)
    ax12.text(6.2, 0.50,   r'$\theta=60^{\circ} $',           fontsize = 30)
    ax12.text(5.9, 0.28,   r'$\theta=55^{\circ} $',           fontsize = 30)
    ax12.text(5.2, 0.18,   r'$\theta=45^{\circ} $',           fontsize = 30)
    ax12.text(4.9, 0.10,   r'$\theta=41^{\circ} $',           fontsize = 30)
    ax12.text(4.6, 0.055,  r'$\theta=38^{\circ}  \, (i=0)$',  fontsize = 30)

    ax12.semilogy()

    ax12.set_xlim(3.5,   7.5)
    ax12.set_ylim(0.04,  3.5)

    ax12.set_xticks([4, 5, 6, 7])
    ax12.set_xticklabels([r'$4$', r'$5$', r'$6$', r'$7$'])
    lo12cmin = matplotlib.ticker.LogLocator(base = 10.0, subs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks = 12)
    ax12.yaxis.set_minor_locator(locmin)
    ax12.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax12.yaxis.set_tick_params(which = 'minor', length = 5)
    ax12.set_yticks([1e-1, 1e0])
    ax12.set_yticklabels([r'$10^{-1}$', r'$10^{0}$'])

    ax12.tick_params(axis = 'both', labelsize = 30)

    ax12.set_xlabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    ax12.text(0.05,  0.70,   r'\boldmath$F_2^p$',         transform = ax12.transAxes, size = 60)
    ax12.text(0.15,  0.70,   r'$(\, \times\, 2^{\, i})$', transform = ax12.transAxes, size = 40)

    ax12.xaxis.set_tick_params(which = 'major', length = 10)
    ax12.xaxis.set_tick_params(which = 'minor', length = 5)
    ax12.yaxis.set_tick_params(which = 'major', length = 10)
    ax12.yaxis.set_tick_params(which = 'minor', length = 5)

    ax12.yaxis.set_ticks_position('both')
    ax12.xaxis.set_ticks_position('both')
    ax12.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax12.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')

    minorLocator = MultipleLocator(0.2)
    ax12.xaxis.set_minor_locator(minorLocator)

    handles = [hand['JLab p']]
    label1  = r'\textbf{\textrm{Hall C}}'
    labels  = [label1,label2]
    ax12.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    #################################
    #--Plot NC cross-section from HERA at sqrt(s) = 318
    #################################

    hera_10026 = data[10026] ## HERA e+p \sqrt(s)=318 NC
    hera_10030 = data[10030] ## HERA e-p \sqrt(s)=318 NC

    nbins = 25 

    DATA = {}
    DATA['HERA 10026'] = pd.DataFrame(hera_10026) 
    DATA['HERA 10030'] = pd.DataFrame(hera_10030) 

    theory = {}
    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_xbins(DATA[exp],'HERA NC')
            PLOT[exp] = get_plot(query)

            theory[exp] = get_theory(PLOT[exp],nbins,loop=False)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            Q2    = PLOT[exp]['Q2'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax21.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            Q2   = theory[exp]['Q2'][key]
            mean = theory[exp]['value'][key]*2.0**float(key)
            std  = theory[exp]['std'][key]  *2.0**float(key)
            down = mean - std
            up   = mean + std
            thy_plot ,= ax21.plot(Q2,mean,linestyle='solid',color='black')
            thy_band  = ax21.fill_between(Q2,down,up,color='gold', alpha=1.0)


    ax21.text(3.4,1.65e7, r'$x=2.8\cdot 10^{-5}\, (i=24)$', fontsize = 22)
    ax21.text(5.5,8.65e6, r'$x=4.6\cdot 10^{-5}$',          fontsize = 22)
    ax21.text(8.0,4.6e6,  r'$x=7.3\cdot 10^{-5}$',          fontsize = 22)
    ax21.text(15.0,2.7e6, r'$x=1.2\cdot 10^{-4}$',          fontsize = 22)
    ax21.text(22.0,1.4e6, r'$x=1.9\cdot 10^{-4}$',          fontsize = 22)
    ax21.text(35.0,7.5e5, r'$x=3.1\cdot 10^{-4}$',          fontsize = 22)
    ax21.text(45.0,3.9e5, r'$x=4.4\cdot 10^{-4}$',          fontsize = 22)
    ax21.text(60.0,1.8e5, r'$x=6.2\cdot 10^{-4}$',          fontsize = 22)
    ax21.text(80.0,8.5e4, r'$x=7.7\cdot 10^{-4}$',          fontsize = 22)
    ax21.text(86.0,4e4,   r'$x=9.1 \cdot 10^{-4}$',         fontsize = 22)
    ax21.text(160.0,2e4,  r'$x=0.0014$',                    fontsize = 22)
    ax21.text(260.0,1e4,  r'$x=0.0024$',                    fontsize = 22)
    ax21.text(440.0,4500, r'$x=0.0035$',                    fontsize = 22)
    ax21.text(740.0,2000, r'$x=0.0056$',                    fontsize = 22)
    ax21.text(1.5e3,970,  r'$x=0.0083$',                    fontsize = 22)
    ax21.text(1.7e3,410,  r'$x=0.013$',                     fontsize = 22)
    ax21.text(2.7e3,190,  r'$x=0.021$',                     fontsize = 22)
    ax21.text(3.9e3,77,   r'$x=0.032$',                     fontsize = 22)
    ax21.text(6.5e3,35,   r'$x=0.052$',                     fontsize = 22)
    ax21.text(1.15e4,19,  r'$x=0.084$',                     fontsize = 22)
    ax21.text(12.0,5.5,   r'$x=0.13$',                      fontsize = 22)
    ax21.text(18.0,2.25,  r'$x=0.18$',                      fontsize = 22)
    ax21.text(36.0,0.97,  r'$x=0.25$',                      fontsize = 22)
    ax21.text(43.0,0.25,  r'$x=0.40$',                      fontsize = 22)
    ax21.text(30.0,0.02,  r'$x=0.65 \, (i=0)$',             fontsize = 22)

    ax21.semilogy()
    ax21.semilogx()

    ax21.set_xlim(1.0,   2e5)
    ax21.set_ylim(0.004, 4e7)

    ax21.set_xticks([1.0, 10.0, 1e2, 1e3, 1e4, 1e5])
    ax21.set_xticklabels([r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    locmin = matplotlib.ticker.LogLocator(base = 10.0, subs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks = 12)
    ax21.yaxis.set_minor_locator(locmin)
    ax21.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax21.yaxis.set_tick_params(which = 'minor', length = 5)
    ax21.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
    ax21.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$'])

    ax21.tick_params(axis = 'both', labelsize = 30)

    ax21.set_xlabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    #ax21.text(1.2, 5e-2,  r'\boldmath$\sigma_r^{p,NC}$', size = 60)
    #ax21.text(13.0,7e-2, r'$(\, \times\, 2^{\, i})$',   size = 40)
    ax21.text(0.05, 0.10,  r'\boldmath$\sigma_r^{p \textrm{(NC)}}$', transform = ax21.transAxes, size = 60)
    ax21.text(0.05, 0.04, r'$(\, \times\, 2^{\, i})$',               transform = ax21.transAxes, size = 40)

    ax21.text(2.0e3, 3.0e4, r'$\sqrt{s}=318 \,\rm{GeV}$', fontsize = 40)

    ax21.xaxis.set_tick_params(which = 'major', length = 10)
    ax21.xaxis.set_tick_params(which = 'minor', length = 5)
    ax21.yaxis.set_tick_params(which = 'major', length = 10)
    ax21.yaxis.set_tick_params(which = 'minor', length = 5)

    ax21.yaxis.set_ticks_position('both')
    ax21.xaxis.set_ticks_position('both')
    ax21.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax21.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')


    handles = [hand['HERA 10026'],hand['HERA 10030']]
    label1  = r'\textbf{\textrm{HERA NC}} \boldmath$e^+p$'
    label2  = r'\textbf{\textrm{HERA NC}} \boldmath$e^-p$'
    labels  = [label1,label2]
    ax21.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    #################################
    #--Plot NC cross-section from HERA at sqrt(s) other than 318
    #################################

    hera_10027 = data[10027] ## HERA e+p \sqrt(s)=300 NC
    hera_10028 = data[10028] ## HERA e+p \sqrt(s)=251 NC
    hera_10029 = data[10029] ## HERA e+p \sqrt(s)=225 NC

    nbins = 30

    DATA = {}
    DATA['HERA 10027'] = pd.DataFrame(hera_10027) 
    DATA['HERA 10028'] = pd.DataFrame(hera_10028) 
    DATA['HERA 10029'] = pd.DataFrame(hera_10029)
 
    theory = {}
    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_xbins(DATA[exp],'HERA other')
            PLOT[exp] = get_plot(query)

            theory[exp] = get_theory(PLOT[exp],nbins,loop=False)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            Q2    = PLOT[exp]['Q2'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax22.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            Q2   = theory[exp]['Q2'][key]
            mean = theory[exp]['value'][key]*2.0**float(key)
            std  = theory[exp]['std'][key]  *2.0**float(key)
            down = mean - std
            up   = mean + std
            thy_plot ,= ax22.plot(Q2,mean,linestyle='solid',color=color)
            thy_band  = ax22.fill_between(Q2,down,up,color='gold', alpha = 1.0)

    ax22.text(2.5,    4.5e8,  r'$x=3.3 \cdot 10^{-5}\, (i=29)$', fontsize = 20)
    ax22.text(5.7,    2.5e8,  r'$x=5.9 \cdot 10^{-5}$',          fontsize = 20)
    ax22.text(19.0,   1.7e8,  r'$x=1.6 \cdot 10^{-4}$',          fontsize = 20)
    ax22.text(35.0,   8e7,    r'$x=3.1 \cdot 10^{-4}$',          fontsize = 20)
    ax22.text(35.0,   4.4e7,  r'$x=4.5 \cdot 10^{-4}$',          fontsize = 20)
    ax22.text(45.0,   2.1e7,  r'$x=6.2 \cdot 10^{-4}$',          fontsize = 20)
    ax22.text(57.0,   1.1e7,  r'$x=7.8 \cdot 10^{-4}$',          fontsize = 20)
    ax22.text(0.5e2,  5e6,    r'$x=9.1 \cdot 10^{-4}$',          fontsize = 20)
    ax22.text(0.9e2,  2.7e6,  r'$x=0.0011$',                     fontsize = 20)
    ax22.text(1e2,    1.15e6, r'$x=0.0015$',                     fontsize = 20)
    ax22.text(1.7e2,  6e5,    r'$x=0.0022$',                     fontsize = 20)
    ax22.text(2.5e2,  2.9e5,  r'$x=0.0034$',                     fontsize = 20)
    ax22.text(3.7e2,  1.2e5,  r'$x=0.0046$',                     fontsize = 20)
    ax22.text(0.45e3, 5.6e4,  r'$x=0.0064$',                     fontsize = 20)
    ax22.text(0.8e3,  2.8e4,  r'$x=0.0085$',                     fontsize = 20)
    ax22.text(0.6e3,  1.2e4,  r'$x=0.0094$',                     fontsize = 20)
    ax22.text(0.8e3,  6200.0, r'$x=0.011$',                      fontsize = 20)
    ax22.text(1.5e3,  3200.0, r'$x=0.0138$',                     fontsize = 20)
    ax22.text(1e3,    1400.0, r'$x=0.0156$',                     fontsize = 20)
    ax22.text(1e3,    700.0,  r'$x=0.0183$',                     fontsize = 20)
    ax22.text(1.9e3,  320.0,  r'$x=0.020$',                      fontsize = 20)
    ax22.text(1.1e3,  150.0,  r'$x=0.025$',                      fontsize = 20)
    ax22.text(2.5e3,  67.0,   r'$x=0.032$',                      fontsize = 20)
    ax22.text(60.0,   28.0,   r'$x=0.05$',                       fontsize = 20)
    ax22.text(60.0,   12.0,   r'$x=0.08$',                       fontsize = 20)
    ax22.text(60.0,   5.3,    r'$x=0.13$',                       fontsize = 20)
    ax22.text(60.0,   2.15,   r'$x=0.18$',                       fontsize = 20)
    ax22.text(160.0,  0.85,   r'$x=0.25$',                       fontsize = 20)
    ax22.text(60.0,   0.3,    r'$x=0.40$',                       fontsize = 20)
    ax22.text(70.0,   0.022,  r'$x=0.65 \, (i=0)$',              fontsize = 20)

    ax22.text(4e3,   28.0, r'$x=0.05$',                          fontsize = 20, color = 'darkgreen')
    ax22.text(6e3,   11.0, r'$x=0.08$',                          fontsize = 20, color = 'darkgreen')
    ax22.text(1.6e3, 4.6,  r'$x=0.13$',                          fontsize = 20, color = 'darkgreen')
    ax22.text(2.3e3, 2.0,  r'$x=0.18$',                          fontsize = 20, color = 'darkgreen')
    ax22.text(2.8e3, 0.8,  r'$x=0.25$',                          fontsize = 20, color = 'darkgreen')
    ax22.text(3.0e3, 0.2,  r'$x=0.40\, (i=1)$',                  fontsize = 20, color = 'darkgreen')

    ax22.semilogy()
    ax22.semilogx()
    ax22.set_xlim(1.0,  4e4)
    ax22.set_ylim(1e-2, 2e9)

    ax22.set_xlabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    #ax22.text(1.50 , 3.0e-2, r'\boldmath$\sigma_r^{p,NC}$', size = 60)
    #ax22.text(13.0 , 5.0e-2, r'$(\, \times\, 2^{\, i})$',   size = 40)
    ax22.text(0.05, 0.10,  r'\boldmath$\sigma_r^{p \textrm{(NC)}}$', transform = ax22.transAxes, size = 60)
    ax22.text(0.05, 0.04, r'$(\, \times\, 2^{\, i})$',               transform = ax22.transAxes, size = 40)

    ax22.set_xticks([1.0, 10.0, 1e2, 1e3, 1e4])
    ax22.set_xticklabels([r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$'])
    locmin = matplotlib.ticker.LogLocator(base = 10.0, subs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks = 12)
    ax22.yaxis.set_minor_locator(locmin)
    ax22.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax22.yaxis.set_tick_params(which = 'minor', length = 3)
    ax22.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9])
    ax22.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$'])

    ax22.yaxis.set_ticks_position('both')
    ax22.xaxis.set_ticks_position('both')
    ax22.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax22.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')

    ax22.xaxis.set_tick_params(which = 'major', length = 10)
    ax22.xaxis.set_tick_params(which = 'minor', length = 5)
    ax22.yaxis.set_tick_params(which = 'major', length = 10)
    ax22.yaxis.set_tick_params(which = 'minor', length = 5)

    ax22.yaxis.set_ticks_position('both')
    ax22.xaxis.set_ticks_position('both')
    ax22.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax22.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')

    ax22.tick_params(axis='both',labelsize=30)

    handles = [hand['HERA 10027'],hand['HERA 10028'],hand['HERA 10029']]
    label1  = r'\textbf{\textrm{HERA}}' + r'$\sqrt{s}=300$'# + ' ' +  r'\textrm{GeV}'
    label2  = r'\textbf{\textrm{HERA}}' + r'$\sqrt{s}=251$'# + ' ' +  r'\textrm{GeV}'
    label3  = r'\textbf{\textrm{HERA}}' + r'$\sqrt{s}=225$'# + ' ' +  r'\textrm{GeV}'
    labels  = [label1,label2,label3]
    ax22.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    py.tight_layout()
    filename = '%s/gallery/DIS-proton'%cwd
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    py.savefig(filename)
    print('Saving figure to %s'%(filename))
    py.close()

def plot_deuteron(wdir, data, kc, istep):

    nrows, ncols = 2, 2
    py.figure(figsize = (ncols * 12, nrows * 14))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)
    ax21 = py.subplot(nrows, ncols, 3)
    ax22 = py.subplot(nrows, ncols, 4)

    #################################
    #--Plot F2d from BCDMS and SLAC
    #################################

    nbins = 19

    bcdms = data[10017] ## BCDMS d
    slac  = data[10011] ## SLAC d

    DATA = {}
    DATA['SLAC']  = pd.DataFrame(slac)
    DATA['BCDMS'] = pd.DataFrame(bcdms)

    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_xbins(DATA[exp],'d')
            PLOT[exp] = get_plot(query)

        theory = get_theory(PLOT,nbins)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            Q2    = PLOT[exp]['Q2'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax11.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

    #--plot theory interpolated between data points
    for key in theory['value']:
        Q2   = theory['Q2'][key]
        mean = theory['value'][key]*2.0**float(key)
        std  = theory['std'][key]  *2.0**float(key)
        down = mean - std
        up   = mean + std
        thy_plot ,= ax11.plot(Q2,mean,linestyle='solid',color='black')
        thy_band  = ax11.fill_between(Q2,down,up,color='gold',alpha=1.0)


    ax11.text(21.0,  9.0e4,  r'$x=0.07 \,(i=15)$', fontsize = 26)
    ax11.text(3.0,   4.0e4,  r'$x=0.09$',          fontsize = 26)
    ax11.text(45.0,  2.0e4,  r'$x=0.10$',          fontsize = 26)
    ax11.text(70.0,  1.0e4,  r'$x=0.14$',          fontsize = 26)
    ax11.text(75.0,  4.5e3,  r'$x=0.18$',          fontsize = 26)
    ax11.text(100.0, 2.0e3,  r'$x=0.23$',          fontsize = 26)
    ax11.text(140.0, 9.0e2,  r'$x=0.28$',          fontsize = 26)
    ax11.text(170.0, 3.0e2,  r'$x=0.35$',          fontsize = 26)
    ax11.text(190.0, 1.0e2,  r'$x=0.43$',          fontsize = 26)
    ax11.text(19.0,  58.0,   r'$x=0.46$',          fontsize = 26)
    ax11.text(17.0,  20.0,   r'$x=0.50$',          fontsize = 26)
    ax11.text(255.0, 4.00,   r'$x=0.53$',          fontsize = 26)
    ax11.text(22.0,  3.0,    r'$x=0.56$',          fontsize = 26)
    ax11.text(23.0,  1.50,   r'$x=0.59$',          fontsize = 26)
    ax11.text(25.0,  0.60,   r'$x=0.61$',          fontsize = 26)
    ax11.text(250.0, 0.14,   r'$x=0.65$',          fontsize = 26)
    ax11.text(250.0, 0.020,  r'$x=0.75$',          fontsize = 26)
    ax11.text(35.0,  0.005,  r'$x=0.85$',          fontsize = 26)
    ax11.text(35.0,  0.0015, r'$x=0.88 \,(i=0)$',  fontsize = 26)

    ax11.semilogy()
    ax11.semilogx()

    ax11.set_xlim(1.5, 1e3)
    ax11.set_ylim(0.001, 2e5)

    ax11.tick_params(axis = 'both', labelsize = 30)

    ax11.yaxis.set_tick_params(which = 'major', length = 10)
    ax11.yaxis.set_tick_params(which = 'minir', length = 5)
    ax11.set_yticks([0.01, 0.1, 1.0, 10.0, 1e2, 1e3, 1e4,1e5])
    ax11.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$',r'$10^5$'])
    locmin = matplotlib.ticker.LogLocator(base = 10.0, subs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks = 12)
    ax11.yaxis.set_minor_locator(locmin)
    ax11.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax11.yaxis.set_tick_params(which = 'minor', length = 5)

    ax11.xaxis.set_tick_params(which = 'major', length = 10)
    ax11.xaxis.set_tick_params(which = 'minor', length = 5)
    ax11.set_xticks([2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2])
    xticklabels=[2e0, 5e0, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2]
    ax11.set_xticklabels(['$%0.0f$'%x for x in xticklabels])

    ax11.set_xlabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    ax11.text(0.05,0.05,r'\boldmath$F_2^D$',      transform = ax11.transAxes, size = 60)
    ax11.text(0.15,0.05,r'$(\times\, 2^{\, i})$', transform = ax11.transAxes, size = 40)

    ax11.yaxis.set_ticks_position('both')
    ax11.xaxis.set_ticks_position('both')
    ax11.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax11.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')

    handles = [hand['BCDMS'],hand['SLAC']]
    label1  = r'\textbf{\textrm{BCDMS}}'
    label2  = r'\textbf{\textrm{SLAC}}'
    labels  = [label1,label2]
    ax11.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #################################
    #--Plot F2d from Hall C
    #################################

    jlab_d = data[10002] ## JLab d 


    E  = 5.5 #--Elab
    Mp = 0.939

    theta = lambda X,Q2,M: np.arcsin(np.sqrt(Q2/(4*E**2 - 2*E*Q2/M/X)))*2*180/np.pi

    data[10002]['theta'] = theta(data[10002]['X'],data[10002]['Q2'],Mp)

    nbins = 6 

    DATA = {}
    DATA['JLab d'] = pd.DataFrame(jlab_d) 

    theory = {}
    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_thetabins(DATA[exp],'JLab')
            PLOT[exp] = get_plot(query)

            theory[exp] = get_theory(PLOT[exp],nbins,loop=False)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            Q2    = PLOT[exp]['Q2'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax12.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            Q2   = theory[exp]['Q2'][key]
            mean = theory[exp]['value'][key]*2.0**float(key)
            std  = theory[exp]['std'][key]  *2.0**float(key)
            down = mean - std
            up   = mean + std
            thy_plot ,= ax12.plot(Q2,mean,linestyle='solid',color='black')
            thy_band  = ax12.fill_between(Q2,down,up,color='gold', alpha = 1.0)


    ax12.text(6.55,0.80,   r'$\theta=70^{\circ} \, (i=5)$',   fontsize = 30)
    ax12.text(6.2, 0.45,   r'$\theta=60^{\circ} $',           fontsize = 30)
    ax12.text(5.9, 0.24,   r'$\theta=55^{\circ} $',           fontsize = 30)
    ax12.text(5.2, 0.16,   r'$\theta=45^{\circ} $',           fontsize = 30)
    ax12.text(4.9, 0.090,  r'$\theta=41^{\circ} $',           fontsize = 30)
    ax12.text(4.6, 0.050,  r'$\theta=38^{\circ}  \, (i=0)$',  fontsize = 30)

    ax12.semilogy()

    ax12.set_xlim(3.5,   7.5)
    ax12.set_ylim(0.04,  3.5)

    ax12.set_xticks([4, 5, 6, 7])
    ax12.set_xticklabels([r'$4$', r'$5$', r'$6$', r'$7$'])
    locmin = matplotlib.ticker.LogLocator(base = 10.0, subs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks = 12)
    ax12.yaxis.set_minor_locator(locmin)
    ax12.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax12.yaxis.set_tick_params(which = 'minor', length = 5)
    ax12.set_yticks([1e-1, 1e0])
    ax12.set_yticklabels([r'$10^{-1}$', r'$10^{0}$'])

    ax12.tick_params(axis = 'both', labelsize = 30)

    ax12.set_xlabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    ax12.text(0.05,  0.70,   r'\boldmath$F_2^D$',         transform = ax12.transAxes, size = 60)
    ax12.text(0.15,  0.70,   r'$(\, \times\, 2^{\, i})$', transform = ax12.transAxes, size = 40)

    ax12.xaxis.set_tick_params(which = 'major', length = 10)
    ax12.xaxis.set_tick_params(which = 'minor', length = 5)
    ax12.yaxis.set_tick_params(which = 'major', length = 10)
    ax12.yaxis.set_tick_params(which = 'minor', length = 5)

    ax12.yaxis.set_ticks_position('both')
    ax12.xaxis.set_ticks_position('both')
    ax12.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax12.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')

    minorLocator = MultipleLocator(0.2)
    ax12.xaxis.set_minor_locator(minorLocator)

    handles = [hand['JLab d'],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{Hall C}}'
    label2  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2]
    ax12.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    #################################
    #--Plot F2d/F2p from NMC and Marathon
    #################################

    nbins = 18
    nmc   = data[10021] ## NMC d/p
    mar   = data[10050] ## MARATHON d/p

    DATA = {}
    DATA['NMC']           = pd.DataFrame(nmc)
    DATA['MARATHON dp']   = pd.DataFrame(mar)

    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_xbins(DATA[exp],'d/p')
            PLOT[exp] = get_plot(query)

        theory = get_theory(PLOT,nbins)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            Q2    = PLOT[exp]['Q2'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax21.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

    #--plot theory interpolated between data points
    for key in theory['value']:
        Q2   = theory['Q2'][key]
        mean = theory['value'][key]*2.0**float(key)
        std  = theory['std'][key]  *2.0**float(key)
        down = mean - std
        up   = mean + std
        thy_plot ,= ax21.plot(Q2,mean,linestyle='solid',color='black')
        thy_band  = ax21.fill_between(Q2,down,up,color='gold',alpha=1.0)

    ax21.text(2.0,  1.2e5,  r'$x=0.005 \, (i=17)$', fontsize = 26)
    ax21.text(4.0,  0.63e5, r'$x=0.008$',           fontsize = 26)
    ax21.text(6.3,  3.1e4,  r'$x=0.013$',           fontsize = 26)
    ax21.text(8.2,  1.5e4,  r'$x=0.018$',           fontsize = 26)
    ax21.text(10.5, 7.8e3,  r'$x=0.025$',           fontsize = 26)
    ax21.text(16.5, 3.6e3,  r'$x=0.035$',           fontsize = 26)
    ax21.text(22.0, 1.9e3,  r'$x=0.05$',            fontsize = 26)
    ax21.text(30.0, 0.9e3,  r'$x=0.07$',            fontsize = 26)
    ax21.text(40,   4.7e2,  r'$x=0.09$',            fontsize = 26)
    ax21.text(52.0, 2.2e2,  r'$x=0.11$',            fontsize = 26)
    ax21.text(56.0, 1.1e2,  r'$x=0.14$',            fontsize = 26)
    ax21.text(74.0, 5.4e1,  r'$x=0.18$',            fontsize = 26)
    ax21.text(75.0, 2.5e1,  r'$x=0.23$',            fontsize = 26)
    ax21.text(108,  12.2,   r'$x=0.28$',            fontsize = 26)
    ax21.text(110,  6.0,    r'$x=0.35$',            fontsize = 26)
    ax21.text(110,  2.9,    r'$x=0.45$',            fontsize = 26)
    ax21.text(110,  1.35,   r'$x=0.55$',            fontsize = 26)
    ax21.text(110,  0.7,    r'$x=0.68 \,(i=0)$',    fontsize = 26)

    ax21.semilogy()
    ax21.semilogx()

    ax21.set_xlim(1.5, 3.5e2)
    ax21.set_ylim(0.5, 2e5)

    ax21.tick_params(axis = 'both', labelsize = 30)

    ax21.yaxis.set_tick_params(which = 'major', length = 10)
    ax21.yaxis.set_tick_params(which = 'minir', length = 5)
    ax21.set_yticks([1.0, 10.0, 1e2, 1e3, 1e4, 1e5])
    ax21.set_yticklabels([r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])
    locmin = matplotlib.ticker.LogLocator(base = 10.0, subs = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks = 12)
    ax21.yaxis.set_minor_locator(locmin)
    ax21.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax21.yaxis.set_tick_params(which = 'minor', length = 5)

    ax21.xaxis.set_tick_params(which = 'major', length = 10)
    ax21.xaxis.set_tick_params(which = 'minor', length = 5)
    ax21.set_xticks([2e0, 5e0, 1e1, 2e1, 5e1, 1e2])
    xticklabels=[2e0, 5e0, 1e1, 2e1, 5e1, 1e2]
    ax21.set_xticklabels(['$%0.0f$'%x for x in xticklabels])

    ax21.set_xlabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    ax21.text(0.60, 0.75,  r'\boldmath$F_2^D/F_2^p$',  transform = ax21.transAxes, size = 60)
    ax21.text(0.85, 0.75,  r'$(\, \times\, 2^{\, i})$',transform = ax21.transAxes, size = 40)

    ax21.yaxis.set_ticks_position('both')
    ax21.xaxis.set_ticks_position('both')
    ax21.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax21.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')

    handles = [hand['NMC'],hand['MARATHON dp']]
    label1  = r'\textbf{\textrm{NMC}}'
    label2  = r'\textbf{\textrm{MARATHON}}'
    labels  = [label1,label2]
    ax21.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #################################
    #--Plot F2n/F2d from BONuS
    #################################

    bonus  = data[10033] ## BONuS n/d 

    nbins = 6 

    DATA = {}
    DATA['BONuS'] = pd.DataFrame(bonus) 

    PLOT   = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'BONuS')
            PLOT[exp] = get_plot(query)

            theory[exp] = get_theory(PLOT[exp],nbins,loop=False,funcX=True)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key]*4.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*4.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax22.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key]*4.0**float(key)
            std  = theory[exp]['std'][key]  *4.0**float(key)
            down = mean - std
            up   = mean + std
            thy_plot ,= ax22.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax22.fill_between(X,down,up,color='gold', alpha = 1.0)

    ax22.text(0.47, 0.05,   r'$Q^2 = 4.0 $'+'$\, (i=5)$',               transform = ax22.transAxes, fontsize = 30)
    ax22.text(0.55, 0.20,   r'$Q^2 = 3.4 $',                            transform = ax22.transAxes, fontsize = 30)
    ax22.text(0.62, 0.35,   r'$Q^2 = 2.9 $',                            transform = ax22.transAxes, fontsize = 30)
    ax22.text(0.74, 0.50,   r'$Q^2 = 2.4 $',                            transform = ax22.transAxes, fontsize = 30)
    ax22.text(0.78, 0.64,   r'$Q^2 = 2.0 $',                            transform = ax22.transAxes, fontsize = 30)
    ax22.text(0.30, 0.80,   r'$Q^2 = 1.7 \: \rm{GeV^2}$'+'$\, (i=0)$',  transform = ax22.transAxes, fontsize = 30)

    ax22.semilogy()

    ax22.set_xlim(0.2,   0.75)
    ax22.set_ylim(0.5,   4e3)

    ax22.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax22.set_xticklabels([r'$0.2$', r'$0.3$', r'$0.4$', r'$0.5$',r'$0.6$',r'$0.7$'])
    ax22.set_yticks([1e0, 1e1, 1e2, 1e3])
    ax22.set_yticklabels([r'$1$', r'$10$', r'$10^2$', r'$10^3$'])

    ax22.tick_params(axis = 'both', labelsize = 30)

    ax22.set_xlabel(r'\boldmath$x$',size=40)

    ax22.text(0.05, 0.90,   r'\boldmath$F_2^n/F_2^D$',   transform = ax22.transAxes, size = 60)
    ax22.text(0.30, 0.90,   r'$(\, \times\, 4^{\, i})$', transform = ax22.transAxes, size = 40)

    ax22.xaxis.set_tick_params(which = 'major', length = 10)
    ax22.xaxis.set_tick_params(which = 'minor', length = 5)
    ax22.yaxis.set_tick_params(which = 'major', length = 10)
    ax22.yaxis.set_tick_params(which = 'minor', length = 5)

    ax22.yaxis.set_ticks_position('both')
    ax22.xaxis.set_ticks_position('both')
    ax22.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax22.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')

    minorLocator = MultipleLocator(0.02)
    ax22.xaxis.set_minor_locator(minorLocator)

    handles = [hand['BONuS']]
    label1  = r'\textbf{\textrm{BONuS}}'
    labels  = [label1]
    #ax22.legend(handles,labels,loc=(0.02,0.6),  fontsize = 35, frameon = 0, handletextpad = 0.1, handlelength = 1.0)
    ax22.legend(handles,labels,loc='upper right',  fontsize = 35, frameon = 0, handletextpad = 0.1, handlelength = 1.0)

    py.tight_layout()
    filename = '%s/gallery/DIS-deuteron'%cwd
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()

def plot_CC(wdir, data, kc, istep):

    nrows, ncols = 1, 1
    fig = py.figure(figsize = (ncols * 12, nrows * 14))
    ax = {}
    ax = fig.add_subplot(nrows, ncols, 1)

    hera_10031 = data[10031] ## HERA e+p \sqrt(s)=318 CC
    hera_10032 = data[10032] ## HERA e-p \sqrt(s)=318 CC

    nbins = 8 

    DATA = {}
    DATA['HERA 10031'] = pd.DataFrame(hera_10031) 
    DATA['HERA 10032'] = pd.DataFrame(hera_10032) 

    PLOT   = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_xbins(DATA[exp],'HERA CC')
            PLOT[exp] = get_plot(query)

            theory[exp] = get_theory(PLOT[exp],nbins,loop=False)

    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            Q2    = PLOT[exp]['Q2'][key]
            val   = PLOT[exp]['value'][key]*5.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*5.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax.errorbar(Q2,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            Q2   = theory[exp]['Q2'][key]
            mean = theory[exp]['value'][key]*5.0**float(key)
            std  = theory[exp]['std'][key]  *5.0**float(key)
            down = mean - std
            up   = mean + std
            thy_plot ,= ax.plot(Q2,mean,linestyle='solid',color='black')
            thy_band  = ax.fill_between(Q2,down,up,color='gold', alpha=1.0)


    ax.text(370.0,  1.4e5, r'$x=0.008 \, (i=7)$', fontsize = 28, color = 'darkgreen')
    ax.text(1200.0, 1.3e4, r'$x=0.013$',          fontsize = 28, color = 'darkgreen')
    ax.text(3700.0, 2e3,   r'$x=0.032$',          fontsize = 28, color = 'darkgreen')
    ax.text(6000.0, 400,   r'$x=0.08$',           fontsize = 28, color = 'darkgreen')
    ax.text(9800.0, 70,    r'$x=0.13$',           fontsize = 28, color = 'darkgreen')
    ax.text(400.0,  11.5,  r'$x=0.25$',           fontsize = 28, color = 'darkgreen')
    ax.text(600.0,  1.4,   r'$x=0.40$',           fontsize = 28, color = 'darkgreen')
    ax.text(1300.0, 0.035, r'$x=0.65 \, (i=0) $', fontsize = 28, color = 'darkgreen')

    ax.text(550.0,  0.6e5, r'$x=0.008 \, (i=7)$', fontsize = 28, color = 'blue')
    ax.text(1200.0, 0.8e4, r'$x=0.013$',          fontsize = 28, color = 'blue')
    ax.text(3700.0, 1000,  r'$x=0.032$',          fontsize = 28, color = 'blue')
    ax.text(4300.0, 180,   r'$x=0.08$',           fontsize = 28, color = 'blue')
    ax.text(0.8e4,  23,    r'$x=0.13$',           fontsize = 28, color = 'blue')
    ax.text(400.0,  5.2,   r'$x=0.25$',           fontsize = 28, color = 'blue')
    ax.text(370.0,  0.3,   r'$x=0.40 \, (i=1)$',  fontsize = 28, color = 'blue')

    ax.semilogy()
    ax.semilogx()

    ax.set_xlim(1.5e2,   4e4)
    ax.set_ylim(1.0e-2,  8e5)

    ax.set_xticks([2e2, 1e3, 1e4, 4e4])
    ax.set_xticklabels([r'$200$', r'$10^3$', r'$10^4$', r'$4\cdot10^4$'])
    ax.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_yticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$1$', r'$10$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])

    ax.tick_params(axis = 'both', labelsize = 30)

    ax.set_xlabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    #ax.text(1.8e2, 2.0e-2, r'\boldmath$\sigma_r^{p,CC}$', size = 60)
    #ax.text(5.6e2, 2.6e-2,  r'$(\, \times\, 5^{\, i})$',   size = 40)
    ax.text(0.05, 0.08,  r'\boldmath$\sigma_r^{p \textrm{(CC)}}$',transform = ax.transAxes, size = 60)
    ax.text(0.05, 0.03,  r'$(\, \times\, 5^{\, i})$'             ,transform = ax.transAxes, size = 40)

    ax.xaxis.set_tick_params(which = 'major', length = 10)
    ax.xaxis.set_tick_params(which = 'minor', length = 5)
    ax.yaxis.set_tick_params(which = 'major', length = 10)
    ax.yaxis.set_tick_params(which = 'minor', length = 5)

    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction = 'in')
    ax.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction = 'in')

    #ax.text(5.0e3,  2.0e4, r'$\sqrt{s}=318 \,\rm{GeV}$', fontsize = 40)

    handles = [hand['HERA 10031'],hand['HERA 10032'],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{HERA CC}} \boldmath$e^+p$'
    label2  = r'\textbf{\textrm{HERA CC}} \boldmath$e^-p$'
    label3  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2,label3]
    ax.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)
    py.tight_layout()

    filename = '%s/gallery/DIS-CC'%cwd
    filename += '.png'
    #filename += '.pdf'
    ax.set_rasterized(True)
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()

def plot_A3(wdir, data, kc, istep):

    nrows, ncols = 1, 2
    py.figure(figsize = (ncols * 5, nrows * 6))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)

    #################################
    #--Plot F2h/F2d from JLab Hall C
    #################################

    hd   = data[10041] ## JLab Hall C h/d

    DATA = {}
    DATA['JLab hd']  = pd.DataFrame(hd)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'JLab')
            nbins = len(query)
            PLOT[exp] = get_plot(query)
            theory[exp] = get_theory(PLOT[exp],nbins,loop=False,funcX=True)


    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax11.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=2.5,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key]*2.0**float(key)
            std  = theory[exp]['std'][key]  *2.0**float(key)
            down = mean - std
            up   = mean + std
            thy_plot ,= ax11.plot(X,mean,linestyle='solid',color='black')
            thy_band  = ax11.fill_between(X,down,up,color='gold',alpha=1.0)


    ax11.set_xlim(0.28, 0.78)
    ax11.set_ylim(1.02, 1.09)

    ax11.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 20)

    ax11.yaxis.set_tick_params(which = 'major', length = 5)
    ax11.yaxis.set_tick_params(which = 'minor', length = 2.5)

    ax11.xaxis.set_tick_params(which = 'major', length = 5)
    ax11.xaxis.set_tick_params(which = 'minor', length = 2.5)

    ax11.set_xlabel(r'\boldmath$x$', size=30)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.text(0.05, 0.85, r'\boldmath$F_2^{^3\rm{He}}/F_2^{D}$', transform = ax11.transAxes, size = 40)

    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.1)
    ax11.xaxis.set_minor_locator(minorLocator)
    ax11.xaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.004)
    majorLocator = MultipleLocator(0.02)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)
    ax11.set_xticks([0.30, 0.40, 0.50, 0.60, 0.70])
    ax11.set_yticks([1.04, 1.06, 1.08])

    handles, labels = [],[]
    handles.append(hand['JLab hd'])
    handles.append((thy_band,thy_plot))
    labels.append(r'\textbf{\textrm{JLab Hall C}}')
    labels.append(r'\textbf{\textrm{JAM}}')
    ax11.legend(handles,labels,loc='lower right', fontsize = 25, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    #################################
    #--Plot F2h/F2t from MARATHON
    #################################

    ht   = data[10051] ## MARATHON h/t

    DATA = {}
    DATA['MARATHON ht']  = pd.DataFrame(ht)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'MARATHON')
            nbins = len(query)
            PLOT[exp] = get_plot(query)
            theory[exp] = get_theory(PLOT[exp],nbins,loop=False,funcX=True)


    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key]*2.0**float(key)
            alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
            color,marker,ms = get_details(exp)
            hand[exp] = ax12.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=2.5,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key]*2.0**float(key)
            std  = theory[exp]['std'][key]  *2.0**float(key)
            down = mean - std
            up   = mean + std
            thy_plot ,= ax12.plot(X,mean,linestyle='solid',color='black')
            thy_band  = ax12.fill_between(X,down,up,color='gold',alpha=1.0)


    ax12.set_xlim(0.10, 0.95)
    ax12.set_ylim(1.05, 1.32)

    ax12.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 20)

    ax12.yaxis.set_tick_params(which = 'major', length = 5)
    ax12.yaxis.set_tick_params(which = 'minor', length = 2.5)

    ax12.xaxis.set_tick_params(which = 'major', length = 5)
    ax12.xaxis.set_tick_params(which = 'minor', length = 2.5)

    ax12.set_xlabel(r'\boldmath$x$', size=30)
    ax12.xaxis.set_label_coords(0.95,0.00)

    ax12.text(0.05, 0.85, r'\boldmath$F_2^{^3\rm{He}}/F_2^{^3\rm{H}}$', transform = ax12.transAxes, size = 40)
    #ax12.text(0.50, 0.05, r'$Q^2 = 14 \cdot x ~ {\rm GeV^2}$', transform = ax12.transAxes, size = 20)

    minorLocator = MultipleLocator(0.04)
    majorLocator = MultipleLocator(0.2)
    ax12.xaxis.set_minor_locator(minorLocator)
    ax12.xaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.1)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)
    ax12.set_xticks([0.2,0.4,0.6,0.8])

    handles, labels = [],[]
    handles.append(hand['MARATHON ht'])
    labels.append(r'\textrm{\textbf{MARATHON}}')
    ax12.legend(handles,labels,loc='lower right', fontsize = 25, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()
    filename = '%s/gallery/DIS-A3'%cwd
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()


if __name__ == "__main__":

    wdir = 'results/misc/marathon_LHAPDF'

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

    predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
    labels  = load('%s/data/labels-%d.dat' % (wdir, istep))
    cluster = labels['cluster']

    data = predictions['reactions']['idis']

    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        for ic in range(kc.nc[istep]):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d' % ic] = np.mean(predictions_ic, axis = 0)
            data[idx]['dthy-%d' % ic] = np.std(predictions_ic, axis = 0)
            if 'X' in data[idx]: data[idx]['x'] = data[idx]['X']
            data[idx]['rQ2'] = np.around(data[idx]['Q2'], decimals = 0)
            data[idx]['rx'] = np.around(data[idx]['x'], decimals = 2)


    checkdir('%s/gallery'%cwd)
    plot_proton  (wdir, data, kc, istep)
    plot_deuteron(wdir, data, kc, istep)
    plot_CC      (wdir, data, kc, istep)
    plot_A3      (wdir, data, kc, istep)







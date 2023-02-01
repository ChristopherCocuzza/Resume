#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
from matplotlib import cm
import pylab as py

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator

#--from local  
from analysis.corelib import core

def reweight(wdir,mode=0):

    #--reweight based on all datasets in input file
    #--mode 0: no chi2 factor for weights
    #--mode 1: w/ chi2 factor for weights
    print('calculating weights...')
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()

    #--load data where residuals are computed
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))
    res=np.array(data['res'])
    N = len(res.T[0])
    npts = len(res[0])

    #--compute chi2 for each replica
    CHI2=[]
    for row in res:
        CHI2.append(np.sum(row**2))
    CHI2=np.array(CHI2)

    Min = np.amin(CHI2)

    if mode==0: w = np.exp(-0.5*(CHI2-Min))
    if mode==1: w = CHI2**(0.5*(npts-1)) * np.exp(-0.5*(CHI2-Min))

    #--normalize to number of replicas
    w = w/np.sum(w)

    filename = '%s/data/weights-%d'%(wdir,mode)
    print()
    print('Saving figures to %s'%filename)
    np.save(filename, w)

    #--calculate effective number of replicas
    #--first remove zeros to avoid divergences
    w = np.array([w[i] for i in range(len(w)) if w[i] > 0])

    Neff = int(round(np.exp(np.sum(w*np.log(1/w)))))
    print('Effective number of replicas: %d out of %d'%(Neff,N))









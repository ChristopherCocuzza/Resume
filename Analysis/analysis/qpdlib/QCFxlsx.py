#!/usr/bin/env python
import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT
import pandas as pd

#--matplotlib
import matplotlib
matplotlib.use('Agg')
import pylab as py
from matplotlib.ticker import MultipleLocator

#--from scipy stack 
from scipy.integrate import fixed_quad
from scipy import interpolate

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#-- from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--convert QCF into Excel sheet

def gen_xlsx(wdir,QCF,Q2,xmin=-6,scale='mixed'):

    print('\ngenerating %s at Q2=%s from %s'%(QCF,Q2,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    resman = RESMAN(nworkers = 1, parallel = False, datasets = False)
    parman = resman.parman
    parman.order = replicas[0]['order'][istep]
    qcf = conf[QCF]


    if QCF=='pdf':
        if scale=='mixed':
            X=10**np.linspace(xmin,-1,200)
            X=np.append(X,np.linspace(0.1,0.99,200))
        elif scale=='linear':
            X=np.linspace(10**(xmin),0.99,200)
        flavors = ['up','dp','sp','um','dm','sm','u','d','s','ub','db','sb','ub+db','db-ub'] 

    if QCF=='ppdf':
        if scale=='mixed':
            X=10**np.linspace(xmin,-1,200)
            X=np.append(X,np.linspace(0.1,0.99,200))
        elif scale=='linear':
            X=np.linspace(10**(xmin),0.99,200)
        flavors = ['up','dp','sp','um','dm','sm','u','d','s','ub','db','sb','ub+db','ub-db'] 

    xlsx = {}

    xlsx['QCF'] = [('X*'+QCF) for x in X]
    xlsx['X']  = X
    xlsx['Q2'] = np.ones(len(X))*Q2

    ## compute XF for all replicas
    XF = {}
    n_replicas = len(replicas)
    for i in range(n_replicas):
        lprint('%d/%d' % (i + 1, n_replicas))

        core.mod_conf(istep, replicas[i])
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        for flavor in flavors:
            if flavor not in XF: XF[flavor] = []
            if flavor == 'up':
                func = lambda x: qcf.get_xF(x, Q2, 'u') + qcf.get_xF(x, Q2, 'ub')
            elif flavor == 'dp':
                func = lambda x: qcf.get_xF(x, Q2, 'd') + qcf.get_xF(x, Q2, 'db')
            elif flavor == 'sp':
                func = lambda x: qcf.get_xF(x, Q2, 's') + qcf.get_xF(x, Q2, 'sb')
            elif flavor == 'um':
                func = lambda x: qcf.get_xF(x, Q2, 'u') - qcf.get_xF(x, Q2, 'ub')
            elif flavor == 'dm':
                func = lambda x: qcf.get_xF(x, Q2, 'd') - qcf.get_xF(x, Q2, 'db')
            elif flavor == 'sm':
                func = lambda x: qcf.get_xF(x, Q2, 's') - qcf.get_xF(x, Q2, 'sb')
            elif flavor == 'u':
                func = lambda x: qcf.get_xF(x, Q2, 'u')
            elif flavor == 'd':
                func = lambda x: qcf.get_xF(x, Q2, 'd')
            elif flavor == 'ub+db':
                func = lambda x: qcf.get_xF(x, Q2, 'ub') + qcf.get_xF(x, Q2, 'db')
            elif flavor == 'ub-db':
                func = lambda x: qcf.get_xF(x, Q2, 'ub') - qcf.get_xF(x, Q2, 'db')
            elif flavor == 'db-ub':
                func = lambda x: qcf.get_xF(x, Q2, 'db') - qcf.get_xF(x, Q2, 'ub')
            else:
                func = lambda x: qcf.get_xF(x, Q2, flavor)

            XF[flavor].append([func(x) for x in X])


    for flavor in flavors:
       xlsx[flavor+' mean'] = np.mean(XF[flavor],axis=0) 
       xlsx[flavor+' std']  = np.std (XF[flavor],axis=0) 

    checkdir('%s/data'%wdir)
    filename = '%s/data/xlsx-%s-Q2=%s.xlsx'%(wdir,QCF,Q2)
    xlsx = pd.DataFrame(xlsx)
    xlsx.to_excel(filename,index=False) 

    print('Saving xlsx file for %s at Q2=%s to %s'%(QCF,Q2,filename))











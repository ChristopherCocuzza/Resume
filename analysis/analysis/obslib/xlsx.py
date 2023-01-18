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


def gen_xlsx(process,idx,wdir):

    if process == 'idis':  from obslib.idis.reader import READER
    if process == 'sidis': from obslib.sidis.reader import READER

    print('\ngenerating %s %s values from %s'%(process,idx,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if process not in predictions['reactions']: return

    filters = conf['datasets']['%s'%process]['filters']

    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['%s'%process]={}
    conf['datasets']['%s'%process]['xlsx']={}
    #conf['datasets']['%s'%process]['xlsx'][idx]='%s/expdata/%s.xlsx'%(process,idx)
    conf['datasets']['%s'%process]['xlsx'][idx]='./tabs/%s.xlsx'%(idx)
    conf['datasets']['%s'%process]['norm']={}
    conf['datasets']['%s'%process]['filters']=filters
    conf['%s tabs'%process]=READER().load_data_sets('%s'%process)
    tabs = conf['%s tabs'%process]
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['%s'%process]

    #--get theory by seperating solutions and taking mean
    #--remove any correlated errors from the prediction
    predictions = copy.copy(np.array(data[idx]['prediction-rep'])-np.array(data[idx]['shift-rep']))

    data[idx]['thy']  = np.mean(predictions,axis=0)
    data[idx]['dthy'] = np.std (predictions,axis=0)

    xlsx = {}
    if process=='idis':
        xlsx['X']  = tabs[idx]['X']
        xlsx['Q2'] = tabs[idx]['Q2']
        if 'thetac' in tabs[idx]: xlsx['theta'] = tabs[idx]['thetac']
    elif process=='sidis':
        xlsx['X']  = tabs[idx]['X']
        xlsx['Q2'] = tabs[idx]['Q2']
        xlsx['Z']  = tabs[idx]['Z']
        xlsx['E']  = tabs[idx]['E']

    xlsx['thy'] = data[idx]['thy']
    xlsx['std'] = data[idx]['dthy']

    #xlsx['thy'][xlsx['theta']<22]
    
    checkdir('%s/data'%wdir)
    filename = '%s/data/xlsx-%s-%s.xlsx'%(wdir,process,idx)
    xlsx = pd.DataFrame(xlsx)
    xlsx.to_excel(filename,index=False) 

    print('Saving xlsx file for %s %s to %s'%(process,idx,filename))

    return

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*10,nrows*6))
    ax11 = py.subplot(nrows,ncols,1)

    ax11.errorbar(xlsx['X'][xlsx['theta']<22] ,xlsx['thy'][xlsx['theta']<22] ,yerr=xlsx['std'][xlsx['theta']<22] ,color='black',fmt='.',ms=10,capsize=3.0)

    for ax in [ax11]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0.25,0.43)

    for ax in [ax11]:
        minorLocator = MultipleLocator(0.004)
        majorLocator = MultipleLocator(0.02)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    ax11.set_ylim(0.78,0.88)

    minorLocator = MultipleLocator(0.004)
    majorLocator = MultipleLocator(0.02)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)

    ax11.text(0.05, 0.05, r'\boldmath$F_2^D/F_2^p$'                 ,transform=ax11.transAxes,size=40)
    ax11.set_xlabel(r'\boldmath$x_{\rm B}$',size=30)

    py.tight_layout()
    py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/test.png'%(wdir)

    py.savefig(filename)










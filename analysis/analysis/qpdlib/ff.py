#!/usr/bin/env python
import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

## matplotlib
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#matplotlib.rc('text',usetex=True)
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pylab as py
import matplotlib.gridspec as gridspec

## from scipy stack
from scipy.integrate import quad

## from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

from qcdlib.qpdcalc import QPDCALC

## from fitlib
from fitlib.resman import RESMAN

## from local
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

cmap = matplotlib.cm.get_cmap('plasma')

def gen_xf(wdir, had, flavors = ['g', 'u', 'ub', 'd', 'db', 's', 'sb','c','b'], Q2 = None):

    fflabel='ff%s'%had

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    if Q2==None: Q2 = conf['Q20']
    print('\ngenerating ff-%s from %s at %3.5f' % (had,wdir,Q2))

    replicas = core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) ## set conf as specified in istep

    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('ff-%s not in active distribution' % had)
            return

    resman = RESMAN(nworkers = 1, parallel = False, datasets = False)
    parman = resman.parman

    #jar = load('%s/data/jar-%d.dat' % (wdir, istep))
    #replicas = jar['replicas']

    ff = conf[fflabel]

    ## setup kinematics
    X = np.linspace(0.01, 0.99, 100)

    ## compute XF for all replicas
    XF = {}
    cnt = 0
    for par in replicas:
        lprint('%d/%d' % (cnt+1, len(replicas)))

        core.mod_conf(istep, replicas[cnt])
        parman.set_new_params(replicas[cnt]['params'][istep], initial = True)
        #parman.order = jar['order']
        #parman.set_new_params(par, initial = True)


        for flavor in flavors:
            if flavor not in XF: XF[flavor] = []
            if   flavor=='c' or flavor=='cb' or flavor=='c+cb': 
                if _Q2 < conf['aux'].mc2: _Q2=conf['aux'].mc2+1
                else:                     _Q2=Q2
            elif flavor=='b' or flavor=='bb' or flavor=='b+bb':
                if _Q2 < conf['aux'].mb2: _Q2=conf['aux'].mb2+1
                else:                     _Q2=Q2
            else:                         _Q2=Q2
            if  flavor=='u+ub':
                func=lambda x: ff.get_xF(x,_Q2,'u')+ff.get_xF(x,_Q2,'ub')
            elif flavor=='d+db':
                func=lambda x: ff.get_xF(x,_Q2,'d')+ff.get_xF(x,_Q2,'db')
            elif flavor=='s+sb':
                func=lambda x: ff.get_xF(x,_Q2,'s')+ff.get_xF(x,_Q2,'sb')
            elif flavor=='c+cb':
                func=lambda x: ff.get_xF(x,_Q2,'c')+ff.get_xF(x,_Q2,'cb')
            elif flavor=='b+bb':
                func=lambda x: ff.get_xF(x,_Q2,'b')+ff.get_xF(x,_Q2,'bb')
            else:
                func=lambda x: ff.get_xF(x,_Q2,flavor)

            XF[flavor].append(np.array([func(x) for x in X]))
        cnt += 1
    
    print
    checkdir('%s/data' % wdir)
    filename = '%s/data/ff%s-Q2=%3.5f.dat' % (wdir, had, Q2)
    save({'X': X, 'Q2': Q2, 'XF': XF},filename) 
    print('Saving data to %s'%filename)

    print()

def plot_xf_pion(wdir,Q2=None,mode=0):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    fflabel = 'ffpion'
    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)

    hand = {}
    thy = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    if Q2==None: Q2 = conf['Q20']

    scale = classifier.get_scale(wdir)
    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('%s not in active distribution' % fflabel)
            return

    filename = '%s/data/ffpion-Q2=%3.5f.dat' % (wdir, Q2)
    #--load data if it exists
    try:
        data = load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,'pion',Q2=Q2)
        data = load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]


    flavs = ['up','u','ub','g','c','b','sp']
    for flav in flavs:
        X=data['X']
        if   flav=='up': _data = np.array(data['XF']['u']) + np.array(data['XF']['ub'])
        elif flav=='sp': _data = np.array(data['XF']['s']) + np.array(data['XF']['sb'])
        else:            _data = data['XF'][flav]
        mean = np.mean(_data,axis=0)
        std  = np.std (_data,axis=0)

        if flav=='up': ax,color = ax11,'blue'
        if flav=='ub': ax,color = ax11,'green'
        if flav=='u':  ax,color = ax11,'red'
        if flav=='sp': ax,color = ax11,'magenta'
        if flav=='g':  ax,color = ax12,'blue'
        if flav=='c':  ax,color = ax12,'green'
        if flav=='b':  ax,color = ax12,'red'

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF']['u'])):
                hand[flav] ,= ax.plot(X,_data[i],color=color,alpha=0.3)
      
        #--plot average and standard deviation
        if mode==1:
            hand[flav]  = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.9,zorder=1)



    for ax in [ax11,ax12]:
          ax.set_xlabel(r'\boldmath$z$'    ,size=40)
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
          ax.xaxis.set_label_coords(0.98,0.00)
          ax.set_xlim(0.2,0.90)
          ax.set_xticks([0.2,0.4,0.6,0.8])
          ax.set_ylim(0,0.8)
          ax.set_yticks([0.2,0.4,0.6,0.8])
          minorLocator = MultipleLocator(0.05)
          ax.xaxis.set_minor_locator(minorLocator)
          minorLocator = MultipleLocator(0.05)
          ax.yaxis.set_minor_locator(minorLocator)


    ax12.text(0.65 ,0.75  ,r'\boldmath$z D^{\pi^+}_q$'   ,transform=ax12.transAxes,size=50)

    #ax11.text(0.25 ,0.80  ,r'\boldmath$u^+$'     ,transform=ax11.transAxes,size=30,color='blue')
    #ax11.text(0.35 ,0.70  ,r'\boldmath$u$'       ,transform=ax11.transAxes,size=30,color='red')
    #ax11.text(0.20 ,0.30  ,r'\boldmath$\bar{u}$' ,transform=ax11.transAxes,size=30,color='green')
    #ax11.text(0.25 ,0.80  ,r'\boldmath$g$'       ,transform=ax11.transAxes,size=30,color='blue')
    #ax11.text(0.35 ,0.70  ,r'\boldmath$c$'       ,transform=ax11.transAxes,size=30,color='red')
    #ax11.text(0.20 ,0.30  ,r'\boldmath$b$'       ,transform=ax11.transAxes,size=30,color='green')

    if Q2 == 1.27**2: ax12.text(0.55,0.60,r'$Q^2 = m_c^2$'                                  , transform=ax12.transAxes,size=30)
    else:             ax12.text(0.55,0.60,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax12.transAxes,size=30)

    #if mode==0:
    #    sm   = py.cm.ScalarMappable(cmap=cmap)
    #    sm.set_array([])
    #    cax = fig.add_axes([0.56,0.28,0.20,0.05])
    #    cax.tick_params(axis='both',which='both',labelsize=20,direction='in')
    #    cax.xaxis.set_label_coords(0.65,-1.2)
    #    cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    #    cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=30)


    handles,labels = [],[]
    handles.append(hand['up'])
    handles.append(hand['u'])
    handles.append(hand['ub'])
    handles.append(hand['sp'])
    labels.append(r'\boldmath$u^+$')
    labels.append(r'\boldmath$u$')
    labels.append(r'\boldmath$\bar{u}$')
    labels.append(r'\boldmath$s^+$')
    ax11.legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    handles,labels = [],[]
    handles.append(hand['g'])
    handles.append(hand['c'])
    handles.append(hand['b'])
    labels.append(r'\boldmath$g$')
    labels.append(r'\boldmath$c$')
    labels.append(r'\boldmath$b$')
    ax12.legend(handles,labels,loc='lower right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()

    filename = '%s/gallery/ff-pion-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_xf_kaon(wdir,Q2=None,mode=0):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    fflabel = 'ffkaon'
    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)

    hand = {}
    thy = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    if Q2==None: Q2=conf['Q20']

    scale = classifier.get_scale(wdir)
    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('%s not in active distribution' % fflabel)
            return

    filename = '%s/data/ffkaon-Q2=%3.5f.dat' % (wdir, Q2)
    #--load data if it exists
    try:
        data = load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,'kaon',Q2=Q2)
        data = load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]


    flavs = ['sp','s','up','dp','g','c','b']
    for flav in flavs:
        X=data['X']
        if   flav=='up': _data = np.array(data['XF']['u']) + np.array(data['XF']['ub'])
        elif flav=='dp': _data = np.array(data['XF']['d']) + np.array(data['XF']['db'])
        elif flav=='sp': _data = np.array(data['XF']['s']) + np.array(data['XF']['sb'])
        else:            _data = data['XF'][flav]
        mean = np.mean(_data,axis=0)
        std  = np.std (_data,axis=0)

        if flav=='sp': ax,color = ax11,'blue'
        if flav=='s':  ax,color = ax11,'green'
        if flav=='up': ax,color = ax11,'red'
        if flav=='dp': ax,color = ax11,'magenta'
        if flav=='g':  ax,color = ax12,'blue'
        if flav=='c':  ax,color = ax12,'green'
        if flav=='b':  ax,color = ax12,'red'

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF']['u'])):
                hand[flav] ,= ax.plot(X,_data[i],color=color,alpha=0.3)
      
        #--plot average and standard deviation
        if mode==1:
            hand[flav]  = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.9,zorder=1)



    for ax in [ax11,ax12]:
          ax.set_xlabel(r'\boldmath$z$'    ,size=40)
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
          ax.xaxis.set_label_coords(0.98,0.00)
          ax.set_xlim(0.2,0.90)
          ax.set_xticks([0.2,0.4,0.6,0.8])
          ax.set_ylim(0,0.4)
          ax.set_yticks([0.1,0.2,0.3,0.4])
          minorLocator = MultipleLocator(0.05)
          ax.xaxis.set_minor_locator(minorLocator)
          minorLocator = MultipleLocator(0.05)
          ax.yaxis.set_minor_locator(minorLocator)


    ax12.text(0.65 ,0.75  ,r'\boldmath$z D^{K^+}_q$'   ,transform=ax12.transAxes,size=50)

    #ax11.text(0.25 ,0.80  ,r'\boldmath$u^+$'     ,transform=ax11.transAxes,size=30,color='blue')
    #ax11.text(0.35 ,0.70  ,r'\boldmath$u$'       ,transform=ax11.transAxes,size=30,color='red')
    #ax11.text(0.20 ,0.30  ,r'\boldmath$\bar{u}$' ,transform=ax11.transAxes,size=30,color='green')
    #ax11.text(0.25 ,0.80  ,r'\boldmath$g$'       ,transform=ax11.transAxes,size=30,color='blue')
    #ax11.text(0.35 ,0.70  ,r'\boldmath$c$'       ,transform=ax11.transAxes,size=30,color='red')
    #ax11.text(0.20 ,0.30  ,r'\boldmath$b$'       ,transform=ax11.transAxes,size=30,color='green')

    if Q2 == 1.27**2: ax12.text(0.55,0.60,r'$Q^2 = m_c^2$'                                  , transform=ax12.transAxes,size=30)
    else:             ax12.text(0.55,0.60,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax12.transAxes,size=30)

    #if mode==0:
    #    sm   = py.cm.ScalarMappable(cmap=cmap)
    #    sm.set_array([])
    #    cax = fig.add_axes([0.56,0.28,0.20,0.05])
    #    cax.tick_params(axis='both',which='both',labelsize=20,direction='in')
    #    cax.xaxis.set_label_coords(0.65,-1.2)
    #    cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    #    cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=30)


    handles,labels = [],[]
    handles.append(hand['sp'])
    handles.append(hand['s'])
    handles.append(hand['up'])
    handles.append(hand['dp'])
    labels.append(r'\boldmath$s^+$')
    labels.append(r'\boldmath$s$')
    labels.append(r'\boldmath$u^+$')
    labels.append(r'\boldmath$d^+$')
    ax11.legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    handles,labels = [],[]
    handles.append(hand['g'])
    handles.append(hand['c'])
    handles.append(hand['b'])
    labels.append(r'\boldmath$g$')
    labels.append(r'\boldmath$c$')
    labels.append(r'\boldmath$b$')
    ax12.legend(handles,labels,loc='lower right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)



    py.tight_layout()

    filename = '%s/gallery/ff-kaon-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_xf_hadron(wdir,Q2=None,mode=0):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    fflabel = 'ffhadron'
    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)

    hand = {}
    thy = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    if Q2==None: Q2=conf['Q20']

    scale = classifier.get_scale(wdir)
    if fflabel not in conf['steps'][istep]['active distributions']:
        if fflabel not in conf['steps'][istep]['passive distributions']:
            print('%s not in active distribution' % fflabel)
            return

    filename = '%s/data/ffhadron-Q2=%3.5f.dat' % (wdir, Q2)
    #--load data if it exists
    try:
        data = load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,'hadron',Q2=Q2)
        data = load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]


    flavs = ['u','ub','g','c','b']
    for flav in flavs:
        X=data['X']
        if   flav=='up': _data = np.array(data['XF']['u']) + np.array(data['XF']['ub'])
        elif flav=='dp': _data = np.array(data['XF']['d']) + np.array(data['XF']['db'])
        elif flav=='sp': _data = np.array(data['XF']['s']) + np.array(data['XF']['sb'])
        else:            _data = data['XF'][flav]
        mean = np.mean(_data,axis=0)
        std  = np.std (_data,axis=0)

        if flav=='u':  ax,color = ax11,'red'
        if flav=='ub': ax,color = ax11,'magenta'
        if flav=='g':  ax,color = ax12,'blue'
        if flav=='c':  ax,color = ax12,'green'
        if flav=='b':  ax,color = ax12,'red'

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF']['u'])):
                hand[flav] ,= ax.plot(X,_data[i],color=color,alpha=0.3)
      
        #--plot average and standard deviation
        if mode==1:
            hand[flav]  = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.9,zorder=1)



    for ax in [ax11,ax12]:
          ax.set_xlabel(r'\boldmath$z$'    ,size=40)
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
          ax.xaxis.set_label_coords(0.98,0.00)
          ax.set_xlim(0.2,0.90)
          ax.set_xticks([0.2,0.4,0.6,0.8])
          ax.set_ylim(0,0.4)
          ax.set_yticks([0.1,0.2,0.3,0.4])
          minorLocator = MultipleLocator(0.05)
          ax.xaxis.set_minor_locator(minorLocator)
          minorLocator = MultipleLocator(0.05)
          ax.yaxis.set_minor_locator(minorLocator)


    ax12.text(0.65 ,0.75  ,r'\boldmath$z D^{h^+}_q$'   ,transform=ax12.transAxes,size=50)

    #ax11.text(0.25 ,0.80  ,r'\boldmath$u^+$'     ,transform=ax11.transAxes,size=30,color='blue')
    #ax11.text(0.35 ,0.70  ,r'\boldmath$u$'       ,transform=ax11.transAxes,size=30,color='red')
    #ax11.text(0.20 ,0.30  ,r'\boldmath$\bar{u}$' ,transform=ax11.transAxes,size=30,color='green')
    #ax11.text(0.25 ,0.80  ,r'\boldmath$g$'       ,transform=ax11.transAxes,size=30,color='blue')
    #ax11.text(0.35 ,0.70  ,r'\boldmath$c$'       ,transform=ax11.transAxes,size=30,color='red')
    #ax11.text(0.20 ,0.30  ,r'\boldmath$b$'       ,transform=ax11.transAxes,size=30,color='green')

    if Q2 == 1.27**2: ax12.text(0.55,0.60,r'$Q^2 = m_c^2$'                                  , transform=ax12.transAxes,size=30)
    else:             ax12.text(0.55,0.60,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax12.transAxes,size=30)

    #if mode==0:
    #    sm   = py.cm.ScalarMappable(cmap=cmap)
    #    sm.set_array([])
    #    cax = fig.add_axes([0.56,0.28,0.20,0.05])
    #    cax.tick_params(axis='both',which='both',labelsize=20,direction='in')
    #    cax.xaxis.set_label_coords(0.65,-1.2)
    #    cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    #    cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=30)


    handles,labels = [],[]
    handles.append(hand['u'])
    handles.append(hand['ub'])
    labels.append(r'\boldmath$u$')
    labels.append(r'\boldmath$\bar{u}$')
    ax11.legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    handles,labels = [],[]
    handles.append(hand['g'])
    handles.append(hand['c'])
    handles.append(hand['b'])
    labels.append(r'\boldmath$g$')
    labels.append(r'\boldmath$c$')
    labels.append(r'\boldmath$b$')
    ax12.legend(handles,labels,loc='lower right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)



    py.tight_layout()

    filename = '%s/gallery/ff-hadron-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)



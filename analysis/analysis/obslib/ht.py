#!/usr/bin/env python
import os,sys
from tools.config import load_config,conf
from fitlib.resman import RESMAN
import numpy as np

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab  as py
from matplotlib.ticker import MultipleLocator
from matplotlib import cm

#--from tools
from tools.tools     import load,save,checkdir,lprint

#--from corelib
from analysis.corelib import core,classifier

from analysis.obslib import stf

import kmeanconf as kc

cmap = cm.get_cmap('plasma')

#--plot as function of W2
def gen_ht_W2fixed(wdir,tar='p',W2cut=3.5):
  
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating STF from %s for %s at W2 = %3.5f'%(wdir,tar,W2cut))

    if 'ht4' not in conf['steps'][istep]['active distributions']:
        if 'ht4' not in conf['steps'][istep]['passive distributions']:
                print('ht is not an active or passive distribution')
                return 

    M2 = conf['aux'].M2

    #--setup kinematics for higher twist to be calculated at
    #Xgrid = np.geomspace(1e-5,1e-1,20)
    #Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    #Q2grid = Xgrid/(1-Xgrid)*(W2cut - M2) 
    #conf['idis grid'] = {}
    #conf['idis grid']['X']  = Xgrid 
    #conf['idis grid']['Q2'] = Q2grid 
    conf['idis grid'] = 'predict'
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
    
    idis.data[tar] = {}
    idis.data[tar]['F2'] = np.zeros(idis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics for structure functions to be interpolated to
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))
    Q2 = X/(1-X)*(W2cut - M2) 

    ht4 = conf['ht4']

    if 'ht type' in conf: TYPE = conf['ht type']
    else:                 TYPE = 'mult'

    #--compute X*STF for all replicas
    STF  = []
    ADD  = []     
    MULT = []
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        idis._update()

        #--get LT + TMC structure functions
        if TYPE=='mult': stf  = idis.get_stf(X,Q2,stf='F2',tar=tar)/(1+ht4.get_ht(X,Q2,tar,'F2'))
        if TYPE=='add':  stf  = idis.get_stf(X,Q2,stf='F2',tar=tar) - ht4.get_ht(X,Q2,tar,'F2')

        #--get additive higher twist contributions (multiplied by x and divided by Q2)
        if TYPE=='mult': add  = X*stf*ht4.get_ht(X,Q2,tar,'F2')
        if TYPE=='add':  add  = X*ht4.get_ht(X,Q2,tar,'F2')

        #--get multiplicative higher twist contributions (divided by Q2)
        if TYPE=='mult': mult = ht4.get_ht(X,Q2,tar,'F2')
        if TYPE=='add':  mult = ht4.get_ht(X,Q2,tar,'F2')/stf

        STF .append(stf)
        ADD .append(add)
        MULT.append(mult)

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/ht-%s-W2=%3.5f.dat'%(wdir,tar,W2cut)

    save({'X':X,'Q2':Q2,'ADD':ADD,'MULT':MULT,'STF':STF},filename)
    print ('Saving data to %s'%filename)

#--outdated
def plot_ht_W2fixed(wdir,W2cut=3.0,mode=0):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas

    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1) 
    ax12 = py.subplot(nrows,ncols,2) 
    ax13 = py.subplot(nrows,ncols,3) 


    hand = {}
    replicas = core.get_replicas(wdir)

    scale = classifier.get_scale(wdir)

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    core.mod_conf(istep,replicas[0])
    #conf['idis grid'] = 'prediction'
    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman
    cluster,colors,nc,cluster_order= classifier.get_clusters(wdir,istep,kc) 

    replicas = core.get_replicas(wdir)

    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']

    #--generate structure functions
    TAR = ['p','n']
    STF = {}
    for tar in TAR:
        filename ='%s/data/ht-%s-W2=%3.5f.dat'%(wdir,tar,W2cut)
        try:
            STF[tar] = load(filename)
        except:
            gen_ht_W2fixed(wdir,tar=tar,W2cut=W2cut)
            STF[tar] = load(filename)


    X        = STF['p']['X']
    STF['p'] = STF['p']['XF']
    STF['n'] = STF['n']['XF']

    M2 = conf['aux'].M2
    Q2 = X/(1-X)*(W2cut - M2) 

    if 'ht4' in conf: ht4 = conf['ht4']
    else:
        print('Higher twist corrections not present.')
        return       

    ht_type = 'mult'
    if 'ht type' in conf: ht_type = conf['ht type']

    ##############################################
    #--plot ht
    ##############################################
    cnt = 0
    F2p, F2n, F2pHT, F2nHT = [], [], [], []
    plus, minus = [], []
    for par in replicas:
        parman.set_new_params(par)
        if ht_type=='mult':
            F2pHT.append(ht4.get_ht(X,Q2,'p','F2'))
            F2nHT.append(ht4.get_ht(X,Q2,'n','F2'))
            F2p.append(STF['p'][cnt]*ht4.get_ht(X,Q2,'p','F2')/X)
            F2n.append(STF['n'][cnt]*ht4.get_ht(X,Q2,'n','F2')/X)
            plus.append (STF['n'][cnt]*ht4.get_ht(X,Q2,'n','F2')/X + STF['p'][cnt]*ht4.get_ht(X,Q2,'p','F2')/X)
            minus.append(STF['n'][cnt]*ht4.get_ht(X,Q2,'n','F2')/X - STF['p'][cnt]*ht4.get_ht(X,Q2,'p','F2')/X)
        else: pass
        if mode==0:
            hand['p']     ,= ax11.plot(X,F2pHT[cnt]       ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax11.plot(X,F2nHT[cnt]       ,color='green' ,alpha=0.3)
            hand['p']     ,= ax12.plot(X,F2p[cnt]         ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax12.plot(X,F2n[cnt]         ,color='green' ,alpha=0.3)
            hand['plus']  ,= ax12.plot(X,plus[cnt]        ,color='blue'  ,alpha=0.3)
            hand['minus'] ,= ax12.plot(X,minus[cnt]       ,color='purple',alpha=0.3)
            hand['p']     ,= ax13.plot(X,1/(1+F2pHT[cnt]) ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax13.plot(X,1/(1+F2nHT[cnt]) ,color='green' ,alpha=0.3)
        cnt+=1

    if mode == 1:
        meanp = np.mean(np.array(F2pHT),axis=0)
        stdp  = np.std (np.array(F2pHT),axis=0)
        meann = np.mean(np.array(F2nHT),axis=0)
        stdn  = np.std (np.array(F2nHT),axis=0)
        hand['p'] = ax11.fill_between(X,meanp-stdp,meanp+stdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n'] = ax11.fill_between(X,meann-stdn,meann+stdn,color='green',alpha=0.9, zorder=1)
        meanp = np.mean(np.array(F2p),axis=0)
        stdp  = np.std (np.array(F2p),axis=0)
        meann = np.mean(np.array(F2n),axis=0)
        stdn  = np.std (np.array(F2n),axis=0)
        meanplus  = np.mean(np.array(plus),axis=0)
        stdplus   = np.std (np.array(plus),axis=0)
        meanminus = np.mean(np.array(minus),axis=0)
        stdminus  = np.std (np.array(minus),axis=0)
        hand['p']     = ax12.fill_between(X,meanp-stdp,meanp+stdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n']     = ax12.fill_between(X,meann-stdn,meann+stdn,color='green',alpha=0.9, zorder=1)
        hand['plus']  = ax12.fill_between(X,meanplus-stdplus  ,meanplus+stdplus  ,color='blue'  ,alpha=0.9, zorder=1)
        hand['minus'] = ax12.fill_between(X,meanminus-stdminus,meanminus+stdminus,color='purple',alpha=0.9, zorder=1)
        meanp = np.mean(1/(1+np.array(F2pHT)),axis=0)
        stdp  = np.std (1/(1+np.array(F2pHT)),axis=0)
        meann = np.mean(1/(1+np.array(F2nHT)),axis=0)
        stdn  = np.std (1/(1+np.array(F2nHT)),axis=0)
        hand['p'] = ax13.fill_between(X,meanp-stdp,meanp+stdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n'] = ax13.fill_between(X,meann-stdn,meann+stdn,color='green',alpha=0.9, zorder=1)

    print 
    ##############################################
    h0 =-3.2874
    h1 = 1.9274
    h2 =-2.0701
    ht = h0*X**h1*(1+h2*X)
    hand['CJ15'] ,= ax11.plot(X,ht/Q2,'k--')
   
    ax11.set_ylim(-0.16,0.30)
    ax11.set_yticks([-0.1,0,0.1,0.2])

    ax12.set_ylim(-0.001,0.0038)

    ax13.set_ylim(0.80,1.00)
    ax13.set_yticks([0.85,0.90,0.95])

    ax11.text(0.05,0.78,r'\boldmath$\frac{C_{\rm HT}^N}{Q^2}$',transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.78,r'\boldmath$F_2^N \frac{C^N_{\rm HT}}{Q^2}$',transform=ax12.transAxes,size=40)
    ax13.text(0.05,0.05,r'\boldmath$\frac{F_2^{N,{\rm LT}}}{F_2^{N,{\rm tot}}}$',transform=ax13.transAxes,size=40)


    #ax11.text(0.45,0.40,r'\textrm{\textbf{JAM PRELIMINARY}}',transform=ax11.transAxes,size=20,color='red')

    xmin,xmax = 0.43,0.90
    Q2min = xmin/(1-xmin)*(W2cut - M2) 
    Q2max = xmax/(1-xmax)*(W2cut - M2) 
    for ax in [ax11,ax12,ax13]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(xmin,xmax)
        ax.set_xlabel(r'\boldmath$x$',size=30)
        ax.xaxis.set_label_coords(0.95,0.00)
        ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.5,0.6,0.7,0.8])

    ax11.text(0.35,0.15,r'$W^2 = %d~{\rm GeV}^2$'%W2cut,transform=ax11.transAxes,size=25)
    ax11.text(0.25,0.05,r'$%3.2f < Q^2 < %3.0f~{\rm GeV}^2$'%(Q2min,Q2max),transform=ax11.transAxes,size=25)
    if conf['tmc']=='GP':  ax11.text(0.05,0.05,r'\textbf{\textrm{(GP)}}' ,size=25,transform=ax11.transAxes)
    if conf['tmc']=='AOT': ax11.text(0.05,0.05,r'\textbf{\textrm{(AOT)}}',size=25,transform=ax11.transAxes)

    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.1)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(0.001)
    majorLocator = MultipleLocator(0.002)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.05)
    ax13.yaxis.set_minor_locator(minorLocator)
    ax13.yaxis.set_major_locator(majorLocator)


    handles,labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    handles.append(hand['CJ15'])
    labels.append(r'\boldmath$p$')
    labels.append(r'\boldmath$n$')
    labels.append(r'\textbf{\textrm{CJ15}}')
    ax11.legend(handles,labels,frameon=False,loc='lower right',fontsize=20, handletextpad = 0.5, handlelength = 1.5, ncol=1, columnspacing = 0.5)

    handles,labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    handles.append(hand['plus'])
    handles.append(hand['minus'])
    labels.append(r'\boldmath$p$')
    labels.append(r'\boldmath$n$')
    labels.append(r'\boldmath$n+p$')
    labels.append(r'\boldmath$n-p$')
    ax12.legend(handles,labels,frameon=False,loc='upper right',fontsize=20, handletextpad = 0.5, handlelength = 1.5, ncol=1, columnspacing = 0.5)


    #if mode==0:
    #    sm   = py.cm.ScalarMappable(cmap=cmap)
    #    sm.set_array([])
    #    cax = fig.add_axes([0.45,0.88,0.30,0.04])
    #    cax.tick_params(axis='both',which='both',labelsize=15,direction='in')
    #    cax.xaxis.set_label_coords(0.65,-1.8)
    #    cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    #    cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=20)

    py.tight_layout()
    filename = '%s/gallery/ht-W2=%3.5f'%(wdir,W2cut)
    if mode == 1: filename += '-bands'
    filename += '.png'
    print()
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()


#--plot as function of Q2
#--need to update
def gen_ht_Q2fixed(wdir,tar='p',Q2=None):
  
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating STF from %s for %s at W2 = %3.5f'%(wdir,tar,W2cut))

    if 'ht4' not in conf['steps'][istep]['active distributions']:
        if 'ht4' not in conf['steps'][istep]['passive distributions']:
                print('ht is not an active or passive distribution')
                return 

    M2 = conf['aux'].M2

    #--setup kinematics for higher twist to be calculated at
    #Xgrid = np.geomspace(1e-5,1e-1,20)
    #Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    #Q2grid = Xgrid/(1-Xgrid)*(W2cut - M2) 
    #conf['idis grid'] = {}
    #conf['idis grid']['X']  = Xgrid 
    #conf['idis grid']['Q2'] = Q2grid 
    conf['idis grid'] = 'predict'
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
    
    idis.data[tar] = {}
    idis.data[tar]['F2'] = np.zeros(idis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics for structure functions to be interpolated to
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))
    Q2 = X/(1-X)*(W2cut - M2) 

    ht4 = conf['ht4']
    #--compute X*STF for all replicas        
    XF=[]
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        idis._update()

        xf = X*idis.get_stf(X,Q2,stf='F2',tar=tar)*ht4.get_ht(X,Q2,tar,'F2')
        XF.append(xf)

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/ht-%s-Q2=%3.5f.dat'%(wdir,tar,W2cut)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_ht_Q2fixed(wdir,Q2=None,mode=0):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1) 
    ax12 = py.subplot(nrows,ncols,2) 


    hand = {}
    replicas = core.get_replicas(wdir)

    scale = classifier.get_scale(wdir)

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    if Q2==None: Q2 = conf['Q20']
    core.mod_conf(istep,replicas[0])
    #conf['idis grid'] = 'prediction'
    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman
    cluster,colors,nc,cluster_order= classifier.get_clusters(wdir,istep,kc) 

    replicas = core.get_replicas(wdir)

    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']

    #--generate structure functions
    if Q2 == None: Q2=conf['Q20']
    _stf = 'F2'
    TAR = ['p','n']
    STF = {}
    for tar in TAR:
        filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,_stf,Q2)
        try:
            STF[tar] = load(filename)
        except:
            stf.gen_stf(wdir,Q2,tar=tar,stf=_stf)
            STF[tar] = load(filename)

    X   = STF['p']['X']
    STF['p'] = STF['p']['XF']
    STF['n'] = STF['n']['XF']


    if 'ht4' in conf: ht4 = conf['ht4']
    else:
        print('Higher twist corrections not present.')
        return       

    ht_type = 'mult'
    if 'ht type' in conf: ht_type = conf['ht type']

    ##############################################
    #--plot ht
    ##############################################
    cnt = 0
    F2p, F2n, F2pHT, F2nHT = [], [], [], []
    plus, minus = [], []
    for par in replicas:
        parman.set_new_params(par)
        if ht_type=='mult':
            F2pHT.append(ht4.get_ht(X,Q2,'p','F2'))
            F2nHT.append(ht4.get_ht(X,Q2,'n','F2'))
            F2p.append(STF['p'][cnt]*ht4.get_ht(X,Q2,'p','F2')/X)
            F2n.append(STF['n'][cnt]*ht4.get_ht(X,Q2,'n','F2')/X)
            plus.append (STF['n'][cnt]*ht4.get_ht(X,Q2,'n','F2')/X + STF['p'][cnt]*ht4.get_ht(X,Q2,'p','F2')/X)
            minus.append(STF['n'][cnt]*ht4.get_ht(X,Q2,'n','F2')/X - STF['p'][cnt]*ht4.get_ht(X,Q2,'p','F2')/X)
        else: pass
        if mode==0:
            hand['p']     ,= ax11.plot(X,F2pHT[cnt] ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax11.plot(X,F2nHT[cnt] ,color='green' ,alpha=0.3)
            hand['p']     ,= ax12.plot(X,F2p[cnt]   ,color='red'   ,alpha=0.3)
            hand['n']     ,= ax12.plot(X,F2n[cnt]   ,color='green' ,alpha=0.3)
            hand['plus']  ,= ax12.plot(X,plus[cnt]  ,color='blue'  ,alpha=0.3)
            hand['minus'] ,= ax12.plot(X,minus[cnt] ,color='purple',alpha=0.3)
        cnt+=1

    if mode == 1:
        meanp = np.mean(np.array(F2pHT),axis=0)
        stdp  = np.std (np.array(F2pHT),axis=0)
        meann = np.mean(np.array(F2nHT),axis=0)
        stdn  = np.std (np.array(F2nHT),axis=0)
        hand['p'] = ax11.fill_between(X,meanp-stdp,meanp+stdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n'] = ax11.fill_between(X,meann-stdn,meann+stdn,color='green',alpha=0.9, zorder=1)
        meanp = np.mean(np.array(F2p),axis=0)
        stdp  = np.std (np.array(F2p),axis=0)
        meann = np.mean(np.array(F2n),axis=0)
        stdn  = np.std (np.array(F2n),axis=0)
        meanplus  = np.mean(np.array(plus),axis=0)
        stdplus   = np.std (np.array(plus),axis=0)
        meanminus = np.mean(np.array(minus),axis=0)
        stdminus  = np.std (np.array(minus),axis=0)
        hand['p']     = ax12.fill_between(X,meanp-stdp,meanp+stdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n']     = ax12.fill_between(X,meann-stdn,meann+stdn,color='green',alpha=0.9, zorder=1)
        hand['plus']  = ax12.fill_between(X,meanplus-stdplus  ,meanplus+stdplus  ,color='blue'  ,alpha=0.9, zorder=1)
        hand['minus'] = ax12.fill_between(X,meanminus-stdminus,meanminus+stdminus,color='purple',alpha=0.9, zorder=1)

    print 
    ##############################################
    h0 =-3.2874
    h1 = 1.9274
    h2 =-2.0701
    ht = h0*X**h1*(1+h2*X)/Q2
    hand['CJ15'] ,= ax11.plot(X,ht,'k--')
   
    ax11.set_ylim(-0.09,0.25)
    ax12.set_ylim(-0.03,0.07)

    ax11.text(0.05,0.32,r'\boldmath$\frac{C_{\rm HT}^N}{Q2}$',transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.05,r'\boldmath$F_2^N \frac{C^N_{\rm HT}}{Q^2}$',transform=ax12.transAxes,size=40)

    if Q2=='1.27**2': ax12.text(0.50,0.10,r'$Q^2 = m_c^2$',transform=ax12.transAxes,size=25)
    else:             ax12.text(0.50,0.10,r'$Q^2 = %3.2f~{\rm GeV}^2$'%Q2,transform=ax12.transAxes,size=25)
    
    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0,0.5)
        ax.set_xlabel(r'\boldmath$x$',size=30)
        ax.xaxis.set_label_coords(0.95,0.00)
        ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0,0.1,0.2,0.3,0.4])

    if conf['tmc']=='GP':  ax11.text(0.05,0.05,r'\textbf{\textrm{GP}}' ,size=25,transform=ax11.transAxes)
    if conf['tmc']=='AOT': ax11.text(0.05,0.05,r'\textbf{\textrm{AOT}}',size=25,transform=ax11.transAxes)

    minorLocator = MultipleLocator(0.025)
    majorLocator = MultipleLocator(0.1)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.02)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)

    handles,labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    handles.append(hand['CJ15'])
    labels.append(r'\boldmath$p$')
    labels.append(r'\boldmath$n$')
    labels.append(r'\textbf{\textrm{CJ15}}')
    ax11.legend(handles,labels,frameon=False,loc=2,fontsize=25, handletextpad = 0.5, handlelength = 1.5, ncol=1, columnspacing = 0.5)

    handles,labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    handles.append(hand['plus'])
    handles.append(hand['minus'])
    labels.append(r'\boldmath$p$')
    labels.append(r'\boldmath$n$')
    labels.append(r'\boldmath$n+p$')
    labels.append(r'\boldmath$n-p$')
    ax12.legend(handles,labels,frameon=False,loc=2,fontsize=25, handletextpad = 0.5, handlelength = 1.5, ncol=1, columnspacing = 0.5)


    #if mode==0:
    #    sm   = py.cm.ScalarMappable(cmap=cmap)
    #    sm.set_array([])
    #    cax = fig.add_axes([0.45,0.88,0.30,0.04])
    #    cax.tick_params(axis='both',which='both',labelsize=15,direction='in')
    #    cax.xaxis.set_label_coords(0.65,-1.8)
    #    cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    #    cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=20)

    py.tight_layout()
    filename = '%s/gallery/ht-Q2=%3.5f'%(wdir,Q2)
    if mode == 1: filename += '-bands'
    filename += '.png'
    print()
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()




    

#--need to update
def plot_pol_ht(wdir,Q2=None,mode=0):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas

    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1) 
    ax12 = py.subplot(nrows,ncols,2) 
    ax21 = py.subplot(nrows,ncols,3) 
    ax22 = py.subplot(nrows,ncols,4) 

    hand = {}
    replicas = core.get_replicas(wdir)

    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep = core.get_istep()
    if 'pol ht4' not in conf['steps'][istep]['active distributions']:
        if 'pol ht4' not in conf['steps'][istep]['passive distributions']:
            return

    core.mod_conf(istep,replicas[0])
    conf['idis grid']  = 'prediction'
    conf['pidis grid'] = 'prediction'
    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman
    cluster,colors,nc,cluster_order= classifier.get_clusters(wdir,istep,kc) 

    replicas = core.get_replicas(wdir)

    conf['datasets']['idis']  = {_:{} for _ in ['xlsx','norm']}
    conf['datasets']['pidis'] = {_:{} for _ in ['xlsx','norm']}
    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']
    resman.setup_idis()
    resman.setup_pidis()
  
    idis  = resman.idis_thy
    pidis = resman.pidis_thy
    idis ._update()
    pidis._update()

    ht4    = conf['pol ht4']
    twist3 = False
    if 't3ppdf' in conf: 
        t3ppdf = conf['t3ppdf']
        twist3 = True

    ##############################################
    #--plot ht4
    ##############################################
    X1=10**np.linspace(-4,-1,100)
    X2=np.linspace(0.1,0.99,100)
    X=np.append(X1,X2)
    cnt = 0
    g1p, g1n = [], []
    for par in replicas:
        lprint('Generating polarized ht4 %s/%s'%(cnt+1,len(replicas))) 
        parman.set_new_params(par)
        g1p.append(X*ht4.get_ht(X,1.0,'p','g1'))
        g1n.append(X*ht4.get_ht(X,1.0,'n','g1'))
        if mode==0:
            ax11.plot(X,g1p[cnt],'r-',alpha=0.5)
            ax12.plot(X,g1n[cnt],'r-',alpha=0.5)
        cnt+=1

    if mode == 1:
        meanp = np.mean(np.array(g1p),axis=0)
        stdp  = np.std (np.array(g1p),axis=0)
        meann = np.mean(np.array(g1n),axis=0)
        stdn  = np.std (np.array(g1n),axis=0)
        hand['p'] = ax11 .fill_between(X,(meanp-stdp),(meanp+stdp),color='red',alpha=0.9)
        hand['n'] = ax12 .fill_between(X,(meann-stdn),(meann+stdn),color='red',alpha=0.9)

    print()
 
    ##############################################
    #--plot twist-3 ppdf
    ##############################################
    if twist3:
        cnt = 0
        u, d = [], []
        for par in replicas:
            lprint('Generating twist-3 ppdf %s/%s'%(cnt+1,len(replicas))) 
            parman.set_new_params(par)
            u.append(np.array([t3ppdf.get_xF(x,Q2,'u',evolve=True) for x in X]))
            d.append(np.array([t3ppdf.get_xF(x,Q2,'d',evolve=True) for x in X]))
            if mode==0:
                ax21 .plot(X,u[cnt],'r-',alpha=0.5)
                ax22 .plot(X,d[cnt],'r-',alpha=0.5)
            cnt+=1

        if mode == 1:
            meanu = np.mean(np.array(u),axis=0)
            stdu  = np.std (np.array(u),axis=0)
            meand = np.mean(np.array(d),axis=0)
            stdd  = np.std (np.array(d),axis=0)
            hand['p'] = ax21.fill_between(X,(meanu-stdu),(meanp+stdu),color='red',alpha=0.9)
            hand['n'] = ax22.fill_between(X,(meand-stdd),(meann+stdd),color='red',alpha=0.9)

    print() 
    ##############################################

    ax11.text(0.05,0.05,r'\boldmath$xH_p$',transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.05,r'\boldmath$xH_n$',transform=ax12.transAxes,size=40)
    ax21.text(0.05,0.05,r'\boldmath$xD_u$',transform=ax21.transAxes,size=40)
    ax22.text(0.05,0.05,r'\boldmath$xD_d$',transform=ax22.transAxes,size=40)

    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='major',length=6)
        ax.tick_params(axis='both',which='minor',length=3)
        ax.set_xlim(0.0,1.0)
        ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticks([0,0.2,0.4,0.6,0.8])

    ax11.set_ylim(-0.035,0.035)
    ax12.set_ylim(-0.035,0.035)
    ax21.set_ylim(-0.18,0.18)
    ax22.set_ylim(-0.18,0.18)

    ax11.tick_params(labelbottom=False)
    ax12.tick_params(labelbottom=False)
    ax12.tick_params(labelleft=False)
    ax22.tick_params(labelleft=False)

    ax21.set_xlabel(r'\boldmath$x$',size=30)
    ax22.set_xlabel(r'\boldmath$x$',size=30)
    ax21.xaxis.set_label_coords(0.90,0.00)
    ax22.xaxis.set_label_coords(0.90,0.00)

    for ax in [ax11,ax12]:
        minorLocator = MultipleLocator(0.002)
        majorLocator = MultipleLocator(0.01)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
    for ax in [ax21,ax22]:
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    #ax11.set_xticks([0,0.2,0.4,0.6,0.8])

    #handles,labels = [],[]
    #handles.append(hand['p'])
    #handles.append(hand['n'])
    #handles.append(hand['CJ15'])
    #labels.append(r'\textbf{\textrm{JAM (p)}}')
    #labels.append(r'\textbf{\textrm{JAM (n)}}')
    #labels.append(r'\textbf{\textrm{CJ15}}')

    #ax11.legend(handles,labels,frameon=False,loc=2,fontsize=25, handletextpad = 0.5, handlelength = 1.5)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)
    filename = '%s/gallery/pol-ht-Q2=%3.5f'%(wdir,Q2)
    if mode == 1: filename += '-bands'
    filename += '.png'
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()





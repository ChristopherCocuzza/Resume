#!/usr/bin/env python
from tools.config import load_config,conf
from fitlib.resman import RESMAN
import numpy as np

#--matplotlib
import matplotlib
import pylab  as py
from matplotlib.ticker import MultipleLocator

#--from tools
from tools.tools import load,lprint,save

#--from corelib
from analysis.corelib import core,classifier

import kmeanconf as kc

#--generate offshell corrections
def gen_off(wdir,Q2=None):

    replicas = core.get_replicas(wdir)

    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']

    istep = core.get_istep()
    core.mod_conf(istep,replicas[0])
    #conf['idis grid'] = 'prediction'
    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman

    replicas = core.get_replicas(wdir)
    names    = core.get_replicas_names(wdir)

    Xgrid = np.geomspace(1e-5,1e-1,20)
    Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    Q2grid = [Q2*0.99,Q2*1.01]
    conf['idis grid'] = {}
    conf['idis grid']['X']  = Xgrid 
    conf['idis grid']['Q2'] = Q2grid 
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']
    resman.setup_idis()
    
    idis=resman.idis_thy
    idis.data['p']['F2'] = np.array(idis.X.size)
    idis.data['n']['F2'] = np.array(idis.X.size)
    
    if 'off' in conf: off_type,off = 'stf',conf['off']
    elif 'offpdf' in conf: off_type,off = 'pdf',conf['offpdf']
    else:
        print('Offshell corrections not present.')
        return

    if 'off' not in conf['steps'][istep]['active distributions']: 
        if 'off' not in conf['steps'][istep]['passive distributions']: 
            if 'off pdf' not in conf['steps'][istep]['active distributions']: 
                if 'off pdf' not in conf['steps'][istep]['passive distributions']: 
                    return

    ##############################################
    #--generate offshell
    ##############################################
    X1   = 10**np.linspace(-4,-1,100)
    X2   = np.linspace(0.1,0.98,100)
    X    = np.append(X1,X2)
    Q2   = np.ones(X.size)*Q2
    cnt = 0

    OFF = {}
    OFF['dfD'] = []
    OFF['X'] = X
    for par in replicas:
         
        lprint('Generating offshell %s/%s'%(cnt+1,len(replicas)))
        parman.set_new_params(par)
        idis._update()
        F2p  = idis.get_stf(X,Q2,stf='F2',tar='p')
        F2n  = idis.get_stf(X,Q2,stf='F2',tar='n')
        #--I think this is outdated
        if off_type=='stf':
            pof2 = off.get_offshell(X,Q2,'p','F2',tar='h')
            nof2 = off.get_offshell(X,Q2,'n','F2',tar='h')
            dfD  = (F2p*pof2+F2n*nof2)/(F2p+F2n)
            OFF['dfD'].append(dfD)    
        if off_type=='pdf':
            F2poff  = idis.get_stf(X,Q2,stf='F2',tar='p',off=True,nucleus='d')
            F2noff  = idis.get_stf(X,Q2,stf='F2',tar='n',off=True,nucleus='d')
            dfD  = (F2poff+F2noff)/(F2p+F2n)
            OFF['dfD'].append(dfD)    
        cnt +=1
     
    filename ='%s/data/off-Q2=%3.5f.dat'%(wdir,Q2[0])
    save(OFF,filename) 
    print()
    print('Saving offshell data to %s'%filename)

def plot_off(wdir,Q2=None,mode=0):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)

    j = 0
    hand = {}
    thy  = {}
    replicas = core.get_replicas(wdir)

    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']

    istep = core.get_istep()
    core.mod_conf(istep,replicas[0])
    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman
    cluster,colors,nc,cluster_order= classifier.get_clusters(wdir,istep,kc) 

    replicas = core.get_replicas(wdir)
    names    = core.get_replicas_names(wdir)

    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']
    
    if 'off' in conf: off = conf['off']
    elif 'offpdf' in conf: off = conf['offpdf']
    else:
        print('Offshell corrections not present.')
        return

    if 'off' not in conf['steps'][istep]['active distributions']: 
        if 'off' not in conf['steps'][istep]['passive distributions']: 
            if 'off pdf' not in conf['steps'][istep]['active distributions']: 
                if 'off pdf' not in conf['steps'][istep]['passive distributions']: 
                    return

    #--try to load data.  Generate it if it does not exist
    filename ='%s/data/off-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data = load(filename)
    except:
        gen_off(wdir,Q2)
        data = load(filename)
    ##############################################
    #--plot offshell
    ##############################################
    X   = data['X']

    dfD = data['dfD']
    if mode==0:
        for i in range(len(dfD)):        
            thy[j]    ,= ax11.plot(X,dfD[i],color='firebrick',alpha=0.2)
    if mode == 1:
       meanD = np.mean(np.array(dfD),axis=0)
       stdD  = np.std (np.array(dfD),axis=0)
       thy[j]    = ax11.fill_between(X,meanD-stdD,meanD+stdD,color='red'      ,alpha=0.7,zorder=1)

 
    ##############################################
    X1   = 10**np.linspace(-4,-1,100)
    X2   = np.linspace(0.1,0.98,100)
    X    = np.append(X1,X2)
    #--CJ15 
    C =-3.6735
    x0= 5.7717e-2
    x1=0.36419
    dfcj=C*(X-x0)*(X-x1)*(1+x0-X)
    hand['CJ'] ,= ax11.plot(X,dfcj,'b--')
    #--KP 
    C = 8.10
    x0= 0.448
    x1= 0.05
    dfcj=C*(X-x0)*(X-x1)*(1+x0-X)
    hand['KP'] ,= ax11.plot(X,dfcj,'g--')

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    ax11.text(0.30,0.85,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=30,transform=ax11.transAxes)

    ax11.text(0.05,0.85,r'\boldmath$\delta f^D$',transform=ax11.transAxes,size=40)
 
    for ax in [ax11]:
        ax.set_ylim(-2.2,1.2)
        ax.set_xlim(0,0.90)
        ax.axhline(0,alpha=0.5,color='k',ls='--')
        ax.set_xlabel(r'\boldmath$x$'         ,size=30)
        ax.xaxis.set_label_coords(0.98,0.00)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])


    handles,labels=[],[]

    handles.append(hand['CJ'])
    handles.append(hand['KP'])
    labels.append(r'\textbf{\textrm{CJ15}}')
    labels.append(r'\textbf{\textrm{KP}}')

    ax11.legend(handles,labels,frameon=False,loc='lower left',fontsize=28, handletextpad = 0.5, handlelength = 1.5, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(wspace=0)

    filename = '%s/gallery/off-Q2=%3.5f'%(wdir,Q2)
    if mode == 1: filename += '-bands'
    filename += '.png'
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()









#--polarized offshell (very outdated) 
def plot_pol_off(PLOT,kc,mode=0,iso=False,name='',nrep=None,labelQ2=True,ncol=1,_ax=None):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas

    if _ax==None:
        nrows,ncols=1,1
        fig = py.figure(figsize=(ncols*8,nrows*5))
        ax11 = py.subplot(nrows,ncols,1)
    else: ax11 = _ax

    j = 0
    hand = {}
    thy  = {}
    for plot in PLOT: 
        wdir, q2, color, style, label, alpha, zorder= plot[0], plot[1], plot[2], plot[3], plot[4], plot[5], plot[6]
        replicas = core.get_replicas(wdir)

        load_config('%s/input.py'%wdir)

        istep = core.get_istep()

        if 'pol off' in conf: off = conf['pol off']
        else:
            print('Polarized offshell corrections not present.')
            return

        core.mod_conf(istep,replicas[0])
        conf['idis grid'] = 'prediction'
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
        
        idis=resman.idis_thy
        pidis=resman.pidis_thy
        idis._update()
        pidis._update()
        

        ##############################################
        #--plot offshell
        ##############################################
        X   = np.linspace(0.0,0.98,100)
        Q2   = np.ones(X.size)*q2
        cnt = 0
        df = []
        pof1,nof1 = [],[]
        for par in replicas:
           
            if nrep != None and cnt >= nrep: break 
            #if len(PLOT) == 1: color = colors[cluster[cnt]]
    
            lprint('Generating polarized offshell %s/%s'%(cnt+1,len(replicas))) 
            parman.set_new_params(par,initial=True) 
            #--isospin symmetry
            #g1p  = pidis.get_stf(X,Q2,stf='g1',tar='p')
            #g1n  = pidis.get_stf(X,Q2,stf='g1',tar='n')
            pof1.append(off.get_offshell(X,Q2,'p','g1'))
            nof1.append(off.get_offshell(X,Q2,'n','g1'))
            #df.append((g1p*pof2+g1n*nof2)/(g1p+g1n))

            if mode==0: 
                hand['p'] ,= ax11.plot(X,pof1[cnt],color='firebrick',alpha=1.0)
                hand['n'] ,= ax11.plot(X,nof1[cnt],color='darkgreen',alpha=1.0)
            cnt += 1
   
        if mode == 1:
            meanp = np.mean(np.array(pof1),axis=0)
            stdp  = np.std(np.array(pof1),axis=0)
            meann = np.mean(np.array(nof1),axis=0)
            stdn  = np.std(np.array(nof1),axis=0)
            hand['p'] = ax11.fill_between(X,meanp-stdp,meanp+stdp,color='firebrick',alpha=alpha,zorder=zorder)
            hand['n'] = ax11.fill_between(X,meann-stdn,meann+stdn,color='darkgreen',alpha=alpha,zorder=zorder)

        j+=1
        print
 
    ##############################################

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    if labelQ2: ax11.text(0.60,0.05,r'$Q^2=%s{\rm~GeV^2}$'%q2,size=30,transform=ax11.transAxes)

    ax11.set_ylim(-5.5,5.5)
    ax11.set_xlim(0,1)
    ax11.text(0.05,0.05,r'\boldmath$\delta f^N$',transform=ax11.transAxes,size=40)
    ax11.set_xlabel(r'\boldmath$x$'         ,size=30)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.axhline(0,alpha=0.5,color='k',ls='--')

    for ax in [ax11]:
        minorLocator = MultipleLocator(0.4)
        majorLocator = MultipleLocator(2.0)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0,0.2,0.4,0.6,0.8])


    #handles = [thy[i] for i in range(len(PLOT)) if PLOT[i][4]!=None]
    #labels  = [PLOT[i][4] for i in range(len(PLOT)) if PLOT[i][4]!=None]
    handles, labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    labels.append(r'\textrm{\textbf{JAM p}}')
    labels.append(r'\textrm{\textbf{JAM n}}')

    ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.5, ncol = ncol, columnspacing = 0.5)

    py.tight_layout()

    if _ax==None:
        filename = '%s/gallery/pol-off'%PLOT[0][0]
        if mode == 1: filename += '-bands'
        filename += name
        filename += '.png'
        print('Saving figures to %s'%filename)
        py.savefig(filename)
        py.clf()
    else:
        return ax11 














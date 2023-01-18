import sys,os
import numpy as np

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py

#--from scipy stack 
from scipy.integrate import quad

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.qpdlib.qpdcalc import QPDCALC
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

FLAV=[]
FLAV.append('uv')
FLAV.append('dv')

def gen_xf(wdir,Q2=None):
    
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    if Q2==None: Q2 = conf['Q20']
    print('\ngenerating offshell pdf at Q2 = %s from %s'%(Q2,wdir))
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'off pdf' not in conf['steps'][istep]['active distributions']:
        if 'off pdf' not in conf['steps'][istep]['passive distributions']:
                print('off pdf is not an active or passive distribution')
                return 

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['off pdf']
    #--setup kinematics
    X=10**np.linspace(-6,-1,200)
    X=np.append(X,np.linspace(0.1,0.99,200))

    pdf.evolve(Q2)

    #--compute XF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in XF:  XF[flav]=[]
            if flav=='uv':
                 func=lambda x: pdf.get_xF(x,Q2,'uv')
            elif flav=='dv':
                 func=lambda x: pdf.get_xF(x,Q2,'dv')
            elif flav=='d/u':
                 func=lambda x: pdf.get_xF(x,Q2,'d')/pdf.get_xF(x,Q2,'u')
            elif flav=='db+ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') + pdf.get_xF(x,Q2,'ub')
            elif flav=='db-ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') - pdf.get_xF(x,Q2,'ub')
            elif flav=='db/ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') / pdf.get_xF(x,Q2,'ub')
            elif flav=='s+sb':
                 func=lambda x: pdf.get_xF(x,Q2,'s') + pdf.get_xF(x,Q2,'sb')
            elif flav=='s-sb':
                 func=lambda x: pdf.get_xF(x,Q2,'s') - pdf.get_xF(x,Q2,'sb')
            elif flav=='Rs':
                 func=lambda x: (pdf.get_xF(x,Q2,'s') + pdf.get_xF(x,Q2,'sb'))\
                                /(pdf.get_xF(x,Q2,'db') + pdf.get_xF(x,Q2,'ub'))
            else:
                 func=lambda x: pdf.get_xF(x,Q2,flav) 

            XF[flav].append(np.array([func(x) for x in X]))

    print() 
    checkdir('%s/data'%wdir)
    filename='%s/data/off-pdf-Q2=%3.5f.dat'%(wdir,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_xf(wdir,Q2=None,mode=0):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=3,2
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax21=py.subplot(nrows,ncols,3)
    ax22=py.subplot(nrows,ncols,4)
    ax31=py.subplot(nrows,ncols,5)
    ax32=py.subplot(nrows,ncols,6)

    cmap = matplotlib.cm.get_cmap('hot')

    hand = {}
    thy  = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if Q2==None: Q2 = conf['Q20']

    #idx = [10041,10051]
    idx = [10041,10050,10051]

    scale = classifier.get_scale(wdir,'idis',idx)

    if 'off pdf' not in conf['steps'][istep]['active distributions']:
        if 'off pdf' not in conf['steps'][istep]['passive distributions']:
                print('off pdf is not an active or passive distribution')
                return 

    #--load data if it exists
    filename='%s/data/off-pdf-Q2=%3.5f.dat'%(wdir,Q2)
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,Q2)
        
    #--load pdf data
    filename='%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2)
    pdfs=load(filename)

    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]

    X=data['X']

    for flav in data['XF']:
        mean = np.mean(data['XF'][flav],axis=0)
        std = np.std(data['XF'][flav],axis=0)

        if flav=='uv': ax = ax11
        elif flav=='dv': ax = ax12
        else: continue
        if flav=='uv':      color = 'firebrick'
        elif flav=='dv':    color = 'darkgreen'

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF'][flav])):
                hand[flav] ,= ax.plot(X,data['XF'][flav][i],color=cmap(scale[i]),alpha=0.1)
 
        #--plot average and standard deviation
        if mode==1:
            hand[flav]  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=0.8,zorder=5)

    uv = np.array(data['XF']['uv'])
    dv = np.array(data['XF']['dv'])
    p  = uv + dv
    m  = uv - dv

    meanp = np.mean(p,axis=0)
    stdp  = np.std (p,axis=0)
    meanm = np.mean(m,axis=0)
    stdm  = np.std (m,axis=0)

    #--plot each replica
    if mode==0:
        for i in range(len(data['XF'][flav])):
            hand[flav] ,= ax21.plot(X,p[i],color=cmap(scale[i]),alpha=0.1)
            hand[flav] ,= ax22.plot(X,m[i],color=cmap(scale[i]),alpha=0.1)
 
    #--plot average and standard deviation
    if mode==1:
        hand[flav]  = ax21.fill_between(X,meanp-stdp,meanp+stdp,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax22.fill_between(X,meanm-stdm,meanm+stdm,color='red',alpha=0.8,zorder=5)

    uvon = np.array(pdfs['XF']['uv'])
    dvon = np.array(pdfs['XF']['dv'])
    ratu = uv/uvon
    ratd = dv/dvon

    meanu = np.mean(ratu,axis=0)
    stdu  = np.std (ratu,axis=0)
    meand = np.mean(ratd,axis=0)
    stdd  = np.std (ratd,axis=0)

    #--plot each replica
    if mode==0:
        for i in range(len(data['XF'][flav])):
            hand[flav] ,= ax31.plot(X,ratu[i],color=cmap(scale[i]),alpha=0.1)
            hand[flav] ,= ax32.plot(X,ratd[i],color=cmap(scale[i]),alpha=0.1)
 
    #--plot average and standard deviation
    if mode==1:
        hand[flav]  = ax31.fill_between(X,meanu-stdu,meanu+stdu,color='red',alpha=0.8,zorder=5)
        hand[flav]  = ax32.fill_between(X,meand-stdd,meand+stdd,color='red',alpha=0.8,zorder=5)


    for ax in [ax11,ax12,ax21,ax22,ax31,ax32]:
          ax.set_xlim(0,0.9)
          ax.axhline(0.0,ls='--',color='black',alpha=0.5)
          minorLocator = MultipleLocator(0.04)
          majorLocator = MultipleLocator(0.2)
          ax.xaxis.set_minor_locator(minorLocator)
          ax.xaxis.set_major_locator(majorLocator)
          ax.set_xticks([0.2,0.4,0.6,0.8])
            
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)

    if mode==0:
        ax11.set_ylim(-0.95,0.95)
        ax12.set_ylim(-0.95,0.95)
        ax21.set_ylim(-0.95,0.95)
        ax22.set_ylim(-0.95,0.95)
        ax31.set_ylim(-1.10,1.10)
        ax32.set_ylim(-1.10,1.10)
    if mode==1:
        ax11.set_ylim(-0.25,0.25)
        ax12.set_ylim(-0.25,0.25)
        ax21.set_ylim(-0.35,0.35)
        ax22.set_ylim(-0.35,0.35)
        ax31.set_ylim(-1.10,1.10)
        ax32.set_ylim(-1.10,1.10)


    for ax in [ax31,ax32]:
        ax.set_xlabel(r'\boldmath$x$' ,size=30)
        ax.xaxis.set_label_coords(0.98,0.00)

    if Q2 == 1.27**2: ax11.text(0.65,0.85,r'$Q^2 = m_c^2$'                                  , transform=ax11.transAxes,size=30)
    else:             ax11.text(0.65,0.85,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=30)


    ax11.text(0.05,0.07,r'\boldmath$x \delta u_v$'     ,transform=ax11.transAxes,size=30)
    ax12.text(0.05,0.07,r'\boldmath$x \delta d_v$'     ,transform=ax12.transAxes,size=30)
    ax21.text(0.05,0.07,r'\boldmath$x (\delta u_v + \delta d_v)$' ,transform=ax21.transAxes,size=30)
    ax22.text(0.05,0.07,r'\boldmath$x (\delta u_v - \delta d_v)$' ,transform=ax22.transAxes,size=30)
    ax31.text(0.05,0.07,r'\boldmath$  \delta u_v/u_v$'     ,transform=ax31.transAxes,size=30)
    ax32.text(0.05,0.07,r'\boldmath$  \delta d_v/d_v$'     ,transform=ax32.transAxes,size=30)

    sm   = py.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.72,0.92,0.25,0.05])
    cax.tick_params(axis='both',which='both',labelsize=20,direction='in')
    cax.xaxis.set_label_coords(0.65,-0.5)
    cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=30)


    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.20)

    filename = '%s/gallery/off-pdfs-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'
    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)







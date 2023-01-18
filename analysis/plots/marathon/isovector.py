#!/usr/bin/env python
import os,sys
#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
import pylab  as py


from tools.config import load_config,conf
from fitlib.resman import RESMAN
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--from tools
from tools.tools import checkdir,load,lprint,save

import kmeanconf as kc

#--from corelib
from analysis.corelib import core,classifier

cwd = 'plots/marathon/'

if __name__=="__main__":

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)

    wdir1 = 'results/marathon/final'
    #wdir1 = 'results/marathon/step30'
    #wdir1 = 'results/marathon/wf/av18'
    #wdir1 = 'results/marathon/wf/cdbonn'
    #wdir1 = 'results/marathon/wf/wjc1'
    #wdir1 = 'results/marathon/wf/wjc2'
    #wdir1 = 'results/marathon/wf/ss'
    #wdir1 = 'results/marathon/more/hanjie'
    #wdir1 = 'results/marathon/nomar3'
    #wdir1 = 'results/marathon/GP'
    wdir1 = 'final2'

    WDIR = [wdir1]

    Q2 = 10

    j = 0
    hand = {}
    thy  = {}
    for wdir in WDIR: 
        replicas = core.get_replicas(wdir)

        load_config('%s/input.py'%wdir)

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
        
        ##############################################
        #--plot off-shell PDF ratios
        ##############################################
        data=load('%s/data/off-pdf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))
        pdfs=load('%s/data/pdf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))

        X    = np.array(data['X'])
        uv   = np.array(data['XF']['uv'])
        dv   = np.array(data['XF']['dv'])
        uvon = np.array(pdfs['XF']['uv'])
        dvon = np.array(pdfs['XF']['dv'])
        ratu = uv/uvon
        ratd = dv/dvon

        meanu = np.mean(ratu,axis=0)
        stdu  = np.std (ratu,axis=0)
        meand = np.mean(ratd,axis=0)
        stdd  = np.std (ratd,axis=0)

        ax11.plot(X,meanu,color='darkred',alpha=1.0,zorder=2,lw=2)
        ax11.plot(X,meand,color='blue',alpha=1.0,zorder=1,lw=2)
        ax11.fill_between(X,meanu-stdu,meanu+stdu,color='red'  ,alpha=0.7,zorder=2)
        ax11.fill_between(X,meand-stdd,meand+stdd,color='blue',alpha=0.4,zorder=1)

        j+=1
        ##############################################
        #--plot nuclear PDF ratios
        ##############################################
        data = load('%s/data/nuclear-pdf-Q2=%d.dat'%(wdir,Q2))
        X   = np.array(data['X'])

        upT = np.array(data['up']['p']['t']['total'])
        upH = np.array(data['up']['p']['h']['total'])
        dpT = np.array(data['do']['p']['t']['total'])
        dpH = np.array(data['do']['p']['h']['total'])

        urat = (upT - upH)/(upT + upH) 
        drat = (dpT - dpH)/(dpT + dpH) 

        meanu = np.mean(urat,axis=0)
        stdu  = np.std (urat,axis=0)
        meand = np.mean(drat,axis=0)
        stdd  = np.std (drat,axis=0)
      
        idx = np.nonzero(X < 0.75)

        hand['u']    ,= ax12.plot(X,meanu,color='darkred'  ,alpha=1.0,zorder=2,lw=2)
        hand['d']    ,= ax12.plot(X[idx],meand[idx],color='blue',alpha=1.0,zorder=1,lw=2)
        #hand['uband'] = ax12.fill_between(X,meanu-stdu,meanu+stdu,color='red'  ,alpha=0.7,zorder=2)
        #hand['dband'] = ax12.fill_between(X,meand-stdd,meand+stdd,color='blue' ,alpha=0.4,zorder=1)
        hand['uband'] = ax12.fill_between(X,meanu-stdu,meanu+stdu,color='red'  ,alpha=0.7,zorder=2)
        hand['dband'] = ax12.fill_between(X,meand-stdd,meand+stdd,color='blue' ,alpha=0.4,zorder=1)
  
        j+=1
 
    ##############################################

    ax11.text(0.05,0.60,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=30,transform=ax11.transAxes)

    ax11.text(0.05,0.85,r'\boldmath$\delta q / q$'                                                                  ,transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.78,r'\boldmath$\Delta_3^q$' ,transform=ax12.transAxes,size=50)
 
    for ax in [ax11,ax12]:
        ax.set_xlim(0.15,0.85)
        ax.set_xlabel(r'\boldmath$x$',size=50)
        ax.xaxis.set_label_coords(0.50,-0.05)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticks([0.2,0.4,0.6,0.8])

    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.axhline(0,0,1,color='black',ls=':',alpha=0.5)

    ax11.set_ylim(-1.20,7.50)
    ax12.set_ylim(-0.12,0.12)

    minorLocator = MultipleLocator(0.5)
    majorLocator = MultipleLocator(2.0)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.05)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)

    handles,labels = [],[]
    handles.append((hand['uband'], hand['u']))
    handles.append((hand['dband'], hand['d']))
    labels.append(r'\boldmath$u$')
    labels.append(r'\boldmath$d$')
    ax12.legend(handles,labels,frameon=False,loc=(0.01,0.00),fontsize=32, handletextpad = 0.5, handlelength = 1.2, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    #py.subplots_adjust(hspace=0,wspace=0)


    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/isovector'%cwd
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    print('Saving figure to %s'%filename)
    py.savefig(filename)
    py.clf()
















#!/usr/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
import numpy as np
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--matplotlib
from matplotlib.legend_handler import HandlerBase
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

cwd = 'plots/seaquest'

def plot_impact(WDIR,kc,separate=False):

    nrows,ncols=5,1
    fig = py.figure(figsize=(ncols*12,nrows*3))
    ax11=py.subplot(nrows,ncols,(1,2))
    ax21=py.subplot(nrows,ncols,(3,4))
    ax31=py.subplot(nrows,ncols,5)

    filename = '%s/gallery/impact'%cwd

    j = 0
    hand = {}
    thy = {}
    STD = {}
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        Q2 = 10
        data=load('%s/data/pdf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))
            
        replicas=core.get_replicas(wdir)
        cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
        best_cluster=cluster_order[0]


        for flav in data['XF']:
            X=data['X']
            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            if flav=='db/ub':   ax = ax11
            elif flav=='db-ub': ax = ax21
            else: continue

            #--plot average and standard deviation
            #if j == 0: 
            #    #thy[j]  = ax.fill_between(X,mean-std,mean+std,color='gold',alpha=0.3,zorder=4)
            #    thy[j]  ,= ax.plot(X,mean+std,color='lightgreen',alpha=0.9,ls = '--',zorder=4)
            #    ax           .plot(X,mean-std,color='lightgreen',alpha=0.9,ls = '--',zorder=4)
            if j == 0: thy[j]  = ax.fill_between(X,mean-std,mean+std,color='lightgreen',alpha=0.9,zorder=4)
            if j == 1: thy[j]  = ax.fill_between(X,mean-std,mean+std,color='blue'      ,alpha=0.7,zorder=5)
            if j == 2: thy[j]  = ax.fill_between(X,mean-std,mean+std,color='red'       ,alpha=1.0,zorder=6)

            if flav=='db/ub': STD[j] = std

        j+=1


    for ax in [ax11,ax21,ax31]:
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=40,length=5)
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=40,length=10)
        ax.set_xlim(0.02,0.42)
        ax.set_xticks([0.1,0.2,0.3,0.4])
        minorLocator = MultipleLocator(0.02)
        ax.xaxis.set_minor_locator(minorLocator)

    ax11.tick_params(labelbottom=False)
    ax21.tick_params(labelbottom=False)

    ax11.set_ylim(0,1.7)
    ax11.set_yticks([0.5,1.0,1.5])

    ax21.set_ylim(-0.005,0.048)
    ax21.set_yticks([0.00,0.02,0.04])
    ax21.set_yticklabels([r'$0$',r'$0.02$',r'$0.04$'])

    minorLocator = MultipleLocator(0.1)
    ax11.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.01)
    ax21.yaxis.set_minor_locator(minorLocator)

    ax11.text(0.45,0.10,r'\boldmath$\bar{d}/\bar{u}$'    ,transform=ax11.transAxes,size=70)
    ax21.text(0.57,0.80,r'\boldmath$x(\bar{d}-\bar{u})$' ,transform=ax21.transAxes,size=70)

    ax21.text(0.03,0.14,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax21.transAxes,size=40)

    ax11.axhline(1.0,0,1,ls=':',color='black',alpha=0.5,zorder=5)
    ax21.axhline(0.0,0,1,ls=':',color='black',alpha=0.5,zorder=5)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    handles,labels = [],[]
    #handles.append(('darkgreen',0.9,'--'))
    handles.append(thy[0])
    handles.append(thy[1])
    handles.append(thy[2])
    labels.append(r'\textrm{\textbf{baseline}}')
    labels.append(r'\textrm{\textbf{+STAR}}')
    labels.append(r'\textrm{\textbf{+SeaQuest}}')

    ax11.legend(handles,labels,loc=(0.00,0.02),fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 0.95, ncol = 1, columnspacing = 0.5, handler_map = {tuple:AnyObjectHandler()})

    #--plot ratios
    hand = {}
    X=data['X']
    hand['RHIC'] ,= ax31.plot(X,(STD[1]/STD[0]),color='blue',alpha=0.7,lw=3)
    hand['SQ']   ,= ax31.plot(X,(STD[2]/STD[0]),color='red' ,alpha=1.0,lw=3)

    ax31.text(0.05,0.12,r'\boldmath$\delta/\delta_{\rm baseline}$' ,transform=ax31.transAxes,size=70)
    #ax31.text(0.60,0.08,r'\boldmath$(\bar{d}/\bar{u})$'            ,transform=ax31.transAxes,size=30)

    ax31.axhline(1.0,0,1,ls=':',color='black',alpha=0.5)

    ax31.set_xlabel(r'\boldmath$x$'    ,size=60)
    ax31.xaxis.set_label_coords(0.82,0.00)

    ax31.set_ylim(0,1.2)
    ax31.set_yticks([0.50,1.00])

    minorLocator = MultipleLocator(0.1)
    ax31.yaxis.set_minor_locator(minorLocator)

    handles,labels = [],[]
    handles.append(hand['RHIC'])
    handles.append(hand['SQ'])
    labels.append(r'\textrm{\textbf{+STAR}}')
    labels.append(r'\textrm{\textbf{+SeaQuest}}')

    #ax31.legend(handles,labels,loc=(0.00,-0.02),fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()

    py.subplots_adjust(hspace=0,wspace=0)

    if separate: filename += '-separate'
    #filename+='.png'
    filename+='.pdf'
    ax11.set_rasterized(True)
    ax21.set_rasterized(True)
    ax31.set_rasterized(True)
    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

if __name__ == "__main__":

    #--plot iteratively adding RHIC and SQ
    wdir0 = 'results/upol/step13'
    wdir1 = 'results/upol/step14'
    wdir2 = 'results/sq/final'

    WDIR = [wdir0,wdir1,wdir2]

    plot_impact(WDIR,kc)


    #--plot adding RHIC and SQ separately
    wdir0 = 'results/upol/step13'
    wdir1 = 'results/upol/step14'
    wdir2 = 'results/sq/noRHIC'

    WDIR = [wdir0,wdir1,wdir2]

    plot_impact(WDIR,kc,separate=True)




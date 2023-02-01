#!/usr/bin/env python
import sys, os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
import scipy as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

## matplotlib
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
from matplotlib.ticker import MultipleLocator, FormatStrFormatter ## for minor ticks in x label
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import pylab as py

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from qcdlib
from qcdlib import aux

## from fitpack fitlib
from fitlib.resman import RESMAN

## from obslib
from obslib.dy.reader import READER

## from analysis
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

cwd = 'plots/seaquest'

if __name__ == "__main__":

    wdir = 'results/sq/final'

    print('\ngenerating DY from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    nrows,ncols=3,1
    fig = py.figure(figsize=(ncols*7,nrows*2.5))
    ax11 = py.subplot(nrows,ncols,(1,2))
    ax21 = py.subplot(nrows,ncols,3)

    conf['path2wzrvtab'] = '%s/grids/grids-dy'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['dy']={}
    conf['datasets']['dy']['xlsx']={}
    conf['datasets']['dy']['xlsx'][20001]='dy/expdata/20001.xlsx'
    conf['datasets']['dy']['xlsx'][20002]='dy/expdata/20002.xlsx'
    conf['datasets']['dy']['norm']={}
    conf['datasets']['dy']['filters']=[]
    conf['dy tabs']=READER().load_data_sets('dy')
    tabs = conf['dy tabs']

    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
  
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['dy']
  
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in tabs:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic, axis = 0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic, axis=0)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    thy_plot = {}
    thy_band = {}
    #--plot data
    for idx in tabs:
        xt     = tabs[idx]['xt']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==20001: color,marker,ms,cs,alp = 'darkblue' ,'^',5 ,2.0,0.8
        if idx==20002: color,marker,ms,cs,alp = 'darkred'  ,'.',10,3.0,1.0
        hand[idx] = ax11.errorbar(xt,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=ms,capsize=cs,alpha=alp,zorder=5)

    #--plot mean and std of all replicas
    for idx in tabs:
        for ic in range(nc):
            if idx==20001: color,alp = 'blue' ,0.2
            if idx==20002: color,alp = 'red'  ,0.5
            xt  = tabs[idx]['xt']
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]
            down = thy - std
            up   = thy + std

            thy_plot[idx] ,= ax11.plot(xt,thy,color=color,alpha=0.5)
            thy_band[idx]  = ax11.fill_between(xt,down,up,color=color,alpha=alp)
    
    #######################
    #--plot data/theory
    #######################

    for idx in tabs:
        for ic in range(nc):
            xt = conf['dy tabs'][idx]['xt']
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            ratio = data[idx]['value']/thy
            alpha = data[idx]['alpha']
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]
            if idx==20001: color,marker,ms,cs,alp = 'darkblue','^',5 ,2.0,0.8
            if idx==20002: color,marker,ms,cs,alp = 'darkred'  ,'.',10,3.0,1.0
            ax21.errorbar(xt,ratio,yerr=alpha/thy,color=color,fmt=marker,ms=ms,capsize=cs,alpha=alp,zorder=5)
            if idx==20001: color,alp = 'blue' ,0.2
            if idx==20002: color,alp = 'red'  ,0.5
            ax21.fill_between(xt,1-std/thy,1+std/thy,color=color,alpha=alp)


    for ax in [ax11]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30,labelbottom=False)
    
    for ax in [ax21]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    handles, labels = [],[]
    handles.append(hand[20002])
    handles.append(hand[20001])
    handles.append((thy_band[20002],thy_plot[20002]))
    handles.append((thy_band[20001],thy_plot[20001]))
    labels.append(r'\textbf{\textrm{SeaQuest}}')
    labels.append(r'\textbf{\textrm{NuSea}}')
    labels.append(r'\textbf{\textrm{JAM (SeaQuest)}}')
    labels.append(r'\textbf{\textrm{JAM (NuSea)}}')
    ax11.legend(handles,labels,frameon=False,fontsize=22,loc='upper left', ncol = 2, handletextpad = 0.3, handlelength = 0.9, columnspacing = 0.8)

    ax21.set_xlabel(r'\boldmath$x_2$',size=40)
    ax21.xaxis.set_label_coords(0.75,-0.02)

    ax11.set_ylim(0.59,1.60)
    ax21.set_ylim(0.75,1.25)

    ax11.axhline(1,0,1,color='black',ls='--',alpha=0.5)
    ax21.axhline(1,0,1,color='black',ls='--')
   
    ax11.text(0.05,0.08,r'\boldmath$\sigma_{pD}^{\rm DY}/2\sigma_{pp}^{\rm DY}$' ,transform = ax11.transAxes,size=45)
    ax21.text(0.05,0.75,r'\textbf{\textrm{data/theory}}'                         ,transform = ax21.transAxes,size=35)


    #ax11.text(0.06,0.60,r'$4.7< \langle M_{\ell\ell} \rangle <6.4$'                ,transform = ax11.transAxes,size=16, color='darkred')
    #ax11.text(0.02,0.32,r'$4.6< \langle M_{\ell\ell} \rangle <12.9$ \textrm{GeV}'  ,transform = ax11.transAxes,size=16, color='darkblue')
    ax11.text(0.06,0.60,r'$0.48 < x_1 < 0.69$'  ,transform = ax11.transAxes,size=16, color='darkred')
    ax11.text(0.06,0.32,r'$0.36 < x_1 < 0.56$'  ,transform = ax11.transAxes,size=16, color='darkblue')


    for ax in [ax11,ax21]:
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticks([0.0,0.2,0.4])
        ax.set_xlim(0.00,0.42)

    for ax in [ax11]:
        majorLocator = MultipleLocator(0.2)
        minorLocator = MultipleLocator(0.05)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    for ax in [ax21]:
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    ax11.set_yticks([0.8,1,1.2,1.4])
    ax21.set_yticks([0.8,1.0,1.2])

    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace=0)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/SQ'%cwd
    #filename+='.png'
    filename+='.pdf'
    ax11.set_rasterized(True)
    ax21.set_rasterized(True)


    py.savefig(filename)
    print()
    print('Saving DY plot to %s'%filename)




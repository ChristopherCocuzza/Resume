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

cwd = 'plots/thesis'

def plot_ratios(wdir):

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
    filename = '%s/gallery/DY-ratios'%cwd
    filename+='.png'
    #filename+='.pdf'
    ax11.set_rasterized(True)
    ax21.set_rasterized(True)


    py.savefig(filename)
    print()
    print('Saving DY ratios plot to %s'%filename)

def plot_xsec(wdir):

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

    predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
    labels  = load('%s/data/labels-%d.dat' % (wdir, istep))
    cluster = labels['cluster']

    data = predictions['reactions']['dy']

    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        for ic in range(kc.nc[istep]):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d' % ic] = np.mean(predictions_ic, axis = 0)
            data[idx]['dthy-%d' % ic] = np.std(predictions_ic, axis = 0)

    Q2_bins = []
    Q2_bins.append([15, 20])
    Q2_bins.append([20, 24])
    Q2_bins.append([24, 28])
    Q2_bins.append([28, 35])
    Q2_bins.append([35, 40])  
    Q2_bins.append([40, 45])  
    Q2_bins.append([45, 52])  
    Q2_bins.append([52, 60])  
    Q2_bins.append([60, 68])  
    Q2_bins.append([68, 75])  
    Q2_bins.append([100, 140])
    Q2_bins.append([140, 160])
    Q2_bins.append([160, 280])



    nrows, ncols = 1, 1
    fig = py.figure(figsize = (ncols * 16.0, nrows * 18.0))
    ax11 = py.subplot(nrows, ncols, 1)


    #--plot observable
    n = 10.0
    hand = {}
    for _ in data:
        if _ != 10001: continue
        for i in range(len(Q2_bins)):
            #--skip highest Q2 bin
            if i == 12: continue
            Q2_min, Q2_max = Q2_bins[i]
            for ic in range(kc.nc[istep]):
                if ic != 0: continue
                thyk = data[_]['thy-%d' % ic]
                value = data[_]['value']
                df = pd.DataFrame(data[_])
                df = df.query('Q2>%f and Q2<%f' % (Q2_min, Q2_max))
                xF = df.xF
                thy  = df['thy-%d' % ic]*n**i
                dthy = df['dthy-%d' % ic]*n**i
                value = df.value*n**i
                alpha = df.alpha*n**i
                #--sort by ascending X
                zt = sorted(zip(xF,thy))
                zv = sorted(zip(xF,value))
                za = sorted(zip(xF,alpha))
                zd = sorted(zip(xF,dthy))
                thy   = np.array([zt[i][1] for i in range(len(zt))])
                value = np.array([zv[i][1] for i in range(len(zv))])
                alpha = np.array([za[i][1] for i in range(len(za))])
                dthy  = np.array([zd[i][1] for i in range(len(zd))])
                xF    = sorted(xF)
                up   = thy + dthy
                down = thy - dthy
                ax,color,marker,ms = ax11, 'firebrick','.',10
                hand[_]   = ax.errorbar(xF, value, alpha, color = color, marker = marker, linestyle = 'none', ms=ms,capsize=3.0)
                thy_plot ,= ax.plot(xF, thy, color = 'black', linestyle = '-')
                thy_band  = ax.fill_between(xF,down,up,color='gold')


    ax11.tick_params(axis = 'both', which='both', top = True, right = True, direction='in', labelsize = 30)

    for ax in [ax11]:
        ax.semilogy()
        ax.set_xlim(-0.02, 1.15)
        ax.set_ylim(3e-2, 3e11)

        ax.yaxis.set_tick_params(which = 'major', length = 10)
        ax.yaxis.set_tick_params(which = 'minor', length = 5)

        ax.xaxis.set_tick_params(which = 'major', length = 10)
        ax.xaxis.set_tick_params(which = 'minor', length = 5)
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels([r'$0$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'])


    ax11.text(0.62, 0.08, r'$Q^2 \, \in \,[15,20]\,\mathrm{GeV^2}$', transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.13, r'$Q^2 \, \in \,[20,24]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.20, r'$Q^2 \, \in \,[24,28]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.28, r'$Q^2 \, \in \,[28,35]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.35, r'$Q^2 \, \in \,[35,40]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.42, r'$Q^2 \, \in \,[40,45]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.48, r'$Q^2 \, \in \,[45,52]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.55, r'$Q^2 \, \in \,[52,60]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.62, r'$Q^2 \, \in \,[60,68]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.68, r'$Q^2 \, \in \,[68,75]$',                 transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.70, 0.75, r'$Q^2 \, \in \,[100,140]$',               transform = ax11.transAxes, fontsize = 30)
    ax11.text(0.62, 0.83, r'$Q^2 \, \in \,[140,160]\, (i=11)$',      transform = ax11.transAxes, fontsize = 30)

    ax11.text(0.70, 0.05, r'$(i=0)$', transform = ax11.transAxes, fontsize = 30)

    ax11.text(0.12 ,0.05 ,r'\boldmath$M_{\ell \ell}^3 ~ \frac{d^2 \sigma^{\rm DY}}{dM_{\ell \ell} dx_F}$', transform = ax11.transAxes, size=60)
    ax11.text(0.42 ,0.07 ,r'$(\times\, 10^{\, i})$',                    transform = ax11.transAxes, size = 40)

    ax11.set_xlim(-0.02, 1.15)
    ax11.set_xlabel(r'\boldmath$x_{\rm F}$', size = 50)
    ax11.xaxis.set_label_coords(0.85,0.00)

    ax11.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax11.set_xticklabels([r'$0$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])


    handles = [hand[10001],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{FNAL E866}}' + ' ' + r'\boldmath$pp$'
    label2  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2] 
    ax11.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/DY-xsec'%cwd
    filename+='.png'
    #filename+='.pdf'
    ax11.set_rasterized(True)

    py.savefig(filename)
    print()
    print('Saving DY xsec plot to %s'%filename)
    py.close()


if __name__ == "__main__":

    wdir = 'results/misc/marathon_LHAPDF'

    print('\ngenerating DY from %s'%(wdir))


    plot_xsec(wdir)
    plot_ratios(wdir)





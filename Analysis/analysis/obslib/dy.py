#!/usr/bin/env python
import sys, os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT
import pandas as pd
import scipy as sp

## matplotlib
import matplotlib
matplotlib.use('Agg')
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

def get_plot(query,cluster_i=0):

    #--generate dictionary with everything needed for plot

    plot = {_:{} for _ in ['theory','X','Q2','value','alpha','std']}
    for key in query:
        theory = query[key]['thy-%d' % cluster_i]
        std    = query[key]['dthy-%d' % cluster_i]
        X      = query[key]['X']
        Q2     = query[key]['Q2']
        value  = query[key]['value']
        alpha  = query[key]['alpha']
        #--sort by ascending Q2
        zx = sorted(zip(Q2,X))
        zt = sorted(zip(Q2,theory))
        zv = sorted(zip(Q2,value))
        za = sorted(zip(Q2,alpha))
        zs = sorted(zip(Q2,std))
        plot['X'][key]      = np.array([zx[i][1] for i in range(len(zx))])
        plot['theory'][key] = np.array([zt[i][1] for i in range(len(zt))])
        plot['value'][key]  = np.array([zv[i][1] for i in range(len(zv))])
        plot['alpha'][key]  = np.array([za[i][1] for i in range(len(za))])
        plot['std'][key]    = np.array([zs[i][1] for i in range(len(zs))])
        plot['Q2'][key]     = np.array(sorted(Q2))

    return plot

def plot_dy(wdir, data, kc, istep, Q2_bins, dpi = 200):

    if 10001 not in data: return
    if 10002 not in data: return

    nrows, ncols = 2, 2
    fig = py.figure(figsize = (ncols * 12.0, nrows * 14.0))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)
    ax21 = py.subplot(nrows, ncols, 3)
    ax22 = py.subplot(nrows, ncols, 4)


    #--plot observable
    n = 10.0
    hand = {}
    for _ in data:
        for i in range(len(Q2_bins)):
            #--skip highest Q2 bin
            if i == 7: continue
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
                if _ == 10001: ax,color,marker,ms = ax11, 'firebrick','.',10
                if _ == 10002: ax,color,marker,ms = ax12, 'darkgreen','^',7
                hand[_]   = ax.errorbar(xF, value, alpha, color = color, marker = marker, linestyle = 'none', ms=ms,capsize=3.0)
                thy_plot ,= ax.plot(xF, thy, color = 'black', linestyle = '-')
                thy_band  = ax.fill_between(xF,down,up,color='gold')


    ax11.tick_params(axis = 'both', which='both', top = True, right = True, direction='in', labelsize = 30,labelbottom=False)
    ax12.tick_params(axis = 'both', which='both', top = True, right = True, labelleft=False, direction='in', labelsize = 30,labelbottom=False)

    for ax in [ax11,ax12]:
        ax.semilogy()
        ax.set_xlim(-0.02, 1.15)
        ax.set_ylim(3e-2, 5e6)

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


    ax11.text(0.80, 1.0e-1, r'$Q^2 \, \in \,[35,45]\,\mathrm{GeV^2}$', fontsize = 30)
    ax11.text(0.80, 8.0e-1, r'$Q^2 \, \in \,[45,52]$',   fontsize = 30)
    ax11.text(0.80, 7.0e0,  r'$Q^2 \, \in \,[52,60]$',   fontsize = 30)
    ax11.text(0.80, 6.0e1,  r'$Q^2 \, \in \,[60,68]$',   fontsize = 30)
    ax11.text(0.80, 3.0e2,  r'$Q^2 \, \in \,[68,75]$',   fontsize = 30)
    ax11.text(0.80, 1.0e3,  r'$Q^2 \, \in \,[100,140]$', fontsize = 30)
    ax11.text(0.71, 2.0e4,  r'$Q^2 \, \in \,[140,160]\, (i=6)$', fontsize = 30)

    ax12.text(0.80, 1.0e-1, r'$Q^2 \, \in \,[35,45]\,\mathrm{GeV^2}$', fontsize = 30)
    ax12.text(0.80, 8.0e-1, r'$Q^2 \, \in \,[45,52]$',   fontsize = 30)
    ax12.text(0.80, 7.0e0,  r'$Q^2 \, \in \,[52,60]$',   fontsize = 30)
    ax12.text(0.80, 6.0e1,  r'$Q^2 \, \in \,[60,68]$',   fontsize = 30)
    ax12.text(0.80, 3.0e2,  r'$Q^2 \, \in \,[68,75]$',   fontsize = 30)
    ax12.text(0.80, 1.0e3,  r'$Q^2 \, \in \,[100,140]$', fontsize = 30)
    ax12.text(0.80, 8.0e3,  r'$Q^2 \, \in \,[140,160]$', fontsize = 30)


    ax11.text(0.95, 5.0e-2, r'$(i=0)$', fontsize = 30)
    ax12.text(0.95, 5.0e-2, r'$(i=0)$', fontsize = 30)
    ax12.text(0.95, 4.0e3,  r'$(i=6)$', fontsize = 30)

    ax11.text(0.10 ,1.0e-1 ,r'\boldmath$M^3 \frac{d^2 \sigma}{dM dx_F}$',size=60)
    ax11.text(0.47 ,1.7e-1,r'$(\times\, 10^{\, i})$', size = 40)

    handles = [hand[10001],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{FNAL E866}}' + ' ' + r'\boldmath$pp$'
    label2  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2] 
    ax11.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    handles = [hand[10002]]
    label1  = r'\textbf{\textrm{FNAL E866}}' + ' ' + r'\boldmath$pd$'
    labels  = [label1] 
    ax12.legend(handles,labels,loc='upper right', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    #--plot ratio
    #ratio_1 = 2.5
    #ratio_2 = -3.0
    #--put limits on points to include
    ratio_1 = 1.5
    ratio_2 = -1.5
    for _ in data:
        cnt = 0
        for i in range(len(Q2_bins)):
            #--skip highest Q2 bin
            if i == 7: continue
            Q2_min, Q2_max = Q2_bins[i]
            for ic in range(kc.nc[istep]):
                if ic != 0: continue
                thyk = data[_]['thy-%d' % ic]
                ratio= data[_]['value'] / thyk
                data[_].update({'ratio' : ratio})
                df = pd.DataFrame(data[_])
                df = df.query('Q2>%f and Q2<%f' % (Q2_min, Q2_max))
                df = df.query('ratio<%f' % (ratio_1))
                df = df.query('ratio>%f' % (ratio_2))
                thy = df['thy-%d' % ic]
                xF = df.xF
                value = df.value
                alpha = df.alpha
                if _ == 10001:
                    ax21.errorbar(xF, value/thy + cnt, alpha/thy, color = 'firebrick', marker = '.', capsize = 3.0, linestyle = 'none', ms = 10)
                if _ == 10002:
                    ax22.errorbar(xF, value/thy + cnt, alpha/thy, color = 'darkgreen', marker = '^', capsize = 3.0, linestyle = 'none', ms = 7)
                cnt += 2.0


    #ax21.text(0.02, 13.40, r'\boldmath$|$' + r'\textbf{\textrm{data/theory}}' + r'\boldmath$|$' + r'\boldmath$<1.5$',size=40)
    ax21.text(0.02, 13.40, r'\textbf{\textrm{data/theory}}',size=40)

    ax21.text(0.80,  1.40, r'$Q^2 \, \in \,[35,45]\,\mathrm{GeV^2}$', fontsize = 30)
    ax21.text(0.80,  3.40, r'$Q^2 \, \in \,[45,52]$',   fontsize = 30)
    ax21.text(0.80,  5.40, r'$Q^2 \, \in \,[52,60]$',   fontsize = 30)
    ax21.text(0.80,  7.40, r'$Q^2 \, \in \,[60,68]$',   fontsize = 30)
    ax21.text(0.80,  9.40, r'$Q^2 \, \in \,[68,75]$',   fontsize = 30)
    ax21.text(0.80, 11.40, r'$Q^2 \, \in \,[100,140]$', fontsize = 30)
    ax21.text(0.80, 13.40, r'$Q^2 \, \in \,[140,160]$', fontsize = 30)

    ax22.text(0.80,  1.40, r'$Q^2 \, \in \,[35,45]\,\mathrm{GeV^2}$', fontsize = 30)
    ax22.text(0.80,  3.40, r'$Q^2 \, \in \,[45,52]$',   fontsize = 30)
    ax22.text(0.80,  5.40, r'$Q^2 \, \in \,[52,60]$',   fontsize = 30)
    ax22.text(0.80,  7.40, r'$Q^2 \, \in \,[60,68]$',   fontsize = 30)
    ax22.text(0.80,  9.40, r'$Q^2 \, \in \,[68,75]$',   fontsize = 30)
    ax22.text(0.80, 11.40, r'$Q^2 \, \in \,[100,140]$', fontsize = 30)
    ax22.text(0.80, 13.40, r'$Q^2 \, \in \,[140,160]$', fontsize = 30)

    ax21.tick_params(axis = 'both', which='both', top = True, right = True, direction='in', labelsize = 30)
    ax22.tick_params(axis = 'both', which='both', top = True, right = True, labelleft=False, direction='in', labelsize = 30)

    for ax in [ax21,ax22]:
        ax.set_xlim(-0.02, 1.15)
        ax.set_ylim(0, 14)
        for j in [0,2,4,6,8,10,12]:
            ax.axhline(j,0,1,color='black')
            ax.axhline(j+1,0,1,color='black',ls='--',alpha=0.5)
        ax.set_xlabel(r'\boldmath$x_{\rm F}$', size = 40)
        ax.xaxis.set_label_coords(0.85,0.00)

        ax.yaxis.set_tick_params(which = 'major', length = 10)
        ax.yaxis.set_tick_params(which = 'minor', length = 5)

        ax.xaxis.set_tick_params(which = 'major', length = 10)
        ax.xaxis.set_tick_params(which = 'minor', length = 5)
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels([r'$0$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])

    ax21.set_yticks([1,3,5,7,9,11,13])
    ax21.set_yticklabels([r'$1.0$',r'$1.0$',r'$1.0$',r'$1.0$',r'$1.0$',r'$1.0$',r'$1.0$'])

    ax21.set_yticks([ 0.5,  1,  1.5,\
                      2.5,  3,  3.5,\
                      4.5,  5,  5.5,\
                      6.5,  7,  7.5,\
                      8.5,  9,  9.5,\
                     10.5, 11, 11.5,\
                     12.5, 13, 13.5])
    ax21.set_yticklabels([r'$0.5$',r'$1.0$',r'$1.5$',\
                          r'$0.5$',r'$1.0$',r'$1.5$',\
                          r'$0.5$',r'$1.0$',r'$1.5$',\
                          r'$0.5$',r'$1.0$',r'$1.5$',\
                          r'$0.5$',r'$1.0$',r'$1.5$',\
                          r'$0.5$',r'$1.0$',r'$1.5$',\
                          r'$0.5$',r'$1.0$',r'$1.5$'])

    ax22.set_yticks([ 0.5,  1,  1.5,\
                      2.5,  3,  3.5,\
                      4.5,  5,  5.5,\
                      6.5,  7,  7.5,\
                      8.5,  9,  9.5,\
                     10.5, 11, 11.5,\
                     12.5, 13, 13.5])
    ax22.set_yticklabels([])

    py.tight_layout()
    py.subplots_adjust(wspace=0,hspace=0)
    print('Saving DY observable plot to %s/gallery/dy-%d.png'%(wdir,istep))
    py.savefig('%s/gallery/dy-%d.png' % (wdir, istep))
    py.close()

def plot_obs(wdir, kc, dpi = 200):

    print('\nplotting dy data from %s' % (wdir))

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

    predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
    if 'dy' not in predictions['reactions']:
        print('DY is not in data file')
        return
    labels  = load('%s/data/labels-%d.dat' % (wdir, istep))
    cluster = labels['cluster']

    data = predictions['reactions']['dy']

    for idx in data:
        predictions = copy.copy(data[idx]['prediction-rep'])
        del data[idx]['prediction-rep']
        del data[idx]['residuals-rep']
        del data[idx]['shift-rep']
        del data[idx]['rres-rep']
        for ic in range(kc.nc[istep]):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d' % ic] = np.mean(predictions_ic, axis = 0)
            data[idx]['dthy-%d' % ic] = np.std(predictions_ic, axis = 0)

    Q2_bins = []
    # Q2_bins.append([30, 40])
    Q2_bins.append([35, 45])    ## label pp:  37 < Q2 < 42   pd: idem
    Q2_bins.append([45, 52])    ## label pp:  47 < Q2 < 49   pd: idem
    Q2_bins.append([52, 60])    ## label pp:  54 < Q2 < 58   pd:  55 < Q2 < 57
    Q2_bins.append([60, 68])    ## label pp:  63 < Q2 < 66   pd:  62 < Q2 < 65
    Q2_bins.append([68, 75])    ## label pp:  70 < Q2 < 73   pd:  70 < Q2 < 72
    Q2_bins.append([100, 140])  ## label pp: 118 < Q2 < 131  pd: 124 < Q2 < 129
    Q2_bins.append([140, 160])  ## label pp: 148 < Q2 < 156  pd: 145 < Q2 < 154
    Q2_bins.append([160, 280])  ## label pp: 173 < Q2 < 222  pd: 173 < Q2 < 280

    plot_dy(wdir, data, kc, istep, Q2_bins, dpi)

    return

def plot_E866_ratio(wdir,kc,mode=1):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'dy' not in predictions['reactions']: return
    data = predictions['reactions']['dy']
    if 20001 not in data: return
    print('\ngenerating DY from %s'%(wdir))
    if 'dy' not in predictions['reactions']: return

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)

    conf['path2wzrvtab'] = '%s/grids/grids-dy'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['dy']={}
    conf['datasets']['dy']['xlsx']={}
    conf['datasets']['dy']['xlsx'][20001]='dy/expdata/20001.xlsx'
    conf['datasets']['dy']['norm']={}
    conf['datasets']['dy']['filters']=[]
    conf['dy tabs']=READER().load_data_sets('dy')
    tabs = conf['dy tabs']

    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
  
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']
  
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
    #--plot data
    for idx in tabs:
        xF     = tabs[idx]['xF']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==20001: ax,color,marker = ax11,'darkgreen',    '.'
        hand[idx] = ax.errorbar(xF,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=6.0,capsize=3.0)

    #--plot all replicas
    if mode==0:
        for i in range(len(replicas)):
            for idx in [20001]:
               if idx==20001: ax = ax11
               xF = tabs[idx]['xF']
               thy = data[idx]['prediction-rep'][i]

               thy_plot ,= ax.plot(xF,thy,color='red',alpha=0.2)

    #--plot mean and std of all replicas
    if mode==1:
        for idx in [20001]:
            for ic in range(nc):
                if idx==20001: ax = ax11
                xF = tabs[idx]['xF']
                thy = data[idx]['thy-%d'%ic]
                std = data[idx]['dthy-%d'%ic]
                down = thy - std
                up   = thy + std

                thy_plot ,= ax.plot(xF,thy,color='black')
                thy_band  = ax.fill_between(xF,down,up,color='gold',alpha=1.0)
    
    for ax in [ax11]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    if mode == 1:
        handles = [hand[20001],(thy_band,thy_plot)]
        label1 = r'\textbf{\textrm{FNAL E866}}'
        label2 = r'\textbf{\textrm{JAM}}'
        labels = [label1,label2] 
        ax11.legend(handles,labels,frameon=False,fontsize=20,loc='upper right', ncol = 1, handletextpad = 0.3, handlelength = 1.0)


    ax11.set_xlabel(r'\boldmath$x_F$',size=30)
    ax11.xaxis.set_label_coords(0.95,-0.02)

    ax11.set_xlim(0.00,0.55)
    ax11.set_ylim(0.55,1.30)

    ax11.text(0.45,0.10,r'\boldmath$\sigma_{pd}/2\sigma_{pp}$'     ,transform = ax11.transAxes,size=50)

    for ax in [ax11]:
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    for ax in [ax11]:
        majorLocator = MultipleLocator(0.2)
        minorLocator = MultipleLocator(0.05)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        majorLocator = MultipleLocator(0.2)
        minorLocator = MultipleLocator(0.05)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    ax11.set_xticks([0.2,0.4])

    py.tight_layout()
    #py.subplots_adjust(hspace = 0, wspace=0.15)

    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/dy-ratio'%wdir
    if mode == 1: filename+='-band'
    filename+='.png'

    py.savefig(filename)
    print()
    print('Saving DY plot to %s'%filename)

def plot_SQ(wdir,kc,mode=1):

    print('\ngenerating DY from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'dy' not in predictions['reactions']: return
    if 20002 not in predictions['reactions']['dy']: return

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)

    conf['path2wzrvtab'] = '%s/grids/grids-dy'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['dy']={}
    conf['datasets']['dy']['xlsx']={}
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
    #--plot data
    for idx in tabs:
        xF     = tabs[idx]['xF']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==20002: ax,color,marker = ax11,'darkgreen',    '.'
        hand[idx] = ax.errorbar(xF,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=6.0,capsize=3.0)

    #--plot all replicas
    if mode==0:
        for i in range(len(replicas)):
            for idx in [20002]:
               if idx==20002: ax = ax11
               xF = tabs[idx]['xF']
               thy = data[idx]['prediction-rep'][i]

               thy_plot ,= ax.plot(xF,thy,color='red',alpha=0.2)

    #--plot mean and std of all replicas
    if mode==1:
        for idx in [20002]:
            for ic in range(nc):
                if idx==20002: ax = ax11
                xF = tabs[idx]['xF']
                thy = data[idx]['thy-%d'%ic]
                std = data[idx]['dthy-%d'%ic]
                down = thy - std
                up   = thy + std

                thy_plot ,= ax.plot(xF,thy,color='black')
                thy_band  = ax.fill_between(xF,down,up,color='gold',alpha=1.0)
    
    for ax in [ax11]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    if mode == 1:
        handles = [hand[20002],(thy_band,thy_plot)]
        label1 = r'\textbf{\textrm{FNAL E906}}'
        label2 = r'\textbf{\textrm{JAM}}'
        labels = [label1,label2] 
        ax11.legend(handles,labels,frameon=False,fontsize=20,loc='upper right', ncol = 1, handletextpad = 0.3, handlelength = 1.0)


    ax11.set_xlabel(r'\boldmath$x_F$',size=30)
    ax11.xaxis.set_label_coords(0.95,-0.02)

    ax11.set_xlim(0.00,0.60)
    ax11.set_ylim(0.70,1.50)

    ax11.text(0.45,0.10,r'\boldmath$\sigma_{pd}/2\sigma_{pp}$'     ,transform = ax11.transAxes,size=50)

    for ax in [ax11]:
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    for ax in [ax11]:
        majorLocator = MultipleLocator(0.2)
        minorLocator = MultipleLocator(0.05)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        majorLocator = MultipleLocator(0.2)
        minorLocator = MultipleLocator(0.05)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    ax11.set_xticks([0.2,0.4])

    py.tight_layout()
    #py.subplots_adjust(hspace = 0, wspace=0.15)

    checkdir('%s/gallery'%wdir)
    filename = '%s/gallery/dy-SQ'%wdir
    if mode == 1: filename+='-band'
    filename+='.png'

    py.savefig(filename)
    print()
    print('Saving DY plot to %s'%filename)




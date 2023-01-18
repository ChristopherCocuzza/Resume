#!/usr/bin/env python
import sys, os
import numpy as np
import copy
import pandas as pd
import scipy as sp
from scipy.interpolate import griddata

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import kmeanconf as kc

## matplotlib
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.latex.preview']=True
import pylab as py
from matplotlib.ticker import MultipleLocator

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack fitlib
from fitlib.resman import RESMAN

## from analysis
from analysis.corelib import core
from analysis.corelib import classifier

cwd = 'plots/marathon/'

def get_Q2bins(data,kind):
    query = {}
    if kind == 'MARATHON':
        query[0]  = data.query('Q2 > 1.5 and Q2 <= 20')
    if kind == 'NMC':
        #query[10] = data.query('Q2 > 15.0 and Q2 <= 20.0')
        #query[9]  = data.query('Q2 > 12.0 and Q2 <= 15.0')
        #query[8]  = data.query('Q2 > 10.0 and Q2 <= 12.0')
        #query[7]  = data.query('Q2 > 8.0 and Q2 <= 10.0')
        #query[6]  = data.query('Q2 > 6.0 and Q2 <= 8.0')
        #query[5]  = data.query('Q2 > 5.0 and Q2 <= 6.0')
        #query[4]  = data.query('Q2 > 4.0 and Q2 <= 5.0')
        #query[3]  = data.query('Q2 > 3.0 and Q2 <= 4.0')
        #query[2]  = data.query('Q2 > 2.5 and Q2 <= 3.0')
        #query[1]  = data.query('Q2 > 2.0 and Q2 <= 2.5')
        #query[0]  = data.query('Q2 > 1.0 and Q2 <= 2.0')
        #query[0] = data.query('Q2 > 1.5 and Q2 <= 20.0')
        query[0] = data.query('X > 0.15 and X <= 0.40')

    return query

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

def get_theory(PLOT,nbins,loop=True,funcQ2=True,funcX=False):

    #--interpolate theoretical values across Q2 or X

    theory = {}

    if funcQ2: svar = 'Q2'
    if funcX:  svar = 'X'

    theory = {_:{} for _ in [svar,'value','std']}
    for key in range(nbins):
        var = []
        thy = []
        std = []
        
        #--if plotting for multiple experiments, loop over and combine
        if loop:
            for exp in PLOT:
                var.extend(PLOT[exp][svar][key])
                thy.extend(PLOT[exp]['theory'][key])
                std.extend(PLOT[exp]['std'][key])
        else:
            var.extend(PLOT[svar][key])
            thy.extend(PLOT['theory'][key])
            std.extend(PLOT['std'][key])

        #--if nothing in bin, skip
        if len(var) == 0: continue

        vmin = np.min(var)
        vmax = np.max(var)
        theory[svar][key]  = np.array([0.195,0.225,0.255,0.285,0.315,0.345,0.375,0.405,0.435,0.465,0.495,0.525,0.555,0.585,0.615,0.645,0.675,0.705,0.735,0.765,0.795,0.825])

        #--if more than one value, interpolate between them
        if len(var) > 1:
            theory['value'][key] = griddata(np.array(var),np.array(thy),theory[svar][key],method='linear')
            theory['std'][key]   = griddata(np.array(var),np.array(std),theory[svar][key],method='linear')
        else:
            theory['value'][key] = np.ones(100)*thy 
            theory['std'][key]   = np.ones(100)*std


    return theory

def get_details(exp):

    #--get details for plotting

    if exp=='MARATHON dp':   color, marker, ms = 'blue'     , 'o', 5 
    if exp=='MARATHON ht':   color, marker, ms = 'darkgreen', '^', 5 
    if exp=='NMC':           color, marker, ms = 'blue'     , '*', 5 
    return color,marker,ms

if __name__=='__main__':

    nrows, ncols = 1, 2
    py.figure(figsize = (ncols * 9, nrows * 7))
    ax11 = py.subplot(nrows, ncols, 1)

    WDIR = {}
    WDIR[1]  = 'results/marathon/step29pos'          #--pdfs and off-offshell fitted
    WDIR[2]  = 'results/marathon/more/offonly-nooff' #--pdfs fixed, off-shell set to zero
    WDIR[3]  = 'results/marathon/more/offonly'       #--pdfs fixed, off-shell fitted

    data = {}
    for part in WDIR:
        wdir = WDIR[part]
        load_config('%s/input.py' % wdir)
        istep = core.get_istep()
        replicas = core.get_replicas(wdir)
        core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

        predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
        labels  = load('%s/data/labels-%d.dat' % (wdir, istep))
        cluster = labels['cluster']

        data[part] = predictions['reactions']['idis']

        for idx in data[part]:
            predictions = copy.copy(data[part][idx]['prediction-rep'])
            del data[part][idx]['prediction-rep']
            del data[part][idx]['residuals-rep']
            del data[part][idx]['shift-rep']
            for ic in range(kc.nc[istep]):
                predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
                data[part][idx]['thy-%d' % ic] = np.mean(predictions_ic, axis = 0)
                data[part][idx]['dthy-%d' % ic] = np.std(predictions_ic, axis = 0)
                if 'X' in data[part][idx]: data[part][idx]['x'] = data[part][idx]['X']
                data[part][idx]['rQ2'] = np.around(data[part][idx]['Q2'], decimals = 0)
                data[part][idx]['rx']  = np.around(data[part][idx]['x'], decimals = 2)


    #################################
    #--Plot F2h/F2t from MARATHON
    #################################

    theory = {}
    for key in data:
        DATA = pd.DataFrame(data[key][10051])

        for cluster_i in range(kc.nc[istep]):
            if cluster_i != 0: continue

            query = get_Q2bins(DATA,'MARATHON')
            nbins = len(query)
            PLOT = get_plot(query)
            theory[key] = get_theory(PLOT,nbins,loop=False,funcX=True)



    X      = theory[1]['X'][0]
    val1   = theory[1]['value'][0]
    val2   = theory[2]['value'][0]
    val3   = theory[3]['value'][0]
    _std1  = theory[1]['std'][0]
    _std2  = theory[2]['std'][0]
    _std3  = theory[3]['std'][0]

    rat1  = val3/val2
    rat2  = val1/val3

    std1  = rat1*np.sqrt((_std3/val3)**2 + (_std2/val2)**2)
    std2  = rat2*np.sqrt((_std1/val1)**2 + (_std3/val3)**2)

    hand = {}
    #hand['rat1'] = ax11.fill_between(X, rat1 - std1 , rat1 + std1, alpha = 0.5, color='firebrick')
    #hand['rat2'] = ax11.fill_between(X, rat2 - std2 , rat2 + std2, alpha = 0.5, color='darkblue')
    hand['rat1'] = ax11.errorbar(X, rat1, yerr = std1 , alpha = 1.0, capsize = 2.5, marker = 'o', linestyle = 'none', ms = 3, color='firebrick')
    hand['rat2'] = ax11.errorbar(X+0.005, rat2, yerr = std2 , alpha = 1.0, capsize = 2.5, marker = 'o', linestyle = 'none', ms = 3, color='darkblue')

    ax11.set_xlim(0.15, 0.85)
    ax11.set_ylim(0.915, 1.015)

    ax11.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 20)

    ax11.yaxis.set_tick_params(which = 'major', length = 5)
    ax11.yaxis.set_tick_params(which = 'minor', length = 2.5)

    ax11.xaxis.set_tick_params(which = 'major', length = 5)
    ax11.xaxis.set_tick_params(which = 'minor', length = 2.5)

    ax11.set_xlabel(r'\boldmath$x$',size=50)
    ax11.xaxis.set_label_coords(0.50,-0.05)

    ax11.axhline(1,0,1,ls=':',alpha=0.5,color='black')

    ax11.text(0.05, 0.45, r'\textrm{\textbf{Ratio of Fit Results for}}'                                        , transform = ax11.transAxes, size = 25)
    ax11.text(0.05, 0.35, r'\textrm{\textbf{MARATHON}}' + r' ' + r'\boldmath$F_2^{^3{\rm He}}/F_2^{^3{\rm H}}$', transform = ax11.transAxes, size = 25)

    ax11.text(1.05, 0.25, r'\textrm{\textbf{Fit 1: Fit PDFs, Fit Off-shell}}', transform = ax11.transAxes, size = 25)
    ax11.text(1.05, 0.15, r'\textrm{\textbf{Fit 2: Fix PDFs, Fit Off-shell}}', transform = ax11.transAxes, size = 25)
    ax11.text(1.05, 0.05, r'\textrm{\textbf{Fit 3: Fix PDFs, No  Off-shell}}', transform = ax11.transAxes, size = 25)

    ax11.text(0.70, 0.90, r'$Q^2 = 14 x ~ {\rm GeV^2}$', transform = ax11.transAxes, size = 25)

    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.10)
    ax11.xaxis.set_minor_locator(minorLocator)
    ax11.xaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.005)
    majorLocator = MultipleLocator(0.02)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)

    handles, labels = [],[]
    handles.append(hand['rat2'])
    handles.append(hand['rat1'])
    labels.append(r'\textbf{\textrm{Fit 1/Fit 2 (Effect of PDFs)}}')
    labels.append(r'\textbf{\textrm{Fit 2/Fit 3 (Effect of Off-Shell)}}')
    ax11.legend(handles,labels,loc='lower left', fontsize = 25, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    py.tight_layout()
    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/ratios.png'%cwd
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()






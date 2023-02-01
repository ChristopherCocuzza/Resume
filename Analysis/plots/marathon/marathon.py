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

    plot = {_:{} for _ in ['theory','X','Q2','value','alpha','std','shifts']}
    for key in query:
        theory = query[key]['thy-%d' % cluster_i]
        std    = query[key]['dthy-%d' % cluster_i]
        X      = query[key]['X']
        Q2     = query[key]['Q2']
        value  = query[key]['value']
        alpha  = query[key]['alpha']
        shifts = query[key]['shifts']
        #--sort by ascending Q2
        zx = sorted(zip(Q2,X))
        zt = sorted(zip(Q2,theory))
        zv = sorted(zip(Q2,value))
        za = sorted(zip(Q2,alpha))
        zs = sorted(zip(Q2,std))
        zsh= sorted(zip(Q2,shifts))
        plot['X'][key]      = np.array([zx[i][1]  for i in range(len(zx))])
        plot['theory'][key] = np.array([zt[i][1]  for i in range(len(zt))])
        plot['value'][key]  = np.array([zv[i][1]  for i in range(len(zv))])
        plot['alpha'][key]  = np.array([za[i][1]  for i in range(len(za))])
        plot['std'][key]    = np.array([zs[i][1]  for i in range(len(zs))])
        plot['shifts'][key] = np.array([zsh[i][1] for i in range(len(zsh))])
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
        theory[svar][key]  = np.geomspace(vmin,vmax,100)

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

    if exp=='MARATHON dp':   color, marker, ms = 'black', 'o', 4 
    if exp=='MARATHON ht':   color, marker, ms = 'black', 'o', 4 
    return color,marker,ms

if __name__=='__main__':

    nrows, ncols = 1, 2
    py.figure(figsize = (ncols * 5, nrows * 6))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)

    wdir1 = 'results/marathon/final'
    wdir2 = 'results/upol/step17'
    wdir1 = 'final2'
    WDIR = [wdir1,wdir2]

    j = 0
    JAMcolor = 'red'
    JAMlw    = 2
    for wdir in WDIR:
        load_config('%s/input.py' % wdir)
        istep = core.get_istep()
        replicas = core.get_replicas(wdir)
        core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

        predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
        labels  = load('%s/data/labels-%d.dat' % (wdir, istep))
        cluster = labels['cluster']

        data = predictions['reactions']['idis']

        for idx in data:
            predictions = copy.copy(data[idx]['prediction-rep'])
            shifts = copy.copy(data[idx]['shift-rep'])
            data[idx]['shifts'] = np.mean(shifts,axis=0) 
            del data[idx]['prediction-rep']
            del data[idx]['residuals-rep']
            del data[idx]['shift-rep']
            for ic in range(kc.nc[istep]):
                predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
                data[idx]['thy-%d' % ic] = np.mean(predictions_ic, axis = 0)
                data[idx]['dthy-%d' % ic] = np.std(predictions_ic, axis = 0)
                if 'X' in data[idx]: data[idx]['x'] = data[idx]['X']
                data[idx]['rQ2'] = np.around(data[idx]['Q2'], decimals = 0)
                data[idx]['rx'] = np.around(data[idx]['x'], decimals = 2)



        #################################
        #--Plot F2d/F2p from MARATHON
        #################################

        dp   = data[10050] ## MARATHON d/p

        DATA = {}
        DATA['MARATHON dp']   = pd.DataFrame(dp)

        PLOT = {}
        theory = {}
        for cluster_i in range(kc.nc[istep]):
            if cluster_i != 0: continue

            for exp in DATA:
                query = get_Q2bins(DATA[exp],'MARATHON')
                nbins = len(query)
                PLOT[exp] = get_plot(query)
                theory[exp] = get_theory(PLOT[exp],nbins,loop=False,funcX=True)

        hand = {}
        #--plot data points
        for exp in PLOT:
            for key in PLOT[exp]['X']:
                X     = PLOT[exp]['X'][key]
                val   = PLOT[exp]['value'][key]
                alpha = PLOT[exp]['alpha'][key]
                color,marker,ms = get_details(exp)
                hand[exp] = ax11.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=2.5,linestyle='none',zorder=4)

                #--plot correlated shifts
                if j==0:
                    center = 0.77
                    shifts = PLOT[exp]['shifts'][key]
                    pos, neg = np.zeros(len(alpha)),np.zeros(len(alpha))
                    for i in range(len(shifts)):
                        if shifts[i] > 0.0: pos[i] = shifts[i]
                        else:               neg[i] = shifts[i]

                    #ax11.fill_between(X, center, center + pos, color = 'g', alpha = 0.5)
                    #ax11.fill_between(X, center, center + neg, color = 'b', alpha = 0.5)

            #--plot theory interpolated between data points
            for key in theory[exp]['value']:
                X    = theory[exp]['X'][key]
                mean = theory[exp]['value'][key]
                std  = theory[exp]['std'][key]
                down = mean - std
                up   = mean + std
                if j==0:
                    thy_plot ,= ax11.plot(X,mean,linestyle='solid',lw=JAMlw,color=JAMcolor,zorder=3)
                    thy_band  = ax11.fill_between(X,down,up,color=JAMcolor,alpha=0.2,zorder=3)
                if j==1:
                    nooff_plot ,= ax11.plot(X,mean,linestyle='dashed',color='green',zorder=2,lw=2)
                    nooff_band  = ax11.fill_between(X,down,up,color='gold',alpha=0.0,zorder=2)


        #################################
        #--Plot F2h/F2t from MARATHON
        #################################

        ht   = data[10051] ## MARATHON h/t

        DATA = {}
        DATA['MARATHON ht']  = pd.DataFrame(ht)

        PLOT = {}
        theory = {}
        for cluster_i in range(kc.nc[istep]):
            if cluster_i != 0: continue

            for exp in DATA:
                query = get_Q2bins(DATA[exp],'MARATHON')
                nbins = len(query)
                PLOT[exp] = get_plot(query)
                theory[exp] = get_theory(PLOT[exp],nbins,loop=False,funcX=True)


        #--plot data points
        for exp in PLOT:
            for key in range(nbins):
                X     = PLOT[exp]['X'][key]
                val   = PLOT[exp]['value'][key]*2.0**float(key)
                alpha = PLOT[exp]['alpha'][key]*2.0**float(key)
                color,marker,ms = get_details(exp)
                hand[exp] = ax12.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=2.5,linestyle='none',zorder=4)

                #--plot correlated shifts
                if j==0:
                    center = 1.1
                    shifts = PLOT[exp]['shifts'][key]
                    pos, neg = np.zeros(len(alpha)),np.zeros(len(alpha))
                    for i in range(len(shifts)):
                        if shifts[i] > 0.0: pos[i] = shifts[i]
                        else:               neg[i] = shifts[i]

                    #ax12.fill_between(X, center, center + pos, color = 'g', alpha = 0.5)
                    #ax12.fill_between(X, center, center + neg, color = 'b', alpha = 0.5)

            #--plot theory interpolated between data points
            for key in theory[exp]['value']:
                X    = theory[exp]['X'][key]
                mean = theory[exp]['value'][key]*2.0**float(key)
                std  = theory[exp]['std'][key]  *2.0**float(key)
                down = mean - std
                up   = mean + std
                if j==0:
                    thy_plot ,= ax12.plot(X,mean,linestyle='solid',color=JAMcolor,lw=JAMlw,zorder=3)
                    thy_band  = ax12.fill_between(X,down,up,color='red',alpha=0.2,zorder=3)
                if j==1:
                    nooff_plot ,= ax12.plot(X,mean,linestyle='dashed',color='green',zorder=2,lw=2)
                    nooff_band  = ax12.fill_between(X,down,up,color='gold',alpha=0.0,zorder=2)

        j += 1


    ax11.set_xlim(0.150, 0.44)
    ax11.set_ylim(0.740, 0.88)

    ax11.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 20)

    ax11.yaxis.set_tick_params(which = 'major', length = 5)
    ax11.yaxis.set_tick_params(which = 'minor', length = 2.5)

    ax11.xaxis.set_tick_params(which = 'major', length = 5)
    ax11.xaxis.set_tick_params(which = 'minor', length = 2.5)

    ax11.set_xlabel(r'\boldmath$x$', size=30)
    ax11.xaxis.set_label_coords(0.97,0.00)

    #ax11.text(0.05, 0.05, r'\textrm{\textbf{MARATHON}}', transform = ax11.transAxes, size = 25)

    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.10)
    ax11.xaxis.set_minor_locator(minorLocator)
    ax11.xaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.05)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)

    #handles, labels = [],[]
    #handles.append(hand['MARATHON dp'])
    #handles.append((thy_band,thy_plot))
    #labels.append(r'\boldmath$F_2^D/F_2^p$')
    #labels.append(r'\textbf{\textrm{JAM}}')
    #ax11.legend(handles,labels,loc='upper right', fontsize = 22, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    ax12.set_xlim(0.10, 0.95)
    ax12.set_ylim(1.07, 1.33)

    ax12.tick_params(axis = 'both', which = 'both', direction='in', top = True, right = True, labelsize = 20)

    ax12.yaxis.set_tick_params(which = 'major', length = 5)
    ax12.yaxis.set_tick_params(which = 'minor', length = 2.5)

    ax12.xaxis.set_tick_params(which = 'major', length = 5)
    ax12.xaxis.set_tick_params(which = 'minor', length = 2.5)

    ax12.set_xlabel(r'\boldmath$x$', size=30)
    ax12.xaxis.set_label_coords(0.97,0.00)

    #ax12.text(0.50, 0.05, r'$Q^2 = 14 \cdot x ~ {\rm GeV^2}$', transform = ax12.transAxes, size = 20)
    ax12.text(0.40, 0.05, r'$Q^2 = 14 x ~ {\rm GeV^2}$', transform = ax12.transAxes, size = 25)

    minorLocator = MultipleLocator(0.04)
    majorLocator = MultipleLocator(0.2)
    ax12.xaxis.set_minor_locator(minorLocator)
    ax12.xaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.1)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)
    ax12.set_xticks([0.2,0.4,0.6,0.8])

    ax11.text(0.53, 0.85, r'\boldmath$F_2^D/F_2^p$'                   , transform = ax11.transAxes, size = 35)
    ax12.text(0.07, 0.85, r'\boldmath$F_2^{^3\rm{He}}/F_2^{^3\rm{H}}$', transform = ax12.transAxes, size = 35)

    handles, labels = [],[]
    handles.append(hand['MARATHON ht'])
    handles.append((thy_band,thy_plot))
    handles.append((nooff_band,nooff_plot))
    labels.append(r'\textbf{\textrm{{\huge MARATHON}}}')
    labels.append(r'\textbf{\textrm{JAM}}')
    #labels.append(r'\boldmath$F_2^{^3\rm{He}}/F_2^{^3\rm{H}}$')
    labels.append(r'\textrm{\textbf{on-shell fit}}')
    ax11.legend(handles,labels,loc='lower left', fontsize = 20, frameon = 0, handletextpad = 0.3, handlelength = 1.4)

    py.tight_layout()
    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/marathon'%cwd
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    py.close()






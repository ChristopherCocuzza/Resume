#!/usr/bin/env python
import sys, os
import numpy as np
import copy
import pandas as pd
import scipy as sp
from scipy.interpolate import griddata

## matplotlib
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.latex.preview']=True
import pylab as py
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack fitlib
from fitlib.resman import RESMAN

## from fitpack analysis
from analysis.corelib import core
from analysis.corelib import classifier

def get_Q2bins(data,kind):
    query = {}
    #--proton A1/Apa
    if kind == 'p A1':
        query[6]  = data.query('Q2 > 60.0')                
        query[5]  = data.query('Q2 > 40.0 and Q2 <= 60.0') 
        query[4]  = data.query('Q2 > 20.0 and Q2 <= 40.0') 
        query[3]  = data.query('Q2 > 10.0 and Q2 <= 20.0') 
        query[2]  = data.query('Q2 > 5.00 and Q2 <= 10.0') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 5.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00') 
    if kind == 'p Apa':
        query[8]  = data.query('Q2 > 20.0') 
        query[7]  = data.query('Q2 > 15.0 and Q2 <= 20.0') 
        query[6]  = data.query('Q2 > 10.0 and Q2 <= 15.0') 
        query[5]  = data.query('Q2 > 7.00 and Q2 <= 10.0')
        query[4]  = data.query('Q2 > 5.00 and Q2 <= 7.00')
        query[3]  = data.query('Q2 > 4.00 and Q2 <= 5.00') 
        query[2]  = data.query('Q2 > 3.00 and Q2 <= 4.00') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 3.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00') 
    if kind == 'p Apa EG1b':
        query[8]  = data.query('Q2 > 4.50') 
        query[7]  = data.query('Q2 > 4.00 and Q2 <= 4.50') 
        query[6]  = data.query('Q2 > 3.50 and Q2 <= 4.00') 
        query[5]  = data.query('Q2 > 3.00 and Q2 <= 3.50') 
        query[4]  = data.query('Q2 > 2.50 and Q2 <= 3.00') 
        query[3]  = data.query('Q2 > 2.20 and Q2 <= 2.50') 
        query[2]  = data.query('Q2 > 2.00 and Q2 <= 2.20') 
        query[1]  = data.query('Q2 > 1.80 and Q2 <= 2.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 1.80') 
    if kind == 'p Apa DVCS':
        query[6]  = data.query('Q2 > 4.50') 
        query[5]  = data.query('Q2 > 3.50 and Q2 <= 4.50') 
        query[4]  = data.query('Q2 > 3.00 and Q2 <= 3.50') 
        query[3]  = data.query('Q2 > 2.50 and Q2 <= 3.00') 
        query[2]  = data.query('Q2 > 2.20 and Q2 <= 2.50') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 2.20') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00') 

    #--deuteron A1/Apa
    if kind == 'd A1':
        query[4]  = data.query('Q2 > 40.0') 
        query[3]  = data.query('Q2 > 20.0 and Q2 <= 40.0') 
        query[2]  = data.query('Q2 > 10.0 and Q2 <= 20.0') 
        query[1]  = data.query('Q2 > 5.00 and Q2 <= 10.0') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 5.00')
    if kind == 'd Apa':
        query[8]  = data.query('Q2 > 20.0') 
        query[7]  = data.query('Q2 > 15.0 and Q2 <= 20.0') 
        query[6]  = data.query('Q2 > 10.0 and Q2 <= 15.0') 
        query[5]  = data.query('Q2 > 7.00 and Q2 <= 10.0')
        query[4]  = data.query('Q2 > 5.00 and Q2 <= 7.00')
        query[3]  = data.query('Q2 > 4.00 and Q2 <= 5.00') 
        query[2]  = data.query('Q2 > 3.00 and Q2 <= 4.00') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 3.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00') 
    if kind == 'd Apa EG1b':
        query[8]  = data.query('Q2 > 4.50') 
        query[7]  = data.query('Q2 > 4.00 and Q2 <= 4.50') 
        query[6]  = data.query('Q2 > 3.50 and Q2 <= 4.00') 
        query[5]  = data.query('Q2 > 3.00 and Q2 <= 3.50') 
        query[4]  = data.query('Q2 > 2.50 and Q2 <= 3.00') 
        query[3]  = data.query('Q2 > 2.20 and Q2 <= 2.50') 
        query[2]  = data.query('Q2 > 2.00 and Q2 <= 2.20') 
        query[1]  = data.query('Q2 > 1.80 and Q2 <= 2.00') 
        query[0]  = data.query('Q2 > 1.60 and Q2 <= 1.80') 
    if kind == 'd Apa DVCS':
        query[5]  = data.query('Q2 > 3.50') 
        query[4]  = data.query('Q2 > 3.00 and Q2 <= 3.50') 
        query[3]  = data.query('Q2 > 2.50 and Q2 <= 3.00') 
        query[2]  = data.query('Q2 > 2.20 and Q2 <= 2.50') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 2.20') 
        query[0]  = data.query('Q2 > 1.60 and Q2 <= 2.00') 

    #--proton A2/Ape/Atpe
    if kind == 'p A2':
        query[2]  = data.query('Q2 > 5.00') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 5.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00')
    if kind == 'p Ape':
        query[8]  = data.query('Q2 > 20.0') 
        query[7]  = data.query('Q2 > 15.0 and Q2 <= 20.0') 
        query[6]  = data.query('Q2 > 10.0 and Q2 <= 15.0') 
        query[5]  = data.query('Q2 > 7.00 and Q2 <= 10.0')
        query[4]  = data.query('Q2 > 5.00 and Q2 <= 7.00')
        query[3]  = data.query('Q2 > 4.00 and Q2 <= 5.00') 
        query[2]  = data.query('Q2 > 3.00 and Q2 <= 4.00') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 3.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00') 
    if kind == 'p Atpe':
        query[7]  = data.query('Q2 > 15.0') 
        query[6]  = data.query('Q2 > 10.0 and Q2 <= 15.0') 
        query[5]  = data.query('Q2 > 7.00 and Q2 <= 10.0')
        query[4]  = data.query('Q2 > 5.00 and Q2 <= 7.00')
        query[3]  = data.query('Q2 > 4.00 and Q2 <= 5.00') 
        query[2]  = data.query('Q2 > 3.00 and Q2 <= 4.00') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 3.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00') 

    #--deuteron Ape/Atpe
    if kind == 'd Ape':
        query[8]  = data.query('Q2 > 20.0') 
        query[7]  = data.query('Q2 > 15.0 and Q2 <= 20.0') 
        query[6]  = data.query('Q2 > 10.0 and Q2 <= 15.0') 
        query[5]  = data.query('Q2 > 7.00 and Q2 <= 10.0')
        query[4]  = data.query('Q2 > 5.00 and Q2 <= 7.00')
        query[3]  = data.query('Q2 > 4.00 and Q2 <= 5.00') 
        query[2]  = data.query('Q2 > 3.00 and Q2 <= 4.00') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 3.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00') 
    if kind == 'd Atpe':
        query[7]  = data.query('Q2 > 15.0') 
        query[6]  = data.query('Q2 > 10.0 and Q2 <= 15.0') 
        query[5]  = data.query('Q2 > 7.00 and Q2 <= 10.0')
        query[4]  = data.query('Q2 > 5.00 and Q2 <= 7.00')
        query[3]  = data.query('Q2 > 4.00 and Q2 <= 5.00') 
        query[2]  = data.query('Q2 > 3.00 and Q2 <= 4.00') 
        query[1]  = data.query('Q2 > 2.00 and Q2 <= 3.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 2.00') 

    #--helium 
    if kind == 'h A1':
        query[0]  = data.query('Q2 > 1.00') 
    if kind == 'h Apa':
        query[3]  = data.query('Q2 > 10.0') 
        query[2]  = data.query('Q2 > 5.00 and Q2 <= 10.0')
        query[1]  = data.query('Q2 > 3.00 and Q2 <= 5.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 3.00') 
    if kind == 'h A2':
        query[0]  = data.query('Q2 > 1.00') 
    if kind == 'h Ape':
        query[3]  = data.query('Q2 > 10.0') 
        query[2]  = data.query('Q2 > 5.00 and Q2 <= 10.0')
        query[1]  = data.query('Q2 > 3.00 and Q2 <= 5.00') 
        query[0]  = data.query('Q2 > 1.00 and Q2 <= 3.00') 
 
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
                var. extend(PLOT[exp][svar][key])
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

    if exp=='COMPASS1':    color, marker, ms = 'firebrick', '*', 6
    if exp=='COMPASS2':    color, marker, ms = 'firebrick', '*', 6
    if exp=='EMC' :        color, marker, ms = 'darkgreen', '^', 6
    if exp=='SMC1':        color, marker, ms = 'blue'     , 'o', 6
    if exp=='SMC2':        color, marker, ms = 'blue'     , 'o', 6

    if exp=='HERMES':      color, marker, ms = 'firebrick', '*', 6
    if exp=='SLACE143':    color, marker, ms = 'darkgreen', '^', 6
    if exp=='SLACE155':    color, marker, ms = 'blue'     , 'o', 6
    if exp=='SLACE80E130': color, marker, ms = 'black'    , 'o', 6

    if exp=='EG1b4.2':     color, marker, ms = 'firebrick', '*', 6
    if exp=='EG1b5.7':     color, marker, ms = 'blue'     , 'o', 6

    if exp=='DVCS4.8':     color, marker, ms = 'firebrick', '*', 6
    if exp=='DVCS6.0':     color, marker, ms = 'blue'     , '^', 6

    if exp=='DVCS5.7':     color, marker, ms = 'firebrick', '*', 6

    if exp=='SLAC E155x':  color, marker, ms = 'firebrick', '*', 6

    if exp=='SLACE142':    color, marker, ms = 'darkgreen', '^', 6

    if exp=='SLACE154':    color, marker, ms = 'firebrick', '*', 6
    if exp=='JLabE06E014': color, marker, ms = 'darkgreen', '^', 6   
    if exp=='JLabE99E117': color, marker, ms = 'blue'     , 'o', 6   

    return color,marker,ms

def plot_proton_A1_Apa(wdir, data, kc, istep):

    if 10002 not in data: return #--COMPASS A1
    if 10003 not in data: return #--COMPASS A1
    if 10004 not in data: return #--EMC A1
    if 10007 not in data: return #--HERMES Apa
    if 10017 not in data: return #--JLab DVCS Apa
    if 10022 not in data: return #--SLAC E143 Apa
    if 10029 not in data: return #--SLAC E155 Apa
    if 10032 not in data: return #--SLAC E80E130 Apa
    if 10035 not in data: return #--SMC A1
    if 10036 not in data: return #--SMC A1
    if 10043 not in data: return #--JLab EG1b Apa (E=4.2)
    if 10044 not in data: return #--JLab EG1b Apa (E=5.7)

    nrows, ncols = 2, 2
    py.figure(figsize = (ncols * 12.0, nrows * 14.0))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)
    ax21 = py.subplot(nrows, ncols, 3)
    ax22 = py.subplot(nrows, ncols, 4)

    #######################################
    #--Plot A1 from COMPASS, EMC, HERMES, SMC
    #######################################

    compass1 = data[10002] #--COMPASS A1
    compass2 = data[10003] #--COMPASS A1
    emc      = data[10004] #--EMC A1
    smc1     = data[10035] #--SMC A1
    smc2     = data[10036] #--SMC A1

    DATA = {}
    DATA['COMPASS1']  = pd.DataFrame(compass1)
    DATA['COMPASS2']  = pd.DataFrame(compass2)
    DATA['EMC']       = pd.DataFrame(emc)
    DATA['SMC1']      = pd.DataFrame(smc1)
    DATA['SMC2']       = pd.DataFrame(smc2)

    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'p A1')
            PLOT[exp] = get_plot(query)

        nbins = len(query)
        theory = get_theory(PLOT,nbins,funcX=True,funcQ2=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax11.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

    #--plot theory interpolated between data points
    for key in theory['value']:
        X   = theory['X'][key]
        mean = theory['value'][key] + N*float(key)
        std  = theory['std'][key]
        down = mean - std
        up   = mean + std
        thy_plot ,= ax11.plot(X,mean,linestyle='solid',color='black')
        thy_band  = ax11.fill_between(X,down,up,color='gold',alpha=1.0)
        ax11.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')

    #--plot labels
    ax11.text(0.52, 0.89,  r'$Q^2 > 60~{\rm GeV} \, (i=6)$'  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.51, 0.75,  r'$40 < Q^2 < 60$'                , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.44, 0.59,  r'$20 < Q^2 < 40$'                , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.27, 0.44,  r'$10 < Q^2 < 20$'                , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.68, 0.32,  r'$5 < Q^2 < 10$'                 , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.57, 0.17,  r'$2 < Q^2 < 5$'                  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.27, 0.03,  r'$1 < Q^2 < 2\, (i=0)$'          , transform = ax11.transAxes, fontsize = 25)

    ax11.semilogx()
    ax11.set_xlim(0.005, 0.7)
    ax11.set_ylim(-0.2, 7.2)

    ax11.tick_params(axis = 'both', labelsize = 30)

    ax11.yaxis.set_tick_params(which = 'major', length = 10)
    ax11.yaxis.set_tick_params(which = 'minor', length = 5)

    ax11.xaxis.set_tick_params(which = 'major', length = 10)
    ax11.xaxis.set_tick_params(which = 'minor', length = 5)
    ax11.set_xticks([0.01,0.1])
    ax11.set_xticklabels([r'$0.01$',r'$0.1$'])

    ax11.text(0.05,0.65,r'\boldmath$A_1^p$'     , transform = ax11.transAxes, size = 60)
    ax11.text(0.16,0.66,r'$(+\, i)$', transform = ax11.transAxes, size = 40)

    ax11.set_xlabel(r'\boldmath$x$', size=40)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.yaxis.set_ticks_position('both')
    ax11.xaxis.set_ticks_position('both')
    ax11.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax11.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax11.yaxis.set_major_locator(majorLocator)
    ax11.yaxis.set_minor_locator(minorLocator)

    handles = [hand['COMPASS1'],hand['EMC'],hand['SMC1'],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{COMPASS}}'
    label2  = r'\textbf{\textrm{EMC}}'
    label3  = r'\textbf{\textrm{SMC}}'
    label4  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2,label3,label4]
    ax11.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #######################################
    #--Plot Apa from  HERMES, SLAC E143, SLAC E155, and SLAC E80E130
    #######################################

    hermes      = data[10007] #--HERMES Apa
    slace143    = data[10022] #--SLAC E143 Apa
    slace155    = data[10029] #--SLAC E155 Apa
    slace80e130 = data[10032] #--SLAC E80E130 Apa

    DATA = {}
    DATA['HERMES']      = pd.DataFrame(hermes)
    DATA['SLACE143']    = pd.DataFrame(slace143)
    DATA['SLACE155']    = pd.DataFrame(slace155)
    DATA['SLACE80E130'] = pd.DataFrame(slace80e130)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'p Apa')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax12.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax12.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax12.fill_between(X,down,up,color='gold',alpha=1.0)
            ax12.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax12.text(0.46, 0.92,  r'$Q^2 > 20~{\rm GeV} \, (i=9)$'  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.46, 0.81,  r'$15 < Q^2 < 20$'                , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.36, 0.69,  r'$10 < Q^2 < 15$'                , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.36, 0.57,  r'$7 < Q^2 < 10$'                 , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.28, 0.47,  r'$5 < Q^2 < 7$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.19, 0.36,  r'$4 < Q^2 < 5$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.26, 0.25,  r'$3 < Q^2 < 4$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.10, 0.13,  r'$2 < Q^2 < 3$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.73, 0.04,  r'$1 < Q^2 < 2\, (i=0)$'          , transform = ax12.transAxes, fontsize = 25)


    ax12.semilogx()
    ax12.set_xlim(0.01, 0.90)
    ax12.set_ylim(-0.2, 9.0)

    ax12.set_xlabel(r'\boldmath$x$', size=40)
    ax12.xaxis.set_label_coords(0.95,0.00)

    ax12.tick_params(axis = 'both', labelsize = 30)

    ax12.yaxis.set_tick_params(which = 'major', length = 10)
    ax12.yaxis.set_tick_params(which = 'minor', length = 5)

    ax12.xaxis.set_tick_params(which = 'major', length = 10)
    ax12.xaxis.set_tick_params(which = 'minor', length = 5)
    ax12.set_xticks([0.02,0.1,0.5])
    ax12.set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])

    ax12.text(0.05,0.65,r'\boldmath$A_{\parallel}^p$' , transform = ax12.transAxes, size = 60)
    ax12.text(0.16,0.66,r'$(+\, i)$'      , transform = ax12.transAxes, size = 40)

    ax12.yaxis.set_ticks_position('both')
    ax12.xaxis.set_ticks_position('both')
    ax12.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax12.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax12.yaxis.set_major_locator(majorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)

    handles = [hand['HERMES'],hand['SLACE143'],hand['SLACE155'],hand['SLACE80E130']]
    label1  = r'\textbf{\textrm{HERMES}}'
    label2  = r'\textbf{\textrm{SLAC E143}}'
    label3  = r'\textbf{\textrm{SLAC E155}}'
    label4  = r'\textbf{\textrm{SLAC E80E130}}'
    labels  = [label1,label2,label3,label4]
    ax12.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #################################
    #--Plot Apa from JLab EG1b
    #################################

    eg1b1 = data[10043] #--JLab EG1b Apa (E=4.2)
    eg1b2 = data[10044] #--JLab EG1b Apa (E=5.7)

    DATA = {}
    DATA['EG1b4.2'] = pd.DataFrame(eg1b1)
    DATA['EG1b5.7'] = pd.DataFrame(eg1b2)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'p Apa EG1b')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 2.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax21.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax21.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax21.fill_between(X,down,up,color='gold',alpha=1.0)
            ax21.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax21.text(0.49, 0.78,  r'$Q^2 > 4.5~{\rm GeV} \, (i=9)$'  , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.46, 0.69,  r'$4.0 < Q^2 < 4.5$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.34, 0.60,  r'$3.5 < Q^2 < 4.0$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.20, 0.50,  r'$3.0 < Q^2 < 3.5$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.67, 0.40,  r'$2.5 < Q^2 < 3.0$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.57, 0.31,  r'$2.2 < Q^2 < 2.5$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.49, 0.21,  r'$2.0 < Q^2 < 2.2$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.31, 0.12,  r'$1.8 < Q^2 < 2.0$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.40, 0.03,  r'$1.0 < Q^2 < 1.8\, (i=0)$'       , transform = ax21.transAxes, fontsize = 25)


    ax21.set_xlim(0.18, 0.65)
    ax21.set_ylim(-0.5, 21)

    ax21.set_xlabel(r'\boldmath$x$', size=40)
    ax21.xaxis.set_label_coords(0.95,0.00)

    ax21.tick_params(axis = 'both', labelsize = 30)

    ax21.yaxis.set_tick_params(which = 'major', length = 10)
    ax21.yaxis.set_tick_params(which = 'minor', length = 5)

    ax21.xaxis.set_tick_params(which = 'major', length = 10)
    ax21.xaxis.set_tick_params(which = 'minor', length = 5)

    ax21.text(0.05,0.70,r'\boldmath$A_{\parallel}^p$' , transform = ax21.transAxes, size = 60)
    ax21.text(0.16,0.71,r'$(+\, 2 i)$'                , transform = ax21.transAxes, size = 40)

    ax21.yaxis.set_ticks_position('both')
    ax21.xaxis.set_ticks_position('both')
    ax21.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax21.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(4.00)
    minorLocator = MultipleLocator(0.80)
    ax21.yaxis.set_major_locator(majorLocator)
    ax21.yaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(0.20)
    minorLocator = MultipleLocator(0.04)
    ax21.xaxis.set_major_locator(majorLocator)
    ax21.xaxis.set_minor_locator(minorLocator)

    handles = [hand['EG1b4.2'],hand['EG1b5.7']]
    label1  = r'\textbf{\textrm{E=4.2 GeV}}'
    label2  = r'\textbf{\textrm{E=5.7 GeV}}'
    labels  = [label1,label2]
    ax21.legend(handles,labels,loc=(0.02,0.80), fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)
    ax21.text(0.05,0.93,r'\textrm{\textbf{JLab EG1b}}',transform=ax21.transAxes,size=40)

    #################################
    #--Plot Apa from JLab DVCS
    #################################

    dvcs = data[10017] #--JLab DVCS (E=4.8,6.0)

    DVCS = pd.DataFrame(dvcs)

    DATA = {}
    DATA['DVCS4.8'] = DVCS.query("Elab < 5.0")
    DATA['DVCS6.0'] = DVCS.query("Elab > 5.0")

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'p Apa DVCS')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax22.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax22.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax22.fill_between(X,down,up,color='gold',alpha=1.0)
            ax22.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax22.text(0.46, 0.93,  r'$Q^2 > 4.5~{\rm GeV} \, (i=7)$'  , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.34, 0.78,  r'$3.5 < Q^2 < 4.5$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.73, 0.63,  r'$3.0 < Q^2 < 3.5$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.65, 0.49,  r'$2.5 < Q^2 < 3.0$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.57, 0.35,  r'$2.2 < Q^2 < 2.5$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.49, 0.20,  r'$2.0 < Q^2 < 2.2$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.40, 0.05,  r'$1.0 < Q^2 < 2.0\, (i=0)$'     , transform = ax22.transAxes, fontsize = 25)


    ax22.set_xlim(0.18, 0.65)
    ax22.set_ylim(-0.2, 7)

    ax22.set_xlabel(r'\boldmath$x$', size=40)
    ax22.xaxis.set_label_coords(0.95,0.00)

    ax22.tick_params(axis = 'both', labelsize = 30)

    ax22.yaxis.set_tick_params(which = 'major', length = 10)
    ax22.yaxis.set_tick_params(which = 'minor', length = 5)

    ax22.xaxis.set_tick_params(which = 'major', length = 10)
    ax22.xaxis.set_tick_params(which = 'minor', length = 5)

    ax22.text(0.05,0.70,r'\boldmath$A_{\parallel}^p$' , transform = ax22.transAxes, size = 60)
    ax22.text(0.16,0.71,r'$(+\, i)$'                , transform = ax22.transAxes, size = 40)

    ax22.yaxis.set_ticks_position('both')
    ax22.xaxis.set_ticks_position('both')
    ax22.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax22.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax22.yaxis.set_major_locator(majorLocator)
    ax22.yaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(0.20)
    minorLocator = MultipleLocator(0.04)
    ax22.xaxis.set_major_locator(majorLocator)
    ax22.xaxis.set_minor_locator(minorLocator)

    handles = [hand['DVCS4.8'],hand['DVCS6.0']]
    label1  = r'\textbf{\textrm{E=4.8 GeV}}'
    label2  = r'\textbf{\textrm{E=6.0 GeV}}'
    labels  = [label1,label2]
    ax22.legend(handles,labels,loc=(0.02,0.80), fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)
    ax22.text(0.05,0.93,r'\textrm{\textbf{JLab DVCS}}',transform=ax22.transAxes,size=40)

    py.tight_layout()
    py.savefig('%s/gallery/pidis-proton-A1-Apa.png' % (wdir))
    print('Saving figure to %s/gallery/pidis-proton-A1-Apa.png'%(wdir))
    py.close()

def plot_deuteron_A1_Apa(wdir, data, kc, istep):

    if 10001 not in data: return #--COMPASS A1
    if 10006 not in data: return #--HERMES Apa
    if 10016 not in data: return #--JLab DVCS Apa
    if 10021 not in data: return #--SLAC E143 Apa
    if 10027 not in data: return #--SLAC E155 Apa
    if 10033 not in data: return #--SMC A1
    if 10034 not in data: return #--SMC A1
    if 10039 not in data: return #--JLab EG1b Apa (E=4.2)
    if 10040 not in data: return #--JLab EG1b Apa (E=5.7)

    nrows, ncols = 2, 2
    py.figure(figsize = (ncols * 12.0, nrows * 14.0))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)
    ax21 = py.subplot(nrows, ncols, 3)
    ax22 = py.subplot(nrows, ncols, 4)

    #######################################
    #--Plot A1 from COMPASS and SMC
    #######################################

    compass1 = data[10001] #--COMPASS A1
    smc1     = data[10033] #--SMC A1
    smc2     = data[10034] #--SMC A1

    DATA = {}
    DATA['COMPASS1']  = pd.DataFrame(compass1)
    DATA['SMC1']      = pd.DataFrame(smc1)
    DATA['SMC2']      = pd.DataFrame(smc2)

    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'d A1')
            PLOT[exp] = get_plot(query)

        nbins = len(query)
        theory = get_theory(PLOT,nbins,funcX=True,funcQ2=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax11.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

    #--plot theory interpolated between data points
    for key in theory['value']:
        X   = theory['X'][key]
        mean = theory['value'][key] + N*float(key)
        std  = theory['std'][key]
        down = mean - std
        up   = mean + std
        thy_plot ,= ax11.plot(X,mean,linestyle='solid',color='black')
        thy_band  = ax11.fill_between(X,down,up,color='gold',alpha=1.0)
        ax11.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')

    #--plot labels
    ax11.text(0.56, 0.86,  r'$Q^2 > 40~{\rm GeV} \, (i=4)$'  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.44, 0.62,  r'$20 < Q^2 < 40$'                , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.35, 0.43,  r'$10 < Q^2 < 20$'                , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.57, 0.25,  r'$5 < Q^2 < 10$'                 , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.41, 0.03,  r'$m_c^2 < Q^2 < 5\, (i=0)$'      , transform = ax11.transAxes, fontsize = 25)

    ax11.semilogx()
    ax11.set_xlim(0.005, 0.7)
    ax11.set_ylim(-0.2, 5.0)

    ax11.tick_params(axis = 'both', labelsize = 30)

    ax11.yaxis.set_tick_params(which = 'major', length = 10)
    ax11.yaxis.set_tick_params(which = 'minor', length = 5)

    ax11.xaxis.set_tick_params(which = 'major', length = 10)
    ax11.xaxis.set_tick_params(which = 'minor', length = 5)
    ax11.set_xticks([0.01,0.1])
    ax11.set_xticklabels([r'$0.01$',r'$0.1$'])

    ax11.text(0.05,0.65,r'\boldmath$A_1^d$'     , transform = ax11.transAxes, size = 60)
    ax11.text(0.16,0.66,r'$(+\, i)$', transform = ax11.transAxes, size = 40)

    ax11.set_xlabel(r'\boldmath$x$', size=40)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.yaxis.set_ticks_position('both')
    ax11.xaxis.set_ticks_position('both')
    ax11.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax11.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax11.yaxis.set_major_locator(majorLocator)
    ax11.yaxis.set_minor_locator(minorLocator)

    handles = [hand['COMPASS1'],hand['SMC1'],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{COMPASS}}'
    label2  = r'\textbf{\textrm{SMC}}'
    label3  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2,label3]
    ax11.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #######################################
    #--Plot Apa from  HERMES, SLAC E143, and SLAC E155
    #######################################

    hermes      = data[10006] #--HERMES Apa
    slace143    = data[10021] #--SLAC E143 Apa
    slace155    = data[10027] #--SLAC E155 Apa

    DATA = {}
    DATA['HERMES']      = pd.DataFrame(hermes)
    DATA['SLACE143']    = pd.DataFrame(slace143)
    DATA['SLACE155']    = pd.DataFrame(slace155)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'d Apa')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax12.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax12.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax12.fill_between(X,down,up,color='gold',alpha=1.0)
            ax12.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax12.text(0.46, 0.92,  r'$Q^2 > 20~{\rm GeV} \, (i=9)$'  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.46, 0.80,  r'$15 < Q^2 < 20$'                , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.36, 0.68,  r'$10 < Q^2 < 15$'                , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.36, 0.57,  r'$7 < Q^2 < 10$'                 , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.29, 0.46,  r'$5 < Q^2 < 7$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.22, 0.35,  r'$4 < Q^2 < 5$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.23, 0.24,  r'$3 < Q^2 < 4$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.10, 0.13,  r'$2 < Q^2 < 3$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.73, 0.04,  r'$m_c^2 < Q^2 < 2\, (i=0)$'      , transform = ax12.transAxes, fontsize = 25)


    ax12.semilogx()
    ax12.set_xlim(0.01, 0.90)
    ax12.set_ylim(-0.2, 9.0)

    ax12.set_xlabel(r'\boldmath$x$', size=40)
    ax12.xaxis.set_label_coords(0.95,0.00)

    ax12.tick_params(axis = 'both', labelsize = 30)

    ax12.yaxis.set_tick_params(which = 'major', length = 10)
    ax12.yaxis.set_tick_params(which = 'minor', length = 5)

    ax12.xaxis.set_tick_params(which = 'major', length = 10)
    ax12.xaxis.set_tick_params(which = 'minor', length = 5)
    ax12.set_xticks([0.02,0.1,0.5])
    ax12.set_xticklabels([r'$0.02$',r'$0.1$',r'$0.5$'])

    ax12.text(0.05,0.65,r'\boldmath$A_{\parallel}^d$' , transform = ax12.transAxes, size = 60)
    ax12.text(0.16,0.66,r'$(+\, i)$'      , transform = ax12.transAxes, size = 40)

    ax12.yaxis.set_ticks_position('both')
    ax12.xaxis.set_ticks_position('both')
    ax12.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax12.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax12.yaxis.set_major_locator(majorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)

    handles = [hand['HERMES'],hand['SLACE143'],hand['SLACE155']]
    label1  = r'\textbf{\textrm{HERMES}}'
    label2  = r'\textbf{\textrm{SLAC E143}}'
    label3  = r'\textbf{\textrm{SLAC E155}}'
    labels  = [label1,label2,label3]
    ax12.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #################################
    #--Plot Apa from JLab EG1b
    #################################

    eg1b1 = data[10039] #--JLab EG1b Apa (E=4.2)
    eg1b2 = data[10040] #--JLab EG1b Apa (E=5.7)

    DATA = {}
    DATA['EG1b4.2'] = pd.DataFrame(eg1b1)
    DATA['EG1b5.7'] = pd.DataFrame(eg1b2)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'d Apa EG1b')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 2.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax21.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax21.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax21.fill_between(X,down,up,color='gold',alpha=1.0)
            ax21.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax21.text(0.52, 0.78,  r'$Q^2 > 4.5~{\rm GeV} \, (i=9)$'  , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.46, 0.69,  r'$4.0 < Q^2 < 4.5$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.36, 0.59,  r'$3.5 < Q^2 < 4.0$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.22, 0.49,  r'$3.0 < Q^2 < 3.5$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.67, 0.40,  r'$2.5 < Q^2 < 3.0$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.57, 0.31,  r'$2.2 < Q^2 < 2.5$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.49, 0.21,  r'$2.0 < Q^2 < 2.2$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.43, 0.12,  r'$1.8 < Q^2 < 2.0$'               , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.40, 0.03,  r'$m_c^2 < Q^2 < 1.8\, (i=0)$'     , transform = ax21.transAxes, fontsize = 25)


    ax21.set_xlim(0.18, 0.65)
    ax21.set_ylim(-0.5, 21)

    ax21.set_xlabel(r'\boldmath$x$', size=40)
    ax21.xaxis.set_label_coords(0.95,0.00)

    ax21.tick_params(axis = 'both', labelsize = 30)

    ax21.yaxis.set_tick_params(which = 'major', length = 10)
    ax21.yaxis.set_tick_params(which = 'minor', length = 5)

    ax21.xaxis.set_tick_params(which = 'major', length = 10)
    ax21.xaxis.set_tick_params(which = 'minor', length = 5)

    ax21.text(0.05,0.70,r'\boldmath$A_{\parallel}^d$' , transform = ax21.transAxes, size = 60)
    ax21.text(0.16,0.71,r'$(+\, 2 i)$'                , transform = ax21.transAxes, size = 40)

    ax21.yaxis.set_ticks_position('both')
    ax21.xaxis.set_ticks_position('both')
    ax21.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax21.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(4.00)
    minorLocator = MultipleLocator(0.80)
    ax21.yaxis.set_major_locator(majorLocator)
    ax21.yaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(0.20)
    minorLocator = MultipleLocator(0.04)
    ax21.xaxis.set_major_locator(majorLocator)
    ax21.xaxis.set_minor_locator(minorLocator)

    handles = [hand['EG1b4.2'],hand['EG1b5.7']]
    label1  = r'\textbf{\textrm{E=4.2 GeV}}'
    label2  = r'\textbf{\textrm{E=5.7 GeV}}'
    labels  = [label1,label2]
    ax21.legend(handles,labels,loc=(0.02,0.80), fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)
    ax21.text(0.05,0.93,r'\textrm{\textbf{JLab EG1b}}',transform=ax21.transAxes,size=40)

    #################################
    #--Plot Apa from JLab DVCS
    #################################

    dvcs = data[10016] #--JLab DVCS (E=4.8,6.0)

    DVCS = pd.DataFrame(dvcs)

    DATA = {}
    DATA['DVCS5.7'] = DVCS 

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'d Apa DVCS')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax22.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax22.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax22.fill_between(X,down,up,color='gold',alpha=1.0)
            ax22.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax22.text(0.42, 0.90,  r'$Q^2 > 3.5~{\rm GeV} \, (i=5)$'  , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.30, 0.72,  r'$3.0 < Q^2 < 3.5$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.15, 0.55,  r'$2.5 < Q^2 < 3.0$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.65, 0.39,  r'$2.2 < Q^2 < 2.5$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.55, 0.22,  r'$2.0 < Q^2 < 2.2$'               , transform = ax22.transAxes, fontsize = 25)
    ax22.text(0.44, 0.05,  r'$m_c^2 < Q^2 < 2.0\, (i=0)$'     , transform = ax22.transAxes, fontsize = 25)


    ax22.set_xlim(0.18, 0.58)
    ax22.set_ylim(-0.2, 5.9)

    ax22.set_xlabel(r'\boldmath$x$', size=40)
    ax22.xaxis.set_label_coords(0.95,0.00)

    ax22.tick_params(axis = 'both', labelsize = 30)

    ax22.yaxis.set_tick_params(which = 'major', length = 10)
    ax22.yaxis.set_tick_params(which = 'minor', length = 5)

    ax22.xaxis.set_tick_params(which = 'major', length = 10)
    ax22.xaxis.set_tick_params(which = 'minor', length = 5)

    ax22.text(0.05,0.70,r'\boldmath$A_{\parallel}^d$' , transform = ax22.transAxes, size = 60)
    ax22.text(0.16,0.71,r'$(+\, i)$'                , transform = ax22.transAxes, size = 40)

    ax22.yaxis.set_ticks_position('both')
    ax22.xaxis.set_ticks_position('both')
    ax22.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax22.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax22.yaxis.set_major_locator(majorLocator)
    ax22.yaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(0.20)
    minorLocator = MultipleLocator(0.04)
    ax22.xaxis.set_major_locator(majorLocator)
    ax22.xaxis.set_minor_locator(minorLocator)

    handles = [hand['DVCS5.7']]
    label1  = r'\textbf{\textrm{E=5.7 GeV}}'
    labels  = [label1]
    ax22.legend(handles,labels,loc=(0.02,0.85), fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)
    ax22.text(0.05,0.93,r'\textrm{\textbf{JLab DVCS}}',transform=ax22.transAxes,size=40)

    py.tight_layout()
    py.savefig('%s/gallery/pidis-deuteron-A1-Apa.png' % (wdir))
    print('Saving figure to %s/gallery/pidis-deuteron-A1-Apa.png'%(wdir))
    py.close()

def plot_proton_A2_Ape(wdir, data, kc, istep):

    if 10008 not in data: return #--HERMES A2
    if 10023 not in data: return #--SLAC E143 Ape
    if 10028 not in data: return #--SLAC E155 Ape
    if 10031 not in data: return #--SLAC E155x Atpe 

    nrows, ncols = 2, 2
    fig  = py.figure(figsize = (ncols * 12.0, nrows * 14.0))
    gs   = gridspec.GridSpec(2,4)
    ax11 = fig.add_subplot(gs[0,:2],)
    ax12 = fig.add_subplot(gs[0,2:])
    ax21 = fig.add_subplot(gs[1,1:3])

    #######################################
    #--Plot A2 from HERMES
    #######################################

    hermes = data[10008] #--HERMES A2

    DATA = {}
    DATA['HERMES']  = pd.DataFrame(hermes)

    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'p A2')
            PLOT[exp] = get_plot(query)

        nbins = len(query)
        theory = get_theory(PLOT,nbins,funcX=True,funcQ2=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax11.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

    #--plot theory interpolated between data points
    for key in theory['value']:
        X   = theory['X'][key]
        mean = theory['value'][key] + N*float(key)
        std  = theory['std'][key]
        down = mean - std
        up   = mean + std
        thy_plot ,= ax11.plot(X,mean,linestyle='solid',color='black')
        thy_band  = ax11.fill_between(X,down,up,color='gold',alpha=1.0)
        ax11.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')

    #--plot labels
    ax11.text(0.25, 0.74,  r'$Q^2 > 5~{\rm GeV} \, (i=2)$'   , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.05, 0.40,  r'$2 < Q^2 < 5$'                  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.69, 0.09,  r'$m_c^2 < Q^2 < 2\, (i=0)$'      , transform = ax11.transAxes, fontsize = 25)

    ax11.semilogx()
    ax11.set_xlim(0.04, 0.8)
    ax11.set_ylim(-0.2, 2.8)

    ax11.tick_params(axis = 'both', labelsize = 30)

    ax11.yaxis.set_tick_params(which = 'major', length = 10)
    ax11.yaxis.set_tick_params(which = 'minor', length = 5)

    ax11.xaxis.set_tick_params(which = 'major', length = 10)
    ax11.xaxis.set_tick_params(which = 'minor', length = 5)
    ax11.set_xticks([0.05,0.1,0.5])
    ax11.set_xticklabels([r'$0.03$',r'$0.1$','$0.5$'])

    ax11.text(0.05,0.80,r'\boldmath$A_2^p$'     , transform = ax11.transAxes, size = 60)
    ax11.text(0.16,0.81,r'$(+\, i)$', transform = ax11.transAxes, size = 40)

    ax11.set_xlabel(r'\boldmath$x$', size=40)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.yaxis.set_ticks_position('both')
    ax11.xaxis.set_ticks_position('both')
    ax11.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax11.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(1.00)
    minorLocator = MultipleLocator(0.20)
    ax11.yaxis.set_major_locator(majorLocator)
    ax11.yaxis.set_minor_locator(minorLocator)

    handles = [hand['HERMES'],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{HERMES}}'
    label2  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2]
    ax11.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #######################################
    #--Plot Ape from SLAC E143 and SLAC E155
    #######################################

    slace143    = data[10023] #--SLAC E143 Ape
    slace155    = data[10028] #--SLAC E155 Ape

    DATA = {}
    DATA['SLACE143']    = pd.DataFrame(slace143)
    DATA['SLACE155']    = pd.DataFrame(slace155)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'p Ape')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax12.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax12.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax12.fill_between(X,down,up,color='gold',alpha=1.0)
            ax12.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax12.text(0.56, 0.88,  r'$Q^2 > 20~{\rm GeV} \, (i=8)$'  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.49, 0.77,  r'$15 < Q^2 < 20$'                , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.30, 0.67,  r'$10 < Q^2 < 15$'                , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.21, 0.57,  r'$7 < Q^2 < 10$'                 , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.18, 0.45,  r'$5 < Q^2 < 7$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.06, 0.34,  r'$4 < Q^2 < 5$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.81, 0.23,  r'$3 < Q^2 < 4$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.60, 0.13,  r'$2 < Q^2 < 3$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.17, 0.02,  r'$m_c^2 < Q^2 < 2\, (i=0)$'      , transform = ax12.transAxes, fontsize = 25)


    ax12.semilogx()
    ax12.set_xlim(0.04, 0.90)
    ax12.set_ylim(-0.2, 9.0)

    ax12.set_xlabel(r'\boldmath$x$', size=40)
    ax12.xaxis.set_label_coords(0.95,0.00)

    ax12.tick_params(axis = 'both', labelsize = 30)

    ax12.yaxis.set_tick_params(which = 'major', length = 10)
    ax12.yaxis.set_tick_params(which = 'minor', length = 5)

    ax12.xaxis.set_tick_params(which = 'major', length = 10)
    ax12.xaxis.set_tick_params(which = 'minor', length = 5)
    ax12.set_xticks([0.05,0.1,0.5])
    ax12.set_xticklabels([r'$0.05$',r'$0.1$',r'$0.5$'])

    ax12.text(0.05,0.80,r'\boldmath$A_{\perp}^p$' , transform = ax12.transAxes, size = 60)
    ax12.text(0.16,0.81,r'$(+\, i)$'      , transform = ax12.transAxes, size = 40)

    ax12.yaxis.set_ticks_position('both')
    ax12.xaxis.set_ticks_position('both')
    ax12.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax12.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax12.yaxis.set_major_locator(majorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)

    handles = [hand['SLACE143'],hand['SLACE155']]
    label1  = r'\textbf{\textrm{SLAC E143}}'
    label2  = r'\textbf{\textrm{SLAC E155}}'
    labels  = [label1,label2]
    ax12.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #################################
    #--Plot Atpe from SLAC E155x
    #################################

    slace155x = data[10031] #--SLAC E155x Atpe

    DATA = {}
    DATA['SLAC E155x'] = pd.DataFrame(slace155x)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'p Atpe')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax21.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax21.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax21.fill_between(X,down,up,color='gold',alpha=1.0)
            ax21.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax21.text(0.47, 0.90,  r'$Q^2 > 15~{\rm GeV} \, (i=7)$'  , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.36, 0.78,  r'$10 < Q^2 < 15$'                , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.22, 0.65,  r'$7 < Q^2 < 10$'                 , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.15, 0.52,  r'$5 < Q^2 < 7$'                  , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.59, 0.42,  r'$4 < Q^2 < 5$'                  , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.39, 0.29,  r'$3 < Q^2 < 4$'                  , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.66, 0.17,  r'$2 < Q^2 < 3$'                  , transform = ax21.transAxes, fontsize = 25)
    ax21.text(0.68, 0.05,  r'$m_c^2 < Q^2 < 2\, (i=0)$'      , transform = ax21.transAxes, fontsize = 25)


    ax21.semilogx()
    ax21.set_xlim(0.05, 0.95)
    ax21.set_ylim(-0.5, 7.8)

    ax21.set_xlabel(r'\boldmath$x$', size=40)
    ax21.xaxis.set_label_coords(0.95,0.00)

    ax21.tick_params(axis = 'both', labelsize = 30)

    ax21.yaxis.set_tick_params(which = 'major', length = 10)
    ax21.yaxis.set_tick_params(which = 'minor', length = 5)

    ax21.xaxis.set_tick_params(which = 'major', length = 10)
    ax21.xaxis.set_tick_params(which = 'minor', length = 5)
    ax21.set_xticks([0.06,0.1,0.5])
    ax21.set_xticklabels([r'$0.06$',r'$0.1$',r'$0.5$'])

    ax21.text(0.05,0.80,r'\boldmath$\tilde{A}_{\perp}^p$' , transform = ax21.transAxes, size = 60)
    ax21.text(0.16,0.81,r'$(+\, i)$'                , transform = ax21.transAxes, size = 40)

    ax21.yaxis.set_ticks_position('both')
    ax21.xaxis.set_ticks_position('both')
    ax21.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax21.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax21.yaxis.set_major_locator(majorLocator)
    ax21.yaxis.set_minor_locator(minorLocator)

    handles = [hand['SLAC E155x']]
    label1  = r'\textbf{\textrm{SLAC E155x}}'
    labels  = [label1]
    ax21.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()
    py.savefig('%s/gallery/pidis-proton-A2-Ape.png' % (wdir))
    print('Saving figure to %s/gallery/pidis-proton-A2-Ape.png'%(wdir))
    py.close()

def plot_deuteron_A2_Ape(wdir, data, kc, istep):

    if 10020 not in data: return #--SLAC E143 Ape
    if 10026 not in data: return #--SLAC E155 Ape
    if 10030 not in data: return #--SLAC E155x Atpe

    nrows, ncols = 1, 2
    py.figure(figsize = (ncols * 12.0, nrows * 14.0))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)

    #######################################
    #--Plot Ape from SLAC E143 and SLAC E155
    #######################################

    slace143    = data[10020] #--SLAC E143 Ape
    slace155    = data[10026] #--SLAC E155 Ape

    DATA = {}
    DATA['SLACE143']    = pd.DataFrame(slace143)
    DATA['SLACE155']    = pd.DataFrame(slace155)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'d Ape')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax11.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax11.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax11.fill_between(X,down,up,color='gold',alpha=1.0)
            ax11.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax11.text(0.56, 0.88,  r'$Q^2 > 20~{\rm GeV} \, (i=8)$'  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.49, 0.77,  r'$15 < Q^2 < 20$'                , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.30, 0.67,  r'$10 < Q^2 < 15$'                , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.21, 0.57,  r'$7 < Q^2 < 10$'                 , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.18, 0.45,  r'$5 < Q^2 < 7$'                  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.06, 0.34,  r'$4 < Q^2 < 5$'                  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.81, 0.23,  r'$3 < Q^2 < 4$'                  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.60, 0.13,  r'$2 < Q^2 < 3$'                  , transform = ax11.transAxes, fontsize = 25)
    ax11.text(0.17, 0.02,  r'$m_c^2 < Q^2 < 2\, (i=0)$'      , transform = ax11.transAxes, fontsize = 25)


    ax11.semilogx()
    ax11.set_xlim(0.04, 0.90)
    ax11.set_ylim(-0.2, 9.0)

    ax11.set_xlabel(r'\boldmath$x$', size=40)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.tick_params(axis = 'both', labelsize = 30)

    ax11.yaxis.set_tick_params(which = 'major', length = 10)
    ax11.yaxis.set_tick_params(which = 'minor', length = 5)

    ax11.xaxis.set_tick_params(which = 'major', length = 10)
    ax11.xaxis.set_tick_params(which = 'minor', length = 5)
    ax11.set_xticks([0.05,0.1,0.5])
    ax11.set_xticklabels([r'$0.05$',r'$0.1$',r'$0.5$'])

    ax11.text(0.05,0.75,r'\boldmath$A_{\perp}^d$' , transform = ax11.transAxes, size = 60)
    ax11.text(0.16,0.76,r'$(+\, i)$'      , transform = ax11.transAxes, size = 40)

    ax11.yaxis.set_ticks_position('both')
    ax11.xaxis.set_ticks_position('both')
    ax11.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax11.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax11.yaxis.set_major_locator(majorLocator)
    ax11.yaxis.set_minor_locator(minorLocator)

    handles = [hand['SLACE143'],hand['SLACE155'],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{SLAC E143}}'
    label2  = r'\textbf{\textrm{SLAC E155}}'
    label3  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2,label3]
    ax11.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #################################
    #--Plot Atpe from SLAC E155x
    #################################

    slace155x = data[10030] #--SLAC E155x Atpe

    DATA = {}
    DATA['SLAC E155x'] = pd.DataFrame(slace155x)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'d Atpe')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax12.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax12.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax12.fill_between(X,down,up,color='gold',alpha=1.0)
            ax12.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax12.text(0.47, 0.90,  r'$Q^2 > 15~{\rm GeV} \, (i=7)$'  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.36, 0.78,  r'$10 < Q^2 < 15$'                , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.22, 0.65,  r'$7 < Q^2 < 10$'                 , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.15, 0.52,  r'$5 < Q^2 < 7$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.59, 0.42,  r'$4 < Q^2 < 5$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.39, 0.29,  r'$3 < Q^2 < 4$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.66, 0.17,  r'$2 < Q^2 < 3$'                  , transform = ax12.transAxes, fontsize = 25)
    ax12.text(0.68, 0.05,  r'$m_c^2 < Q^2 < 2\, (i=0)$'      , transform = ax12.transAxes, fontsize = 25)


    ax12.semilogx()
    ax12.set_xlim(0.05, 0.95)
    ax12.set_ylim(-0.5, 7.8)

    ax12.set_xlabel(r'\boldmath$x$', size=40)
    ax12.xaxis.set_label_coords(0.95,0.00)

    ax12.tick_params(axis = 'both', labelsize = 30)

    ax12.yaxis.set_tick_params(which = 'major', length = 10)
    ax12.yaxis.set_tick_params(which = 'minor', length = 5)

    ax12.xaxis.set_tick_params(which = 'major', length = 10)
    ax12.xaxis.set_tick_params(which = 'minor', length = 5)
    ax12.set_xticks([0.06,0.1,0.5])
    ax12.set_xticklabels([r'$0.06$',r'$0.1$',r'$0.5$'])

    ax12.text(0.05,0.75,r'\boldmath$\tilde{A}_{\perp}^d$' , transform = ax12.transAxes, size = 60)
    ax12.text(0.16,0.76,r'$(+\, i)$'                , transform = ax12.transAxes, size = 40)

    ax12.yaxis.set_ticks_position('both')
    ax12.xaxis.set_ticks_position('both')
    ax12.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax12.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax12.yaxis.set_major_locator(majorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)

    handles = [hand['SLAC E155x']]
    label1  = r'\textbf{\textrm{SLAC E155x}}'
    labels  = [label1]
    ax12.legend(handles,labels,loc='upper left', fontsize = 35, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()
    py.savefig('%s/gallery/pidis-deuteron-A2-Ape.png' % (wdir))
    print('Saving figure to %s/gallery/pidis-deuteron-A2-Ape.png'%(wdir))
    py.close()

def plot_helium_neutron(wdir, data, kc, istep):

    if 10010 not in data: return #--JLab E06-014 Apa (h)
    if 10011 not in data: return #--JLab E06-014 Ape (h)
    if 10014 not in data: return #--JLab E99-117 Apa (h)
    if 10015 not in data: return #--JLab E99-117 Ape (h)
    if 10018 not in data: return #--SLAC E142 A1 (h)
    if 10019 not in data: return #--SLAC E142 A2 (h)
    if 10024 not in data: return #--SLAC E154 Ape (h)
    if 10025 not in data: return #--SLAC E154 Apa (h)
    if 10005 not in data: return #--HERMES A1 (n)

    nrows, ncols = 2, 2
    py.figure(figsize = (ncols * 6.0, nrows * 7.0))
    ax11 = py.subplot(nrows, ncols, 1)
    ax12 = py.subplot(nrows, ncols, 2)
    ax21 = py.subplot(nrows, ncols, 3)
    ax22 = py.subplot(nrows, ncols, 4)

    #######################################
    #--Plot A1 from SLAC E142
    #######################################

    slace142 = data[10018] #--SLAC E142 A1 (h)
    hermes   = data[10005] #--HERMES A1 (n)

    DATA = {}
    DATA['SLACE142']  = pd.DataFrame(slace142)
    DATA['HERMES']    = pd.DataFrame(hermes)

    Q2 = np.append(DATA['SLACE142']['Q2'],DATA['HERMES']['Q2'])
    Q2min,Q2max = np.round(np.min(Q2),1),np.round(np.max(Q2),1)

    theory = {}
    PLOT = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'h A1')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax11.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax11.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax11.fill_between(X,down,up,color='gold',alpha=1.0)
            ax11.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')

    #--plot labels
    ax11.text(0.05, 0.05,  r'$m_c^2 < Q^2 < 5.5$'      , transform = ax11.transAxes, fontsize = 30)

    ax11.semilogx()
    ax11.set_xlim(0.05, 0.6)
    ax11.set_ylim(-0.5, 0.5)

    ax11.tick_params(axis = 'both', labelsize = 20)

    ax11.yaxis.set_tick_params(which = 'major', length = 5)
    ax11.yaxis.set_tick_params(which = 'minor', length = 3)

    ax11.xaxis.set_tick_params(which = 'major', length = 5)
    ax11.xaxis.set_tick_params(which = 'minor', length = 3)
    ax11.set_xticks([0.1,0.4])
    ax11.set_xticklabels([r'$0.1$',r'$0.4$'])

    ax11.text(0.05,0.70,r'\boldmath$A_1$'     , transform = ax11.transAxes, size = 40)
    #ax11.text(0.16,0.71,r'$(+\, i)$', transform = ax11.transAxes, size = 20)

    ax11.set_xlabel(r'\boldmath$x$', size=30)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.yaxis.set_ticks_position('both')
    ax11.xaxis.set_ticks_position('both')
    ax11.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax11.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(0.20)
    minorLocator = MultipleLocator(0.04)
    ax11.yaxis.set_major_locator(majorLocator)
    ax11.yaxis.set_minor_locator(minorLocator)

    handles = [hand['HERMES'],hand['SLACE142'],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{HERMES \boldmath$n$}}'
    label2  = r'\textbf{\textrm{SLAC E142 \boldmath$^3$He}}'
    label3  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2,label3]
    ax11.legend(handles,labels,loc='upper left', fontsize = 20, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #######################################
    #--Plot Apa from  SLAC E154, JLab E06-014, JLab E99-117
    #######################################

    jlabe06014  = data[10010] #--JLab E06-014 Apa (h)
    jlabe99117  = data[10014] #--JLab E99-117 Apa (h)
    slace154    = data[10025] #--SLAC E154 Apa (h)

    DATA = {}
    DATA['SLACE154']       = pd.DataFrame(slace154)
    DATA['JLabE06E014']    = pd.DataFrame(jlabe06014)
    DATA['JLabE99E117']    = pd.DataFrame(jlabe99117)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'h Apa')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax12.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax12.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax12.fill_between(X,down,up,color='gold',alpha=1.0)
            ax12.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')

    #--plot labels
    ax12.text(0.55, 0.87,  r'$Q^2 > 10~{\rm GeV} \, (i=3)$', transform = ax12.transAxes, fontsize = 15)
    ax12.text(0.05, 0.57,  r'$5 < Q^2 < 10$'               , transform = ax12.transAxes, fontsize = 15)
    ax12.text(0.05, 0.34,  r'$3 < Q^2 < 5$'                , transform = ax12.transAxes, fontsize = 15)
    ax12.text(0.20, 0.04,  r'$m_c^2 < Q^2 < 3\, (i=0)$'    , transform = ax12.transAxes, fontsize = 15)


    ax12.semilogx()
    ax12.set_xlim(0.03, 0.90)
    ax12.set_ylim(-0.2, 3.6)

    ax12.set_xlabel(r'\boldmath$x$', size=30)
    ax12.xaxis.set_label_coords(0.95,0.00)

    ax12.tick_params(axis = 'both', labelsize = 20)

    ax12.yaxis.set_tick_params(which = 'major', length = 5)
    ax12.yaxis.set_tick_params(which = 'minor', length = 3)

    ax12.xaxis.set_tick_params(which = 'major', length = 5)
    ax12.xaxis.set_tick_params(which = 'minor', length = 3)
    ax12.set_xticks([0.1,0.5])
    ax12.set_xticklabels([r'$0.1$',r'$0.5$'])

    ax12.text(0.05,0.70,r'\boldmath$A_{\parallel}^h$' , transform = ax12.transAxes, size = 40)
    ax12.text(0.19,0.71,r'$(+\, i)$'      , transform = ax12.transAxes, size = 30)

    ax12.yaxis.set_ticks_position('both')
    ax12.xaxis.set_ticks_position('both')
    ax12.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax12.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax12.yaxis.set_major_locator(majorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)

    handles = [hand['SLACE154'],hand['JLabE06E014'],hand['JLabE99E117']]
    label1  = r'\textbf{\textrm{SLAC E154}}'
    label2  = r'\textbf{\textrm{JLab E06-014}}'
    label3  = r'\textbf{\textrm{SLAC E99-117}}'
    labels  = [label1,label2,label3]
    ax12.legend(handles,labels,loc='upper left', fontsize = 20, frameon = 0, handletextpad = 0.3, handlelength = 1.0)


    #################################
    #--Plot A2 from SLAC E142 
    #################################

    slace142 = data[10019] #--SLAC E142 A2 (h)

    DATA = {}
    DATA['SLACE142'] = pd.DataFrame(slace142)

    Q2 = DATA['SLACE142']['Q2']
    Q2min,Q2max = np.round(np.min(Q2),1),np.round(np.max(Q2),1)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'h A2')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax21.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax21.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax21.fill_between(X,down,up,color='gold',alpha=1.0)
            ax21.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')


    #--plot labels
    ax21.text(0.05, 0.05,  r'$m_c^2 < Q^2 < 5.5$'      , transform = ax21.transAxes, fontsize = 30)

    ax21.set_xlim(0.06, 0.50)
    ax21.set_ylim(-0.3, 0.3)

    ax21.set_xlabel(r'\boldmath$x$', size=30)
    ax21.xaxis.set_label_coords(0.95,0.00)

    ax21.tick_params(axis = 'both', labelsize = 20)

    ax21.yaxis.set_tick_params(which = 'major', length = 5)
    ax21.yaxis.set_tick_params(which = 'minor', length = 3)

    ax21.xaxis.set_tick_params(which = 'major', length = 5)
    ax21.xaxis.set_tick_params(which = 'minor', length = 3)

    ax21.text(0.05,0.70,r'\boldmath$A_2^h$' , transform = ax21.transAxes, size = 40)
    #ax21.text(0.16,0.71,r'$(+\, i)$'                , transform = ax21.transAxes, size = 30)

    ax21.yaxis.set_ticks_position('both')
    ax21.xaxis.set_ticks_position('both')
    ax21.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax21.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(0.20)
    minorLocator = MultipleLocator(0.04)
    ax21.yaxis.set_major_locator(majorLocator)
    ax21.yaxis.set_minor_locator(minorLocator)
    majorLocator = MultipleLocator(0.20)
    minorLocator = MultipleLocator(0.04)
    ax21.xaxis.set_major_locator(majorLocator)
    ax21.xaxis.set_minor_locator(minorLocator)

    handles = [hand['SLACE142']]
    label1  = r'\textbf{\textrm{SLAC E142}}'
    labels  = [label1]
    ax21.legend(handles,labels,loc='upper left', fontsize = 20, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    #######################################
    #--Plot Ape from  SLAC E154, JLab E06-014, JLab E99-117
    #######################################

    jlabe06014  = data[10011] #--JLab E06-014 Ape (h)
    jlabe99117  = data[10015] #--JLab E99-117 Ape (h)
    slace154    = data[10024] #--SLAC E154 Ape (h)

    DATA = {}
    DATA['SLACE154']       = pd.DataFrame(slace154)
    DATA['JLabE06E014']    = pd.DataFrame(jlabe06014)
    DATA['JLabE99E117']    = pd.DataFrame(jlabe99117)

    PLOT = {}
    theory = {}
    for cluster_i in range(kc.nc[istep]):
        if cluster_i != 0: continue

        for exp in DATA:
            query = get_Q2bins(DATA[exp],'h Ape')
            PLOT[exp] = get_plot(query)

            nbins = len(query)
            theory[exp] = get_theory(PLOT[exp],nbins,funcX=True,funcQ2=False,loop=False)

    N = 1.0
    hand = {}
    #--plot data points
    for exp in PLOT:
        for key in range(nbins):
            X     = PLOT[exp]['X'][key]
            val   = PLOT[exp]['value'][key] + N*float(key)
            alpha = PLOT[exp]['alpha'][key]
            color,marker,ms = get_details(exp)
            hand[exp] = ax22.errorbar(X,val,alpha,marker=marker,color=color,ms=ms,capsize=3,linestyle='none')

        #--plot theory interpolated between data points
        for key in theory[exp]['value']:
            X    = theory[exp]['X'][key]
            mean = theory[exp]['value'][key] + N*float(key)
            std  = theory[exp]['std'][key]
            down = mean - std
            up   = mean + std
            thy_plot ,= ax22.plot(X,mean,linestyle='solid',color=color)
            thy_band  = ax22.fill_between(X,down,up,color='gold',alpha=1.0)
            ax22.plot(np.linspace(np.min(X),np.max(X),10),np.ones(10)*float(N*key),color='purple',ls=':')

    #--plot labels
    ax22.text(0.55, 0.87,  r'$Q^2 > 10~{\rm GeV} \, (i=3)$', transform = ax22.transAxes, fontsize = 15)
    ax22.text(0.05, 0.57,  r'$5 < Q^2 < 10$'               , transform = ax22.transAxes, fontsize = 15)
    ax22.text(0.22, 0.34,  r'$3 < Q^2 < 5$'                , transform = ax22.transAxes, fontsize = 15)
    ax22.text(0.20, 0.04,  r'$m_c^2 < Q^2 < 3\, (i=0)$'    , transform = ax22.transAxes, fontsize = 15)


    ax22.semilogx()
    ax22.set_xlim(0.03, 0.90)
    ax22.set_ylim(-0.2, 3.6)

    ax22.set_xlabel(r'\boldmath$x$', size=30)
    ax22.xaxis.set_label_coords(0.95,0.00)

    ax22.tick_params(axis = 'both', labelsize = 20)

    ax22.yaxis.set_tick_params(which = 'major', length = 5)
    ax22.yaxis.set_tick_params(which = 'minor', length = 3)

    ax22.xaxis.set_tick_params(which = 'major', length = 5)
    ax22.xaxis.set_tick_params(which = 'minor', length = 3)
    ax22.set_xticks([0.1,0.5])
    ax22.set_xticklabels([r'$0.1$',r'$0.5$'])

    ax22.text(0.05,0.70,r'\boldmath$A_{\perp}^h$' , transform = ax22.transAxes, size = 40)
    ax22.text(0.19,0.71,r'$(+\, i)$'      , transform = ax22.transAxes, size = 30)

    ax22.yaxis.set_ticks_position('both')
    ax22.xaxis.set_ticks_position('both')
    ax22.tick_params(axis = 'y', which = 'both', labelleft = True, labelright = False, direction='in')
    ax22.tick_params(axis = 'x', which = 'both', labeltop = False, labelbottom = True, direction='in')

    majorLocator = MultipleLocator(2.00)
    minorLocator = MultipleLocator(0.40)
    ax22.yaxis.set_major_locator(majorLocator)
    ax22.yaxis.set_minor_locator(minorLocator)

    handles = [hand['SLACE154'],hand['JLabE06E014'],hand['JLabE99E117']]
    label1  = r'\textbf{\textrm{SLAC E154}}'
    label2  = r'\textbf{\textrm{JLab E06-014}}'
    label3  = r'\textbf{\textrm{SLAC E99-117}}'
    labels  = [label1,label2,label3]
    ax22.legend(handles,labels,loc='upper left', fontsize = 20, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()
    py.savefig('%s/gallery/pidis-helium.png' % (wdir))
    print('Saving figure to %s/gallery/pidis-helium.png'%(wdir))
    py.close()

def plot_obs(wdir, kc):

    print('\nplotting pidis data from %s' % (wdir))

    load_config('%s/input.py' % wdir)
    istep = core.get_istep()
    replicas = core.get_replicas(wdir)
    core.mod_conf(istep, replicas[0]) #--set conf as specified in istep

    predictions = load('%s/data/predictions-%d.dat' % (wdir, istep))
    if 'pidis' not in predictions['reactions']:
        print('inclusive polarized DIS is not in data file')
        return
    labels  = load('%s/data/labels-%d.dat' % (wdir, istep))
    cluster = labels['cluster']

    data = predictions['reactions']['pidis']

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
            if 'X' in data[idx]: data[idx]['x'] = data[idx]['X']
            data[idx]['rQ2'] = np.around(data[idx]['Q2'], decimals = 0)
            data[idx]['rx'] = np.around(data[idx]['x'], decimals = 2)


    plot_proton_A1_Apa  (wdir, data, kc, istep)
    plot_deuteron_A1_Apa(wdir, data, kc, istep)
    plot_proton_A2_Ape  (wdir, data, kc, istep)
    plot_deuteron_A2_Ape(wdir, data, kc, istep)
    plot_helium_neutron (wdir, data, kc, istep)

 
    return






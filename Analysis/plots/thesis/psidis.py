#!/usr/bin/env python
import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--matplotlib
import matplotlib
matplotlib.use('Agg')
import pylab as py
from matplotlib.ticker import MultipleLocator

#--from scipy stack 
from scipy.integrate import fixed_quad
from scipy import interpolate

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#-- from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from obslib
from obslib.psidis.reader import READER

import kmeanconf as kc

cwd = 'plots/thesis'

def plot_pion(wdir,kc):

    print('\ngenerating pion PSIDIS plots from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax21 = py.subplot(nrows,ncols,3)
    ax22 = py.subplot(nrows,ncols,4)

    filters = conf['datasets']['psidis']['filters']

    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['psidis']={}
    conf['datasets']['psidis']['filters']=filters
    conf['datasets']['psidis']['xlsx']={}
    conf['datasets']['psidis']['xlsx'][20004]='psidis/expdata/20004.xlsx'
    conf['datasets']['psidis']['xlsx'][20005]='psidis/expdata/20005.xlsx'
    conf['datasets']['psidis']['xlsx'][20017]='psidis/expdata/20017.xlsx'
    conf['datasets']['psidis']['xlsx'][20018]='psidis/expdata/20018.xlsx'
    conf['datasets']['psidis']['xlsx'][20008]='psidis/expdata/20008.xlsx'
    conf['datasets']['psidis']['xlsx'][20009]='psidis/expdata/20009.xlsx'
    conf['datasets']['psidis']['xlsx'][20021]='psidis/expdata/20021.xlsx'
    conf['datasets']['psidis']['xlsx'][20022]='psidis/expdata/20022.xlsx'
    conf['datasets']['psidis']['norm']={}
    conf['psidis tabs']=READER().load_data_sets('psidis')
    tabs = conf['psidis tabs']
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['psidis']

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in tabs:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    thy_band = {}
    #--plot data
    for idx in tabs:
        X = tabs[idx]['X']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==20004: ax,color = ax11,'firebrick'
        if idx==20005: ax,color = ax12,'firebrick' 
        if idx==20017: ax,color = ax11,'darkgreen'
        if idx==20018: ax,color = ax12,'darkgreen'
        if idx==20008: ax,color = ax21,'firebrick'
        if idx==20009: ax,color = ax22,'firebrick' 
        if idx==20021: ax,color = ax21,'darkgreen'
        if idx==20022: ax,color = ax22,'darkgreen'

        hand[idx] = ax.errorbar(X,values,yerr=alpha,color=color,fmt='o',ms=2,capsize=3.0)

        #if idx in [20004,20005,20008,20009]: continue

        for ic in range(nc):
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]
            down = thy - std
            up   = thy + std
            #thy_plot ,= ax.plot(X,thy,color=color)
            thy_band[idx]  = ax.fill_between(X,down,up,color=color,alpha=0.5)

    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_yticks([0.25,0.5,0.75])

    for ax in [ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        #ax.set_yticks([0.25,0.5,0.75])

    for ax in [ax11,ax12,ax21,ax22]:
        ax.semilogx()
        ax.set_xlim(0.004,0.6)
        ax.set_xticks([0.01,0.1])
        ax.set_xticklabels([r'$0.01$',r'$0.1$'])
        #minorLocator = MultipleLocator(0.1)
        #majorLocator = MultipleLocator(1.0)
        #ax.xaxis.set_minor_locator(minorLocator)
        #ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.axhline(0,0,1,color='black',alpha=0.2)

    for ax in [ax21,ax22]:
        ax.set_xlabel(r'\boldmath$x$',size=40)
        ax.xaxis.set_label_coords(0.90,0.00)

    for ax in [ax12,ax22]:
        ax.tick_params(axis='both',which='both',labelleft=False)

    ax11.set_ylim(-0.10,1.0)
    ax12.set_ylim(-0.10,1.0)
    ax21.set_ylim(-0.10,0.5)
    ax22.set_ylim(-0.10,0.5)

    minorLocator = MultipleLocator(0.10)
    majorLocator = MultipleLocator(0.20)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.10)
    majorLocator = MultipleLocator(0.20)
    ax21.yaxis.set_minor_locator(minorLocator)
    ax21.yaxis.set_major_locator(majorLocator)
    ax22.yaxis.set_minor_locator(minorLocator)
    ax22.yaxis.set_major_locator(majorLocator)

    #ax11.set_yticks([0,0.1,0.2])
    #ax11.set_yticklabels([r'',r'$0.1$',r'$0.2$'])
    #ax12.set_yticks([0,20,40,60,80])
    #ax12.set_yticklabels([r'',r'$20$',r'$40$',r'$60$',r''])


    ax11.text(0.05, 0.75, r'\boldmath$A^{p,\pi^+}_{1}$',transform=ax11.transAxes,size=40)
    ax12.text(0.05, 0.75, r'\boldmath$A^{p,\pi^-}_{1}$',transform=ax12.transAxes,size=40)
    ax21.text(0.05, 0.75, r'\boldmath$A^{D,\pi^+}_{1}$',transform=ax21.transAxes,size=40)
    ax22.text(0.05, 0.75, r'\boldmath$A^{D,\pi^-}_{1}$',transform=ax22.transAxes,size=40)

    handles, labels = [],[]
    handles.append(hand[20004])
    handles.append(hand[20017])

    labels.append(r'\textbf{\textrm{HERMES}}')
    labels.append(r'\textbf{\textrm{COMPASS}}')

    ax11.legend(handles,labels,frameon=False,fontsize=22,loc=(0.0,0.35),handletextpad = 0.5, handlelength = 1.0)
    

    handles, labels = [],[]
    handles.append(thy_band[20004])
    handles.append(thy_band[20017])
    labels.append(r'\textbf{\textrm{JAM (HERMES)}}')
    labels.append(r'\textbf{\textrm{JAM (COMPASS)}}')

    ax12.legend(handles,labels,frameon=False,fontsize=22,loc=(0.0,0.35),handletextpad = 0.5, handlelength = 1.0)

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/PSIDIS-pion.png'%(cwd)

    py.savefig(filename)
    print()
    print('Saving PSIDIS pion plot to %s'%filename)

def plot_kaon(wdir,kc):

    print('\ngenerating kaon PSIDIS plots from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax13 = py.subplot(nrows,ncols,3)
    ax21 = py.subplot(nrows,ncols,4)
    ax22 = py.subplot(nrows,ncols,5)

    filters = conf['datasets']['psidis']['filters']
    
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['psidis']={}
    conf['datasets']['psidis']['filters']=filters
    conf['datasets']['psidis']['xlsx']={}
    conf['datasets']['psidis']['xlsx'][20012]='psidis/expdata/20012.xlsx'
    conf['datasets']['psidis']['xlsx'][20013]='psidis/expdata/20013.xlsx'
    conf['datasets']['psidis']['xlsx'][20014]='psidis/expdata/20014.xlsx'
    conf['datasets']['psidis']['xlsx'][20019]='psidis/expdata/20019.xlsx'
    conf['datasets']['psidis']['xlsx'][20020]='psidis/expdata/20020.xlsx'
    conf['datasets']['psidis']['xlsx'][20025]='psidis/expdata/20025.xlsx'
    conf['datasets']['psidis']['xlsx'][20026]='psidis/expdata/20026.xlsx'
    conf['datasets']['psidis']['norm']={}
    conf['psidis tabs']=READER().load_data_sets('psidis')
    tabs = conf['psidis tabs']
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    names = core.get_replicas_names(wdir)  

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['psidis']

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in tabs:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    thy_band = {}
    #--plot data
    for idx in tabs:
        X = tabs[idx]['X']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==20012: ax,color = ax21,'firebrick'
        if idx==20013: ax,color = ax22,'firebrick' 
        if idx==20014: ax,color = ax13,'firebrick'
        if idx==20019: ax,color = ax11,'darkgreen'
        if idx==20020: ax,color = ax12,'darkgreen'
        if idx==20025: ax,color = ax21,'darkgreen' 
        if idx==20026: ax,color = ax22,'darkgreen'

        hand[idx] = ax.errorbar(X,values,yerr=alpha,color=color,fmt='o',ms=2,capsize=3.0)

        for ic in range(nc):
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]
            down = thy - std
            up   = thy + std
            #thy_plot ,= ax.plot(X,thy,color=color)
            thy_band[idx]  = ax.fill_between(X,down,up,color=color,alpha=0.5)

    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_yticks([0.25,0.5,0.75])

    for ax in [ax13,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        #ax.set_yticks([0.25,0.5,0.75])

    for ax in [ax11,ax12,ax13,ax21,ax22]:
        ax.semilogx()
        ax.set_xlim(0.004,0.6)
        ax.set_xticks([0.01,0.1])
        ax.set_xticklabels([r'$0.01$',r'$0.1$'])
        #minorLocator = MultipleLocator(0.1)
        #majorLocator = MultipleLocator(1.0)
        #ax.xaxis.set_minor_locator(minorLocator)
        #ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.axhline(0,0,1,color='black',alpha=0.2)

    for ax in [ax13,ax21,ax22]:
        ax.set_xlabel(r'\boldmath$x$',size=40)
        ax.xaxis.set_label_coords(0.90,0.00)

    for ax in [ax12,ax13,ax22]:
        ax.tick_params(axis='both',which='both',labelleft=False)

    ax11.set_ylim(-0.18,1.0)
    ax12.set_ylim(-0.18,1.0)
    ax13.set_ylim(-0.18,1.0)
    ax21.set_ylim(-0.18,0.5)
    ax22.set_ylim(-0.18,0.5)

    minorLocator = MultipleLocator(0.10)
    majorLocator = MultipleLocator(0.20)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.10)
    majorLocator = MultipleLocator(0.20)
    ax21.yaxis.set_minor_locator(minorLocator)
    ax21.yaxis.set_major_locator(majorLocator)
    ax22.yaxis.set_minor_locator(minorLocator)
    ax22.yaxis.set_major_locator(majorLocator)

    ax11.text(0.05, 0.75, r'\boldmath$A^{p,K^+}_{1}$'        ,transform=ax11.transAxes,size=40)
    ax12.text(0.05, 0.75, r'\boldmath$A^{p,K^-}_{1}$'        ,transform=ax12.transAxes,size=40)
    ax13.text(0.05, 0.75, r'\boldmath$A^{p,K^{\pm}}_{1}$',transform=ax13.transAxes,size=40)
    ax21.text(0.05, 0.75, r'\boldmath$A^{D,K^+}_{1}$'        ,transform=ax21.transAxes,size=40)
    ax22.text(0.05, 0.75, r'\boldmath$A^{D,K^-}_{1}$'        ,transform=ax22.transAxes,size=40)

    #handles = [hand[20012],hand[20019],(thy_band,thy_plot)]
    #label1  = r'\textbf{\textrm{HERMES}}' 
    #label2  = r'\textbf{\textrm{COMPASS}}' 
    #label3  = r'\textbf{\textrm{JAM}}'
    #labels  = [label1,label2,label3] 
    #ax11.legend(handles,labels,frameon=False,fontsize=22,loc=(0.0,0.25),handletextpad = 0.5, handlelength = 1.0)

    handles, labels = [],[]
    handles.append(hand[20012])
    handles.append(hand[20019])

    labels.append(r'\textbf{\textrm{HERMES}}')
    labels.append(r'\textbf{\textrm{COMPASS}}')

    ax11.legend(handles,labels,frameon=False,fontsize=22,loc=(0.0,0.35),handletextpad = 0.5, handlelength = 1.0)
    
    handles, labels = [],[]
    handles.append(thy_band[20012])
    handles.append(thy_band[20019])
    labels.append(r'\textbf{\textrm{JAM (HERMES)}}')
    labels.append(r'\textbf{\textrm{JAM (COMPASS)}}')

    ax12.legend(handles,labels,frameon=False,fontsize=22,loc=(0.0,0.35),handletextpad = 0.5, handlelength = 1.0)


    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/PSIDIS-kaon.png'%(cwd)

    py.savefig(filename)
    print()
    print('Saving PSIDIS kaon plot to %s'%filename)

def plot_hadron(wdir,kc):

    print('\ngenerating hadron PSIDIS plots from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    nrows,ncols=3,2
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax21 = py.subplot(nrows,ncols,3)
    ax22 = py.subplot(nrows,ncols,4)
    ax31 = py.subplot(nrows,ncols,5)
    ax32 = py.subplot(nrows,ncols,6)

    filters = conf['datasets']['psidis']['filters']

    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['psidis']={}
    conf['datasets']['psidis']['filters']=filters
    conf['datasets']['psidis']['xlsx']={}
    conf['datasets']['psidis']['xlsx'][20000]='psidis/expdata/20000.xlsx'
    conf['datasets']['psidis']['xlsx'][20001]='psidis/expdata/20001.xlsx'
    conf['datasets']['psidis']['xlsx'][20002]='psidis/expdata/20002.xlsx'
    conf['datasets']['psidis']['xlsx'][20003]='psidis/expdata/20003.xlsx'
    conf['datasets']['psidis']['xlsx'][20006]='psidis/expdata/20006.xlsx'
    conf['datasets']['psidis']['xlsx'][20007]='psidis/expdata/20007.xlsx'
    conf['datasets']['psidis']['xlsx'][20010]='psidis/expdata/20010.xlsx'
    conf['datasets']['psidis']['xlsx'][20011]='psidis/expdata/20011.xlsx'
    conf['datasets']['psidis']['xlsx'][20015]='psidis/expdata/20015.xlsx'
    conf['datasets']['psidis']['xlsx'][20016]='psidis/expdata/20016.xlsx'
    conf['datasets']['psidis']['xlsx'][20023]='psidis/expdata/20023.xlsx'
    conf['datasets']['psidis']['xlsx'][20024]='psidis/expdata/20024.xlsx'
    conf['datasets']['psidis']['norm']={}
    conf['psidis tabs']=READER().load_data_sets('psidis')
    tabs = conf['psidis tabs']
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['psidis']

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in tabs:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    thy_band = {}
    #--plot data
    for idx in tabs:
        X = tabs[idx]['X']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==20000: ax,color = ax11,'blue'
        if idx==20001: ax,color = ax12,'blue' 
        if idx==20002: ax,color = ax21,'blue'
        if idx==20003: ax,color = ax22,'blue'
        if idx==20006: ax,color = ax11,'firebrick'
        if idx==20007: ax,color = ax12,'firebrick' 
        if idx==20010: ax,color = ax21,'firebrick'
        if idx==20011: ax,color = ax22,'firebrick'
        if idx==20015: ax,color = ax31,'firebrick'
        if idx==20016: ax,color = ax32,'firebrick'
        if idx==20023: ax,color = ax21,'darkgreen'
        if idx==20024: ax,color = ax22,'darkgreen'

        hand[idx] = ax.errorbar(X,values,yerr=alpha,color=color,fmt='o',ms=2,capsize=3.0)

        for ic in range(nc):
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]
            down = thy - std
            up   = thy + std
            #thy_plot ,= ax.plot(X,thy,color=color)
            thy_band[idx]  = ax.fill_between(X,down,up,color=color,alpha=0.5)

    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_yticks([0.25,0.5,0.75])

    for ax in [ax31,ax32]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        #ax.set_yticks([0.25,0.5,0.75])

    for ax in [ax11,ax12,ax21,ax22,ax31,ax32]:
        ax.semilogx()
        ax.set_xlim(0.004,0.6)
        ax.set_xticks([0.01,0.1])
        ax.set_xticklabels([r'$0.01$',r'$0.1$'])
        #minorLocator = MultipleLocator(0.1)
        #majorLocator = MultipleLocator(1.0)
        #ax.xaxis.set_minor_locator(minorLocator)
        #ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.axhline(0,0,1,color='black',alpha=0.2)

    for ax in [ax31,ax32]:
        ax.set_xlabel(r'\boldmath$x$',size=40)
        ax.xaxis.set_label_coords(0.90,0.00)

    for ax in [ax12,ax22,ax32]:
        ax.tick_params(axis='both',which='both',labelleft=False)

    ax11.set_ylim(-0.10,0.99)
    ax12.set_ylim(-0.10,0.99)
    ax21.set_ylim(-0.10,0.50)
    ax22.set_ylim(-0.10,0.50)
    ax31.set_ylim(-0.50,0.50)
    ax32.set_ylim(-0.50,0.50)

    minorLocator = MultipleLocator(0.10)
    majorLocator = MultipleLocator(0.20)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.10)
    majorLocator = MultipleLocator(0.20)
    ax21.yaxis.set_minor_locator(minorLocator)
    ax21.yaxis.set_major_locator(majorLocator)
    ax22.yaxis.set_minor_locator(minorLocator)
    ax22.yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(0.10)
    majorLocator = MultipleLocator(0.20)
    ax31.yaxis.set_minor_locator(minorLocator)
    ax31.yaxis.set_major_locator(majorLocator)
    ax32.yaxis.set_minor_locator(minorLocator)
    ax32.yaxis.set_major_locator(majorLocator)

    ax11.text(0.05, 0.75, r'\boldmath$A^{p,h^+}_{1}$',transform=ax11.transAxes,size=40)
    ax12.text(0.05, 0.75, r'\boldmath$A^{p,h^-}_{1}$',transform=ax12.transAxes,size=40)
    ax21.text(0.05, 0.75, r'\boldmath$A^{D,h^+}_{1}$',transform=ax21.transAxes,size=40)
    ax22.text(0.05, 0.75, r'\boldmath$A^{D,h^-}_{1}$',transform=ax22.transAxes,size=40)
    ax31.text(0.05, 0.75, r'\boldmath$A^{^3{\rm He},h^+}_{1}$',transform=ax31.transAxes,size=40)
    ax32.text(0.05, 0.75, r'\boldmath$A^{^3{\rm He},h^-}_{1}$',transform=ax32.transAxes,size=40)

    #handles = [hand[20006],hand[20023],hand[20000],(thy_band,thy_plot)]
    #label1  = r'\textbf{\textrm{HERMES}}' 
    #label2  = r'\textbf{\textrm{COMPASS}}' 
    #label3  = r'\textbf{\textrm{SMC}}' 
    #label4  = r'\textbf{\textrm{JAM}}'
    #labels  = [label1,label2,label3,label4]
    #ax11.legend(handles,labels,frameon=False,fontsize=22,loc=(0.0,0.25),handletextpad = 0.5, handlelength = 1.0)

    handles, labels = [],[]
    handles.append(hand[20006])
    handles.append(hand[20023])
    handles.append(hand[20000])

    labels.append(r'\textbf{\textrm{HERMES}}')
    labels.append(r'\textbf{\textrm{COMPASS}}')
    labels.append(r'\textbf{\textrm{SMC}}')

    ax11.legend(handles,labels,frameon=False,fontsize=22,loc=(0.0,0.35),handletextpad = 0.5, handlelength = 1.0)
    
    handles, labels = [],[]
    handles.append(thy_band[20006])
    handles.append(thy_band[20023])
    handles.append(thy_band[20000])
    labels.append(r'\textbf{\textrm{JAM (HERMES)}}')
    labels.append(r'\textbf{\textrm{JAM (COMPASS)}}')
    labels.append(r'\textbf{\textrm{JAM (SMC)}}')

    ax12.legend(handles,labels,frameon=False,fontsize=22,loc=(0.0,0.35),handletextpad = 0.5, handlelength = 1.0)


    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0)

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/PSIDIS-hadron.png'%(cwd)

    py.savefig(filename)
    print()
    print('Saving PSIDIS hadron plot to %s'%filename)


if __name__ == "__main__":

    wdir = 'results/star/final'

    plot_pion  (wdir, kc)
    plot_kaon  (wdir, kc)
    plot_hadron(wdir, kc)





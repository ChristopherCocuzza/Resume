#!/usr/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--matplotlib
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

#--from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from obslib
from obslib.wzrv.theory import WZRV
from obslib.wzrv.reader import READER

import kmeanconf as kc

cwd = 'plots/seaquest'

if __name__ == "__main__":

    wdir = 'results/sq/final'

    print('\ngenerating STAR WZRV from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    nrows,ncols=3,1
    fig = py.figure(figsize=(ncols*7,nrows*2.5))
    ax11 = py.subplot(nrows,ncols,(1,2))
    ax21 = py.subplot(nrows,ncols,3)

    conf['path2wzrvtab'] = '%s/grids/grids-wzrv'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['xlsx']={}
    conf['datasets']['wzrv']['xlsx'][2020]='wzrv/expdata/2020.xlsx'
    conf['datasets']['wzrv']['norm']={}
    conf['datasets']['wzrv']['filters']=[]
    conf['wzrv tabs']=READER().load_data_sets('wzrv')
    tabs = conf['wzrv tabs']

    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
  
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['wzrv']
  
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
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==2020: ax,color,marker = ax11,'black','.'
        eta = tabs[idx]['eta']
        xerr = np.zeros((2,len(eta)))
        hand[idx] = ax.errorbar(eta,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=10,capsize=3.0,zorder=5)

    #--plot mean and std of all replicas
    for idx in [2020]:
        for ic in range(nc):
            if idx==2020: ax = ax11
            eta0, eta1 = [], []
            thy0, thy1 = [], []
            std0, std1 = [], []
            up0 , up1  = [], []
            down0,down1= [], []
            boson = conf['wzrv tabs'][idx]['boson']
            eta = tabs[idx]['eta']
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]
            down = thy - std
            up   = thy + std

            thy_plot ,= ax.plot(eta,thy,color='red')
            thy_band  = ax.fill_between(eta,down,up,color='red',alpha=0.5)
    
    for idx in [2020]:
        if idx==2020: ax,color = ax21,'black'
        for ic in range(nc):
            if nc > 1: color = colors[cluster[ic]]
            eta = tabs[idx]['eta']
            thy = data[idx]['thy-%d'%ic]
            ratio = data[idx]['value']/thy
            alpha = data[idx]['alpha']
            ax.errorbar(eta,ratio,yerr=alpha/thy,color=color,fmt='.',ms=10,capsize=3.0,zorder=5)
            ax.fill_between(eta,1-std/thy,1+std/thy,color='red',alpha=0.5)
            ax.axhline(1,0,3,color='black',ls='--')

    for ax in [ax11]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30,labelbottom=False)

    for ax in [ax21]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    handles = [hand[2020],(thy_band,thy_plot)]
    label1 = r'\textbf{\textrm{STAR}}'
    label2 = r'\textbf{\textrm{JAM}}'
    labels = [label1,label2] 
    ax11.legend(handles,labels,frameon=False,fontsize=25,loc='upper right', ncol = 1, handletextpad = 0.3, handlelength = 1.0)

    ax21.set_xlabel(r'\boldmath$\eta_{\ell}$',size=40)
    ax21.xaxis.set_label_coords(0.97,-0.02)

    ax11.set_ylim(0.50,9.50)
    ax21.set_ylim(0.50,1.50)

    ax11.text(0.05,0.75, r'\boldmath$\sigma_{pp}^{W^+}/\sigma_{pp}^{W^-}$'    ,transform = ax11.transAxes, size=50)
    ax11.text(0.35,0.17, r'$\sqrt{S} = 510$'+' '+r'\textrm{GeV}'    ,transform = ax11.transAxes, size=23)
    ax11.text(0.35,0.05, r'\rm{$p_T^{\ell}$ $>$ 25 GeV}'            ,transform = ax11.transAxes, size=23)
    ax21.text(0.02,0.80,r'\textbf{\textrm{data/theory}}'            ,transform = ax21.transAxes, size=35)

    for ax in [ax11,ax21]:
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        majorLocator = MultipleLocator(0.5)
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xlim(-1.4,1.4)

    for ax in [ax11]:
        majorLocator = MultipleLocator(2)
        minorLocator = MultipleLocator(0.5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    for ax in [ax21]:
        majorLocator = MultipleLocator(0.30)
        minorLocator = MultipleLocator(0.10)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_yticks([0.60,1.00,1.40])


    py.tight_layout()
    py.subplots_adjust(hspace = 0.00, wspace=0.00)

    checkdir('%s/gallery'%cwd)
    filename='%s/gallery/STAR'%(cwd)
    #filename+='.png'
    filename+='.pdf'
    ax11.set_rasterized(True)
    ax21.set_rasterized(True)

    py.savefig(filename)
    print()
    print('Saving STAR wzrv plot to %s'%filename)











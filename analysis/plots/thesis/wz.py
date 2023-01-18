#!/usr/bin/env python
import sys, os
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

#--matplotlib
import pylab as py
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

#--from scipy stack 
from scipy.integrate import fixed_quad
from scipy import interpolate
from scipy.interpolate import griddata

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
from obslib.zrap.theory  import ZRAP
from obslib.wasym.theory import WASYM
from obslib.zrap.reader  import READER as ZREADER
from obslib.wasym.reader import READER as WREADER

import kmeanconf as kc

cwd = 'plots/thesis'

def integrate(dsig,Y):
    #--integrate dsig/dy using Gauss. quadrature
    Ymin,Ymax = np.min(Y), np.max(Y)
    gauss_Y = 99 
    x,w = np.polynomial.legendre.leggauss(gauss_Y)
    Y_int = 0.5*(x*(Ymax - Ymin) + Ymin + Ymax)  
    jac = 0.5*(Ymax - Ymin)
    integrand = griddata(Y,dsig,Y_int,method='cubic')
    return np.sum(w*integrand*jac)


if __name__ == "__main__":

    wdir = 'results/misc/marathon_LHAPDF'

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    nrows,ncols=4,2
    fig = py.figure(figsize=(ncols*7,nrows*2))
    gs  = gridspec.GridSpec(4,2)
    ax11 = fig.add_subplot(gs[0:2,0])
    ax21 = fig.add_subplot(gs[2  ,0])
    ax31 = fig.add_subplot(gs[3  ,0])

    #######################
    #--plot Z boson data
    #######################

    conf['path2zraptab'] = '%s/grids/grids-zrap'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['zrap']={}
    conf['datasets']['zrap']['xlsx']={}
    conf['datasets']['zrap']['xlsx'][1000]='zrap/expdata/1000.xlsx'
    conf['datasets']['zrap']['xlsx'][1001]='zrap/expdata/1001.xlsx'
    conf['datasets']['zrap']['norm']={}
    conf['datasets']['zrap']['filters']=[]
    conf['zrap tabs']=ZREADER().load_data_sets('zrap')
    tabs = conf['zrap tabs']
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['zrap']

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
    #--plot data
    for idx in tabs:
        Y = tabs[idx]['Y']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if 'normalization' in tabs[idx]: norm   = tabs[idx]['normalization'][0]
        else: norm = 2*integrate(values,Y)
        if idx==1000: color = 'darkgreen'
        if idx==1001: color = 'firebrick' 
        hand[idx] = ax11.errorbar(Y,values/norm,yerr=alpha/norm,color=color,fmt='o',ms=2,capsize=3.0)

    #--plot mean and std of all replicas
    for idx in data:
        for ic in range(nc):
            Y = conf['zrap tabs'][idx]['Y']
            if 'normalization' in tabs[idx]: norm   = tabs[idx]['normalization'][0]
            else: norm = 2*integrate(values,Y)
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]/norm
            std = data[idx]['dthy-%d'%ic]/norm
            down = thy - std
            up   = thy + std
            thy_plot ,= ax11.plot(Y,thy,color='black')
            thy_band  = ax11.fill_between(Y,down,up,color='gold',alpha=1.0)

    #######################
    #--plot ratio
    #######################


    for idx in data:
        if idx==1000: ax,color,label = ax21,'darkgreen',r'\textbf{\textrm{CDF(Z)}}'
        if idx==1001: ax,color,label = ax31,'firebrick',r'\textbf{\textrm{D0(Z)}}'
        for ic in range(nc):
            Y = conf['zrap tabs'][idx]['Y']
            if nc > 1: color = colors[cluster[ic]]
            if 'normalization' in tabs[idx]: norm   = tabs[idx]['normalization'][0]
            else: norm = 2*integrate(values,Y)
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]/norm
            ratio = data[idx]['value']/thy
            alpha = data[idx]['alpha']
            ax.errorbar(Y,ratio,yerr=alpha/thy,color=color,fmt='.',ms=10,capsize=3.0)
            ax.fill_between(Y,1-std/thy,1+std/thy,color='gold',alpha=1.0)
            ax.text(0.02,0.78,label,fontsize=25,transform=ax.transAxes)
            ax.axhline(1,0,3,color='black',ls='--')
            

    for ax in [ax11,ax21]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_xlim(0,3)

    for ax in [ax31]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0,3)

    for ax in [ax11]:
        ax.set_ylim(0,0.32)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_yticks([0,0.1,0.2,0.3])
        ax.set_yticklabels([r'',r'$0.1$',r'$0.2$',r'$0.3$'])

    for ax in [ax21,ax31]:
        ax.set_ylim(0.75,1.25)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    ax31.set_xlabel(r'\boldmath$y_Z$',size=30)
    ax31.xaxis.set_label_coords(0.85,-0.02)

    ax11.text(0.1, 0.02, r'\boldmath$\frac{1}{\sigma} \frac{d\sigma(Z/\gamma^*)}{dy}$',size=50)
    ax21.text(0.1, 0.8,  r'\textbf{\textrm{data/theory}}'    ,size=30)

    ax11.text(0.1,0.15, r'$\sqrt{s} = 1.96$'+' '+r'\textrm{TeV}'   ,fontsize=25)

    handles = [hand[1000],hand[1001],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{CDF(Z)}}' 
    label2  = r'\textbf{\textrm{D0(Z)}}' 
    label3  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2,label3] 
    ax11.legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.5, handlelength = 1.5)

    ax11.set_rasterized(True)
    ax21.set_rasterized(True)
    ax31.set_rasterized(True)

    #######################
    #--plot W boson data
    #######################

    ax11 = fig.add_subplot(gs[0:2,1])
    ax21 = fig.add_subplot(gs[2  ,1])
    ax31 = fig.add_subplot(gs[3  ,1])

    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    conf['path2wasymtab'] = '%s/grids/grids-wasym'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wasym']={}
    conf['datasets']['wasym']['xlsx']={}
    conf['datasets']['wasym']['xlsx'][1000]='wasym/expdata/1000.xlsx'
    conf['datasets']['wasym']['xlsx'][1001]='wasym/expdata/1001.xlsx'
    conf['datasets']['wasym']['norm']={}
    conf['datasets']['wasym']['filters']=[]
    conf['wasym tabs']=WREADER().load_data_sets('wasym')
    tabs = conf['wasym tabs']
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    data = predictions['reactions']['wasym']

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
    #--plot data
    for idx in tabs:
        Y = tabs[idx]['Y']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==1000: color = 'darkgreen'
        if idx==1001: color = 'firebrick'
        hand[idx] = ax11.errorbar(Y,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)

    #--plot mean and std of all replicas
    for idx in data:
        for ic in range(nc):
            Y = tabs[idx]['Y']
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]
            down = thy - std
            up   = thy + std
            thy_plot ,= ax11.plot(Y,thy,color='black')
            thy_band  = ax11.fill_between(Y,down,up,color='gold',alpha=1.0)

    #######################
    #--plot ratio
    #######################


    for idx in data:
        if idx==1000: ax,color,label = ax21,'darkgreen',r'\textbf{\textrm{CDF(W)}}'
        if idx==1001: ax,color,label = ax31,'firebrick' ,r'\textbf{\textrm{D0(W)}}'
        for ic in range(nc):
            Y = conf['wasym tabs'][idx]['Y']
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            std = data[idx]['dthy-%d'%ic]
            ratio = data[idx]['value']/thy
            alpha = data[idx]['alpha']
            ax.errorbar(Y,ratio,yerr=alpha/thy,color=color,fmt='.',ms=10,capsize=3.0)
            ax.fill_between(Y,1-std/thy,1+std/thy,color='gold',alpha=1.0)
            if idx==1000: ax.text(0.74,0.78,label,fontsize=25,transform=ax.transAxes)
            if idx==1001: ax.text(0.79,0.78,label,fontsize=25,transform=ax.transAxes)
            ax.axhline(1,0,3,color='black',ls='--')
            

    for ax in [ax11,ax21]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_xlim(0,3)

    for ax in [ax31]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0,3)

    for ax in [ax11]:
        ax.set_ylim(0,0.85)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_yticks([0.2,0.4,0.6,0.8])
        ax.set_yticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])

    for ax in [ax21,ax31]:
        ax.set_ylim(0.75,1.25)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    ax31.set_xlabel(r'\boldmath$y_W$',size=30)
    ax31.xaxis.set_label_coords(0.85,-0.02)

    ax11.text(2.2, 0.04, r'\boldmath$A_W$',size=50)
    ax21.text(0.1, 0.8, r'\textbf{\textrm{data/theory}}'    ,size=30)

    ax11.text(0.1,0.30, r'$\sqrt{s} = 1.96$'+' '+r'\textrm{TeV}'   ,fontsize=25)

    handles,labels = [], []
    handles.append(hand[1000])
    handles.append(hand[1001])
    #handles.append((thy_band,thy_plot))
    labels.append(r'\textbf{\textrm{CDF(W)}}') 
    labels.append(r'\textbf{\textrm{D0(W)}}')
    #labels.append(r'\textbf{\textrm{JAM}}')
    ax11.legend(handles,labels,frameon=False,fontsize=22,loc='upper left',handletextpad = 0.5, handlelength = 1.5)


    py.tight_layout()
    py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%cwd)
    filename='%s/gallery/WZboson'%cwd
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax21.set_rasterized(True)
    ax31.set_rasterized(True)

    py.savefig(filename)
    print('Saving WZ boson plot to %s'%filename)











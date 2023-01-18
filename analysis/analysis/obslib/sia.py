import sys, os
import numpy as np
import copy
import pandas as pd
import scipy as sp

## matplotlib
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
from matplotlib.ticker import MultipleLocator, FormatStrFormatter ## for minor ticks in x label
#matplotlib.rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
matplotlib.rc('text', usetex = True)
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.pyplot as py
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec

from scipy.stats import norm


#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from tools
from tools.tools import load,save,checkdir
from tools.config import conf, load_config

def plot_pion(wdir,kc):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    sia_pi_idx=[
         [1030]  #  ${\rm ARGUS}~(\pi^{\pm})$
        ,[1029]  #  ${\rm BELLE}~(\pi^{\pm})$
        ,[1028]  #  ${\rm BABAR}~(\pi^{\pm})$
        ,[1001   # ${\rm TASSO}~(\pi^{\pm})$
         ,1002   # ${\rm TASSO}~(\pi^{\pm})$
         ,1003   # ${\rm TASSO}~(\pi^{\pm})$
         ,1005    # ${\rm TASSO}~(\pi^{\pm})$
         ,1006]   # ${\rm TASSO}~(\pi^{\pm})$
        ,[1007,1008]  #  ${\rm TPC}~(\pi^{\pm})$  #  ${\rm TPC}~(\pi^{\pm})$
        ,[1010]  #  ${\rm TPC(c)}~(\pi^{\pm})$
        ,[1011]  #  ${\rm TPC(b)}~(\pi^{\pm})$
        #,[1013]  #  ${\rm TOPAZ}~(\pi^{\pm})$
        ,[1019]  #  ${\rm OPAL}~(\pi^{\pm})$
        ,[1023]  #  ${\rm OPAL(c)}~(\pi^{\pm})$
        ,[1024]  #  ${\rm OPAL(b)}~(\pi^{\pm})$
        ,[1018]  #  ${\rm ALEPH}~(\pi^{\pm})$
        ,[1025]  #  ${\rm DELPHI}~(\pi^{\pm})$
        ,[1027]  #  ${\rm DELPHI(b)}~(\pi^{\pm})$
        ,[1014]  #  ${\rm SLD}~(\pi^{\pm})$
        ,[1016]  #  ${\rm SLD(c)}~(\pi^{\pm})$
        ,[1017]  #  ${\rm SLD(b)}~(\pi^{\pm})$
        ]

    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster = labels['cluster']
    if 'sia' not in predictions['reactions']: return
    for i in range(len(sia_pi_idx)):
        for k in range(len(sia_pi_idx[i])):
            if sia_pi_idx[i][k] not in predictions['reactions']['sia']: return
    
    nrows,ncols=4,4
    fig = py.figure(figsize=(ncols*7,nrows*4))

    cnt=0
    for idx in sia_pi_idx:
        cnt+=1
        if idx[0]==None: continue

        ax=py.subplot(nrows,ncols,cnt)
        for _ in idx:

            tab=predictions['reactions']['sia'][_]
            thy=np.mean(np.array(tab['prediction-rep']),axis=0)
            dthy=np.std(np.array(tab['prediction-rep']),axis=0)

            exp=tab['value']
            alpha=tab['alpha']
            msg=r'\boldmath{$\rm %s$}'%(tab['col'][0].replace('_',''))
            ax.text(0.02,0.88,msg,transform=ax.transAxes,size=30)        
            ax.fill_between(tab['z'],(thy-dthy)/thy,(thy+dthy)/thy,color='r')
            ax.errorbar(tab['z'],exp/thy,yerr=alpha/thy,fmt='k.')
        ax.tick_params(axis='both', which='major', labelsize=30,direction='in')

        ax.set_ylim(0.5,1.5)
        ax.set_xlim(0,1)
        if any([cnt==_ for _ in [1,2,3,4,5,6,7,8,9,10,11,12]]):
            ax.set_xticklabels([])
        if any([cnt==_ for _ in [2,3,4,6,7,8,10,11,12,14,15,16]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in [1,5,9,13]]):
            ax.set_yticks([0.6,0.8,1.0,1.2,1.4])
        if any([cnt==_ for _ in [13,14,15,16]]):
            ax.set_xticks([0.2,0.4,0.6,0.8])
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xlabel(r'\boldmath$z_h$',size=50)
        if cnt==1:
            ax.set_ylabel(r'\boldmath${\rm data/theory}$',size=50)
            ax.yaxis.set_label_coords(-0.15, -1.2)
            ax.text(0.78,0.1,r'\boldmath${\pi^{\pm}}$',color='r',transform=ax.transAxes,size=60)
    #py.tight_layout()
    py.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.05, hspace=0.1)
    filename = '%s/gallery/sia-pion.png'%wdir
    py.savefig(filename)
    print('Saving SIA pion figure to %s'%filename)

def plot_kaon(wdir,kc):
    
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    sia_k_idx=[
        [2030]   #${\rm ARGUS}~(K^{\pm})$
        ,[2029]  #${\rm BELLE}~(K^{\pm})$
        ,[2028]  #${\rm BABAR}~(K^{\pm})$
        ,[2002   #${\rm TASSO}~(K^{\pm})$ 
          ,2003  #${\rm TASSO}~(K^{\pm})$ 
          ,2005] #${\rm TASSO}~(K^{\pm})$
        ,[2007   #${\rm TPC}~(K^{\pm})$
          ,2008] #${\rm TPC}~(K^{\pm})$
        #,[2013]  #${\rm TOPAZ}~(K^{\pm})$
        ,[2019] #${\rm OPAL}~(K^{\pm})$
        ,[2023] #${\rm OPAL(c)}~(K^{\pm})$
        ,[2024] #${\rm OPAL(b)}~(K^{\pm})$
        ,[2018] #${\rm ALEPH}~(K^{\pm})$
        ,[2031 #${\rm DELPHI}~(K^{\pm})$
          ,2025] #${\rm DELPHI}~(K^{\pm})$
        ,[2027] #${\rm DELPHI(b)}~(K^{\pm})$
        ,[2014] #${\rm SLD}~(K^{\pm})$
        ,[2016] #${\rm SLD(c)}~(K^{\pm})$
        ,[2017] #${\rm SLD(b)}~(K^{\pm})$
      ]
    
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster = labels['cluster']
    if 'sia' not in predictions['reactions']: return
    for i in range(len(sia_k_idx)):
        for k in range(len(sia_k_idx[i])):
            if sia_k_idx[i][k] not in predictions['reactions']['sia']: return
    
    nrows,ncols=4,4
    fig = py.figure(figsize=(ncols*7,nrows*4))
    AX={}
    cnt=0
    for idx in sia_k_idx:

        cnt+=1
        if idx[0]==None: continue

        ax=py.subplot(nrows,ncols,cnt)
        AX[cnt]=ax
        for _ in idx:

            tab=predictions['reactions']['sia'][_]
            thy=np.mean(np.array(tab['prediction-rep']),axis=0)
            dthy=np.std(np.array(tab['prediction-rep']),axis=0)

            exp=tab['value']
            alpha=tab['alpha']
            msg=r'\boldmath{$\rm %s$}'%(tab['col'][0].replace('_',''))
            ax.text(0.02,0.88,msg,transform=ax.transAxes,size=30)
            ax.fill_between(tab['z'],(thy-dthy)/thy,(thy+dthy)/thy,color='b')
            ax.errorbar(tab['z'],exp/thy,yerr=alpha/thy,fmt='k.')
        ax.tick_params(axis='both', which='major', labelsize=30,direction='in')
        ax.set_ylim(0.5,1.5)
        ax.set_xlim(0,1)
        if any([cnt==_ for _ in [1,2,3,4,5,6,7,8,9,10]]):
            ax.set_xticklabels([])
        if any([cnt==_ for _ in [2,3,4,6,7,8,10,11,12,14]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in [1,5,9,13]]):
            ax.set_yticks([0.6,0.8,1.0,1.2,1.4])
        if any([cnt==_ for _ in [11,12,13,14]]):
            ax.set_xticks([0.2,0.4,0.6,0.8])
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xlabel(r'\boldmath$z_h$',size=50)
        if cnt==1:
            ax.set_ylabel(r'\boldmath${\rm data/theory}$',size=50)
            ax.yaxis.set_label_coords(-0.15, -1.2)
            ax.text(0.75,0.1,r'\boldmath${K^{\pm}}$',color='b',transform=ax.transAxes,size=60)

    #py.tight_layout()
    py.subplots_adjust(left=0.07, bottom=0.05, right=0.99, top=0.99, wspace=0.05, hspace=0.1)
    filename = '%s/gallery/sia-kaon.png'%wdir
    py.savefig(filename)
    print('Saving SIA kaon figure to %s'%filename)

def plot_hadron(wdir,kc):
    
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    
    sia_h_idx=[[4011  # ${\rm TASSO}~(h^{\pm})$
        ,4012]  # ${\rm TASSO}~(h^{\pm})$
        ,[4004] # ${\rm TPC}~(h^{\pm})$
        ,[4007] # ${\rm OPAL}~(h^{\pm})$
        ,[4005] # ${\rm OPAL(b)}~(h^{\pm})$
        ,[4006] # ${\rm OPAL(c)}~(h^{\pm})$
        ,[4000] # ${\rm ALEPH}~(h^{\pm})$
        ,[4001] # ${\rm DELPHI}~(h^{\pm})$
        ,[4013] # ${\rm DELPHI(b)}~(h^{\pm})$
        ,[4002] # ${\rm SLD}~(h^{\pm})$
        ,[4014] # ${\rm SLD(c)}~(h^{\pm})$
        ,[4015] # ${\rm SLD(b)}~(h^{\pm})$
        ]

    
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster = labels['cluster']
    if 'sia' not in predictions['reactions']: return
    for i in range(len(sia_h_idx)):
        for k in range(len(sia_h_idx[i])):
            if sia_h_idx[i][k] not in predictions['reactions']['sia']: return

    nrows,ncols=3,4
    fig = py.figure(figsize=(ncols*7,nrows*4))

    cnt=0
    for idx in sia_h_idx:
        cnt+=1
        if idx[0]==None: continue

        ax=py.subplot(nrows,ncols,cnt)
        for _ in idx:

            tab=predictions['reactions']['sia'][_]
            thy=np.mean(np.array(tab['prediction-rep']),axis=0)
            dthy=np.std(np.array(tab['prediction-rep']),axis=0)

            exp=tab['value']
            alpha=tab['alpha']
            msg=r'\boldmath{$\rm %s$}'%(tab['col'][0].replace('_',''))
            ax.text(0.02,0.88,msg,transform=ax.transAxes,size=30)      
            ax.fill_between(tab['z'],(thy-dthy)/thy,(thy+dthy)/thy,color='g')
            ax.errorbar(tab['z'],exp/thy,yerr=alpha/thy,fmt='k.')
        ax.tick_params(axis='both', which='major', labelsize=30,direction='in')

        ax.set_ylim(0.5,1.5)
        ax.set_xlim(0,1)
        if any([cnt==_ for _ in [1,2,3,4,5,6,7,8]]):
            ax.set_xticklabels([])
        if any([cnt==_ for _ in [2,3,4,6,7,8,10,11,12,14,15,16]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in [1,5,9,13]]):
            ax.set_yticks([0.6,0.8,1.0,1.2,1.4])
        if any([cnt==_ for _ in [9,10,11]]):
            ax.set_xticks([0.2,0.4,0.6,0.8])
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xlabel(r'\boldmath$z_h$',size=50)
        if cnt==1:
            ax.set_ylabel(r'\boldmath${\rm data/theory}$',size=50)
            ax.yaxis.set_label_coords(-0.15, -0.6)
            ax.text(0.1,0.1,r'\boldmath${h^{\pm}}$',color='g',transform=ax.transAxes,size=60)

    #py.tight_layout()
    py.subplots_adjust(left=0.07, bottom=0.06, right=0.99, top=0.99, wspace=0.05, hspace=0.1)
    filename = '%s/gallery/sia-hadron.png'%wdir
    py.savefig(filename)
    print('Saving SIA hadron figure to %s'%filename)













#!/usr/bin/env python
import sys, os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerBase
from matplotlib import cm
import pylab as py

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack fitlib
from fitlib.resman import RESMAN

## from fitpack analysis
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

cwd = 'plots/star'

if __name__ == "__main__":

  #--with PSIDIS
  wdir1 = 'results/star/final'
  wdir2 = 'results/pol/step31'
  wdir3 = 'results/star/pos2'

  WDIR = [wdir1,wdir2,wdir3]

  nrows,ncols=2,2
  N = nrows*ncols
  fig = py.figure(figsize=(ncols*9,nrows*5))
  axs,axLs = {},{}
  for i in range(N):
      axs[i+1] = py.subplot(nrows,ncols,i+1)
      divider = make_axes_locatable(axs[i+1])
      axLs[i+1] = divider.append_axes("right",size=3.50,pad=0,sharey=axs[i+1])
      axLs[i+1].spines['left'].set_visible(False)
      axLs[i+1].yaxis.set_ticks_position('right')
      py.setp(axLs[i+1].get_xticklabels(),visible=True)

      axs[i+1].spines['right'].set_visible(False)

  filename = '%s/gallery/helicity'%cwd

  Q2 = 10
  hand = {}
  j = 0
  for wdir in WDIR:

      hand[j] = {}

      load_config('%s/input.py'%wdir)
      istep=core.get_istep()

      data =load('%s/data/ppdf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))
      udata=load('%s/data/pdf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))

      replicas=core.get_replicas(wdir)
      cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
      best_cluster=cluster_order[0]

      X=data['X']
      idx1 = np.nonzero(X <= 0.1)
      idx2 = np.nonzero(X >= 0.1)

      flavs = ['up','dp','ub','db']
      for flav in flavs:

          ppdf  = np.array(data ['XF'][flav])
          if flav=='up':   pdf = np.array(udata['XF']['u']) + np.array(udata['XF']['ub'])
          elif flav=='dp': pdf = np.array(udata['XF']['d']) + np.array(udata['XF']['db'])
          else:            pdf = np.array(udata['XF'][flav])
          plus  = X*(pdf + ppdf)/2.0
          minus = X*(pdf - ppdf)/2.0

          meanp = np.mean(plus ,axis=0)
          stdp  = np.std (plus ,axis=0)
          meanm = np.mean(minus,axis=0)
          stdm  = np.std (minus,axis=0)

          if flav=='up' : ax,axL = axs[1],axLs[1]
          if flav=='dp' : ax,axL = axs[2],axLs[2]
          if flav=='ub' : ax,axL = axs[3],axLs[3]
          if flav=='db' : ax,axL = axs[4],axLs[4]

          if flav=='up' :  color1,color2,alpha  = 'red','darkviolet' ,0.8
          if flav=='dp' :  color1,color2,alpha  = 'dodgerblue','green',0.8
          if flav=='ub' :  color1,color2,alpha  = 'red','darkviolet' ,0.8
          if flav=='db' :  color1,color2,alpha  = 'dodgerblue','green',0.8

          if j==0:
              hand[j][flav+'+'] = ax.fill_between(X,(meanp-stdp),(meanp+stdp),color=color1,alpha=alpha,zorder=2)
              axL.                   fill_between(X,(meanp-stdp),(meanp+stdp),color=color1,alpha=alpha,zorder=2)
              hand[j][flav+'-'] = ax.fill_between(X,(meanm-stdm),(meanm+stdm),color=color2,alpha=alpha,zorder=2)
              axL.                   fill_between(X,(meanm-stdm),(meanm+stdm),color=color2,alpha=alpha,zorder=2)
          if j==1:
              hand[j][flav+'+'] = ax.fill_between(X,(meanp-stdp),(meanp+stdp),color=color1,alpha=0.1,zorder=1)
              axL.                   fill_between(X,(meanp-stdp),(meanp+stdp),color=color1,alpha=0.1,zorder=1)
              hand[j][flav+'-'] = ax.fill_between(X,(meanm-stdm),(meanm+stdm),color=color2,alpha=0.1,zorder=1)
              axL.                   fill_between(X,(meanm-stdm),(meanm+stdm),color=color2,alpha=0.1,zorder=1)
          if j==2:
              hand[j][flav+'+'] = ax.fill_between(X,(meanp-stdp),(meanp+stdp),facecolor='none',alpha=0.4,zorder=3,hatch='//',edgecolor='black')
              axL.                   fill_between(X,(meanp-stdp),(meanp+stdp),facecolor='none',alpha=0.4,zorder=3,hatch='//',edgecolor='black')
              hand[j][flav+'-'] = ax.fill_between(X,(meanm-stdm),(meanm+stdm),facecolor='none',alpha=0.4,zorder=3,hatch='//',edgecolor='black')
              axL.                   fill_between(X,(meanm-stdm),(meanm+stdm),facecolor='none',alpha=0.4,zorder=3,hatch='//',edgecolor='black')


      j+=1

  for i in range(N):
      axs[i+1].set_xlim(8e-3,0.1)
      axs[i+1].semilogx()

      minorLocator = MultipleLocator(0.1)
      axs[i+1].yaxis.set_minor_locator(minorLocator)
      axLs[i+1].axhline(0,0,1,color='black',linestyle='-' ,alpha=0.2, lw=1)
      axs[i+1] .axhline(0,0,1,color='black',linestyle='-' ,alpha=0.2, lw=1)

      axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
      axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
      axs[i+1].set_xticks([0.01,0.1])
      axs[i+1].set_xticklabels([r'$0.01$',r''])

      axLs[i+1].tick_params(axis='both', which='major', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=10)
      axLs[i+1].tick_params(axis='both', which='minor', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=5)

  axs[1].set_ylim(-0.00,0.16)  
  axs[2].set_ylim(-0.00,0.07)  
  axs[3].set_ylim(-0.00,0.0099)  
  axs[4].set_ylim(-0.00,0.0099)  



  minorLocator = MultipleLocator(0.01)
  axs[1].yaxis.set_minor_locator(minorLocator)

  minorLocator = MultipleLocator(0.01)
  axs[2].yaxis.set_minor_locator(minorLocator)

  minorLocator = MultipleLocator(0.001)
  axs[3].yaxis.set_minor_locator(minorLocator)

  minorLocator = MultipleLocator(0.001)
  axs[4].yaxis.set_minor_locator(minorLocator)

  axs[1].set_yticks([0,0.05,0.1,0.15])
  axs[2].set_yticks([0,0.02,0.04,0.06])
  axs[3].set_yticks([0,0.002,0.004,0.006,0.008])
  axs[4].set_yticks([0,0.002,0.004,0.006,0.008])

  axs[1].set_yticklabels([r'$0$',r'$0.05$' ,r'$0.10$' ,r'$0.15$'])
  axs[2].set_yticklabels([r'$0$',r'$0.02$' ,r'$0.04$' ,r'$0.06$'])
  axs[3].set_yticklabels([r'$0$',r'$0.002$',r'$0.004$',r'$0.006$',r'$0.008$'])
  axs[4].set_yticklabels([r'$0$',r'$0.002$',r'$0.004$',r'$0.006$',r'$0.008$'])

  axLs[1].set_xlim(0.1,0.8)
  axLs[1].set_xticks([0.1,0.3,0.5,0.7])

  axLs[2].set_xlim(0.1,0.8)
  axLs[2].set_xticks([0.1,0.3,0.5,0.7])

  axLs[3].set_xlim(0.1,0.4)
  axLs[3].set_xticks([0.1,0.2,0.3])

  axLs[4].set_xlim(0.1,0.4)
  axLs[4].set_xticks([0.1,0.2,0.3])

  minorLocator = MultipleLocator(0.1)
  axLs[1].xaxis.set_minor_locator(minorLocator)
  axLs[2].xaxis.set_minor_locator(minorLocator)


  axLs[3].set_xlabel(r'\boldmath$x$',size=50)
  axLs[3].xaxis.set_label_coords(0.92,0.00)
  axLs[4].set_xlabel(r'\boldmath$x$',size=50)
  axLs[4].xaxis.set_label_coords(0.92,0.00)

  axs[2].text(0.08,0.40,r'$Q^2 = %s$~'%Q2 + r'\textrm{GeV}'+r'$^2$', transform=axs[2].transAxes,size=30)
  axs[1].text(0.08,0.250,r'\textrm{\textbf{light: no RHIC \boldmath$W/Z$}}', transform=axs[1].transAxes,size=20)

  blank ,= axs[1].plot([0],[0],alpha=0)

  fs = 35
  handles, labels = [],[]
  handles.append(hand[0]['up+'])
  handles.append(hand[0]['up-'])
  handles.append(hand[2]['up+'])
  labels.append(r'\boldmath$x (u+\bar{u})^{\uparrow}$')
  labels.append(r'\boldmath$x (u+\bar{u})^{\downarrow}$')
  labels.append(r'\textrm{\textbf{+pos}}')
  axs[1].legend(handles,labels,loc='upper left',fontsize=fs,frameon=0,handletextpad=0.3,handlelength=1.0,labelspacing=0.6)

  handles, labels = [],[]
  handles.append(hand[0]['dp+'])
  handles.append(hand[0]['dp-'])
  labels.append(r'\boldmath$x (d+\bar{d})^{\uparrow}$')
  labels.append(r'\boldmath$x (d+\bar{d})^{\downarrow}$')
  axs[2].legend(handles,labels,loc='upper left',fontsize=fs,frameon=0,handletextpad=0.3,handlelength=1.0,labelspacing=0.6)

  handles, labels = [],[]
  handles.append(hand[0]['ub+'])
  handles.append(hand[0]['ub-'])
  labels.append(r'\boldmath$x \bar{u}^{\uparrow}$')
  labels.append(r'\boldmath$x \bar{u}^{\downarrow}$')
  axs[3].legend(handles,labels,loc='upper left',fontsize=fs,frameon=0,handletextpad=0.3,handlelength=1.0,labelspacing=0.6)

  handles, labels = [],[]
  handles.append(hand[0]['db+'])
  handles.append(hand[0]['db-'])
  labels.append(r'\boldmath$x \bar{d}^{\uparrow}$')
  labels.append(r'\boldmath$x \bar{d}^{\downarrow}$')
  axs[4].legend(handles,labels,loc='upper left',fontsize=fs,frameon=0,handletextpad=0.3,handlelength=1.0,labelspacing=0.6)



  py.tight_layout()
  #py.subplots_adjust(wspace=0.02,hspace=0)
  #py.subplots_adjust(left=0.10)

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)









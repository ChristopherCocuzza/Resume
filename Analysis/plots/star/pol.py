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

  wdir1 = 'results/star/final'
  wdir2 = 'results/star/pos2'

  WDIR = [wdir1,wdir2]

  nrows,ncols=2,1
  N = nrows*ncols
  fig = py.figure(figsize=(ncols*9,nrows*5))
  ax11 = py.subplot(nrows,ncols,1)
  ax12 = py.subplot(nrows,ncols,2)

  filename = '%s/gallery/polarization'%cwd

  Q2 = 10
  hand = {}
  j = 0
  for wdir in WDIR:

      hand[j] = {}

      load_config('%s/input.py'%wdir)
      istep=core.get_istep()

      data =load('%s/data/ppdf-Q2=%3.5f.dat'%(wdir,Q2))
      udata=load('%s/data/pdf-Q2=%3.5f.dat' %(wdir,Q2))

      replicas=core.get_replicas(wdir)
      cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
      best_cluster=cluster_order[0]

      X=data['X']

      flavs = ['u','d','ub','db']
      for flav in flavs:

          ppdf = np.array(data ['XF'][flav])
          pdf  = np.array(udata['XF'][flav])
          rat = ppdf/pdf

          mean = np.mean(rat,axis=0)
          std  = np.std (rat,axis=0)

          if flav=='u' :  ax = ax11
          if flav=='d' :  ax = ax11
          if flav=='ub' : ax = ax12
          if flav=='db' : ax = ax12

          if flav=='u' :  color,alpha,zorder = 'coral'      ,1.0, 2
          if flav=='d' :  color,alpha,zorder = 'skyblue'    ,1.0, 1
          if flav=='ub' : color,alpha,zorder = 'red'        ,1.0, 2
          if flav=='db' : color,alpha,zorder = 'dodgerblue' ,1.0, 1

          if j==0:
              hand[j][flav] = ax.fill_between(X,(mean-std),(mean+std),color=color,alpha=alpha,zorder=zorder)
              if flav=='db' or flav=='d':
                  ax.plot(X,mean+std,color=color,alpha=0.5,zorder=3)
                  ax.plot(X,mean-std,color=color,alpha=0.5,zorder=3)
          if j==1:
              hand[j][flav] = ax.fill_between(X,(mean-std),(mean+std),facecolor='none',alpha=0.4,zorder=zorder,hatch='//',edgecolor='black')
              if flav=='db' or flav=='d':
                  ax.plot(X,mean+std,color='black',alpha=0.2,zorder=3)
                  ax.plot(X,mean-std,color='black',alpha=0.2,zorder=3)


      j+=1

  for ax in [ax11,ax12]:
      ax.set_ylim(-1.20,1.20)  
      ax.set_yticks([-1,-0.5,0,0.5,1.0])


      minorLocator = MultipleLocator(0.025)
      ax.xaxis.set_minor_locator(minorLocator)

      minorLocator = MultipleLocator(0.1)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.axhline(0,0,1,color='black',linestyle='-' ,alpha=0.2, lw=1)
      ax.axhline(0,0,1,color='black',linestyle='-' ,alpha=0.2, lw=1)

      ax.tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
      ax.tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
      ax.set_yticks([-1.0,-0.5,0,0.5,1.0])
      ax.set_yticklabels([r'$-1.0$',r'$-0.5$',r'$0$',r'$0.5$',r'$1.0$'])

      ax.tick_params(axis='both', which='major', top=True, right=True, left=True, labelright=False, direction='in',labelsize=30,length=10)
      ax.tick_params(axis='both', which='minor', top=True, right=True, left=True, labelright=False, direction='in',labelsize=30,length=5)
      ax.set_xlabel(r'\boldmath$x$',size=40)
      ax.xaxis.set_label_coords(0.95,0.00)

  #ax12.tick_params(labelleft=False)

  ax11.set_xlim(8e-3,0.8)
  ax11.set_xticks([0.1,0.3,0.5,0.7])

  ax12.set_xlim(8e-3,0.4)
  ax12.set_xticks([0.1,0.2,0.3])

  #ax11.text(0.35,0.08,r'$Q^2 = %s$~'%Q2 + r'\textrm{GeV}'+r'$^2$', transform=ax11.transAxes,size=30)

  blank ,= ax11.plot([0],[0],alpha=0)

  fs = 25
  handles, labels = [],[]
  handles.append(hand[0]['u'])
  handles.append(hand[0]['d'])
  labels.append(r'\boldmath$\Delta u/ u$')
  labels.append(r'\boldmath$\Delta d/ d$')
  legend1 = ax11.legend(handles,labels,loc='upper left',fontsize=fs,frameon=0,handletextpad=0.3,handlelength=1.0,labelspacing=0.6)
  ax11.add_artist(legend1)

  handles, labels = [],[]
  handles.append(hand[1]['d'])
  labels.append(r'\textbf{\textrm{+pos}}')
  legend2 = ax11.legend(handles,labels,loc='lower left',fontsize=fs,frameon=0,handletextpad=0.3,handlelength=1.0,labelspacing=0.6)

  handles, labels = [],[]
  handles.append(hand[0]['ub'])
  handles.append(hand[0]['db'])
  labels.append(r'\boldmath$\Delta \bar{u}/ \bar{u}$')
  labels.append(r'\boldmath$\Delta \bar{d}/ \bar{d}$')
  legend1 = ax12.legend(handles,labels,loc='upper left',fontsize=fs,frameon=0,handletextpad=0.3,handlelength=1.0,labelspacing=0.6)
  ax12.add_artist(legend1)
  
  handles, labels = [],[]
  handles.append(hand[1]['db'])
  labels.append(r'\textbf{\textrm{+pos}}')
  legend2 = ax12.legend(handles,labels,loc='lower left',fontsize=fs,frameon=0,handletextpad=0.3,handlelength=1.0,labelspacing=0.6)

  py.tight_layout()
  #py.subplots_adjust(wspace=0,hspace=0.2)

  #filename+='.png'
  filename+='.pdf'
  ax11.set_rasterized(True)
  ax12.set_rasterized(True)

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)









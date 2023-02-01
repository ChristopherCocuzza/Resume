#!/usr/bin/env python
import sys, os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerBase
import pylab as py

from scipy.interpolate import interp1d

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack fitlib
from fitlib.resman import RESMAN

## from fitpack analysis
from analysis.corelib import core
from analysis.corelib import classifier
import kmeanconf as kc

from qcdlib.qpdcalc import QPDCALC
from qcdlib.ppdf import PPDF
from tools.config import conf
from qcdlib import aux, alphaS, mellin

## for DSSV
from analysis.qpdlib.sets.DSSV.DSSVcalc import DSSV

cwd = 'plots/star'

def load_lattice(flav):

    F = open('plots/star/data/lat_%s.csv'%flav,'r')
    L = F.readlines()
    F.close()
   
    L = [l.strip() for l in L]
    L = [[x for x in l.split()] for l in L]
    L = np.transpose(L)[0]
   
    X, up, do = [],[],[]
    for i in range(len(L)):
        if i==0: continue
        X .append(float(L[i].split(',')[0]))
        do.append(float(L[i].split(',')[1]))
        up.append(float(L[i].split(',')[2]))
   
    X,do,up = np.array(X),np.array(do),np.array(up)
    return X, do, up

def load_asym(model):

  if model=='MC': 
      F = open('plots/star/data/mesoncloud.csv','r')
      L = F.readlines()
      F.close()

      L = [l.strip() for l in L]
      L = [[x for x in l.split()] for l in L]
      L = np.transpose(L)[0]

      X, A = [],[]
      for i in range(len(L)):
          if i==0: continue
          X  .append(float(L[i].split(',')[0]))
          A  .append(float(L[i].split(',')[1]))

      X,A = np.array(X),np.array(A)
      return X, A

  if model=='stat': 
      F = open('plots/star/data/statmodel.csv','r')
      L = F.readlines()
      F.close()

      L = [l.strip() for l in L]
      L = [[x for x in l.split()] for l in L]
      #L = np.transpose(L)[0]

      X, UB, DB, UBUP, UBDO, DBUP, DBDO = [],[],[],[],[],[],[]
      for i in range(len(L)):
          if i==0: continue
          X   .append(float(L[i][0].split(',')[0]))
          UB  .append(float(L[i][0].split(',')[1]))
          DB  .append(float(L[i][0].split(',')[2]))
          UBUP.append(float(L[i][0].split(',')[3]))
          UBDO.append(float(L[i][0].split(',')[4]))
          DBUP.append(float(L[i][0].split(',')[5]))
          DBDO.append(float(L[i][0].split(',')[6]))

      X,UB,DB,UBUP,UBDO,DBUP,DBDO = np.array(X),np.array(UB),np.array(DB),np.array(UBUP),np.array(UBDO),np.array(DBUP),np.array(DBDO)

      sigUB = UBUP - UBDO
      sigDB = DBUP - DBDO

      sig = np.sqrt(sigUB**2 + sigDB**2)

      A = UB-DB
      up  = A + sig
      do  = A - sig
      return X, A, up, do

def load_DSSV(flav):

    F = open('plots/star/data/DSSV19_%s.csv'%flav,'r')
    L = F.readlines()
    F.close()
   
    L = [l.strip() for l in L]
    L = [[x for x in l.split()] for l in L]
    L = np.transpose(L)[0]
   
    X, up, do = [],[],[]
    for i in range(len(L)):
        if i==0: continue
        X .append(float(L[i].split(',')[0]))
        do.append(float(L[i].split(',')[1]))
        up.append(float(L[i].split(',')[2]))
   
    X,do,up = np.array(X),np.array(do),np.array(up)
    return X, do, up

if __name__ == "__main__":

    #--new
    wdir1 = 'results/star/final'
    #--old
    wdir2 = 'results/old_star/final'

    #--without W
    #--new
    wdir3 = 'results/star/noRHIC2'
    #--old
    wdir4 = 'results/pol/step31'

    WDIR = [wdir1,wdir2,wdir3,wdir4]

    nrows,ncols=1,2
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs,axLs = {},{}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)
        divider = make_axes_locatable(axs[i+1])
        axLs[i+1] = divider.append_axes("right",size=3.51,pad=0,sharey=axs[i+1])
        axLs[i+1].set_xlim(0.1,0.9)
        axLs[i+1].spines['left'].set_visible(False)
        axLs[i+1].yaxis.set_ticks_position('right')
        py.setp(axLs[i+1].get_xticklabels(),visible=True)

        axs[i+1].spines['right'].set_visible(False)

    hand = {}
    j = 0

    Q2 = 10

    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()


        data=load('%s/data/ppdf-Q2=%3.5f.dat'%(wdir,Q2))

        X=data['X']

        flav = 'ub-db'
        mean = np.mean(data['XF'][flav],axis=0)
        std = np.std(data['XF'][flav],axis=0)

        #--plot average and standard deviation
        if j == 0:
            ax,axL = axs[1],axLs[1]
            thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
            axL.           fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
        if j == 1:
            ax,axL = axs[1],axLs[1]
            thy_band1 = ax.fill_between(X,(mean-std),(mean+std),fc='yellow',alpha=0.15,zorder=2,ec='darkgoldenrod',lw=2.0)
            axL.           fill_between(X,(mean-std),(mean+std),fc='yellow',alpha=0.15,zorder=2,ec='darkgoldenrod',lw=2.0)
        if j == 2:
            ax,axL = axs[2],axLs[2]
            thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
            axL.           fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
        if j == 3:
            ax,axL = axs[2],axLs[2]
            thy_band1 = ax.fill_between(X,(mean-std),(mean+std),fc='yellow',alpha=0.15,zorder=2,ec='darkgoldenrod',lw=2.0)
            axL.           fill_between(X,(mean-std),(mean+std),fc='yellow',alpha=0.15,zorder=2,ec='darkgoldenrod',lw=2.0)

        j+=1


    for i in range(N):
          axs[i+1].set_xlim(8e-3,0.1)
          axs[i+1].semilogx()

          axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
          axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
          axs[i+1].set_xticks([0.01,0.1])
          axs[i+1].set_xticklabels([r'$0.01$',r'$0.1$'])

          axLs[i+1].set_xlim(0.1,0.5)

          axLs[i+1].tick_params(axis='both', which='major', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=8)
          axLs[i+1].tick_params(axis='both', which='minor', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=4)
          axLs[i+1].set_xticks([0.3])
          axLs[i+1].set_xticklabels([r'$0.3$'])
          minorLocator = MultipleLocator(0.01)
          majorLocator = MultipleLocator(0.04)
          axs[i+1].yaxis.set_minor_locator(minorLocator)
          axs[i+1].yaxis.set_major_locator(majorLocator)
          minorLocator = MultipleLocator(0.1)
          axLs[i+1].xaxis.set_minor_locator(minorLocator)

    axs [2].tick_params(labelleft=False)
    axLs[2].tick_params(labelleft=False)

    axs [1].set_ylim(-0.015,0.100)
    axs [1] .set_yticks([0,0.02,0.04,0.06,0.08])
    axLs[1].set_yticklabels([r'$0$',r'$0.02$',r'$0.04$',r'$0.06$',r'$0.08$'])
   
    axs [2].set_ylim(-0.015,0.100)
    axs [2].set_yticks([0,0.02,0.04,0.06,0.08])
    axLs[2].set_yticklabels([r'$0$',r'$0.02$',r'$0.04$',r'$0.06$',r'$0.08$'])

    for i in range(N):
        axs [i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axLs[i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)

    axs [1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
    axLs[1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
    axs [2].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
    axLs[2].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)

    axLs[1].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[1].xaxis.set_label_coords(0.95,0.00)
    axLs[2].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[2].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.07,0.83,r'\boldmath{$x (\Delta \bar{u} \!-\! \Delta \bar{d})$}', transform=axs[1].transAxes,size=40)

    axs[1].text(0.07,0.05,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[1].transAxes,size=25)
    #axs[2] .text(0.07,0.85,r'$2.5 < Q^2 < %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[2].transAxes,size=30)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(thy_band1)

    labels.append(r'\textrm{\textbf{JAM (New)}}')
    labels.append(r'\textrm{\textbf{JAM (Old)}}')

    #legend1 = axLs[1].legend(handles,labels,loc=(-1.20,0.63),fontsize=24,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    legend1 = axLs[1].legend(handles,labels,loc='upper right',fontsize=24,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axLs[1].add_artist(legend1)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(thy_band1)

    labels.append(r'\textrm{\textbf{No \boldmath$W$ (New)}}')
    labels.append(r'\textrm{\textbf{No \boldmath$W$ (Old)}}')
    axs[2].legend(handles,labels,loc='upper left',fontsize=24,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.0, handler_map = {tuple:AnyObjectHandler()})

    

    py.tight_layout()
    py.subplots_adjust(hspace=0,wspace=0,top=0.99,right=0.99,left=0.05)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/compare'%cwd
    filename+='.png'
    #filename+='.pdf'
    axs[1] .set_rasterized(True)
    axLs[1].set_rasterized(True)
    axs[2] .set_rasterized(True)
    axLs[2].set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)
 





 
  
        
        
        
        
        
        
        
        
        
        
        
        

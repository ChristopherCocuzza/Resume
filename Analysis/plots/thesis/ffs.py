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
from scipy.integrate import cumtrapz,quad,fixed_quad

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

import lhapdf

#--for DSSV
import analysis.qpdlib.sets.DSSV.dssvlib as dssvlib
from analysis.qpdlib.sets.DSSV.DSSVcalc import DSSV

cwd = 'plots/thesis'


def plot_pion():
   
    wdir1 = 'results/star/final'

    WDIR = [wdir1]

    nrows,ncols=3,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)

    hand = {}
    j = 0

    Q2 = 100

    flavs = ['u','d','s','ub','db','sb','c','b','g']

    JAM20 = QPDCALC('JAM20-SIDIS_FF_pion_nlo',ismc=True)
    MAP10 = QPDCALC('MAPFF10NNLOPIp',ismc=True)
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data=load('%s/data/ffpion-Q2=%3.5f.dat'%(wdir,Q2))

        for flav in flavs:

            X=data['X']
            if   flav=='u':  ax = axs[1]
            elif flav=='d':  ax = axs[2]
            elif flav=='s':  ax = axs[3]
            elif flav=='ub': ax = axs[4]
            elif flav=='db': ax = axs[5]
            elif flav=='sb': ax = axs[6]
            elif flav=='c':  ax = axs[7]
            elif flav=='b':  ax = axs[8]
            elif flav=='g':  ax = axs[9]

            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            #--plot average and standard deviation
            if j == 0:
                thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)

            #--plot JAM20
            if j==0:
                ff = JAM20.get_xpdf(flav,X,Q2=Q2) 

                color = 'blue'
                alpha = 1.0
                hand['JAM20'] = ax.fill_between(X,ff['xfmin'],ff['xfmax'],fc=color,alpha=alpha,zorder=5,lw=1.5)

            #--plot MAP10
            if j==0:
                ff = MAP10.get_xpdf(flav,X,Q2=Q2) 

                color = 'green'
                alpha = 0.8
                hand['MAP10'] = ax.fill_between(X,ff['xfmin'],ff['xfmax'],fc=color,alpha=alpha,zorder=4,lw=1.5)

        j+=1



   
    for i in range(N):
          axs[i+1].set_xlim(0.2,0.9)

          axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
          axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
          axs[i+1].set_xticks([0.2,0.4,0.6,0.8])

          minorLocator = MultipleLocator(0.05)
          axs[i+1].xaxis.set_minor_locator(minorLocator)

    axs [1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axs [5].tick_params(labelbottom=False)
    axs [6].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axs [3].tick_params(labelleft=False)
    axs [5].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axs [8].tick_params(labelleft=False)
    axs [9].tick_params(labelleft=False)


    for i in [1,2,3]:
        axs[i].set_ylim(0,0.70)

        axs[i].set_yticks([0.2,0.4,0.6])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)
   
    for i in [4,5,6]:
        axs[i].set_ylim(0,0.70)

        axs[i].set_yticks([0.2,0.4,0.6])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

    for i in [7,8,9]:
        axs[i].set_ylim(0,0.70)

        axs[i].set_yticks([0.2,0.4,0.6])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

    axs[7].set_xlabel(r'\boldmath$z$',size=40)   
    axs[7].xaxis.set_label_coords(0.95,0.00)
    axs[8].set_xlabel(r'\boldmath$z$',size=40)   
    axs[8].xaxis.set_label_coords(0.95,0.00)
    axs[9].set_xlabel(r'\boldmath$z$',size=40)   
    axs[9].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{u}$'       , transform=axs[1].transAxes,size=40)
    axs[2].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{d}$'       , transform=axs[2].transAxes,size=40)
    axs[3].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{s}$'       , transform=axs[3].transAxes,size=40)
    axs[4].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{\bar{u}}$' , transform=axs[4].transAxes,size=40)
    axs[5].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{\bar{d}}$' , transform=axs[5].transAxes,size=40)
    axs[6].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{\bar{s}}$' , transform=axs[6].transAxes,size=40)
    axs[7].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{c}$'       , transform=axs[7].transAxes,size=40)
    axs[8].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{b}$'       , transform=axs[8].transAxes,size=40)
    axs[9].text(0.80,0.83,r'\boldmath$z D^{\pi^+}_{g}$'       , transform=axs[9].transAxes,size=40)

    axs[6].text(0.07,0.85,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[6].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(hand['JAM20'])
    handles.append(hand['MAP10'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{JAM20-SIDIS}')
    labels.append(r'\textrm{MAPFF1.0}')

    legend1 = axs[2].legend(handles,labels,loc='upper left',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axs[2].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/FFs-pion'%cwd
    filename+='.png'
    #filename+='.pdf'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

def plot_kaon():
   
    wdir1 = 'results/star/final'

    WDIR = [wdir1]

    nrows,ncols=3,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)

    hand = {}
    j = 0

    Q2 = 100

    flavs = ['u','d','s','ub','db','sb','c','b','g']

    JAM20 = QPDCALC('JAM20-SIDIS_FF_kaon_nlo',ismc=True)
    MAP10 = QPDCALC('MAPFF10NNLOKAp',ismc=True)
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data=load('%s/data/ffkaon-Q2=%3.5f.dat'%(wdir,Q2))

        for flav in flavs:

            X=data['X']
            if   flav=='u':  ax = axs[1]
            elif flav=='d':  ax = axs[2]
            elif flav=='s':  ax = axs[3]
            elif flav=='ub': ax = axs[4]
            elif flav=='db': ax = axs[5]
            elif flav=='sb': ax = axs[6]
            elif flav=='c':  ax = axs[7]
            elif flav=='b':  ax = axs[8]
            elif flav=='g':  ax = axs[9]

            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            #--plot average and standard deviation
            if j == 0:
                thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)

            #--plot JAM20
            if j==0:
                ff = JAM20.get_xpdf(flav,X,Q2=Q2) 

                color = 'blue'
                alpha = 1.0
                hand['JAM20'] = ax.fill_between(X,ff['xfmin'],ff['xfmax'],fc=color,alpha=alpha,zorder=5,lw=1.5)

            #--plot MAP10
            if j==0:
                ff = MAP10.get_xpdf(flav,X,Q2=Q2) 

                color = 'green'
                alpha = 0.8
                hand['MAP10'] = ax.fill_between(X,ff['xfmin'],ff['xfmax'],fc=color,alpha=alpha,zorder=4,lw=1.5)

        j+=1



   
    for i in range(N):
          axs[i+1].set_xlim(0.2,0.9)

          axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
          axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
          axs[i+1].set_xticks([0.2,0.4,0.6,0.8])

          minorLocator = MultipleLocator(0.05)
          axs[i+1].xaxis.set_minor_locator(minorLocator)

    axs [1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axs [5].tick_params(labelbottom=False)
    axs [6].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axs [3].tick_params(labelleft=False)
    axs [5].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axs [8].tick_params(labelleft=False)
    axs [9].tick_params(labelleft=False)


    for i in [1,2,3]:
        axs[i].set_ylim(0,0.40)

        axs[i].set_yticks([0.1,0.2,0.3])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)
   
    for i in [4,5,6]:
        axs[i].set_ylim(0,0.50)

        axs[i].set_yticks([0.1,0.2,0.3,0.4])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

    for i in [7,8,9]:
        axs[i].set_ylim(0,0.30)

        axs[i].set_yticks([0.1,0.2])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

    axs[7].set_xlabel(r'\boldmath$z$',size=40)   
    axs[7].xaxis.set_label_coords(0.95,0.00)
    axs[8].set_xlabel(r'\boldmath$z$',size=40)   
    axs[8].xaxis.set_label_coords(0.95,0.00)
    axs[9].set_xlabel(r'\boldmath$z$',size=40)   
    axs[9].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.80,0.83,r'\boldmath$z D^{K^+}_{u}$'       , transform=axs[1].transAxes,size=40)
    axs[2].text(0.80,0.83,r'\boldmath$z D^{K^+}_{d}$'       , transform=axs[2].transAxes,size=40)
    axs[3].text(0.80,0.83,r'\boldmath$z D^{K^+}_{s}$'       , transform=axs[3].transAxes,size=40)
    axs[4].text(0.80,0.83,r'\boldmath$z D^{K^+}_{\bar{u}}$' , transform=axs[4].transAxes,size=40)
    axs[5].text(0.80,0.83,r'\boldmath$z D^{K^+}_{\bar{d}}$' , transform=axs[5].transAxes,size=40)
    axs[6].text(0.80,0.83,r'\boldmath$z D^{K^+}_{\bar{s}}$' , transform=axs[6].transAxes,size=40)
    axs[7].text(0.80,0.83,r'\boldmath$z D^{K^+}_{c}$'       , transform=axs[7].transAxes,size=40)
    axs[8].text(0.80,0.83,r'\boldmath$z D^{K^+}_{b}$'       , transform=axs[8].transAxes,size=40)
    axs[9].text(0.80,0.83,r'\boldmath$z D^{K^+}_{g}$'       , transform=axs[9].transAxes,size=40)

    axs[4].text(0.07,0.85,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[4].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(hand['JAM20'])
    handles.append(hand['MAP10'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{JAM20-SIDIS}')
    labels.append(r'\textrm{MAPFF1.0}')

    legend1 = axs[2].legend(handles,labels,loc='upper left',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axs[2].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/FFs-kaon'%cwd
    filename+='.png'
    #filename+='.pdf'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

def plot_hadron():
   
    wdir1 = 'results/star/final'

    WDIR = [wdir1]

    nrows,ncols=3,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)

    hand = {}
    j = 0

    Q2 = 100

    flavs = ['u','d','s','ub','db','sb','c','b','g']

    JAM20 = QPDCALC('JAM20-SIDIS_FF_hadron_nlo',ismc=True)
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data=load('%s/data/ffhadron-Q2=%3.5f.dat'%(wdir,Q2))

        for flav in flavs:

            X=data['X']
            if   flav=='u':  ax = axs[1]
            elif flav=='d':  ax = axs[2]
            elif flav=='s':  ax = axs[3]
            elif flav=='ub': ax = axs[4]
            elif flav=='db': ax = axs[5]
            elif flav=='sb': ax = axs[6]
            elif flav=='c':  ax = axs[7]
            elif flav=='b':  ax = axs[8]
            elif flav=='g':  ax = axs[9]

            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            #--plot average and standard deviation
            if j == 0:
                thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)

            #--plot JAM20
            if j==0:
                ff = JAM20.get_xpdf(flav,X,Q2=Q2) 

                color = 'blue'
                alpha = 1.0
                hand['JAM20'] = ax.fill_between(X,ff['xfmin'],ff['xfmax'],fc=color,alpha=alpha,zorder=5,lw=1.5)
                #hand['JAM20'] = ax.fill_between(X,ff['xfmin'],ff['xfmax'],fc=color,alpha=alpha,zorder=5,ec='lightgray',lw=1.5)
                #ax .plot(X,ff['xfmax'],color=ec,alpha=0.4,zorder=5,lw=2.0)
                #ax .plot(X,ff['xfmin'],color=ec,alpha=0.4,zorder=5,lw=2.0)

        j+=1



   
    for i in range(N):
          axs[i+1].set_xlim(0.2,0.9)

          axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
          axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
          axs[i+1].set_xticks([0.2,0.4,0.6,0.8])

          minorLocator = MultipleLocator(0.05)
          axs[i+1].xaxis.set_minor_locator(minorLocator)

    axs [1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axs [5].tick_params(labelbottom=False)
    axs [6].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axs [3].tick_params(labelleft=False)
    axs [5].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axs [8].tick_params(labelleft=False)
    axs [9].tick_params(labelleft=False)


    for i in [1,2,3]:
        axs[i].set_ylim(0,0.90)

        axs[i].set_yticks([0.2,0.4,0.6,0.8])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)
   
    for i in [4,5,6]:
        axs[i].set_ylim(0,0.85)

        axs[i].set_yticks([0.2,0.4,0.6,0.8])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

    for i in [7,8,9]:
        axs[i].set_ylim(0,0.85)

        axs[i].set_yticks([0.2,0.4,0.6,0.8])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

    axs[7].set_xlabel(r'\boldmath$z$',size=40)   
    axs[7].xaxis.set_label_coords(0.95,0.00)
    axs[8].set_xlabel(r'\boldmath$z$',size=40)   
    axs[8].xaxis.set_label_coords(0.95,0.00)
    axs[9].set_xlabel(r'\boldmath$z$',size=40)   
    axs[9].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.80,0.83,r'\boldmath$z D^{h^+}_{u}$'       , transform=axs[1].transAxes,size=40)
    axs[2].text(0.80,0.83,r'\boldmath$z D^{h^+}_{d}$'       , transform=axs[2].transAxes,size=40)
    axs[3].text(0.80,0.83,r'\boldmath$z D^{h^+}_{s}$'       , transform=axs[3].transAxes,size=40)
    axs[4].text(0.80,0.83,r'\boldmath$z D^{h^+}_{\bar{u}}$' , transform=axs[4].transAxes,size=40)
    axs[5].text(0.80,0.83,r'\boldmath$z D^{h^+}_{\bar{d}}$' , transform=axs[5].transAxes,size=40)
    axs[6].text(0.80,0.83,r'\boldmath$z D^{h^+}_{\bar{s}}$' , transform=axs[6].transAxes,size=40)
    axs[7].text(0.80,0.83,r'\boldmath$z D^{h^+}_{c}$'       , transform=axs[7].transAxes,size=40)
    axs[8].text(0.80,0.83,r'\boldmath$z D^{h^+}_{b}$'       , transform=axs[8].transAxes,size=40)
    axs[9].text(0.80,0.83,r'\boldmath$z D^{h^+}_{g}$'       , transform=axs[9].transAxes,size=40)

    axs[4].text(0.07,0.85,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[4].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(hand['JAM20'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{JAM20-SIDIS}')

    legend1 = axs[2].legend(handles,labels,loc='upper left',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axs[2].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/FFs-hadron'%cwd
    filename+='.png'
    #filename+='.pdf'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

def plot_all():
   
    wdir1 = 'results/star/final'

    WDIR = [wdir1]

    nrows,ncols=3,3
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*9,nrows*5))
    axs = {}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)

    hand = {}
    j = 0

    Q2 = 100

    flavs = ['u','d','s','ub','db','sb','c','b','g']

    hads = ['pion','kaon','hadron','res']

    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data = {}
        data['pion']   = load('%s/data/ffpion-Q2=%3.5f.dat'%(wdir,Q2))
        data['kaon']   = load('%s/data/ffkaon-Q2=%3.5f.dat'%(wdir,Q2))
        data['hadron'] = load('%s/data/ffhadron-Q2=%3.5f.dat'%(wdir,Q2))

        data['res'] = {}
        data['res']['X'] = data['pion']['X']
        data['res']['XF'] = {}

        for had in hads:

            if   had=='pion':   color,zorder = 'red'  , 4
            elif had=='kaon':   color,zorder = 'blue' , 2
            elif had=='hadron': color,zorder = 'green', 3
            elif had=='res':    color,zorder = 'gold' , 1

            for flav in flavs:
                data['res']['XF'][flav] = np.array(data['hadron']['XF'][flav]) - np.array(data['pion']['XF'][flav]) - np.array(data['kaon']['XF'][flav])

                X=data[had]['X']
                if   flav=='u':  ax = axs[1]
                elif flav=='d':  ax = axs[2]
                elif flav=='s':  ax = axs[3]
                elif flav=='ub': ax = axs[4]
                elif flav=='db': ax = axs[5]
                elif flav=='sb': ax = axs[6]
                elif flav=='c':  ax = axs[7]
                elif flav=='b':  ax = axs[8]
                elif flav=='g':  ax = axs[9]

                mean = np.mean(data[had]['XF'][flav],axis=0)
                std  = np.std (data[had]['XF'][flav],axis=0)

                #--plot average and standard deviation
                if j == 0:
                    hand[had] = ax.fill_between(X,(mean-std),(mean+std),color=color,alpha=0.8,zorder=zorder)


        j+=1



   
    for i in range(N):
          axs[i+1].set_xlim(0.2,0.9)

          axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
          axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
          axs[i+1].set_xticks([0.2,0.4,0.6,0.8])

          minorLocator = MultipleLocator(0.05)
          axs[i+1].xaxis.set_minor_locator(minorLocator)

    axs [1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axs [5].tick_params(labelbottom=False)
    axs [6].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axs [3].tick_params(labelleft=False)
    axs [5].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axs [8].tick_params(labelleft=False)
    axs [9].tick_params(labelleft=False)


    for i in [1,2,3]:
        axs[i].set_ylim(0,0.90)

        axs[i].set_yticks([0.2,0.4,0.6,0.8])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)
   
    for i in [4,5,6]:
        axs[i].set_ylim(0,0.85)

        axs[i].set_yticks([0.2,0.4,0.6,0.8])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

    for i in [7,8,9]:
        axs[i].set_ylim(0,0.85)

        axs[i].set_yticks([0.2,0.4,0.6,0.8])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

    axs[7].set_xlabel(r'\boldmath$z$',size=40)   
    axs[7].xaxis.set_label_coords(0.95,0.00)
    axs[8].set_xlabel(r'\boldmath$z$',size=40)   
    axs[8].xaxis.set_label_coords(0.95,0.00)
    axs[9].set_xlabel(r'\boldmath$z$',size=40)   
    axs[9].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.80,0.83,r'\boldmath$z D_{u}$'       , transform=axs[1].transAxes,size=40)
    axs[2].text(0.80,0.83,r'\boldmath$z D_{d}$'       , transform=axs[2].transAxes,size=40)
    axs[3].text(0.80,0.83,r'\boldmath$z D_{s}$'       , transform=axs[3].transAxes,size=40)
    axs[4].text(0.80,0.83,r'\boldmath$z D_{\bar{u}}$' , transform=axs[4].transAxes,size=40)
    axs[5].text(0.80,0.83,r'\boldmath$z D_{\bar{d}}$' , transform=axs[5].transAxes,size=40)
    axs[6].text(0.80,0.83,r'\boldmath$z D_{\bar{s}}$' , transform=axs[6].transAxes,size=40)
    axs[7].text(0.80,0.83,r'\boldmath$z D_{c}$'       , transform=axs[7].transAxes,size=40)
    axs[8].text(0.80,0.83,r'\boldmath$z D_{b}$'       , transform=axs[8].transAxes,size=40)
    axs[9].text(0.80,0.83,r'\boldmath$z D_{g}$'       , transform=axs[9].transAxes,size=40)

    axs[6].text(0.07,0.85,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[6].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(hand['pion'])
    handles.append(hand['kaon'])
    handles.append(hand['hadron'])
    handles.append(hand['res'])

    labels.append(r'\boldmath$\pi^+$')
    labels.append(r'\boldmath$K^+$')
    labels.append(r'\boldmath$h^+$')
    labels.append(r'\boldmath$\delta h^+$')

    legend1 = axs[2].legend(handles,labels,loc='upper left',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=2,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axs[2].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/FFs-all'%cwd
    filename+='.png'
    #filename+='.pdf'
    for i in range(N):
        axs[i+1] .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)




if __name__ == "__main__":

    plot_pion()
    plot_kaon()
    plot_hadron()
    plot_all()




 
  
        
        
        
        
        
        
        
        
        
        
        
        

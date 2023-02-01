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

def load_DSSV_gluon():

    F = open('plots/thesis/data/DSSV19gluon.csv','r')
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


def plot_ppdfs():
   
    wdir1 = 'results/star/final'
    wdir2 = 'results/star/pos2'
    wdir3 = 'results/star/noRHIC2'

    WDIR = [wdir1,wdir2,wdir3]

    nrows,ncols=3,2
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

    flavs = ['up','dp','ub','db','sp','sm']

    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data=load('%s/data/ppdf-Q2=%3.5f.dat'%(wdir,Q2))

        for flav in flavs:

            X=data['X']
            if   flav=='up': ax,axL = axs[1],axLs[1]
            elif flav=='dp': ax,axL = axs[2],axLs[2]
            elif flav=='ub': ax,axL = axs[3],axLs[3]
            elif flav=='db': ax,axL = axs[4],axLs[4]
            elif flav=='sp': ax,axL = axs[5],axLs[5]
            elif flav=='sm': ax,axL = axs[6],axLs[6]

            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            #--plot average and standard deviation
            if j == 0:
                thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
                axL.           fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
            if j == 1:
                color='none'
                alpha=0.4
                thy_band1 = ax.fill_between(X,(mean-std),(mean+std),facecolor=color,alpha=alpha,zorder=3,hatch='//',edgecolor='black')
                axL.           fill_between(X,(mean-std),(mean+std),facecolor=color,alpha=alpha,zorder=3,hatch='//',edgecolor='black')
            if j == 2:
                thy_band2 = ax.fill_between(X,(mean-std),(mean+std),fc='yellow',alpha=0.15,zorder=2,ec='darkgoldenrod',lw=2.0)
                axL.           fill_between(X,(mean-std),(mean+std),fc='yellow',alpha=0.15,zorder=2,ec='darkgoldenrod',lw=2.0)


        j+=1



   
    for i in range(N):
          axs[i+1].set_xlim(8e-3,0.1)
          axs[i+1].semilogx()

          axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
          axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
          axs[i+1].set_xticks([0.01,0.03,0.1])
          axs[i+1].set_xticklabels([r'$0.01$',r'$0.03$',r'$0.1$'])

          axLs[i+1].set_xlim(0.1,0.9)

          axLs[i+1].tick_params(axis='both', which='major', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=8)
          axLs[i+1].tick_params(axis='both', which='minor', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=4)
          axLs[i+1].set_xticks([0.3,0.5,0.7])
          minorLocator = MultipleLocator(0.1)
          axLs[i+1].xaxis.set_minor_locator(minorLocator)

    axs [1].tick_params(labelbottom=False)
    axLs[1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axLs[2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axLs[3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axLs[4].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axLs[2].tick_params(labelleft=False)
    axs [4].tick_params(labelleft=False)
    axLs[4].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axLs[6].tick_params(labelleft=False)


    for i in [1,2]:
        axs[i].set_ylim(-0.15,0.39)

        axs[i].set_yticks([-0.10,0,0.1,0.2,0.3])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)
   
    for i in [3,4]:
        axs[i].set_ylim(-0.03,0.05)

        axs[i].set_yticks([-0.02,0,0.02,0.04])
        minorLocator = MultipleLocator(0.01)
        axs[i].yaxis.set_minor_locator(minorLocator)

    for i in [5,6]:
        axs[i].set_ylim(-0.11,0.11)

        axs[i].set_yticks([-0.08,-0.04,0,0.04,0.08])
        minorLocator = MultipleLocator(0.01)
        axs[i].yaxis.set_minor_locator(minorLocator)



    for i in range(N):
        axs [i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axLs[i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axs [i+1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
        axLs[i+1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)


    axLs[5].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[5].xaxis.set_label_coords(0.95,0.00)
    axLs[6].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[6].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.07,0.83,r'\boldmath{$x \Delta u^+$'                 , transform=axs[1].transAxes,size=40)
    axs[2].text(0.07,0.83,r'\boldmath{$x \Delta d^+$'                 , transform=axs[2].transAxes,size=40)
    axs[3].text(0.07,0.83,r'\boldmath{$x \Delta \bar{u}$'             , transform=axs[3].transAxes,size=40)
    axs[4].text(0.07,0.83,r'\boldmath{$x \Delta \bar{d}$'             , transform=axs[4].transAxes,size=40)
    axs[5].text(0.07,0.83,r'\boldmath{$x \Delta s^+$', transform=axs[5].transAxes,size=40)
    axs[6].text(0.07,0.83,r'\boldmath{$x \Delta s^-$', transform=axs[6].transAxes,size=40)

    axs[2].text(0.07,0.60,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[2].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_band2)
    handles.append(thy_band0)
    handles.append(thy_band1)

    labels.append(r'\boldmath${\rm no} \, W$')
    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{\textbf{+pos}}')

    legend1 = axLs[2].legend(handles,labels,loc='upper right',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axLs[2].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/PPDFs'%cwd
    filename+='.png'
    #filename+='.pdf'
    for i in range(N):
        axs[i+1] .set_rasterized(True)
        axLs[i+1].set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

def plot_ppdfs_groups():
   
    #wdir1 = 'results/star/final'
    wdir1 = 'results/star/pos2'

    WDIR = [wdir1]

    nrows,ncols=3,2
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

    flavs = ['up','dp','ub','db','sp','sm']
    NNPDF = QPDCALC('NNPDFpol11_100',ismc=True)
    dssv = DSSV()

    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data=load('%s/data/ppdf-Q2=%3.5f.dat'%(wdir,Q2))

        for flav in flavs:

            X=data['X']
            if   flav=='up': ax,axL = axs[1],axLs[1]
            elif flav=='dp': ax,axL = axs[2],axLs[2]
            elif flav=='ub': ax,axL = axs[3],axLs[3]
            elif flav=='db': ax,axL = axs[4],axLs[4]
            elif flav=='sp': ax,axL = axs[5],axLs[5]
            elif flav=='sm': ax,axL = axs[6],axLs[6]

            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            #--plot average and standard deviation
            if j == 0:
                thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
                axL.           fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)

            #--plot NNPDFpol1.1
            if j==0:
                if flav=='sm': pass
                else:
                    ppdf = NNPDF.get_xpdf(flav,X,Q2=Q2) 

                    color = 'lightsteelblue'
                    ec    = 'gray'
                    alpha = 0.6
                    hand['NNPDF'] = ax.fill_between(X,ppdf['xfmin'],ppdf['xfmax'],fc=color,alpha=alpha,zorder=1,ec='lightgray',lw=1.5)
                    axL.               fill_between(X,ppdf['xfmin'],ppdf['xfmax'],fc=color,alpha=alpha,zorder=1,ec='lightgray',lw=1.5)
                    ax .plot(X,ppdf['xfmax'],color=ec,alpha=0.4,zorder=5,lw=2.0)
                    axL.plot(X,ppdf['xfmax'],color=ec,alpha=0.4,zorder=5,lw=2.0)
                    ax .plot(X,ppdf['xfmin'],color=ec,alpha=0.4,zorder=5,lw=2.0)
                    axL.plot(X,ppdf['xfmin'],color=ec,alpha=0.4,zorder=5,lw=2.0)

            #--plot DSSV08
            if j==0:
                if flav=='sm': pass
                else:
                    X=10**np.linspace(-4,-1,200)
                    X=np.append(X,np.linspace(0.1,0.99,200))
                    if   flav=='up': ppdf = dssv.xfxQ2( 2,X,Q2) + dssv.xfxQ2(-2,X,Q2) 
                    elif flav=='dp': ppdf = dssv.xfxQ2( 1,X,Q2) + dssv.xfxQ2(-1,X,Q2) 
                    elif flav=='ub': ppdf = dssv.xfxQ2(-2,X,Q2)
                    elif flav=='db': ppdf = dssv.xfxQ2(-1,X,Q2)
                    elif flav=='sp': ppdf = dssv.xfxQ2( 3,X,Q2) + dssv.xfxQ2(-3,X,Q2) 
                    mean = ppdf[0]
                    std  = 0
                    for i in range(1,20):
                        std += (ppdf[i] - ppdf[-i])**2
                    std = np.sqrt(std)/2.0

                    color = 'springgreen'
                    ec    = 'green'
                    alpha = 0.7
                    hand['DSSV'] = ax.fill_between(X,mean-std,mean+std,fc=color,alpha=alpha,zorder=1,ec='limegreen',lw=1.5)
                    axL.              fill_between(X,mean-std,mean+std,fc=color,alpha=alpha,zorder=1,ec='limegreen',lw=1.5)
                    ax .plot(X,mean+std,color='mediumspringgreen',alpha=0.5,zorder=5)
                    axL.plot(X,mean+std,color='mediumspringgreen',alpha=0.5,zorder=5)
                    ax .plot(X,mean-std,color='mediumspringgreen',alpha=0.5,zorder=5)
                    axL.plot(X,mean-std,color='mediumspringgreen',alpha=0.5,zorder=5)


        j+=1



   
    for i in range(N):
          axs[i+1].set_xlim(8e-3,0.1)
          axs[i+1].semilogx()

          axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
          axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
          axs[i+1].set_xticks([0.01,0.03,0.1])
          axs[i+1].set_xticklabels([r'$0.01$',r'$0.03$',r'$0.1$'])

          axLs[i+1].set_xlim(0.1,0.9)

          axLs[i+1].tick_params(axis='both', which='major', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=8)
          axLs[i+1].tick_params(axis='both', which='minor', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=4)
          axLs[i+1].set_xticks([0.3,0.5,0.7])
          minorLocator = MultipleLocator(0.1)
          axLs[i+1].xaxis.set_minor_locator(minorLocator)

    axs [1].tick_params(labelbottom=False)
    axLs[1].tick_params(labelbottom=False)
    axs [2].tick_params(labelbottom=False)
    axLs[2].tick_params(labelbottom=False)
    axs [3].tick_params(labelbottom=False)
    axLs[3].tick_params(labelbottom=False)
    axs [4].tick_params(labelbottom=False)
    axLs[4].tick_params(labelbottom=False)


    axs [2].tick_params(labelleft=False)
    axLs[2].tick_params(labelleft=False)
    axs [4].tick_params(labelleft=False)
    axLs[4].tick_params(labelleft=False)
    axs [6].tick_params(labelleft=False)
    axLs[6].tick_params(labelleft=False)


    for i in [1,2]:
        axs[i].set_ylim(-0.15,0.39)

        axs[i].set_yticks([-0.10,0,0.1,0.2,0.3])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)
   
    for i in [3,4]:
        axs[i].set_ylim(-0.03,0.05)

        axs[i].set_yticks([-0.02,0,0.02,0.04])
        minorLocator = MultipleLocator(0.01)
        axs[i].yaxis.set_minor_locator(minorLocator)

    for i in [5,6]:
        axs[i].set_ylim(-0.11,0.11)

        axs[i].set_yticks([-0.08,-0.04,0,0.04,0.08])
        minorLocator = MultipleLocator(0.01)
        axs[i].yaxis.set_minor_locator(minorLocator)



    for i in range(N):
        axs [i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axLs[i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axs [i+1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
        axLs[i+1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)

    axLs[5].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[5].xaxis.set_label_coords(0.95,0.00)
    axLs[6].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[6].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.07,0.83,r'\boldmath{$x \Delta u^+$'                 , transform=axs[1].transAxes,size=40)
    axs[2].text(0.07,0.83,r'\boldmath{$x \Delta d^+$'                 , transform=axs[2].transAxes,size=40)
    axs[3].text(0.07,0.83,r'\boldmath{$x \Delta \bar{u}$'             , transform=axs[3].transAxes,size=40)
    axs[4].text(0.07,0.83,r'\boldmath{$x \Delta \bar{d}$'             , transform=axs[4].transAxes,size=40)
    axs[5].text(0.07,0.83,r'\boldmath{$x \Delta s^+$', transform=axs[5].transAxes,size=40)
    axs[6].text(0.07,0.83,r'\boldmath{$x \Delta s^-$', transform=axs[6].transAxes,size=40)

    axs[2].text(0.07,0.60,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[2].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(hand['NNPDF'])
    handles.append(hand['DSSV'])

    labels.append(r'\textrm{\textbf{JAM + pos}}')
    labels.append(r'\textrm{NNPDFpol1.1}')
    labels.append(r'\textrm{DSSV08}')
    axLs[2].legend(handles,labels,loc='upper right',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.0, handler_map = {tuple:AnyObjectHandler()})

    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/PPDFs-groups'%cwd
    filename+='.png'
    #filename+='.pdf'
    for i in range(N):
        axs[i+1] .set_rasterized(True)
        axLs[i+1].set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

def plot_gluon():
  
    wdir1 = 'results/star/final'
    wdir2 = 'results/star/pos2'

    WDIR = [wdir1,wdir2]

    nrows,ncols=2,1
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
    flav = 'g'
    NNPDF = QPDCALC('NNPDFpol11_100',ismc=True)
    dssv = DSSV()

    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data=load('%s/data/ppdf-Q2=%3.5f.dat'%(wdir,Q2))

        X=data['X']

        mean = np.mean(data['XF'][flav],axis=0)
        std = np.std(data['XF'][flav],axis=0)

        #--plot average and standard deviation
        ax,axL = axs[2],axLs[2]
        if j == 1:
            thy_band1 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
            axL.           fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)

        #--plot replicas
        ax,axL = axs[1],axLs[1]
        for i in range(len(data['XF'][flav])):

            if j == 0:
                color='blue'
                alpha=0.2
                ax .plot(X,data['XF'][flav][i],color=color,alpha=alpha,zorder=2)
                axL.plot(X,data['XF'][flav][i],color=color,alpha=alpha,zorder=2)
                thy_plot0 ,= ax.plot(0,0,color=color,alpha=1.0,zorder=2)
            if j == 1:
                color='red'
                alpha=0.7
                ax .plot(X,data['XF'][flav][i],color=color,alpha=alpha,zorder=3)
                axL.plot(X,data['XF'][flav][i],color=color,alpha=alpha,zorder=3)
                thy_plot1 ,= ax.plot(0,0,color=color,alpha=1.0,zorder=2)



        j+=1



    #--plot NNPDFpol1.1
    ax,axL = axs[2],axLs[2]
    ppdf = NNPDF.get_xpdf(flav,X,Q2=Q2) 

    color = 'lightsteelblue'
    ec    = 'gray'
    alpha = 0.6
    hand['NNPDF'] = ax.fill_between(X,ppdf['xfmin'],ppdf['xfmax'],fc=color,alpha=alpha,zorder=1,ec='lightgray',lw=1.5)
    axL.               fill_between(X,ppdf['xfmin'],ppdf['xfmax'],fc=color,alpha=alpha,zorder=1,ec='lightgray',lw=1.5)
    ax .plot(X,ppdf['xfmax'],color=ec,alpha=0.4,zorder=5,lw=2.0)
    axL.plot(X,ppdf['xfmax'],color=ec,alpha=0.4,zorder=5,lw=2.0)
    ax .plot(X,ppdf['xfmin'],color=ec,alpha=0.4,zorder=5,lw=2.0)
    axL.plot(X,ppdf['xfmin'],color=ec,alpha=0.4,zorder=5,lw=2.0)

    #--plot DSSV08
    #ax,axL = axs[2],axLs[2]
    #X=10**np.linspace(-4,-1,200)
    #X=np.append(X,np.linspace(0.1,0.99,200))
    #ppdf = dssv.xfxQ2(21,X,Q2)
    #mean = ppdf[0]
    #std  = 0
    #for i in range(1,20):
    #    std += (ppdf[i] - ppdf[-i])**2
    #std = np.sqrt(std)/2.0

    #color = 'springgreen'
    #ec    = 'green'
    #alpha = 0.7
    #hand['DSSV'] = ax.fill_between(X,mean-std,mean+std,fc=color,alpha=alpha,zorder=1,ec='limegreen',lw=1.5)
    #axL.              fill_between(X,mean-std,mean+std,fc=color,alpha=alpha,zorder=1,ec='limegreen',lw=1.5)
    #ax .plot(X,mean+std,color='mediumspringgreen',alpha=0.5,zorder=5)
    #axL.plot(X,mean+std,color='mediumspringgreen',alpha=0.5,zorder=5)
    #ax .plot(X,mean-std,color='mediumspringgreen',alpha=0.5,zorder=5)
    #axL.plot(X,mean-std,color='mediumspringgreen',alpha=0.5,zorder=5)

    #--plot DSSV19
    X, do, up = load_DSSV_gluon()
    mean = (do + up)/2.0
    std  = np.abs(up - do)/2.0
    color = 'springgreen'
    ec    = 'green'
    alpha = 0.7
    hand['DSSV'] = ax.fill_between(X,mean-std,mean+std,fc=color,alpha=alpha,zorder=1,ec='limegreen',lw=1.5)
    axL.              fill_between(X,mean-std,mean+std,fc=color,alpha=alpha,zorder=1,ec='limegreen',lw=1.5)
    ax .plot(X,mean+std,color='mediumspringgreen',alpha=0.5,zorder=5)
    axL.plot(X,mean+std,color='mediumspringgreen',alpha=0.5,zorder=5)
    ax .plot(X,mean-std,color='mediumspringgreen',alpha=0.5,zorder=5)
    axL.plot(X,mean-std,color='mediumspringgreen',alpha=0.5,zorder=5)
 

    for i in range(N):
          axs[i+1].set_xlim(8e-3,0.1)
          #axs[i+1].set_xlim(1e-2,0.1)
          axs[i+1].semilogx()

          axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
          axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
          axs[i+1].set_xticks([0.01,0.03,0.1])
          axs[i+1].set_xticklabels([r'$0.01$',r'$0.03$',r'$0.1$'])

          axLs[i+1].set_xlim(0.1,0.9)

          axLs[i+1].tick_params(axis='both', which='major', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=8)
          axLs[i+1].tick_params(axis='both', which='minor', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=4)
          axLs[i+1].set_xticks([0.3,0.5,0.7])
          minorLocator = MultipleLocator(0.1)
          axLs[i+1].xaxis.set_minor_locator(minorLocator)

    for i in [1]:
        axs[i].set_ylim(-0.50,0.50)

        axs[i].set_yticks([-0.4,-0.2,0,0.2,0.4])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)
  
    for i in [2]:
        axs[i].set_ylim(-0.20,0.25)

        axs[i].set_yticks([-0.1,0,0.1,0.2])
        minorLocator = MultipleLocator(0.05)
        axs[i].yaxis.set_minor_locator(minorLocator)

 
    for i in range(N):
        axs [i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axLs[i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axs [i+1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
        axLs[i+1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)

    axs [1].tick_params(labelbottom=False)
    axLs[1].tick_params(labelbottom=False)

    axLs[2].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[2].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.07,0.08,r'\boldmath{$x \Delta g$'                 , transform=axs[1].transAxes,size=40)

    axLs[1].text(0.20,0.05,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axLs[1].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_plot0)
    handles.append(thy_plot1)

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{\textbf{+pos}}')

    legend1 = axLs[1].legend(handles,labels,loc='upper right',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axLs[1].add_artist(legend1)

    handles, labels = [],[]
    handles.append(thy_band1)
    handles.append(hand['NNPDF'])
    handles.append(hand['DSSV'])

    labels.append(r'\textrm{\textbf{JAM + pos}}')
    labels.append(r'\textrm{NNPDFpol1.1}')
    labels.append(r'\textrm{DSSV19}')
    axLs[2].legend(handles,labels,loc='lower right',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.0, handler_map = {tuple:AnyObjectHandler()})


    py.tight_layout()
    py.subplots_adjust(hspace=0.02,wspace=0.02,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/gluon'%cwd
    filename+='.png'
    #filename+='.pdf'
    for i in range(N):
        axs[i+1] .set_rasterized(True)
        axLs[i+1].set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)

if __name__ == "__main__":

    #plot_ppdfs()
    #plot_ppdfs_groups()

    plot_gluon()
    #plot_gluon_groups()





 
  
        
        
        
        
        
        
        
        
        
        
        
        

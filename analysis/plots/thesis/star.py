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
      F = open('plots/thesis/data/mesoncloud.csv','r')
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
      F = open('plots/thesis/data/statmodel.csv','r')
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

def plot_asym():
   
    wdir1 = 'results/star/final'
    wdir2 = 'results/star/pos2'
    wdir3 = 'results/star/noRHIC2'

    WDIR = [wdir1,wdir2,wdir3]
 
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

    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        ax,axL = axs[1],axLs[1]

        data=load('%s/data/ppdf-Q2=%3.5f.dat'%(wdir,Q2))

        X=data['X']

        flav = 'ub-db'
        mean = np.mean(data['XF'][flav],axis=0)
        std = np.std(data['XF'][flav],axis=0)

        #--plot average and standard deviation
        if j == 0:
            thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
            axL.           fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
            ax,axL = axs[2],axLs[2]
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
        if j == 3:
            thy_band3 = ax.fill_between(X,(mean-std),(mean+std),fc='mediumspringgreen',alpha=0.5,zorder=1,ec='green',lw=1.5)
            axL.           fill_between(X,(mean-std),(mean+std),fc='mediumspringgreen',alpha=0.5,zorder=1,ec='green',lw=1.5)
            ax.            plot(X,mean+std,color='mediumspringgreen',alpha=0.5,zorder=5)
            axL.           plot(X,mean+std,color='mediumspringgreen',alpha=0.5,zorder=5)
            ax.            plot(X,mean-std,color='mediumspringgreen',alpha=0.5,zorder=5)
            axL.           plot(X,mean-std,color='mediumspringgreen',alpha=0.5,zorder=5)

        #--plot unpolarized asymmetry
        if j==0:
            data=load('%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2))
            X=data['X']

            flav = 'db-ub'
            mean = np.mean(data['XF'][flav],axis=0)
            std  = np.std(data['XF'][flav],axis=0)

            ax,axL = axs[1],axLs[1]
            #hand['upol'] = ax.fill_between(X,(-mean-std),(-mean+std),color='cyan',alpha=0.6,zorder=4)
            #axL.              fill_between(X,(-mean-std),(-mean+std),color='cyan',alpha=0.6,zorder=4)

        j+=1

    #--plot JAM17
    #ax,axL = axs[1],axLs[1]

    #JAM17 = QPDCALC('JAM17_PPDF_nlo',ismc=True)
    #ppdf = JAM17.get_xpdf('ub-db',X,Q2=Q2) 

    #color = 'cyan'
    #alpha = 0.3
    #hand['JAM17'] = ax.fill_between(X,ppdf['xfmin'],ppdf['xfmax'],color=color,alpha=alpha,zorder=1)
    #axL.               fill_between(X,ppdf['xfmin'],ppdf['xfmax'],color=color,alpha=alpha,zorder=1)
    #ax .plot(X,ppdf['xfmax'],color=color,alpha=alpha,zorder=5)
    #axL.plot(X,ppdf['xfmax'],color=color,alpha=alpha,zorder=5)
    #ax .plot(X,ppdf['xfmin'],color=color,alpha=alpha,zorder=5)
    #axL.plot(X,ppdf['xfmin'],color=color,alpha=alpha,zorder=5)

    #--plot NNPDFpol1.1
    ax,axL = axs[2],axLs[2]

    JAM17 = QPDCALC('NNPDFpol11_100',ismc=True)
    ppdf = JAM17.get_xpdf('ub-db',X,Q2=Q2) 

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
    ax,axL = axs[2],axLs[2]

    dssv = DSSV()
    X=10**np.linspace(-4,-1,200)
    X=np.append(X,np.linspace(0.1,0.99,200))
    ub = dssv.xfxQ2(-2,X,Q2)
    db = dssv.xfxQ2(-1,X,Q2)
    ppdf = ub - db
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

    axs [1].tick_params(labelbottom=False)
    axLs[1].tick_params(labelbottom=False)

    axs [1].set_ylim(-0.015,0.075)
    axs [1] .set_yticks([0,0.02,0.04,0.06])
    axLs[1].set_yticklabels([r'$0$',r'$0.02$',r'$0.04$',r'$0.06$'])
   
    axs [2].set_ylim(-0.015,0.075)
    axs [2].set_yticks([0,0.02,0.04,0.06])
    axLs[2].set_yticklabels([r'$0$',r'$0.02$',r'$0.04$',r'$0.06$'])

    for i in range(N):
        axs [i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axLs[i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)

    axs [1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
    axLs[1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
    axs [2].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
    axLs[2].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)

    axLs[2].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[2].xaxis.set_label_coords(0.95,0.00)

    axs[1].text(0.07,0.83,r'\boldmath{$x (\Delta \bar{u} \!-\! \Delta \bar{d})$}', transform=axs[1].transAxes,size=40)

    axs[2].text(0.07,0.83,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[2].transAxes,size=25)
    #axs[2] .text(0.07,0.85,r'$2.5 < Q^2 < %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[2].transAxes,size=30)

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

    #legend1 = axLs[1].legend(handles,labels,loc=(-1.20,0.63),fontsize=24,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    legend1 = axLs[1].legend(handles,labels,loc='upper right',fontsize=24,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axLs[1].add_artist(legend1)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(hand['NNPDF'])
    handles.append(hand['DSSV'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{NNPDFpol1.1}')
    labels.append(r'\textrm{DSSV08}')
    axLs[2].legend(handles,labels,loc='upper right',fontsize=24,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.0, handler_map = {tuple:AnyObjectHandler()})

    

    py.tight_layout()
    py.subplots_adjust(hspace=0,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/asymmetry'%cwd
    filename+='.png'
    #filename+='.pdf'
    axs[1] .set_rasterized(True)
    axLs[1].set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)





def plot_models():

    wdir1 = 'results/star/final'
    wdir2 = 'results/star/pos2'

    WDIR = [wdir1,wdir2]

    nrows,ncols=1,1
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*12,nrows*7))
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

        ax,axL = axs[1],axLs[1]

        data=load('%s/data/ppdf-Q2=%3.5f.dat'%(wdir,Q2))

        X=data['X']

        flav = 'ub-db'
        mean = np.mean(data['XF'][flav],axis=0)
        std = np.std(data['XF'][flav],axis=0)

        #--plot average and standard deviation
        if j == 0:
            thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)
            axL.           fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=2)

        #--plot chiral soliton model at Q2 =4
        if j==0:
            data=load('%s/data/pdf-Q2=%3.5f.dat'%(wdir,4))
            X=data['X']

            flav = 'db-ub'
            mean = np.mean(data['XF'][flav],axis=0)
            std  = np.std(data['XF'][flav],axis=0)


            ax,axL = axs[1],axLs[1]
            color,alpha = 'blue', 0.7
            hand['CS']  = ax.fill_between(X,2.0*X**(0.12)*(mean-std),2.0*X**(0.12)*(mean+std),color=color,alpha=alpha,zorder=3)
            axL.             fill_between(X,2.0*X**(0.12)*(mean-std),2.0*X**(0.12)*(mean+std),color=color,alpha=alpha,zorder=3)
            #hand['CS'] ,= ax.plot(X,2.0*X**(0.12)*mean,color=color,alpha=alpha,zorder=3,ls='-',lw=2.1)
            #axL.             plot(X,2.0*X**(0.12)*mean,color=color,alpha=alpha,zorder=3,ls='-',lw=2.1)

        j+=1


    #--plot statistical model (at Q2=10)
    ax,axL = axs[1],axLs[1]
    X,stat,up,do = load_asym('stat')
    mean = (up+do)/2.0
    color = 'green'
    hand['stat']  = ax.fill_between(X,do,up,color=color,alpha=1.0,zorder=4)
    axL.               fill_between(X,do,up,color=color,alpha=1.0,zorder=4)
    #hand['stat'] ,= ax.plot(X,mean,color=color,alpha=1.0,zorder=4,ls='--',lw=2.1)
    #axL.               plot(X,mean,color=color,alpha=1.0,zorder=4,ls='--',lw=2.1)


    #--plot meson cloud model (at Q2 = 2.5)
    ax,axL = axs[1],axLs[1]
    X,MC = load_asym('MC')
    hand['MC'] ,= ax .plot(X,MC,color='magenta',ls='-.',alpha=1.0,zorder=3,lw=2.1)
    axL              .plot(X,MC,color='magenta',ls='-.',alpha=1.0,zorder=3,lw=2.1)


    #--plot bag/Pauli model (at Q2 = 4)
    #--need to evolve from 0.2 to 4
    #--set input scale to 0.2.  set parameters so that ub - db = 0.12*(1-x)**8
    
    #conf['Q20'] = 0.2
    #conf['aux'] = aux.AUX()
    #conf['order'] = 'NLO'
    #conf['alphaS'] = alphaS.ALPHAS()
    #conf['mellin'] = mellin.MELLIN()
    #ppdf = PPDF()

    #ppdf.params['uv1'][0] =  0.3
 
    #ppdf.params['ub1'][0] =  0.06
    #ppdf.params['ub1'][1] =  0.00
    #ppdf.params['ub1'][2] =  8.00
    #ppdf.params['ub1'][3] =  0.00
    #ppdf.params['ub1'][4] =  0.00

    #ppdf.params['db1'][0] = -0.06
    #ppdf.params['db1'][1] =  0.00
    #ppdf.params['db1'][2] =  8.00
    #ppdf.params['db1'][3] =  0.00
    #ppdf.params['db1'][4] =  0.00

    #ppdf.params['uv1'][0]  = 0.00
    #ppdf.params['dv1'][0]  = 0.00
    #ppdf.params['sea1'][0] = 0.00

    #ppdf.setup()

    #X=10**np.linspace(-6,-1,200)
    #X=np.append(X,np.linspace(0.1,0.99,200))
    #bag = []
    #for i in range(len(X)):
    #    ub = ppdf.get_xF(X[i],4,'ub')
    #    db = ppdf.get_xF(X[i],4,'db')
    #    bag.append(ub-db)

    #hand['bag'] ,= ax.plot(X,bag,color='darkblue',ls='--',alpha=1.0,zorder=5,lw=2.1)
    #axL              .plot(X,bag,color='darkblue',ls='--',alpha=1.0,zorder=5,lw=2.1)

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

    axs [1].set_ylim(-0.015,0.075)
    axs [1] .set_yticks([0,0.02,0.04,0.06])
    axLs[1].set_yticklabels([r'$0$',r'$0.02$',r'$0.04$',r'$0.06$'])
   
    for i in range(N):
        axs [i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)
        axLs[i+1].axhline(0  ,         color='k',linestyle='-' ,alpha=0.2,lw=1.0)

    axs [1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)
    axLs[1].axvline(0.1,ymax=1.0,color='k',linestyle=':' ,alpha=0.2,lw=1.0)

    axLs[1].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[1].xaxis.set_label_coords(0.95,0.00)

    axLs[1].text(0.00,0.88,r'\boldmath{$x (\Delta \bar{u} \!-\! \Delta \bar{d})$}', transform=axLs[1].transAxes,size=40)

    #axs[1].text(0.07,0.83,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axs[2].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(hand['stat'])
    handles.append(hand['CS'])
    handles.append(hand['MC'])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{\textbf{statistical}}')
    labels.append(r'\textrm{\textbf{chiral soliton}}')
    labels.append(r'\textrm{\textbf{meson cloud}}')

    legend1 = axs[1].legend(handles,labels,loc='upper left',fontsize=26,frameon=0,handletextpad=0.5,handlelength=0.86,ncol=2,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axs[1].add_artist(legend1)
    

    py.tight_layout()
    py.subplots_adjust(hspace=0,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/asymmetry-models'%cwd
    filename+='.png'
    #filename+='.pdf'
    axs[1] .set_rasterized(True)
    axLs[1].set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)






def plot_pol():

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

  filename+='.png'
  #filename+='.pdf'
  ax11.set_rasterized(True)
  ax12.set_rasterized(True)

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)







def load_lattice_spin(flav):

    F = open('plots/thesis/data/lat_%s.csv'%flav,'r')
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

def load_NNPDF_spin(Q2):

    NNPDF = lhapdf.mkPDFs('NNPDFpol11_100')
    L = len(NNPDF)
    X = np.linspace(0.01,0.98,100)
    X = 10**np.linspace(-2,np.log10(0.99),100)
    up = np.zeros((L,len(X)))
    dp = np.zeros((L,len(X)))
    ub = np.zeros((L,len(X)))
    db = np.zeros((L,len(X)))
    UP, DP, UB, DB = [],[],[],[]
    for i in range(L):
        for j in range(len(X)):
            up[i][j] = NNPDF[i].xfxQ2(-2,X[j],Q2)/X[j] + NNPDF[i].xfxQ2(2,X[j],Q2)/X[j]
            dp[i][j] = NNPDF[i].xfxQ2(-1,X[j],Q2)/X[j] + NNPDF[i].xfxQ2(1,X[j],Q2)/X[j]
            ub[i][j] = NNPDF[i].xfxQ2(-2,X[j],Q2)/X[j]
            db[i][j] = NNPDF[i].xfxQ2(-1,X[j],Q2)/X[j]
        
        moment_temp = cumtrapz(up[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        UP.append((moment_max - moment_temp)[0])

        moment_temp = cumtrapz(dp[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        DP.append((moment_max - moment_temp)[0])

        moment_temp = cumtrapz(ub[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        UB.append((moment_max - moment_temp)[0])

        moment_temp = cumtrapz(db[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        DB.append((moment_max - moment_temp)[0])

    UP, DP, UB, DB = np.array(UP), np.array(DP), np.array(UB), np.array(DB)

    return UP, DP, UB, DB

def load_DSSV_spin(Q2):

    dssv = DSSV()
    L = dssv.L
    X = 10**np.linspace(-2,np.log10(0.99),100)
    u  = dssv.xfxQ2(2 ,X,Q2)/X
    d  = dssv.xfxQ2(1 ,X,Q2)/X
    ub = dssv.xfxQ2(-2,X,Q2)/X
    db = dssv.xfxQ2(-1,X,Q2)/X
    up = u + ub
    dp = d + db
    UP = np.zeros(2*L+1)
    DP = np.zeros(2*L+1)
    UB = np.zeros(2*L+1)
    DB = np.zeros(2*L+1)
    for i in range(-L,L+1):
        moment_temp = cumtrapz(up[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        UP[i] = (moment_max - moment_temp)[0]

        moment_temp = cumtrapz(dp[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        DP[i] = (moment_max - moment_temp)[0]

        moment_temp = cumtrapz(ub[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        UB[i] = (moment_max - moment_temp)[0]

        moment_temp = cumtrapz(db[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        DB[i] = (moment_max - moment_temp)[0]

    UPstd, DPstd, UBstd, DBstd = 0,0,0,0
    for i in range(1,20):
        UPstd += (UP[i] - UP[-i])**2
        DPstd += (DP[i] - DP[-i])**2
        UBstd += (UB[i] - UB[-i])**2
        DBstd += (DB[i] - DB[-i])**2

    UPstd = np.sqrt(UPstd)/2.0
    DPstd = np.sqrt(DPstd)/2.0
    UBstd = np.sqrt(UBstd)/2.0
    DBstd = np.sqrt(DBstd)/2.0

    UP = np.mean(UP)
    DP = np.mean(DP)
    UB = np.mean(UB)
    DB = np.mean(DB)


    return UP, DP, UB, DB, UPstd, DPstd, UBstd, DBstd

def plot_spin():

    #--choose if truncated
    trunc = True

    #--without positivity
    wdir1 = 'results/star/final'
    wdir2 = 'results/star/noRHIC2'
    #--with positivity
    wdir3 = 'results/star/pos2'
    wdir4 = 'results/star/noRHICpos2'

    WDIR = [wdir1,wdir2,wdir3,wdir4]

    nrows,ncols=1,2
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
  
    thy  = {}
    hand = {}
    j = 0
    Q2 = 4
    for wdir in WDIR:
  
        load_config('%s/input.py'%wdir)
        istep=core.get_istep()
 
        if trunc==False: data =load('%s/data/ppdf-moment-%d-Q2=%d.dat'%(wdir,1,Q2))
        if trunc==True:  data =load('%s/data/ppdf-moment-trunc-%d-Q2=%d.dat'%(wdir,1,Q2))
  
        replicas=core.get_replicas(wdir)
        cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
        best_cluster=cluster_order[0]
  
        #flavs = ['u','d','ub','db','s+sb','g']
        flavs = ['up','dp','ub','db']
  
        mean1,std1,mean2,std2 = [],[],[],[]
        for flav in flavs:
            if trunc==True:
                if flav=='g':
                    momentp, momentn = [],[]
                    for i in range(len(data['moments'][flav])):
                        if data['moments'][flav][i][0] > 0: momentp.append(data['moments'][flav][i][0])
                        if data['moments'][flav][i][0] < 0: momentn.append(data['moments'][flav][i][0])
                    mean1.append(np.mean(momentp))
                    std1 .append(np.std (momentp))
                    mean1.append(np.mean(momentn))
                    std1 .append(np.std (momentn))
  
                elif flav=='up':
                    moment = []
                    for i in range(len(data['moments']['u'])):
                        moment.append(data['moments']['u'][i][0]+data['moments']['ub'][i][0])
  
                    mean1.append(np.mean(moment))
                    std1 .append(np.std (moment))

                elif flav=='dp':
                    moment = []
                    for i in range(len(data['moments']['d'])):
                        moment.append(data['moments']['d'][i][0]+data['moments']['db'][i][0])
  
                    mean1.append(-np.mean(moment))
                    std1 .append( np.std (moment))

                elif flav=='ub':
                    moment = []
                    for i in range(len(data['moments'][flav])):
                        moment.append(data['moments'][flav][i][0])
  
                    mean2.append(np.mean(moment))
                    std2 .append(np.std (moment))

                elif flav=='db':
                    moment = []
                    for i in range(len(data['moments'][flav])):
                        moment.append(data['moments'][flav][i][0])
  
                    mean2.append(-np.mean(moment))
                    std2 .append( np.std (moment))


            else:

                    mean2.append(np.mean(data['moments'][flav],axis=0))
                    std2 .append(np.std (data['moments'][flav],axis=0))
          
  
        X = np.array([0,1])

        #print(mean1,mean2)
        #print(std1,std2)
 
        colors1 = ['red','dodgerblue','darkviolet','green']
        colors2 = ['yellow','cyan','violet','limegreen']
        if j==0: left,right,hatch,zorder,color,alpha =  -0.18, 0.18, None, 2, 'red'    ,1.0
        if j==1: left,right,hatch,zorder,color,alpha =  -0.21, 0.21, None, 1, 'cyan' ,0.3
        if j==2: left,right,hatch,zorder,color,alpha =  -0.04, 0.04, None, 3, 'black'  ,1.0
        if j==3: left,right,hatch,zorder,color,alpha =  -0.05, 0.05, '///', 3, 'black'  ,1.0

        for i in range(len(X)):
            up1 = mean1[i] + std1[i]
            do1 = mean1[i] - std1[i]
            up2 = mean2[i] + std2[i]
            do2 = mean2[i] - std2[i]
            if j in [0,1]:
                hand[j] = ax11.fill_between([X[i]+left,X[i]+right],[do1,do1],[up1,up1],hatch=hatch,facecolor=color,edgecolor=color,zorder=zorder,alpha=alpha)
                hand[j] = ax12.fill_between([X[i]+left,X[i]+right],[do2,do2],[up2,up2],hatch=hatch,facecolor=color,edgecolor=color,zorder=zorder,alpha=alpha)
            if j in [2]:
                hand[j] = ax11.fill_between([X[i]+left,X[i]+right],[do1,do1],[up1,up1],hatch=hatch,facecolor='black',edgecolor='black',zorder=zorder,alpha=0.9)
                hand[j] = ax12.fill_between([X[i]+left,X[i]+right],[do2,do2],[up2,up2],hatch=hatch,facecolor='black',edgecolor='black',zorder=zorder,alpha=0.9)
            if j in [3]:
                hand[j] = ax11.fill_between([X[i]+left,X[i]+right],[do1,do1],[up1,up1],hatch=hatch,facecolor='none',edgecolor='black',zorder=zorder,alpha=0.5,lw=1)
                hand[j] = ax12.fill_between([X[i]+left,X[i]+right],[do2,do2],[up2,up2],hatch=hatch,facecolor='none',edgecolor='black',zorder=zorder,alpha=0.5,lw=1)
            if j in [4]:
                ax11.errorbar(X[i]+0.10,mean1[i],yerr=std1[i],capsize=4.0,color='darkblue',alpha=1.0,ms=0.0,marker='o',zorder=5)
                ax12.errorbar(X[i]+0.10,mean2[i],yerr=std2[i],capsize=4.0,color='darkblue',alpha=1.0,ms=0.0,marker='o',zorder=5)

        j+=1
  
 
   
    #--plot NNPDF
    up, dp, ub, db = load_NNPDF_spin(Q2)

    l = 0.25
    color = 'darkgreen'
    #NNPDF  = ax11.errorbar([0.0+l],np.mean(up),yerr=np.std(up),capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    #ax11         .errorbar([1.0+l],np.mean(-dp),yerr=np.std(dp),capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    NNPDF  = ax12.errorbar([0.0+l],np.mean(ub),yerr=np.std(ub),capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    ax12         .errorbar([1.0+l],np.mean(-db),yerr=np.std(db),capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)

    #--plot DSSV
    up, dp, ub, db, upstd, dpstd, ubstd, dbstd = load_DSSV_spin(Q2)

    l = 0.3
    color = 'darkblue'
    #DSSV  = ax11.errorbar([0.0+l],up,upstd,capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    #ax11        .errorbar([1.0+l],-dp,dpstd,capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    DSSV  = ax12.errorbar([0.0+l],ub,ubstd,capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    ax12        .errorbar([1.0+l],-db,dbstd,capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)


 
    #--plot lattice data
    up =  0.432*2
    dp = -0.213*2
    uperr = 2*0.008
    dperr = 2*0.008

    l = 0.3
    color1, color2 = 'darkgreen','limegreen'
    #lat1  = ax11.errorbar([0.0+l],up,yerr=uperr,capsize=4.0,color=color1,alpha=1.0,ms=0.0,marker='o',zorder=2)
    #ax11        .errorbar([1.0+l],dp,yerr=dperr,capsize=4.0,color=color1,alpha=1.0,ms=0.0,marker='o',zorder=2)
    #lat2 ,= ax11.plot(    [0.0+l],up,                       color=color2,alpha=1.0,ms=15,marker='o',zorder=1)
    #ax11        .plot(    [1.0+l],dp,                       color=color2,alpha=1.0,ms=15,marker='o',zorder=1)


    for flav in ['ub','db']:
        X, do, up = load_lattice(flav)
        idx = np.nonzero(X >= 0.01)
        X, do, up = X[idx], do[idx], up[idx]

        moment_temp = cumtrapz(do, X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        do = (moment_max - moment_temp)[0]

        moment_temp = cumtrapz(up, X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        up = (moment_max - moment_temp)[0]

        mean = (do+up)/2.0
        std  = np.abs(up-do)/2.0

        if flav=='ub': x = 2.0
        if flav=='db': x = 3.0
        #ax11.errorbar([x+l],mean,yerr=std,capsize=4.0,color=color1 ,alpha=1.0,ms=0.0,marker='o',zorder=2)
        #ax11.plot(    [x+l],mean,                     color=color2 ,alpha=1.0,ms=15 ,marker='o',zorder=1)
    
    #--point for legend
    #ax11.errorbar(0.90,0.57,yerr=std,capsize=4.0,color=color1 ,alpha=1.0,ms=0.0,marker='o',zorder=2)
    #ax11.plot(    0.90,0.57,                     color=color2 ,alpha=1.0,ms=15 ,marker='o',zorder=1)


    for ax in [ax11,ax12]: 
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=5,pad=10)
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=2,pad=10)
 
    ax12.tick_params(labelleft=False,labelright=True)
 
    ax11.set_xlim(-0.4,1.4)
    ax11.set_xticks([0,1])
    ax11.set_xticklabels([r'\boldmath$\Delta u^+$',r'\boldmath$-\Delta d^+$'])

    ax12.set_xlim(-0.4,1.4)
    ax12.set_xticks([0,1])
    ax12.set_xticklabels([r'\boldmath$\Delta \bar{u}$',r'\boldmath$-\Delta \bar{d}$'])

    ax11.set_ylim(0.25,0.85)   
    ax11.set_yticks([0.4,0.6,0.8])
    minorLocator = MultipleLocator(0.10)
    ax11.yaxis.set_minor_locator(minorLocator)
  
    ax12.set_ylim(-0.022,0.105)   
    ax12.set_yticks([-0.02,0,0.02,0.04,0.06,0.08,0.10])
    ax12.set_yticklabels([r'',r'$0$',r'$0.02$',r'$0.04$',r'$0.06$',r'$0.08$',r''])
    minorLocator = MultipleLocator(0.01)
    ax12.yaxis.set_minor_locator(minorLocator)

    
    l = 0.26
    ax11.axhline(0,0,1,color='black',linestyle='-' ,alpha=0.2, lw=1.0)
    ax12.axhline(0,0,1,color='black',linestyle='-' ,alpha=0.2, lw=1.0)
    #ax11.plot([-1.0 ,2.0-l] ,[0,0],color='black',linestyle='-',alpha=0.2,lw=1.0)
    #ax11.plot([2.0+l,3.0-l] ,[0,0],color='black',linestyle='-',alpha=0.2,lw=1.0)
    #ax11.plot([3.0+l,4.0]   ,[0,0],color='black',linestyle='-',alpha=0.2,lw=1.0)
  
    if trunc==True:  ax11.text(0.45,0.75,r'\boldmath$\int_{0.01}^{1} {\rm d} x \Delta q$',     transform=ax11.transAxes,size=40)
    if trunc==False: ax11.text(0.45,0.75,r'\boldmath$\int_{0}^{1} {\rm d} x \Delta q$',        transform=ax11.transAxes,size=40)
  
    #ax11.text(0.03,0.43,r'$Q^2 = %d$~'%Q2 + r'\textrm{GeV}'+r'$^2$', transform=ax11.transAxes,size=25)
    #ax11.text(0.52,0.14,r'\textrm{\textbf{light: no RHIC \boldmath$W/Z$}}',   transform=ax11.transAxes,size=20)
    #ax11.text(0.52,0.05,r'\textrm{\textbf{dark:  with RHIC \boldmath$W/Z$}}', transform=ax11.transAxes,size=20)

    pos1 ,= ax11.plot(0,0,color='black',lw=10,alpha=0.9) 
    pos2  = ax11.fill_between([0],[0],[0],hatch=None,facecolor='none',edgecolor='black',zorder=zorder,alpha=0.7)

    blank ,= ax11.plot(0,0,color='white',alpha=0.0)

    handles,labels = [],[]
    handles.append(hand[1])
    handles.append(hand[0])
    handles.append(hand[3])
    handles.append(hand[2])
    labels.append(r'\textrm{\textbf{no \boldmath$W$ \hspace{0.20cm} +pos}}') 
    labels.append(r'\textrm{\textbf{JAM \hspace{0.45cm} +pos}}') 
    labels.append(r'') 
    labels.append(r'') 
    ax11.legend(handles,labels,loc=(0.01,0.05),fontsize=25,frameon=0,handletextpad=0.5,handlelength=1.0,ncol=2,columnspacing=0.35)

    handles,labels = [],[]
    handles.append(NNPDF)
    handles.append(DSSV)
    labels.append(r'\textrm{NNPDFpol1.1}')
    labels.append(r'\textrm{DSSV08}')
    ax12.legend(handles,labels,loc=(0.01,0.70),fontsize=25,frameon=0,handletextpad=0.5,handlelength=1.0,ncol=1,columnspacing=0.35)

  
    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/spin'%cwd
    filename+='.png'
    #filename+='.pdf'
    ax11 .set_rasterized(True)
    ax12 .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)






if __name__ == "__main__":

    #plot_asym()
    #plot_pol()
    #plot_spin() 

    plot_models()





 
  
        
        
        
        
        
        
        
        
        
        
        
        

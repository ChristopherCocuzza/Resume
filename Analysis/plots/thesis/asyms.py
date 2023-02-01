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

from tools.config import conf
from qcdlib import aux, alphaS, mellin

cwd = 'plots/thesis'

if __name__ == "__main__":

    wdir1 = 'results/star/final'

    WDIR = [wdir1]

    nrows,ncols=1,1
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
        thy_band0 = ax.fill_between(X,(mean-std),(mean+std),color='blue',alpha=1.0,zorder=2)
        axL.           fill_between(X,(mean-std),(mean+std),color='blue',alpha=1.0,zorder=2)

        #--plot unpolarized asymmetry
        data=load('%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2))
        X=data['X']

        flav = 'db-ub'
        mean = np.mean(data['XF'][flav],axis=0)
        std  = np.std(data['XF'][flav],axis=0)

        hand['upol'] = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=0.9,zorder=4)
        axL.              fill_between(X,(mean-std),(mean+std),color='red',alpha=0.9,zorder=4)

        j+=1


    #--plot NNPDFpol1.1
    #ax,axL = axs[2],axLs[2]

    #JAM17 = QPDCALC('NNPDFpol11_100',ismc=True)
    #ppdf = JAM17.get_xpdf('ub-db',X,Q2=Q2) 

    #color = 'lightsteelblue'
    #ec    = 'gray'
    #alpha = 0.6
    #hand['NNPDF'] = ax.fill_between(X,ppdf['xfmin'],ppdf['xfmax'],fc=color,alpha=alpha,zorder=1,ec='lightgray',lw=1.5)
    #axL.               fill_between(X,ppdf['xfmin'],ppdf['xfmax'],fc=color,alpha=alpha,zorder=1,ec='lightgray',lw=1.5)
    #ax .plot(X,ppdf['xfmax'],color=ec,alpha=0.4,zorder=5,lw=2.0)
    #axL.plot(X,ppdf['xfmax'],color=ec,alpha=0.4,zorder=5,lw=2.0)
    #ax .plot(X,ppdf['xfmin'],color=ec,alpha=0.4,zorder=5,lw=2.0)
    #axL.plot(X,ppdf['xfmin'],color=ec,alpha=0.4,zorder=5,lw=2.0)

    #--plot DSSV08
    #ax,axL = axs[2],axLs[2]

    #dssv = DSSV()
    #X=10**np.linspace(-4,-1,200)
    #X=np.append(X,np.linspace(0.1,0.99,200))
    #ub = dssv.xfxQ2(-2,X,Q2)
    #db = dssv.xfxQ2(-1,X,Q2)
    #ppdf = ub - db
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

    axLs[1].text(0.50,0.83,r'\textbf{\textrm{JAM}}', transform=axLs[1].transAxes,size=40)
    axLs[1].text(0.40,0.70,r'$Q^2 = %d$~'%Q2  + r'\textrm{GeV}'+r'$^2$', transform=axLs[1].transAxes,size=25)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    blank ,= axs[1].plot(0,0,color='white',alpha=0.0)

    handles, labels = [],[]
    handles.append(thy_band0)
    handles.append(hand['upol'])

    labels.append(r'\boldmath$x(\Delta \bar{u} - \Delta \bar{d})$')
    labels.append(r'\boldmath$x(\bar{d} - \bar{u})$')

    legend1 = axs[1].legend(handles,labels,loc='upper left',fontsize=30,frameon=0,handletextpad=0.5,handlelength=0.9,ncol=1,columnspacing=1.5, handler_map = {tuple:AnyObjectHandler()})
    axs[1].add_artist(legend1)

    py.tight_layout()
    py.subplots_adjust(hspace=0,top=0.99,right=0.99,left=0.10)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/asymmetries'%cwd
    #filename+='.png'
    filename+='.pdf'
    axs[1] .set_rasterized(True)
    axLs[1].set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)
 





 
  
        
        
        
        
        
        
        
        
        
        
        
        

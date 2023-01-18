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

        #--plot chiral soliton model at Q2 =4
        #if j==0:
        #    data=load('%s/data/pdf-%d-Q2=4.dat'%(wdir,istep))
        #    X=data['X']

        #    flav = 'db-ub'
        #    mean = np.mean(data['XF'][flav],axis=0)
        #    std  = np.std(data['XF'][flav],axis=0)


        #    ax,axL = axs[1],axLs[1]
        #    color,alpha = 'lightgray', 0.9
        #    #hand['CS']  = ax.fill_between(X,2.0*X**(0.12)*(mean-std),2.0*X**(0.12)*(mean+std),color=color,alpha=alpha,zorder=3)
        #    #axL.             fill_between(X,2.0*X**(0.12)*(mean-std),2.0*X**(0.12)*(mean+std),color=color,alpha=alpha,zorder=3)
        #    #hand['CS'] ,= ax.plot(X,2.0*X**(0.12)*mean,color=color,alpha=alpha,zorder=3,ls='-',lw=2.1)
        #    #axL.             plot(X,2.0*X**(0.12)*mean,color=color,alpha=alpha,zorder=3,ls='-',lw=2.1)

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
   
    #--plot DSSV19 (currently taken from plot and not from grids) 
    #ax,axL = axs[2],axLs[2]

    #X=10**np.linspace(-3,-1,200)
    #X=np.append(X,np.linspace(0.1,0.99,200))
    #x, ub_up, ub_do = load_DSSV('ub')

    #ub_up = interp1d(x, ub_up, kind='cubic')
    #ub_do = interp1d(x, ub_do, kind='cubic')

    #ub_up = np.array([ub_up(x) for x in X])
    #ub_do = np.array([ub_do(x) for x in X])

    #x, db_up, db_do = load_DSSV('db')

    #db_up = interp1d(x, db_up, kind='cubic')
    #db_do = interp1d(x, db_do, kind='cubic')

    #db_up = np.array([db_up(x) for x in X])
    #db_do = np.array([db_do(x) for x in X])

    #ub_mean = (ub_up + ub_do)/2.0
    #db_mean = (db_up + db_do)/2.0
    #ub_std  = (ub_up - ub_do)/2.0
    #db_std  = (db_up - db_do)/2.0
    #mean = ub_mean - db_mean
    #std  = np.sqrt(ub_std**2 + db_std**2)    

    #color = 'blue'
    #alpha = 0.4
    #hand['DSSV'] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=alpha,zorder=6)
    #axL.              fill_between(X,mean-std,mean+std,color=color,alpha=alpha,zorder=6)


    #--plot statistical model (at Q2=10)
    ax,axL = axs[1],axLs[1]
    X,stat,up,do = load_asym('stat')
    mean = (up+do)/2.0
    #hand['stat']  = ax.fill_between(X,do,up,color='lightgreen',alpha=1.0,zorder=4)
    #axL.               fill_between(X,do,up,color='lightgreen',alpha=1.0,zorder=4)
    #hand['stat'] ,= ax.plot(X,mean,color='lightgreen',alpha=1.0,zorder=4,ls='--',lw=2.1)
    #axL.               plot(X,mean,color='lightgreen',alpha=1.0,zorder=4,ls='--',lw=2.1)


    #--plot meson cloud model (at Q2 = 2.5)
    ax,axL = axs[1],axLs[1]
    X,MC = load_asym('MC')
    #hand['MC'] ,= ax .plot(X,MC,color='magenta',ls='-.',alpha=1.0,zorder=3,lw=2.1)
    #axL              .plot(X,MC,color='magenta',ls='-.',alpha=1.0,zorder=3,lw=2.1)


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
    #filename+='.png'
    filename+='.pdf'
    axs[1] .set_rasterized(True)
    axLs[1].set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)
 





 
  
        
        
        
        
        
        
        
        
        
        
        
        

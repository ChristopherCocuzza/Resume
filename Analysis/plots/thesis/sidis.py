#!/usr/bin/env python
import sys, os
import numpy as np
import copy
import pandas as pd
import scipy as sp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

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
from matplotlib.ticker import MultipleLocator

from scipy.stats import norm


#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from tools
from tools.tools import load,save,checkdir
from tools.config import conf, load_config


import kmeanconf as kc

cwd = 'plots/thesis'

def plot_pion(wdir,kc):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if 1005 not in predictions['reactions']['sidis']: return 
    if 1006 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:  data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    for xrange in X:
        xmin,xmax=xrange
        DP=pd.DataFrame(data[1005]).query('X>%f and X<%f'%(xmin,xmax))
        DM=pd.DataFrame(data[1006]).query('X>%f and X<%f'%(xmin,xmax))

        if len(DP.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            for D in [DP,DM]:
                d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
                if len(d)<=1: continue
                f=2**alpha
                #print(ymin)
                if ymin==0.1:  color='r'
                elif ymin==0.15: color='darkorange'
                elif ymin==0.2: color='g'
                elif ymin==0.3: color='c'
                elif ymin==0.5: color='m'
                else: color='k'

                #ax.fill_between(d.Z,(d.thy-d.dthy)*f, (d.thy+d.dthy)*f,color=color)
                if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy*f,color=color,ls='--')
                if '-' in d['hadron'].values[0]: ax.plot(d.Z,d.thy*f,color=color,ls=':')

                p=ax.errorbar(d.Z,d.value*f,d.alpha*f,fmt='.',color=color)
                H[ymin]=p

        ax.set_ylim(5e-2,50.5)
        ax.set_xlim(0.15,0.85)
        ax.semilogy()
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.tick_params(axis='both', which='both', top=True, right=True,direction='in',labelsize=20)
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticks([0.2,0.4,0.6,0.8])
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            ax.set_ylabel(r'\boldmath$\mathrm{d} M_{h}/\mathrm{d} z_h\times 2^i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        if cnt==1 or cnt==5:
            ax.set_yticks([0.1,1,10])
            ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.3,0.9,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:
            from matplotlib.lines import Line2D
            h = [Line2D([0], [0], color='k', lw=5,ls='--'),
                Line2D([0], [0], color='k', lw=5,ls=':')]
            l = [r'\boldmath{$\pi^+$}',r'\boldmath{$\pi^-$}']
            ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)
        #Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]

    H=[H[_]  for _ in [0.1,0.15,0.2,0.3,0.5]]
    L=[ r'$0.1~<y<0.15~(i=0)$'
       ,r'$0.15<y<0.2~~(i=1)$'
       ,r'$0.2~<y<0.3~~(i=2)$'
       ,r'$0.3~<y<0.5~~(i=3)$'
       ,r'$0.5~<y<0.7~~(i=4)$'
      ]
    ax1.legend(H,L,loc=3,fontsize=20,frameon=0,handletextpad=0)


    py.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.98, wspace=0, hspace=0)
    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/SIDIS-pion.png'%cwd
    py.savefig(filename)
    print('Saving SIDIS pion figure to %s'%filename)

def plot_kaon(wdir,kc):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if 2005 not in predictions['reactions']['sidis']: return 
    if 2006 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:  data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])


    # kaon
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    for xrange in X:
        xmin,xmax=xrange
        DP=pd.DataFrame(data[2005]).query('X>%f and X<%f'%(xmin,xmax))
        DM=pd.DataFrame(data[2006]).query('X>%f and X<%f'%(xmin,xmax))

        if len(DP.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            for D in [DP,DM]:
                d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
                if len(d)<=1: continue
                f=2**alpha
                #print(ymin)
                if ymin==0.1:  color='r'
                elif ymin==0.15: color='darkorange'
                elif ymin==0.2: color='g'
                elif ymin==0.3: color='c'
                elif ymin==0.5: color='m'
                else: color='k'

                #ax.fill_between(d.Z,(d.thy-d.dthy)*f, (d.thy+d.dthy)*f,color=color)
                if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy*f,color=color,ls='--')
                if '-' in d['hadron'].values[0]: ax.plot(d.Z,d.thy*f,color=color,ls=':')

                p=ax.errorbar(d.Z,d.value*f,d.alpha*f,fmt='.',color=color)
                H[ymin]=p

        ax.set_ylim(5e-3,20)
        ax.set_xlim(0.15,0.85)
        ax.semilogy()
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.tick_params(axis='both', which='both', top=True, right=True,direction='in',labelsize=20)
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticks([0.2,0.4,0.6,0.8])
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            ax.set_ylabel(r'\boldmath$\mathrm{d} M_{h}/\mathrm{d} z_h\times 2^i$',size=30)
            ax.yaxis.set_label_coords(-0.20, -0.05)

        if cnt==1 or cnt==5:
            ax.set_yticks([0.01,0.1,1,10])
            ax.set_yticklabels([r'$0.01$',r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.3,0.9,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:
            from matplotlib.lines import Line2D
            h = [Line2D([0], [0], color='k', lw=5,ls='--'),
                Line2D([0], [0], color='k', lw=5,ls=':')]
            l = [r'\boldmath{$K^+$}',r'\boldmath{$K^-$}']
            ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)
        #Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]

    H=[H[_]  for _ in [0.1,0.15,0.2,0.3,0.5]]
    L=[ r'$0.1~<y<0.15~(i=0)$'
       ,r'$0.15<y<0.2~~(i=1)$'
       ,r'$0.2~<y<0.3~~(i=2)$'
       ,r'$0.3~<y<0.5~~(i=3)$'
       ,r'$0.5~<y<0.7~~(i=4)$'
      ]
    ax1.legend(H,L,loc=3,fontsize=20,frameon=0,handletextpad=0)


    py.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.98, wspace=0, hspace=0)
    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/SIDIS-kaon.png'%cwd
    py.savefig(filename)
    print('Saving SIDIS kaon figure to %s'%filename)

def plot_hadron(wdir,kc):
   
    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
 
    predictions=load('%s/data/predictions-%d.dat'%(wdir,istep))
    labels  = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster = labels['cluster']
    if 'sidis' not in predictions['reactions']: return
    if 3000 not in predictions['reactions']['sidis']: return 
    if 3001 not in predictions['reactions']['sidis']: return 
    data=predictions['reactions']['sidis']

    for _ in data:
        data[_]['thy'] =np.mean(data[_]['prediction-rep'], axis=0)
        data[_]['dthy']=np.std(data[_]['prediction-rep'], axis=0)
        del data[_]['prediction-rep']
        del data[_]['residuals-rep']
        del data[_]['shift-rep']
        del data[_]['rres-rep']
        if 'X' not in data[_]:  data[_]['X']=0.5*(data[_]['xmin']+data[_]['xmax'])
        if 'Z' not in data[_]:  data[_]['Z']=0.5*(data[_]['zmin']+data[_]['zmax'])
        if 'Y' not in data[_]:  data[_]['Y']=0.5*(data[_]['ymin']+data[_]['ymax'])


    # pi
    X=[[0.004,0.01],[0.01,0.02],[0.02,0.03],[0.03,0.04],[0.04,0.06],[0.06,0.1]
      ,[0.1,0.14],[0.14,0.18],[0.18,0.4]]

    Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]
    nrows,ncols=2,4
    fig = py.figure(figsize=(ncols*4,nrows*5))

    cnt=0
    H={}
    for xrange in X:
        xmin,xmax=xrange
        DP=pd.DataFrame(data[3000]).query('X>%f and X<%f'%(xmin,xmax))
        DM=pd.DataFrame(data[3001]).query('X>%f and X<%f'%(xmin,xmax))

        if len(DP.values)==0: continue
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        alpha=-1
        for yrange in Y:
            alpha+=1
            ymin,ymax=yrange
            for D in [DP,DM]:
                d=D.query('Y>%f  and Y<%f'%(ymin,ymax))
                if len(d)<=1: continue
                f=2**alpha
                #print(ymin)
                if ymin==0.1:  color='r'
                elif ymin==0.15: color='darkorange'
                elif ymin==0.2: color='g'
                elif ymin==0.3: color='c'
                elif ymin==0.5: color='m'
                else: color='k'

                #ax.fill_between(d.Z,(d.thy-d.dthy)*f, (d.thy+d.dthy)*f,color=color)
                if '+' in d['hadron'].values[0]: ax.plot(d.Z,d.thy*f,color=color,ls='--')
                if '-' in d['hadron'].values[0]: ax.plot(d.Z,d.thy*f,color=color,ls=':')

                p=ax.errorbar(d.Z,d.value*f,d.alpha*f,fmt='.',color=color)
                H[ymin]=p

        ax.set_ylim(5e-2,50.5)
        ax.set_xlim(0.15,0.85)
        ax.semilogy()
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.tick_params(axis='both', which='both', top=True, right=True,direction='in',labelsize=20)
        if any([cnt==_ for _ in[2,3,4,6,7,8]]):
            ax.set_yticklabels([])
        if any([cnt==_ for _ in[1,2,3,4]]):
            ax.set_xticklabels([])

        if cnt==8: 
            ax.set_xlabel(r'\boldmath$z_h$',size=30)
            ax.xaxis.set_label_coords(0.95, -0.01)
            ax.set_xticks([0.2,0.4,0.6,0.8])
            ax.set_xticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r''])

        if cnt==1:
            ax.set_ylabel(r'\boldmath$\mathrm{d} M_{h}/\mathrm{d} z_h\times 2^i$',size=30)
            ax.yaxis.set_label_coords(-0.15, -0.05)

        if cnt==1 or cnt==5:
            ax.set_yticks([0.1,1,10])
            ax.set_yticklabels([r'$0.1$',r'$1$',r'$10$'])


        if cnt==1:msg=r'\boldmath{$x_{\rm Bj}\in[0.01,0.02]$}'
        if cnt==2:msg=r'\boldmath{$x_{\rm Bj}\in[0.02,0.03]$}'
        if cnt==3:msg=r'\boldmath{$x_{\rm Bj}\in[0.03,0.04]$}'
        if cnt==4:msg=r'\boldmath{$x_{\rm Bj}\in[0.04,0.06]$}'
        if cnt==5:msg=r'\boldmath{$x_{\rm Bj}\in[0.06,0.10]$}'
        if cnt==6:msg=r'\boldmath{$x_{\rm Bj}\in[0.10,0.14]$}'
        if cnt==7:msg=r'\boldmath{$x_{\rm Bj}\in[0.14,0.18]$}'
        if cnt==8:msg=r'\boldmath{$x_{\rm Bj}\in[0.18,0.40]$}'


        ax.text(0.3,0.9,msg,transform=ax.transAxes,size=20)
        if cnt==1:ax1=ax
        if cnt==2:
            from matplotlib.lines import Line2D
            h = [Line2D([0], [0], color='k', lw=5,ls='--'),
                Line2D([0], [0], color='k', lw=5,ls=':')]
            l = [r'\boldmath{$h^+$}',r'\boldmath{$h^-$}']
            ax.legend(h,l,fontsize=30,loc=3,frameon=0,handlelength=2.5,handletextpad=0.4)
        #Y=[[0.1,0.15],[0.15,0.2],[0.2,0.3],[0.3,0.5],[0.5,0.7]]

    H=[H[_]  for _ in [0.1,0.15,0.2,0.3,0.5]]
    L=[ r'$0.1~<y<0.15~(i=0)$'
       ,r'$0.15<y<0.2~~(i=1)$'
       ,r'$0.2~<y<0.3~~(i=2)$'
       ,r'$0.3~<y<0.5~~(i=3)$'
       ,r'$0.5~<y<0.7~~(i=4)$'
      ]
    ax1.legend(H,L,loc=3,fontsize=20,frameon=0,handletextpad=0)


    py.subplots_adjust(left=0.08, bottom=0.05, right=0.98, top=0.98, wspace=0, hspace=0)
    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/SIDIS-hadron.png'%cwd
    py.savefig(filename)
    print('Saving SIDIS hadron figure to %s'%filename)


if __name__ == "__main__":

    wdir = 'results/star/final'

    plot_pion  (wdir, kc)
    plot_kaon  (wdir, kc)
    plot_hadron(wdir, kc)






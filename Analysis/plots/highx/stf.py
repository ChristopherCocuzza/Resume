#!/usr/bin/env python
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from tools.config import load_config,conf
from fitlib.resman import RESMAN
import numpy as np

#--matplotlib
import matplotlib
import pylab  as py
from matplotlib.ticker import MultipleLocator

#--from tools
from tools.tools import load,lprint,save

#--from corelib
from analysis.corelib import core,classifier

import kmeanconf as kc

cwd = 'plots/highx'

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth

def load_F2nF2p():

  #--load AKP17 and CJ15 data
  F = open('plots/marathon/data/F2nF2p.csv','r')
  L = F.readlines()
  F.close()

  L = [l.strip() for l in L]
  L = [[x for x in l.split()] for l in L]
  L = np.transpose(L)[0]

  X, CJ, AKP = [],[],[]
  for i in range(len(L)):
      if i==0: continue
      X  .append(float(L[i].split(',')[0]))
      AKP.append(float(L[i].split(',')[1]))
      CJ .append(float(L[i].split(',')[2]))

  X,CJ,AKP = np.array(X),np.array(CJ),np.array(AKP)

  x = np.linspace(0.195,0.825,50)

  CJ = smooth(CJ,10)
  AKP = smooth(AKP,10)

  return x,CJ,AKP

def load_F2DF2N(group):

  if group=='AKP17':
      F = open('plots/marathon/data/F2DF2N/AKP.csv','r')
      L = F.readlines()
      F.close()
      
      L = [l.strip() for l in L]
      L = [[x for x in l.split()] for l in L]
      L = np.transpose(L)[0]

      X,upper,lower = [],[],[]

      for i in range(len(L)):
          if i==0: continue
          X    .append(float(L[i].split(',')[0]))
          upper.append(float(L[i].split(',')[1]))
          lower.append(float(L[i].split(',')[2]))

      X,upper,lower=np.array(X),np.array(upper),np.array(lower)
 
  if group=='CJ15':
      filename = 'plots/marathon/data/F2DF2N/CJ15.out'

      F = open(filename,'r')
      L = F.readlines()
      F.close()
      
      L = [l.strip() for l in L]
      L = [[x for x in l.split()] for l in L]
      L = np.transpose(L)
      
      X,thy,err = [],[],[] 
      for i in range(len(L)):
          if len(L[i]) < 9: continue
          if L[i][8]!='test_DN': continue
          if L[i][1] != '10.000': continue
          X  .append(float(L[i][0]))
          thy.append(float(L[i][3]))
          err.append(float(L[i][4]))

      X,thy,err = np.array(X),np.array(thy),np.array(err)

      lower = thy - err
      upper = thy + err

      lower = smooth(lower,10)
      upper = smooth(upper,10)

  return X,lower,upper

def plot_EMC():

    wdir1 = 'results/highx/W2cuts/W2cut35'
    wdir2 = 'results/highx/wfs/AV18'
    wdir3 = 'results/highx/wfs/CDBonn'
    wdir4 = 'results/highx/wfs/WJC1'
    wdir5 = 'results/highx/wfs/WJC2'
    wdir6 = 'results/highx/HT/add'
    wdir7 = 'results/highx/HT/iso'
    wdir8 = 'results/highx/HT/addiso'
    wdir9 = 'results/highx/GP/W2cut35'
    #wdir8 = 'noSLAC'
    #wdir9 = 'noJLab'

    WDIR = [wdir1,wdir2,wdir3,wdir4,wdir5,wdir6,wdir7,wdir8,wdir9]

    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax21 = py.subplot(nrows,ncols,3)
    ax22 = py.subplot(nrows,ncols,4)

    j = 0
    hand = {}
    thy  = {}

    Q2 = 10

    UP = []
    DO = []

    for wdir in WDIR:    

        load_config('%s/input.py'%wdir)

        D = load('%s/data/stf-d-F2-Q2=%3.5f.dat'%(wdir,Q2))
        p = load('%s/data/stf-p-F2-Q2=%3.5f.dat'%(wdir,Q2))
        n = load('%s/data/stf-n-F2-Q2=%3.5f.dat'%(wdir,Q2))

        ##############################################
        #--plot EMC ratio
        ##############################################
        X = D['X']

        rat = 2*np.array(D['XF'])/(np.array(p['XF']) + np.array(n['XF']))
        mean = np.mean(np.array(rat),axis=0)
        std  = np.std (np.array(rat),axis=0)
        if j==0: 
            hand[j] = ax11.fill_between(X,mean-std,mean+std,color='red'    ,alpha=1.0,zorder=1)
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='red'    ,alpha=1.0,zorder=5)
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='red'    ,alpha=1.0,zorder=3)
            hand[j] = ax22.fill_between(X,mean-std,mean+std,color='red'    ,alpha=1.0,zorder=2)
        if j==1: 
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='green'  ,alpha=1.0,zorder=4)
        if j==2: 
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='blue'   ,alpha=1.0,zorder=1)
        if j==3: 
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='magenta',alpha=1.0,zorder=2)
        if j==4: 
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='orange' ,alpha=1.0,zorder=3)
        if j==5: 
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='blue'   ,alpha=1.0,zorder=1)
        if j==6: 
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='green'  ,alpha=1.0,zorder=2)
        if j==7: 
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='magenta',alpha=1.0,zorder=4)
        if j==8: 
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='orange',alpha=1.0,zorder=4)

        UP.append(mean+std)
        DO.append(mean-std)

        j+=1

    UP = np.array(UP)
    DO = np.array(DO)

    UP = np.max(UP,axis=0)
    DO = np.min(DO,axis=0)

    hand['all'] = ax22.fill_between(X,DO,UP,color='blue',alpha=1.0,zorder=1)
 
    ##############################################
    #--plot CJ15 and AKP17 F2D/F2N
    X,lower,upper = load_F2DF2N('AKP17')
    #hand['AKP17 DN']  = ax21.fill_between(X,lower,upper,color='darkcyan',alpha=0.3,zorder=1)

    X,lower,upper = load_F2DF2N('CJ15')
    hand['CJ'] = ax11.fill_between(X,lower,upper,color='lightgreen',alpha=0.5,zorder=1)
    ##############################################

    ax21.text(0.65,0.05,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=30,transform=ax21.transAxes)

    ax11.text(0.05,0.50,r'\boldmath$R(D)$',transform=ax11.transAxes,size=50)

    ax11.text(0.05,0.08,r'\textbf{\textrm{(a)}}',size=30,transform=ax11.transAxes)
    ax12.text(0.05,0.08,r'\textbf{\textrm{(b)}}',size=30,transform=ax12.transAxes)
    ax21.text(0.05,0.08,r'\textbf{\textrm{(c)}}',size=30,transform=ax21.transAxes)
    ax22.text(0.05,0.08,r'\textbf{\textrm{(d)}}',size=30,transform=ax22.transAxes)

 
    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_ylim(0.96,1.09)
        ax.set_xlim(0.01,0.88)
        ax.axhline(1,alpha=0.5,color='k',ls='--',zorder=10)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.05)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])

    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom=False)

    for ax in [ax12,ax22]:
        ax.tick_params(labelleft=False)

    ax21.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax22.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax21.xaxis.set_label_coords(0.78,0.00)
    ax22.xaxis.set_label_coords(0.78,0.00)

    handles,labels=[],[]

    handles.append(hand[0])
    #handles.append(hand['KP'])
    handles.append(hand['CJ'])
    labels.append(r'\textbf{\textrm{JAM}}')
    #labels.append(r'\textbf{\textrm{KP}}')
    labels.append(r'\textbf{\textrm{CJ15 (AV18)}}')

    ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    handles,labels=[],[]

    handles.append(hand[1])
    handles.append(hand[2])
    handles.append(hand[3])
    handles.append(hand[4])
    labels.append(r'\textbf{\textrm{AV18}}')
    labels.append(r'\textbf{\textrm{CD-Bonn}}')
    labels.append(r'\textbf{\textrm{WJC-1}}')
    labels.append(r'\textbf{\textrm{WJC-2}}')

    ax12.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 2, columnspacing = 0.5)

    handles,labels=[],[]

    handles.append(hand[5])
    handles.append(hand[6])
    handles.append(hand[7])
    handles.append(hand[8])
    labels.append(r'\textbf{\textrm{HT add.}}')
    labels.append(r'\textbf{\textrm{HT \boldmath$p=n$}}')
    labels.append(r'\textbf{\textrm{HT add. \boldmath$p=n$}}')
    labels.append(r'\textbf{\textrm{GP TMCs}}')

    ax21.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    handles,labels=[],[]

    handles.append(hand['all'])
    #handles.append(hand['KP'])
    labels.append(r'\textbf{\textrm{JAM + systematics}}')
    #labels.append(r'\textbf{\textrm{KP}}')

    ax22.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0.04)

    filename = '%s/gallery/EMC'%(cwd)
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()

def plot_np_ratio():

    wdir1 = 'results/highx/W2cuts/W2cut35'
    wdir2 = 'results/highx/wfs/AV18'
    wdir3 = 'results/highx/wfs/CDBonn'
    wdir4 = 'results/highx/wfs/WJC1'
    wdir5 = 'results/highx/wfs/WJC2'
    wdir6 = 'results/highx/HT/add'
    wdir7 = 'results/highx/HT/iso'
    wdir8 = 'results/highx/HT/addiso'
    wdir9 = 'results/highx/GP/W2cut35'
    #wdir8 = 'noSLAC'
    #wdir9 = 'noJLab'

    WDIR = [wdir1,wdir2,wdir3,wdir4,wdir5,wdir6,wdir7,wdir8,wdir9]

    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax21 = py.subplot(nrows,ncols,3)
    ax22 = py.subplot(nrows,ncols,4)

    j = 0
    hand = {}
    thy  = {}

    Q2 = 10

    UP = []
    DO = []

    for wdir in WDIR:    

        load_config('%s/input.py'%wdir)

        p = load('%s/data/stf-p-F2-Q2=%3.5f.dat'%(wdir,Q2))
        n = load('%s/data/stf-n-F2-Q2=%3.5f.dat'%(wdir,Q2))

        ##############################################
        #--plot EMC ratio
        ##############################################
        X = p['X']

        rat = np.array(n['XF'])/np.array(p['XF'])
        mean = np.mean(np.array(rat),axis=0)
        std  = np.std (np.array(rat),axis=0)
        if j==0: 
            hand[j] = ax11.fill_between(X,mean-std,mean+std,color='red'    ,alpha=1.0,zorder=1)
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='red'    ,alpha=1.0,zorder=5)
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='red'    ,alpha=1.0,zorder=3)
            hand[j] = ax22.fill_between(X,mean-std,mean+std,color='red'    ,alpha=1.0,zorder=2)
        if j==1: 
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='green'  ,alpha=1.0,zorder=4)
        if j==2: 
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='blue'   ,alpha=1.0,zorder=1)
        if j==3: 
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='magenta',alpha=1.0,zorder=2)
        if j==4: 
            hand[j] = ax12.fill_between(X,mean-std,mean+std,color='orange' ,alpha=1.0,zorder=3)
        if j==5: 
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='blue'   ,alpha=1.0,zorder=1)
        if j==6: 
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='green'  ,alpha=1.0,zorder=2)
        if j==7: 
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='magenta',alpha=1.0,zorder=4)
        if j==8: 
            hand[j] = ax21.fill_between(X,mean-std,mean+std,color='orange',alpha=1.0,zorder=4)

        UP.append(mean+std)
        DO.append(mean-std)

        j+=1

    UP = np.array(UP)
    DO = np.array(DO)

    UP = np.max(UP,axis=0)
    DO = np.min(DO,axis=0)

    hand['all'] = ax22.fill_between(X,DO,UP,color='blue',alpha=1.0,zorder=1)
 
    ##############################################
    ##--CJ15 
    #CJ = load_CJ()
    #hand['CJ'] = ax11.fill_between(CJ['X'],CJ['min'],CJ['max'],color='blue',alpha=1.0,zorder=3)
    ##--KP 
    #KP = load_KP()
    #hand['KP'] = ax11.fill_between(KP['X'],KP['min'],KP['max'],color='green',alpha=1.0,zorder=3)
    #hand['KP'] = ax22.fill_between(KP['X'],KP['min'],KP['max'],color='green',alpha=1.0,zorder=3)

    ax21.text(0.65,0.50,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=30,transform=ax21.transAxes)

    ax11.text(0.45,0.80,r'\boldmath$F_2^n/F_2^p$',transform=ax11.transAxes,size=50)

    ax11.text(0.88,0.85,r'\textbf{\textrm{(a)}}',size=30,transform=ax11.transAxes)
    ax12.text(0.88,0.85,r'\textbf{\textrm{(b)}}',size=30,transform=ax12.transAxes)
    ax21.text(0.88,0.85,r'\textbf{\textrm{(c)}}',size=30,transform=ax21.transAxes)
    ax22.text(0.88,0.85,r'\textbf{\textrm{(d)}}',size=30,transform=ax22.transAxes)

 
    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_ylim(0.30,1.00)
        ax.set_xlim(0.01,0.88)
        #ax.axhline(1,alpha=0.5,color='k',ls='--',zorder=10)
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])

    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom=False)

    for ax in [ax12,ax22]:
        ax.tick_params(labelleft=False)

    ax21.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax22.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax21.xaxis.set_label_coords(0.78,0.00)
    ax22.xaxis.set_label_coords(0.78,0.00)

    handles,labels=[],[]

    handles.append(hand[0])
    #handles.append(hand['KP'])
    #handles.append(hand['CJ'])
    labels.append(r'\textbf{\textrm{JAM}}')
    #labels.append(r'\textbf{\textrm{KP}}')
    #labels.append(r'\textbf{\textrm{CJ15 (AV18)}}')

    ax11.legend(handles,labels,frameon=False,loc='lower left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    handles,labels=[],[]

    handles.append(hand[1])
    handles.append(hand[2])
    handles.append(hand[3])
    handles.append(hand[4])
    labels.append(r'\textbf{\textrm{AV18}}')
    labels.append(r'\textbf{\textrm{CD-Bonn}}')
    labels.append(r'\textbf{\textrm{WJC-1}}')
    labels.append(r'\textbf{\textrm{WJC-2}}')

    ax12.legend(handles,labels,frameon=False,loc='lower left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 2, columnspacing = 0.5)

    handles,labels=[],[]

    handles.append(hand[5])
    handles.append(hand[6])
    handles.append(hand[7])
    handles.append(hand[8])
    labels.append(r'\textbf{\textrm{HT add.}}')
    labels.append(r'\textbf{\textrm{HT \boldmath$p=n$}}')
    labels.append(r'\textbf{\textrm{HT add. \boldmath$p=n$}}')
    labels.append(r'\textbf{\textrm{GP TMCs}}')

    ax21.legend(handles,labels,frameon=False,loc='lower left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    handles,labels=[],[]

    handles.append(hand['all'])
    #handles.append(hand['KP'])
    labels.append(r'\textbf{\textrm{JAM + systematics}}')
    #labels.append(r'\textbf{\textrm{KP}}')

    ax22.legend(handles,labels,frameon=False,loc='lower left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0.04)

    filename = '%s/gallery/np_ratio'%(cwd)
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()

if __name__ == "__main__":

    plot_np_ratio()
    plot_EMC()
























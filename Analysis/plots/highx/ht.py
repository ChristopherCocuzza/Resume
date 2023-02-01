#!/usr/bin/env python
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from tools.config import load_config,conf
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

def load_KP():

  F = open('plots/highx/data/KPoff.csv','r')
  L = F.readlines()
  F.close()

  L = [l.strip() for l in L]
  L = [[x for x in l.split()] for l in L]
  L = np.transpose(L)[0]

  data = {}

  data['X']     = []
  data['min']   = []
  data['max']   = []
  for i in range(len(L)):
      if i==0: continue
      data['X']  .append(float(L[i].split(',')[0]))
      data['min'].append(float(L[i].split(',')[1]))
      data['max'].append(float(L[i].split(',')[2]))

  data['X']   = np.array(data['X']) 
  data['min'] = np.array(data['min']) 
  data['max'] = np.array(data['max']) 


  return data

def load_CJ():

  F = open('plots/highx/data/CJoff.csv','r')
  L = F.readlines()
  F.close()

  L = [l.strip() for l in L]
  L = [[x for x in l.split()] for l in L]
  L = np.transpose(L)[0]

  data = {}

  data['X']     = []
  data['min']   = []
  data['max']   = []
  for i in range(len(L)):
      if i==0: continue
      data['X']  .append(float(L[i].split(',')[0]))
      data['min'].append(float(L[i].split(',')[1]))
      data['max'].append(float(L[i].split(',')[2]))

  data['X']   = np.array(data['X']) 
  data['min'] = np.array(data['min']) 
  data['max'] = np.array(data['max']) 

  return data

def plot_ht_mult():

    wdir1 = 'results/highx/W2cuts/W2cut35'
    wdir2 = 'results/highx/HT/add'
    wdir3 = 'results/highx/HT/iso'
    wdir4 = 'results/highx/HT/addiso'
    wdir5 = 'results/highx/GP/W2cut35'
    wdir6 = 'results/highx/wfs/AV18'
    wdir7 = 'results/highx/wfs/CDBonn'
    wdir8 = 'results/highx/wfs/WJC1'
    wdir9 = 'results/highx/wfs/WJC2'

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

    W2cut = 3.5
    M2 = 0.9389**2


    UP,DO = {},{}
    UP['p'] = []
    DO['p'] = []
    UP['n'] = []
    DO['n'] = []

    for wdir in WDIR:    

        hand[j] = {}
        load_config('%s/input.py'%wdir)
        istep = core.get_istep()

        data = {}

        for tar in ['p','n']:
            filename ='%s/data/ht-%s-W2=%3.5f.dat'%(wdir,tar,W2cut)
            data[tar] = load(filename)

        X   = data['p']['X']
        Q2  = X/(1-X)*(W2cut - M2)

        #--load multiplicative contribution CHT/Q2 = H2/Q2/STF(LT+TMC)
        STFp = data['p']['MULT']
        STFn = data['n']['MULT']

        ##############################################
        #--plot multiplicative higher twist
        ##############################################

        mean,std = {},{}
        mean['p'] = np.mean(np.array(STFp),axis=0)
        std['p']  = np.std (np.array(STFp),axis=0)
        mean['n'] = np.mean(np.array(STFn),axis=0)
        std['n']  = np.std (np.array(STFn),axis=0)

        for tar in ['p','n']:
            if tar=='p': color,zorder='red', 1
            if tar=='n': color,zorder='blue',2
            alpha = 0.8
            if j==0:
                hand[j][tar] = ax11.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color=color    ,alpha=alpha,zorder=zorder)
                #hand[j][tar] = ax12.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color='red'    ,alpha=1.0,zorder=5)
                #hand[j][tar] = ax21.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color='red'    ,alpha=1.0,zorder=3)
                #hand[j][tar] = ax22.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color='red'    ,alpha=1.0,zorder=2)
            #--additive
            if j==1: 
                hand[j][tar] = ax21.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color=color   ,alpha=alpha,zorder=zorder)
            #--iso.
            if j==2: 
                hand[j][tar] = ax11.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color='green'  ,alpha=alpha,zorder=zorder)
            #--additive iso.
            #if j==3: 
            #    hand[j][tar] = ax21.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color='green',alpha=1.0,zorder=4)
            #--GP
            if j==4: 
                hand[j][tar] = ax12.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color=color,alpha=alpha,zorder=zorder)
            #--don't plot otherwise

            #--do not add p = n results to systematics
            if j==2: continue
            if j==3: continue
            #--do not add GP to systematics
            if j==4: continue
            UP[tar].append(mean[tar]+std[tar])
            DO[tar].append(mean[tar]-std[tar])

        j+=1

    hand['all'] = {}
    alpha = 0.8
    for tar in ['p','n']:
        UP[tar] = np.array(UP[tar])
        DO[tar] = np.array(DO[tar])

        UP[tar] = np.max(UP[tar],axis=0)
        DO[tar] = np.min(DO[tar],axis=0)

        if tar=='p': color,zorder='red', 1
        if tar=='n': color,zorder='blue',2
        hand['all'][tar] = ax22.fill_between(X,DO[tar],UP[tar],color=color,alpha=alpha,zorder=zorder)
 
    ##############################################
    h0 =-3.2874
    h1 = 1.9274
    h2 =-2.0701
    ht = h0*X**h1*(1+h2*X)
    hand['CJ'] ,= ax12.plot(X,ht/Q2,color='green',ls='--',zorder=5)
   

    xmin,xmax = 0.38,0.90
    Q2min = xmin/(1-xmin)*(W2cut - M2) 
    Q2max = xmax/(1-xmax)*(W2cut - M2)
    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(xmin,xmax)
        ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.4,0.5,0.6,0.7,0.8])
        ax.set_ylim(-0.05,0.3)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    ax21.text(0.08,0.85,r'$W^2 = %2.1f~{\rm GeV}^2$'%W2cut,transform=ax21.transAxes,size=25)
    ax21.text(0.02,0.75,r'$%3.2f < Q^2 < %3.0f~{\rm GeV}^2$'%(Q2min,Q2max),transform=ax21.transAxes,size=25)

    ax11.text(0.35,0.75,r'\boldmath$\frac{C_{\rm HT}^N}{Q^2}$',transform=ax11.transAxes,size=50)
    ax11.text(0.25,0.04,r'\textrm{\textbf{AOT mult.}}'        ,transform=ax11.transAxes,size=35)
    ax12.text(0.25,0.04,r'\textrm{\textbf{GP mult.}}'         ,transform=ax12.transAxes,size=35)
    ax21.text(0.25,0.04,r'\textrm{\textbf{AOT add.}}'         ,transform=ax21.transAxes,size=35)
    ax22.text(0.25,0.04,r'\textrm{\textbf{JAM+systematics}}'  ,transform=ax22.transAxes,size=35)

    ax11.text(0.15,0.05,r'\textbf{\textrm{(a)}}',size=30,transform=ax11.transAxes)
    ax12.text(0.15,0.05,r'\textbf{\textrm{(b)}}',size=30,transform=ax12.transAxes)
    ax21.text(0.15,0.05,r'\textbf{\textrm{(c)}}',size=30,transform=ax21.transAxes)
    ax22.text(0.15,0.05,r'\textbf{\textrm{(d)}}',size=30,transform=ax22.transAxes)

    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom=False)

    for ax in [ax12,ax22]:
        ax.tick_params(labelleft=False)

    ax21.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax22.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax21.xaxis.set_label_coords(0.95,0.00)
    ax22.xaxis.set_label_coords(0.95,0.00)
    
    handles,labels=[],[]

    handles.append(hand[0]['p'])
    handles.append(hand[0]['n'])
    handles.append(hand[2]['p'])
    labels.append(r'\boldmath$C_{\rm HT}^p$')
    labels.append(r'\boldmath$C_{\rm HT}^n$')
    labels.append(r'\boldmath$C_{\rm HT}^p=C_{\rm HT}^n$')

    ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    handles,labels=[],[]

    handles.append(hand['CJ'])
    labels.append(r'\textbf{\textrm{CJ15 (GP \boldmath$p=n$)}}')

    ax12.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0.04)

    filename = '%s/gallery/ht_mult'%(cwd)
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)

    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()

def plot_ht_add():

    wdir1 = 'results/highx/W2cuts/W2cut35'
    wdir2 = 'results/highx/HT/add'
    wdir3 = 'results/highx/HT/iso'
    wdir4 = 'results/highx/HT/addiso'
    wdir5 = 'results/highx/GP/W2cut35'
    wdir6 = 'results/highx/wfs/AV18'
    wdir7 = 'results/highx/wfs/CDBonn'
    wdir8 = 'results/highx/wfs/WJC1'
    wdir9 = 'results/highx/wfs/WJC2'

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

    W2cut = 3.5
    M2 = 0.9389**2


    UP,DO = {},{}
    UP['p'] = []
    DO['p'] = []
    UP['n'] = []
    DO['n'] = []

    for wdir in WDIR:    

        hand[j] = {}
        load_config('%s/input.py'%wdir)
        istep = core.get_istep()

        data = {}

        for tar in ['p','n']:
            filename ='%s/data/ht-%s-W2=%3.5f.dat'%(wdir,tar,W2cut)
            data[tar] = load(filename)

        X   = data['p']['X']
        Q2  = X/(1-X)*(W2cut - M2)

        #--load additive contribution STF(LT+TMC)*CHT/Q2 = H2/Q2
        STFp = data['p']['ADD']
        STFn = data['n']['ADD']

        ##############################################
        #--plot multiplicative higher twist
        ##############################################

        mean,std = {},{}
        mean['p'] = np.mean(np.array(STFp),axis=0)
        std['p']  = np.std (np.array(STFp),axis=0)
        mean['n'] = np.mean(np.array(STFn),axis=0)
        std['n']  = np.std (np.array(STFn),axis=0)

        for tar in ['p','n']:
            if tar=='p': color,zorder='red' ,2
            if tar=='n': color,zorder='blue',1
            alpha = 0.8
            if j==0:
                hand[j][tar] = ax11.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color=color    ,alpha=alpha,zorder=zorder)
            #--additive
            if j==1: 
                hand[j][tar] = ax21.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color=color   ,alpha=alpha,zorder=zorder)
            #--iso.
            #if j==2: 
            #    hand[j][tar] = ax11.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color='green'  ,alpha=1.0,zorder=2)
            #--additive iso.
            if j==3: 
                hand[j][tar] = ax21.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color='green',alpha=alpha,zorder=zorder)
            #--GP
            if j==4: 
                hand[j][tar] = ax12.fill_between(X,mean[tar]-std[tar],mean[tar]+std[tar],color=color,alpha=alpha,zorder=zorder)
            #--don't plot otherwise

            #--do not add p = n results to systematics
            if j==2: continue
            if j==3: continue
            #--do not add GP to systematics
            if j==4: continue
            UP[tar].append(mean[tar]+std[tar])
            DO[tar].append(mean[tar]-std[tar])

        j+=1

    hand['all'] = {}
    alpha = 0.8
    for tar in ['p','n']:
        UP[tar] = np.array(UP[tar])
        DO[tar] = np.array(DO[tar])

        UP[tar] = np.max(UP[tar],axis=0)
        DO[tar] = np.min(DO[tar],axis=0)

        if tar=='p': color,zorder='red' ,2
        if tar=='n': color,zorder='blue',1
        hand['all'][tar] = ax22.fill_between(X,DO[tar],UP[tar],color=color,alpha=alpha,zorder=zorder)
 
    ##############################################

    xmin,xmax = 0.38,0.90
    Q2min = xmin/(1-xmin)*(W2cut - M2) 
    Q2max = xmax/(1-xmax)*(W2cut - M2)
    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(xmin,xmax)
        ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.4,0.5,0.6,0.7,0.8])
        ax.set_ylim(-0.003,0.009)
        minorLocator = MultipleLocator(0.001)
        majorLocator = MultipleLocator(0.002)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    ax21.text(0.61,0.85,r'$W^2 = %2.1f~{\rm GeV}^2$'%W2cut,transform=ax21.transAxes,size=25)
    ax21.text(0.55,0.75,r'$%3.2f < Q^2 < %3.0f~{\rm GeV}^2$'%(Q2min,Q2max),transform=ax21.transAxes,size=25)

    ax11.text(0.35,0.75,r'\boldmath$\frac{xH_2^N}{Q^2}$'      ,transform=ax11.transAxes,size=50)
    ax11.text(0.25,0.04,r'\textrm{\textbf{AOT mult.}}'        ,transform=ax11.transAxes,size=35)
    ax12.text(0.25,0.04,r'\textrm{\textbf{GP mult.}}'         ,transform=ax12.transAxes,size=35)
    ax21.text(0.25,0.04,r'\textrm{\textbf{AOT add.}}'         ,transform=ax21.transAxes,size=35)
    ax22.text(0.25,0.04,r'\textrm{\textbf{JAM+systematics}}'  ,transform=ax22.transAxes,size=35)

    ax11.text(0.15,0.05,r'\textbf{\textrm{(a)}}',size=30,transform=ax11.transAxes)
    ax12.text(0.15,0.05,r'\textbf{\textrm{(b)}}',size=30,transform=ax12.transAxes)
    ax21.text(0.15,0.05,r'\textbf{\textrm{(c)}}',size=30,transform=ax21.transAxes)
    ax22.text(0.15,0.05,r'\textbf{\textrm{(d)}}',size=30,transform=ax22.transAxes)

    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom=False)

    for ax in [ax12,ax22]:
        ax.tick_params(labelleft=False)

    ax21.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax22.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax21.xaxis.set_label_coords(0.95,0.00)
    ax22.xaxis.set_label_coords(0.95,0.00)
    
    handles,labels=[],[]

    handles.append(hand[0]['p'])
    handles.append(hand[0]['n'])
    handles.append(hand[3]['p'])
    labels.append(r'\boldmath$H_2^p$')
    labels.append(r'\boldmath$H_2^n$')
    labels.append(r'\boldmath$H_2^p=H_2^n$')

    ax11.legend(handles,labels,frameon=False,loc='upper right',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0.04)

    filename = '%s/gallery/ht_add'%(cwd)
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

    plot_ht_add()
    plot_ht_mult()

    sys.exit()
    X        = STF['p']['X']
    STF['p'] = STF['p']['XF']
    STF['n'] = STF['n']['XF']


    ht_type = 'mult'
    if 'ht type' in conf: ht_type = conf['ht type']

    ##############################################
    #--plot ht
    ##############################################

    if mode == 1:
        meanp = np.mean(np.array(F2p),axis=0)
        stdp  = np.std (np.array(F2p),axis=0)
        meann = np.mean(np.array(F2n),axis=0)
        stdn  = np.std (np.array(F2n),axis=0)
        meanplus  = np.mean(np.array(plus),axis=0)
        stdplus   = np.std (np.array(plus),axis=0)
        meanminus = np.mean(np.array(minus),axis=0)
        stdminus  = np.std (np.array(minus),axis=0)
        hand['p']     = ax12.fill_between(X,meanp-stdp,meanp+stdp,color='red'  ,alpha=0.9, zorder=2)
        hand['n']     = ax12.fill_between(X,meann-stdn,meann+stdn,color='green',alpha=0.9, zorder=1)
        hand['plus']  = ax12.fill_between(X,meanplus-stdplus  ,meanplus+stdplus  ,color='blue'  ,alpha=0.9, zorder=1)
        hand['minus'] = ax12.fill_between(X,meanminus-stdminus,meanminus+stdminus,color='purple',alpha=0.9, zorder=1)

    print 
    ##############################################


    #if mode==0:
    #    sm   = py.cm.ScalarMappable(cmap=cmap)
    #    sm.set_array([])
    #    cax = fig.add_axes([0.45,0.88,0.30,0.04])
    #    cax.tick_params(axis='both',which='both',labelsize=15,direction='in')
    #    cax.xaxis.set_label_coords(0.65,-1.8)
    #    cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
    #    cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=20)

    py.tight_layout()
    filename = '%s/gallery/ht-W2=%3.5f'%(wdir,W2cut)
    if mode == 1: filename += '-bands'
    filename += '.png'
    print()
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()























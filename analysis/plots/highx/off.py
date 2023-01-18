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

def plot_off():

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

        filename ='%s/data/off-Q2=%3.5f.dat'%(wdir,Q2)
        data = load(filename)

        ##############################################
        #--plot offshell
        ##############################################
        X   = data['X']

        dfD = data['dfD']
        mean = np.mean(np.array(dfD),axis=0)
        std  = np.std (np.array(dfD),axis=0)
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
    X1   = 10**np.linspace(-4,-1,100)
    X2   = np.linspace(0.1,0.98,100)
    X    = np.append(X1,X2)
    #--CJ15 
    CJ = load_CJ()
    #C =-3.6735
    #x0= 5.7717e-2
    #x1=0.36419
    #dfcj=C*(X-x0)*(X-x1)*(1+x0-X)
    #hand['CJ'] ,= ax11.plot(X,dfcj,'b--')
    hand['CJ'] = ax11.fill_between(CJ['X'],CJ['min'],CJ['max'],color='blue',alpha=1.0,zorder=3)
    #--KP 
    KP = load_KP()
    #C = 8.10
    #x0= 0.448
    #x1= 0.05
    #dfcj=C*(X-x0)*(X-x1)*(1+x0-X)
    #hand['KP'] ,= ax11.plot(X,dfcj,'g--')
    hand['KP'] = ax11.fill_between(KP['X'],KP['min'],KP['max'],color='green',alpha=1.0,zorder=3)
    hand['KP'] = ax22.fill_between(KP['X'],KP['min'],KP['max'],color='green',alpha=1.0,zorder=3)

    ax21.text(0.65,0.05,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=30,transform=ax21.transAxes)

    ax11.text(0.45,0.80,r'\boldmath$\delta f$',transform=ax11.transAxes,size=50)

    ax11.text(0.05,0.08,r'\textbf{\textrm{(a)}}',size=30,transform=ax11.transAxes)
    ax12.text(0.05,0.08,r'\textbf{\textrm{(b)}}',size=30,transform=ax12.transAxes)
    ax21.text(0.05,0.08,r'\textbf{\textrm{(c)}}',size=30,transform=ax21.transAxes)
    ax22.text(0.05,0.08,r'\textbf{\textrm{(d)}}',size=30,transform=ax22.transAxes)

 
    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_ylim(-1.2,2.2)
        ax.set_xlim(0.01,0.88)
        ax.axhline(0,alpha=0.5,color='k',ls='--',zorder=10)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
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
    handles.append(hand['KP'])
    handles.append(hand['CJ'])
    labels.append(r'\textbf{\textrm{JAM}}')
    labels.append(r'\textbf{\textrm{KP}}')
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
    handles.append(hand['KP'])
    labels.append(r'\textbf{\textrm{JAM + systematics}}')
    labels.append(r'\textbf{\textrm{KP}}')

    ax22.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0.04)

    filename = '%s/gallery/off'%(cwd)
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()

def plot_off2():

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

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)

    j = 0
    hand = {}
    thy  = {}

    Q2 = 10

    UPu = []
    DOu = []
    UPd = []
    DOd = []

    for wdir in WDIR:    

        hand[j] = {}
        load_config('%s/input.py'%wdir)
        istep = core.get_istep()

        ##############################################
        #--plot off-shell PDF ratios
        ##############################################
        data=load('%s/data/off-pdf-Q2=%3.5f.dat'%(wdir,int(Q2)))
        pdfs=load('%s/data/pdf-Q2=%3.5f.dat'%(wdir,int(Q2)))

        X    = np.array(data['X'])
        uv   = np.array(data['XF']['uv'])
        dv   = np.array(data['XF']['dv'])
        uvon = np.array(pdfs['XF']['uv'])
        dvon = np.array(pdfs['XF']['dv'])
        ratu = uv/uvon
        ratd = dv/dvon

        meanu = np.mean(ratu,axis=0)
        stdu  = np.std (ratu,axis=0)
        meand = np.mean(ratd,axis=0)
        stdd  = np.std (ratd,axis=0)

        alpha = 0.8
        if j==0: 
            hand[j] = ax11.fill_between(X,meanu-stdu,meanu+stdu,color='red'    ,alpha=1.0,zorder=2)
            hand[j] = ax12.fill_between(X,meand-stdd,meand+stdd,color='red'    ,alpha=1.0,zorder=2)
        #if j==1: 
        #    hand[j] = ax12.fill_between(X,meanu-stdu,meanu+stdu,color='green'  ,alpha=1.0,zorder=4)
        #if j==2: 
        #    hand[j] = ax12.fill_between(X,meanu-stdu,meanu+stdu,color='blue'   ,alpha=1.0,zorder=1)
        #if j==3: 
        #    hand[j] = ax12.fill_between(X,meanu-stdu,meanu+stdu,color='magenta',alpha=1.0,zorder=2)
        #if j==4: 
        #    hand[j] = ax12.fill_between(X,meanu-stdu,meanu+stdu,color='orange' ,alpha=1.0,zorder=3)
        #if j==5: 
        #    hand[j] = ax21.fill_between(X,meanu-stdu,meanu+stdu,color='blue'   ,alpha=1.0,zorder=1)
        #if j==6: 
        #    hand[j] = ax21.fill_between(X,meanu-stdu,meanu+stdu,color='green'  ,alpha=1.0,zorder=2)
        #if j==7: 
        #    hand[j] = ax21.fill_between(X,meanu-stdu,meanu+stdu,color='magenta',alpha=1.0,zorder=4)
        #if j==8: 
        #    hand[j] = ax21.fill_between(X,meanu-stdu,meanu+stdu,color='orange',alpha=1.0,zorder=4)

        UPu.append(meanu+stdu)
        DOu.append(meanu-stdu)
        UPd.append(meand+stdd)
        DOd.append(meand-stdd)

        j+=1

    alpha = 1.0
    UPu = np.max(np.array(UPu),axis=0)
    DOu = np.min(np.array(DOu),axis=0)
    UPd = np.max(np.array(UPd),axis=0)
    DOd = np.min(np.array(DOd),axis=0)

    color,zorder='blue', 1
    hand['all'] = ax11.fill_between(X,DOu,UPu,color=color,alpha=alpha,zorder=zorder)
    hand['all'] = ax12.fill_between(X,DOd,UPd,color=color,alpha=alpha,zorder=zorder)
 
    ##############################################


    ax12.text(0.05,0.85,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=25,transform=ax12.transAxes)

    ax11.text(0.05,0.50,r'\boldmath$\delta u/u$',transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.50,r'\boldmath$\delta d/d$',transform=ax12.transAxes,size=40)
 
    for ax in [ax11,ax12]:
        ax.set_xlim(0.15,0.85)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.set_ylim(-1.20,7.50)
        minorLocator = MultipleLocator(0.5)
        majorLocator = MultipleLocator(2.0)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.axhline(0,0,1,color='black',ls=':',alpha=0.5)
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    for ax in [ax12]:
        ax.tick_params(labelleft=False)
    
    ax11.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax12.set_xlabel(r'\boldmath$x$'         ,size=40)
    ax11.xaxis.set_label_coords(0.78,0.00)
    ax12.xaxis.set_label_coords(0.78,0.00)

    handles,labels=[],[]

    handles,labels=[],[]

    handles.append(hand[0])
    handles.append(hand['all'])
    labels.append(r'\textbf{\textrm{JAM}}')
    labels.append(r'\textbf{\textrm{JAM + systematics}}')

    ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=25, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)


    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0.04)

    filename = '%s/gallery/off2'%(cwd)
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)

    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()


 





if __name__ == "__main__":

    #plot_off()
    plot_off2()
























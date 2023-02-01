#!/usr/bin/env python
import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version

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

from analysis.qpdlib.qpdcalc import QPDCALC

import kmeanconf as kc

cwd = 'plots/highx'

def plot_du_ratio(SETS):

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

        data=load('%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2))

        ##############################################
        #--plot offshell
        ##############################################
        X   = data['X']

        rat  = data['XF']['d/u']
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
    #--plot other PDF sets
    for SET in SETS:
        _SET, _color, _alpha = SETS[SET][0], SETS[SET][1], SETS[SET][2]

        pdf = _SET.get_xpdf('d/u',X,Q2)

        hand[SET] = ax11.fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,zorder=1)
    ##############################################

    ax21.text(0.35,0.85,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=30,transform=ax21.transAxes)

    ax11.text(0.45,0.80,r'\boldmath$d/u$',transform=ax11.transAxes,size=50)

    ax11.text(0.88,0.85,r'\textbf{\textrm{(a)}}',size=30,transform=ax11.transAxes)
    ax12.text(0.88,0.85,r'\textbf{\textrm{(b)}}',size=30,transform=ax12.transAxes)
    ax21.text(0.88,0.85,r'\textbf{\textrm{(c)}}',size=30,transform=ax21.transAxes)
    ax22.text(0.88,0.85,r'\textbf{\textrm{(d)}}',size=30,transform=ax22.transAxes)

 
    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_ylim(0,0.9)
        ax.set_xlim(0.01,0.88)
        #ax.axhline(0,alpha=0.5,color='k',ls='--',zorder=10)
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
        #ax.set_yticks([0.0,0.2,0.4,0.6])


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
    handles.append(hand['CJ15'])
    handles.append(hand['ABMP16'])
    handles.append(hand['NNPDF40'])
    handles.append(hand['MSHT20'])
    handles.append(hand['CT18'])
    labels.append(r'\textbf{\textrm{JAM}}')
    labels.append(r'\textrm{CJ15}')
    labels.append(r'\textrm{ABMP16}')
    labels.append(r'\textrm{NNPDF4.0}')
    labels.append(r'\textrm{MSHT20}')
    labels.append(r'\textrm{CT18}')

    ax11.legend(handles,labels,frameon=False,loc='lower left',fontsize=22, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

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
    labels.append(r'\textbf{\textrm{JAM + systematics}}')

    ax22.legend(handles,labels,frameon=False,loc='lower left',fontsize=28, handletextpad = 0.5, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0.04,top=0.98)

    filename = '%s/gallery/du_ratio'%(cwd)
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

    #--update NNPDF and MSHT
    #--get PDF sets for comparison
    SETS = {}
    SETS['CJ15']    = (QPDCALC('CJ15nlo'                   ,ismc=False)                    ,'green'    ,0.4) 
    SETS['ABMP16']  = (QPDCALC('ABMP16_3_nlo'              ,ismc=False)                    ,'darkblue' ,0.5)
    SETS['NNPDF40'] = (QPDCALC('NNPDF40_nnlo_as_01180'     ,ismc=True ,central_only=False) ,'gold'     ,0.5)
    SETS['MSHT20']  = (QPDCALC('MSHT20nnlo_as118'          ,ismc=False)                    ,'cyan'     ,0.3)
    SETS['CT18']    = (QPDCALC('CT18NNLO'                  ,ismc=True ,central_only=False) ,'gray'     ,0.5)

    plot_du_ratio(SETS)
























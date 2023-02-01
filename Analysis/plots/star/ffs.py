#!/usr/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT


#--matplotlib
from matplotlib.ticker import MultipleLocator
import pylab as py
import matplotlib.gridspec as gridspec
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#-- from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier
import kmeanconf as kc

from analysis.qpdlib import ff

#--from obslib
from obslib.wzrv.theory import WZRV
from obslib.wzrv.reader import READER


cwd = 'plots/star'

if __name__=="__main__":

    wdir = 'step36new'
    #wdir = 'results/pol/step36'
    Q2   = 100
    
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    try:
        _ffpion = load('%s/data/ffpion-Q2=%3.5f.dat' % (wdir, Q2))
    except:
        ff.gen_xf(wdir,'pion',Q2=Q2)
        _ffpion = load('%s/data/ffpion-Q2=%3.5f.dat' % (wdir, Q2))

    ffpion={}
    for flav in ['c', 'b', 'd', 'g', 'db', 's', 'u', 'sb', 'ub']:
        ffpion[flav]={}
        ffpion[flav]['xf0']=np.mean(_ffpion['XF'][flav],axis=0)
        ffpion[flav]['dxf']=np.std(_ffpion['XF'][flav],axis=0)
    
    try:
        _ffkaon = load('%s/data/ffkaon-Q2=%3.5f.dat' % (wdir, Q2))
    except:
        ff.gen_xf(wdir,'kaon',Q2=Q2)
        _ffkaon = load('%s/data/ffkaon-Q2=%3.5f.dat' % (wdir, Q2))

    ffkaon={}
    for flav in ['c', 'b', 'd', 'g', 'db', 's', 'u', 'sb', 'ub']:
        ffkaon[flav]={}
        ffkaon[flav]['xf0']=np.mean(_ffkaon['XF'][flav],axis=0)
        ffkaon[flav]['dxf']=np.std(_ffkaon['XF'][flav],axis=0)
    
    try:
        _ffhadron = load('%s/data/ffhadron-Q2=%3.5f.dat' % (wdir, Q2))
    except:
        ff.gen_xf(wdir,'hadron',Q2=Q2)
        _ffhadron = load('%s/data/ffhadron-Q2=%3.5f.dat' % (wdir, Q2))

    ffhadron={}
    for flav in ['c', 'b', 'd', 'g', 'db', 's', 'u', 'sb', 'ub']:
        ffhadron[flav]={}
        ffhadron[flav]['xf0']=np.mean(_ffhadron['XF'][flav],axis=0)
        ffhadron[flav]['dxf']=np.std(_ffhadron['XF'][flav],axis=0)
        
        
    ffpiK={}
    for flav in ['c', 'b', 'd', 'g', 'db', 's', 'u', 'sb', 'ub']:
        ffpiK[flav]={}
        ffpiK[flav]['xf0']=np.mean( np.array(_ffhadron['XF'][flav]) 
                                   -np.array(_ffpion['XF'][flav])
                                   -np.array(_ffkaon['XF'][flav]),axis=0)
        ffpiK[flav]['dxf']=np.std(  np.array(_ffhadron['XF'][flav]) 
                                   -np.array(_ffpion['XF'][flav])
                                   -np.array(_ffkaon['XF'][flav]),axis=0)

    
    Z=_ffpion['X']

    nrows,ncols=3,3
    fig = py.figure(figsize=(ncols*7,nrows*4))
    cnt=0
    AX={}
    H={}
    for flav in ['u','d','s','ub','db','sb','c','b','g']:
        cnt+=1
        ax=py.subplot(nrows,ncols,cnt)

        up=ffpiK[flav]['xf0']+ffpiK[flav]['dxf']
        do=ffpiK[flav]['xf0']-ffpiK[flav]['dxf']
        H['dh']=ax.fill_between(Z,do,up,color='y',alpha=0.5)

        up=ffpion[flav]['xf0']+ffpion[flav]['dxf']
        do=ffpion[flav]['xf0']-ffpion[flav]['dxf']
        H['pi']=ax.fill_between(Z,do,up,color='r',alpha=0.7)

        up=ffkaon[flav]['xf0']+ffkaon[flav]['dxf']
        do=ffkaon[flav]['xf0']-ffkaon[flav]['dxf']
        H['K']=ax.fill_between(Z,do,up,color='b',alpha=0.6)

        up=ffhadron[flav]['xf0']+ffhadron[flav]['dxf']
        do=ffhadron[flav]['xf0']-ffhadron[flav]['dxf']
        H['h']=ax.fill_between(Z,do,up,color='g',alpha=0.6)


        AX[cnt]=ax


    for _ in AX: 
        AX[_].tick_params(axis='both', which='both', labelsize=30,direction='in')
        #AX[_].set_xlim(8e-3,1)
        AX[_].set_xticks([.2,.3,.4,.5,.6,.7,.8,.9])
        #ax.set_ylim(0,2)
        AX[_].semilogy()
        AX[_].set_xlim(0.15,0.95)
        AX[_].set_ylim(0.0006,2)

    for _ in [1,2,3,4,5,6]: AX[_].set_xticklabels([])
    for _ in [2,3,5,6,8,9]: AX[_].set_yticklabels([])
    for _ in [7,8,9]:
        #AX[_].set_xticklabels([r'$0.2$',r'$0.3$',r'$0.4$',r'$0.5$',r'$0.6$',r'$0.7$',r'$0.8$',''])
        AX[_].set_xticklabels([r'$0.2$',r'',r'$0.4$',r'',r'$0.6$',r'',r'$0.8$',''])
        AX[_].xaxis.set_label_coords(0.95, -0.01)
        AX[_].set_xlabel(r'\boldmath$z$',size=40)
    for _ in AX: AX[_].set_yticks([0.001,0.01,0.1,1])
    for _ in [1,4,7]:
        AX[_].set_yticklabels([r'$0.001$',r'$0.01$',r'$0.1$',r'$1$'])

    #['u','d','s','ub','db','sb','c','b','g']:
    AX[1].text(0.7,0.8,r'\boldmath{$zD_u$}',size=40,transform=AX[1].transAxes)   
    AX[2].text(0.7,0.8,r'\boldmath{$zD_d$}',size=40,transform=AX[2].transAxes)   
    AX[3].text(0.7,0.8,r'\boldmath{$zD_s$}',size=40,transform=AX[3].transAxes)   
    AX[4].text(0.7,0.8,r'\boldmath{$zD_{\bar{u}}$}',size=40,transform=AX[4].transAxes)   
    AX[5].text(0.7,0.8,r'\boldmath{$zD_{\bar{d}}$}',size=40,transform=AX[5].transAxes)   
    AX[6].text(0.7,0.8,r'\boldmath{$zD_{\bar{s}}$}',size=40,transform=AX[6].transAxes)   
    AX[7].text(0.7,0.8,r'\boldmath{$zD_{c}$}',size=40,transform=AX[7].transAxes)   
    AX[8].text(0.7,0.8,r'\boldmath{$zD_{b}$}',size=40,transform=AX[8].transAxes)   
    AX[9].text(0.7,0.8,r'\boldmath{$zD_{g}$}',size=40,transform=AX[9].transAxes)   

    AX[8].legend([H[_] for _ in ['pi','K','h','dh']], 
                 [r'\boldmath{$\pi^+$}',r'\boldmath{$K^+$}'
                  ,r'\boldmath{$h^+$}',r'\boldmath{$\delta h^+$}'],fontsize=30,loc='lower left'
                ,frameon=0,ncol=2,handletextpad=0.5, columnspacing=0.5) 

    AX[1].text(0.08,0.10,r'$\mu^2=%d{\rm ~GeV^2}$'%Q2,size=35,transform=AX[1].transAxes)   

    py.subplots_adjust(left=0.05, bottom=0.05, right=0.99, top=0.98, wspace=0.03, hspace=0.05)
    filename = '%s/gallery/ffs.png'%cwd
    py.savefig(filename)
    print('Saving figure to %s'%filename)
    
    
    
    
    
    
    
    
    
    
    
    
    






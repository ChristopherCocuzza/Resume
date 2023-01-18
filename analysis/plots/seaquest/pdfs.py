#!/usr/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
import numpy as np
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python3/sets'

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--matplotlib
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py

#--from scipy stack 
from scipy.integrate import quad

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from qcdlib
from qcdlib.qpdcalc import QPDCALC

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

cwd = 'plots/seaquest'

def plot_xf_main(WDIR,kc,SETS,Q2):

    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax21=py.subplot(nrows,ncols,3)
    ax22=py.subplot(nrows,ncols,4)

    filename = '%s/gallery/PDFs'%cwd

    j = 0
    hand = {}
    thy  = {}
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data=load('%s/data/pdf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))
            
        replicas=core.get_replicas(wdir)
        cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
        best_cluster=cluster_order[0]

        X=data['X']

        for flav in data['XF']:
            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            if flav=='uv' or flav=='dv': ax = ax11
            elif flav=='g':              ax = ax12
            elif flav=='db+ub':          ax = ax21
            elif flav=='s+sb':           ax = ax22
            else: continue

            #--plot average and standard deviation
            if flav=='g':
                mean /= 10.0
                std  /= 10.0

            thy[j]  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=0.9,zorder=5)

            #--plot other PDF sets
            if j==0:
                for SET in SETS:
                    _SET, _color, _alpha, _zorder = SETS[SET][0], SETS[SET][1], SETS[SET][2], SETS[SET][3]

                    pdf = _SET.get_xpdf(flav,X,Q2)

                    if flav=='g':
                        pdf['xfmin'] /= 10.0
                        pdf['xfmax'] /= 10.0

                    hand[SET] = ax.fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,zorder=_zorder)

        j+=1


    for ax in [ax11,ax12,ax21,ax22]:
          ax.set_xlim(1e-2,0.9)
          ax.semilogx()
            
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
          ax.set_xticks([0.01,0.1])
          ax.set_xticklabels([r'$0.01$',r'$0.1$'])

    ax11.tick_params(axis='both', which='both', labelbottom=False)
    ax12.tick_params(axis='both', which='both', labelbottom=False)
    ax21.tick_params(axis='both', which='both')
    ax22.tick_params(axis='both', which='both')

    ax11.set_ylim(0.0,0.7)
    ax12.set_ylim(0.0,0.5)
    ax21.set_ylim(0.0,0.7)
    ax22.set_ylim(0.0,0.7)

    ax11.set_yticks([0.2,0.4,0.6])
    ax12.set_yticks([0.2,0.4])
    ax21.set_yticks([0.2,0.4,0.6])
    ax22.set_yticks([0.2,0.4,0.6])

    minorLocator = MultipleLocator(0.05)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax21.yaxis.set_minor_locator(minorLocator)
    ax22.yaxis.set_minor_locator(minorLocator)

    for ax in [ax21,ax22]:
        ax.set_xlabel(r'\boldmath$x$' ,size=50)
        ax.xaxis.set_label_coords(0.95,0.00)

    ax11.text(0.80 ,0.70  ,r'\boldmath{$xu_{v}$}'            , transform=ax11.transAxes,size=40)
    ax11.text(0.53 ,0.15  ,r'\boldmath{$xd_{v}$}'            , transform=ax11.transAxes,size=40)
    ax12.text(0.05 ,0.10  ,r'\boldmath{$xg/10$}'             , transform=ax12.transAxes,size=40)
    ax21.text(0.05 ,0.10  ,r'\boldmath{$x(\bar{d}+\bar{u})$}', transform=ax21.transAxes,size=40)
    ax22.text(0.57 ,0.30  ,r'\boldmath{$x(s+\bar{s})$}'      , transform=ax22.transAxes,size=40)

    ax21.text(0.55,0.80,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax21.transAxes,size=30)

    #ax21.axhline(0.0,ls=':',color='black',alpha=0.5)

    handles, labels = [],[]
    handles.append(thy[0])
    labels.append(r'\textrm{\textbf{JAM}}')
    if 'NNPDF' in hand:
        handles.append(hand['NNPDF'])
        labels .append(r'\textbf{\textrm{NNPDF3.1}}')
    if 'ABMP16' in hand: 
        handles.append(hand['ABMP16'])
        labels .append(r'\textbf{\textrm{ABMP16}}')
    ax12.legend(handles,labels,loc=(0.48,0.45), fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    handles,labels = [],[]
    if 'CJ15' in hand:
        handles.append(hand['CJ15'])
        labels .append(r'\textbf{\textrm{CJ15}}')
    if 'CT18' in hand: 
        handles.append(hand['CT18'])
        labels .append(r'\textbf{\textrm{CT18}}')
    ax22.legend(handles,labels,loc=(0.48,0.60), fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.15, right = 0.97, top = 0.97)

    #filename+='.png'
    filename+='.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)

    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_xf_ratio(WDIR,kc,SETS,Q2):

    nrows,ncols=2,1
    fig = py.figure(figsize=(ncols*12,nrows*15/2))
    ax11=py.subplot(nrows,ncols,1)
    ax21=py.subplot(nrows,ncols,2)

    filename = '%s/gallery/PDFs-ratio'%cwd

    j = 0
    hand = {}
    thy = {}
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        data=load('%s/data/pdf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))
            
        replicas=core.get_replicas(wdir)
        cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
        best_cluster=cluster_order[0]


        for flav in data['XF']:
            X=data['X']
            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            if   flav=='db/ub': ax = ax11
            elif flav=='db-ub': ax = ax21
            else: continue

            #--plot average and standard deviation
            thy[j]  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=1.0,zorder=5)


        j+=1

    #--plot other PDF sets
    for SET in SETS:
        _SET, _color, _alpha, _zorder = SETS[SET][0], SETS[SET][1], SETS[SET][2], SETS[SET][3]

        if SET in ['CJ15','NNPDF']:  ax,flav = ax11, 'db/ub'
        if SET in ['ABMP16','CT18']: ax,flav = ax21, 'db-ub'

        if flav=='db/ub':
            X = X[:300]

        pdf = _SET.get_xpdf(flav,X,Q2)

        hand[SET] = ax.fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,zorder=_zorder)

    for ax in [ax11,ax21]:
        ax.set_xlim(0.02,0.42)
        ax.set_xticks([0.1,0.2,0.3,0.4])
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=40,length=10)
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=40,length=5)
        minorLocator = MultipleLocator(0.02)
        ax.xaxis.set_minor_locator(minorLocator)

    ax11.tick_params(labelbottom=False)

    ax21.set_xlabel(r'\boldmath$x$'    ,size=60)
    ax21.xaxis.set_label_coords(0.82,0.00)

    ax11.set_ylim(0.4,1.9)
    ax11.set_yticks([0.5,1.0,1.5])


    minorLocator = MultipleLocator(0.1)
    ax11.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.005)
    ax21.yaxis.set_minor_locator(minorLocator)

    ax21.set_ylim(-0.000,0.045)
    ax21.set_yticks([0.00,0.02,0.04])
    ax21.set_yticklabels([r'$0$',r'$0.02$',r'$0.04$'])

    ax11.text(0.04,0.80,r'\boldmath{$\bar{d}/\bar{u}$}'    ,transform=ax11.transAxes,size=70)
    ax21.text(0.05,0.10,r'\boldmath{$x(\bar{d}-\bar{u})$}' ,transform=ax21.transAxes,size=70)

    ax21.text(0.65,0.40,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax21.transAxes,size=40)

    ax11.axhline(1.0,ls=':',color='black',alpha=0.5,zorder=10)
    #ax21.axhline(0.0,ls=':',color='black',alpha=0.5,zorder=10)

    hand['blank'] ,= ax.plot([0],[0],alpha=0)
    handles,labels = [],[]
    handles.append(thy[0])
    handles.append(hand['NNPDF'])
    handles.append(hand['CJ15'])
    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textbf{\textrm{NNPDF3.1}}')
    labels.append(r'\textbf{\textrm{CJ15}}')

    ax11.legend(handles,labels,loc=(0.02,0.01),  fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    handles,labels = [],[]
    handles.append(thy[0])
    handles.append(hand['ABMP16'])
    handles.append(hand['CT18'])
    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textbf{\textrm{ABMP16}}')
    labels.append(r'\textbf{\textrm{CT18}}')
    ax21.legend(handles,labels,loc='upper right',  fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.20)

    #filename+='.png'
    filename+='.pdf'
    ax11.set_rasterized(True)
    ax21.set_rasterized(True)
    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

if __name__ == "__main__":

    wdir1 = 'results/sq/final'

    WDIR = [wdir1]

    #--get PDF sets for comparison
    SETS = {}
    SETS['ABMP16'] = (QPDCALC('ABMP16_3_nlo'              ,ismc=False)                    ,'cornflowerblue'     ,0.8 ,6)
    SETS['NNPDF']  = (QPDCALC('NNPDF31_nlo_as_0118'       ,ismc=True ,central_only=False) ,'gold'               ,0.4 ,1)
    SETS['CJ15']   = (QPDCALC('CJ15nlo'                   ,ismc=False)                    ,'gray'               ,0.5 ,2) 
    SETS['CT18']   = (QPDCALC('CT18NLO'                   ,ismc=True ,central_only=False) ,'lightgreen'         ,0.8 ,2)
    #SETS['MSHT20'] = (QPDCALC('MSHT20nnlo_as118'          ,ismc=False)                    ,'cyan'              ,0.4 ,2)

    Q2 = 10

    plot_xf_main (WDIR,kc,SETS,Q2)
    plot_xf_ratio(WDIR,kc,SETS,Q2)







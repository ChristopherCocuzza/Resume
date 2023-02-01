#!/usr/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
import numpy as np
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version

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

#--from local
from analysis.qpdlib.qpdcalc import QPDCALC
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

cwd = 'plots/highx'

def plot_xf_main(WDIR,kc,SETS):

    nrows,ncols=3,2
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax21=py.subplot(nrows,ncols,3)
    ax22=py.subplot(nrows,ncols,4)
    ax31=py.subplot(nrows,ncols,5)
    ax32=py.subplot(nrows,ncols,6)

    filename = '%s/gallery/PDFs'%cwd

    j = 0
    hand = {}
    thy  = {}
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        Q2 = 10
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
            elif flav=='db-ub':          ax = ax22
            elif flav=='s+sb':           ax = ax31
            elif flav=='Rs':             ax = ax32
            else: continue

            #--plot average and standard deviation
            if flav=='g':
                mean /= 10.0
                std  /= 10.0

            where = [1 for i in range(len(X))]
            if flav=='Rs':
                where = []
                for x in X:
                    if x < 0.2: where.append(1)
                    if x > 0.2: where.append(0)

            thy[j]  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=0.9,zorder=5,where=where)

            #--plot other PDF sets
            if j==0:
                for SET in SETS:
                    _SET, _color, _alpha = SETS[SET][0], SETS[SET][1], SETS[SET][2]

                    pdf = _SET.get_xpdf(flav,X,Q2)

                    if flav=='g':
                        pdf['xfmin'] /= 10.0
                        pdf['xfmax'] /= 10.0

                    where = [1 for i in range(len(X))]
                    if flav=='Rs':
                        where = []
                        for x in X:
                            if x < 0.2: where.append(1)
                            if x > 0.2: where.append(0)

                    hand[SET] = ax.fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,where=where,zorder=1)

        j+=1


    for ax in [ax11,ax12,ax21,ax22,ax31,ax32]:
          ax.set_xlim(1e-2,1)
          ax.semilogx()
            
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
          ax.set_xticks([0.01,0.1,1])
          ax.set_xticklabels([r'$0.01$',r'$0.1$',r'$1$'])

    ax11.tick_params(axis='both', which='both', labelbottom=False)
    ax12.tick_params(axis='both', which='both', labelbottom=False)
    ax21.tick_params(axis='both', which='both', labelbottom=False)
    ax22.tick_params(axis='both', which='both', labelbottom=False)

    ax11.set_ylim(0,0.7)
    ax12.set_ylim(0,0.5)
    ax21.set_ylim(-0.05,0.7)
    ax22.set_ylim(-0.04,0.08)
    ax31.set_ylim(0,0.7)
    ax32.set_ylim(0,1.2)

    ax11.set_yticks([0.2,0.4,0.6])
    ax12.set_yticks([0.2,0.4])
    ax21.set_yticks([0,0.2,0.4,0.6])
    ax22.set_yticks([-0.02,0,0.02,0.04,0.06])
    ax31.set_yticks([0.2,0.4,0.6])
    ax32.set_yticks([0.5,1.0])

    minorLocator = MultipleLocator(0.05)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax21.yaxis.set_minor_locator(minorLocator)
    ax31.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.005)
    ax22.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.1)
    ax32.yaxis.set_minor_locator(minorLocator)

    for ax in [ax31,ax32]:
        ax.set_xlabel(r'\boldmath$x$' ,size=30)
        ax.xaxis.set_label_coords(0.80,0.00)

    ax11.text(0.78 ,0.70  ,r'\boldmath{$xu_{v}$}'            , transform=ax11.transAxes,size=40)
    ax11.text(0.53 ,0.15  ,r'\boldmath{$xd_{v}$}'            , transform=ax11.transAxes,size=40)
    ax12.text(0.65 ,0.25  ,r'\boldmath{$xg/10$}'             , transform=ax12.transAxes,size=40)
    ax21.text(0.10 ,0.20  ,r'\boldmath{$x(\bar{d}+\bar{u})$}', transform=ax21.transAxes,size=40)
    ax22.text(0.20 ,0.10  ,r'\boldmath{$x(\bar{d}-\bar{u})$}', transform=ax22.transAxes,size=40)
    ax31.text(0.50 ,0.40  ,r'\boldmath{$x(s+\bar{s})$}'      , transform=ax31.transAxes,size=40)
    ax32.text(0.05 ,0.05  ,r'\boldmath{$R_s$}'               , transform=ax32.transAxes,size=40)

    ax12.text(0.05,0.08,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax12.transAxes,size=30)

    ax21.axhline(0.0,ls=':',color='black',alpha=0.5)
    ax22.axhline(0.0,ls=':',color='black',alpha=0.5)
    ax32.axvline(0.2,ls=':',color='black',alpha=0.5)

    handles, labels = [],[]
    handles.append(thy[0])
    labels.append(r'\textrm{\textbf{JAM}}')
    ax11.legend(handles,labels,loc='upper left',  fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    if len(SETS) > 0:

        handles = [hand['CJ15'],hand['JAM20']]  
        label1 = r'\textbf{\textrm{CJ15}}'
        label2 = r'\textbf{\textrm{JAM20}}'
        labels = [label1,label2]
        ax12.legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

        handles = [hand['NNPDF'],hand['ABMP16']]  
        label1  = r'\textbf{\textrm{NNPDF3.1}}'
        label2  = r'\textbf{\textrm{ABMP16}}'
        labels = [label1,label2]
        ax21.legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

        handles = [hand['MMHT'],hand['CT18']]  
        label1  = r'\textbf{\textrm{MMHT}}'
        label2  = r'\textbf{\textrm{CT18}}'
        labels = [label1,label2]
        ax31.legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.20)

    filename+='.png'
    #filename+='.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    ax31.set_rasterized(True)
    ax32.set_rasterized(True)

    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_xf_du(WDIR,kc,SETS):

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)

    filename = '%s/gallery/PDFs-ratio'%cwd

    j = 0
    hand = {}
    thy = {}
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()

        Q2 = 10
        data=load('%s/data/pdf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))
            
        replicas=core.get_replicas(wdir)
        cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
        best_cluster=cluster_order[0]


        for flav in data['XF']:
            X=data['X']
            mean = np.mean(data['XF'][flav],axis=0)
            std = np.std(data['XF'][flav],axis=0)

            if   flav=='d/u'  : ax = ax11
            elif flav=='db/ub': ax = ax12
            else: continue

            #--plot average and standard deviation
            thy[j]  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=0.9,zorder=5)

            #--plot other PDF sets
            if j==0:
                for SET in SETS:
                    _SET, _color, _alpha = SETS[SET][0], SETS[SET][1], SETS[SET][2]

                    if flav=='db/ub':
                        X = X[:300]

                    pdf = _SET.get_xpdf(flav,X,Q2)

                    hand[SET] = ax.fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,zorder=1)

        j+=1


    for ax in [ax11,ax12]:
          ax.set_xlabel(r'\boldmath$x$'    ,size=30)
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
          ax.xaxis.set_label_coords(0.95,0.00)

    ax11.set_xlim(0,0.95)
    ax11.set_xticks([0,0.2,0.4,0.6,0.8])

    ax12.set_xlim(0,0.40)
    ax12.set_xticks([0,0.1,0.2,0.3])

    ax11.set_ylim(0,1.0)
    ax11.set_yticks([0.2,0.4,0.6,0.8,1.0])

    ax12.set_ylim(0,1.9)
    ax12.set_yticks([0.5,1.0,1.5])


    minorLocator = MultipleLocator(0.05)
    ax11.xaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_minor_locator(minorLocator)

    minorLocator = MultipleLocator(0.02)
    ax12.xaxis.set_minor_locator(minorLocator)

    minorLocator = MultipleLocator(0.1)
    ax12.yaxis.set_minor_locator(minorLocator)

    ax11.text(0.10 ,0.10  ,r'\boldmath{$d/u$}'             ,transform=ax11.transAxes,size=40)
    ax12.text(0.50 ,0.10  ,r'\boldmath{$\bar{d}/\bar{u}$}' ,transform=ax12.transAxes,size=40)

    ax11.text(0.35,0.52,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=30)

    handles1,labels1 = [],[]
    handles1.append(thy[0])
    labels1.append(r'\textrm{\textbf{JAM}}')
    handles2,labels2 = [],[]

    if len(SETS) > 0:
        handles1.append(hand['CJ15'])
        handles1.append(hand['JAM20'])
        handles1.append(hand['NNPDF'])
        handles2.append(hand['ABMP16'])
        handles2.append(hand['MMHT'])
        handles2.append(hand['CT18'])

        labels1.append(r'\textbf{\textrm{CJ15}}')
        labels1.append(r'\textbf{\textrm{JAM20}}')
        labels1.append(r'\textbf{\textrm{NNPDF3.1}}')
        labels2.append(r'\textbf{\textrm{ABMP16}}')
        labels2.append(r'\textbf{\textrm{MMHT}}')
        labels2.append(r'\textbf{\textrm{CT18}}')

    if len(handles1) <= 2: loc = (0.20,0.80)
    if len(handles1) >  2: loc = (0.10,0.65)

    ax11.legend(handles1,labels1,loc=loc,           fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 2, columnspacing = 0.5)
    ax12.legend(handles2,labels2,loc='lower left',  fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)


    py.tight_layout()

    filename+='.png'
    #filename+='.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)


if __name__ == "__main__":

    #wdir1 = 'results/marathon/step30'
    wdir1 = 'step12'

    WDIR = [wdir1]

    #--get PDF sets for comparison
    SETS = {}
    SETS['CJ15']   = (QPDCALC('CJ15nlo'                   ,ismc=False)                    ,'green'    ,0.4) 
    SETS['JAM20']  = (QPDCALC('JAM20-SIDIS_PDF_proton_nlo',ismc=True ,central_only=False) ,'magenta'  ,0.2)
    SETS['ABMP16'] = (QPDCALC('ABMP16_3_nlo'              ,ismc=False)                    ,'darkblue' ,0.5)
    SETS['NNPDF']  = (QPDCALC('NNPDF31_nlo_as_0118'       ,ismc=True ,central_only=False) ,'gold'     ,0.5)
    SETS['MMHT']   = (QPDCALC('MMHT2014nlo68cl'           ,ismc=False)                    ,'cyan'     ,0.3)
    SETS['CT18']   = (QPDCALC('CT18NNLO'                  ,ismc=True ,central_only=False) ,'gray'     ,0.5)

    plot_xf_main(WDIR,kc,SETS)
    plot_xf_du  (WDIR,kc,SETS)




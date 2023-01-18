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
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
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

cwd = 'plots/thesis'

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
    handles.append(hand['NNPDF'])
    handles.append(hand['ABMP16'])
    labels.append(r'\textrm{\textbf{JAM}}')
    labels .append(r'\textbf{\textrm{NNPDF4.0}}')
    labels .append(r'\textbf{\textrm{ABMP16}}')
    ax12.legend(handles,labels,loc=(0.48,0.45), fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    handles,labels = [],[]
    handles.append(hand['CJ15'])
    handles.append(hand['CT18'])
    handles.append(hand['MSHT20'])
    labels .append(r'\textbf{\textrm{CJ15}}')
    labels .append(r'\textbf{\textrm{CT18}}')
    labels .append(r'\textbf{\textrm{MSHT20}}')
    ax22.legend(handles,labels,loc=(0.48,0.45), fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.15, right = 0.97, top = 0.97)

    filename+='.png'
    #filename+='.pdf'
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

    filename = '%s/gallery/PDFs-asymmetry'%cwd

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
    labels.append(r'\textbf{\textrm{NNPDF4.0}}')
    labels.append(r'\textbf{\textrm{CJ15}}')

    ax11.legend(handles,labels,loc=(0.02,0.01),  fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    handles,labels = [],[]
    handles.append(hand['MSHT20'])
    handles.append(hand['ABMP16'])
    handles.append(hand['CT18'])
    labels .append(r'\textbf{\textrm{MSHT20}}')
    labels.append(r'\textbf{\textrm{ABMP16}}')
    labels.append(r'\textbf{\textrm{CT18}}')
    ax21.legend(handles,labels,loc='upper right',  fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.20)

    filename+='.png'
    #filename+='.pdf'
    ax11.set_rasterized(True)
    ax21.set_rasterized(True)
    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_impact(WDIR,kc):

    nrows,ncols=5,1
    fig = py.figure(figsize=(ncols*12,nrows*3))
    ax11=py.subplot(nrows,ncols,(1,2))
    ax21=py.subplot(nrows,ncols,(3,4))
    ax31=py.subplot(nrows,ncols,5)

    filename = '%s/gallery/SQ-impact'%cwd

    j = 0
    hand = {}
    thy = {}
    STD = {}
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

            if flav=='db/ub':   ax = ax11
            elif flav=='db-ub': ax = ax21
            else: continue

            #--plot average and standard deviation
            #if j == 0: 
            #    #thy[j]  = ax.fill_between(X,mean-std,mean+std,color='gold',alpha=0.3,zorder=4)
            #    thy[j]  ,= ax.plot(X,mean+std,color='lightgreen',alpha=0.9,ls = '--',zorder=4)
            #    ax           .plot(X,mean-std,color='lightgreen',alpha=0.9,ls = '--',zorder=4)
            if j == 0: thy[j]  = ax.fill_between(X,mean-std,mean+std,color='lightgreen',alpha=0.9,zorder=4)
            if j == 1: thy[j]  = ax.fill_between(X,mean-std,mean+std,color='blue'      ,alpha=0.7,zorder=5)
            if j == 2: thy[j]  = ax.fill_between(X,mean-std,mean+std,color='red'       ,alpha=1.0,zorder=6)

            if flav=='db/ub': STD[j] = std

        j+=1


    for ax in [ax11,ax21,ax31]:
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=40,length=5)
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=40,length=10)
        ax.set_xlim(0.02,0.42)
        ax.set_xticks([0.1,0.2,0.3,0.4])
        minorLocator = MultipleLocator(0.02)
        ax.xaxis.set_minor_locator(minorLocator)

    ax11.tick_params(labelbottom=False)
    ax21.tick_params(labelbottom=False)

    ax11.set_ylim(0,1.7)
    ax11.set_yticks([0.5,1.0,1.5])

    ax21.set_ylim(-0.005,0.048)
    ax21.set_yticks([0.00,0.02,0.04])
    ax21.set_yticklabels([r'$0$',r'$0.02$',r'$0.04$'])

    minorLocator = MultipleLocator(0.1)
    ax11.yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.01)
    ax21.yaxis.set_minor_locator(minorLocator)

    ax11.text(0.45,0.10,r'\boldmath$\bar{d}/\bar{u}$'    ,transform=ax11.transAxes,size=70)
    ax21.text(0.57,0.80,r'\boldmath$x(\bar{d}-\bar{u})$' ,transform=ax21.transAxes,size=70)

    ax21.text(0.03,0.14,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax21.transAxes,size=40)

    ax11.axhline(1.0,0,1,ls=':',color='black',alpha=0.5,zorder=5)
    ax21.axhline(0.0,0,1,ls=':',color='black',alpha=0.5,zorder=5)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]


    handles,labels = [],[]
    #handles.append(('darkgreen',0.9,'--'))
    handles.append(thy[0])
    handles.append(thy[1])
    handles.append(thy[2])
    labels.append(r'\textrm{\textbf{baseline}}')
    labels.append(r'\textrm{\textbf{+STAR}}')
    labels.append(r'\textrm{\textbf{+SeaQuest}}')

    ax11.legend(handles,labels,loc=(0.00,0.02),fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 0.95, ncol = 1, columnspacing = 0.5, handler_map = {tuple:AnyObjectHandler()})

    #--plot ratios
    hand = {}
    X=data['X']
    hand['RHIC'] ,= ax31.plot(X,(STD[1]/STD[0]),color='blue',alpha=0.7,lw=3)
    hand['SQ']   ,= ax31.plot(X,(STD[2]/STD[0]),color='red' ,alpha=1.0,lw=3)

    ax31.text(0.05,0.12,r'\boldmath$\delta/\delta_{\rm baseline}$' ,transform=ax31.transAxes,size=70)
    #ax31.text(0.60,0.08,r'\boldmath$(\bar{d}/\bar{u})$'            ,transform=ax31.transAxes,size=30)

    ax31.axhline(1.0,0,1,ls=':',color='black',alpha=0.5)

    ax31.set_xlabel(r'\boldmath$x$'    ,size=60)
    ax31.xaxis.set_label_coords(0.82,0.00)

    ax31.set_ylim(0,1.2)
    ax31.set_yticks([0.50,1.00])

    minorLocator = MultipleLocator(0.1)
    ax31.yaxis.set_minor_locator(minorLocator)

    handles,labels = [],[]
    handles.append(hand['RHIC'])
    handles.append(hand['SQ'])
    labels.append(r'\textrm{\textbf{+STAR}}')
    labels.append(r'\textrm{\textbf{+SeaQuest}}')

    #ax31.legend(handles,labels,loc=(0.00,-0.02),fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()

    py.subplots_adjust(hspace=0,wspace=0)

    filename+='.png'
    #filename+='.pdf'
    ax11.set_rasterized(True)
    ax21.set_rasterized(True)
    ax31.set_rasterized(True)
    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_history(WDIR,kc):

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*12,nrows*9))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)

    filename = '%s/gallery/PDFs-history'%cwd

    M = 1
    if M == 1: r = (-0.2,0.2)
    if M == 2: r = (-0.02,0.02)

    j = 0
    hand = {}
    thy = {}

    PLOT1 = [('yellow',0.2),('lightgray',0.8),('lightgreen',0.9),('blue',0.7),('red',1.0)]
    PLOT2 = [('yellow',0.2),('lightgray',0.8),('lightgreen',0.9),('blue',0.7),('red',1.0)]
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

            if flav != 'db-ub': continue

            #--plot average and standard deviation of asymmetry
            if j == 0: 
                thy[j]  = ax11.fill_between(X,mean-std,mean+std,color=PLOT1[j][0],alpha=PLOT1[j][1],zorder=1)
            if j == 1: 
                thy[j]  = ax11.fill_between(X,mean-std,mean+std,color=PLOT1[j][0],alpha=PLOT1[j][1],zorder=3)
            if j == 2: 
                thy[j]  = ax11.fill_between(X,mean-std,mean+std,color=PLOT1[j][0],alpha=PLOT1[j][1],zorder=4)
            if j == 3: 
                thy[j]  = ax11.fill_between(X,mean-std,mean+std,color=PLOT1[j][0],alpha=PLOT1[j][1],zorder=5)
            if j == 4: 
                thy[j]  = ax11.fill_between(X,mean-std,mean+std,color=PLOT1[j][0],alpha=PLOT1[j][1],zorder=6)

        #--plot histogram of integrated asymmetry
        data=load('%s/data/pdf-moment-%d-Q2%d.dat'%(wdir,M,int(Q2)))

        for flav in data['moments']:
            X=data['X']
            if flav != 'db-ub': continue

            moment = []
            for i in range(len(data['moments'][flav])):
                moment.append(data['moments'][flav][i][0])

            moment = np.array(moment)

            #--plot histogram of moments
            if j == 0: 
                       ax12.hist(moment,density=True,range=r,bins=30,alpha=PLOT2[j][1],color=PLOT2[j][0],zorder=1) 
                       ax12.hist(moment,density=True,range=r,bins=30,alpha=1.0        ,color=PLOT2[j][0],zorder=6,histtype='step',lw=3.0) 
            if j == 1: ax12.hist(moment,density=True,range=r,bins=30,alpha=PLOT2[j][1],color=PLOT2[j][0],zorder=2) 
            if j == 2: ax12.hist(moment,density=True,range=r,bins=30,alpha=PLOT2[j][1],color=PLOT2[j][0],zorder=3) 
            if j == 3: ax12.hist(moment,density=True,range=r,bins=30,alpha=PLOT2[j][1],color=PLOT2[j][0],zorder=4) 
            if j == 4: ax12.hist(moment,density=True,range=r,bins=30,alpha=PLOT2[j][1],color=PLOT2[j][0],zorder=5) 

        j+=1

  

    for ax in [ax11]:
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=40,length=5)
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=40,length=10)
        ax.set_xlim(0.01,0.50)
        ax.semilogx()
        ax.set_xticks([0.01,0.1])
        ax.set_xticklabels([r'$0.01$',r'$0.1$'])
        ax.set_ylim(-0.023,0.088)
        ax.set_yticks([0.00,0.02,0.04,0.06,0.08])
        ax.set_yticklabels([r'$0$',r'$0.02$',r'$0.04$',r'$0.06$',r'$0.08$'])
        minorLocator = MultipleLocator(0.01)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.axhline(0.0,0,1,ls=':',color='black',alpha=0.5,zorder=5)

    for ax in [ax12]:
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=40,length=5)
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=40,length=10)
        ax.tick_params(left=False,labelleft=False)


    ax11.text(0.05,0.06,r'\boldmath$x(\bar{d}-\bar{u})$' ,transform=ax11.transAxes,size=60)

    ax11.text(0.71,0.60,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=35)

    ax12.set_ylabel(r'\textrm{\textbf{Normalized Yield}}', size=45)

    if M==1: ax12.text(0.02,0.30,r'\boldmath$\int_{0.01}^{1} {\rm d}x (\bar{d} - \bar{u})$',  transform=ax12.transAxes,size=60)
    if M==2: ax12.text(0.02,0.30,r'\boldmath$\int_{0.01}^{1} {\rm d}x~x(\bar{d} - \bar{u})$', transform=ax12.transAxes,size=60)
    #ax12.text(0.02,0.20,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$',               transform=ax12.transAxes,size=35)

    ax11.set_xlabel(r'\boldmath$x$'    ,size=60)
    ax11.xaxis.set_label_coords(0.95,0.00)

    class AnyObjectHandler(HandlerBase):

        def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
            l1 = py.Line2D([x0,y0+width], [0.8*height,0.8*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            l2 = py.Line2D([x0,y0+width], [0.2*height,0.2*height], color = orig_handle[0], alpha = orig_handle[1], ls = orig_handle[2])
            return [l1,l2]

    handles,labels = [],[]
    handles.append(thy[0])
    handles.append(thy[1])
    handles.append(thy[2])
    handles.append(thy[3])
    handles.append(thy[4])
    labels.append(r'\textrm{\textbf{DIS~(no NMC)}}')
    labels.append(r'\textrm{\textbf{+NMC}}')
    labels.append(r'\textrm{\textbf{+\boldmath$W/Z$/jet}}')
    labels.append(r'\textrm{\textbf{+NuSea}}')
    labels.append(r'\textrm{\textbf{+STAR/SeaQuest}}')

    ax11.legend(handles,labels,loc='upper left',fontsize = 32, frameon = 0, handletextpad = 0.3, handlelength = 0.85, ncol = 2, columnspacing = 0.5, handler_map = {tuple:AnyObjectHandler()})

    handles = [] 

    handles.extend([Rectangle((0,0),1,1,color=p[0],alpha=p[1]) for p in PLOT2])

    ax12.legend(handles,labels,loc='upper left',fontsize = 32, frameon = 0, handletextpad = 0.3, handlelength = 0.85, ncol = 1, columnspacing = 0.5)


    py.tight_layout()

    py.subplots_adjust(hspace=0.15,wspace=0.08)

    filename+='.png'
    #filename+='.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

if __name__ == "__main__":

    wdir1 = 'results/sq/final'

    #--PLOT PDFS AND ASYMMETRY
    
    WDIR = [wdir1]


    #--get PDF sets for comparison
    SETS = {}
    SETS['CJ15']    = (QPDCALC('CJ15nlo'                   ,ismc=False)                    ,'green'    ,0.4 ,2) 
    SETS['ABMP16']  = (QPDCALC('ABMP16_3_nlo'              ,ismc=False)                    ,'darkblue' ,0.5 ,6)
    SETS['NNPDF']   = (QPDCALC('NNPDF40_nnlo_as_01180'     ,ismc=True ,central_only=False) ,'gold'     ,0.5 ,1)
    SETS['MSHT20']  = (QPDCALC('MSHT20nnlo_as118'          ,ismc=False)                    ,'cyan'     ,0.3 ,2)
    SETS['CT18']    = (QPDCALC('CT18NNLO'                  ,ismc=True ,central_only=False) ,'gray'     ,0.5 ,2)

    Q2 = 10

    plot_xf_main (WDIR,kc,SETS,Q2)
    plot_xf_ratio(WDIR,kc,SETS,Q2)


    #--PLOT SEAQUEST AND STAR IMPACT

    #--plot iteratively adding RHIC and SQ
    wdir0 = 'results/upol/step13'
    wdir1 = 'results/upol/step14'
    wdir2 = 'results/sq/final'

    WDIR = [wdir0,wdir1,wdir2]

    plot_impact(WDIR,kc)


    #--PLOT HISTORY OF ASYMMETRY

    #--plot starting from very basic baseline
    wdir0 = 'results/sq/noNMC'    #--DIS only, no NMC
    wdir1 = 'results/sq/DISonly'  #--+NMC
    wdir2 = 'results/sq/baseline' #--+Jets, Z, lepton
    wdir3 = 'results/upol/step13' #--+E866
    wdir4 = 'results/sq/final'    #--+RHIC and SeaQuest

    WDIR = [wdir0,wdir1,wdir2,wdir3,wdir4]

    plot_history(WDIR,kc)




















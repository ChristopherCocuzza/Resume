import sys,os
import numpy as np
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py

#--from scipy stack 
from scipy.integrate import quad, cumtrapz

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

import warnings
warnings.filterwarnings("ignore")

FLAV=[]
FLAV.append('g')
FLAV.append('u')
FLAV.append('d')
FLAV.append('uv')
FLAV.append('dv')
FLAV.append('d/u')
FLAV.append('db')
FLAV.append('ub')
FLAV.append('db/ub')
FLAV.append('db+ub')
FLAV.append('db-ub')
FLAV.append('s')
FLAV.append('sb')
FLAV.append('s+sb')
FLAV.append('s-sb')
FLAV.append('Rs')

cmap = matplotlib.cm.get_cmap('plasma')

def gen_xf(wdir,Q2):
    
    print('\ngenerating pdf at Q2 = %s from %s'%(Q2,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics
    X=10**np.linspace(-6,-1,200)
    X=np.append(X,np.linspace(0.101,0.99,200))

    pdf.evolve(Q2)

    #--compute XF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in XF:  XF[flav]=[]
            if flav=='uv':
                 func=lambda x: pdf.get_xF(x,Q2,'uv')
            elif flav=='dv':
                 func=lambda x: pdf.get_xF(x,Q2,'dv')
            elif flav=='d/u':
                 func=lambda x: pdf.get_xF(x,Q2,'d')/pdf.get_xF(x,Q2,'u')
            elif flav=='db+ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') + pdf.get_xF(x,Q2,'ub')
            elif flav=='db-ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') - pdf.get_xF(x,Q2,'ub')
            elif flav=='db/ub':
                 func=lambda x: pdf.get_xF(x,Q2,'db') / pdf.get_xF(x,Q2,'ub')
            elif flav=='s+sb':
                 func=lambda x: pdf.get_xF(x,Q2,'s') + pdf.get_xF(x,Q2,'sb')
            elif flav=='s-sb':
                 func=lambda x: pdf.get_xF(x,Q2,'s') - pdf.get_xF(x,Q2,'sb')
            elif flav=='Rs':
                 func=lambda x: (pdf.get_xF(x,Q2,'s') + pdf.get_xF(x,Q2,'sb'))\
                                /(pdf.get_xF(x,Q2,'db') + pdf.get_xF(x,Q2,'ub'))
            else:
                 func=lambda x: pdf.get_xF(x,Q2,flav) 

            XF[flav].append(np.array([func(x) for x in X]))

    print() 
    checkdir('%s/data'%wdir)
    filename='%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_xf_main(wdir,Q2,mode=0,SETS={}):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=3,2
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*8,nrows*4))
    axs,axLs = {},{}
    for i in range(N):
        axs[i+1] = py.subplot(nrows,ncols,i+1)
        divider = make_axes_locatable(axs[i+1])
        axLs[i+1] = divider.append_axes("right",size=3.10,pad=0,sharey=axs[i+1])
        axLs[i+1].set_xlim(0.1,0.9)
        axLs[i+1].spines['left'].set_visible(False)
        axLs[i+1].yaxis.set_ticks_position('right')
        py.setp(axLs[i+1].get_xticklabels(),visible=True)

        axs[i+1].spines['right'].set_visible(False)
    nrows,ncols=3,2

    hand = {}
    thy  = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if Q2==None: Q2 = 1.27**2

    scale = classifier.get_scale(wdir)

    filename='%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,Q2)
        data=load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]

    X=data['X']

    for flav in data['XF']:
        mean = np.mean(data['XF'][flav],axis=0)
        std  = np.std (data['XF'][flav],axis=0)

        if flav=='uv' or flav=='dv': ax,axL = axs[1],axLs[1]
        elif flav=='g':              ax,axL = axs[2],axLs[2]
        elif flav=='db+ub':          ax,axL = axs[3],axLs[3]
        elif flav=='db-ub':          ax,axL = axs[4],axLs[4]
        elif flav=='s+sb':           ax,axL = axs[5],axLs[5]
        elif flav=='Rs':             ax,axL = axs[6],axLs[6]
        else: continue

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF'][flav])):
                #--if plotting one step, use clusters
                if flav=='g': data['XF'][flav][i] /= 10.0

                thy ,= ax.plot(X,np.array(data['XF'][flav][i]),color=cmap(scale[i]),alpha=0.3)
                axL      .plot(X,np.array(data['XF'][flav][i]),color=cmap(scale[i]),alpha=0.3)
 
        #--plot average and standard deviation
        if mode==1:
            if flav=='g':
                mean /= 10.0
                std  /= 10.0

            where = [1 for i in range(len(X))]
            if flav=='Rs':
                where = []
                for x in X:
                    if x < 0.2: where.append(1)
                    if x > 0.2: where.append(0)

            thy  = ax.fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=5,where=np.array(where))
            axL      .fill_between(X,(mean-std),(mean+std),color='red',alpha=1.0,zorder=5,where=np.array(where))

        #--plot other PDF sets
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

            hand[SET] = ax.fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,where=np.array(where),zorder=1)
            axL           .fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,where=np.array(where),zorder=1)


    for i in range(N):
        axs[i+1].set_xlim(1e-3,0.1)
        axs[i+1].semilogx()

        axs[i+1].tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
        axs[i+1].tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
        axs[i+1].tick_params(axis='x',    which='major', pad = 8)
        axs[i+1].set_xticks([0.001,0.01,0.1])
        axs[i+1].set_xticklabels([r'$10^{-3}$',r'$10^{-2}$',r'$0.1$'])

        axLs[i+1].set_xlim(0.1,1.0)

        axLs[i+1].tick_params(axis='both', which='major', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=8)
        axLs[i+1].tick_params(axis='both', which='minor', top=True, right=True, left=False, labelright=False, direction='in',labelsize=30,length=4)
        axLs[i+1].tick_params(axis='x',    which='major', pad = 8)
        axLs[i+1].set_xticks([0.3,0.5,0.7])
        axLs[i+1].set_xticklabels([r'$0.3$',r'$0.5$',r'$0.7$'])

    for i in [1,2,3,4]:
        axs[i] .tick_params(labelbottom=False)
        axLs[i].tick_params(labelbottom=False)

    axs[1].set_ylim(0,0.7)
    axs[2].set_ylim(0,0.8)
    axs[3].set_ylim(-0.05,1.0)
    axs[4].set_ylim(-0.09,0.09)
    axs[5].set_ylim(0,0.9)
    axs[6].set_ylim(0,1.2)

    axs[1].set_yticks([0.2,0.4,0.6])
    axs[2].set_yticks([0.2,0.4,0.6])
    axs[3].set_yticks([0,0.2,0.4,0.6,0.8])
    axs[4].set_yticks([-0.08,-0.04,0,0.04,0.08])
    axs[5].set_yticks([0.2,0.4,0.6,0.8])
    axs[6].set_yticks([0.5,1.0])

    for i in range(N):
        axs [i+1].axvline(0.1,color='k',linestyle=':' ,alpha=0.5)
        axLs[i+1].axvline(0.1,color='k',linestyle=':' ,alpha=0.5)

    axs [3].axhline(0.0,ls='--',color='black',alpha=0.5)
    axs [4].axhline(0.0,ls='--',color='black',alpha=0.5)
    axLs[3].axhline(0.0,ls='--',color='black',alpha=0.5)
    axLs[4].axhline(0.0,ls='--',color='black',alpha=0.5)

    axLs[5].set_xlabel(r'\boldmath$x$',size=40)
    axLs[6].set_xlabel(r'\boldmath$x$',size=40)   
    axLs[5].xaxis.set_label_coords(0.92,0.00)
    axLs[6].xaxis.set_label_coords(0.92,0.00)

    axLs[1].text(0.50 ,0.50  ,r'\boldmath{$xu_{v}$}'            , transform=axLs[1].transAxes,size=30)
    axLs[1].text(0.01 ,0.20  ,r'\boldmath{$xd_{v}$}'            , transform=axLs[1].transAxes,size=30)
    axLs[2].text(0.05 ,0.25  ,r'\boldmath{$xg/10$}'             , transform=axLs[2].transAxes,size=30)
    axs[3] .text(0.10 ,0.10  ,r'\boldmath{$x(\bar{d}+\bar{u})$}', transform=axs[3] .transAxes,size=30)
    axLs[4].text(0.20 ,0.10  ,r'\boldmath{$x(\bar{d}-\bar{u})$}', transform=axLs[4].transAxes,size=30)
    axs[5] .text(0.10 ,0.10  ,r'\boldmath{$x(s+\bar{s})$}',       transform=axs[5] .transAxes,size=30)
    axs[6] .text(0.10 ,0.10  ,r'\boldmath{$R_s$}',                transform=axs[6] .transAxes,size=30)

    if Q2 == 1.27**2: axs[2].text(0.05,0.08,r'$Q^2 = m_c^2$'                                  , transform=axs[2].transAxes,size=30)
    else:             axs[2].text(0.05,0.08,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=axs[2].transAxes,size=30)

    minorLocator = MultipleLocator(0.05)
    axs[1].yaxis.set_minor_locator(minorLocator)
    axs[2].yaxis.set_minor_locator(minorLocator)
    axs[3].yaxis.set_minor_locator(minorLocator)
    axs[5].yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.005)
    axs[4].yaxis.set_minor_locator(minorLocator)
    minorLocator = MultipleLocator(0.1)
    axs[6].yaxis.set_minor_locator(minorLocator)

    if len(SETS) > 0:

        handles = [hand['CJ15'],hand['JAM20']]  
        label1 = r'\textbf{\textrm{CJ15}}'
        label2 = r'\textbf{\textrm{JAM20}}'
        labels = [label1,label2]
        axLs[2].legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

        handles = [hand['NNPDF'],hand['ABMP16']]  
        label1  = r'\textbf{\textrm{NNPDF3.1}}'
        label2  = r'\textbf{\textrm{ABMP16}}'
        labels = [label1,label2]
        axLs[2].legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

        handles = [hand['MMHT'],hand['CT18']]  
        label1  = r'\textbf{\textrm{MMHT}}'
        label2  = r'\textbf{\textrm{CT18}}'
        labels = [label1,label2]
        axLs[5].legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

    if mode==0:
        sm   = py.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.77,0.95,0.20,0.02])
        cax.tick_params(axis='both',which='both',labelsize=20,direction='in')
        cax.xaxis.set_label_coords(0.65,-1.4)
        cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
        cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=30)

    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.20)

    filename = '%s/gallery/pdfs-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_xf_du(wdir,Q2,mode=0,SETS={}):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)

    hand = {}
    thy = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    scale = classifier.get_scale(wdir)

    #--load data if it exists
    filename='%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,Q2)
        data=load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]


    for flav in data['XF']:
        X=data['X']
        mean = np.mean(data['XF'][flav],axis=0)
        std  = np.std (data['XF'][flav],axis=0)

        if   flav=='d/u'  : ax = ax11
        elif flav=='db/ub': ax = ax12
        else: continue

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i],color=cmap(scale[i]),alpha=0.3)
      
        #--plot average and standard deviation
        if mode==1:

            thy  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=1.0,zorder=5)

        #--plot other PDF sets
        for SET in SETS:
            _SET, _color, _alpha = SETS[SET][0], SETS[SET][1], SETS[SET][2]

            if flav=='db/ub':
                X = X[:250]

            pdf = _SET.get_xpdf(flav,X,Q2)

            hand[SET] = ax.fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,zorder=1)


    for ax in [ax11,ax12]:
          ax.set_xlabel(r'\boldmath$x$'    ,size=40)
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

    ax12.axhline(1.0,ls='--',color='black',alpha=0.5)

    minorLocator = MultipleLocator(0.05)
    ax11.xaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_minor_locator(minorLocator)

    minorLocator = MultipleLocator(0.02)
    ax12.xaxis.set_minor_locator(minorLocator)

    minorLocator = MultipleLocator(0.1)
    ax12.yaxis.set_minor_locator(minorLocator)

    ax11.text(0.10 ,0.10  ,r'\boldmath{$d/u$}'             ,transform=ax11.transAxes,size=30)
    ax12.text(0.05 ,0.85  ,r'\boldmath{$\bar{d}/\bar{u}$}' ,transform=ax12.transAxes,size=30)

    if Q2 == 1.27**2: ax11.text(0.50,0.60,r'$Q^2 = m_c^2$'                                  , transform=ax11.transAxes,size=30)
    else:             ax11.text(0.50,0.60,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=30)

    if mode==1:
        if len(SETS) > 0:
            handles1, handles2 = [],[]
            labels1, labels2 = [],[]
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
            if len(handles1) >  2: loc = (0.20,0.70)

            ax11.legend(handles1,labels1,loc=loc,           fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 2, columnspacing = 0.5)
            ax12.legend(handles2,labels2,loc='lower left',  fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    if mode==0:
        sm   = py.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.56,0.28,0.20,0.05])
        cax.tick_params(axis='both',which='both',labelsize=20,direction='in')
        cax.xaxis.set_label_coords(0.65,-1.2)
        cbar = py.colorbar(sm,cax=cax,orientation='horizontal',ticks=[0.2,0.4,0.6,0.8])
        cbar.set_label(r'\boldmath${\rm scaled}~\chi^2_{\rm red}$',size=30)

    py.tight_layout()

    filename = '%s/gallery/pdfs-du-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_xf_highx(wdir,Q2,mode=0,SETS={}):
    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*6))
    ax11=py.subplot(nrows,ncols,1)

    hand = {}
    thy = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    #--load data if it exists
    filename='%s/data/pdf-Q2=%3.5f.dat'%(wdir,Q2)
    #--load data if it exists
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        gen_xf(wdir,Q2)
        data=load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]


    for flav in data['XF']:
        X=data['X']
        mean = np.mean(data['XF'][flav],axis=0)
        std  = np.std (data['XF'][flav],axis=0)

        if   flav=='d/u':   ax,color,zorder = ax11,'green', 1
        elif flav=='u':     ax,color,zorder = ax11,'red'  , 3
        elif flav=='d':     ax,color,zorder = ax11,'blue' , 2
        else: continue

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF'][flav])):
                ax.plot(X,data['XF'][flav][i],color=color,alpha=0.3)
      
        #--plot average and standard deviation
        if mode==1:

            thy  = ax.fill_between(X,mean-std,mean+std,color=color,alpha=1.0,zorder=zorder)

        #--plot other PDF sets
        for SET in SETS:
            _SET, _color, _alpha = SETS[SET][0], SETS[SET][1], SETS[SET][2]

            if flav=='db/ub':
                X = X[:250]

            pdf = _SET.get_xpdf(flav,X,Q2)

            hand[SET] = ax.fill_between(X,pdf['xfmin'],pdf['xfmax'],color=_color,alpha=_alpha,zorder=1)


    for ax in [ax11]:
          ax.semilogy()
          ax.set_xlabel(r'\boldmath$x$'    ,size=40)
          ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
          ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
          ax.xaxis.set_label_coords(0.95,0.00)

    ax11.set_xlim(0.1,0.95)
    ax11.set_xticks([0.2,0.4,0.6,0.8])

    ax11.set_ylim(1e-4,1.0)
    #ax11.set_yticks([0.2,0.4,0.6,0.8,1.0])

    #ax11.axhline(1.0,ls='--',color='black',alpha=0.5)

    minorLocator = MultipleLocator(0.05)
    ax11.xaxis.set_minor_locator(minorLocator)
    #ax11.yaxis.set_minor_locator(minorLocator)

    ax11.text(0.35 ,0.92  ,r'\boldmath{$u$}'             ,color='red'  ,transform=ax11.transAxes,size=30)
    ax11.text(0.10 ,0.75  ,r'\boldmath{$d$}'             ,color='blue' ,transform=ax11.transAxes,size=30)
    ax11.text(0.70 ,0.87  ,r'\boldmath{$d/u$}'           ,color='green',transform=ax11.transAxes,size=30)

    if Q2 == 1.27**2: ax11.text(0.05,0.05,r'$Q^2 = m_c^2$'                                  , transform=ax11.transAxes,size=30)
    else:             ax11.text(0.05,0.05,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=30)

    if mode==1:
        if len(SETS) > 0:
            handles1, handles2 = [],[]
            labels1, labels2 = [],[]
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
            if len(handles1) >  2: loc = (0.20,0.70)

            ax11.legend(handles1,labels1,loc=loc,           fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 2, columnspacing = 0.5)
            ax11.legend(handles2,labels2,loc='lower left',  fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0, ncol = 1, columnspacing = 0.5)

    py.tight_layout()

    filename = '%s/gallery/pdfs-highx-Q2=%3.5f'%(wdir,Q2)
    if mode==1: filename += '-bands'

    filename+='.png'
    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

def plot_xf(wdir,Q2=None,mode=0,sets=False):

    #--get PDF sets for comparison
    SETS = {}
    if sets:
        SETS['CJ15']   = (QPDCALC('CJ15nlo',ismc=False),'green',0.4) 
        SETS['JAM20']  = (QPDCALC('JAM20-SIDIS_PDF_proton_nlo',ismc=True,central_only=False),'magenta',0.2)
        SETS['ABMP16'] = (QPDCALC('ABMP16_3_nlo',ismc=False),'darkblue',0.5)
        SETS['NNPDF']  = (QPDCALC('NNPDF31_nlo_as_0118',ismc=True,central_only=False),'gold',0.5)
        SETS['MMHT']   = (QPDCALC('MMHT2014nlo68cl',ismc=False),'cyan',0.3)
        SETS['CT18']   = (QPDCALC('CT18NNLO',ismc=True,central_only=False),'gray',0.5)

    
    plot_xf_main (wdir,Q2,mode=mode,SETS=SETS)
    plot_xf_du   (wdir,Q2,mode=mode,SETS=SETS)
    plot_xf_highx(wdir,Q2,mode=mode,SETS=SETS)


#--pdf moments
def gen_moments(wdir, Q2 = 10, flavors = ['quark','g'], mom = 1, xmin = 1e-5, xmax = 0.95):
    ## get truncated second moment integrated from a range of x_min to 1
    load_config('%s/input.py' % wdir)
    istep = core.get_istep()

    replicas = core.get_replicas(wdir)
    ## 'conf' will be modified for each replica individually later in the loop over 'replicas'
    ## the reason for doing this is that 'fix parameters' has to be set correctly for each replica

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
            print('pdf-proton not an active or passive distribution')
            return

    resman = RESMAN(nworkers = 1, parallel = False, datasets = False)
    parman = resman.parman
    parman.order = replicas[0]['order'][istep]
    ## make sure 'parman' uses the same order for active distributions as all the replicas do

    pdf = conf['pdf']

    ## setup kinematics
    xs = np.geomspace(xmin,0.1,100)
    xs = np.append(xs, np.linspace(0.1, xmax, 100))
    if Q2 == None: Q2 = conf['Q20']
    print('\ngenerating moment %d for pdf-proton from %s at Q2 = %3.2f from %3.2f to %3.2f' % (mom, wdir, Q2, xmin, xmax))

    ## compute moments for all replicas
    moments = {}
    n_replicas = len(replicas)

    power = mom - 2

    for i in range(n_replicas): ## using 'scipy.integrate.cumtrapz' takes about 9.984 seconds for 100 x points, 4 flavors and 516 replicas
        lprint('%d/%d' % (i + 1, n_replicas))

        parman.order = copy.copy(replicas[i]['order'][istep])
        parman.set_new_params(replicas[i]['params'][istep], initial = True)

        for flavor in flavors:
            if flavor not in moments:
                moments[flavor] = []
            if flavor == 'quark':
                func = lambda x:  x**power*(pdf.get_xF(x, Q2, 'u') + pdf.get_xF(x, Q2, 'ub') + pdf.get_xF(x, Q2, 'd') + pdf.get_xF(x, Q2, 'db') + \
                                           pdf.get_xF(x, Q2, 's') + pdf.get_xF(x, Q2, 'sb') + pdf.get_xF(x, Q2, 'c') + pdf.get_xF(x, Q2, 'cb'))
            if flavor == 'db-ub':
                func = lambda x:  x**power*(pdf.get_xF(x, Q2, 'db') - pdf.get_xF(x, Q2, 'ub'))
            else:
                func = lambda x:  x**power*pdf.get_xF(x, Q2, flavor)

            function_values = [func(_) for _ in xs]
            moment_temp = cumtrapz(function_values, xs, initial = 0.0)
            moment_temp = np.array(moment_temp)
            moment_max = moment_temp[-1]
            moments[flavor].append(moment_max - moment_temp)
    print()

    checkdir('%s/data' % wdir)
    save({'X': xs, 'Q2': Q2, 'moments': moments}, '%s/data/pdf-moment-%d-Q2%d.dat' % (wdir,mom,Q2))

#--evolution
def gen_evolution(wdir,x=0.2):
    
    print('\ngenerating pdf at x = %s from %s'%(x,wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics
    #Q2=np.linspace(conf['Q20'],100,100)
    Q2=np.geomspace(1,1000,100)

    #--compute XF for all replicas        
    XF={}

    cnt=0
    for par in replicas:
        core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        for flav in FLAV:
            if flav not in XF:  XF[flav]=[]
            if flav=='uv':
                 func=lambda q2: pdf.get_xF(x,q2,'uv')
            elif flav=='dv':
                 func=lambda q2: pdf.get_xF(x,q2,'dv')
            elif flav=='d/u':
                 func=lambda q2: pdf.get_xF(x,q2,'d')/pdf.get_xF(x,q2,'u')
            elif flav=='db+ub':
                 func=lambda q2: pdf.get_xF(x,q2,'db') + pdf.get_xF(x,q2,'ub')
            elif flav=='db-ub':
                 func=lambda q2: pdf.get_xF(x,q2,'db') - pdf.get_xF(x,q2,'ub')
            elif flav=='db/ub':
                 func=lambda q2: pdf.get_xF(x,q2,'db') / pdf.get_xF(x,q2,'ub')
            elif flav=='s+sb':
                 func=lambda q2: pdf.get_xF(x,q2,'s') + pdf.get_xF(x,q2,'sb')
            elif flav=='s-sb':
                 func=lambda q2: pdf.get_xF(x,q2,'s') - pdf.get_xF(x,q2,'sb')
            elif flav=='Rs':
                 func=lambda q2: (pdf.get_xF(x,q2,'s') + pdf.get_xF(x,q2,'sb'))\
                                /(pdf.get_xF(x,q2,'db') + pdf.get_xF(x,q2,'ub'))
            else:
                 func=lambda q2: pdf.get_xF(x,q2,flav) 

            XF[flav].append(np.array([func(q2) for q2 in Q2]))


    print() 
    checkdir('%s/data'%wdir)
    filename='%s/data/pdf-evolution-X=%3.5f.dat'%(wdir,x)

    save({'X':x,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_evolution(wdir,x=0.2,mode=0):

    #--mode 0: plot each replica
    #--mode 1: plot average and standard deviation of replicas 

    nrows,ncols=1,2
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*8,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)

    hand = {}
    thy  = {}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    scale = classifier.get_scale(wdir)

    filename='%s/data/pdf-evolution-X=%3.5f.dat'%(wdir,x)
    #--load data if it exists
    try:
        data=load(filename)
    #--generate data and then load it if it does not exist
    except:
        print('blah')
        gen_evolution(wdir,x)
        data=load(filename)
        
    replicas=core.get_replicas(wdir)
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]

    Q2=data['Q2']

    for flav in data['XF']:
        mean = np.mean(data['XF'][flav],axis=0)
        std  = np.std (data['XF'][flav],axis=0)

        if   flav=='u': ax, axL = ax11, ax12
        else: continue
        mean = np.mean(data['XF'][flav],axis=0)
        std  = np.std (data['XF'][flav],axis=0)

        #--plot each replica
        if mode==0:
            for i in range(len(data['XF'][flav])):
                thy ,= ax .plot(Q2,np.array(data['XF'][flav][i]),color=cmap(scale[i]),alpha=0.3)
                thy ,= axL.plot(Q2,np.array(data['XF'][flav][i]),color=cmap(scale[i]),alpha=0.3)
 
        #--plot average and standard deviation
        if mode==1:
            thy  = ax .fill_between(Q2,(mean-std),(mean+std),color='red',alpha=1.0,zorder=5)
            thy  = axL.fill_between(Q2,(mean-std),(mean+std),color='red',alpha=1.0,zorder=5)


    for ax in [ax11,ax12]:
        ax.set_xlim(2.0,1000)
        ax.semilogx()

        ax.tick_params(axis='both', which='major', top=True, direction='in',labelsize=30,length=8)
        ax.tick_params(axis='both', which='minor', top=True, direction='in',labelsize=30,length=4)
        ax.tick_params(axis='x',    which='major', pad = 8)
        #ax.set_xticks([0.01,0.1])
        #ax.set_xticklabels([r'$10^{-2}$',r'$0.1$'])
        ax.set_xlabel(r'\boldmath$Q^2$',size=30)
        ax.xaxis.set_label_coords(0.88,-0.02)

    ax12.semilogy()

    ax11.set_ylim(0.1,7.0)
    ax11.set_yticks([2,4,6])

    ax12.set_ylim(0.1,7.0)
    #ax12.set_yticks([2,4,6])

    minorLocator = MultipleLocator(0.5)
    ax11.yaxis.set_minor_locator(minorLocator)

    #for ax in [ax11]:
    #    ax.axvline(0.1,color='k',linestyle=':' ,alpha=0.5)
    #    ax.axvline(0.1,color='k',linestyle=':' ,alpha=0.5)

    #ax11.axhline(0.0,ls='--',color='black',alpha=0.5)
    #ax11.axhline(0.0,ls='--',color='black',alpha=0.5)

    ax11.set_ylabel(r'\boldmath$u(x,Q^2)$',size=20)

    ax11.text(0.75 ,0.20  ,r'\boldmath{$x=%s$}'%x             , transform=ax11.transAxes,size=25)


    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace = 0.20)

    filename='%s/gallery/pdf-evolution-X=%3.5f'%(wdir,x)
    if mode==1: filename += '-bands'

    filename+='.png'

    checkdir('%s/gallery'%wdir)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)




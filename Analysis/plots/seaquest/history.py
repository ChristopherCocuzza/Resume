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
from matplotlib.legend_handler import HandlerBase
from matplotlib import cm
from matplotlib.patches import Rectangle
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

cwd = 'plots/seaquest'

def plot_history(WDIR,kc):

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*12,nrows*9))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)

    filename = '%s/gallery/history'%cwd

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

    #filename+='.png'
    filename+='.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

if __name__ == "__main__":

    #--plot starting from very basic baseline
    wdir0 = 'results/sq/noNMC'    #--DIS only, no NMC
    wdir1 = 'results/sq/DISonly'  #--+NMC
    wdir2 = 'results/sq/baseline' #--+Jets, Z, lepton
    wdir3 = 'results/upol/step13' #--+E866
    wdir4 = 'results/sq/final'    #--+RHIC and SeaQuest

    WDIR = [wdir0,wdir1,wdir2,wdir3,wdir4]

    plot_history(WDIR,kc)






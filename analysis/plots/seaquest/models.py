#!/usr/bin/env python
import sys,os
import matplotlib
import copy
matplotlib.use('Agg')
import numpy as np
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--matplotlib
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py

#--from scipy stack 
from scipy.integrate import quad
from scipy.optimize import least_squares

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

from obslib.LeadingBaryon.theory import STFUNCS as PISTFUNCS
from obslib.LeadingBaryon.theory import LN

from obslib.idis.theory import THEORY

cwd = 'plots/seaquest'

covexp='%s/data/covexp'%cwd #--cov exp
covmon='%s/data/covmon'%cwd #--cov mon
PV    ='%s/data/PV'    %cwd #--Pauli-Villars
Regge ='%s/data/Regge' %cwd #--Regge cov exp
IMFexp='%s/data/IMFexp'%cwd #--IMF exp

paths = [covexp,covmon,PV,Regge,IMFexp]

def get_chi2(par):
    conf['ln'].L_p2pin = par
    nfixed = 0.447/2.0
    return 3*conf['ln'].get_fNmom()/2.0-nfixed

def gen_data():

    X=np.linspace(0.01,0.5,100)
    Q2=10

    for path in paths:
        load_config('%s/input.py'%path)
        Lambda = conf['params']['p->pi,n']['lambda']['value']
        resman=RESMAN(parallel=False,datasets=False)
        conf['ln']=LN()
        conf['ln'].L_p2pin = Lambda
        #--fit lambda to match <n> = 0.447 
        fit = least_squares(get_chi2,Lambda).x[0]
        replicas = core.get_replicas(path)
        istep=max(replicas[0]['order'].keys()) #--same for each model
        func=conf['ln'].get_dbub               #--function of x,Q2
        XF=[]
        for i in range(len(replicas)):
            #if i>10: break
            lprint('%s progress: %i/%i'%(path,i+1,len(replicas)))
            resman.parman.order=copy.copy(replicas[i]['order'][istep])
            resman.parman.set_new_params(replicas[i]['params'][istep],initial=True)
            conf['ln'].L_p2pin = fit
            XF.append([])
            for j in range(len(X)):
                XF[i].append(X[j]*func(X[j],Q2))

        filename = '%s/dbub.dat'%(path)
        save({'X':X,'Q2':Q2,'XF':XF},filename)
        print('')
    
def plot_models(WDIR,kc):

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*12,nrows*9))
    ax11=py.subplot(nrows,ncols,(1,1))

    filename = '%s/gallery/models'%cwd

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

            if flav=='db-ub':   ax = ax11
            else: continue

            thy[j]  = ax.fill_between(X,mean-std,mean+std,color='red',alpha=1.0,zorder=4)

        j+=1

    #--plot models
    X=np.linspace(0.01,0.5,100)

    mean, std = {}, {}
    UP,DO = [],[]
    for path in paths:
        data = load('%s/dbub.dat'%(path))
        mean[path] = np.mean(data['XF'],axis=0)
        std[path]  = np.std (data['XF'],axis=0)
        up = mean[path] + std[path]
        do = mean[path] - std[path]
        UP.append(up)
        DO.append(do)

    UP,DO = np.array(UP), np.array(DO)
    UP = np.max(UP,axis=0)
    DO = np.min(DO,axis=0)

    hand['models'] = ax11.fill_between(X,DO,UP,color='gold',alpha=0.7)
  
    #for path in paths:
    #    if path==IMFexp: color='gray'
    #    if path==covexp: color='b'
    #    if path==covmon: color='g'
    #    if path==Regge:  color='gold'
    #    if path==PV:     color='orange'
    #    up = mean[path] + std[path]
    #    do = mean[path] - std[path]
    #    hand[path] = ax11.fill_between(X,do,up,fc=color,alpha=0.4)

    for ax in [ax11]:
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=40,length=5)
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=40,length=10)
        ax.set_xlim(0.02,0.42)
        ax.set_xticks([0.1,0.2,0.3,0.4])
        minorLocator = MultipleLocator(0.02)
        ax.xaxis.set_minor_locator(minorLocator)

    ax11.set_ylim(0.000,0.048)
    ax11.set_yticks([0,0.01,0.02,0.03,0.04])
    ax11.set_yticklabels([r'$0$',r'$0.01$',r'$0.02$',r'$0.03$',r'$0.04$'])

    minorLocator = MultipleLocator(0.005)
    ax11.yaxis.set_minor_locator(minorLocator)

    ax11.text(0.04,0.08,r'\boldmath$x(\bar{d}-\bar{u})$' ,transform=ax11.transAxes,size=60)

    ax11.text(0.65,0.60,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=40)

    ax11.set_xlabel(r'\boldmath$x$'    ,size=60)
    ax11.xaxis.set_label_coords(0.82,0.00)

    handles,labels = [],[]
    handles.append(thy[0])
    handles.append(hand['models'])
    #handles.append(hand[IMFexp])
    #handles.append(hand[covexp])
    #handles.append(hand[covmon])
    #handles.append(hand[Regge])
    #handles.append(hand[PV])
    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{\textbf{pion cloud}}')
    #labels.append(r'\textrm{\textbf{\boldmath$s$ exp}}')
    #labels.append(r'\textrm{\textbf{\boldmath$t$ exp}}')
    #labels.append(r'\textrm{\textbf{\boldmath$t$ mon}}')
    #labels.append(r'\textrm{\textbf{Regge}}')
    #labels.append(r'\textrm{\textbf{P-V}}')

    ax11.legend(handles,labels,loc='upper right',fontsize = 40, frameon = 0, handletextpad = 0.3, handlelength = 0.95, ncol = 1, columnspacing = 0.5)

    py.tight_layout()

    py.subplots_adjust(hspace=0,wspace=0)

    #filename+='.png'
    filename+='.pdf'
    ax11.set_rasterized(True)
    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    py.clf()
    print ('Saving figure to %s'%filename)

if __name__ == "__main__":

    wdir0 = 'results/sq/final'

    WDIR = [wdir0]

    #--generate data if not already done
    #gen_data()

    plot_models(WDIR,kc)






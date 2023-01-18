#!/usr/bin/env python
import sys, os
import matplotlib
matplotlib.use('Agg')
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerBase
import pylab as py
matplotlib.rcParams['hatch.linewidth']=1.0
from scipy.integrate import cumtrapz,quad,fixed_quad

## from fitpack tools
from tools.tools     import load, save, checkdir, lprint
from tools.config    import conf, load_config

## from fitpack fitlib
from fitlib.resman import RESMAN

## from fitpack analysis
from analysis.corelib import core
from analysis.corelib import classifier
import kmeanconf as kc

from qcdlib.qpdcalc import QPDCALC

import lhapdf

#--for DSSV
import analysis.qpdlib.sets.DSSV.dssvlib as dssvlib
from analysis.qpdlib.sets.DSSV.DSSVcalc import DSSV


cwd = 'plots/star'


def load_lattice(flav):

    F = open('plots/star/data/lat_%s.csv'%flav,'r')
    L = F.readlines()
    F.close()
   
    L = [l.strip() for l in L]
    L = [[x for x in l.split()] for l in L]
    L = np.transpose(L)[0]
   
    X, up, do = [],[],[]
    for i in range(len(L)):
        if i==0: continue
        X .append(float(L[i].split(',')[0]))
        do.append(float(L[i].split(',')[1]))
        up.append(float(L[i].split(',')[2]))
   
    X,do,up = np.array(X),np.array(do),np.array(up)
    return X, do, up

def load_NNPDF(Q2):

    NNPDF = lhapdf.mkPDFs('NNPDFpol11_100')
    L = len(NNPDF)
    X = np.linspace(0.01,0.98,100)
    X = 10**np.linspace(-2,np.log10(0.99),100)
    up = np.zeros((L,len(X)))
    dp = np.zeros((L,len(X)))
    ub = np.zeros((L,len(X)))
    db = np.zeros((L,len(X)))
    UP, DP, UB, DB = [],[],[],[]
    for i in range(L):
        for j in range(len(X)):
            up[i][j] = NNPDF[i].xfxQ2(-2,X[j],Q2)/X[j] + NNPDF[i].xfxQ2(2,X[j],Q2)/X[j]
            dp[i][j] = NNPDF[i].xfxQ2(-1,X[j],Q2)/X[j] + NNPDF[i].xfxQ2(1,X[j],Q2)/X[j]
            ub[i][j] = NNPDF[i].xfxQ2(-2,X[j],Q2)/X[j]
            db[i][j] = NNPDF[i].xfxQ2(-1,X[j],Q2)/X[j]
        
        moment_temp = cumtrapz(up[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        UP.append((moment_max - moment_temp)[0])

        moment_temp = cumtrapz(dp[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        DP.append((moment_max - moment_temp)[0])

        moment_temp = cumtrapz(ub[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        UB.append((moment_max - moment_temp)[0])

        moment_temp = cumtrapz(db[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        DB.append((moment_max - moment_temp)[0])

    UP, DP, UB, DB = np.array(UP), np.array(DP), np.array(UB), np.array(DB)

    return UP, DP, UB, DB

def load_DSSV(Q2):

    dssv = DSSV()
    L = dssv.L
    X = 10**np.linspace(-2,np.log10(0.99),100)
    u  = dssv.xfxQ2(2 ,X,Q2)/X
    d  = dssv.xfxQ2(1 ,X,Q2)/X
    ub = dssv.xfxQ2(-2,X,Q2)/X
    db = dssv.xfxQ2(-1,X,Q2)/X
    up = u + ub
    dp = d + db
    UP = np.zeros(2*L+1)
    DP = np.zeros(2*L+1)
    UB = np.zeros(2*L+1)
    DB = np.zeros(2*L+1)
    for i in range(-L,L+1):
        moment_temp = cumtrapz(up[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        UP[i] = (moment_max - moment_temp)[0]

        moment_temp = cumtrapz(dp[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        DP[i] = (moment_max - moment_temp)[0]

        moment_temp = cumtrapz(ub[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        UB[i] = (moment_max - moment_temp)[0]

        moment_temp = cumtrapz(db[i], X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        DB[i] = (moment_max - moment_temp)[0]

    UPstd, DPstd, UBstd, DBstd = 0,0,0,0
    for i in range(1,20):
        UPstd += (UP[i] - UP[-i])**2
        DPstd += (DP[i] - DP[-i])**2
        UBstd += (UB[i] - UB[-i])**2
        DBstd += (DB[i] - DB[-i])**2

    UPstd = np.sqrt(UPstd)/2.0
    DPstd = np.sqrt(DPstd)/2.0
    UBstd = np.sqrt(UBstd)/2.0
    DBstd = np.sqrt(DBstd)/2.0

    UP = np.mean(UP)
    DP = np.mean(DP)
    UB = np.mean(UB)
    DB = np.mean(DB)


    return UP, DP, UB, DB, UPstd, DPstd, UBstd, DBstd

if __name__ == "__main__":
   
    #--choose if truncated
    trunc = True

    #--without positivity
    wdir1 = 'results/star/final'
    wdir2 = 'results/star/noRHIC2'
    #--with positivity
    wdir3 = 'results/star/pos2'
    wdir4 = 'results/star/noRHICpos2'

    WDIR = [wdir1,wdir2,wdir3,wdir4]

    nrows,ncols=1,2
    N = nrows*ncols
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
  
    thy  = {}
    hand = {}
    j = 0
    Q2 = 4
    for wdir in WDIR:
  
        load_config('%s/input.py'%wdir)
        istep=core.get_istep()
 
        if trunc==False: data =load('%s/data/ppdf-moment-%d-Q2=%d.dat'%(wdir,1,Q2))
        if trunc==True:  data =load('%s/data/ppdf-moment-trunc-%d-Q2=%d.dat'%(wdir,1,Q2))
  
        replicas=core.get_replicas(wdir)
        cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
        best_cluster=cluster_order[0]
  
        #flavs = ['u','d','ub','db','s+sb','g']
        flavs = ['up','dp','ub','db']
  
        mean1,std1,mean2,std2 = [],[],[],[]
        for flav in flavs:
            if trunc==True:
                if flav=='g':
                    momentp, momentn = [],[]
                    for i in range(len(data['moments'][flav])):
                        if data['moments'][flav][i][0] > 0: momentp.append(data['moments'][flav][i][0])
                        if data['moments'][flav][i][0] < 0: momentn.append(data['moments'][flav][i][0])
                    mean1.append(np.mean(momentp))
                    std1 .append(np.std (momentp))
                    mean1.append(np.mean(momentn))
                    std1 .append(np.std (momentn))
  
                elif flav=='up':
                    moment = []
                    for i in range(len(data['moments']['u'])):
                        moment.append(data['moments']['u'][i][0]+data['moments']['ub'][i][0])
  
                    mean1.append(np.mean(moment))
                    std1 .append(np.std (moment))

                elif flav=='dp':
                    moment = []
                    for i in range(len(data['moments']['d'])):
                        moment.append(data['moments']['d'][i][0]+data['moments']['db'][i][0])
  
                    mean1.append(-np.mean(moment))
                    std1 .append( np.std (moment))

                elif flav=='ub':
                    moment = []
                    for i in range(len(data['moments'][flav])):
                        moment.append(data['moments'][flav][i][0])
  
                    mean2.append(np.mean(moment))
                    std2 .append(np.std (moment))

                elif flav=='db':
                    moment = []
                    for i in range(len(data['moments'][flav])):
                        moment.append(data['moments'][flav][i][0])
  
                    mean2.append(-np.mean(moment))
                    std2 .append( np.std (moment))


            else:

                    mean2.append(np.mean(data['moments'][flav],axis=0))
                    std2 .append(np.std (data['moments'][flav],axis=0))
          
  
        X = np.array([0,1])

        #print(mean1,mean2)
        #print(std1,std2)
 
        colors1 = ['red','dodgerblue','darkviolet','green']
        colors2 = ['yellow','cyan','violet','limegreen']
        if j==0: left,right,hatch,zorder,color,alpha =  -0.18, 0.18, None, 2, 'red'    ,1.0
        if j==1: left,right,hatch,zorder,color,alpha =  -0.21, 0.21, None, 1, 'cyan' ,0.3
        if j==2: left,right,hatch,zorder,color,alpha =  -0.04, 0.04, None, 3, 'black'  ,1.0
        if j==3: left,right,hatch,zorder,color,alpha =  -0.05, 0.05, '///', 3, 'black'  ,1.0

        for i in range(len(X)):
            up1 = mean1[i] + std1[i]
            do1 = mean1[i] - std1[i]
            up2 = mean2[i] + std2[i]
            do2 = mean2[i] - std2[i]
            if j in [0,1]:
                hand[j] = ax11.fill_between([X[i]+left,X[i]+right],[do1,do1],[up1,up1],hatch=hatch,facecolor=color,edgecolor=color,zorder=zorder,alpha=alpha)
                hand[j] = ax12.fill_between([X[i]+left,X[i]+right],[do2,do2],[up2,up2],hatch=hatch,facecolor=color,edgecolor=color,zorder=zorder,alpha=alpha)
            if j in [2]:
                hand[j] = ax11.fill_between([X[i]+left,X[i]+right],[do1,do1],[up1,up1],hatch=hatch,facecolor='black',edgecolor='black',zorder=zorder,alpha=0.9)
                hand[j] = ax12.fill_between([X[i]+left,X[i]+right],[do2,do2],[up2,up2],hatch=hatch,facecolor='black',edgecolor='black',zorder=zorder,alpha=0.9)
            if j in [3]:
                hand[j] = ax11.fill_between([X[i]+left,X[i]+right],[do1,do1],[up1,up1],hatch=hatch,facecolor='none',edgecolor='black',zorder=zorder,alpha=0.5,lw=1)
                hand[j] = ax12.fill_between([X[i]+left,X[i]+right],[do2,do2],[up2,up2],hatch=hatch,facecolor='none',edgecolor='black',zorder=zorder,alpha=0.5,lw=1)
            if j in [4]:
                ax11.errorbar(X[i]+0.10,mean1[i],yerr=std1[i],capsize=4.0,color='darkblue',alpha=1.0,ms=0.0,marker='o',zorder=5)
                ax12.errorbar(X[i]+0.10,mean2[i],yerr=std2[i],capsize=4.0,color='darkblue',alpha=1.0,ms=0.0,marker='o',zorder=5)

        j+=1
  
 
   
    #--plot NNPDF
    up, dp, ub, db = load_NNPDF(Q2)

    l = 0.25
    color = 'darkgreen'
    #NNPDF  = ax11.errorbar([0.0+l],np.mean(up),yerr=np.std(up),capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    #ax11         .errorbar([1.0+l],np.mean(-dp),yerr=np.std(dp),capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    NNPDF  = ax12.errorbar([0.0+l],np.mean(ub),yerr=np.std(ub),capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    ax12         .errorbar([1.0+l],np.mean(-db),yerr=np.std(db),capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)

    #--plot DSSV
    up, dp, ub, db, upstd, dpstd, ubstd, dbstd = load_DSSV(Q2)

    l = 0.3
    color = 'darkblue'
    #DSSV  = ax11.errorbar([0.0+l],up,upstd,capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    #ax11        .errorbar([1.0+l],-dp,dpstd,capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    DSSV  = ax12.errorbar([0.0+l],ub,ubstd,capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)
    ax12        .errorbar([1.0+l],-db,dbstd,capsize=4.0,color=color,alpha=1.0,ms=5,fmt='o',zorder=2)


 
    #--plot lattice data
    up =  0.432*2
    dp = -0.213*2
    uperr = 2*0.008
    dperr = 2*0.008

    l = 0.3
    color1, color2 = 'darkgreen','limegreen'
    #lat1  = ax11.errorbar([0.0+l],up,yerr=uperr,capsize=4.0,color=color1,alpha=1.0,ms=0.0,marker='o',zorder=2)
    #ax11        .errorbar([1.0+l],dp,yerr=dperr,capsize=4.0,color=color1,alpha=1.0,ms=0.0,marker='o',zorder=2)
    #lat2 ,= ax11.plot(    [0.0+l],up,                       color=color2,alpha=1.0,ms=15,marker='o',zorder=1)
    #ax11        .plot(    [1.0+l],dp,                       color=color2,alpha=1.0,ms=15,marker='o',zorder=1)


    for flav in ['ub','db']:
        X, do, up = load_lattice(flav)
        idx = np.nonzero(X >= 0.01)
        X, do, up = X[idx], do[idx], up[idx]

        moment_temp = cumtrapz(do, X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        do = (moment_max - moment_temp)[0]

        moment_temp = cumtrapz(up, X, initial = 0.0)
        moment_temp = np.array(moment_temp)
        moment_max = moment_temp[-1]
        up = (moment_max - moment_temp)[0]

        mean = (do+up)/2.0
        std  = np.abs(up-do)/2.0

        if flav=='ub': x = 2.0
        if flav=='db': x = 3.0
        #ax11.errorbar([x+l],mean,yerr=std,capsize=4.0,color=color1 ,alpha=1.0,ms=0.0,marker='o',zorder=2)
        #ax11.plot(    [x+l],mean,                     color=color2 ,alpha=1.0,ms=15 ,marker='o',zorder=1)
    
    #--point for legend
    #ax11.errorbar(0.90,0.57,yerr=std,capsize=4.0,color=color1 ,alpha=1.0,ms=0.0,marker='o',zorder=2)
    #ax11.plot(    0.90,0.57,                     color=color2 ,alpha=1.0,ms=15 ,marker='o',zorder=1)


    for ax in [ax11,ax12]: 
        ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=5,pad=10)
        ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=2,pad=10)
 
    ax12.tick_params(labelleft=False,labelright=True)
 
    ax11.set_xlim(-0.4,1.4)
    ax11.set_xticks([0,1])
    ax11.set_xticklabels([r'\boldmath$\Delta u^+$',r'\boldmath$-\Delta d^+$'])

    ax12.set_xlim(-0.4,1.4)
    ax12.set_xticks([0,1])
    ax12.set_xticklabels([r'\boldmath$\Delta \bar{u}$',r'\boldmath$-\Delta \bar{d}$'])

    ax11.set_ylim(0.25,0.85)   
    ax11.set_yticks([0.4,0.6,0.8])
    minorLocator = MultipleLocator(0.10)
    ax11.yaxis.set_minor_locator(minorLocator)
  
    ax12.set_ylim(-0.022,0.105)   
    ax12.set_yticks([-0.02,0,0.02,0.04,0.06,0.08,0.10])
    ax12.set_yticklabels([r'',r'$0$',r'$0.02$',r'$0.04$',r'$0.06$',r'$0.08$',r''])
    minorLocator = MultipleLocator(0.01)
    ax12.yaxis.set_minor_locator(minorLocator)

    
    l = 0.26
    ax11.axhline(0,0,1,color='black',linestyle='-' ,alpha=0.2, lw=1.0)
    ax12.axhline(0,0,1,color='black',linestyle='-' ,alpha=0.2, lw=1.0)
    #ax11.plot([-1.0 ,2.0-l] ,[0,0],color='black',linestyle='-',alpha=0.2,lw=1.0)
    #ax11.plot([2.0+l,3.0-l] ,[0,0],color='black',linestyle='-',alpha=0.2,lw=1.0)
    #ax11.plot([3.0+l,4.0]   ,[0,0],color='black',linestyle='-',alpha=0.2,lw=1.0)
  
    if trunc==True:  ax11.text(0.45,0.75,r'\boldmath$\int_{0.01}^{1} {\rm d} x \Delta q$',     transform=ax11.transAxes,size=40)
    if trunc==False: ax11.text(0.45,0.75,r'\boldmath$\int_{0}^{1} {\rm d} x \Delta q$',        transform=ax11.transAxes,size=40)
  
    #ax11.text(0.03,0.43,r'$Q^2 = %d$~'%Q2 + r'\textrm{GeV}'+r'$^2$', transform=ax11.transAxes,size=25)
    #ax11.text(0.52,0.14,r'\textrm{\textbf{light: no RHIC \boldmath$W/Z$}}',   transform=ax11.transAxes,size=20)
    #ax11.text(0.52,0.05,r'\textrm{\textbf{dark:  with RHIC \boldmath$W/Z$}}', transform=ax11.transAxes,size=20)

    pos1 ,= ax11.plot(0,0,color='black',lw=10,alpha=0.9) 
    pos2  = ax11.fill_between([0],[0],[0],hatch=None,facecolor='none',edgecolor='black',zorder=zorder,alpha=0.7)

    blank ,= ax11.plot(0,0,color='white',alpha=0.0)

    handles,labels = [],[]
    handles.append(hand[1])
    handles.append(hand[0])
    handles.append(hand[3])
    handles.append(hand[2])
    labels.append(r'\textrm{\textbf{no \boldmath$W$ \hspace{0.20cm} +pos}}') 
    labels.append(r'\textrm{\textbf{JAM \hspace{0.45cm} +pos}}') 
    labels.append(r'') 
    labels.append(r'') 
    ax11.legend(handles,labels,loc=(0.01,0.05),fontsize=25,frameon=0,handletextpad=0.5,handlelength=1.0,ncol=2,columnspacing=0.35)

    handles,labels = [],[]
    handles.append(NNPDF)
    handles.append(DSSV)
    labels.append(r'\textrm{NNPDFpol1.1}')
    labels.append(r'\textrm{DSSV08}')
    ax12.legend(handles,labels,loc=(0.01,0.70),fontsize=25,frameon=0,handletextpad=0.5,handlelength=1.0,ncol=1,columnspacing=0.35)

  
    py.tight_layout()
    py.subplots_adjust(wspace=0.03,hspace=0)

    checkdir('%s/gallery'%cwd)
    filename = '%s/gallery/spin'%cwd
    #filename+='.png'
    filename+='.pdf'
    ax11 .set_rasterized(True)
    ax12 .set_rasterized(True)

    py.savefig(filename)
    print ('Saving figure to %s'%filename)
 





 
  
        
        
        
        
        
        
        
        
        
        
        
        

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

#--from obslib
from obslib.wzrv.theory import WZRV
from obslib.wzrv.reader import READER


cwd = 'plots/star'

if __name__=="__main__":

    wdir1 = 'results/star/final'
    wdir2 = 'results/star/noasym2'

    WDIR = [wdir1,wdir2]

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*7,nrows*6))
    ax11 = py.subplot(nrows,ncols,1)

    conf['path2wzrvtab'] = '%s/grids/grids-wzrv'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['xlsx']={}
    conf['datasets']['wzrv']['xlsx'][1000]='wzrv/expdata/1000.xlsx'
    conf['datasets']['wzrv']['norm']={}
    conf['datasets']['wzrv']['filters']=[]
    conf['wzrv tabs']=READER().load_data_sets('wzrv')
    tabs = conf['wzrv tabs']
    #--plot data
    hand = {}
    for idx in tabs:
        if idx==1000: color = 'black'
        values = tabs[idx]['value']
        alpha = np.sqrt(np.array(tabs[idx]['stat_u'])**2 + np.array(tabs[idx]['syst_u'])**2)
        if 'eta' in tabs[idx]:
            eta = tabs[idx]['eta']
        else: 
            eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
            eta_min = tabs[idx]['eta_min']
            eta_max = tabs[idx]['eta_max']
        n = int(len(eta)/2)
        hand[idx] = ax11.errorbar(eta[:n],values[:n],yerr=alpha[:n],color=color,fmt='o',ms=5.0,capsize=3.0)
        ax11            .errorbar(eta[n:],values[n:],yerr=alpha[n:],color=color,fmt='o',ms=5.0,capsize=3.0)

    j = 0
    for wdir in WDIR:

        load_config('%s/input.py'%wdir)
        istep=core.get_istep()
        replicas=core.get_replicas(wdir)
        core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
        predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
        data = predictions['reactions']['wzrv']
 
        #--get theory by seperating solutions and taking mean
        cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)
        for idx in tabs:
            predictions = copy.copy(data[idx]['prediction-rep'])
            for ic in range(nc):
                predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
                data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
                data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)


            #--compute cross-section for all replicas        
            thy  = data[idx]['thy-0']
            dthy = data[idx]['dthy-0']
            
            up   = thy + dthy 
            down = thy - dthy 

            if j == 0:
                thy_plot0 ,= ax11.plot(eta[:n],thy[:n],color='red',alpha=1.0,lw=2)
                ax11             .plot(eta[n:],thy[n:],color='red',alpha=1.0,lw=2)
                thy_band0  = ax11.fill_between(eta[:n],down[:n],up[:n],color='red',alpha=0.2)
                ax11             .fill_between(eta[n:],down[n:],up[n:],color='red',alpha=0.2)
            if j == 1:
                thy_plot1 ,= ax11.plot(eta[:n],thy[:n],color='green',alpha=1.0,ls='--',lw=2)
                ax11             .plot(eta[n:],thy[n:],color='green',alpha=1.0,ls='--',lw=2)

        j += 1


    for ax in [ax11]:
        ax.set_xlim(-1.9,1.9)
        minorLocator = MultipleLocator(0.25)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    ax11.axhline(0,0,1,ls='-',color='black',alpha=0.2,lw=1.0)
    ax11.set_xlabel(r'\boldmath$\eta_\ell$',size=40)
    ax11.xaxis.set_label_coords(0.95,-0.01)
    ax11.set_xticks([-1.0,0.0,1.0])
    ax11.text(0.08,0.03,r'\boldmath$A_L^{W}$'                                  ,transform=ax11.transAxes,size=50)
    ax11.text(0.65,0.43,r'$\sqrt{s}=510~$'+r'\textrm{GeV}'                 ,transform=ax11.transAxes,size=20)
    ax11.text(0.67,0.34,r'$p_T^\ell > 25~$'+r'\hspace{0.00cm}\textrm{GeV}' ,transform=ax11.transAxes,size=20)
    ax11.text(0.04,0.61,r'$W^-$'                                  ,transform=ax11.transAxes,size=25)
    ax11.text(0.04,0.30,r'$W^+$'                                  ,transform=ax11.transAxes,size=25)

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    ax11.set_ylim(-0.90,0.90)
    ax11.set_yticks([-0.5,0,0.5])
    ax11.set_yticklabels([r'$-0.5$',r'$0$',r'$0.5$'])

    handles, labels = [],[]
    handles.append((thy_band0,thy_plot0))
    handles.append(thy_plot1)
    handles.append(hand[1000])

    labels.append(r'\textrm{\textbf{JAM}}')
    labels.append(r'\textrm{\textbf{\boldmath$\Delta \bar{u} \!=\! \Delta \bar{d}$ fit}}')
    labels.append(r'\textrm{\textbf{STAR}}')
    ax11.legend(handles,labels,frameon=False,fontsize=25,loc='upper left',handletextpad=0.3,handlelength=1.2,ncol=2,columnspacing=2.3)

    py.tight_layout()
    py.subplots_adjust(hspace=0,top=0.99)

    checkdir('%s/gallery'%cwd)
    filename='%s/gallery/star'%(cwd)
    #filename += '.png'
    filename += '.pdf'
    ax11.set_rasterized(True)

    py.savefig(filename)
    print('Saving A_L plot to %s'%filename)






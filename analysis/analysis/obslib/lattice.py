import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
import pylab as py
import matplotlib.gridspec as gridspec

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#-- from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from obslib
from obslib.wzrv.theory import WZRV
from obslib.wzrv.reader import READER


def plot_obs(wdir,kc,mode=1):

    plot_lattice_spin(wdir,kc,mode=mode)
    plot_lattice_momentum(wdir,kc,mode=mode)

def plot_lattice_spin(wdir,kc,mode=1):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'SU23' not in conf['steps'][istep]['datasets']: return 
    if 60001 not in conf['steps'][istep]['datasets']['SU23']:
        if 60002 not in conf['steps'][istep]['datasets']['SU23']: return 

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)

    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['SU23']={}
    conf['datasets']['SU23']['xlsx']={}
    conf['datasets']['SU23']['xlsx'][60001]='SU23/expdata/60001.xlsx'
    conf['datasets']['SU23']['xlsx'][60002]='SU23/expdata/60002.xlsx'
    conf['datasets']['SU23']['norm']={}
    conf['datasets']['SU23']['filters']=[]
    conf['SU23 tabs']=READER().load_data_sets('SU23')
    tabs = conf['SU23 tabs']

    print('generating lattice moments from %s'%(wdir))
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    data = predictions['reactions']['SU23']
 
    #--get theory by seperating solutions and taking mean
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)
    for idx in tabs:
        if idx not in data: continue
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    hand = {}
    #--plot data
    for idx in tabs:
        if idx not in data: continue
        if idx==60001: color = 'black'
        if idx==60002: color = 'darkgreen'
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==60001: X = [0,1,2]
        if idx==60002: X = [3]
        hand[idx] = ax11.errorbar(X,values,yerr=alpha,color=color,fmt='o',ms=5.0,capsize=3.0)

        #--compute observable for all replicas        
        cnt=0
        if mode==0:
            for i in range(len(replicas)):
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                thy = data[idx]['prediction-rep'][i]
                for i in range(len(X)):
                    thy_plot ,= ax11.plot(X[i],thy[i],color='red',alpha=0.3,marker='o')

        if mode==1:
            thy  = data[idx]['thy-0']
            dthy = data[idx]['dthy-0']
            
            up   = thy + dthy 
            down = thy - dthy 

            #thy_plot = ax11.errorbar(X,thy,yerr=dthy,color='black',alpha=1.0,fmt='o',ms=5.0,capsize=3.0)
            for i in range(len(X)):
                thy_plot = ax11.fill_between([X[i]-0.1,X[i]+0.1],[down[i],down[i]],[up[i],up[i]],color='red',alpha=1.0)


    ax11.set_ylim(-0.60,1.10)
    ax11.set_xlim(-0.20,3.20)

    #minorLocator = MultipleLocator(0.1)
    #majorLocator = MultipleLocator(0.5)
    #ax11.xaxis.set_minor_locator(minorLocator)
    #ax11.xaxis.set_major_locator(majorLocator)
    #ax11.xaxis.set_tick_params(which='major',length=6)
    #ax11.xaxis.set_tick_params(which='minor',length=3)
    ax11.set_xticks([0,1,2,3])
    ax11.set_xticklabels([r'\boldmath$\Delta u^+$',r'\boldmath$\Delta d^+$',r'\boldmath$\Delta s^+$',r'\boldmath$g$'])

    ax11.tick_params(axis='both',which='both',top=True,direction='in',labelsize=20,pad=10)

    ax11.axhline(0,0,1,color='black',alpha=0.2,lw=1.0)

    handles, labels = [],[]
    handles.append(hand[60001])
    if 60002 in hand: handles.append(hand[60002])
    handles.append(thy_plot)

    labels.append(r'\textrm{\textbf{ETMC}}')
    if 60002 in hand: labels.append(r'\textrm{\textbf{\boldmath$\chi$QCD}}')
    labels.append(r'\textrm{\textbf{JAM}}')
    ax11.legend(handles,labels,frameon=False,fontsize=20,loc='upper right',handletextpad=0.5,handlelength=1.5,ncol=2,columnspacing=1.0)

    #py.tight_layout()
    #py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/lattice.png'%(wdir)

    py.savefig(filename)
    print('Saving lattice plot to %s'%filename)
    py.clf()

def plot_lattice_momentum(wdir,kc,mode=1):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'SU23' not in conf['steps'][istep]['datasets']: return 

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)

    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['SU23']={}
    conf['datasets']['SU23']['xlsx']={}
    conf['datasets']['SU23']['xlsx'][60011]='SU23/expdata/60011.xlsx'
    conf['datasets']['SU23']['norm']={}
    conf['datasets']['SU23']['filters']=[]
    conf['SU23 tabs']=READER().load_data_sets('SU23')
    tabs = conf['SU23 tabs']

    print('generating lattice moments from %s'%(wdir))
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    data = predictions['reactions']['SU23']
 
    #--get theory by seperating solutions and taking mean
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)
    for idx in tabs:
        if idx not in data: continue
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    hand = {}
    #--plot data
    for idx in tabs:
        if idx not in data: continue
        if idx==60011: color = 'black'
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==60011: X = [0,1,2,3]
        hand[idx] = ax11.errorbar(X,values,yerr=alpha,color=color,fmt='o',ms=5.0,capsize=3.0)

        #--compute observable for all replicas        
        cnt=0
        if mode==0:
            for i in range(len(replicas)):
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                thy = data[idx]['prediction-rep'][i]
                thy_plot ,= ax11.plot(eta,thy,color='red',alpha=0.3)

        if mode==1:
            thy  = data[idx]['thy-0']
            dthy = data[idx]['dthy-0']
            
            up   = thy + dthy 
            down = thy - dthy 

            #thy_plot = ax11.errorbar(X,thy,yerr=dthy,color='black',alpha=1.0,fmt='o',ms=5.0,capsize=3.0)
            for i in range(len(X)):
                thy_plot = ax11.fill_between([X[i]-0.1,X[i]+0.1],[down[i],down[i]],[up[i],up[i]],color='red',alpha=1.0)


    ax11.set_ylim( 0.00,0.70)
    ax11.set_xlim(-0.20,3.20)

    #minorLocator = MultipleLocator(0.1)
    #majorLocator = MultipleLocator(0.5)
    #ax11.xaxis.set_minor_locator(minorLocator)
    #ax11.xaxis.set_major_locator(majorLocator)
    #ax11.xaxis.set_tick_params(which='major',length=6)
    #ax11.xaxis.set_tick_params(which='minor',length=3)
    ax11.set_xticks([0,1,2,3])
    ax11.set_xticklabels([r'\boldmath$x u^+$',r'\boldmath$x d^+$',r'\boldmath$x s^+$',r'\boldmath$x g$'])

    ax11.tick_params(axis='both',which='both',top=True,direction='in',labelsize=20,pad=10)

    ax11.axhline(0,0,1,color='black',alpha=0.2,lw=1.0)

    handles, labels = [],[]
    handles.append(hand[60011])
    handles.append(thy_plot)

    labels.append(r'\textrm{\textbf{ETMC}}')
    labels.append(r'\textrm{\textbf{JAM}}')
    ax11.legend(handles,labels,frameon=False,fontsize=20,loc='upper left',handletextpad=0.5,handlelength=1.5,ncol=1,columnspacing=1.0)

    #py.tight_layout()
    #py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/lattice_mom.png'%(wdir)

    py.savefig(filename)
    print('Saving lattice plot to %s'%filename)
    py.clf()





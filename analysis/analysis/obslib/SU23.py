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

    plot_SU23(wdir,kc,mode=mode)

def plot_SU23(wdir,kc,mode=1):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'SU23' not in conf['steps'][istep]['datasets']: return 
    #else:
    #    if 20001 not in conf['steps'][istep]['datasets']['SU23']: return 
    #    if 20002 not in conf['steps'][istep]['datasets']['SU23']: return
 
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)

    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['SU23']={}
    conf['datasets']['SU23']['xlsx']={}
    conf['datasets']['SU23']['xlsx'][10001]='SU23/expdata/10001.xlsx'
    conf['datasets']['SU23']['xlsx'][10002]='SU23/expdata/10002.xlsx'
    conf['datasets']['SU23']['xlsx'][20001]='SU23/expdata/20001.xlsx'
    conf['datasets']['SU23']['xlsx'][20002]='SU23/expdata/20002.xlsx'
    conf['datasets']['SU23']['norm']={}
    conf['datasets']['SU23']['filters']=[]
    conf['SU23 tabs']=READER().load_data_sets('SU23')
    tabs = conf['SU23 tabs']

    print('generating SU23 from %s'%(wdir))
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
        if idx in [10001,20001]: color = 'firebrick'
        if idx in [10002,20002]: color = 'darkgreen'
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        #hand[idx] = ax11.errorbar(1,values,yerr=alpha,color=color,fmt='o',ms=5.0,capsize=3.0)
        hand[idx] = ax11.errorbar(values,1,xerr=alpha,yerr=0,color=color,fmt='o',ms=5.0,capsize=3.0)

        #--compute observable for all replicas        
        cnt=0
        if mode==0:
            for i in range(len(replicas)):
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                thy = data[idx]['prediction-rep'][i]
                thy_plot ,= ax11.plot(thy,1.01,color='black',alpha=1.0)

        if mode==1:
            thy  = data[idx]['thy-0']
            dthy = data[idx]['dthy-0']
            
            up   = thy + dthy 
            down = thy - dthy 

            #thy_plot = ax11.errorbar(1.01,thy,yerr=dthy,color='black',alpha=1.0,fmt='o',ms=5.0,capsize=3.0)
            thy_plot = ax11.errorbar(thy,1.01,xerr=dthy,yerr=0,color='black',alpha=1.0,fmt='o',ms=5.0,capsize=3.0)
            #thy_band  = ax11.fill_between(eta,down,up,color='gold',alpha=1.0)


    ax11.set_ylim( 0.99,1.02)
    ax11.set_xlim(-0.40,1.90)

    minorLocator = MultipleLocator(0.1)
    majorLocator = MultipleLocator(0.5)
    ax11.xaxis.set_minor_locator(minorLocator)
    ax11.xaxis.set_major_locator(majorLocator)
    ax11.xaxis.set_tick_params(which='major',length=6)
    ax11.xaxis.set_tick_params(which='minor',length=3)
    ax11.set_xticks([0.0,0.5,1.0,1.5])

    ax11.text(0.68,0.75,r'\boldmath$g_A$'                 ,transform=ax11.transAxes,size=25)
    ax11.text(0.30,0.75,r'\boldmath$a_8$'                 ,transform=ax11.transAxes,size=25)
    ax11.text(0.75,0.63,r'\textrm{\textbf{JAM20}}'        ,transform=ax11.transAxes,size=25)
    if 10001 in data: ax11.text(0.75,0.30,r'\textrm{\textbf{Nom.}}'         ,transform=ax11.transAxes,size=25)
    if 20001 in data: ax11.text(0.75,0.30,r'\textrm{\textbf{JAM17}}'        ,transform=ax11.transAxes,size=25)

    ax11.axvline(0,0,1,color='black',alpha=0.5,ls=':')

    ax11.tick_params(axis='both',which='both',top=True,left=False,direction='in',labelsize=20,labelleft=False)


    #handles, labels = [],[]
    #if mode==0: handles.append(thy_plot)
    #if mode==1: handles.append(thy_plot)

    #labels.append(r'\textrm{\textbf{JAM}}')
    #ax11.legend(handles,labels,frameon=False,fontsize=20,loc='upper left',handletextpad=0.5,handlelength=1.5,ncol=2,columnspacing=1.0)

    #py.tight_layout()
    #py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/SU23.png'%(wdir)

    py.savefig(filename)
    print('Saving SU23 plot to %s'%filename)
    py.clf()






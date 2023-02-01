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

    plot_A_L       (wdir,kc,mode=mode)
    plot_A_L_PHENIX(wdir,kc,mode=mode)
    plot_A_LL      (wdir,kc,mode=mode)

def plot_A_L(wdir,kc,mode=1):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'wzrv' not in conf['steps'][istep]['datasets']: return 
    else:
        if 1000 not in conf['steps'][istep]['datasets']['wzrv']: return 
        if 1020 not in conf['steps'][istep]['datasets']['wzrv']: return 
 
    nrows,ncols=4,1
    fig = py.figure(figsize=(ncols*7,nrows*2))
    ax11 = py.subplot(nrows,ncols,(1,3))
    ax31 = py.subplot(nrows,ncols,4)

    conf['path2wzrvtab'] = '%s/grids/grids-wzrv'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['xlsx']={}
    conf['datasets']['wzrv']['xlsx'][1000]='wzrv/expdata/1000.xlsx'
    conf['datasets']['wzrv']['xlsx'][1020]='wzrv/expdata/1020.xlsx'
    conf['datasets']['wzrv']['xlsx'][1021]='wzrv/expdata/1021.xlsx'
    conf['datasets']['wzrv']['norm']={}
    conf['datasets']['wzrv']['filters']=[]
    conf['wzrv tabs']=READER().load_data_sets('wzrv')
    tabs = conf['wzrv tabs']

    print('generating A_L from %s'%(wdir))
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

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

    hand = {}
    #--plot data
    for idx in tabs:
        if idx==1000: color = 'firebrick'
        if idx==1020: color = 'darkgreen'
        if idx==1021: color = 'purple'
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if 'eta' in tabs[idx]:
            eta = tabs[idx]['eta']
            hand[idx] = ax11.errorbar(eta,values,yerr=alpha,color=color,fmt='o',ms=5.0,capsize=3.0)
        else: 
            eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
            eta_min = tabs[idx]['eta_min']
            eta_max = tabs[idx]['eta_max']
        n = int(len(eta)/2)
        hand[idx] = ax11.errorbar(eta,values,yerr=alpha,color=color,fmt='o',ms=5.0,capsize=3.0)

        #--compute cross-section for all replicas        
        cnt=0
        if mode==0:
            for i in range(len(replicas)):
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                thy = data[idx]['prediction-rep'][i]
                thy_plot ,= ax11.plot(eta[:n],thy[:n],color='red',alpha=0.3)
                ax11            .plot(eta[n:],thy[n:],color='red',alpha=0.3)


        if mode==1:
            thy  = data[idx]['thy-0']
            dthy = data[idx]['dthy-0']
            
            up   = thy + dthy 
            down = thy - dthy 

            if idx==1000:
                thy_plot ,= ax11.plot(eta[:n],thy[:n],color='black',alpha=1.0)
                ax11            .plot(eta[n:],thy[n:],color='black',alpha=1.0)
                thy_band  = ax11.fill_between(eta[:n],down[:n],up[:n],color='gold',alpha=1.0)
                ax11            .fill_between(eta[n:],down[n:],up[n:],color='gold',alpha=1.0)
            elif idx in [1020,1021]:
                thy_point  = ax11.errorbar(eta[:n],thy[:n],yerr=dthy[:n],color='black',alpha=1.0,fmt='o',ms=5.0,capsize=3.0,zorder=5)
                ax11             .errorbar(eta[n:],thy[n:],yerr=dthy[n:],color='black',alpha=1.0,fmt='o',ms=5.0,capsize=3.0,zorder=5)


        for ic in range(nc):
            if nc > 1: color = colors[cluster[ic]]
            ratio = values/thy
            ax31.errorbar(eta,ratio,yerr=alpha/thy,color=color,fmt='.',ms=10,capsize=3.0)

    for ax in [ax11,ax31]:
        ax.set_xlim(-1.6,1.6)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
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

    ax31.axhline(1,0,1,color='black',ls='--')
    ax11.axhline(0,0,1,ls='--',color='black',alpha=0.5)
    ax31.set_xlabel(r'\boldmath$\eta$',size=30)
    ax31.xaxis.set_label_coords(0.95,0.00)
    ax31.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0])
    ax11.text(0.85,0.90,r'\boldmath$A_L$'                  ,transform=ax11.transAxes,size=40)
    ax11.text(0.64,0.42,r'$\sqrt{s}=510~$'+r'\textrm{GeV}' ,transform=ax11.transAxes,size=25)
    ax11.text(0.55,0.35,r'$25 < p_T < 50~$'+r'\textrm{GeV}',transform=ax11.transAxes,size=25)
    ax31.text(0.03,0.08,r'\textbf{\textrm{data/theory}}'   ,transform=ax31.transAxes,size=30)

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20,labelbottom=False)
    ax31.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20)


    ax11.set_ylim(-0.7,0.7)
    ax11.set_yticks([-0.5,0,0.5])
    ax31.set_ylim(0.4,1.6)

    handles, labels = [],[]
    handles.append(hand[1000])
    handles.append(hand[1020])
    if mode==0: handles.append(thy_plot)
    if mode==1: handles.append((thy_band,thy_plot))

    labels.append(r'\textrm{\textbf{STAR \boldmath$W^{\pm},p_T>25$}}')
    labels.append(r'\textrm{\textbf{PHENIX \boldmath$W^{\pm}/Z,p_T>30$}}')
    labels.append(r'\textrm{\textbf{JAM}}')
    ax11.legend(handles,labels,frameon=False,fontsize=20,loc='upper left',handletextpad=0.5,handlelength=1.5,ncol=1,columnspacing=1.0)

    py.tight_layout()
    py.subplots_adjust(hspace=0)

    filename = 'A_L'
    if mode==1: filename += '-bands'
    filename += '.png'

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/%s'%(wdir,filename)

    py.savefig(filename)
    print('Saving A_L plot to %s'%filename)

def plot_A_L_PHENIX(wdir,kc,mode=1):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'wzrv' not in conf['steps'][istep]['datasets']: return 
    else:
        if 1021 not in conf['steps'][istep]['datasets']['wzrv']: return 
 
    nrows,ncols=4,1
    fig = py.figure(figsize=(ncols*7,nrows*2))
    ax11 = py.subplot(nrows,ncols,(1,3))
    ax31 = py.subplot(nrows,ncols,4)

    conf['path2wzrvtab'] = '%s/grids/grids-wzrv'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['xlsx']={}
    conf['datasets']['wzrv']['xlsx'][1021]='wzrv/expdata/1021.xlsx'
    conf['datasets']['wzrv']['norm']={}
    conf['datasets']['wzrv']['filters']=[]
    conf['wzrv tabs']=READER().load_data_sets('wzrv')
    tabs = conf['wzrv tabs']

    print('generating A_L PHENIX from %s'%(wdir))
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

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

    hand = {}
    #--plot data
    for idx in tabs:
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if 'eta' in tabs[idx]:
            eta = tabs[idx]['eta']
            hand[idx] = ax11.errorbar(eta,values,yerr=alpha,color=color,fmt='o',ms=5.0,capsize=3.0)
        else: 
            eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
            eta_min = tabs[idx]['eta_min']
            eta_max = tabs[idx]['eta_max']
        n = int(len(eta)/2)
        for i in range(len(eta)):
            boson=tabs[idx]['boson'][i]
            if 'W+' in boson: color,shift,_hand = 'firebrick', 0.0, 'W+'
            if 'W-' in boson: color,shift,_hand = 'darkgreen', 0.1, 'W-'
            hand[_hand] = ax11.errorbar(eta[i]+shift,values[i],yerr=alpha[i],c=color,fmt='o',ms=5.0,capsize=3.0)

        #--compute cross-section for all replicas        
        cnt=0
        if mode==0:
            for i in range(len(replicas)):
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                thy = data[idx]['prediction-rep'][i]
                thy_plot ,= ax11.plot(eta[:n],thy[:n],color='red',alpha=0.3)
                ax11            .plot(eta[n:],thy[n:],color='red',alpha=0.3)


        if mode==1:
            thy  = data[idx]['thy-0']
            dthy = data[idx]['dthy-0']
            
            up   = thy + dthy 
            down = thy - dthy 

            if idx==1021:
                thy_plot ,= ax11.plot(eta[:n]+0.0,thy[:n],color='firebrick',alpha=1.0)
                ax11            .plot(eta[n:]+0.1,thy[n:],color='darkgreen',alpha=1.0)
                thy_band  = ax11.fill_between(eta[:n]+0.0,down[:n],up[:n],color='firebrick',alpha=0.3)
                ax11            .fill_between(eta[n:]+0.1,down[n:],up[n:],color='darkgreen',alpha=0.3)
            #elif idx in [1020,1021]:
            #    thy_point  = ax11.errorbar(eta[:n],thy[:n],yerr=dthy[:n],color='black',alpha=1.0,fmt='o',ms=5.0,capsize=3.0,zorder=5)
            #    ax11             .errorbar(eta[n:],thy[n:],yerr=dthy[n:],color='black',alpha=1.0,fmt='o',ms=5.0,capsize=3.0,zorder=5)


        for ic in range(nc):
            if nc > 1: color = colors[cluster[ic]]
            ratio = values/thy
            diff  = values - thy
            for i in range(len(eta)):
                boson=tabs[idx]['boson'][i]
                if 'W+' in boson: color,shift = 'firebrick', 0.0
                if 'W-' in boson: color,shift = 'darkgreen', 0.1
                ax31.errorbar(eta[i]+shift,diff[i],yerr=alpha[i],color=color,fmt='.',ms=10,capsize=3.0)

    for ax in [ax11,ax31]:
        ax.set_xlim(-2.0,2.0)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
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

    ax31.axhline(0,0,1,color='black',ls='--')
    ax11.axhline(0,0,1,ls='--',color='black',alpha=0.5)
    ax31.set_xlabel(r'\boldmath$\eta$',size=30)
    ax31.xaxis.set_label_coords(0.95,0.00)
    ax31.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0])
    ax11.text(0.80,0.85,r'\boldmath$A_L$'                  ,transform=ax11.transAxes,size=40)
    ax11.text(0.60,0.75,r'$\sqrt{s}=510~$'+r'\textrm{GeV}' ,transform=ax11.transAxes,size=25)
    ax11.text(0.60,0.68,r'$p_T > 16~$'+r'\textrm{GeV}'     ,transform=ax11.transAxes,size=25)
    ax31.text(0.30,0.08,r'\textbf{\textrm{data-theory}}'   ,transform=ax31.transAxes,size=30)

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20,labelbottom=False)
    ax31.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20)


    ax11.set_ylim(-1.2,1.2)
    ax11.set_yticks([-0.5,0,0.5])
    ax31.set_ylim(-1.0,1.0)

    handles, labels = [],[]
    handles.append(hand['W+'])
    handles.append(hand['W-'])
    if mode==0: handles.append(thy_plot)
    if mode==1: handles.append((thy_band,thy_plot))

    labels.append(r'\textrm{\textbf{PHENIX \boldmath$W^+/Z$}}')
    labels.append(r'\textrm{\textbf{PHENIX \boldmath$W^-/Z$}}')
    labels.append(r'\textrm{\textbf{JAM}}')
    ax11.legend(handles,labels,frameon=False,fontsize=20,loc='lower left',handletextpad=0.5,handlelength=1.5,ncol=1,columnspacing=1.0)

    py.tight_layout()
    py.subplots_adjust(hspace=0)

    filename = 'A_L_PHENIX'
    if mode==1: filename += '-bands'
    filename += '.png'

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/%s'%(wdir,filename)

    py.savefig(filename)
    print('Saving A_L plot to %s'%filename)

def plot_A_LL(wdir,kc,mode=1):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if 'wzrv' not in conf['steps'][istep]['datasets']: return 
    else:
        if 1002 not in conf['steps'][istep]['datasets']['wzrv']: return 
        if 1003 not in conf['steps'][istep]['datasets']['wzrv']: return 

    nrows,ncols=2,1
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)
    ax21 = py.subplot(nrows,ncols,2)

    conf['path2wzrvtab'] = '%s/grids/grids-wzrv'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['xlsx']={}
    conf['datasets']['wzrv']['xlsx'][1002]='wzrv/expdata/1002.xlsx'
    conf['datasets']['wzrv']['xlsx'][1003]='wzrv/expdata/1003.xlsx'
    conf['datasets']['wzrv']['norm']={}
    conf['datasets']['wzrv']['filters']=[]
    conf['wzrv tabs']=READER().load_data_sets('wzrv')
    tabs = conf['wzrv tabs']

    print('generating A_LL from %s'%(wdir))
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

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

    hand = {}
    #--plot data
    for idx in tabs:
        if idx==1002: ax,color = ax11,'firebrick'
        if idx==1003: ax,color = ax21,'darkgreen'
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if 'eta' in tabs[idx]:
            eta = tabs[idx]['eta']
            hand[idx] = ax.errorbar(eta,values,yerr=alpha,color=color,fmt='o',ms=5.0,capsize=3.0)
        else: 
            eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
            eta_min = tabs[idx]['eta_min']
            eta_max = tabs[idx]['eta_max']
            xerr = np.zeros((2,len(eta)))
            hand[idx] = ax.errorbar(eta,values,xerr=xerr,yerr=alpha,color=color,fmt='o',ms=5.0,capsize=3.0)

        #--compute cross-section for all replicas        
        cnt=0
        if mode==0:
            for i in range(len(replicas)):
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                thy = data[idx]['prediction-rep'][i]
                thy_plot ,= ax.plot(eta,thy,color='red',alpha=0.3)


        if mode==1:
            thy  = data[idx]['thy-0']
            dthy = data[idx]['dthy-0']
            
            up   = thy + dthy 
            down = thy - dthy 

            thy_plot ,= ax.plot(eta,thy,color='black',alpha=1.0)
            thy_band  = ax.fill_between(eta,down,up,color='gold',alpha=1.0)

    for ax in [ax11,ax21]:
        ax.set_xlim(0.0,1.6)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
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

    ax11.axhline(0,0,1,ls='--',color='black',alpha=0.5)
    ax21.axhline(0,0,1,ls='--',color='black',alpha=0.5)
    ax21.set_xlabel(r'\boldmath$\eta$',size=30)
    ax21.xaxis.set_label_coords(0.95,0.00)
    ax21.set_xticks([0.0,0.5,1.0])
    ax11.text(0.80,0.85,r'\boldmath$A_{LL}$'              ,transform=ax11.transAxes,size=40)
    ax21.text(0.05,0.15,r'$\sqrt{s}=510~$'+r'\textrm{GeV}',transform=ax21.transAxes,size=25)
    ax21.text(0.05,0.05,r'$p_T > 25~$'+r'\textrm{GeV}'    ,transform=ax21.transAxes,size=25)

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20,labelbottom=False)
    ax21.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20)

    ax11.set_ylim(-0.30,0.30)
    ax21.set_ylim(-0.50,0.15)
    ax11.set_yticks([-0.2,0,0.2])
    ax21.set_yticks([-0.4,-0.2,0])

    handles, labels = [],[]
    handles.append(hand[1002])
    handles.append(hand[1003])
    handles.append((thy_band,thy_plot))

    labels.append(r'\textrm{\textbf{STAR (W$^+$)}}')
    labels.append(r'\textrm{\textbf{STAR (W$^-$)}}')
    labels.append(r'\textrm{\textbf{JAM}}')
    ax11.legend(handles,labels,frameon=False,fontsize=20,loc='upper left',handletextpad=0.5,handlelength=1.5,ncol=2,columnspacing=1.0)

    py.tight_layout()
    py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%wdir)
    filename='%s/gallery/A_LL.png'%(wdir)

    py.savefig(filename)
    print('Saving A_LL plot to %s'%filename)





import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

#--matplotlib
import matplotlib
matplotlib.use('Agg')
import pylab as py
from matplotlib.ticker import MultipleLocator

#--from scipy stack 
from scipy.integrate import fixed_quad
from scipy import interpolate

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from obslib
from obslib.wzrv.reader import READER

def plot_obs(wdir,kc,mode=1):

    print('\ngenerating WZRV from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'wzrv' not in predictions['reactions']: return

    nrows,ncols=2,2
    gridspec = dict(hspace=0, height_ratios=[1,0.5])
    fig,axs = py.subplots(nrows=nrows,ncols=ncols,gridspec_kw=gridspec,figsize=(ncols*7,nrows*3))
    ax11 = axs[0][0] 
    ax12 = axs[0][1] 
    ax21 = axs[1][0] 
    ax22 = axs[1][1] 

    conf['path2wzrvtab'] = '%s/grids/grids-wzrv'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['xlsx']={}
    conf['datasets']['wzrv']['xlsx'][2010]='wzrv/expdata/2010.xlsx'
    conf['datasets']['wzrv']['xlsx'][2011]='wzrv/expdata/2011.xlsx'
    conf['datasets']['wzrv']['xlsx'][2012]='wzrv/expdata/2012.xlsx'
    conf['datasets']['wzrv']['xlsx'][2013]='wzrv/expdata/2013.xlsx'
    conf['datasets']['wzrv']['xlsx'][2014]='wzrv/expdata/2014.xlsx'
    conf['datasets']['wzrv']['xlsx'][2016]='wzrv/expdata/2016.xlsx'
    conf['datasets']['wzrv']['xlsx'][2017]='wzrv/expdata/2017.xlsx'
    conf['datasets']['wzrv']['norm']={}
    conf['datasets']['wzrv']['filters']=[]
    conf['wzrv tabs']=READER().load_data_sets('wzrv')
    tabs = conf['wzrv tabs']

    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
  
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['wzrv']
  
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in tabs:
        if idx not in data: continue
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic, axis = 0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic, axis=0)

    flag = True
    for idx in [2010,2012,2016,2017]:
        if idx in data: flag = False
    if flag: return

    #######################
    #--plot absolute values
    #######################

    hand = {}
    axes = {}
    axes['ax'] = {}
    axes['ax'][2010] = ax11
    axes['ax'][2011] = ax11
    axes['ax'][2012] = ax11
    axes['ax'][2013] = ax11
    axes['ax'][2014] = ax11
    axes['ax'][2016] = ax12
    axes['ax'][2017] = ax12
    axes['color'] = {}
    axes['color'][2010] = 'firebrick' 
    axes['color'][2011] = 'darkturquoise' 
    axes['color'][2012] = 'darkmagenta' 
    axes['color'][2013] = 'darkturquoise' 
    axes['color'][2014] = 'darkturquoise' 
    axes['color'][2016] = 'firebrick' 
    axes['color'][2017] = 'darkgreen' 
    axes['marker'] = {}
    axes['marker'][2010] = '*'
    axes['marker'][2011] = '^'
    axes['marker'][2012] = '.'
    axes['marker'][2013] = '^'
    axes['marker'][2014] = '^'
    axes['marker'][2016] = 'o'
    axes['marker'][2017] = '*'

    #--plot data
    for idx in tabs:
        if idx not in data: continue
        if idx not in axes['ax']: continue
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        ax     = axes['ax'][idx]
        color  = axes['color'][idx]
        marker = axes['marker'][idx]
        if 'eta' in tabs[idx]:
            eta = tabs[idx]['eta']
            hand[idx] = ax.errorbar(eta,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=6.0,capsize=3.0)
        else: 
            eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
            eta_min = tabs[idx]['eta_min']
            eta_max = tabs[idx]['eta_max']
            xerr = np.zeros((2,len(eta)))
            hand[idx] = ax.errorbar(eta,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=6.0,capsize=3.0)

    #--plot cross-section for all replicas
    if mode == 0:      
        cnt=0
        for i in range(len(replicas)):
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))
            color = colors[cluster[i]]

            for idx in [2010,2011,2012,2016,2017]:
                if idx not in data: continue
                ax = axes['ax'][idx]
                if 'eta' in tabs[idx]: eta = tabs[idx]['eta']
                else: eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
                repdata=data[idx]['prediction-rep'][i]
                thy_plot = ax.plot(eta ,repdata ,color=color,alpha=0.1)

    #--plot mean and std of all replicas
    if mode==1:
        for idx in [2010,2011,2012,2016,2017]:
            if idx not in data: continue
            for ic in range(nc):
                ax     = axes['ax'][idx]
                color  = axes['color'][idx]
                marker = axes['marker'][idx]
                if 'eta' in tabs[idx]: eta = tabs[idx]['eta']
                else: eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
                thy = data[idx]['thy-%d'%ic]
                std = data[idx]['dthy-%d'%ic]
                down = thy - std
                up   = thy + std
                thy_plot ,= ax.plot(eta,thy,color='black')
                thy_band  = ax.fill_between(eta,down,up,color='gold',alpha=1.0)
    
    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30,labelbottom=False)

    ax12.axhline(0,0,3,alpha=0.5,color='black',ls='--')

    for ax in [ax11,ax21]: ax.set_xlim(0.0,2.6)
    for ax in [ax12,ax22]: ax.set_xlim(2.0,4.6)

    ax11.text(0.05, 0.05, r'\boldmath$A_l$'          , transform = ax11.transAxes, fontsize=40)

    ax11.text(0.05, 0.82, r'\textrm{\textbf{CMS}}'   , transform = ax11.transAxes, fontsize=40)
    ax12.text(0.70, 0.82, r'\textrm{\textbf{LHCb}}'  , transform = ax12.transAxes, fontsize=40)

    ax11.set_ylim( 0.00,0.30)
    ax12.set_ylim(-0.55,0.30)

    majorLocator = MultipleLocator(0.1)
    minorLocator = MultipleLocator(0.025)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)
    majorLocator = MultipleLocator(0.2)
    minorLocator = MultipleLocator(0.05)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)

    ax11.set_yticks([0.1,0.2])

    for ax in [ax11,ax12]:
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    #######################
    #--plot data - theory or data/theory
    #######################

    axes['ax'] = {}
    axes['ax'][2010] = ax21
    axes['ax'][2011] = ax21
    axes['ax'][2012] = ax21
    axes['ax'][2013] = ax21
    axes['ax'][2014] = ax21
    axes['ax'][2016] = ax22
    axes['ax'][2017] = ax22

    for idx in data:
        if idx not in axes['ax']: continue
        ax     = axes['ax'][idx]
        color  = axes['color'][idx]
        marker = axes['marker'][idx]
        ms = 8
        for ic in range(nc):
            eta0, eta1 = [], []
            thy0, thy1, ratio0, ratio1, alpha0, alpha1 = [], [], [], [], [], []
            boson = conf['wzrv tabs'][idx]['boson']
            if 'eta' in conf['wzrv tabs'][idx]: eta = conf['wzrv tabs'][idx]['eta']
            else: eta = (conf['wzrv tabs'][idx]['eta_min'] + conf['wzrv tabs'][idx]['eta_max'])/2.0
            thy   = data[idx]['thy-%d'%ic]
            value = data[idx]['value']
            alpha = data[idx]['alpha']
            std = data[idx]['dthy-%d'%ic]
            if ax in [ax21]: measure = value/thy
            if ax in [ax21]: yerr    = alpha/value
            if ax in [ax21]: thyerr  = std/thy
            if ax in [ax22]: measure = value-thy
            if ax in [ax22]: yerr    = alpha
            if ax in [ax22]: thyerr  = std 
            ax.errorbar(eta,measure,yerr=yerr,color=color,linestyle='none',marker=marker,ms=ms,capsize=3.0)
            #if ax in [ax21]: ax.fill_between(eta,1-thyerr,1+thyerr,color='blue',alpha=0.2)
            #if ax in [ax22]: ax.fill_between(eta,0-thyerr,0+thyerr,color='blue',alpha=0.2)
             

    for ax in [ax21,ax22]: 
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlabel(r'\boldmath$|\eta|$',size=30)
        ax.xaxis.set_label_coords(0.90,-0.02)

    for ax in [ax21,ax22]:
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    for ax in [ax21]:
        ax.set_ylim(0.85,1.15)
        ax.axhline(1,0,3,alpha=1.0,color='black',ls='--')
        minorLocator = MultipleLocator(0.025)
        majorLocator = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.set_yticks([0.9,1.0,1.1])

    for ax in [ax22]:
        ax.set_ylim(-0.05,0.05)
        ax.axhline(0,0,3,alpha=1.0,color='black',ls='--')
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.04)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    ax21.set_xticks([1,2])
    ax22.set_xticks([3,4])

    handles,labels=[],[]
    handles.append(hand[2011])
    handles.append(hand[2012])
    handles.append(hand[2010])
    if 2011 in hand: labels.append(r'\boldmath$\sqrt{s}=7,p_T>25$')
    if 2012 in hand: labels.append(r'\boldmath$\sqrt{s}=7,p_T>35$')
    if 2010 in hand: labels.append(r'\boldmath$\sqrt{s}=8,p_T>25$')
    ax11.legend(handles,labels,frameon=False,fontsize=22,loc='lower right', ncol = 1, handletextpad = 0.3, handlelength = 1.0)

    handles,labels=[],[]
    if mode==0: handles.append(thy_plot)
    if mode==1: handles.append((thy_band,thy_plot))
    handles.append(hand[2016])
    handles.append(hand[2017])
    labels.append(r'\textbf{\textrm{JAM}}')
    if 2016 in hand: labels.append(r'\boldmath$\sqrt{s}=7,p_T>20$')
    if 2017 in hand: labels.append(r'\boldmath$\sqrt{s}=8,p_T>20$')
    ax12.legend(handles,labels,frameon=False,fontsize=22,loc='lower left', ncol = 1, handletextpad = 0.3, handlelength = 1.0)

    ax21.text(0.05, 0.80, r'\textrm{\textbf{data/theory}}'  , transform = ax21.transAxes, fontsize=25)
    ax22.text(0.05, 0.80, r'\textrm{\textbf{data-theory}}'  , transform = ax22.transAxes, fontsize=25)

    py.tight_layout()
    py.subplots_adjust(hspace = 0.0, wspace=0.20)

    checkdir('%s/gallery'%wdir)
    if mode == 0: filename='%s/gallery/wzrv.png'%(wdir)
    if mode == 1: filename='%s/gallery/wzrv-bands.png'%(wdir)

    py.savefig(filename)
    print()
    print('Saving wzrv plot to %s'%filename)

def plot_star(wdir,kc,mode=1):

    print('\ngenerating STAR WZRV from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'wzrv' not in predictions['reactions']: return
    if 2020 not in predictions['reactions']['wzrv']: return

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*7,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)

    conf['path2wzrvtab'] = '%s/grids/grids-wzrv'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['xlsx']={}
    conf['datasets']['wzrv']['xlsx'][2020]='wzrv/expdata/2020.xlsx'
    conf['datasets']['wzrv']['norm']={}
    conf['datasets']['wzrv']['filters']=[]
    conf['wzrv tabs']=READER().load_data_sets('wzrv')
    tabs = conf['wzrv tabs']

    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
  
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['wzrv']
  
    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in tabs:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic, axis = 0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic, axis=0)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in tabs:
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==2020: ax,color,marker = ax11,'darkgreen','.'
        if 'eta' in tabs[idx]:
            eta = tabs[idx]['eta']
            hand[idx] = ax.errorbar(eta,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=6.0,capsize=3.0)

        else: 
            eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
            eta_min = tabs[idx]['eta_min']
            eta_max = tabs[idx]['eta_max']
            xerr = np.zeros((2,len(eta)))
            hand[idx] = ax.errorbar(eta,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=6.0,capsize=3.0)

    #--plot cross-section for all replicas
    if mode == 0:      
        cnt=0
        for i in range(len(replicas)):
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))
            color = colors[cluster[i]]

            for idx in [2020]:
                if idx==2020: ax = ax11
                eta0, eta1 = [], []
                repdata0, repdata1 = [], []
                boson = tabs[idx]['boson']
                #--separate W+ data from W- data
                for j in range(len(boson)):
                    if boson[j]=='W':
                        if 'eta' in tabs[idx]: eta = tabs[idx]['eta']
                        else: eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
                        repdata=data[idx]['prediction-rep'][i]
                        break
                    if boson[j]=='W+':
                        if 'eta' in tabs[idx]: eta0.append(tabs[idx]['eta'][j])
                        else: eta0.append((tabs[idx]['eta_min'][j] + tabs[idx]['eta_max'][j])/2.0)
                        repdata0.append(data[idx]['prediction-rep'][i][j])
                    if boson[j]=='W-':
                        if 'eta' in tabs[idx]: eta1.append(tabs[idx]['eta'][j])
                        else: eta1.append((tabs[idx]['eta_min'][j] + tabs[idx]['eta_max'][j])/2.0)
                        repdata1.append(data[idx]['prediction-rep'][i][j])

                if boson[0]=='W':
                    ax.plot(eta ,repdata ,color=color,alpha=0.1)
                else:  
                    ax.plot(eta0,repdata0,color=color,alpha=0.1)
                    ax.plot(eta1,repdata1,color=color,alpha=0.1)

    #--plot mean and std of all replicas
    if mode==1:
        for idx in [2020]:
            for ic in range(nc):
                if idx==2020: ax = ax11
                eta0, eta1 = [], []
                thy0, thy1 = [], []
                std0, std1 = [], []
                up0 , up1  = [], []
                down0,down1= [], []
                boson = conf['wzrv tabs'][idx]['boson']
                #--separate W+ data from W- data
                for j in range(len(boson)):
                    if boson[j]=='W':
                        if 'eta' in tabs[idx]: eta = tabs[idx]['eta']
                        else: eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
                        thy = data[idx]['thy-%d'%ic]
                        std = data[idx]['dthy-%d'%ic]
                        down = thy - std
                        up   = thy + std
                        break
                    if boson[j]=='W+':
                        if 'eta' in tabs[idx]: eta0.append(tabs[idx]['eta'][j])
                        else: eta0.append((tabs[idx]['eta_min'][j] + tabs[idx]['eta_max'][j])/2.0)
                        thy0 .append(data[idx]['thy-%d'%ic][j])
                        std0 .append(data[idx]['dthy-%d'%ic][j])
                    if boson[j]=='W-':
                        if 'eta' in tabs[idx]: eta1.append(tabs[idx]['eta'][j])
                        else: eta1.append((tabs[idx]['eta_min'][j] + tabs[idx]['eta_max'][j])/2.0)
                        thy1 .append(data[idx]['thy-%d'%ic][j])
                        std1 .append(data[idx]['dthy-%d'%ic][j])

                if boson[0]=='W':
                    thy_plot ,= ax.plot(eta,thy,color='black')
                    thy_band  = ax.fill_between(eta,down,up,color='gold',alpha=1.0)
                else:  
                    down0 = np.array(thy0) - np.array(std0)
                    up0   = np.array(thy0) + np.array(std0)
                    down1 = np.array(thy1) - np.array(std1)
                    up1   = np.array(thy1) + np.array(std1)
                    thy_plot ,= ax.plot(eta0,thy0,color='black')
                    thy_band  = ax.fill_between(eta0,down0,up0,color='gold',alpha=1.0)
                    thy_plot ,= ax.plot(eta1,thy1,color='black')
                    thy_band  = ax.fill_between(eta1,down1,up1,color='gold',alpha=1.0)
    
    for ax in [ax11]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    if mode == 1:
        handles = [hand[2020],(thy_band,thy_plot)]
        label1 = r'\textbf{\textrm{STAR}}'
        label2 = r'\textbf{\textrm{JAM}}'
        labels = [label1,label2] 
        ax11.legend(handles,labels,frameon=False,fontsize=20,loc='upper right', ncol = 1, handletextpad = 0.3, handlelength = 1.0)


    ax11.set_xlabel(r'\boldmath$\eta$',size=30)
    ax11.xaxis.set_label_coords(0.95,-0.02)

    ax11.set_xlim(-1.3,1.3)
    ax11.set_ylim(0.5,9.0)

    ax11.text(0.05,0.85, r'\boldmath$\sigma_{W^+}/\sigma_{W^-}$'    ,transform = ax11.transAxes, size=40)
    ax11.text(0.05,0.75, r'$\sqrt{s} = 510$'+' '+r'\textrm{GeV}'    ,transform = ax11.transAxes, size=25)
    ax11.text(0.05,0.65, r'\rm{$p_T$ $>$ 25 GeV}'                   ,transform = ax11.transAxes, size=25)

    for ax in [ax11]:
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    for ax in [ax11]:
        majorLocator = MultipleLocator(0.5)
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        majorLocator = MultipleLocator(2)
        minorLocator = MultipleLocator(0.5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)



    py.tight_layout()
    #py.subplots_adjust(hspace = 0, wspace=0.15)

    checkdir('%s/gallery'%wdir)
    if mode == 0: filename='%s/gallery/wzrv-star.png'%(wdir)
    if mode == 1: filename='%s/gallery/wzrv-star-bands.png'%(wdir)

    py.savefig(filename)
    print()
    print('Saving STAR wzrv plot to %s'%filename)





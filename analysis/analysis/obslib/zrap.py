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

#-- from qcdlib
from qcdlib import aux

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

#--from obslib
from obslib.zrap.reader import READER

def plot_obs(wdir,kc,mode=1):

    print('\ngenerating Z rapidity plots from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'zrap' not in predictions['reactions']: return

    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax21 = py.subplot(nrows,ncols,3)
    ax22 = py.subplot(nrows,ncols,4)

    conf['path2zraptab'] = '%s/grids/grids-zrap'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['zrap']={}
    conf['datasets']['zrap']['xlsx']={}
    conf['datasets']['zrap']['xlsx'][1000]='zrap/expdata/1000.xlsx'
    conf['datasets']['zrap']['xlsx'][1001]='zrap/expdata/1001.xlsx'
    conf['datasets']['zrap']['norm']={}
    conf['datasets']['zrap']['filters']=[]
    conf['zrap tabs']=READER().load_data_sets('zrap')
    tabs = conf['zrap tabs']
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['zrap']

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in tabs:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in tabs:
        Y = tabs[idx]['Y']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==1000: ax,color = ax12,'darkgreen'
        if idx==1001: ax,color = ax11,'firebrick' 
        hand[idx] = ax.errorbar(Y,values,yerr=alpha,color=color,fmt='o',ms=2,capsize=3.0)

        #--compute cross-section for all replicas
        if mode==0:   
            cnt=0
            for i in range(len(replicas)):
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                repdata = data[idx]['prediction-rep'][i]
                thy_plot ,= ax.plot(Y,repdata,color='red',alpha=0.1)

        #--plot mean and std of all replicas
        if mode==1:
           for ic in range(nc):
               if nc > 1: color = colors[cluster[ic]]
               thy = data[idx]['thy-%d'%ic]
               std = data[idx]['dthy-%d'%ic]
               down = thy - std
               up   = thy + std
               thy_plot ,= ax.plot(Y,thy,color='black')
               thy_band  = ax.fill_between(Y,down,up,color='gold',alpha=1.0)

    #######################
    #--plot ratio
    #######################


    for idx in tabs:
        if idx==1000: ax,color,label = ax22,'darkgreen',r'\textbf{\textrm{CDF(Z)}}'
        if idx==1001: ax,color,label = ax21,'firebrick',r'\textbf{\textrm{D0(Z)}}'
        for ic in range(nc):
            Y = conf['zrap tabs'][idx]['Y']
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            ratio = data[idx]['value']/thy
            alpha = data[idx]['alpha']
            ax.errorbar(Y,ratio,yerr=alpha/thy,color=color,fmt='.',ms=10,capsize=3.0)
            ax.text(0.02,0.78,label,fontsize=25,transform=ax.transAxes)
            ax.axhline(1,0,3,color='black',ls='--')
            

    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_xlim(0,3)

    for ax in [ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0,3)

    for ax in [ax11,ax12]:
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    for ax in [ax21,ax22]:
        ax.set_ylim(0.75,1.25)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xlabel(r'\boldmath$y_Z$',size=30)
        ax.xaxis.set_label_coords(0.85,-0.02)

    
    ax11.set_ylim(0,0.3)
    ax12.set_ylim(0,80)

    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.1)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)
    minorLocator = MultipleLocator(5)
    majorLocator = MultipleLocator(20)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)

    ax11.set_yticks([0,0.1,0.2])
    ax11.set_yticklabels([r'',r'$0.1$',r'$0.2$'])
    ax12.set_yticks([0,20,40,60,80])
    ax12.set_yticklabels([r'',r'$20$',r'$40$',r'$60$',r''])


    ax11.text(0.05, 0.05, r'\boldmath$\frac{1}{\sigma} \frac{d\sigma(Z/\gamma^*)}{dy_Z}$',transform=ax11.transAxes,size=40)
    ax12.text(0.05, 0.05, r'\boldmath$\frac{d\sigma(Z/\gamma^*)}{dy_Z}$'                 ,transform=ax12.transAxes,size=40)
    ax21.text(0.05, 0.05,  r'\textbf{\textrm{data/theory}}'                              ,transform=ax21.transAxes,size=30)

    ax11.text(0.1,0.15, r'$\sqrt{s} = 1.96$'+' '+r'\textrm{TeV}'   ,fontsize=25)

    if mode==0: handles = [hand[1000],hand[1001],thy_plot]
    if mode==1: handles = [hand[1000],hand[1001],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{CDF(Z)}}' 
    label2  = r'\textbf{\textrm{D0(Z)}}' 
    label3  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2,label3] 
    ax12.legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.5, handlelength = 1.0)

    py.tight_layout()
    py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%wdir)
    if mode==0: filename='%s/gallery/zrap.png'%(wdir)
    if mode==1: filename='%s/gallery/zrap-bands.png'%(wdir)

    py.savefig(filename)
    print()
    print('Saving Z boson cross-section/ratio plot to %s'%filename)

#--plot ATLAS data (not included in current analyses)
def plot_ATLAS(wdir,kc,mode=1):

    print('\ngenerating Z rapidity ATLAS plot from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'zrap' not in predictions['reactions']: return

    nrows,ncols=2,1
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)
    ax21 = py.subplot(nrows,ncols,2)

    conf['path2zraptab'] = '%s/grids/grids-zrap'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['zrap']={}
    conf['datasets']['zrap']['xlsx']={}
    conf['datasets']['zrap']['xlsx'][1010]='zrap/expdata/1010.xlsx'
    conf['datasets']['zrap']['norm']={}
    conf['datasets']['zrap']['filters']=[]
    conf['zrap tabs']=READER().load_data_sets('zrap')
    tabs = conf['zrap tabs']
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    data = predictions['reactions']['zrap']

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    #--get theory by seperating solutions and taking mean
    for idx in tabs:
        predictions = copy.copy(data[idx]['prediction-rep'])
        for ic in range(nc):
            predictions_ic = [predictions[i] for i in range(len(predictions)) if cluster[i] == ic]
            data[idx]['thy-%d'%ic]  = np.mean(predictions_ic,axis=0)
            data[idx]['dthy-%d'%ic] = np.std(predictions_ic,axis=0)

    #######################
    #--plot absolute values
    #######################

    hand = {}
    #--plot data
    for idx in tabs:
        Y = tabs[idx]['Y']
        values = tabs[idx]['value']
        alpha  = data[idx]['alpha']
        if idx==1010: ax,color = ax11,'firebrick' 
        hand[idx] = ax.errorbar(Y,values,yerr=alpha,color=color,fmt='o',ms=2,capsize=3.0)

        #--compute cross-section for all replicas
        if mode==0:   
            cnt=0
            for i in range(len(replicas)):
                cnt+=1
                lprint('%d/%d'%(cnt,len(replicas)))
                repdata = data[idx]['prediction-rep'][i]
                thy_plot ,= ax.plot(Y,repdata,color='red',alpha=0.1)

        #--plot mean and std of all replicas
        if mode==1:
           for ic in range(nc):
               if nc > 1: color = colors[cluster[ic]]
               thy = data[idx]['thy-%d'%ic]
               std = data[idx]['dthy-%d'%ic]
               down = thy - std
               up   = thy + std
               thy_plot ,= ax.plot(Y,thy,color='black')
               thy_band  = ax.fill_between(Y,down,up,color='gold',alpha=1.0)

    #######################
    #--plot ratio
    #######################


    for idx in tabs:
        if idx==1010: ax,color,label = ax21,'firebrick',r'\textbf{\textrm{ATLAS(Z)}}'
        for ic in range(nc):
            Y = conf['zrap tabs'][idx]['Y']
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            ratio = data[idx]['value']/thy
            alpha = data[idx]['alpha']
            ax.errorbar(Y,ratio,yerr=alpha/thy,color=color,fmt='.',ms=10,capsize=3.0)
            #ax.text(0.02,0.78,label,fontsize=25,transform=ax.transAxes)
            ax.axhline(1,0,3,color='black',ls='--')
            

    for ax in [ax11]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_xlim(0,3.5)

    for ax in [ax21]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0,3.5)

    for ax in [ax11]:
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    for ax in [ax21]:
        ax.set_ylim(0.75,1.25)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xlabel(r'\boldmath$y_Z$',size=30)
        ax.xaxis.set_label_coords(0.75,-0.02)

    
    ax11.set_ylim(0,150)

    minorLocator = MultipleLocator(10)
    majorLocator = MultipleLocator(50)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)

    ax11.set_yticks([0,50,100])
    ax11.set_yticklabels([r'',r'$50$',r'$100$',r''])


    ax11.text(0.05, 0.05, r'\boldmath$\frac{d\sigma(Z/\gamma^*)}{dy_Z}$'                 ,transform=ax11.transAxes,size=40)
    ax21.text(0.05, 0.05,  r'\textbf{\textrm{data/theory}}'                              ,transform=ax21.transAxes,size=30)

    ax11.text(0.05,0.40, r'$\sqrt{s} = 7$'+' '+r'\textrm{TeV}'   ,transform=ax11.transAxes, fontsize=25)

    if mode==0: handles = [hand[1010],thy_plot]
    if mode==1: handles = [hand[1010],(thy_band,thy_plot)]
    label1  = r'\textbf{\textrm{ATLAS(Z)}}' 
    label2  = r'\textbf{\textrm{JAM}}'
    labels  = [label1,label2] 
    ax11.legend(handles,labels,frameon=False,fontsize=22,loc='lower right',handletextpad = 0.5, handlelength = 1.0)

    py.tight_layout()
    py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%wdir)
    if mode==0: filename='%s/gallery/zrap-ATLAS.png'%(wdir)
    if mode==1: filename='%s/gallery/zrap-ATLAS-bands.png'%(wdir)

    py.savefig(filename)
    print()
    print('Saving Z boson ATLAS cross-section/ratio plot to %s'%filename)









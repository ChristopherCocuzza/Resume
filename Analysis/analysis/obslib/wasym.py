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
from obslib.wasym.reader import READER

def plot_obs(wdir,kc,mode=1):

    print('\ngenerating W asymmetry plots from %s'%(wdir))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    if 'wasym' not in predictions['reactions']: return

    nrows,ncols=4,1
    fig = py.figure(figsize=(ncols*7,nrows*2))
    ax11 = py.subplot(nrows,ncols,(1,2))
    ax21 = py.subplot(nrows,ncols,3)
    ax31 = py.subplot(nrows,ncols,4)

    conf['path2wasymtab'] = '%s/grids/grids-wasym'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wasym']={}
    conf['datasets']['wasym']['xlsx']={}
    conf['datasets']['wasym']['xlsx'][1000]='wasym/expdata/1000.xlsx'
    conf['datasets']['wasym']['xlsx'][1001]='wasym/expdata/1001.xlsx'
    conf['datasets']['wasym']['norm']={}
    conf['datasets']['wasym']['filters']=[]
    conf['wasym tabs']=READER().load_data_sets('wasym')
    tabs = conf['wasym tabs']
    
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
   
    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    data = predictions['reactions']['wasym']

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
        if idx==1000: color = 'darkgreen'
        if idx==1001: color = 'firebrick'
        hand[idx] = ax11.errorbar(Y,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)

    #--compute cross-section for all replicas
    if mode == 0:   
        cnt=0
        for i in range(len(replicas)):
            cnt+=1
            lprint('%d/%d'%(cnt,len(replicas)))

            for idx in tabs:
                Y = tabs[idx]['Y']
                repdata=data[idx]['prediction-rep'][i]
                ax11.plot(Y,repdata,color='red',alpha=0.1)

    #--plot mean and std of all replicas
    if mode==1:
        for idx in data:
            for ic in range(nc):
                Y = tabs[idx]['Y']
                if nc > 1: color = colors[cluster[ic]]
                thy = data[idx]['thy-%d'%ic]
                std = data[idx]['dthy-%d'%ic]
                down = thy - std
                up   = thy + std
                thy_plot ,= ax11.plot(Y,thy,color='black')
                thy_band  = ax11.fill_between(Y,down,up,color='gold',alpha=1.0)

    #######################
    #--plot ratio
    #######################


    for idx in data:
        if idx==1000: ax,color,label = ax21,'darkgreen',r'\textbf{\textrm{CDF(W)}}'
        if idx==1001: ax,color,label = ax31,'firebrick' ,r'\textbf{\textrm{D0(W)}}'
        for ic in range(nc):
            Y = conf['wasym tabs'][idx]['Y']
            if nc > 1: color = colors[cluster[ic]]
            thy = data[idx]['thy-%d'%ic]
            ratio = data[idx]['value']/thy
            alpha = data[idx]['alpha']
            ax.errorbar(Y,ratio,yerr=alpha/thy,color=color,fmt='.',ms=10,capsize=3.0)
            if idx==1000: ax.text(0.74,0.78,label,fontsize=25,transform=ax.transAxes)
            if idx==1001: ax.text(0.79,0.78,label,fontsize=25,transform=ax.transAxes)
            ax.axhline(1,0,3,color='black',ls='--')
            

    for ax in [ax11,ax21]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_xlim(0,3)

    for ax in [ax31]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0,3)

    for ax in [ax11]:
        ax.set_ylim(0,0.8)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_yticks([0.2,0.4,0.6,0.8])
        ax.set_yticklabels([r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$'])

    for ax in [ax21,ax31]:
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

    ax31.set_xlabel(r'\boldmath$y_W$',size=30)
    ax31.xaxis.set_label_coords(0.85,-0.02)

    ax11.text(2.4, 0.04, r'\boldmath$A_W$',size=40)
    ax21.text(0.1, 0.8, r'\textbf{\textrm{data/theory}}'    ,size=30)

    ax11.text(0.1,0.30, r'$\sqrt{s} = 1.96$'+' '+r'\textrm{TeV}'   ,fontsize=25)

    if mode == 0:
        ax11.plot([],[],color='red',label=r'\textbf{\textrm{JAM}}')
        ax11.legend(frameon=False,fontsize=22,loc='upper left',handletextpad = 0.5, handlelength = 1.5)

    if mode == 1:
        handles,labels = [], []
        handles.append(hand[1000])
        handles.append(hand[1001])
        #handles.append((thy_band,thy_plot))
        labels.append(r'\textbf{\textrm{CDF(W)}}') 
        labels.append(r'\textbf{\textrm{D0(W)}}')
        #labels.append(r'\textbf{\textrm{JAM}}')
        ax11.legend(handles,labels,frameon=False,fontsize=22,loc='upper left',handletextpad = 0.5, handlelength = 1.5)

    py.tight_layout()
    py.subplots_adjust(hspace=0)

    checkdir('%s/gallery'%wdir)
    if mode==0: filename='%s/gallery/wasym.png'%(wdir)
    if mode==1: filename='%s/gallery/wasym-bands.png'%(wdir)

    py.savefig(filename)
    print()
    print('Saving W asymmetry cross-section/ratio plot to %s'%filename)












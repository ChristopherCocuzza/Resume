#!/usr/bin/env python
import sys, os
import matplotlib
matplotlib.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT

#--matplotlib
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
from obslib.wzrv.theory import WZRV
from obslib.wzrv.reader import READER

import kmeanconf as kc

cwd = 'plots/highx'

if __name__=="__main__":

    wdir = 'results/marathon/step30'

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))

    nrows,ncols=4,2
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax21 = py.subplot(nrows,ncols,3)
    ax22 = py.subplot(nrows,ncols,4)
    ax31 = py.subplot(nrows,ncols,5)
    ax32 = py.subplot(nrows,ncols,6)
    ax41 = py.subplot(nrows,ncols,7)
    ax42 = py.subplot(nrows,ncols,8)

    conf['path2wzrvtab'] = '%s/grids/grids-wzrv'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['datasets'] = {}
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['xlsx']={}
    conf['datasets']['wzrv']['xlsx'][2000]='wzrv/expdata/2000.xlsx'
    conf['datasets']['wzrv']['xlsx'][2003]='wzrv/expdata/2003.xlsx'
    conf['datasets']['wzrv']['xlsx'][2006]='wzrv/expdata/2006.xlsx'
    conf['datasets']['wzrv']['xlsx'][2007]='wzrv/expdata/2007.xlsx'
    conf['datasets']['wzrv']['xlsx'][2009]='wzrv/expdata/2009.xlsx'
    conf['datasets']['wzrv']['xlsx'][2010]='wzrv/expdata/2010.xlsx'
    conf['datasets']['wzrv']['xlsx'][2011]='wzrv/expdata/2011.xlsx'
    conf['datasets']['wzrv']['xlsx'][2012]='wzrv/expdata/2012.xlsx'
    conf['datasets']['wzrv']['xlsx'][2013]='wzrv/expdata/2013.xlsx'
    conf['datasets']['wzrv']['xlsx'][2014]='wzrv/expdata/2014.xlsx'
    conf['datasets']['wzrv']['xlsx'][2015]='wzrv/expdata/2015.xlsx'
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
        if idx==2000: ax,color,marker = ax11,'firebrick',    '.'
        if idx==2003: ax,color,marker = ax11,'darkgreen',    '*'
        if idx==2006: ax,color,marker = ax11,'darkturquoise','^'
        if idx==2007: ax,color,marker = ax21,'darkgreen',    '.'
        if idx==2015: ax,color,marker = ax22,'firebrick',    '.'
        if idx==2009: ax,color,marker = ax22,'darkgreen',    '*'
        if idx==2010: ax,color,marker = ax21,'firebrick',    '*'
        if idx==2011: ax,color,marker = ax12,'darkturquoise','.'
        if idx==2012: ax,color,marker = ax12,'darkgreen',    '*'
        if idx==2013: ax,color,marker = ax12,'firebrick',    '^'
        if idx==2014: ax,color,marker = ax12,'purple',       'p'
        if 'eta' in tabs[idx]:
            eta = tabs[idx]['eta']
            hand[idx] = ax.errorbar(eta,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=6.0,capsize=3.0)

        else: 
            eta = (tabs[idx]['eta_min'] + tabs[idx]['eta_max'])/2.0
            eta_min = tabs[idx]['eta_min']
            eta_max = tabs[idx]['eta_max']
            xerr = np.zeros((2,len(eta)))
            hand[idx] = ax.errorbar(eta,values,yerr=alpha,color=color,linestyle='none',marker=marker,ms=6.0,capsize=3.0)

    #--plot mean and std of all replicas
    for idx in [2000,2007,2009,2010,2011,2012,2015]:
        for ic in range(nc):
            if idx==2000: ax = ax11
            if idx==2007: ax = ax21
            if idx==2010: ax = ax21
            if idx==2011: ax = ax12
            if idx==2012: ax = ax12
            if idx==2015: ax = ax22
            if idx==2009: ax = ax22
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
    
    for ax in [ax11,ax12,ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30,labelbottom=False)

    handles = [hand[2000],hand[2003],hand[2006],(thy_band,thy_plot)]
    label1 = r'\textbf{\textrm{D0(e)}}'
    label2 = r'\textbf{\textrm{CDF(e)}}'
    label3 = r'\textbf{\textrm{D0(\boldmath$\mu$)}}'
    label4 = r'\textbf{\textrm{JAM}}'
    labels = [label1,label2,label3,label4] 
    ax11.legend(handles,labels,frameon=False,fontsize=20,loc='lower left', ncol = 1, handletextpad = 0.3, handlelength = 1.0)

    handles = [hand[2011],hand[2012],hand[2013],hand[2014]]
    label1 = r'\textbf{\textrm{CMS(\boldmath$\mu$)(2011)}}'
    label2 = r'\textbf{\textrm{CMS(e)(2011)}}'
    label3 = r'\textbf{\textrm{CMS(e)(2010)}}'
    label4 = r'\textbf{\textrm{CMS(\boldmath$\mu$)(2010)}}'
    labels = [label1,label2,label3,label4] 
    ax12.legend(handles,labels,frameon=False,fontsize=20,loc='upper left', ncol = 2, handletextpad = 0.3, handlelength = 1.0, columnspacing = 1.0)

    handles = [hand[2007],hand[2010]]
    label1 = r'\textbf{\textrm{ATLAS(2012)}}'
    label2 = r'\textbf{\textrm{CMS}}'
    labels = [label1,label2] 
    ax21.legend(handles,labels,frameon=False,fontsize=20,loc='lower left', ncol = 1, handletextpad = 0.3, handlelength = 1.0)

    handles = [hand[2015],hand[2009]]
    label1 = r'\textbf{\textrm{ATLAS(2011)}}'
    label2 = r'\textbf{\textrm{ATLAS(2010)}}'
    labels = [label1,label2] 
    ax22.legend(handles,labels,frameon=False,fontsize=20,loc='upper right',ncol = 1, handletextpad = 0.3, handlelength = 1.0)


    ax11.set_ylabel(r'\boldmath$A_l$',size=30)
    #ax21.set_ylabel(r'\boldmath$\frac{d\sigma_W}{d|\eta|}$',size=30)
    ax21.set_ylabel(r'\boldmath$d\sigma_W/d|\eta|$' + ' ' + r'\textbf{\textrm{(pb)}}',size=30)
    ax21.set_xlabel(r'\boldmath$|\eta|$',size=30)
    ax22.set_xlabel(r'\boldmath$|\eta|$',size=30)

    ax11.axhline(0,0,3,alpha=0.5,color='black',ls='--')
    ax11.set_ylim(-0.5,0.30)
    ax12.set_ylim(0.05,0.39)
    ax21.set_ylim(250,850)
    ax22.set_ylim(250,850)

    ax11.set_xlim(0,3.0)
    ax21.set_xlim(0,3.0)
    ax12.set_xlim(0,3.0)
    ax22.set_xlim(0,3.0)

    ax11.text(1.75,0.22, r'$\sqrt{s} = 1.96$'+' '+r'\textrm{TeV}'   ,fontsize=25)
    ax12.text(1.4,0.075, r'$\sqrt{s} = 7$'   +' '+r'\textrm{TeV}'   ,fontsize=25)
    ax21.text(1.7,275,   r'$\sqrt{s} = 8$'   +' '+r'\textrm{TeV}'   ,fontsize=25)
    ax22.text(0.1,770,   r'$\sqrt{s} = 7$'   +' '+r'\textrm{TeV}'   ,fontsize=25)

    ax11.text(1.75,0.12, r'\rm{$p_T$ $>$ 25 GeV}',fontsize=25)

    ax12.text(0.1,0.20,  r'\rm{$p_T$ $>$ 25 GeV}',fontsize=25)
    ax12.text(0.1,0.07,  r'\rm{$p_T$ $>$ 35 GeV}',fontsize=25)

    ax21.text(1.9,525,   r'\rm{$p_T$ $>$ 25 GeV}',fontsize=25)

    ax22.text(0.1,660,   r'\rm{$p_T$ $>$ 20 GeV}',fontsize=25)
    ax22.text(0.1,520,   r'\rm{$p_T$ $>$ 25 GeV}',fontsize=25)

    ax22.text(1.3,425,   r'\rm{$p_T$ $>$ 20 GeV}',fontsize=25)
    ax22.text(1.3,260,   r'\rm{$p_T$ $>$ 25 GeV}',fontsize=25)


    ax21.text(2.40,690,  r'\rm{$W^+$}'           ,fontsize=25)
    ax21.text(2.40,390,  r'\rm{$W^-$}'           ,fontsize=25)

    ax22.text(2.40,550,  r'\rm{$W^+$}'           ,fontsize=25)
    ax22.text(2.40,300,  r'\rm{$W^-$}'           ,fontsize=25)


    for ax in [ax11,ax12,ax21,ax22]:
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    for ax in [ax11]:
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    for ax in [ax12]:
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    for ax in [ax21]:
        minorLocator = MultipleLocator(10)
        majorLocator = MultipleLocator(100)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.set_yticks([300,400,500,600,700,800])
        ax.set_yticklabels([r'',r'$400$',r'',r'$600$',r'',r'$800$'])

    for ax in [ax22]:
        minorLocator = MultipleLocator(10)
        majorLocator = MultipleLocator(100)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.set_yticks([300,400,500,600,700,800])
        ax.set_yticklabels([r'',r'$400$',r'',r'$600$',r'',r'$800$'])



    #######################
    #--plot data/theory
    #######################

    for idx in data:
        if idx==2000: ax,color,marker,ms = ax31,'firebrick',     '.', 10 
        if idx==2003: ax,color,marker,ms = ax31,'darkgreen',     '*', 8 
        if idx==2006: ax,color,marker,ms = ax31,'darkturquoise', '^', 8 
        if idx==2007: ax,color,marker,ms = ax41,'darkgreen',     '.', 10 
        if idx==2015: ax,color,marker,ms = ax42,'firebrick',     '.', 10 
        if idx==2009: ax,color,marker,ms = ax42,'darkgreen',     '*', 8 
        if idx==2010: ax,color,marker,ms = ax41,'firebrick',     '*', 8 
        if idx==2011: ax,color,marker,ms = ax32,'darkturquoise', '.', 10 
        if idx==2012: ax,color,marker,ms = ax32,'darkgreen',     '*', 8 
        if idx==2013: ax,color,marker,ms = ax32,'firebrick',     '^', 8 
        if idx==2014: ax,color,marker,ms = ax32,'purple',        'p', 8 
        for ic in range(nc):
            #color = colors[ic]
            eta0, eta1 = [], []
            thy0, thy1, ratio0, ratio1, alpha0, alpha1 = [], [], [], [], [], []
            boson = conf['wzrv tabs'][idx]['boson']
            #--separate W+ data from W- data
            for j in range(len(boson)):
                if boson[j]=='W':
                    if 'eta' in conf['wzrv tabs'][idx]: eta = conf['wzrv tabs'][idx]['eta']
                    else: eta = (conf['wzrv tabs'][idx]['eta_min'] + conf['wzrv tabs'][idx]['eta_max'])/2.0
                    thy   = data[idx]['thy-%d'%ic]
                    ratio = data[idx]['value']/thy
                    alpha = data[idx]['alpha']
                    break
                if boson[j]=='W+':
                    if 'eta' in conf['wzrv tabs'][idx]: eta0.append(conf['wzrv tabs'][idx]['eta'][j])
                    else: eta0.append((conf['wzrv tabs'][idx]['eta_min'][j] + conf['wzrv tabs'][idx]['eta_max'][j])/2.0)
                    thy0.append(data[idx]['thy-%d'%ic][j])
                    ratio0.append(data[idx]['value'][j]/thy0[j])
                    alpha0.append(data[idx]['alpha'][j])
                if boson[j]=='W-':
                    if 'eta' in conf['wzrv tabs'][idx]: eta1.append(conf['wzrv tabs'][idx]['eta'][j])
                    else: eta1.append((conf['wzrv tabs'][idx]['eta_min'][j] + conf['wzrv tabs'][idx]['eta_max'][j])/2.0)
                    thy1.append(data[idx]['thy-%d'%ic][j])
                    ratio1.append(data[idx]['value'][j]/data[idx]['thy-%d'%ic][j])
                    alpha1.append(data[idx]['alpha'][j])

            if boson[0]=='W':  
                ax.errorbar(eta,ratio,yerr=alpha/thy,color=color,linestyle='none',marker=marker,ms=ms,capsize=3.0)
                ax.axhline(1,0,3,alpha=1.0,color='black',ls='--')
                #ax.text(0.02,0.80,label,fontsize=25,transform = ax.transAxes)
            else:  
                ax.errorbar(np.array(eta0)    ,ratio0,yerr=np.array(alpha0)/np.array(thy0),color=color,linestyle='none',marker=marker,ms=ms,capsize=3.0)
                ax.errorbar(np.array(eta1)+0.1,ratio1,yerr=np.array(alpha1)/np.array(thy1),color=color,linestyle='none',marker=marker,ms=ms,capsize=3.0)
                ax.axhline(1,0,3,alpha=1.0,color='black',ls='--')
                ax.axhline(1,0,3,alpha=1.0,color='black',ls='--')
                #ax.text(0.02,0.80,label+r' $\textbf{\textrm{$\mathbf{(W^+)}$}}$',fontsize=25,transform = axp.transAxes)
                #ax.text(0.02,0.80,label+r' $\textbf{\textrm{$\mathbf{(W^-)}$}}$',fontsize=25,transform = axm.transAxes)
             

    for ax in [ax41,ax42]: 
        ax.set_xlabel(r'\boldmath$|\eta|$',size=30)
        ax.xaxis.set_label_coords(0.85,-0.02)
        ax.set_xticks([0,1,2])

    for ax in [ax31,ax32]:
        ax.tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
        ax.set_xlim(0,3.0)

    for ax in [ax41]:
        ax.tick_params(axis='x',which='both',top=True,direction='in',labelsize=30)
        ax.tick_params(axis='y',which='both',right=True,direction='in',labelsize=30)
        ax.set_xlim(0,3.0)

    for ax in [ax42]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.set_xlim(0,3.0)


    for ax in [ax31,ax32,ax41,ax42]:
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(1.0)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    for ax in [ax31]:
        ax.set_ylim(0.60,1.40)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.set_yticks([0.7,1,1.3])
        ax.set_yticklabels([r'$0.70$',r'$1.00$',r'$1.30$'])

    for ax in [ax32]:
        ax.set_ylim(0.75,1.25)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.set_yticks([0.8,1,1.2])
        ax.set_yticklabels([r'$0.80$',r'$1.00$',r'$1.20$'])

    for ax in [ax41]:
        ax.set_ylim(0.95,1.05)
        minorLocator = MultipleLocator(0.04)
        majorLocator = MultipleLocator(0.2)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        ax.set_yticks([0.97,1,1.03])
        ax.set_yticklabels([r'$0.97$',r'$1.00$',r'$1.03$'])

    for ax in [ax42]:
        ax.set_ylim(0.93,1.07)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.05)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)

    handles = [hand[2000],hand[2003],hand[2006]]
    label1 = r'\textbf{\textrm{D0(e)}}'
    label2 = r'\textbf{\textrm{CDF(e)}}'
    label3 = r'\textbf{\textrm{D0(\boldmath$\mu$)}}'
    labels = [label1,label2,label3] 
    ax31.legend(handles,labels,frameon=False,fontsize=20,loc='upper right', ncol = 1, handletextpad = 0.3, handlelength = 1.0)

    handles = [hand[2011],hand[2012],hand[2013],hand[2014]]
    label1 = r'\textbf{\textrm{CMS(\boldmath$\mu$)(2011)}}'
    label2 = r'\textbf{\textrm{CMS(e)(2011)}}'
    label3 = r'\textbf{\textrm{CMS(e)(2010)}}'
    label4 = r'\textbf{\textrm{CMS(\boldmath$\mu$)(2010)}}'
    labels = [label1,label2,label3,label4] 
    ax32.legend(handles,labels,frameon=False,fontsize=20,loc='upper right', ncol = 2, handletextpad = 0.3, handlelength = 1.0, columnspacing = 1.0)

    handles = [hand[2007],hand[2010]]
    label1 = r'\textbf{\textrm{ATLAS(2012)}}'
    label2 = r'\textbf{\textrm{CMS}}'
    labels = [label1,label2] 
    ax41.legend(handles,labels,frameon=False,fontsize=20,loc='upper right', ncol = 1, handletextpad = 0.3, handlelength = 1.0)

    handles = [hand[2015],hand[2009]]
    label1 = r'\textbf{\textrm{ATLAS(2011)}}'
    label2 = r'\textbf{\textrm{ATLAS(2010)}}'
    labels = [label1,label2] 
    ax42.legend(handles,labels,frameon=False,fontsize=20,loc='upper right',ncol = 1, handletextpad = 0.3, handlelength = 1.0)


    ax41.set_ylabel(r'\textbf{\textrm{data/theory}}',size=30)
    ax41.yaxis.set_label_coords(-0.15,1.00)

    py.tight_layout()
    py.subplots_adjust(hspace = 0, wspace=0.15)

    checkdir('%s/gallery'%cwd)
    filename='%s/gallery/lepton'%(cwd)
    filename += '.png'
    #filename += '.pdf'
    ax11.set_rasterized(True)
    ax12.set_rasterized(True)
    ax21.set_rasterized(True)
    ax22.set_rasterized(True)
    ax31.set_rasterized(True)
    ax32.set_rasterized(True)
    ax41.set_rasterized(True)
    ax42.set_rasterized(True)

    py.savefig(filename)
    print()
    print('Saving wzrv plot to %s'%filename)












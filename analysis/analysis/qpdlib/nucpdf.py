#!/usr/bin/env python
import os,sys
from tools.config import load_config,conf
from fitlib.resman import RESMAN
import numpy as np

#--matplotlib
import matplotlib
import pylab  as py
from matplotlib.ticker import MultipleLocator

#--from tools
from tools.tools import load,lprint,save

#--from corelib
from analysis.corelib import core,classifier

from qcdlib import pdf as PDF
from obslib.idis.theory import OFFSHELL_MODEL

#--generate nuclear PDFs
def gen_nuclear_pdf(wdir,Q2):

    replicas = core.get_replicas(wdir)

    load_config('%s/input.py'%wdir)

    istep = core.get_istep()
    core.mod_conf(istep,replicas[0])
    #conf['idis grid'] = 'prediction'
    resman=RESMAN(parallel=False,datasets=False)
    parman = resman.parman

    replicas = core.get_replicas(wdir)
    names    = core.get_replicas_names(wdir)

    Xgrid = np.geomspace(1e-5,1e-1,20)
    Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    Q2grid = [Q2*0.99,Q2*1.01]
    conf['idis grid'] = {}
    conf['idis grid']['X']  = Xgrid 
    conf['idis grid']['Q2'] = Q2grid 
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    jar = load('%s/data/jar-%d.dat'%(wdir,istep))
    parman.order = jar['order']
    replicas = jar['replicas']
    resman.setup_idis()
    
    idis=resman.idis_thy
    idis.data['p']['F2'] = np.array(idis.X.size)
    idis.data['n']['F2'] = np.array(idis.X.size)
   
    off = conf['off pdf']

    offshell_model = OFFSHELL_MODEL()

    ##############################################
    #--generate offshell
    ##############################################
    X1   = 10**np.linspace(-4,-1,100)
    X2   = np.linspace(0.1,0.98,100)
    X    = np.append(X1,X2)
    cnt = 0

    gX = idis.gX
    gW = idis.gW
    XM,  gXM = np.meshgrid(X,gX)
    Q2M, gWM = np.meshgrid(Q2,gW)
    a = XM
    pdf    = conf['pdf']
    mellin = idis.mellin
    N = mellin.N

    #nuclei = ['p','n']
    nuclei = ['p']
    NPDF = {_: {} for _ in ['up','do']}
    NPDF['up'] = {_:{} for _ in nuclei}
    NPDF['do'] = {_:{} for _ in nuclei}
    for _ in nuclei:
        NPDF['up'][_]['d'] = {_: [] for _ in ['onshell','offshell','total']}
        NPDF['up'][_]['h'] = {_: [] for _ in ['onshell','offshell','total']}
        NPDF['up'][_]['t'] = {_: [] for _ in ['onshell','offshell','total']}
        NPDF['do'][_]['d'] = {_: [] for _ in ['onshell','offshell','total']}
        NPDF['do'][_]['h'] = {_: [] for _ in ['onshell','offshell','total']}
        NPDF['do'][_]['t'] = {_: [] for _ in ['onshell','offshell','total']}
    NPDF['X'] = X
    for par in replicas:
         
        lprint('Generating nuclear PDFs %s/%s'%(cnt+1,len(replicas)))
        parman.set_new_params(par)

        for nucleon in nuclei:
            for nucleus in ['d','h','t']:
                if nucleus=='d':
                    b   = idis.ymaxD
                    smf = idis.dsmf
                    YM = 0.5*(b-a)*gXM + 0.5*(a+b)
                    JM = 0.5*(b-a)
                    XM_YM = XM/YM
                    fon22 = smf.get_fXX2('f22','onshell' ,XM,Q2M,YM)
                    fof22 = smf.get_fXX2('f22','offshell',XM,Q2M,YM)
                if nucleus=='h': 
                    b = idis.ymaxH
                    smf = idis.hsmf
                    YM = 0.5*(b-a)*gXM + 0.5*(a+b)
                    JM = 0.5*(b-a)
                    XM_YM = XM/YM
                    fon22 = smf.get_fXX2('f22%s'%nucleon,'onshell' ,XM,Q2M,YM)
                    fof22 = smf.get_fXX2('f22%s'%nucleon,'offshell',XM,Q2M,YM)
                if nucleus=='t': 
                    b = idis.ymaxT
                    smf = idis.hsmf
                    YM = 0.5*(b-a)*gXM + 0.5*(a+b)
                    JM = 0.5*(b-a)
                    XM_YM = XM/YM
                    if nucleon=='p':
                        fon22 = smf.get_fXX2('f22n','onshell' ,XM,Q2M,YM)
                        fof22 = smf.get_fXX2('f22n','offshell',XM,Q2M,YM)
                    if nucleon=='n':
                        fon22 = smf.get_fXX2('f22p','onshell' ,XM,Q2M,YM)
                        fof22 = smf.get_fXX2('f22p','offshell',XM,Q2M,YM)

                U , D  = np.zeros(np.shape(XM_YM)), np.zeros(np.shape(XM_YM))
                dU, dD = np.zeros(np.shape(XM_YM)), np.zeros(np.shape(XM_YM))
                for i in range(len(XM_YM)):
                    for j in range(len(XM_YM[0])):
                        U[i][j]  = pdf.get_xF(XM_YM[i][j],Q2,'u')/XM_YM[i][j]
                        D[i][j]  = pdf.get_xF(XM_YM[i][j],Q2,'d')/XM_YM[i][j]
                        dU[i][j] = off.get_xF(XM_YM[i][j],Q2,'u')/XM_YM[i][j]
                        dD[i][j] = off.get_xF(XM_YM[i][j],Q2,'d')/XM_YM[i][j]

                U , D  = np.array(U) , np.array(D)
                dU, dD = np.array(dU), np.array(dD)

                #--switch onshell pieces
                if nucleon=='n': D, U = U, D

                #--switch offshell pieces
                q = [0,dU, dD]
                q = offshell_model.get_model(q,nucleon,nucleus)
                dU, dD = q[1],q[2]

                uon   = fon22*U
                don   = fon22*D
                uoff  = fof22*dU
                doff  = fof22*dD

                uon  = X*np.einsum('ij,ij,ij->j',gWM,JM,uon /YM)
                don  = X*np.einsum('ij,ij,ij->j',gWM,JM,don /YM)
                uoff = X*np.einsum('ij,ij,ij->j',gWM,JM,uoff/YM)
                doff = X*np.einsum('ij,ij,ij->j',gWM,JM,doff/YM)

                NPDF['up'][nucleon][nucleus]['onshell'] .append(uon)
                NPDF['up'][nucleon][nucleus]['offshell'].append(uoff)
                NPDF['up'][nucleon][nucleus]['total']   .append(uon+uoff)
                NPDF['do'][nucleon][nucleus]['onshell'] .append(don)
                NPDF['do'][nucleon][nucleus]['offshell'].append(doff)
                NPDF['do'][nucleon][nucleus]['total']   .append(don+doff)


        cnt +=1
     
    if Q2 == 1.27**2: filename = '%s/data/nuclear-pdf.dat'%(wdir)
    else:             filename = '%s/data/nuclear-pdf-Q2=%d.dat'%(wdir,Q2)
    save(NPDF,filename) 
    print()
    print('Saving nuclear PDF data to %s'%filename)

def plot_nuclear_pdf(PLOT,kc,mode=1):

    nrows,ncols=3,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)
    ax21 = py.subplot(nrows,ncols,3)
    ax22 = py.subplot(nrows,ncols,4)
    ax31 = py.subplot(nrows,ncols,5)
    ax32 = py.subplot(nrows,ncols,6)

    j = 0
    hand = {}
    thy  = {}
    for plot in PLOT: 
        wdir, q2, color, style, label, alpha, zorder= plot[0], plot[1], plot[2], plot[3], plot[4], plot[5], plot[6]
        replicas = core.get_replicas(wdir)

        load_config('%s/input.py'%wdir)

        istep = core.get_istep()
        core.mod_conf(istep,replicas[0])
        resman=RESMAN(parallel=False,datasets=False)
        parman = resman.parman
        cluster,colors,nc,cluster_order= classifier.get_clusters(wdir,istep,kc) 

        replicas = core.get_replicas(wdir)
        names    = core.get_replicas_names(wdir)

        jar = load('%s/data/jar-%d.dat'%(wdir,istep))
        parman.order = jar['order']
        replicas = jar['replicas']
        
        if 'off pdf' in conf: off = conf['off pdf']
        else:
            print('Offshell corrections not present.')
            return

        if 'off pdf' not in conf['steps'][istep]['active distributions']: 
            if 'off pdf' not in conf['steps'][istep]['passive distributions']: 
                return

        #--try to load data.  Generate it if it does not exist
        try:
            if q2 == 1.27**2: data = load('%s/data/nuclear-pdf.dat'%(wdir))
            else:             data = load('%s/data/nuclear-pdf-Q2=%d.dat'%(wdir,q2))
        except:
            gen_nuclear_pdf(wdir,q2)
            if q2 == 1.27**2: data = load('%s/data/nuclear-pdf.dat'%(wdir))
            else:             data = load('%s/data/nuclear-pdf-Q2=%d.dat'%(wdir,q2))

        if q2 == 1.27**2: pdfs = load('%s/data/pdf-%s.dat'%(wdir,istep))
        else:             pdfs = load('%s/data/pdf-%s-Q2=%d.dat'%(wdir,istep,q2))

        ##############################################
        #--plot offshell
        ##############################################
        X   = np.array(data['X'])

        for pdf in data:
            if pdf=='X': continue
            for nucleon in data[pdf]:
                for nucleus in data[pdf][nucleon]:

                    if pdf=='up':
                        if nucleon=='p': ax,axoff,axrat = ax11,ax21,ax31
                        else:            continue
                    if pdf=='do':
                        if nucleon=='p': ax,axoff,axrat = ax12,ax22,ax32
                        else:            continue

                    if nucleus=='d': color,alpha='darkblue' ,0.7
                    if nucleus=='h': color,alpha='magenta'    ,0.7
                    if nucleus=='t': color,alpha='goldenrod',0.4

                    qon  = np.array(data[pdf][nucleon][nucleus]['onshell'])
                    qoff = np.array(data[pdf][nucleon][nucleus]['offshell'])
                    q    = qon + qoff
                    qrat = qoff/q

                    meanon  = np.mean(qon,axis=0)
                    stdon   = np.std (qon,axis=0)               
                    meanoff = np.mean(qoff,axis=0)
                    stdoff  = np.std (qoff,axis=0)               
                    mean    = np.mean(q,axis=0)
                    std     = np.std (q,axis=0)               
                    meanrat = np.mean(qrat,axis=0)
                    stdrat  = np.std (qrat,axis=0)               

                    if mode == 0:
                        for i in range(len(q)):
                            hand[nucleus] ,= ax.plot(X,q[i]   ,color=color,alpha=alpha,zorder=zorder)
                            axoff              .plot(X,qoff[i],color=color,alpha=alpha,zorder=zorder)
                            axrat              .plot(X,qrat[i],color=color,alpha=alpha,zorder=zorder)
                    if mode == 1:
                        hand[nucleus] = ax.fill_between(X,mean   -std   ,mean   +std   ,color=color,alpha=alpha,zorder=zorder,hatch=style)
                        axoff             .fill_between(X,meanoff-stdoff,meanoff+stdoff,color=color,alpha=alpha,zorder=zorder,hatch=style)
                        axrat             .fill_between(X,meanrat-stdrat,meanrat+stdrat,color=color,alpha=alpha,zorder=zorder,hatch=style)
 
        u = np.array(pdfs['XF']['u'])
        d = np.array(pdfs['XF']['d'])
        meanu = np.mean(u,axis=0)
        stdu  = np.std (u,axis=0)
        meand = np.mean(d,axis=0)
        stdd  = np.std (d,axis=0)
        color, alpha = 'darkgreen',0.5
        if mode==0:
            for i in range(len(u)):
                hand['free'] ,= ax11.plot(X,u[i],color=color,alpha=alpha,zorder=zorder)
                ax12                .plot(X,d[i],color=color,alpha=alpha,zorder=zorder)
        if mode==1:
            hand['free'] = ax11.fill_between(X,meanu-stdu,meanu+stdu,color=color,alpha=alpha,zorder=zorder,hatch=style)
            ax12               .fill_between(X,meand-stdd,meand+stdd,color=color,alpha=alpha,zorder=zorder,hatch=style)
        ax21.axhline(0,0,1,color=color,alpha=alpha)
        ax22.axhline(0,0,1,color=color,alpha=alpha)
        ax31.axhline(0,0,1,color=color,alpha=alpha)
        ax32.axhline(0,0,1,color=color,alpha=alpha)


 
        j+=1
 
    ##############################################

    ax12.text(0.60,0.30,r'$Q^2=%s{\rm~GeV^2}$'%q2,size=30,transform=ax12.transAxes)

    ax11.text(0.05,0.05,r'\boldmath$x u^{p/A}$' ,transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.05,r'\boldmath$x d^{p/A}$' ,transform=ax12.transAxes,size=40)
    ax21.text(0.60,0.80,r'\boldmath$x u^{p/A ({\rm off)}}$',transform=ax21.transAxes,size=40)
    ax22.text(0.60,0.80,r'\boldmath$x d^{p/A ({\rm off)}}$',transform=ax22.transAxes,size=40)
    ax31.text(0.05,0.80,r'\boldmath$  u^{p/A ({\rm off)}}/u^{p/A}$',transform=ax31.transAxes,size=40)
    ax32.text(0.05,0.80,r'\boldmath$  d^{p/A ({\rm off)}}/d^{p/A}$',transform=ax32.transAxes,size=40)
 
    for ax in [ax11,ax12,ax21,ax22,ax31,ax32]:
        ax.set_xlim(0.01,0.90)
        ax.set_xlabel(r'\boldmath$x$'         ,size=30)
        ax.xaxis.set_label_coords(0.98,0.00)
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticks([0.2,0.4,0.6,0.8])

    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30,labelbottom=False)

    for ax in [ax21,ax22]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30,labelbottom=False)
        ax.axhline(0,0,1,color='black',ls=':',alpha=0.5)

    for ax in [ax31,ax32]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.axhline(0,0,1,color='black',ls=':',alpha=0.5)

    ax11.set_ylim(0,0.70)
    ax12.set_ylim(0,0.45)
    ax21.set_ylim(-0.015,0.022)
    ax22.set_ylim(-0.015,0.022)
    ax31.set_ylim(-0.04,0.12)
    ax32.set_ylim(-0.28,0.18)

    minorLocator = MultipleLocator(0.05)
    majorLocator = MultipleLocator(0.2)
    ax11.yaxis.set_minor_locator(minorLocator)
    ax11.yaxis.set_major_locator(majorLocator)
    ax11.set_yticks([0.2,0.4,0.6])

    minorLocator = MultipleLocator(0.05)
    majorLocator = MultipleLocator(0.2)
    ax12.yaxis.set_minor_locator(minorLocator)
    ax12.yaxis.set_major_locator(majorLocator)
    ax12.set_yticks([0.2,0.4])

    minorLocator = MultipleLocator(0.0025)
    majorLocator = MultipleLocator(0.01)
    ax21.yaxis.set_minor_locator(minorLocator)
    ax21.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(0.0025)
    majorLocator = MultipleLocator(0.01)
    ax22.yaxis.set_minor_locator(minorLocator)
    ax22.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(0.01)
    majorLocator = MultipleLocator(0.05)
    ax31.yaxis.set_minor_locator(minorLocator)
    ax31.yaxis.set_major_locator(majorLocator)

    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.10)
    ax32.yaxis.set_minor_locator(minorLocator)
    ax32.yaxis.set_major_locator(majorLocator)


    handles,labels = [],[]
    handles.append(hand['free'])
    handles.append(hand['d'])
    handles.append(hand['h'])
    handles.append(hand['t'])
    labels.append(r'\textrm{\textbf{Free}}')
    labels.append(r'\boldmath$d$')
    labels.append(r'\boldmath$^3 {\rm He}$')
    labels.append(r'\boldmath$^3 {\rm H}$')
    ax12.legend(handles,labels,frameon=False,loc='upper right',fontsize=28, handletextpad = 0.5, handlelength = 1.5, ncol = 1, columnspacing = 0.5)


    py.tight_layout()
    py.subplots_adjust(hspace=0)


    filename = '%s/gallery/nuclear-pdfs'%PLOT[0][0]
    if mode==1: filename+='-bands'
    filename += '.png'
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()

def plot_nuclear_pdf_rat(PLOT,kc,mode=1):

    nrows,ncols=1,2
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)

    j = 0
    hand = {}
    thy  = {}
    for plot in PLOT: 
        wdir, q2, color, style, label, alpha, zorder= plot[0], plot[1], plot[2], plot[3], plot[4], plot[5], plot[6]
        replicas = core.get_replicas(wdir)

        load_config('%s/input.py'%wdir)

        istep = core.get_istep()
        core.mod_conf(istep,replicas[0])
        resman=RESMAN(parallel=False,datasets=False)
        parman = resman.parman
        cluster,colors,nc,cluster_order= classifier.get_clusters(wdir,istep,kc) 

        replicas = core.get_replicas(wdir)
        names    = core.get_replicas_names(wdir)

        jar = load('%s/data/jar-%d.dat'%(wdir,istep))
        parman.order = jar['order']
        replicas = jar['replicas']
        
        if 'off pdf' in conf: off = conf['off pdf']
        else:
            print('Offshell corrections not present.')
            return

        if 'off pdf' not in conf['steps'][istep]['active distributions']: 
            if 'off pdf' not in conf['steps'][istep]['passive distributions']: 
                return

        #--try to load data.  Generate it if it does not exist
        try:
            if q2 == 1.27**2: data = load('%s/data/nuclear-pdf.dat'%(wdir))
            else:             data = load('%s/data/nuclear-pdf-Q2=%d.dat'%(wdir,q2))
        except:
            gen_nuclear_pdf(wdir,q2)
            if q2 == 1.27**2: data = load('%s/data/nuclear-pdf.dat'%(wdir))
            else:             data = load('%s/data/nuclear-pdf-Q2=%d.dat'%(wdir,q2))


        ##############################################
        #--plot offshell
        ##############################################
        X   = np.array(data['X'])

        upT = np.array(data['up']['p']['t']['total'])
        upH = np.array(data['up']['p']['h']['total'])
        dpT = np.array(data['do']['p']['t']['total'])
        dpH = np.array(data['do']['p']['h']['total'])

        urat = (upT - upH)/(upT + upH) 
        drat = (dpT - dpH)/(dpT + dpH) 

        #new = []
        #for i in range(len(urat)):
        #    val = urat[i][-20] #--x = 0.81
        #    if val < 0: continue
        #    new.append(urat[i])

        #urat = np.array(new)

        meanu = np.mean(urat,axis=0)
        stdu  = np.std (urat,axis=0)
        meand = np.mean(drat,axis=0)
        stdd  = np.std (drat,axis=0)
       
        color = 'firebrick'
        alpha = 0.8

        if mode==0:
            for i in range(len(urat)): 
                ax11.plot(X,urat[i],color=color,alpha=alpha,zorder=zorder,ls=style)
            for i in range(len(drat)): 
                ax12.plot(X,drat[i],color=color,alpha=alpha,zorder=zorder,ls=style)
        if mode==1: 
            ax11.fill_between(X,meanu-stdu,meanu+stdu,color=color,alpha=alpha,zorder=zorder,hatch=style)
            ax12.fill_between(X,meand-stdd,meand+stdd,color=color,alpha=alpha,zorder=zorder,hatch=style)
  
        j+=1
 
    ##############################################

    ax12.text(0.05,0.05,r'$Q^2=%s{\rm~GeV^2}$'%q2,size=30,transform=ax11.transAxes)

    #ax11.text(0.05,0.80,r'\boldmath$(u^{p/^3 {\rm H}}-u^{p/^3 {\rm He}})/(u^{p/^3 {\rm H}}+u^{p/^3 {\rm He}})$' ,transform=ax11.transAxes,size=40)
    #ax12.text(0.05,0.80,r'\boldmath$(d^{p/^3 {\rm H}}-d^{p/^3 {\rm He}})/(d^{p/^3 {\rm H}}+d^{p/^3 {\rm He}})$' ,transform=ax12.transAxes,size=40)
    ax11.text(0.05,0.80,r'\boldmath$\frac{u^{p/^3 {\rm H}}-u^{p/^3 {\rm He}}}{u^{p/^3 {\rm H}}+u^{p/^3 {\rm He}}}$' ,transform=ax11.transAxes,size=40)
    ax12.text(0.05,0.10,r'\boldmath$\frac{d^{p/^3 {\rm H}}-d^{p/^3 {\rm He}}}{d^{p/^3 {\rm H}}+d^{p/^3 {\rm He}}}$' ,transform=ax12.transAxes,size=40)
 
    minorLocator = MultipleLocator(0.02)
    majorLocator = MultipleLocator(0.1)
    ax11.xaxis.set_minor_locator(minorLocator)
    ax11.xaxis.set_major_locator(majorLocator)
    for ax in [ax11,ax12]:
        ax.set_xlim(0.01,0.90)
        ax.set_xlabel(r'\boldmath$x$',size=30)
        ax.xaxis.set_label_coords(0.98,0.00)
        minorLocator = MultipleLocator(0.05)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticks([0.2,0.4,0.6,0.8])

    for ax in [ax11,ax12]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.axhline(0,0,1,color='black',ls=':',alpha=0.5)

    ax11.set_ylim(-0.05,0.07)
    ax12.set_ylim(-0.15,0.05)
    #handles,labels = [],[]
    #handles.append(hand['d'])
    #handles.append(hand['h'])
    #handles.append(hand['t'])
    #labels.append(r'\boldmath$d$')
    #labels.append(r'\boldmath$^3 {\rm He}$')
    #labels.append(r'\boldmath$^3 {\rm H}$')
    #ax12.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.5, ncol = 1, columnspacing = 0.5)


    #py.tight_layout()
    #py.subplots_adjust(hspace=0,wspace=0)

    filename = '%s/gallery/nuclear-pdfs-rat'%PLOT[0][0]
    if mode==1: filename += '-bands'
    filename += '.png'
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()

def plot_tony_request(PLOT,kc,mode=1):

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)

    j = 0
    hand = {}
    thy  = {}
    for plot in PLOT: 
        wdir, q2, color, style, label, alpha, zorder= plot[0], plot[1], plot[2], plot[3], plot[4], plot[5], plot[6]
        replicas = core.get_replicas(wdir)

        load_config('%s/input.py'%wdir)

        istep = core.get_istep()
        core.mod_conf(istep,replicas[0])
        resman=RESMAN(parallel=False,datasets=False)
        parman = resman.parman
        cluster,colors,nc,cluster_order= classifier.get_clusters(wdir,istep,kc) 

        replicas = core.get_replicas(wdir)
        names    = core.get_replicas_names(wdir)

        jar = load('%s/data/jar-%d.dat'%(wdir,istep))
        parman.order = jar['order']
        replicas = jar['replicas']
        
        if 'off pdf' in conf: off = conf['off pdf']
        else:
            print('Offshell corrections not present.')
            return

        if 'off pdf' not in conf['steps'][istep]['active distributions']: 
            if 'off pdf' not in conf['steps'][istep]['passive distributions']: 
                return

        #--try to load data.  Generate it if it does not exist
        try:
            if q2 == 1.27**2: data = load('%s/data/nuclear-pdf.dat'%(wdir))
            else:             data = load('%s/data/nuclear-pdf-Q2=%d.dat'%(wdir,q2))
        except:
            gen_nuclear_pdf(wdir,q2)
            if q2 == 1.27**2: data = load('%s/data/nuclear-pdf.dat'%(wdir))
            else:             data = load('%s/data/nuclear-pdf-Q2=%d.dat'%(wdir,q2))

        #--load free PDFs
        if q2 == 1.27**2: fdata = load('%s/data/pdf-%s.dat'%(wdir,istep))
        else:             fdata = load('%s/data/pdf-%s-Q2=%d.dat'%(wdir,istep,q2))

        ##############################################
        #--plot offshell
        ##############################################
        X   = np.array(data['X'])

        upH = np.array(data['up']['p']['h']['total'])
        dpH = np.array(data['do']['p']['h']['total'])
        unH = np.array(data['do']['p']['t']['total'])
        dnH = np.array(data['up']['p']['t']['total'])
       
 
        u   = np.array(fdata['XF']['u'])
        d   = np.array(fdata['XF']['d'])

        uH  = 2*upH + unH
        dH  = 2*dpH + dnH

        q3H = uH - dH
        q3  = u  - d

        rat = q3H/q3

        mean = np.mean(rat,axis=0)
        std  = np.std (rat,axis=0)
      
 
        color = 'firebrick'
        alpha = 0.8

        if mode==0:
            for i in range(len(rat)): 
                ax11.plot(X,rat[i],color=color,alpha=alpha,zorder=zorder,ls=style)
        if mode==1: 
            ax11.fill_between(X,mean-std,mean+std,color=color,alpha=alpha,zorder=zorder,hatch=style)
  
        j+=1
 
    ##############################################

    ax11.text(0.05,0.05,r'$Q^2=%s{\rm~GeV^2}$'%q2,size=30,transform=ax11.transAxes)

    ax11.text(0.05,0.80,r'\boldmath$R^{(^3 {\rm He})}$' ,transform=ax11.transAxes,size=40)
 
    #minorLocator = MultipleLocator(0.02)
    #majorLocator = MultipleLocator(0.1)
    #ax11.xaxis.set_minor_locator(minorLocator)
    #ax11.xaxis.set_major_locator(majorLocator)
    for ax in [ax11]:
        ax.semilogx()
        ax.set_xlim(0.01,0.90)
        ax.set_xlabel(r'\boldmath$x$',size=30)
        ax.xaxis.set_label_coords(0.98,0.00)
        #minorLocator = MultipleLocator(0.05)
        #majorLocator = MultipleLocator(0.2)
        #ax.xaxis.set_minor_locator(minorLocator)
        #ax.xaxis.set_major_locator(majorLocator)
        #ax.set_xticks([0.2,0.4,0.6,0.8])

    for ax in [ax11]:
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.axhline(1,0,1,color='black',ls=':',alpha=0.5)

    ax11.set_ylim(0.5,1.5)


    py.tight_layout()


    filename = '%s/gallery/nuclear-pdfs-tony'%PLOT[0][0]
    if mode==1: filename += '-bands'
    filename += '.png'
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()















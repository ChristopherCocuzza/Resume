import os
import numpy as np
import math

import matplotlib
matplotlib.use('Agg')
import pylab as py


#--from tools
from tools.tools     import checkdir,save,load
import tools.config
from tools.config    import load_config, conf, options
from tools.inputmod  import INPUTMOD

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

def plot_params(wdir,dist,hist=False):

    #--hist: If False, plot point by point.  If True, plot histogram.

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas = core.get_replicas(wdir) 
    core.mod_conf(istep,replicas[0])

    if dist not in conf['steps'][istep]['active distributions']:
        return

    clusters,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    _order = replicas[0]['order'][istep]

    #--get correct order from dist
    order  = []
    idx    = []
    for i in range(len(_order)):
        if _order[i][1] != dist: continue
        order.append(_order[i][2])
        idx.append(i)

    #--get correct params from dist
    params = np.zeros((len(order),len(replicas)))
    for i in range(len(order)):
        for j in range(len(replicas)):
            params[i][j] = replicas[j]['params'][istep][idx[i]]

    #--sort alphabetically
    z = sorted(zip(order,params))
    order  = [z[i][0] for i in range(len(z))]
    params = [z[i][1] for i in range(len(z))]

    #--get names for organization
    try:
        names = [(order[i].split()[0],order[i].split()[1]) for i in range(len(order))]
        n0    = sorted(list(set(names[i][0] for i in range(len(names)))))
        n1    = sorted(list(set(names[i][1] for i in range(len(names)))))
    except:
        names = [(order[i].split()[0]) for i in range(len(order))]
        n0    = sorted(list(set(names[i][0] for i in range(len(names)))))
        n1    = sorted(list(set('' for i in range(len(names)))))

    #--create plot with enough space for # of parameters
    nrows,ncols = len(n0),len(n1)
    fig = py.figure(figsize=(ncols*7,nrows*4))
    X = np.linspace(1,len(replicas),len(replicas))

    #--create plot
    for i in range(len(order)):
        j  = n0.index(names[i][0])
        k  = [names[m][1] for m in range(len(names)) if names[m][0]==n0[j]].index(names[i][1])
        idx = j*ncols + k + 1
        ax = py.subplot(nrows,ncols, idx)
        ax.set_title('%s'%(order[i]), size=30)
        for j in range(nc):
            color  = colors[clusters[j]]
            par = [params[i][k] for k in range(len(params[i])) if clusters[k]==j]
            mean = np.mean(par)
            std  = np.std(par)
            meanl = r'mean: %6.5f'%mean
            stdl  = r'std: %6.5f'%std
            if hist:
                ax.hist(par,color=color,alpha=0.6,edgecolor='black')
                ax.axvline(mean,ymin=0,ymax=1,ls='--',color=color,alpha=0.8,label=meanl)
                ax.axvspan(mean-std,mean+std,alpha=0.2,color=color,label=stdl)
            else:
                ax.scatter(X,par,color=color) 
                ax.axhline(mean,xmin=0,xmax=1,ls='--',color=color,alpha=0.8)
                ax.axhspan(mean-std,mean+std,alpha=0.2,color=color)
        #ax.legend(loc='best',frameon=False,fontsize=15)
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20)

    if hist: filename='%s/gallery/params-%s-hist.png'%(wdir,dist)
    else:    filename='%s/gallery/params-%s.png'%(wdir,dist)
    checkdir('%s/gallery'%wdir)
    py.tight_layout()
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()
     
def plot_norms(wdir,exp):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas = core.get_replicas(wdir) 
    core.mod_conf(istep,replicas[0])

    if exp not in conf['steps'][istep]['datasets']: return

    #--choose histogram range based on experiment
    if   exp=='idis': xmin,xmax = 0.95,1.05
    elif exp=='jet':  xmin,xmax = 0.70,1.30
    elif exp=='dy':   xmin,xmax = 0.70,1.30
    else:             xmin,xmax = 0.80,1.20

    #--load predictions to get norms
    predictions = load('%s/data/predictions-%d.dat'%(wdir,istep))
    data = predictions['reactions'][exp]

    clusters,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc)

    _order = replicas[0]['order'][istep]

    #--get correct order from dist
    order  = []
    idx    = []
    for i in range(len(_order)):
        if _order[i][0] != 2:   continue
        if _order[i][1] != exp: continue
        order.append(_order[i][2])
        idx.append(i)

    if len(order)==0: return

    #--get correct params from dist
    params = np.zeros((len(order),len(replicas)))
    for i in range(len(order)):
        for j in range(len(replicas)):
            params[i][j] = replicas[j]['params'][istep][idx[i]]

    #--sort numerically
    z = sorted(zip(order,params))
    order  = [z[i][0] for i in range(len(z))]
    params = [z[i][1] for i in range(len(z))]

    #--create plot with enough space for # of parameters
    if len(order) > 4:
        nrows = math.ceil(len(order)/4.0)
        ncols = 4
        fig = py.figure(figsize=(ncols*7,nrows*4))
    else:
        nrows = 1
        ncols = len(order)
        fig = py.figure(figsize=(ncols*7,nrows*4))
    X = np.linspace(1,len(replicas),len(replicas))

    #--create plot
    for i in range(len(order)):
        if order[i] not in data: continue
        ax = py.subplot(nrows,ncols, i+1)
        name = get_exp_name(exp,order[i])
        ax.set_title(r'\textrm{%s} %s'%(order[i],name), size=30)
        for j in range(nc):
            color  = colors[clusters[j]]
            par = [params[i][k] for k in range(len(params[i])) if clusters[k]==j]
            mean = np.mean(par)
            std  = np.std(par)
            meanl = r'\textrm{mean}: $%6.4f$'%mean
            stdl  = r'\textrm{std}: $%6.4f$'%std
            ax.hist(par,color=color,alpha=0.6,edgecolor='black',bins=40,range=(xmin,xmax))
            ax.text(0.02,0.90,meanl,transform=ax.transAxes,size=20)
            ax.text(0.02,0.80,stdl, transform=ax.transAxes,size=20)
        #--plot norm uncertainty
        if '*norm_c'  in data[order[i]]: norm_c = (data[order[i]]['*norm_c']/data[order[i]]['value'])[0]
        if  'norm_c'  in data[order[i]]: norm_c = (data[order[i]]['norm_c'] /data[order[i]]['value'])[0]
        if  'norm_c ' in data[order[i]]: norm_c = (data[order[i]]['norm_c '] /data[order[i]]['value'])[0]
        if  '-norm_c' in data[order[i]]: norm_c = (data[order[i]]['-norm_c'] /data[order[i]]['value'])[0]
        ax.axvline(1.0,ymin=0,ymax=1,ls='--',color='darkblue',alpha=0.8)
        ax.set_xlim(xmin,xmax)
        hand_norm = ax.axvspan(1.0-norm_c,1.0+norm_c,alpha=0.2,color='darkblue')
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20)

    filename='%s/gallery/norms-%s-hist.png'%(wdir,exp)
    checkdir('%s/gallery'%wdir)
    py.tight_layout()
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def get_exp_name(exp,idx):

    if exp=='idis':
        if idx == 10002:   name = r'\textrm{JLab Hall C $d$}'
        elif idx == 10003: name = r'\textrm{JLab Hall C $p$}'
        elif idx == 10010: name = r'\textrm{SLAC $p$}'
        elif idx == 10011: name = r'\textrm{SLAC $d$}'
        elif idx == 10016: name = r'\textrm{BCDMS $p$}'
        elif idx == 10017: name = r'\textrm{BCDMS $d$}'
        elif idx == 10020: name = r'\textrm{NMC $p$}'
        elif idx == 10021: name = r'\textrm{NMC $d/p$}'
        elif idx == 10033: name = r'\textrm{BONUS $n/d$}'
        elif idx == 10041: name = r'\textrm{JLab Hall C $^3$He$/d$}'
        elif idx == 10050: name = r'\textrm{MARATHON $d/p$}'
        elif idx == 10051: name = r'\textrm{MARATHON $^3$He/$^3$H}'
        else: name = None
    elif exp=='pidis':
        if idx == 10001:   name = r'\textrm{COMPASS $d$ $A_1$}'
        elif idx == 10002: name = r'\textrm{COMPASS $p$ $A_1$}'
        elif idx == 10003: name = r'\textrm{COMPASS $p$ $A_1$}'
        elif idx == 10004: name = r'\textrm{EMC $p$ $A_1$}'
        elif idx == 10005: name = r'\textrm{HERMES $n$ $A_1$}'
        elif idx == 10006: name = r'\textrm{HERMES $d$ $A_{\parallel}$}'
        elif idx == 10007: name = r'\textrm{HERMES $p$ $A_{\parallel}$}'
        elif idx == 10018: name = r'\textrm{SLAC(E142) $h$ $A_1$}'
        elif idx == 10021: name = r'\textrm{SLAC(E143) $d$ $A_{\parallel}$}'
        elif idx == 10022: name = r'\textrm{SLAC(E143) $p$ $A_{\parallel}$}'
        elif idx == 10025: name = r'\textrm{SLAC(E154) $h$ $A_{\parallel}$}'
        elif idx == 10027: name = r'\textrm{SLAC(E155) $d$ $A_{\parallel}$}'
        elif idx == 10029: name = r'\textrm{SLAC(E155) $p$ $A_{\parallel}$}'
        elif idx == 10032: name = r'\textrm{SLACE80E130 $p$ $A_{\parallel}$}'
        else: name = None
    elif exp=='jet':
        if idx == 10002:   name = r'\textrm{CDF}'
        else: name = None
    elif exp=='dy':
        if   idx == 10001:   name = r'\textrm{E866 $pp$}'
        elif idx == 10002:   name = r'\textrm{E866 $pd$}'
        elif idx == 20001:   name = r'\textrm{E866 $pd/2pp$}'
        elif idx == 20002:   name = r'\textrm{E906 $pd/2pp$}'
        else: name = None
    else:
        name = None

    return name




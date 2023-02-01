#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text',usetex=True)
from matplotlib import cm
import pylab as py

#--for biclustering
from sklearn.datasets import make_biclusters
#from sklearn.datasets import samples_generator as sg
#from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

#--for cluster finding
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn import metrics
#from sklearn.datasets.samples_generator import make_blobs


#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator

#--from local  
from analysis.corelib import core

def _gen_labels_by_chi2(wdir,istep):

    print('\t by chi2 ...')

    #--load data where residuals are computed
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))
    res=np.array(data['res'])
    #--compute chi2/npts
    chi2dof=[]
    for row in res:
        chi2dof.append(np.sum(row**2)/len(row))
    chi2dof=np.array(chi2dof)
    return chi2dof

def _gen_labels_by_echi2(wdir,istep):

    print('\t by echi2 ...')

    #--load data where residuals are computed
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))
    res=np.array(data['res'])
    #--compute chi2/npts
    echi2=[]
    for row in res:
        echi2.append(0)#np.sum(row**2)/len(row))
    echi2=np.array(echi2)

    nexp=0
    for reaction in data['reactions']:
        for idx in data['reactions'][reaction]:
            nexp+=1
            res=np.array(data['reactions'][reaction][idx]['residuals-rep'])
            for i in range(len(res)):
                echi2[i]+=(np.sum(res[i]**2)/len(res[i]))
    return echi2/nexp

def hook(params,order):
    sample=[]
    for i in range(len(order)):
        if order[i][0]!=1: continue
        #if order[i][1]!='pdf': continue
        #if 'g1'  in order[i][2]: continue 
        #if 'uv1' in order[i][2]: continue 
        #if 'dv1' in order[i][2]: continue 
        #if 'db1' in order[i][2]: continue 
        #if 'ub1' in order[i][2]: continue 
        #if 's1' in order[i][2]: continue 
        #if 'sb1' in order[i][2]: continue 
        sample.append(params[i])
    return sample

def _gen_labels_by_cluster(wdir,istep,nc,hook=None):

    print('\t by kmeans ...')

    replicas=core.get_replicas(wdir)
    samples=[]
    for replica in replicas:
        params = replica['params'][istep]
        order  = replica['order'][istep]

        if hook==None:
            samples.append(params)
        else:
            samples.append(hook(params,order))

    #--affinity propagation
    #af = AffinityPropagation(preference=-50).fit(samples)
    #cluster_centers_indices = af.cluster_centers_indices_
    #return af.labels_

    #--agglomerative
    #clustering = AgglomerativeClustering(nc).fit(samples)
    #return clustering.labels_

    #brc = Birch(branching_factor=3, n_clusters=None, threshold=0.5,compute_labels=True)
    #brc.fit(samples) 
    #return brc.labels_

    #--kmean   
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(samples)
    return kmeans.labels_
   
def gen_labels(wdir,kc):

    print('\ngenerating labels for %s\n'%wdir)

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()

    if istep not in kc.nc:
        print("istep not in kmeanconf.py")
        sys.exit()

    nc=kc.nc[istep]
    hook=kc.hooks[istep]

    istep=core.get_istep()
    labels={}
    labels['chi2dof']  = _gen_labels_by_chi2(wdir,istep)
    labels['echi2dof'] = _gen_labels_by_echi2(wdir,istep)
    labels['cluster']  = _gen_labels_by_cluster(wdir,istep,nc,hook)
    checkdir('%s/data'%wdir)
    save(labels,'%s/data/labels-%d.dat'%(wdir,istep))

def get_clusters(wdir,istep,kc): 

    nc=kc.nc[istep]
    hook=kc.hooks[istep]

    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster  = labels['cluster']
    chi2dof  = labels['chi2dof']
    echi2dof = labels['echi2dof']

    #--get clusters idx ordering
    echi2_means = [np.mean([echi2dof[i] for i in range(len(echi2dof))
           if cluster[i]==j ])  for j in range(nc)]

    order=np.argsort(echi2_means)

    clist=['r','c','g','y']
    colors={order[i]:clist[i] for i in range(nc)}
    return cluster,colors,nc,order

#--plots

def plot_chi2_dist(wdir,istep,kc):

    #--needs revisions NS (09/02/19)

    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    chi2     = labels['chi2dof']
    cluster,colors,nc,order = get_clusters(wdir,istep,kc) 

    nrows=1
    ncols=1
    #--plot labeled residuals
    ax=py.subplot(nrows,ncols,1)

    chi2=[_ for _ in chi2 if _<2]
    Mean=np.average(chi2)
    dchi=np.std(chi2)
    chi2min=Mean-2*dchi
    chi2max=Mean+2*dchi
    R=(chi2min,chi2max)
    ax.hist(chi2,bins=60,range=R,histtype='step',label='size=%d'%len(chi2))
    for j in range(nc):
        chi2_=[chi2[i] for i in range(len(chi2)) if cluster[i]==j]
        c=colors[j]
        print (c,np.mean(chi2_),np.mean(chi2))
        label='cluster id=%d  size=%d'%(j,len(chi2_))
        ax.hist(chi2_,bins=30,range=R,histtype='step',label=label,color=c)
        mean=np.average(chi2_)
        ax.axvline(mean,color=c,ls=':')
        

    ax.set_xlabel('chi2/npts')
    ax.set_ylabel('yiled')
    ax.legend()
    ax.text(0.1,0.8,'step %d'%istep,transform=ax.transAxes)
    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/chi2-dist-%d.jpg'%(wdir,istep))
    py.close()

def plot_chi2_dist_per_exp(wdir,kc,reaction,idx):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster,colors,nc,order = get_clusters(wdir,istep,kc) 
    chi2     = labels['chi2dof']
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))

    if reaction not in data['reactions']: 
        print('%s not in step'%reaction)
        return
    if idx not in data['reactions'][reaction]: 
        print('%s %s not in step'%(reaction,idx))
        return

    res=np.array(data['reactions'][reaction][idx]['residuals-rep'])
    chi2=np.array([np.sum(r**2)/len(r) for r in res])
    #chi2=np.array([_ for _ in chi2 if _<5])

    Mean=np.average(chi2)
    
    dchi=np.std(chi2)
    chi2min=Mean-2*dchi
    chi2max=Mean+2*dchi
    R=(chi2min,chi2max)
    nrows=1
    ncols=1
    #--plot labeled residuals
    fig = py.figure(figsize=(nrows*7,ncols*4))
    ax=py.subplot(nrows,ncols,1)

    ax.hist(chi2,bins=30,range=R,histtype='step',label='size=%d'%len(chi2))
    for j in range(nc):
        _chi2=[chi2[i] for i in range(len(chi2)) if cluster[i]==j]
        c=colors[j]
        print (c, np.mean(_chi2),reaction,idx)
        label='cluster id=%d  size=%d'%(j,len(_chi2))
        ax.hist(_chi2,bins=30,range=R,histtype='step',label=label,color=c)
    ax.set_xlabel(r'\textrm{\textbf{chi2/npts}}',size=25)
    ax.set_ylabel(r'\textrm{\textbf{yield}}',size=25)
    ax.legend()
    ax.text(0.1,0.8,'step %d'%istep,transform=ax.transAxes)
    ax.tick_params(axis='both',which='both',direction='in',right=True,top=True,labelsize=15)
    py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/chi2-dist-%d-%s-%d.png'%(wdir,istep,reaction,idx))
    py.close()

def print_chi2_per_exp(wdir,kc):

    load_config('%s/input.py'%wdir)
    istep = core.get_istep()
    labels   = load('%s/data/labels-%d.dat'%(wdir,istep))
    cluster,colors,nc,order = get_clusters(wdir,istep,kc) 
    chi2     = labels['chi2dof']
    data=load('%s/data/predictions-%d.dat'%(wdir,istep))

    npts     = 0
    res2tot  = {colors[i]:0 for i in range(len(colors))}
    chi2tot  = {colors[i]:0 for i in range(len(colors))}
    echi2tot = {colors[i]:0 for i in range(len(colors))}
    for reaction in data['reactions']:
        print
        print ('reaction: %s'%reaction)
        for idx in data['reactions'][reaction]:
            print('idx: %s'%idx)
                       
            res=np.array(data['reactions'][reaction][idx]['residuals-rep'])
            chi2=np.array([np.sum(r**2)/len(r) for r in res])

            res2=np.array([np.sum(r**2) for r in res])

            npts += len(res[1])

            for j in range(nc):
                _chi2=[chi2[i] for i in range(len(chi2)) if cluster[i]==j]
                _res2=[res2[i] for i in range(len(res2)) if cluster[i]==j]
                c=colors[j]
                print('color: %s, chi2: %3.2f'%(c,np.mean(_chi2)))
                echi2tot[c] += np.mean(_chi2)
                res2tot[c]  += np.mean(_res2)

    for j in range(nc):
        chi2tot[colors[j]] = res2tot[colors[j]]/npts

    print('Total chi2:')
    print(chi2tot)
    print('Total echi2:')
    print(echi2tot)

#--get scale for color coding replicas
def get_scale(wdir,reaction='all',idx=[]):

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas = core.get_replicas(wdir)
    nrep = len(replicas)
    ones = np.ones(nrep)*0.4

    predictions = load('%s/data/predictions-%s.dat'%(wdir,istep))

    rchi2 = []
    if reaction=='all':
        for i in range(nrep):
            npts,chi2 = 0,0
            for reaction in predictions['reactions']:
                data = predictions['reactions'][reaction]
                for _ in data:
                    thy = data[_]['prediction-rep'][i]
                    res = (data[_]['value']-thy)/data[_]['alpha']
                    npts += res.size
                    chi2 += np.sum(res**2)
            rchi2.append(chi2/npts)

    else:

        if reaction not in predictions['reactions']: return ones
        data = predictions['reactions'][reaction]

        for _ in idx:
            if _ not in list(data): return ones

        for i in range(nrep):
            npts,chi2 = 0,0
            for _ in idx:
                if _ not in data: continue
                thy = data[_]['prediction-rep'][i]
                res = (data[_]['value']-thy)/data[_]['alpha']
                npts += res.size
                chi2 += np.sum(res**2)
            rchi2.append(chi2/npts)

    amin = np.amin(rchi2)
    amax = np.amax(rchi2)
    scale = np.interp(rchi2, (amin,amax), (0,1))

    return scale












#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.randomstr import id_generator

#--from fitlib
from fitlib.resman import RESMAN

#--from local
import analysis.corelib.core

def gen_dataframe(order,obsres,nsize):
    data={} 
    data['size']=0
    data['order']=order
    data['params']=np.zeros((nsize,len(order)))
    data['reactions']={}
    for reaction in obsres:
        tabs={}
        for idx in obsres[reaction].tabs:
            npts=obsres[reaction].tabs[idx]['value'].size
            tabs[idx]=np.zeros((nsize,npts))
        data['reactions'][reaction]=tabs
    return data

def gen_samples(wdir,nsamples=100,nsize=10):

    load_config('%s/input.py'%wdir)
    conf['bootstrap']=False
    istep=core.get_istep()
    core.mod_conf(istep) #--set conf as specified in istep   

    conf['predict'] = True
    resman=RESMAN(nworkers=5,parallel=True,datasets=True)
    parman=resman.parman
    order=parman.order

    obsres={}
    if 'idis'   in conf['datasets'] : obsres['idis']   = resman.idisres
    if 'pidis'  in conf['datasets'] : obsres['pidis']  = resman.pidisres
    if 'sidis'  in conf['datasets'] : obsres['sidis']  = resman.sidisres
    if 'psidis' in conf['datasets'] : obsres['psidis'] = resman.psidisres
    if 'dy'     in conf['datasets'] : obsres['dy']     = resman.dyres
    if 'sia'    in conf['datasets'] : obsres['sia']    = resman.siares
   

    print('\ngen ml samples using the setup as in %s\n'%wdir)

    checkdir('%s/mlsamples'%wdir)

    data=gen_dataframe(parman.order,obsres,nsize)

    for _ in range(nsamples):
        lprint('progress: %d/%d'%(_,nsamples))
        par=parman.gen_flat(setup=True)
        res,rres,nres=resman.get_residuals(par)

        #--fill dataframe
        i=data['size']
        data['params'][i]=par
        for reaction in obsres:
            for idx in data['reactions'][reaction]:
                prediction=copy.copy(obsres[reaction].tabs[idx]['prediction'])
                data['reactions'][reaction][idx][i]=prediction
        data['size']+=1

        if data['size']==nsize:
            save(data,'%s/mlsamples/%s'%(wdir,id_generator(12)))
            data=gen_dataframe(parman.order,obsres,nsize)

    print     

    #--close resman
    resman.shutdown()

def collect_exp_data(wdir):

    load_config('%s/input.py'%wdir)
    conf['bootstrap']=False
    istep=core.get_istep()
    core.mod_conf(istep) #--set conf as specified in istep   

    conf['predict'] = True
    resman=RESMAN(nworkers=1,parallel=False,datasets=True)

    obsres={}
    if 'idis'   in conf['datasets'] : obsres['idis']   = resman.idisres
    if 'pidis'  in conf['datasets'] : obsres['pidis']  = resman.pidisres
    if 'sidis'  in conf['datasets'] : obsres['sidis']  = resman.sidisres
    if 'psidis' in conf['datasets'] : obsres['psidis'] = resman.psidisres
    if 'dy'     in conf['datasets'] : obsres['dy']     = resman.dyres
    if 'sia'    in conf['datasets'] : obsres['sia']    = resman.siares
   

    print('\ncollecting exp data using the setup as in %s\n'%wdir)

    data={} 
    for reaction in obsres:
        tabs={}
        for idx in obsres[reaction].tabs:
            tabs[idx]={}
            tabs[idx]['value']=obsres[reaction].tabs[idx]['value']
            tabs[idx]['alpha']=obsres[reaction].tabs[idx]['alpha']
        data[reaction]=tabs

    save(data,'%s/expdata.dat'%(wdir))

    #--close resman
    resman.shutdown()










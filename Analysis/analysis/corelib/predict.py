#!/usr/bin/env python
import os,sys
import subprocess
import numpy as np
import scipy as sp
import pandas as pd
import copy

#--from tools
from tools           import config
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.corelib import core

def get_predictions(wdir,force=False,ncores=1,mod_conf=None,name=''):

    #--if force=False: previous predictions will not be repeated
    #--if force=True : previous predictions will be overwritten
    
    if mod_conf == None:
        load_config('%s/input.py'%wdir)
        replicas=core.get_replicas(wdir)
    else:
        config.conf = copy.deepcopy(mod_conf)
        replicas=core.get_replicas(wdir,mod_conf=conf)

    names= core.get_replicas_names(wdir)

    conf['bootstrap']=False
    istep=core.get_istep()
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep

    #--get replicas that are already done if force=False
    if force==False:
        try:
            if mod_conf == None: done = load('%s/data/predictions-%d.dat'%(wdir,istep))
            else:                done = load('%s/data/predictions-%d-%s.dat'%(wdir,istep,name))
            done['res'].tolist()
            done['rres'].tolist()
            done['nres'].tolist()
            flag=True
        except: flag=False
    else: flag=False


    #--choose parallelization based on what experiments are present
    nopar = ['idis','pidis','sidis','psidis','sia','SU23']
    _ncores = 1
    parallel = False
    for exp in conf['steps'][istep]['datasets']:
        if exp not in nopar:
            parallel = True
            _ncores = ncores

    #--run RESMAN
    conf['predict'] = True
    resman=RESMAN(nworkers=_ncores,parallel=parallel,datasets=True)
    parman=resman.parman
    order=parman.order
    datasets = copy.copy(conf['datasets'])
       
    obsres={}
    if 'idis'     in conf['datasets'] : obsres['idis']     = resman.idis_res
    if 'pidis'    in conf['datasets'] : obsres['pidis']    = resman.pidis_res
    if 'SU23'     in conf['datasets'] : obsres['SU23']     = resman.SU23_res
    if 'sidis'    in conf['datasets'] : obsres['sidis']    = resman.sidis_res
    if 'psidis'   in conf['datasets'] : obsres['psidis']   = resman.psidis_res
    if 'dy'       in conf['datasets'] : obsres['dy']       = resman.dy_res
    if 'wzrv'     in conf['datasets'] : obsres['wzrv']     = resman.wzrv_res
    if 'wasym'    in conf['datasets'] : obsres['wasym']    = resman.wasym_res
    if 'zrap'     in conf['datasets'] : obsres['zrap']     = resman.zrap_res
    if 'sia'      in conf['datasets'] : obsres['sia']      = resman.sia_res
    if 'qpdf'     in conf['datasets'] : obsres['qpdf']     = resman.qpdf_res
    if 'dy-pion'  in conf['datasets'] : obsres['dy-pion']  = resman.dy_pion_res
    if 'pion_qT'  in conf['datasets'] : obsres['pion_qT']  = resman.pion_qTres
    if 'ln'       in conf['datasets'] : obsres['ln']       = resman.ln_res
    if 'jet'      in conf['datasets'] : obsres['jet']      = resman.jet_res
    if 'pjet'     in conf['datasets'] : obsres['pjet']     = resman.pjet_res
    if 'wc'       in conf['datasets'] : obsres['wc']       = resman.wc_res 
    #--setup big table to store all we want
    data={}
    data['name'] = [] 
    data['order']=order
    data['params']=[]
    data['reactions']={}
    data['res']=[]
    data['rres']=[]
    data['nres']=[]

    for _ in obsres:
        tabs=copy.copy(obsres[_].tabs)
        #--create a space to store all the predictions from replicas
        for idx in tabs:
            tabs[idx]['prediction-rep']=[]
            tabs[idx]['residuals-rep']=[]
            tabs[idx]['shift-rep']=[]
            tabs[idx]['rres-rep']=[]
        data['reactions'][_]=tabs

    print('\ngen predictions using %s\n'%wdir)

    checkdir('%s/data'%wdir)
    #--total replica count
    cnt =0
    for replica in replicas:
        lprint('progress: %d/%d'%(cnt+1,len(replicas)))

        #--convert arrays into lists
        for _ in ['res','rres','nres']:
            data[_]=list(data[_])

        #--skip replicas that have already been generated
        if flag:
            if names[cnt] in done['name']:
                i = done['name'].index(names[cnt])
                data['name']  .append(done['name'][i])
                data['res']   .append(done['res'][i])
                data['rres']  .append(done['rres'][i])
                data['nres']  .append(done['nres'][i])
                data['params']=np.append(data['params'],done['params'][i])
                for _ in obsres:
                    for idx in data['reactions'][_]:
                        data['reactions'][_][idx]['prediction-rep'].append(done['reactions'][_][idx]['prediction-rep'][i])
                        data['reactions'][_][idx]['residuals-rep'].append(done['reactions'][_][idx]['residuals-rep'][i])
                        data['reactions'][_][idx]['shift-rep'].append(done['reactions'][_][idx]['shift-rep'][i])
                        if 'rres-rep' in done['reactions'][_][idx]:
                            if done['reactions'][_][idx]['rres-rep'] == []: pass
                            else: data['reactions'][_][idx]['rres-rep'].append(done['reactions'][_][idx]['rres-rep'][i])
                cnt +=1
                continue

        #--update passive parameters
        core.mod_conf(istep,replica)
        conf['datasets'] = copy.copy(datasets)

        data['name'].append(names[cnt])
        cnt+=1
        parman.par=copy.copy(replica['params'][istep])
        parman.order=copy.copy(replica['order'][istep])
        data['params']=np.append(data['params'],parman.par)

        #--compute residuals (==theory)
        res,rres,nres=resman.get_residuals(parman.par)
        data['res'].append(res)
        data['rres'].append(rres)
        data['nres'].append(nres)

        #--save predictions of the current step and current replica at data
        for _ in obsres:
            for idx in data['reactions'][_]:
                prediction= copy.copy(obsres[_].tabs[idx]['prediction'])
                residuals = copy.copy(obsres[_].tabs[idx]['residuals'])
                shift     = copy.copy(obsres[_].tabs[idx]['shift'])
                rres      = obsres[_]._get_rres(idx)
                data['reactions'][_][idx]['prediction-rep'].append(prediction)
                data['reactions'][_][idx]['residuals-rep'].append(residuals)
                data['reactions'][_][idx]['shift-rep'].append(shift)
                data['reactions'][_][idx]['rres-rep'].append(rres)

        #--save results after each replica
        #--convert tables to numpy array before saving
        for _ in ['res','rres','nres']:
            data[_]=np.array(data[_])

        if mod_conf==None:
            save(data,'%s/data/predictions-%d.dat'%(wdir,istep))
        else:
            save(data,'%s/data/predictions-%d-%s.dat'%(wdir,istep,name))
    print 

    #--close resman
    resman.shutdown()








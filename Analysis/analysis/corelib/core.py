#!/usr/bin/env python
import os,sys
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


def mod_conf(istep, replica=None):

    step=conf['steps'][istep]

    #--remove pdf/ff that is not in the step
    distributions=list(conf['params'])  #--pdf,ppdf,ffpion,ffkaon,...
    for dist in distributions:
        if  dist in step['active distributions']:  continue
        elif 'passive distributions' in step and dist in step['passive distributions']:  continue
        else:
            del conf['params'][dist] 

    if np.any(replica)!=None:
        #--set fixed==True for passive distributions
        if 'passive distributions' in step:
            for dist in step['passive distributions']:
                for par in conf['params'][dist]:
                    if conf['params'][dist][par]['fixed']==False:
                        conf['params'][dist][par]['fixed']=True

                    #--set prior parameters values for passive distributions
                    for _istep in step['dep']:
                        prior_order=replica['order'][_istep]
                        prior_params=replica['params'][_istep]
                        for i in range(len(prior_order)):
                            _,_dist,_par = prior_order[i]
                            if  dist==_dist and par==_par:
                                #--update values in conf
                                conf['params'][dist][par]['value']=prior_params[i]
                                #--update values at class
                                if dist not in conf: continue
                                PAR  = conf[dist].PAR
                                idx = conf[dist].PAR.index(par.split()[1])
                                flav = par.split()[0]
                                conf[dist].params[flav][idx] = prior_params[i]
                                conf[dist].setup()

    #--remove datasets not in the step
    datasets=list(conf['datasets']) #--idis,dy,....
    for dataset in datasets:
        if  dataset in step['datasets']:  

            #--remove entry from xlsx
            xlsx=list(conf['datasets'][dataset]['xlsx'])
            for idx in xlsx:
                if  idx in step['datasets'][dataset]:
                    continue
                else:
                    del conf['datasets'][dataset]['xlsx'][idx]

            #--remove entry from norm
            norm=list(conf['datasets'][dataset]['norm'])
            for idx in norm:
                if  idx in step['datasets'][dataset]:
                    continue
                else:
                    del conf['datasets'][dataset]['norm'][idx]
        else:
            del conf['datasets'][dataset]  

    if 'passive distributions' in conf['steps'][istep] and replica != None:
        if len(conf['steps'][istep]['passive distributions']) > 0: 
            get_passive_data(istep,replica)
 

def get_replicas(wdir,mod_conf=None):
    """
    load the msr files
    """
    replicas=sorted(os.listdir('%s/msr-inspected'%wdir))
    replicas=[load('%s/msr-inspected/%s'%(wdir,_)) for _ in replicas]

    #--update order with passive parameters from prior steps
    if mod_conf == None:
        load_config('%s/input.py'%wdir)
    else:
        config.conf = copy.deepcopy(mod_conf)

    istep = get_istep()
    #--get parameters from passive distributions
    if 'passive distributions' in conf['steps'][istep]:
       dep = conf['steps'][istep]['dep']
       for step in dep:
           for j in range(len(replicas)):
               prior_order = replicas[j]['order'][step]
               for i in range(len(prior_order)):
                   if prior_order[i] not in replicas[j]['order'][istep]:
                       dist = prior_order[1]
                       if dist not in conf['steps'][istep]['passive distributions']: continue
                       replicas[j]['order'][istep].append(prior_order[i])
                       replicas[j]['params'][istep]=np.append(replicas[j]['params'][istep],replicas[j]['params'][step][i])

    #--get parameters from distributions where all parameters are fixed
    dists = conf['steps'][istep]['active distributions'] + (conf['steps'][istep]['passive distributions'])
    for dist in dists:
        flag = True
        for par in conf['params'][dist]:
            if conf['params'][dist][par]['fixed']: continue
            flag = False
        if flag:
           fixed_order  = [[1,dist,_] for _ in conf['params'][dist].keys()]
           fixed_params = [conf['params'][dist][_] for _ in conf['params'][dist].keys()]
           for j in range(len(replicas)):
               for i in range(len(fixed_order)):
                   if fixed_order[i] not in replicas[j]['order'][istep]:
                       replicas[j]['order'][istep].append(fixed_order[i])
                       replicas[j]['params'][istep]=np.append(replicas[j]['params'][istep],fixed_params[i]['value'])

    return replicas

def get_replicas_names(wdir):
    replicas=sorted(os.listdir('%s/msr-inspected'%wdir))
    return replicas

def get_istep():
    #--pick last step
    return sorted(conf['steps'])[-1] 

def get_passive_data(istep,replica):
    order = replica['order'][istep]
    for i in range(len(order)):
        if order[i][0] != 2: continue
        exp = order[i][1]
        if exp in conf['datasets']: continue
        conf['datasets'][exp] = {_:{} for _ in ['norm','xlsx']}
    



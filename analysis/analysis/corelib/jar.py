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
from tools.inputmod  import INPUTMOD
from tools.randomstr import id_generator

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

def gen_jar_file(wdir,kc):

    print('\ngen jar file using (best cluster) %s\n'%wdir)

    replicas=core.get_replicas(wdir)
    conf={}
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
    best_cluster=cluster_order[0]

    conf['order']=replicas[0]['order'][istep]
    conf['replicas']=[]

    cnt=0
    for i in range(len(replicas)):
        replica=replicas[i]
        #if cluster[i]!=best_cluster: continue
        cnt+=1
        conf['replicas'].append(replica['params'][istep])
    print('number of  replicas = %d'%cnt)
  
    checkdir('%s/data'%wdir)
    save(conf,'%s/data/jar-%d.dat'%(wdir,istep))



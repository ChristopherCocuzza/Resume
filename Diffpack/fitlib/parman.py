import sys
from tools.config import load_config, conf
from numpy.random import uniform
import numpy as np
import pandas as pd

class PARMAN:

    def __init__(self):
        self.get_ordered_free_params()

    def get_ordered_free_params(self):
        self.par=[]
        self.order=[]
        self.pmin=[]
        self.pmax=[]

        if 'check lims' not in conf: conf['check lims']=False

        for k in conf['params']:
            for kk in conf['params'][k]:
                if  conf['params'][k][kk]['fixed']==False:
                    p=conf['params'][k][kk]['value']
                    pmin=conf['params'][k][kk]['min']
                    pmax=conf['params'][k][kk]['max']
                    self.pmin.append(pmin)
                    self.pmax.append(pmax)
                    if p<pmin or p>pmax:
                       if conf['check lims']: raise ValueError('par limits are not consistent with central: %s %s'%(k,kk))

                    self.par.append(p)
                    self.order.append([1,k,kk])

        if 'datasets' in conf:
            for k in conf['datasets']:
                for kk in conf['datasets'][k]['norm']:
                    if  conf['datasets'][k]['norm'][kk]['fixed']==False:
                        p=conf['datasets'][k]['norm'][kk]['value']
                        pmin=conf['datasets'][k]['norm'][kk]['min']
                        pmax=conf['datasets'][k]['norm'][kk]['max']
                        self.pmin.append(pmin)
                        self.pmax.append(pmax)
                        if p<pmin or p>pmax:
                           if conf['check lims']: raise ValueError('par limits are not consistent with central: %s %s'%(k,kk))
                        self.par.append(p)
                        self.order.append([2,k,kk])

        self.pmin=np.array(self.pmin)
        self.pmax=np.array(self.pmax)
        self.par=np.array(self.par)
        self.set_new_params(self.par,initial=True)

    def gen_flat(self,setup=True):
        r=uniform(0,1,len(self.par))
        par=self.pmin + r * (self.pmax-self.pmin)
        if setup: self.set_new_params(par,initial=True)
        return par

    def check_lims(self):
        flag=True
        for k in conf['params']:
            for kk in conf['params'][k]:
                if  conf['params'][k][kk]['fixed']==False:
                    p=conf['params'][k][kk]['value']
                    pmin=conf['params'][k][kk]['min']
                    pmax=conf['params'][k][kk]['max']
                    if  p<pmin or p>pmax:
                        print(k,kk, p,pmin,pmax)
                        flag=False

        if  'datasets' in conf:
            for k in conf['datasets']:
                for kk in conf['datasets'][k]['norm']:
                    if  conf['datasets'][k]['norm'][kk]['fixed']==False:
                        p=conf['datasets'][k]['norm'][kk]['value']
                        pmin=conf['datasets'][k]['norm'][kk]['min']
                        pmax=conf['datasets'][k]['norm'][kk]['max']
                        if p<pmin or p>pmax:
                          flag=False
                          print(k,kk, p,pmin,pmax)

        return flag

    def set_new_params(self,parnew,initial=False):
        self.par=parnew
        self.shifts=0
        semaphore={}

        for i in range(len(self.order)):
            ii,k,kk=self.order[i]
            if  ii==1:
                if k not in semaphore: semaphore[k]=0
                if k not in conf['params']:     continue #--skip parameters that have been removed in latest step
                if kk not in conf['params'][k]: continue #--skip parameters that have been removed in latest step
                if  conf['params'][k][kk]['value']!=parnew[i]:
                    conf['params'][k][kk]['value']=parnew[i]
                    semaphore[k]=1
                    self.shifts+=1
            elif ii==2:
                if k not in conf['datasets']: continue #--skip datasets that have been removed in latest step
                if kk in conf['datasets'][k]['norm']:
                    if  conf['datasets'][k]['norm'][kk]['value']!=parnew[i]:
                        conf['datasets'][k]['norm'][kk]['value']=parnew[i]
                        self.shifts+=1


        #if  initial:
        #    for k in conf['params']: semaphore[k]=1

        for k in conf['params']: semaphore[k]=1
        self.propagate_params(semaphore)

    def gen_report(self):
        if 'jupytermode' not in conf:
            return self._gen_report_v1()
        else:
            return self._gen_report_v2()

    def _gen_report_v1(self):
        L=[]
        cnt=0
        for k in conf['params']:
            for kk in sorted(conf['params'][k]):
                if  conf['params'][k][kk]['fixed']==False:
                    cnt+=1
                    if  conf['params'][k][kk]['value']<0:
                        L.append('%d %10s  %10s  %10.5e'%(cnt,k,kk,conf['params'][k][kk]['value']))
                    else:
                        L.append('%d %10s  %10s   %10.5e'%(cnt,k,kk,conf['params'][k][kk]['value']))

        for k in conf['datasets']:
            for kk in conf['datasets'][k]['norm']:
                if  conf['datasets'][k]['norm'][kk]['fixed']==False:
                    cnt+=1
                    L.append('%d %10s %10s %10s  %10.5e'%(cnt,'norm',k,kk,conf['datasets'][k]['norm'][kk]['value']))
        return L

    def _gen_report_v2(self):
        data={_:[] for _ in ['idx','dist','type','value']}
        cnt=0
        for k in conf['params']:
            for kk in sorted(conf['params'][k]):
                if  conf['params'][k][kk]['fixed']==False:
                    cnt+=1
                    data['idx'].append('%d'%cnt)
                    data['dist'].append('%10s'%k)
                    data['type'].append('%10s'%kk)
                    data['value'].append('%10.2e'%conf['params'][k][kk]['value'])
                    

        for k in conf['datasets']:
            for kk in conf['datasets'][k]['norm']:
                if  conf['datasets'][k]['norm'][kk]['fixed']==False:
                    cnt+=1
                    data['idx'].append('%d'%cnt)
                    data['dist'].append('norm %10s'%k)
                    data['type'].append('%10d'%kk)
                    data['value'].append('%10.2e'%conf['datasets'][k]['norm'][kk]['value'])

        data=pd.DataFrame(data)
        msg=data.to_html(col_space=80,index=False,justify='left')

        return msg



    def propagate_params(self,semaphore):
        flag=False

        if 'QCD'       in semaphore and semaphore['QCD']      ==1: self.set_QCD_params()
        if 'eweak'     in semaphore and semaphore['eweak']    ==1: self.set_eweak_params()

        dists = []
        #--leading power collinear distributions
        dists.extend(['tpdf','diffpippim','tdiffpippim','diffpippim_pythia'])

        for dist in dists:
            if dist in semaphore and semaphore[dist]==1: self.set_dist_params(dist)

    def set_QCD_params(self):

        if conf['params']['QCD']['mc']['fixed']==False:

            conf['aux'].mc=conf['params']['QCD']['mc']['value']

        if conf['params']['QCD']['mb']['fixed']==False:

            conf['aux'].mb=conf['params']['QCD']['mb']['value']

        if conf['params']['QCD']['alphaS0']['fixed']==False:

            conf['aux'].alphaS0= conf['params']['QCD']['alphaS0']['value']

    def set_eweak_params(self):

        if conf['params']['eweak']['s2wMZ']['fixed']==False:

            conf['aux'].s2wMZ = conf['params']['eweak']['s2wMZ']['value']

    def set_params(self,dist,FLAV,PAR,dist2=None):

        #--setup the constraints
        for flav in FLAV:
            for par in PAR:
                if flav+' '+par not in conf['params'][dist]: continue
                if conf['params'][dist][flav+' '+par]['fixed']==True: continue
                if conf['params'][dist][flav+' '+par]['fixed']==False:
                    p    = conf['params'][dist][flav+' '+par]['value']
                    pmin = conf['params'][dist][flav+' '+par]['min']
                    pmax = conf['params'][dist][flav+' '+par]['max']
                    if p < pmin: 
                        conf['params'][dist][flav+' '+par]['value'] = pmin
                        print('WARNING: %s %s below limit, setting to %f'%(flav,par,pmin))
                    if p > pmax: 
                        conf['params'][dist][flav+' '+par]['value'] = pmax
                        print('WARNING: %s %s above limit, setting to %f'%(flav,par,pmax))
                    continue
                reference_flav=conf['params'][dist][flav+' '+par]['fixed']

                if len(reference_flav.split())==2:
                    conf['params'][dist][flav+' '+par]['value']=conf['params'][dist][reference_flav]['value']
                    #print('Setting %s %s %s to %s %s'%(dist, flav, par, dist, reference_flav))

                elif len(reference_flav.split())==3:  #allows one to reference from another distribution
                    reference_dist=reference_flav.split()[0]
                    reference_flav=reference_flav.split()[1] + ' ' + reference_flav.split()[2] 
                    conf['params'][dist][flav+' '+par]['value']=conf['params'][reference_dist][reference_flav]['value']
                    #print('Setting %s %s %s to %s %s'%(dist, flav, par, reference_dist, reference_flav))

        for k in conf['datasets']:
            for kk in conf['datasets'][k]['norm']:
                if conf['datasets'][k]['norm'][kk]['fixed']==True:  continue
                if conf['datasets'][k]['norm'][kk]['fixed']==False: continue
                reference_norm = conf['datasets'][k]['norm'][kk]['fixed']
                conf['datasets'][k]['norm'][kk]['value'] = conf['datasets'][k]['norm'][reference_norm]['value']

        #--update values at the class
        for flav in FLAV:
            idx=0
            for par in PAR:
                if  flav+' '+par in conf['params'][dist]:
                    conf[dist].params[flav][idx]=conf['params'][dist][flav+' '+par]['value']
                    if dist2!=None: 
                        conf[dist2].params[flav][idx]=conf['params'][dist][flav+' '+par]['value']
                else:
                    conf[dist].params[flav][idx]=0
                    if dist2!=None: 
                        conf[dist2].params[flav][idx]=0
                idx+=1
        conf[dist].setup()
        if dist2!=None: 
            conf[dist2].setup()

        #--update values at conf
        for flav in FLAV:
            idx=0
            for par in PAR:
                if  flav+' '+par in conf['params'][dist]:
                    conf['params'][dist][flav+' '+par]['value']= conf[dist].params[flav][idx]
                idx+=1 


    #--generic function for most distributions

    def set_dist_params(self,dist):
        if   dist=='tpdf':  dist2 = 'tpdf-mom'
        else:               dist2 = None
        FLAV=conf[dist].FLAV
        PAR=conf[dist].PAR
        self.set_params(dist,FLAV,PAR,dist2)








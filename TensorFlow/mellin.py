#!/usr/bin/env python
import sys,os
import numpy as np
import time
import argparse
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
from matplotlib.ticker import MultipleLocator
import pylab as py

import tensorflow as tf
from tensorflow import keras

#--from scipy
from scipy.interpolate import griddata

#--from tools
from tools.tools  import load, save, checkdir, lprint, convert, deconvert
from tools.config import conf, load_config

#--from qcdlib
from qcdlib import aux,mellin,alphaS,eweak
from qcdlib import tpdf, diff, tdiff

from reader import READER

conf['Q20']   = 1.0**2
conf['alphaSmode']='backward'
conf['dglap mode']='truncated'
conf['order']='LO'
conf['path2dihadrontab']='%s/grids/dihadron'%os.environ['FITPACK']
conf['aux']=aux.AUX()
conf['mellin']=mellin.MELLIN(npts=4)
conf['alphaS']=alphaS
conf['eweak']=eweak.EWEAK()



class RESMAN:

    def __init__(self,nworkers=2,parallel=True,datasets=True,load_lhapdf=False):

        self.load_lhapdf=load_lhapdf
        self.setup_core()
        self.parman=PARMAN()

        #--if diffpippim in active distributions, update unpolarized observables
        #--otherwise, unpolarized observables are only calculated once at beginning of fit
        conf['update_unpol'] = True
        if ('steps' in conf) and (len(conf['steps']) > 0):
            istep = sorted(conf['steps'].keys())[-1]
            if   'diffpippim' in conf['steps'][istep]['active distributions']: conf['update_unpol'] = True
            else:                                                              conf['update_unpol'] = False

        #--if doing predictions, need to update unpolarized observables
        if 'predict' in conf and conf['predict']: conf['update_unpol'] = True

        if datasets:
            if 'dihadron' in conf['datasets']: self.setup_dihadron()
            if 'moments'  in conf['datasets']: self.setup_moments()

        #--if only idis/pidis/sia/sidis being fit, always turn parallel off
        self.parallel_mode=False
        noparallel = ['dihadron','moments']
        for exp in conf['datasets']:
            if exp in noparallel: continue
            self.parallel_mode=parallel

        if self.parallel_mode:
            self.setup_parallel(nworkers)
        else:
            self.nworkers=1

    def setup_core(self):

        conf['aux']     = aux.AUX()
        conf['mellin']  = mellin.MELLIN(npts=4)
        conf['mellin-pion']=mellin.MELLIN(npts=8,extended=True)
        conf['dmellin'] = mellin.DMELLIN(nptsN=4,nptsM=4)
        conf['alphaS']  = alphaS
        conf['eweak']   = eweak.EWEAK()

        #--setup LHAPDF
        if 'lhapdf_pdf'   in conf: self.lhapdf_pdf   = conf['lhapdf_pdf']
        else:                      self.lhapdf_pdf   = 'JAM22-PDF_proton_nlo'

        if 'lhapdf_ppdf'  in conf: self.lhapdf_ppdf  = conf['lhapdf_ppdf']
        else:                      self.lhapdf_ppdf  = 'JAM22-PPDF_proton_nlo'

        if 'lhapdf_tpdf'  in conf: self.lhapdf_tpdf  = conf['lhapdf_tpdf']
        else:                      self.lhapdf_tpdf  = 'JAM22-transversity_proton_lo_nolat'

        #--number of LHAPDF replicas to load
        N = 50
        if self.load_lhapdf:
            #--load all replicas
            os.environ['LHAPDF_DATA_PATH'] = '%s/qcdlib/lhapdf'%(os.environ['FITPACK'])
            #conf['LHAPDF:PDF']   = lhapdf.mkPDFs(self.lhapdf_pdf )
            #conf['LHAPDF:PPDF']  = lhapdf.mkPDFs(self.lhapdf_ppdf)
            #conf['LHAPDF:TPDF']  = lhapdf.mkPDFs(self.lhapdf_tpdf)
            #--any way to avoid print statements here?
            conf['LHAPDF:PDF']   = [lhapdf.mkPDF(self.lhapdf_pdf ,n) for n in range(N)]
            conf['LHAPDF:PPDF']  = [lhapdf.mkPDF(self.lhapdf_ppdf,n) for n in range(N)]
            conf['LHAPDF:TPDF']  = [lhapdf.mkPDF(self.lhapdf_tpdf,n) for n in range(N)]

        if 'use LHAPDF' in conf and conf['use LHAPDF']: self.xspace = True
        else:                                           self.xspace = False
        #self.xspace = True

        if 'tpdf' in conf['params']:

            conf['tpdf']     = tpdf.TPDF()
            conf['tpdf-mom'] = tpdf.TPDF(mellin.IMELLIN())
            conf['tpdfA']    = conf['tpdf']
            conf['tpdfB']    = conf['tpdf']

        if 'diffpippim' in conf['params']:

            conf['diffpippim'] = diff.DIFF()

        if 'tdiffpippim' in conf['params']:

            conf['tdiffpippim'] = tdiff .TDIFF()

    #--observables

    def setup_dihadron(self):

        conf['path2dihadrontab'] = '%s/grids/dihadron'%os.environ['FITPACK']
        self.dihadron_tabs=obslib.dihadron.reader.READER().load_data_sets('dihadron')
        conf['dihadron tabs'] = self.dihadron_tabs
        if self.xspace: self.dihadron_thy=obslib.dihadron.theory.THEORY(predict=True)
        else:           self.dihadron_thy=obslib.dihadron.mellin.MELLIN()
        conf['dihadron'] = self.dihadron_thy
        self.dihadron_res=obslib.dihadron.residuals.RESIDUALS()

    def setup_moments(self):

        self.moments_tabs=obslib.moments.reader.READER().load_data_sets('moments')
        conf['moments tabs'] = self.moments_tabs
        self.moments_res=obslib.moments.residuals.RESIDUALS()

    #--parallelization

    def setup_parallel(self,nworkers):
        self.parallel=PARALLEL()
        self.parallel.task=self.task
        self.parallel.set_state=self.set_state
        self.parallel.setup_master()
        self.parallel.setup_workers(nworkers)
        self.nworkers=nworkers

    def get_state(self,istate=1):
        state={}
        if istate==1:
            state['istate']=1
            if 'diffpippim'  in conf: state['diffpippim']  = conf['diffpippim'].get_state() 
            if 'tdiffpippim' in conf: state['tdiffpippim'] = conf['tdiffpippim'].get_state() 
            if 'tpdf'        in conf: state['tpdf']        = conf['tpdf'].get_state() 
        return state

    def set_state(self,state):
        if state['istate']==1:
            if 'diffpippim'  in conf: conf['diffpippim'].set_state() 
            if 'tdiffpippim' in conf: conf['tdiffpippim'].set_state() 
            if 'tpdf'        in conf: conf['tpdf'].set_state() 

    def task(self,task):

        pass
        if task['reaction']=='dihadron':
            return self.dihadron_thy.process_request(task)

    def get_residuals(self,par, initial = False):

        self.parman.set_new_params(par, initial = initial)

        data = conf['datasets']

        ###################
        #--parallelization
        ###################
        #--dihadron: not in parallel
        #--moments: not in parallel

        if self.parallel_mode:

            sys.exit()

            state=self.get_state(1)
            self.parallel.update_workers(state)

            #--Level 1
            tasks=[]
            if 'dihadron'      in  data:   tasks.extend(self.dihadron_thy.get_tasks())
            results=self.parallel.send_tasks(tasks)

            for task in results:
                if task['reaction']=='dihadron'     : self.dihadron_thy.update(task)

        #####################
        #--compute residuals
        #####################
        res,rres,nres=[],[],[]
        if 'dihadron' in conf['datasets']:
            if self.xspace==False:
                tasks = self.dihadron_thy.get_tasks()
                for task in tasks: self.dihadron_thy.process_request(task)
                for task in tasks: self.dihadron_thy.update(task)
            out=self.dihadron_res.get_residuals()
            res=np.append(res,out[0])
            rres=np.append(rres,out[1])
            nres=np.append(nres,out[2])
        if 'moments' in conf['datasets']:
            out=self.moments_res.get_residuals()
            res=np.append(res,out[0])
            rres=np.append(rres,out[1])
            nres=np.append(nres,out[2])
        if 'SofferBound' in conf and conf['SofferBound'] and 'tpdf' in conf:
            violations = conf['tpdf'].check_SB()
            res = np.append(res, violations)
            conf['SB chi2'] = np.sum(violations**2)
        if 'D1 positivity' in conf and conf['D1 positivity']\
        or 'D1_positivity' in conf and conf['D1_positivity']:
            if 'diffpippim'        in conf: 
                violations = conf['diffpippim'].check_positivity()
                res = np.append(res, violations)
                conf['D1 pos chi2'] = np.sum(violations**2)
        if 'H1 positivity' in conf and conf['H1 positivity']\
        or 'H1_positivity' in conf and conf['H1_positivity']:
            if 'tdiffpippim'        in conf:
                violations = conf['tdiffpippim'].check_positivity()
                res = np.append(res, violations)
                conf['H1 pos chi2'] = np.sum(violations**2)

        return res,rres,nres

    #--aux funcs

    def get_data_info(self):

        #--compute residuals
        reaction=[]
        if 'dihadron' in conf['datasets']:
            out = self.dihadron_res.get_residuals(calc = False)
            reaction.extend(['dihadron' for _ in out[0]])
        if 'moments' in conf['datasets']:
            out = self.moments_res.get_residuals(calc = False)
            reaction.extend(['moments' for _ in out[0]])
        return reaction

    def gen_report(self,verb=0,level=0):
        L=[]
        if 'dihadron' in conf['datasets']: L.extend(self.dihadron_res.gen_report(verb, level))
        if 'moments'  in conf['datasets']: L.extend(self.moments_res.gen_report(verb, level))

        if 'SofferBound' in conf and conf['SofferBound'] and 'tpdf' in conf:
            L.extend(conf['tpdf'].gen_report(verb,level))
        if 'D1 positivity' in conf and conf['D1 positivity']\
        or 'D1_positivity' in conf and conf['D1_positivity']:
            if 'diffpippim'        in conf: L.extend(conf['diffpippim']       .gen_report(verb,level))
        if 'H1 positivity' in conf and conf['H1 positivity']\
        or 'H1_positivity' in conf and conf['H1_positivity']:
            if 'tdiffpippim'        in conf: L.extend(conf['tdiffpippim']      .gen_report(verb,level))

        if 'jupytermode' not in conf:
            return L
        else:
            msg=''
            for _ in L: msg+=_
            return msg

    def get_chi2(self):
        data={}
        if 'dihadron' in conf['datasets']: data.update(self.dihadron_res.get_chi2())
        if 'moments'  in conf['datasets']: data.update(self.moments_res.get_chi2())
        return data

    def test(self,ntasks=1):
        #--loop over states
        # print('='*20)
        import psutil
        t=time.time()
        for _ in range(ntasks):
            par=self.parman.gen_flat(setup=True)
            res,rres,nres=self.get_residuals(par)
            # res,rres,nres=self.get_residuals(self.parman.par)
            chi2=np.sum(res**2)
            chi2 /= float(len(res))
            print('(%03d/%03d) chi2=%f'%(_ + 1, ntasks, chi2))
        # print('='*20)
        process = psutil.Process(os.getpid())
        memory_rss = process.memory_info().rss / (1024.0 ** 3.0)
        # memory_vms = process.memory_info().vms / (1024.0 ** 3.0) ## memory allocated, including swapped memory
        elapsed_time=time.time()-t
        # print('with %d workers, elapsed time: %f' % (self.nworkers, elapsed_time))
        return elapsed_time, memory_rss

    def shutdown(self):
        if self.parallel_mode:
            self.parallel.stop_workers()

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


  
class MELLIN():
 
    def __init__(self,gen=False):
        self.gen = gen
        self.parts = ['NM','NCM']
        #--M grid
        self.M0 = {}
        self.M0['UU'] = np.array([0.28, 0.40, 0.50, 0.70, 0.75, 0.80, 0.90, 1.00, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00])
        self.M0['UT'] = np.array([0.28, 0.50, 0.70, 0.85, 1.00, 1.20, 1.60, 2.00])

        self.nge = 3
        self.ngM = 6
        self.ngP = 5
 
        self.xg,  self.wg  = np.polynomial.legendre.leggauss(ng)
        self.xge, self.wge = np.polynomial.legendre.leggauss(self.nge)
        self.xgM, self.wgM = np.polynomial.legendre.leggauss(self.ngM)
        self.xgP, self.wgP = np.polynomial.legendre.leggauss(self.ngP)


        self.flavs = ['u','d','s','c','ub','db','sb','cb','g']
        #--unpolarized channels (all are nonzero)
        self.channels_UU =      ['QQ,QQ','QQp,QpQ','QQp,QQp','QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ']
        self.channels_UU.extend(['QQB,GG','GQ,GQ','GQ,QG','QG,GQ','QG,QG','GG,GG','GG,QQB'])

        #--polarized channels (only five are nonzero)
        self.channels_UT =      ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ','GQ,QG']
        
        if 'integrate' in conf: self.integrate = conf['integrate']
        else:                   self.integrate = True

        if 'update_unpol' in conf: self.update_unpol = conf['update_unpol']
        else:                      self.update_unpol = True
        self.DEN = {}

        self.muF2 = lambda PhT: PhT**2

        self.setup()

    def setup(self):
        if self.gen==False:
            self.load_melltab()
            self.mellin = conf['mellin']
            self.setup_interpolation()
        self.setup_lum()
          
    def load_melltab(self):
        self.mtab={}
        path2dihadrontab='./grids/'
        self.tabs = conf['dihadron tabs']


        self.RS = [200]

        for rs in self.RS:
            self.mtab[rs]={}
            self.mtab[rs]['UU']={}
            self.mtab[rs]['UT']={}
            directUT   = '%s/UT' %(path2dihadrontab)
            directUU   = '%s/UU' %(path2dihadrontab)
            #--load UT
            npts=len(os.listdir(directUT))
            for i in range(npts):
                lprint('loading DIHADRON tables of RS = %s (UT) [%s/%s]'%(int(rs),i+1,npts))
                fname='%s/%s.melltab'%(directUT,i)
                self.mtab[rs]['UT'][i]=load(fname)
            print()
            #--load UU
            npts=len(os.listdir(directUU))
            for i in range(npts):
                lprint('loading DIHADRON tables of RS = %s (UU) [%s/%s]'%(int(rs),i+1,npts))
                fname='%s/%s.melltab'%(directUU,i)
                self.mtab[rs]['UU'][i]=load(fname)
            print()

    def setup_lum(self):

        lum = {}

        lum['QQ,QQ'] =     [['u','u','u','u']]
        lum['QQ,QQ'].append(['d','d','d','d'])
        lum['QQ,QQ'].append(['s','s','s','s'])
        lum['QQ,QQ'].append(['c','c','c','c'])
        lum['QQ,QQ'].append(['ub','ub','ub','ub'])
        lum['QQ,QQ'].append(['db','db','db','db'])
        lum['QQ,QQ'].append(['sb','sb','sb','sb'])
        lum['QQ,QQ'].append(['cb','cb','cb','cb'])



        lum['QQp,QpQ'] =     [['u','d','d','u']]
        lum['QQp,QpQ'].append(['u','s','s','u'])
        lum['QQp,QpQ'].append(['u','c','c','u'])

        lum['QQp,QpQ'].append(['d','u','u','d'])
        lum['QQp,QpQ'].append(['d','s','s','d'])
        lum['QQp,QpQ'].append(['d','c','c','d'])

        lum['QQp,QpQ'].append(['s','u','u','s'])
        lum['QQp,QpQ'].append(['s','d','d','s'])
        lum['QQp,QpQ'].append(['s','c','c','s'])

        lum['QQp,QpQ'].append(['c','u','u','c'])
        lum['QQp,QpQ'].append(['c','d','d','c'])
        lum['QQp,QpQ'].append(['c','s','s','c'])

        lum['QQp,QpQ'].append(['ub','db','db','ub'])
        lum['QQp,QpQ'].append(['ub','sb','sb','ub'])
        lum['QQp,QpQ'].append(['ub','cb','cb','ub'])

        lum['QQp,QpQ'].append(['db','ub','ub','db'])
        lum['QQp,QpQ'].append(['db','sb','sb','db'])
        lum['QQp,QpQ'].append(['db','cb','cb','db'])

        lum['QQp,QpQ'].append(['sb','ub','ub','sb'])
        lum['QQp,QpQ'].append(['sb','db','db','sb'])
        lum['QQp,QpQ'].append(['sb','cb','cb','sb'])

        lum['QQp,QpQ'].append(['cb','ub','ub','cb'])
        lum['QQp,QpQ'].append(['cb','db','db','cb'])
        lum['QQp,QpQ'].append(['cb','sb','sb','cb'])

        lum['QQp,QpQ'].append(['u','db','db','u'])
        lum['QQp,QpQ'].append(['u','sb','sb','u'])
        lum['QQp,QpQ'].append(['u','cb','cb','u'])

        lum['QQp,QpQ'].append(['d','ub','ub','d'])
        lum['QQp,QpQ'].append(['d','sb','sb','d'])
        lum['QQp,QpQ'].append(['d','cb','cb','d'])

        lum['QQp,QpQ'].append(['s','ub','ub','s'])
        lum['QQp,QpQ'].append(['s','db','db','s'])
        lum['QQp,QpQ'].append(['s','cb','cb','s'])

        lum['QQp,QpQ'].append(['c','ub','ub','c'])
        lum['QQp,QpQ'].append(['c','db','db','c'])
        lum['QQp,QpQ'].append(['c','sb','sb','c'])

        lum['QQp,QpQ'].append(['ub','d','d','ub'])
        lum['QQp,QpQ'].append(['ub','s','s','ub'])
        lum['QQp,QpQ'].append(['ub','c','c','ub'])

        lum['QQp,QpQ'].append(['db','u','u','db'])
        lum['QQp,QpQ'].append(['db','s','s','db'])
        lum['QQp,QpQ'].append(['db','c','c','db'])

        lum['QQp,QpQ'].append(['sb','u','u','sb'])
        lum['QQp,QpQ'].append(['sb','d','d','sb'])
        lum['QQp,QpQ'].append(['sb','c','c','sb'])

        lum['QQp,QpQ'].append(['cb','u','u','cb'])
        lum['QQp,QpQ'].append(['cb','d','d','cb'])
        lum['QQp,QpQ'].append(['cb','s','s','cb'])




        lum['QQp,QQp'] =     [['u','d','u','d']]
        lum['QQp,QQp'].append(['u','s','u','s'])
        lum['QQp,QQp'].append(['u','c','u','c'])

        lum['QQp,QQp'].append(['d','u','d','u'])
        lum['QQp,QQp'].append(['d','s','d','s'])
        lum['QQp,QQp'].append(['d','c','d','c'])

        lum['QQp,QQp'].append(['s','u','s','u'])
        lum['QQp,QQp'].append(['s','d','s','d'])
        lum['QQp,QQp'].append(['s','c','s','c'])

        lum['QQp,QQp'].append(['c','u','c','u'])
        lum['QQp,QQp'].append(['c','d','c','d'])
        lum['QQp,QQp'].append(['c','s','c','s'])

        lum['QQp,QQp'].append(['ub','db','ub','db'])
        lum['QQp,QQp'].append(['ub','sb','ub','sb'])
        lum['QQp,QQp'].append(['ub','cb','ub','cb'])

        lum['QQp,QQp'].append(['db','ub','db','ub'])
        lum['QQp,QQp'].append(['db','sb','db','sb'])
        lum['QQp,QQp'].append(['db','cb','db','cb'])

        lum['QQp,QQp'].append(['sb','ub','sb','ub'])
        lum['QQp,QQp'].append(['sb','db','sb','db'])
        lum['QQp,QQp'].append(['sb','cb','sb','cb'])

        lum['QQp,QQp'].append(['cb','ub','cb','ub'])
        lum['QQp,QQp'].append(['cb','db','cb','db'])
        lum['QQp,QQp'].append(['cb','sb','cb','sb'])

        lum['QQp,QQp'].append(['u','db','u','db'])
        lum['QQp,QQp'].append(['u','sb','u','sb'])
        lum['QQp,QQp'].append(['u','cb','u','cb'])

        lum['QQp,QQp'].append(['d','ub','d','ub'])
        lum['QQp,QQp'].append(['d','sb','d','sb'])
        lum['QQp,QQp'].append(['d','cb','d','cb'])

        lum['QQp,QQp'].append(['s','ub','s','ub'])
        lum['QQp,QQp'].append(['s','db','s','db'])
        lum['QQp,QQp'].append(['s','cb','s','cb'])

        lum['QQp,QQp'].append(['c','ub','c','ub'])
        lum['QQp,QQp'].append(['c','db','c','db'])
        lum['QQp,QQp'].append(['c','sb','c','sb'])

        lum['QQp,QQp'].append(['ub','d','ub','d'])
        lum['QQp,QQp'].append(['ub','s','ub','s'])
        lum['QQp,QQp'].append(['ub','c','ub','c'])

        lum['QQp,QQp'].append(['db','u','db','u'])
        lum['QQp,QQp'].append(['db','s','db','s'])
        lum['QQp,QQp'].append(['db','c','db','c'])

        lum['QQp,QQp'].append(['sb','u','sb','u'])
        lum['QQp,QQp'].append(['sb','d','sb','d'])
        lum['QQp,QQp'].append(['sb','c','sb','c'])

        lum['QQp,QQp'].append(['cb','u','cb','u'])
        lum['QQp,QQp'].append(['cb','d','cb','d'])
        lum['QQp,QQp'].append(['cb','s','cb','s'])




        lum['QQB,QpQBp'] =     [['u','ub','d','db']]
        lum['QQB,QpQBp'].append(['u','ub','s','sb'])
        lum['QQB,QpQBp'].append(['u','ub','c','cb'])

        lum['QQB,QpQBp'].append(['d','db','u','ub'])
        lum['QQB,QpQBp'].append(['d','db','s','sb'])
        lum['QQB,QpQBp'].append(['d','db','c','cb'])

        lum['QQB,QpQBp'].append(['s','sb','u','ub'])
        lum['QQB,QpQBp'].append(['s','sb','d','db'])
        lum['QQB,QpQBp'].append(['s','sb','c','cb'])

        lum['QQB,QpQBp'].append(['c','cb','u','ub'])
        lum['QQB,QpQBp'].append(['c','cb','d','db'])
        lum['QQB,QpQBp'].append(['c','cb','s','sb'])

        lum['QQB,QpQBp'].append(['ub','u','db','d'])
        lum['QQB,QpQBp'].append(['ub','u','sb','s'])
        lum['QQB,QpQBp'].append(['ub','u','cb','c'])

        lum['QQB,QpQBp'].append(['db','d','ub','u'])
        lum['QQB,QpQBp'].append(['db','d','sb','s'])
        lum['QQB,QpQBp'].append(['db','d','cb','c'])

        lum['QQB,QpQBp'].append(['sb','s','ub','u'])
        lum['QQB,QpQBp'].append(['sb','s','db','d'])
        lum['QQB,QpQBp'].append(['sb','s','cb','c'])

        lum['QQB,QpQBp'].append(['cb','c','ub','u'])
        lum['QQB,QpQBp'].append(['cb','c','db','d'])
        lum['QQB,QpQBp'].append(['cb','c','sb','s'])



        lum['QQB,QBpQp'] =     [['u','ub','db','d']]
        lum['QQB,QBpQp'].append(['u','ub','sb','s'])
        lum['QQB,QBpQp'].append(['u','ub','cb','c'])

        lum['QQB,QBpQp'].append(['d','db','ub','u'])
        lum['QQB,QBpQp'].append(['d','db','sb','s'])
        lum['QQB,QBpQp'].append(['d','db','cb','c'])

        lum['QQB,QBpQp'].append(['s','sb','ub','u'])
        lum['QQB,QBpQp'].append(['s','sb','db','d'])
        lum['QQB,QBpQp'].append(['s','sb','cb','c'])

        lum['QQB,QBpQp'].append(['c','cb','ub','u'])
        lum['QQB,QBpQp'].append(['c','cb','db','d'])
        lum['QQB,QBpQp'].append(['c','cb','sb','s'])

        lum['QQB,QBpQp'].append(['ub','u','d','db'])
        lum['QQB,QBpQp'].append(['ub','u','s','sb'])
        lum['QQB,QBpQp'].append(['ub','u','c','cb'])

        lum['QQB,QBpQp'].append(['db','d','u','ub'])
        lum['QQB,QBpQp'].append(['db','d','s','sb'])
        lum['QQB,QBpQp'].append(['db','d','c','cb'])

        lum['QQB,QBpQp'].append(['sb','s','u','ub'])
        lum['QQB,QBpQp'].append(['sb','s','d','db'])
        lum['QQB,QBpQp'].append(['sb','s','c','cb'])

        lum['QQB,QBpQp'].append(['cb','c','u','ub'])
        lum['QQB,QBpQp'].append(['cb','c','d','db'])
        lum['QQB,QBpQp'].append(['cb','c','s','sb'])






        lum['QQB,QQB'] =     [['u','ub','u','ub']]
        lum['QQB,QQB'].append(['d','db','d','db'])
        lum['QQB,QQB'].append(['s','sb','s','sb'])
        lum['QQB,QQB'].append(['c','cb','c','cb'])
        lum['QQB,QQB'].append(['ub','u','ub','u'])
        lum['QQB,QQB'].append(['db','d','db','d'])
        lum['QQB,QQB'].append(['sb','s','sb','s'])
        lum['QQB,QQB'].append(['cb','c','cb','c'])



        lum['QQB,QBQ'] =     [['u','ub','ub','u']]
        lum['QQB,QBQ'].append(['d','db','db','d'])
        lum['QQB,QBQ'].append(['s','sb','sb','s'])
        lum['QQB,QBQ'].append(['c','cb','cb','c'])
        lum['QQB,QBQ'].append(['ub','u','u','ub'])
        lum['QQB,QBQ'].append(['db','d','d','db'])
        lum['QQB,QBQ'].append(['sb','s','s','sb'])
        lum['QQB,QBQ'].append(['cb','c','c','cb'])



        lum['QQB,GG'] =     [['u','ub','g','g']]
        lum['QQB,GG'].append(['d','db','g','g'])
        lum['QQB,GG'].append(['s','sb','g','g'])
        lum['QQB,GG'].append(['c','cb','g','g'])
        lum['QQB,GG'].append(['ub','u','g','g'])
        lum['QQB,GG'].append(['db','d','g','g'])
        lum['QQB,GG'].append(['sb','s','g','g'])
        lum['QQB,GG'].append(['cb','c','g','g'])



        lum['GQ,GQ'] =     [['g','u' ,'g','u']]
        lum['GQ,GQ'].append(['g','d' ,'g','d'])
        lum['GQ,GQ'].append(['g','s' ,'g','s'])
        lum['GQ,GQ'].append(['g','c' ,'g','c'])
        lum['GQ,GQ'].append(['g','ub','g','ub'])
        lum['GQ,GQ'].append(['g','db','g','db'])
        lum['GQ,GQ'].append(['g','sb','g','sb'])
        lum['GQ,GQ'].append(['g','cb','g','cb'])



        lum['GQ,QG'] =     [['g','u' ,'u' ,'g']]
        lum['GQ,QG'].append(['g','d' ,'d' ,'g'])
        lum['GQ,QG'].append(['g','s' ,'s' ,'g'])
        lum['GQ,QG'].append(['g','c' ,'c' ,'g'])
        lum['GQ,QG'].append(['g','ub','ub','g'])
        lum['GQ,QG'].append(['g','db','db','g'])
        lum['GQ,QG'].append(['g','sb','sb','g'])
        lum['GQ,QG'].append(['g','cb','cb','g'])



        lum['QG,GQ'] =     [['u' ,'g','g','u']]
        lum['QG,GQ'].append(['d' ,'g','g','d'])
        lum['QG,GQ'].append(['s' ,'g','g','s'])
        lum['QG,GQ'].append(['c' ,'g','g','c'])
        lum['QG,GQ'].append(['ub','g','g','ub'])
        lum['QG,GQ'].append(['db','g','g','db'])
        lum['QG,GQ'].append(['sb','g','g','sb'])
        lum['QG,GQ'].append(['cb','g','g','cb'])



        lum['QG,QG'] =     [['u' ,'g','u' ,'g']]
        lum['QG,QG'].append(['d' ,'g','d' ,'g'])
        lum['QG,QG'].append(['s' ,'g','s' ,'g'])
        lum['QG,QG'].append(['c' ,'g','c' ,'g'])
        lum['QG,QG'].append(['ub','g','ub','g'])
        lum['QG,QG'].append(['db','g','db','g'])
        lum['QG,QG'].append(['sb','g','sb','g'])
        lum['QG,QG'].append(['cb','g','cb','g'])



        lum['GG,GG'] =     [['g','g','g','g']]



        lum['GG,QQB'] =     [['g','g','u','ub']]
        lum['GG,QQB'].append(['g','g','d','db'])
        lum['GG,QQB'].append(['g','g','s','sb'])
        lum['GG,QQB'].append(['g','g','c','cb'])
        lum['GG,QQB'].append(['g','g','ub','u'])
        lum['GG,QQB'].append(['g','g','db','d'])
        lum['GG,QQB'].append(['g','g','sb','s'])
        lum['GG,QQB'].append(['g','g','cb','c'])

        self.lum = lum

    def get_pdfs(self,PDF,muF2,ic=False):

        PDF.evolve([muF2])

        data = {}

        for flav in self.flavs:
            data[flav] = PDF.storage[muF2][flav]
            if ic: data[flav] = np.conjugate(data[flav])

        return data

    def get_diffs(self,dist,muF2,Mh,ic=False):

        DIFF = conf[dist]

        DIFF.evolve([muF2],[Mh])

        data = {}

        N = conf['mellin'].N

        for flav in self.flavs:
            data[flav] = DIFF.storage[muF2][Mh][flav]
            if ic: data[flav] = np.conjugate(data[flav])


        return data

    def get_mlum(self,flavs,pdfB,diffC):

        esum = lambda a, b: np.einsum('i, j', a, b)

        B = flavs[1]
        C = flavs[2]

        lum = esum(pdfB[B],diffC[C])

        return lum
  
    def get_asym(self,idx):

        update = self.update_unpol
        if idx not in self.DEN: update = True


        RS  =  self.tabs[idx]['RS']
        if 'etamin' in self.tabs[idx]:
            etamin  =  self.tabs[idx]['etamin']
            etamax  =  self.tabs[idx]['etamax']
            int_eta = True
        else:
            eta     =  self.tabs[idx]['eta']
            int_eta = False

        if 'Mmin' in self.tabs[idx]:
            Mmin  =  self.tabs[idx]['Mmin']
            Mmax  =  self.tabs[idx]['Mmax']
            int_M = True
        else:
            Mh    =  self.tabs[idx]['M']
            int_M = False

        if 'PhTmin' in self.tabs[idx]:
            PhTmin  =  self.tabs[idx]['PhTmin']
            PhTmax  =  self.tabs[idx]['PhTmax']
            int_PhT = True
        else:
            PhT     =  self.tabs[idx]['PhT']
            int_PhT = False

        if self.integrate==False:
            eta     =  self.tabs[idx]['eta']
            int_eta = False
            Mh    =  self.tabs[idx]['M']
            int_M = False
            PhT     =  self.tabs[idx]['PhT']
            int_PhT = False

        l = len(RS)

        mpi = conf['aux'].Mpi

        #--prefactor according to Pavia

        #--prefactor according to us
        pre_num = lambda M, PhT: 2.0*PhT
        pre_den = lambda M, PhT: 2.0*PhT

        nge, xge, wge = self.nge, self.xge, self.wge
        ngM, xgM, wgM = self.ngM, self.xgM, self.wgM
        ngP, xgP, wgP = self.ngP, self.xgP, self.wgP

        if int_eta==False: nge = 1
        if int_M  ==False: ngM = 1
        if int_PhT==False: ngP = 1

        if self.integrate==False: nge,ngM,ngP = 1,1,1

        shape = (l,nge,ngM,ngP)
        ETA = np.zeros(shape)
        M   = np.zeros(shape)
        PHT = np.zeros(shape)

        for m in range(l):

            if int_eta:
                _ETA = 0.5*(etamax[m]-etamin[m])*xge + 0.5*(etamax[m]+etamin[m])
                jac_ETA = 0.5*(etamax[m]-etamin[m])
                bin_ETA = etamax[m]-etamin[m]
            else:
                _ETA = [eta[m]]
                jac_ETA = 1.0
                bin_ETA = 1.0

            if int_M:
                _M = 0.5*(Mmax[m]-Mmin[m])*xgM + 0.5*(Mmax[m]+Mmin[m])
                jac_M = 0.5*(Mmax[m]-Mmin[m])
                bin_M = Mmax[m]-Mmin[m]
            else:
                _M = [Mh[m]]
                jac_M = 1.0 
                bin_M = 1.0

            if int_PhT:
                _PHT = 0.5*(PhTmax[m]-PhTmin[m])*xgP + 0.5*(PhTmax[m]+PhTmin[m])
                jac_PHT = 0.5*(PhTmax[m]-PhTmin[m])
                bin_PHT = PhTmax[m]-PhTmin[m]
            else:
                _PHT = [PhT[m]]
                jac_PHT = 1.0 
                bin_PHT = 1.0

            for i in range(nge):
                for j in range(ngM):
                    for k in range(ngP):
                        ETA[m][i][j][k] = _ETA[i]
                        M  [m][i][j][k] = _M[j]
                        PHT[m][i][j][k] = _PHT[k]
 
 
        ETA = ETA.flatten()
        M   = M.flatten()
        PHT = PHT.flatten()

        NUM = pre_num(M,PHT)*self.get_mxsec(RS[0],PHT,M,ETA,'UT')
        NUM = np.array(NUM)
        NUM = NUM.reshape(shape)
        NUM = np.sum(wgP*NUM,axis=3)
        NUM = np.sum(wgM*NUM,axis=2)
        NUM = np.sum(wge*NUM,axis=1)
        NUM = NUM*jac_ETA*jac_M*jac_PHT/bin_ETA/bin_M/bin_PHT

        if update:
            DEN = pre_den(M,PHT)*self.get_mxsec(RS[0],PHT,M,ETA,'UU')
            DEN = np.array(DEN)
            DEN = DEN.reshape(shape)
            DEN = np.sum(wgP*DEN,axis=3)
            DEN = np.sum(wgM*DEN,axis=2)
            DEN = np.sum(wge*DEN,axis=1)
            DEN = DEN*jac_ETA*jac_M*jac_PHT/bin_ETA/bin_M/bin_PHT
            self.DEN[idx] = DEN

        THY = NUM/self.DEN[idx]

        return THY,NUM,self.DEN[idx]

    def get_mxsec(self,RS,PhT,M,eta,xsec):

        eta = np.array(eta)
        PhT = np.array(PhT)
        M   = np.array(M)
   
        ETA0 = np.array(self.grid[RS][xsec]['eta'])
        PHT0 = np.array(self.grid[RS][xsec]['PhT'])
        M0   = self.M0[xsec]
        VAL  = np.array(self.grid[RS][xsec]['value'])


        #--smoothing factors to make interpolation easier
        aN200 = 5.00
        aD200 = 6.00
        aN500 = 4.00
        aD500 = 6.00
        if RS==200: 
            if xsec=='UT': factor0,factor = PHT0**aN200, PhT**aN200
            if xsec=='UU': factor0,factor = PHT0**aD200, PhT**aD200
        if RS==500: 
            if xsec=='UT': factor0,factor = PHT0**aN500, PhT**aN500
            if xsec=='UU': factor0,factor = PHT0**aD500, PhT**aD500


        interp = []
        for j in range(len(M0)):
            interp.append(griddata((ETA0[j],PHT0[j]),VAL[j]*factor0[j],(eta,PhT),fill_value=0,method='cubic',rescale=True)/factor)

        interp = np.array(interp)

        #--smoothing factors to make M interpolation easier
        if xsec == 'UT': factor0,factor = np.ones(len(M0)), np.ones(len(M))  
        if xsec == 'UU': factor0,factor = np.ones(len(M0)), np.ones(len(M))

        result = np.zeros(len(eta))
        for i in range(len(eta)):
            result[i] = griddata(M0, interp.T[i]*factor0, M[i], fill_value = 0, method='cubic')/factor[i]
        t2 = time.time()

        return result

    def get_mxsec_normalized(self,RS,PhT,M,eta,xsec):

        eta = np.array(eta)
        PhT = np.array(PhT)
        M   = np.array(M)
   
        ETA0 = np.array(self.grid[RS][xsec]['eta'])
        PHT0 = np.array(self.grid[RS][xsec]['PhT'])
        M0   = self.M0[xsec]
        VAL  = np.array(self.grid[RS][xsec]['value'])


        #--smoothing factors to make interpolation easier
        aN200 = 4.00
        aD200 = 6.00
        aN500 = 4.00
        aD500 = 6.00
        if RS==200: 
            if xsec=='UT': factor0,factor = PHT0**aN200, PhT**aN200
            if xsec=='UU': factor0,factor = PHT0**aD200, PhT**aD200
        if RS==500: 
            if xsec=='UT': factor0,factor = PHT0**aN500, PhT**aN500
            if xsec=='UU': factor0,factor = PHT0**aD500, PhT**aD500


        VAL = VAL*factor0

        interp = []
        for j in range(len(M0)):
            #--normalize
            eta  = (eta  - np.mean(ETA0[j])) /np.std(ETA0[j])
            PhT  = (PhT  - np.mean(PHT0[j])) /np.std(PHT0[j])
            ETA0[j] = (ETA0[j] - np.mean(ETA0[j]))/np.std(ETA0[j])
            PHT0[j] = (PHT0[j] - np.mean(PHT0[j]))/np.std(PHT0[j])
            VAL [j] = (VAL [j] - np.mean(VAL[j])) /np.std(VAL[j])
            interp.append(griddata((ETA0[j],PHT0[j]),VAL[j],(eta,PhT),fill_value=0,method='cubic',rescale=True))
            interp[j] = (interp[j]*np.std(interp[j]) + np.mean(interp[j]))/factor


        #print(interp)
        interp = np.array(interp)

        result = np.zeros(len(eta))
        for i in range(len(eta)):
            #M0   = (M0 - np.mean(M0))  /np.std(M0)
            #M[i] = (M[i] - np.mean(M0)) /np.std(M0)
            result[i] = griddata(M0, interp.T[i], M[i], fill_value = 0, method='cubic')
        t2 = time.time()

        #--unnormalize, then divide by factor
        #result = (result*np.std(result) + np.mean(result))/factor

        return result

    #--parallelization
    def setup_interpolation(self):

        self.tasks=[]

        M0 = self.M0

        for rs in self.mtab:
            for xsec in ['UT','UU']:
                cnt = 0
                for i in range(len(self.mtab[rs][xsec])):
                    for j in range(len(M0[xsec])):
                        idx = i*len(M0[xsec]) + j
                        task = {}
                        task['xsec']     = xsec
                        task['RS']       = rs
                        task['eta']      = self.mtab[rs][xsec][i]['eta']
                        task['PhT']      = self.mtab[rs][xsec][i]['PhT']
                        task['M']        = M0[xsec][j]
                        task['reaction'] = 'dihadron'
                        task['task']     = 1
                        task['i']        = i
                        task['j']        = j
                        task['idx']      = idx
                        cnt+=1
                        self.tasks.append(task)

        #--prepare interpolation grid
        self.grid={}
        for rs in self.mtab:
            self.grid[rs] = {}
            for xsec in ['UT','UU']:
                self.grid[rs][xsec] = {}
                self.grid[rs][xsec]['value'] = np.zeros((len(M0[xsec]),len(self.mtab[rs][xsec])))
                self.grid[rs][xsec]['eta']   = np.zeros((len(M0[xsec]),len(self.mtab[rs][xsec])))
                self.grid[rs][xsec]['PhT']   = np.zeros((len(M0[xsec]),len(self.mtab[rs][xsec])))
                for i in range(len(self.mtab[rs][xsec])):
                    for j in range(len(M0[xsec])):
                        idx = i*len(M0[xsec]) + j
                        eta   = self.mtab[rs][xsec][i]['eta']
                        PhT   = self.mtab[rs][xsec][i]['PhT']
                        self.grid[rs][xsec]['eta'][j][i] = eta
                        self.grid[rs][xsec]['PhT'][j][i] = PhT
                        self.grid[rs][xsec]['value'][j][i] = 0.0

    def process_request(self,task):

        N = self.mellin.N
        M = self.mellin.N
        phase =self.mellin.phase
        phase2=self.mellin.phase**2
        W = self.mellin.W*self.mellin.JAC

        RS    = task['RS']     
        eta   = task['eta'] 
        PhT   = task['PhT'] 
        Mh    = task['M'] 
        i     = task['i']
        j     = task['j']
        xsec  = task['xsec']

        update = self.update_unpol
        if 'value' not in task:  update = True

        if xsec=='UU' and update==False: return

        ihA = 0
        ihB = 0
 
        muF2 = self.muF2(PhT)


        if xsec=='UT':
            pdfB   = self.get_pdfs (conf['tpdf']       , muF2)
            pdfBC  = self.get_pdfs (conf['tpdf']       , muF2, ic=True)
            diffC  = self.get_diffs('tdiffpippim'      , muF2, Mh)
            
            ind = 'ij, i, j, ij'

            result = 0
            for channel in self.channels_UT:
                for flavs in self.lum[channel]:
                    lum    = self.get_mlum(flavs,pdfB ,diffC)
                    lumC   = self.get_mlum(flavs,pdfBC,diffC)
                    sigma  = self.mtab[RS]['UT'][i][channel][flavs[0]]['NM']
                    sigmaC = self.mtab[RS]['UT'][i][channel][flavs[0]]['NCM']
                    result += np.real(phase2*np.einsum(ind, lum , W, W, sigma) -\
                                             np.einsum(ind, lumC, W, W, sigmaC))/(-2.0*np.pi**2)

        if xsec=='UU':
            diffC  = self.get_diffs('diffpippim', muF2, Mh)
            
            ind = 'i, i, i'

            result = 0
            for channel in self.channels_UU:
                for flavs in self.lum[channel]:
                    A, B, C = flavs[0], flavs[1], flavs[2]
                    lum    = diffC[C]
                    key    = A + ',' + B
                    sigma  = self.mtab[RS]['UU'][i][channel][key]
                    result += np.imag(phase *np.einsum(ind, lum,  W, sigma))/np.pi


        task['value']=result

        return task

    def get_tasks(self):
        return self.tasks

    def update(self,task):
        RS    = task['RS']
        xsec  = task['xsec']
        i   = task['i']
        j   = task['j']
        self.grid[RS][xsec]['value'][j][i] = task['value']



class MELLIN_NN():
 
    def __init__(self,gen=False):
        self.gen = gen
        self.parts = ['NM','NCM']
        #--M grid
        self.M0 = {}
        self.M0['UU'] = np.array([0.28, 0.40, 0.50, 0.70, 0.75, 0.80, 0.90, 1.00, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00])
        self.M0['UT'] = np.array([0.28, 0.50, 0.70, 0.85, 1.00, 1.20, 1.60, 2.00])
        self.eta = np.linspace(-1 ,1 ,5)
        self.PhT = np.linspace(2.5,15,5)
        PhT, eta = np.meshgrid(self.PhT, self.eta)
        self.PhT0 = PhT.flatten()
        self.eta0 = eta.flatten()

        self.nge = 3
        self.ngM = 6
        self.ngP = 5
 
        self.xg,  self.wg  = np.polynomial.legendre.leggauss(ng)
        self.xge, self.wge = np.polynomial.legendre.leggauss(self.nge)
        self.xgM, self.wgM = np.polynomial.legendre.leggauss(self.ngM)
        self.xgP, self.wgP = np.polynomial.legendre.leggauss(self.ngP)


        self.flavs = ['u','d','s','c','ub','db','sb','cb','g']
        #--unpolarized channels (all are nonzero)
        self.channels_UU =      ['QQ,QQ','QQp,QpQ','QQp,QQp','QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ']
        self.channels_UU.extend(['QQB,GG','GQ,GQ','GQ,QG','QG,GQ','QG,QG','GG,GG','GG,QQB'])

        #--polarized channels (only five are nonzero)
        self.channels_UT =      ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ','GQ,QG']
       
        self.parts = ['NM','NCM']
 
        if 'integrate' in conf: self.integrate = conf['integrate']
        else:                   self.integrate = True

        if 'update_unpol' in conf: self.update_unpol = conf['update_unpol']
        else:                      self.update_unpol = True
        self.DEN = {}

        self.muF2 = lambda PhT: PhT**2

        self.setup()

    def setup(self):
        self.setup_lum()
        self.mellin = conf['mellin']
        self.load_model()
        self.setup_interpolation()
        RS = 200
        self.model = {}
        self.model[RS] = {}
        self.setup_model_UT()
        self.setup_model_UU()
          
    def load_model(self):
        self.mtab={}
        path2dihadrontab='./models/'
        self.tabs = conf['dihadron tabs']


        self.RS = [200]

        for rs in self.RS:
            self.mtab[rs]={}
            self.mtab[rs]['UU']={}
            self.mtab[rs]['UT']={}
            directUT   = '%s/UT' %(path2dihadrontab)
            directUU   = '%s/UU' %(path2dihadrontab)
            #--load UT
            for channel in self.channels_UT:
                if channel not in self.mtab[rs]['UT']: self.mtab[rs]['UT'][channel]={}
                if   channel in ['QQ,QQ','QQp,QpQ']:   flavs = ['u','d','s','c','ub','db','sb','cb']
                elif channel in ['QQB,QQB','QQB,QBQ']: flavs = ['u','d','s','c','ub','db','sb','cb']
                elif channel in ['GQ,QG']:             flavs = ['g']
                for flav in flavs:
                    if flav not in self.mtab[rs]['UT'][channel]: self.mtab[rs]['UT'][channel][flav] = {}
                    for part in self.parts:
                        lprint('loading DIHADRON model of RS = %s (UT) %s/%s-%s'%(int(rs),part,channel,flav))
                        fname = '%s/%s/%s-%s'%(directUT,part,channel,flav)
                        self.mtab[rs]['UT'][channel][flav][part] = keras.models.load_model(fname)
            print()
            #continue
            #--load UU
            for channel in self.channels_UU:
                if channel not in self.mtab[rs]['UU']: self.mtab[rs]['UU'][channel]={}
                if   channel in ['QQ,QQ']:             
                    flavs = ['u,u','d,d','s,s','c,c','ub,ub','db,db','sb,sb','cb,cb']
                elif channel in ['QQp,QpQ','QQp,QQp']: 
                    flavs =      ['u,d'  ,'u,s'  ,'u,c'  ,'d,u'  ,'d,s'  ,'d,c'  ,'s,u'  ,'s,d'  ,'s,c'  ,'c,u'  ,'c,d'  ,'c,s']
                    flavs.extend(['ub,db','ub,sb','ub,cb','db,ub','db,sb','db,cb','sb,ub','sb,db','sb,cb','cb,ub','cb,db','cb,sb'])
                    flavs.extend(['u,db' ,'u,sb' ,'u,cb' ,'d,ub' ,'d,sb' ,'d,cb' ,'s,ub' ,'s,db' ,'s,cb' ,'c,ub' ,'c,db' ,'c,sb'])
                    flavs.extend(['ub,d' ,'ub,s' ,'ub,c' ,'db,u' ,'db,s' ,'db,c' ,'sb,u' ,'sb,d' ,'sb,c' ,'cb,u' ,'cb,d' ,'cb,s'])
                elif channel in ['QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ','QQB,GG']: 
                    flavs = ['u,ub','d,db','s,sb','c,cb','ub,u','db,d','sb,s','cb,c']
                elif channel in ['GQ,GQ','GQ,QG']:             
                    flavs = ['g,u','g,d','g,s','g,c','g,ub','g,db','g,sb','g,cb']
                elif channel in ['QG,GQ','QG,QG']:             
                    flavs = ['u,g','d,g','s,g','c,g','ub,g','db,g','sb,g','cb,g']
                elif channel in ['GG,GG','GG,QQB']:            
                    flavs = ['g,g']
                for flav in flavs:
                    lprint('loading DIHADRON model of RS = %s (UU) %s-%s'%(int(rs),channel,flav))
                    fname = '%s/%s-%s'%(directUU,channel,flav)
                    self.mtab[rs]['UU'][channel][flav] = keras.models.load_model(fname)
            print()

    def setup_lum(self):

        lum = {}

        lum['QQ,QQ'] =     [['u','u','u','u']]
        lum['QQ,QQ'].append(['d','d','d','d'])
        lum['QQ,QQ'].append(['s','s','s','s'])
        lum['QQ,QQ'].append(['c','c','c','c'])
        lum['QQ,QQ'].append(['ub','ub','ub','ub'])
        lum['QQ,QQ'].append(['db','db','db','db'])
        lum['QQ,QQ'].append(['sb','sb','sb','sb'])
        lum['QQ,QQ'].append(['cb','cb','cb','cb'])



        lum['QQp,QpQ'] =     [['u','d','d','u']]
        lum['QQp,QpQ'].append(['u','s','s','u'])
        lum['QQp,QpQ'].append(['u','c','c','u'])

        lum['QQp,QpQ'].append(['d','u','u','d'])
        lum['QQp,QpQ'].append(['d','s','s','d'])
        lum['QQp,QpQ'].append(['d','c','c','d'])

        lum['QQp,QpQ'].append(['s','u','u','s'])
        lum['QQp,QpQ'].append(['s','d','d','s'])
        lum['QQp,QpQ'].append(['s','c','c','s'])

        lum['QQp,QpQ'].append(['c','u','u','c'])
        lum['QQp,QpQ'].append(['c','d','d','c'])
        lum['QQp,QpQ'].append(['c','s','s','c'])

        lum['QQp,QpQ'].append(['ub','db','db','ub'])
        lum['QQp,QpQ'].append(['ub','sb','sb','ub'])
        lum['QQp,QpQ'].append(['ub','cb','cb','ub'])

        lum['QQp,QpQ'].append(['db','ub','ub','db'])
        lum['QQp,QpQ'].append(['db','sb','sb','db'])
        lum['QQp,QpQ'].append(['db','cb','cb','db'])

        lum['QQp,QpQ'].append(['sb','ub','ub','sb'])
        lum['QQp,QpQ'].append(['sb','db','db','sb'])
        lum['QQp,QpQ'].append(['sb','cb','cb','sb'])

        lum['QQp,QpQ'].append(['cb','ub','ub','cb'])
        lum['QQp,QpQ'].append(['cb','db','db','cb'])
        lum['QQp,QpQ'].append(['cb','sb','sb','cb'])

        lum['QQp,QpQ'].append(['u','db','db','u'])
        lum['QQp,QpQ'].append(['u','sb','sb','u'])
        lum['QQp,QpQ'].append(['u','cb','cb','u'])

        lum['QQp,QpQ'].append(['d','ub','ub','d'])
        lum['QQp,QpQ'].append(['d','sb','sb','d'])
        lum['QQp,QpQ'].append(['d','cb','cb','d'])

        lum['QQp,QpQ'].append(['s','ub','ub','s'])
        lum['QQp,QpQ'].append(['s','db','db','s'])
        lum['QQp,QpQ'].append(['s','cb','cb','s'])

        lum['QQp,QpQ'].append(['c','ub','ub','c'])
        lum['QQp,QpQ'].append(['c','db','db','c'])
        lum['QQp,QpQ'].append(['c','sb','sb','c'])

        lum['QQp,QpQ'].append(['ub','d','d','ub'])
        lum['QQp,QpQ'].append(['ub','s','s','ub'])
        lum['QQp,QpQ'].append(['ub','c','c','ub'])

        lum['QQp,QpQ'].append(['db','u','u','db'])
        lum['QQp,QpQ'].append(['db','s','s','db'])
        lum['QQp,QpQ'].append(['db','c','c','db'])

        lum['QQp,QpQ'].append(['sb','u','u','sb'])
        lum['QQp,QpQ'].append(['sb','d','d','sb'])
        lum['QQp,QpQ'].append(['sb','c','c','sb'])

        lum['QQp,QpQ'].append(['cb','u','u','cb'])
        lum['QQp,QpQ'].append(['cb','d','d','cb'])
        lum['QQp,QpQ'].append(['cb','s','s','cb'])




        lum['QQp,QQp'] =     [['u','d','u','d']]
        lum['QQp,QQp'].append(['u','s','u','s'])
        lum['QQp,QQp'].append(['u','c','u','c'])

        lum['QQp,QQp'].append(['d','u','d','u'])
        lum['QQp,QQp'].append(['d','s','d','s'])
        lum['QQp,QQp'].append(['d','c','d','c'])

        lum['QQp,QQp'].append(['s','u','s','u'])
        lum['QQp,QQp'].append(['s','d','s','d'])
        lum['QQp,QQp'].append(['s','c','s','c'])

        lum['QQp,QQp'].append(['c','u','c','u'])
        lum['QQp,QQp'].append(['c','d','c','d'])
        lum['QQp,QQp'].append(['c','s','c','s'])

        lum['QQp,QQp'].append(['ub','db','ub','db'])
        lum['QQp,QQp'].append(['ub','sb','ub','sb'])
        lum['QQp,QQp'].append(['ub','cb','ub','cb'])

        lum['QQp,QQp'].append(['db','ub','db','ub'])
        lum['QQp,QQp'].append(['db','sb','db','sb'])
        lum['QQp,QQp'].append(['db','cb','db','cb'])

        lum['QQp,QQp'].append(['sb','ub','sb','ub'])
        lum['QQp,QQp'].append(['sb','db','sb','db'])
        lum['QQp,QQp'].append(['sb','cb','sb','cb'])

        lum['QQp,QQp'].append(['cb','ub','cb','ub'])
        lum['QQp,QQp'].append(['cb','db','cb','db'])
        lum['QQp,QQp'].append(['cb','sb','cb','sb'])

        lum['QQp,QQp'].append(['u','db','u','db'])
        lum['QQp,QQp'].append(['u','sb','u','sb'])
        lum['QQp,QQp'].append(['u','cb','u','cb'])

        lum['QQp,QQp'].append(['d','ub','d','ub'])
        lum['QQp,QQp'].append(['d','sb','d','sb'])
        lum['QQp,QQp'].append(['d','cb','d','cb'])

        lum['QQp,QQp'].append(['s','ub','s','ub'])
        lum['QQp,QQp'].append(['s','db','s','db'])
        lum['QQp,QQp'].append(['s','cb','s','cb'])

        lum['QQp,QQp'].append(['c','ub','c','ub'])
        lum['QQp,QQp'].append(['c','db','c','db'])
        lum['QQp,QQp'].append(['c','sb','c','sb'])

        lum['QQp,QQp'].append(['ub','d','ub','d'])
        lum['QQp,QQp'].append(['ub','s','ub','s'])
        lum['QQp,QQp'].append(['ub','c','ub','c'])

        lum['QQp,QQp'].append(['db','u','db','u'])
        lum['QQp,QQp'].append(['db','s','db','s'])
        lum['QQp,QQp'].append(['db','c','db','c'])

        lum['QQp,QQp'].append(['sb','u','sb','u'])
        lum['QQp,QQp'].append(['sb','d','sb','d'])
        lum['QQp,QQp'].append(['sb','c','sb','c'])

        lum['QQp,QQp'].append(['cb','u','cb','u'])
        lum['QQp,QQp'].append(['cb','d','cb','d'])
        lum['QQp,QQp'].append(['cb','s','cb','s'])




        lum['QQB,QpQBp'] =     [['u','ub','d','db']]
        lum['QQB,QpQBp'].append(['u','ub','s','sb'])
        lum['QQB,QpQBp'].append(['u','ub','c','cb'])

        lum['QQB,QpQBp'].append(['d','db','u','ub'])
        lum['QQB,QpQBp'].append(['d','db','s','sb'])
        lum['QQB,QpQBp'].append(['d','db','c','cb'])

        lum['QQB,QpQBp'].append(['s','sb','u','ub'])
        lum['QQB,QpQBp'].append(['s','sb','d','db'])
        lum['QQB,QpQBp'].append(['s','sb','c','cb'])

        lum['QQB,QpQBp'].append(['c','cb','u','ub'])
        lum['QQB,QpQBp'].append(['c','cb','d','db'])
        lum['QQB,QpQBp'].append(['c','cb','s','sb'])

        lum['QQB,QpQBp'].append(['ub','u','db','d'])
        lum['QQB,QpQBp'].append(['ub','u','sb','s'])
        lum['QQB,QpQBp'].append(['ub','u','cb','c'])

        lum['QQB,QpQBp'].append(['db','d','ub','u'])
        lum['QQB,QpQBp'].append(['db','d','sb','s'])
        lum['QQB,QpQBp'].append(['db','d','cb','c'])

        lum['QQB,QpQBp'].append(['sb','s','ub','u'])
        lum['QQB,QpQBp'].append(['sb','s','db','d'])
        lum['QQB,QpQBp'].append(['sb','s','cb','c'])

        lum['QQB,QpQBp'].append(['cb','c','ub','u'])
        lum['QQB,QpQBp'].append(['cb','c','db','d'])
        lum['QQB,QpQBp'].append(['cb','c','sb','s'])



        lum['QQB,QBpQp'] =     [['u','ub','db','d']]
        lum['QQB,QBpQp'].append(['u','ub','sb','s'])
        lum['QQB,QBpQp'].append(['u','ub','cb','c'])

        lum['QQB,QBpQp'].append(['d','db','ub','u'])
        lum['QQB,QBpQp'].append(['d','db','sb','s'])
        lum['QQB,QBpQp'].append(['d','db','cb','c'])

        lum['QQB,QBpQp'].append(['s','sb','ub','u'])
        lum['QQB,QBpQp'].append(['s','sb','db','d'])
        lum['QQB,QBpQp'].append(['s','sb','cb','c'])

        lum['QQB,QBpQp'].append(['c','cb','ub','u'])
        lum['QQB,QBpQp'].append(['c','cb','db','d'])
        lum['QQB,QBpQp'].append(['c','cb','sb','s'])

        lum['QQB,QBpQp'].append(['ub','u','d','db'])
        lum['QQB,QBpQp'].append(['ub','u','s','sb'])
        lum['QQB,QBpQp'].append(['ub','u','c','cb'])

        lum['QQB,QBpQp'].append(['db','d','u','ub'])
        lum['QQB,QBpQp'].append(['db','d','s','sb'])
        lum['QQB,QBpQp'].append(['db','d','c','cb'])

        lum['QQB,QBpQp'].append(['sb','s','u','ub'])
        lum['QQB,QBpQp'].append(['sb','s','d','db'])
        lum['QQB,QBpQp'].append(['sb','s','c','cb'])

        lum['QQB,QBpQp'].append(['cb','c','u','ub'])
        lum['QQB,QBpQp'].append(['cb','c','d','db'])
        lum['QQB,QBpQp'].append(['cb','c','s','sb'])






        lum['QQB,QQB'] =     [['u','ub','u','ub']]
        lum['QQB,QQB'].append(['d','db','d','db'])
        lum['QQB,QQB'].append(['s','sb','s','sb'])
        lum['QQB,QQB'].append(['c','cb','c','cb'])
        lum['QQB,QQB'].append(['ub','u','ub','u'])
        lum['QQB,QQB'].append(['db','d','db','d'])
        lum['QQB,QQB'].append(['sb','s','sb','s'])
        lum['QQB,QQB'].append(['cb','c','cb','c'])



        lum['QQB,QBQ'] =     [['u','ub','ub','u']]
        lum['QQB,QBQ'].append(['d','db','db','d'])
        lum['QQB,QBQ'].append(['s','sb','sb','s'])
        lum['QQB,QBQ'].append(['c','cb','cb','c'])
        lum['QQB,QBQ'].append(['ub','u','u','ub'])
        lum['QQB,QBQ'].append(['db','d','d','db'])
        lum['QQB,QBQ'].append(['sb','s','s','sb'])
        lum['QQB,QBQ'].append(['cb','c','c','cb'])



        lum['QQB,GG'] =     [['u','ub','g','g']]
        lum['QQB,GG'].append(['d','db','g','g'])
        lum['QQB,GG'].append(['s','sb','g','g'])
        lum['QQB,GG'].append(['c','cb','g','g'])
        lum['QQB,GG'].append(['ub','u','g','g'])
        lum['QQB,GG'].append(['db','d','g','g'])
        lum['QQB,GG'].append(['sb','s','g','g'])
        lum['QQB,GG'].append(['cb','c','g','g'])



        lum['GQ,GQ'] =     [['g','u' ,'g','u']]
        lum['GQ,GQ'].append(['g','d' ,'g','d'])
        lum['GQ,GQ'].append(['g','s' ,'g','s'])
        lum['GQ,GQ'].append(['g','c' ,'g','c'])
        lum['GQ,GQ'].append(['g','ub','g','ub'])
        lum['GQ,GQ'].append(['g','db','g','db'])
        lum['GQ,GQ'].append(['g','sb','g','sb'])
        lum['GQ,GQ'].append(['g','cb','g','cb'])



        lum['GQ,QG'] =     [['g','u' ,'u' ,'g']]
        lum['GQ,QG'].append(['g','d' ,'d' ,'g'])
        lum['GQ,QG'].append(['g','s' ,'s' ,'g'])
        lum['GQ,QG'].append(['g','c' ,'c' ,'g'])
        lum['GQ,QG'].append(['g','ub','ub','g'])
        lum['GQ,QG'].append(['g','db','db','g'])
        lum['GQ,QG'].append(['g','sb','sb','g'])
        lum['GQ,QG'].append(['g','cb','cb','g'])



        lum['QG,GQ'] =     [['u' ,'g','g','u']]
        lum['QG,GQ'].append(['d' ,'g','g','d'])
        lum['QG,GQ'].append(['s' ,'g','g','s'])
        lum['QG,GQ'].append(['c' ,'g','g','c'])
        lum['QG,GQ'].append(['ub','g','g','ub'])
        lum['QG,GQ'].append(['db','g','g','db'])
        lum['QG,GQ'].append(['sb','g','g','sb'])
        lum['QG,GQ'].append(['cb','g','g','cb'])



        lum['QG,QG'] =     [['u' ,'g','u' ,'g']]
        lum['QG,QG'].append(['d' ,'g','d' ,'g'])
        lum['QG,QG'].append(['s' ,'g','s' ,'g'])
        lum['QG,QG'].append(['c' ,'g','c' ,'g'])
        lum['QG,QG'].append(['ub','g','ub','g'])
        lum['QG,QG'].append(['db','g','db','g'])
        lum['QG,QG'].append(['sb','g','sb','g'])
        lum['QG,QG'].append(['cb','g','cb','g'])



        lum['GG,GG'] =     [['g','g','g','g']]



        lum['GG,QQB'] =     [['g','g','u','ub']]
        lum['GG,QQB'].append(['g','g','d','db'])
        lum['GG,QQB'].append(['g','g','s','sb'])
        lum['GG,QQB'].append(['g','g','c','cb'])
        lum['GG,QQB'].append(['g','g','ub','u'])
        lum['GG,QQB'].append(['g','g','db','d'])
        lum['GG,QQB'].append(['g','g','sb','s'])
        lum['GG,QQB'].append(['g','g','cb','c'])

        self.lum = lum

    def get_pdfs(self,PDF,muF2,ic=False):

        PDF.evolve([muF2])

        data = {}

        for flav in self.flavs:
            data[flav] = PDF.storage[muF2][flav]
            if ic: data[flav] = np.conjugate(data[flav])

        return data

    def get_diffs(self,dist,muF2,Mh,ic=False):

        DIFF = conf[dist]

        DIFF.evolve([muF2],[Mh])

        data = {}

        N = conf['mellin'].N

        for flav in self.flavs:
            data[flav] = DIFF.storage[muF2][Mh][flav]
            if ic: data[flav] = np.conjugate(data[flav])


        return data

    def get_mlum(self,flavs,pdfB,diffC):

        esum = lambda a, b: np.einsum('i, j', a, b)

        B = flavs[1]
        C = flavs[2]

        lum = esum(pdfB[B],diffC[C])

        return lum
  
    def get_asym(self,idx):

        update = self.update_unpol
        if idx not in self.DEN: update = True


        RS  =  self.tabs[idx]['RS']
        if 'etamin' in self.tabs[idx]:
            etamin  =  self.tabs[idx]['etamin']
            etamax  =  self.tabs[idx]['etamax']
            int_eta = True
        else:
            eta     =  self.tabs[idx]['eta']
            int_eta = False

        if 'Mmin' in self.tabs[idx]:
            Mmin  =  self.tabs[idx]['Mmin']
            Mmax  =  self.tabs[idx]['Mmax']
            int_M = True
        else:
            Mh    =  self.tabs[idx]['M']
            int_M = False

        if 'PhTmin' in self.tabs[idx]:
            PhTmin  =  self.tabs[idx]['PhTmin']
            PhTmax  =  self.tabs[idx]['PhTmax']
            int_PhT = True
        else:
            PhT     =  self.tabs[idx]['PhT']
            int_PhT = False

        if self.integrate==False:
            eta     =  self.tabs[idx]['eta']
            int_eta = False
            Mh    =  self.tabs[idx]['M']
            int_M = False
            PhT     =  self.tabs[idx]['PhT']
            int_PhT = False

        l = len(RS)

        mpi = conf['aux'].Mpi

        #--prefactor according to Pavia

        #--prefactor according to us
        pre_num = lambda M, PhT: 2.0*PhT
        pre_den = lambda M, PhT: 2.0*PhT

        nge, xge, wge = self.nge, self.xge, self.wge
        ngM, xgM, wgM = self.ngM, self.xgM, self.wgM
        ngP, xgP, wgP = self.ngP, self.xgP, self.wgP

        if int_eta==False: nge = 1
        if int_M  ==False: ngM = 1
        if int_PhT==False: ngP = 1

        if self.integrate==False: nge,ngM,ngP = 1,1,1

        shape = (l,nge,ngM,ngP)
        ETA = np.zeros(shape)
        M   = np.zeros(shape)
        PHT = np.zeros(shape)

        for m in range(l):

            if int_eta:
                _ETA = 0.5*(etamax[m]-etamin[m])*xge + 0.5*(etamax[m]+etamin[m])
                jac_ETA = 0.5*(etamax[m]-etamin[m])
                bin_ETA = etamax[m]-etamin[m]
            else:
                _ETA = [eta[m]]
                jac_ETA = 1.0
                bin_ETA = 1.0

            if int_M:
                _M = 0.5*(Mmax[m]-Mmin[m])*xgM + 0.5*(Mmax[m]+Mmin[m])
                jac_M = 0.5*(Mmax[m]-Mmin[m])
                bin_M = Mmax[m]-Mmin[m]
            else:
                _M = [Mh[m]]
                jac_M = 1.0 
                bin_M = 1.0

            if int_PhT:
                _PHT = 0.5*(PhTmax[m]-PhTmin[m])*xgP + 0.5*(PhTmax[m]+PhTmin[m])
                jac_PHT = 0.5*(PhTmax[m]-PhTmin[m])
                bin_PHT = PhTmax[m]-PhTmin[m]
            else:
                _PHT = [PhT[m]]
                jac_PHT = 1.0 
                bin_PHT = 1.0

            for i in range(nge):
                for j in range(ngM):
                    for k in range(ngP):
                        ETA[m][i][j][k] = _ETA[i]
                        M  [m][i][j][k] = _M[j]
                        PHT[m][i][j][k] = _PHT[k]
 
 
        ETA = ETA.flatten()
        M   = M.flatten()
        PHT = PHT.flatten()

        NUM = pre_num(M,PHT)*self.get_mxsec(RS[0],PHT,M,ETA,'UT')
        NUM = np.array(NUM)
        NUM = NUM.reshape(shape)
        NUM = np.sum(wgP*NUM,axis=3)
        NUM = np.sum(wgM*NUM,axis=2)
        NUM = np.sum(wge*NUM,axis=1)
        NUM = NUM*jac_ETA*jac_M*jac_PHT/bin_ETA/bin_M/bin_PHT

        if update:
            DEN = pre_den(M,PHT)*self.get_mxsec(RS[0],PHT,M,ETA,'UU')
            DEN = np.array(DEN)
            DEN = DEN.reshape(shape)
            DEN = np.sum(wgP*DEN,axis=3)
            DEN = np.sum(wgM*DEN,axis=2)
            DEN = np.sum(wge*DEN,axis=1)
            DEN = DEN*jac_ETA*jac_M*jac_PHT/bin_ETA/bin_M/bin_PHT
            self.DEN[idx] = DEN

        THY = NUM/self.DEN[idx]

        return THY,NUM,self.DEN[idx]

    def get_mxsec(self,RS,PhT,M,eta,xsec):

        #if xsec=='UU': return np.zeros(len(eta))
        eta = np.array(eta)
        PhT = np.array(PhT)
        M   = np.array(M)
   
        ETA0 = self.grid[RS][xsec]['eta']
        PHT0 = self.grid[RS][xsec]['PhT']
        M0   = self.M0[xsec]
        VAL  = np.array(self.grid[RS][xsec]['value'])


        #--smoothing factors to make interpolation easier
        aN200 = 4.00
        aD200 = 6.00
        aN500 = 4.00
        aD500 = 6.00
        if RS==200: 
            if xsec=='UT': factor0,factor = PHT0**aN200, PhT**aN200
            if xsec=='UU': factor0,factor = PHT0**aD200, PhT**aD200
        if RS==500: 
            if xsec=='UT': factor0,factor = PHT0**aN500, PhT**aN500
            if xsec=='UU': factor0,factor = PHT0**aD500, PhT**aD500


        interp = []
        for j in range(len(M0)):
            interp.append(griddata((ETA0[j],PHT0[j]),VAL[j]*factor0[j],(eta,PhT),fill_value=0,method='cubic',rescale=True)/factor)

        interp = np.array(interp)

        #--smoothing factors to make M interpolation easier
        if xsec == 'UT': factor0,factor = np.ones(len(M0)), np.ones(len(M))  
        if xsec == 'UU': factor0,factor = np.ones(len(M0)), np.ones(len(M))

        result = np.zeros(len(eta))
        for i in range(len(eta)):
            result[i] = griddata(M0, interp.T[i]*factor0, M[i], fill_value = 0, method='cubic')/factor[i]
        t2 = time.time()

        return result

    #--setup models
    def setup_model_UU(self):

        xsec = 'UU'
        self.model[RS][xsec] = {}
        eta0 = self.eta0
        PhT0 = self.PhT0

        N = self.mellin.N

        ETA, PHT    = [],[]
        realN       = []
        #--single Mellin transform
        num_inputs = 3
        L = len(N)
        length=len(eta0)*len(N)
        shape = (len(self.eta0),len(N))
        
        for i in range(len(eta0)):
            ETA.extend(eta0[i]*np.ones(L))
            PHT.extend(PhT0[i]*np.ones(L))
            realN.extend(N.flatten().real)


        for channel in self.channels_UU:
            if channel not in self.model[RS][xsec]: self.model[RS][xsec][channel] = {}
            if   channel in ['QQ,QQ']:             
                flavs = ['u,u','d,d','s,s','c,c','ub,ub','db,db','sb,sb','cb,cb']
            elif channel in ['QQp,QpQ','QQp,QQp']: 
                flavs =      ['u,d'  ,'u,s'  ,'u,c'  ,'d,u'  ,'d,s'  ,'d,c'  ,'s,u'  ,'s,d'  ,'s,c'  ,'c,u'  ,'c,d'  ,'c,s']
                flavs.extend(['ub,db','ub,sb','ub,cb','db,ub','db,sb','db,cb','sb,ub','sb,db','sb,cb','cb,ub','cb,db','cb,sb'])
                flavs.extend(['u,db' ,'u,sb' ,'u,cb' ,'d,ub' ,'d,sb' ,'d,cb' ,'s,ub' ,'s,db' ,'s,cb' ,'c,ub' ,'c,db' ,'c,sb'])
                flavs.extend(['ub,d' ,'ub,s' ,'ub,c' ,'db,u' ,'db,s' ,'db,c' ,'sb,u' ,'sb,d' ,'sb,c' ,'cb,u' ,'cb,d' ,'cb,s'])
            elif channel in ['QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ','QQB,GG']: 
                flavs = ['u,ub','d,db','s,sb','c,cb','ub,u','db,d','sb,s','cb,c']
            elif channel in ['GQ,GQ','GQ,QG']:             
                flavs = ['g,u','g,d','g,s','g,c','g,ub','g,db','g,sb','g,cb']
            elif channel in ['QG,GQ','QG,QG']:             
                flavs = ['u,g','d,g','s,g','c,g','ub,g','db,g','sb,g','cb,g']
            elif channel in ['GG,GG','GG,QQB']:            
                flavs = ['g,g']
            for flav in flavs:
                #A, B, C = flavs[0], flavs[1], flavs[2]
                #key    = A + ',' + B
                stats  = np.load('modelgrids/stats/%s/%s-%s.npy'%(xsec,channel,flav),allow_pickle=True).item()
                mean,std = stats['mean'],stats['std']
                IN = {}
                IN['data']  = np.zeros((num_inputs,len(ETA)))
                IN['data'][0] = (ETA  - mean[0])/std[0]
                IN['data'][1] = (PHT  - mean[1])/std[1]
                IN['data'][2] = (realN- mean[2])/std[2]
                lprint('predicting DIHADRON model (UU) %s-%s'%(channel,flav))
                t1 = time.time()
                predict  = self.mtab[RS]['UU'][channel][flav].predict(IN['data'].T).T
                t2 = time.time()
                #print('time to do prediction',t2-t1)
                real  = predict[0]
                imag  = predict[1]
                real  = (real*std[3] + mean[3])/np.array(PHT)**6
                imag  = (imag*std[4] + mean[4])/np.array(PHT)**6
                sigma = real + 1j*imag
                sigma = np.reshape(sigma,shape)
                self.model[RS][xsec][channel][flav] = sigma
       
    def setup_model_UT(self):

        xsec = 'UT'
        self.model[RS][xsec] = {}
        eta0 = self.eta0
        PhT0 = self.PhT0

        N = self.mellin.N
        M = self.mellin.N
        shape = (len(self.eta0),len(N),len(M))

        M, N = np.meshgrid(M,N)
        N = N.flatten()
        M = M.flatten()

        ETA, PHT    = [],[]
        realN = []
        realM = []
        #--double Mellin transform
        num_inputs = 4
        L = len(N)
        length=len(eta0)*len(N)
        
        for i in range(len(eta0)):
            ETA.extend(eta0[i]*np.ones(L))
            PHT.extend(PhT0[i]*np.ones(L))
            realN.extend(N.real)
            realM.extend(M.real)

 
        for channel in self.channels_UT:
            if channel not in self.model[RS][xsec]: self.model[RS][xsec][channel] = {}
            if   channel in ['QQ,QQ','QQp,QpQ']:   flavs = ['u','d','s','c','ub','db','sb','cb']
            elif channel in ['QQB,QQB','QQB,QBQ']: flavs = ['u','d','s','c','ub','db','sb','cb']
            elif channel in ['GQ,QG']:             flavs = ['g']
            for flav in flavs:
                if flav not in self.model[RS][xsec][channel]: self.model[RS][xsec][channel][flav] = {}
                for part in self.parts:
                    stats  = np.load('modelgrids/stats/%s/%s/%s-%s.npy'%(xsec,part,channel,flav),allow_pickle=True).item()
                    mean,std = stats['mean'],stats['std']
                    IN  = np.zeros((num_inputs,len(ETA)))
                    IN[0] = (ETA  - mean[0])/std[0]
                    IN[1] = (PHT  - mean[1])/std[1]
                    IN[2] = (realN- mean[2])/std[2]
                    IN[3] = (realM- mean[3])/std[3]
                    #for i in range(len(IN[0])):
                    #    print(IN[0][i],IN[1][i],IN[2][i],IN[3][i])
                    #print(len(IN[0]))
                    #sys.exit()
                    lprint('predicting DIHADRON model (UT) %s/%s-%s'%(part,channel,flav))
                    t1 = time.time()
                    predict  = self.mtab[RS]['UT'][channel][flav][part].predict(IN.T).T
                    t2 = time.time()
                    #print('time to do prediction',t2-t1)
                    real  = predict[0]
                    imag  = predict[1]
                    real  = (real*std[4] + mean[4])/np.array(PHT)**6
                    imag  = (imag*std[5] + mean[5])/np.array(PHT)**6
                    sigma = real + 1j*imag
                    sigma = np.reshape(sigma,shape)
                    self.model[RS][xsec][channel][flav][part] = sigma
       

    #--parallelization
    def setup_interpolation(self):

        self.tasks=[]

        M0 = self.M0
        eta0 = self.eta0
        PhT0 = self.PhT0

        for rs in self.mtab:
            for xsec in self.mtab[rs]:
                cnt = 0
                for i in range(len(eta0)):
                    for j in range(len(M0[xsec])):
                        idx = i*len(M0[xsec]) + j
                        task = {}
                        task['xsec']     = xsec
                        task['RS']       = rs
                        task['eta']      = eta0[i]
                        task['PhT']      = PhT0[i]
                        task['M']        = M0[xsec][j]
                        task['reaction'] = 'dihadron'
                        task['task']     = 1
                        task['i']        = i
                        task['j']        = j
                        task['idx']      = idx
                        cnt+=1
                        self.tasks.append(task)

        #--prepare interpolation grid
        self.grid={}
        for rs in self.mtab:
            self.grid[rs] = {}
            for xsec in self.mtab[rs]:
                self.grid[rs][xsec] = {}
                self.grid[rs][xsec]['value'] = np.zeros((len(M0[xsec]),len(eta0)))
                self.grid[rs][xsec]['eta']   = np.zeros((len(M0[xsec]),len(eta0)))
                self.grid[rs][xsec]['PhT']   = np.zeros((len(M0[xsec]),len(eta0)))
                for i in range(len(eta0)):
                    for j in range(len(M0[xsec])):
                        self.grid[rs][xsec]['eta'][j][i] = eta0[i]
                        self.grid[rs][xsec]['PhT'][j][i] = PhT0[i]
                        self.grid[rs][xsec]['value'][j][i] = 0.0

    def process_request(self,task):

        N = self.mellin.N
        M = self.mellin.N
        phase =self.mellin.phase
        phase2=self.mellin.phase**2
        W = self.mellin.W*self.mellin.JAC

        RS    = task['RS']     
        eta   = task['eta'] 
        PhT   = task['PhT'] 
        Mh    = task['M'] 
        i     = task['i']
        j     = task['j']
        xsec  = task['xsec']

        update = self.update_unpol
        if 'value' not in task:  update = True

        if xsec=='UU' and update==False: return

        ihA = 0
        ihB = 0
 
        muF2 = self.muF2(PhT)


        #if xsec=='UU':
        #    task['value'] = 0.0
        #    return

        if xsec=='UT':
            pdfB   = self.get_pdfs (conf['tpdf']       , muF2)
            pdfBC  = self.get_pdfs (conf['tpdf']       , muF2, ic=True)
            diffC  = self.get_diffs('tdiffpippim'      , muF2, Mh)
            
            ind = 'ij, i, j, ij'

            result = 0
            for channel in self.channels_UT:
                for flavs in self.lum[channel]:
                    lum    = self.get_mlum(flavs,pdfB ,diffC)
                    lumC   = self.get_mlum(flavs,pdfBC,diffC)
                    sigma  = self.model[RS]['UT'][channel][flavs[0]]['NM'][i]
                    sigmaC = self.model[RS]['UT'][channel][flavs[0]]['NCM'][i]
                    result += np.real(phase2*np.einsum(ind, lum , W, W, sigma) -\
                                             np.einsum(ind, lumC, W, W, sigmaC))/(-2.0*np.pi**2)

        if xsec=='UU':
            diffC  = self.get_diffs('diffpippim', muF2, Mh)
            
            ind = 'i, i, i'

            result = 0
            for channel in self.channels_UU:
                for flavs in self.lum[channel]:
                    A, B, C = flavs[0], flavs[1], flavs[2]
                    lum    = diffC[C]
                    key    = A + ',' + B
                    sigma  = self.model[RS]['UU'][channel][key][i]
                    result += np.imag(phase *np.einsum(ind, lum,  W, sigma))/np.pi


        task['value']=result

        return task

    def get_tasks(self):
        return self.tasks

    def update(self,task):
        RS    = task['RS']
        xsec  = task['xsec']
        i   = task['i']
        j   = task['j']
        self.grid[RS][xsec]['value'][j][i] = task['value']




  

#--test functions
#--used for basic functions like loading replicas
class CORE():

    def mod_conf(self, istep, replica=None):
    
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
                                    conf['params'][dist][par]['value']=prior_params[i]
    
            #--another version for fixed parameters 
            if 'fix parameters' in step:
                for dist in step['fix parameters']:
                    for par in step['fix parameters'][dist]:
                        conf['params'][dist][par]['fixed']=True
                        #--set prior parameters values for passive distributions
                        for istep in step['dep']:
                            prior_order=replica['order'][istep]
                            prior_params=replica['params'][istep]
                            for i in range(len(prior_order)):
                                _,_dist,_par = prior_order[i]
                                if  dist==_dist and par==_par:
                                    conf['params'][dist][par]['value']=prior_params[i]
                        
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
    
            if 'passive distributions' in conf['steps'][istep]:
                if len(conf['steps'][istep]['passive distributions']) > 0: 
                    self.get_passive_data(istep,replica)
    
    def get_replicas(self, wdir,mod_conf=None):
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
    
        istep = self.get_istep()
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
        if 'passive distributions' not in conf['steps'][istep]: dists = conf['steps'][istep]['active distributions']
        else: dists = conf['steps'][istep]['active distributions'] + (conf['steps'][istep]['passive distributions'])
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
    
    def get_replicas_names(self, wdir):
        replicas=sorted(os.listdir('%s/msr-inspected'%wdir))
        return replicas
    
    def get_istep(self):
        #--pick last step
        return sorted(conf['steps'])[-1] 
    
    def get_passive_data(self,istep,replica):
        order = replica['order'][istep]
        for i in range(len(order)):
            if order[i][0] != 2: continue
            exp = order[i][1]
            if exp in conf['datasets']: continue
            conf['datasets'][exp] = {_:{} for _ in ['norm','xlsx']}

def test_NN(wdir,RS,ng=20,gen_NN=True):

    if RS==200: PhTmin,PhTmax = 2.5, 15
    if RS==500: PhTmin,PhTmax = 4.0, 19
    #--make up data sheets
    L = 30
    idxs = [0,1,2]
    conf['dihadron tabs'] = {}
    for i in idxs:
        conf['dihadron tabs'][i] = {}
        conf['dihadron tabs'][i]['col']     = ['test']
        conf['dihadron tabs'][i]['process'] = ['pp']
        conf['dihadron tabs'][i]['hadrons'] = ['pi+,pi-']
        conf['dihadron tabs'][i]['tar']     = ['pp']
        conf['dihadron tabs'][i]['obs']     = ['A_UT']
        conf['dihadron tabs'][i]['RS']      = np.array([RS]*L)
    
    #--integrate over eta, fixed PhT, plot as function of M
    conf['dihadron tabs'][0]['M']       = np.linspace(0.28,2.00,L)
    conf['dihadron tabs'][0]['PhT']     = np.array([6.0]*L)
    conf['dihadron tabs'][0]['etamin']  = np.array([0.0]*L)
    conf['dihadron tabs'][0]['etamax']  = np.array([1.0]*L)
    #--integrate over eta, fixed M, plot as function of PhT
    conf['dihadron tabs'][1]['PhT']     = np.linspace(PhTmin,PhTmax,L)
    conf['dihadron tabs'][1]['M']       = np.array([0.80]*L)
    conf['dihadron tabs'][1]['etamin']  = np.array([0.0]*L)
    conf['dihadron tabs'][1]['etamax']  = np.array([1.0]*L)
    #--integrate over M and PhT, plot as function of eta
    conf['dihadron tabs'][2]['eta']     = np.linspace(-1,1,L)
    conf['dihadron tabs'][2]['Mmin']    = np.array([0.28]*L)
    conf['dihadron tabs'][2]['Mmax']    = np.array([2.00]*L)
    conf['dihadron tabs'][2]['PhTmin']  = np.array([PhTmin]*L)
    conf['dihadron tabs'][2]['PhTmax']  = np.array([PhTmax]*L)

    #--fixed eta, PhT, function of M
    #conf['dihadron tabs'][0]['M']       = np.linspace(0.28,2.00,L)
    #conf['dihadron tabs'][0]['PhT']     = np.array([5.000]*L)
    #conf['dihadron tabs'][0]['eta']     = np.array([0.333]*L)
    ##--fixed eta, M, function of PhT
    #conf['dihadron tabs'][1]['PhT']     = np.linspace(PhTmin,PhTmax,L)
    #conf['dihadron tabs'][1]['M']       = np.array([0.700]*L)
    #conf['dihadron tabs'][1]['eta']     = np.array([0.333]*L)
    ##--fixed M, PhT, function of eta
    #conf['dihadron tabs'][2]['eta']     = np.linspace(-1,1,L)
    #conf['dihadron tabs'][2]['M']       = np.array([0.700]*L)
    #conf['dihadron tabs'][2]['PhT']     = np.array([5.000]*L)




    core = CORE()

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    conf['SofferBound'] = False

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    #--take single replica
    par = replicas[0]

    parman.set_new_params(par,initial=True)

    if gen_NN:
        MELLNN = MELLIN_NN()
        #--get NN result
        NN = {}
        NN['thy'] = {}
        NN['num'] = {}
        NN['den'] = {}
        t1 = time.time()
        for task in MELLNN.tasks: MELLNN.process_request(task)
        for task in MELLNN.tasks: MELLNN.update(task)
        t2 = time.time()
        t  = t2 - t1
        print('Time to setup NN grids: %3.2f'%t)
        t1 = time.time()
        for idx in conf['dihadron tabs']:
            NN['thy'][idx],NN['num'][idx],NN['den'][idx] = MELLNN.get_asym(idx)
        t2 = time.time()
        t  = t2 - t1
        print('Time to get NN results: %3.2f'%t)
        checkdir('data')
        filename = 'data/mellNN-RS%d.dat'%(RS)
        save(NN,filename)
    else:
        filename = 'data/mellNN-RS%d.dat'%(RS)
        NN = load(filename)
        



    #--get Mellin space result
    MELL = MELLIN()
    t1 = time.time()
    for task in MELL.tasks: MELL.process_request(task)
    for task in MELL.tasks: MELL.update(task)
    t2 = time.time()
    t  = t2 - t1
    print('Time to setup Mellin grids: %3.2f'%t)
    mell = {}
    mell['thy'] = {}
    mell['num'] = {}
    mell['den'] = {}
    t1 = time.time()
    for idx in conf['dihadron tabs']:
        mell['thy'][idx],mell['num'][idx],mell['den'][idx] = MELL.get_asym(idx)
    t2 = time.time()
    t  = t2 - t1
    print('Time to interpolate Mellin results: %3.2f'%t)

    filename = 'data/test_interp-RS%d-ng=%d.dat'%(RS,ng)
    xspace = load(filename)

    #--get % errors
    err = {}
    err['thy'] = {}
    err['num'] = {}
    err['den'] = {}
    for idx in conf['dihadron tabs']:
        err['thy'][idx] = abs((xspace['thy'][idx] - mell['thy'][idx])/xspace['thy'][idx]) * 100
        err['num'][idx] = abs((xspace['num'][idx] - mell['num'][idx])/xspace['num'][idx]) * 100
        err['den'][idx] = abs((xspace['den'][idx] - mell['den'][idx])/xspace['den'][idx]) * 100


    nrows,ncols=3,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax21=py.subplot(nrows,ncols,4)
    ax22=py.subplot(nrows,ncols,5)
    ax23=py.subplot(nrows,ncols,6)
    ax31=py.subplot(nrows,ncols,7)
    ax32=py.subplot(nrows,ncols,8)
    ax33=py.subplot(nrows,ncols,9)

    tabs = conf['dihadron tabs']

    hand = {}

    NNcolor = 'green'
    #--plot z and Mh
    for idx in tabs:

        if idx==0: ax = ax11
        if idx==1: ax = ax12
        if idx==2: ax = ax13

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']    = ax.scatter(M,xspace['thy'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (M,mell  ['thy'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (M,NN    ['thy'][idx],color=NNcolor)
        if idx==1:
            PhT = tabs[idx]['PhT']
            hand['xspace']    = ax.scatter(PhT,xspace['thy'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (PhT,mell  ['thy'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (PhT,NN    ['thy'][idx],color=NNcolor)
        if idx==2:
            eta = tabs[idx]['eta']
            hand['xspace']    = ax.scatter(eta,xspace['thy'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (eta,mell  ['thy'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (eta,NN    ['thy'][idx],color=NNcolor)

        factor = 1e4

        if idx==0: ax = ax21
        if idx==1: ax = ax22
        if idx==2: ax = ax23

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']    = ax.scatter(M,factor*xspace['num'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (M,factor*mell  ['num'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (M,factor*NN    ['num'][idx],color=NNcolor)
        if idx==1:
            PhT = tabs[idx]['PhT']
            hand['xspace']    = ax.scatter(PhT,factor*xspace['num'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (PhT,factor*mell  ['num'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (PhT,factor*NN    ['num'][idx],color=NNcolor)
        if idx==2:
            eta = tabs[idx]['eta']
            hand['xspace']    = ax.scatter(eta,factor*xspace['num'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (eta,factor*mell  ['num'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (eta,factor*NN    ['num'][idx],color=NNcolor)

        if idx==0: ax = ax31
        if idx==1: ax = ax32
        if idx==2: ax = ax33

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']    = ax.scatter(M,factor*xspace['den'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (M,factor*mell  ['den'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (M,factor*NN    ['den'][idx],color=NNcolor)
        if idx==1:
            PhT = tabs[idx]['PhT']
            hand['xspace']    = ax.scatter(PhT,factor*xspace['den'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (PhT,factor*mell  ['den'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (PhT,factor*NN    ['den'][idx],color=NNcolor)
        if idx==2:
            eta = tabs[idx]['eta']
            hand['xspace']    = ax.scatter(eta,factor*xspace['den'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot   (eta,factor*mell  ['den'][idx],color='darkblue')
            hand['NN']       ,= ax.plot   (eta,factor*NN    ['den'][idx],color=NNcolor)

    for ax in [ax11,ax21,ax31]:
        ax.set_xlim(0.25,2.1)
        ax.set_xticks([0.4,0.8,1.2,1.6,2.0])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

    for ax in [ax12,ax22,ax32]: 
        ax.set_xlim(2,17)
        ax.set_xticks([5,10,15])
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)


    for ax in [ax13,ax23,ax33]: 
        ax.set_xlim(-1.1,1.1)
        ax.set_xticks([-0.8,-0.4,0,0.4,0.8])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

    #ax22.set_ylim(-2,2)
    #ax32.set_ylim(-200,200)


    #ax11.set_ylim(-0.1,0.20)
    #ax11.set_yticks([-0.05,0,0.05,0.10,0.15])
    #minorLocator = MultipleLocator(0.01)
    #ax11.yaxis.set_minor_locator(minorLocator)

    #ax12.set_ylim(-0.08,0.08)
    #ax12.set_yticks([-0.05,0,0.05])
    #minorLocator = MultipleLocator(0.01)
    #ax12.yaxis.set_minor_locator(minorLocator)

    #ax13.set_ylim(-0.01,0.04)
    #ax13.set_yticks([0,0.01,0.02,0.03])
    #minorLocator = MultipleLocator(0.005)
    #ax13.yaxis.set_minor_locator(minorLocator)

    ax31.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=30)
    ax32.set_xlabel(r'\boldmath$P_{hT}~[{\rm GeV}]$',size=30)
    ax33.set_xlabel(r'\boldmath$\eta$',size=30)

    ax11.set_ylabel(r'\boldmath$A_{U \perp}$',size=30)
    ax21.set_ylabel(r'\boldmath$\Delta \sigma \times 10^4$',size=30)
    ax31.set_ylabel(r'\boldmath$       \sigma \times 10^4$',size=30)

    for ax in [ax11,ax12,ax13,ax21,ax22,ax23,ax31,ax32,ax33]:
        ax .tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax .tick_params(axis='both',which='minor',size=4)
        ax .tick_params(axis='both',which='major',size=8)
        ax .axhline(0,0,1,color='black',ls='--',alpha=0.5)

    if 'eta' in tabs[0]:
        ax11.text(0.05,0.80,r'\boldmath$\eta=%3.2f$'%tabs[0]['eta'][0] ,transform=ax11.transAxes, size=25)
    else:
        ax11.text(0.05,0.80,r'\boldmath$%3.2f < \eta < %3.2f$'%(tabs[0]['etamin'][0],tabs[0]['etamax'][0]) ,transform=ax11.transAxes, size=25)
    if 'PhT' in tabs[0]:
        ax11.text(0.05,0.70,r'\boldmath$P_{hT} = %3.2f ~ {\rm GeV}$'%(tabs[0]['PhT'][0]) ,transform=ax11.transAxes, size=25)
    else:
        ax11.text(0.05,0.70,r'\boldmath$%3.2f < P_{hT} < %3.2f ~ {\rm GeV}$'%(tabs[0]['PhTmin'][0],tabs[0]['PhTmax'][0]) ,transform=ax11.transAxes, size=25)

    if 'eta' in tabs[1]:
        ax12.text(0.05,0.80,r'\boldmath$\eta=%3.2f$'%tabs[1]['eta'][0] ,transform=ax12.transAxes, size=25)
    else:
        ax12.text(0.05,0.80,r'\boldmath$%3.2f < \eta < %3.2f$'%(tabs[1]['etamin'][0],tabs[1]['etamax'][0]) ,transform=ax12.transAxes, size=25)
    if 'M' in tabs[1]:
        ax12.text(0.05,0.70,r'\boldmath$M_h = %3.2f ~ {\rm GeV}$'%(tabs[1]['M'][0]) ,transform=ax12.transAxes, size=25)
    else:
        ax12.text(0.05,0.70,r'\boldmath$%3.2f < M_h < %3.2f ~ {\rm GeV}$'%(tabs[1]['Mmin'][0],tabs[1]['Mmax'][0]) ,transform=ax12.transAxes, size=25)


    if 'PhT' in tabs[2]:
        ax13.text(0.05,0.80,r'\boldmath$P_{hT} = %3.2f ~ {\rm GeV}$'%(tabs[2]['PhT'][0]) ,transform=ax13.transAxes, size=25)
    else:
        ax13.text(0.05,0.80,r'\boldmath$%3.2f < P_{hT} < %3.2f ~ {\rm GeV}$'%(tabs[2]['PhTmin'][0],tabs[2]['PhTmax'][0]) ,transform=ax13.transAxes, size=25)
    if 'M' in tabs[2]:
        ax13.text(0.05,0.70,r'\boldmath$M_h = %3.2f ~ {\rm GeV}$'%(tabs[2]['M'][0]) ,transform=ax13.transAxes, size=25)
    else:
        ax13.text(0.05,0.70,r'\boldmath$%3.2f < M_h < %3.2f ~ {\rm GeV}$'%(tabs[2]['Mmin'][0],tabs[2]['Mmax'][0]) ,transform=ax13.transAxes, size=25)

    ax13.text(0.05,0.60,r'\boldmath$\sqrt{s} = %d ~ {\rm GeV}$'%(tabs[2]['RS'][0]) ,transform=ax13.transAxes, size=25)
    fs = 30

    handles,labels = [], []
    handles.append(hand['xspace'])
    handles.append(hand['mellin'])
    handles.append(hand['NN'])
    labels.append(r'\textrm{\textbf{Exact}}')
    labels.append(r'\textrm{\textbf{Mellin+interp.}}')
    labels.append(r'\textrm{\textbf{TensorFlow+interp.}}')
    ax33.legend(handles,labels,loc='lower left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.14)
    filename='gallery/test_NN_RS%d'%RS
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()



#--plot theory results for actual data
def load_data():

    conf['datasets'] = {}
    data = {}

    conf['datasets']['dihadron']={}
    conf['datasets']['dihadron']['filters']=[]
    #conf['datasets']['dihadron']['filters'].append("Q2>%f"%Q2cut)
    conf['datasets']['dihadron']['xlsx']={}
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['dihadron']['xlsx'][4000]='dihadron/expdata/4000.xlsx'
    conf['datasets']['dihadron']['xlsx'][4001]='dihadron/expdata/4001.xlsx'
    conf['datasets']['dihadron']['xlsx'][4010]='dihadron/expdata/4010.xlsx'
    conf['datasets']['dihadron']['xlsx'][4011]='dihadron/expdata/4011.xlsx'
    conf['datasets']['dihadron']['xlsx'][4020]='dihadron/expdata/4020.xlsx'
    ##------------------------------------------------------------------------------------------------------------------
    conf['datasets']['dihadron']['xlsx'][4100]='dihadron/expdata/4100.xlsx'
    conf['datasets']['dihadron']['xlsx'][4101]='dihadron/expdata/4101.xlsx'
    conf['datasets']['dihadron']['xlsx'][4110]='dihadron/expdata/4110.xlsx'
    conf['datasets']['dihadron']['xlsx'][4111]='dihadron/expdata/4111.xlsx'
    conf['datasets']['dihadron']['xlsx'][4120]='dihadron/expdata/4120.xlsx'
    ##------------------------------------------------------------------------------------------------------------------
    conf['datasets']['dihadron']['xlsx'][4200]='dihadron/expdata/4200.xlsx'
    conf['datasets']['dihadron']['xlsx'][4201]='dihadron/expdata/4201.xlsx'
    conf['datasets']['dihadron']['xlsx'][4210]='dihadron/expdata/4210.xlsx'
    conf['datasets']['dihadron']['xlsx'][4211]='dihadron/expdata/4211.xlsx'
    conf['datasets']['dihadron']['xlsx'][4220]='dihadron/expdata/4220.xlsx'
    ##------------------------------------------------------------------------------------------------------------------
    conf['datasets']['dihadron']['xlsx'][4300]='dihadron/expdata/4300.xlsx'
    conf['datasets']['dihadron']['xlsx'][4301]='dihadron/expdata/4301.xlsx'
    conf['datasets']['dihadron']['xlsx'][4310]='dihadron/expdata/4310.xlsx'
    conf['datasets']['dihadron']['xlsx'][4311]='dihadron/expdata/4311.xlsx'
    conf['datasets']['dihadron']['xlsx'][4320]='dihadron/expdata/4320.xlsx'
    ##------------------------------------------------------------------------------------------------------------------
    conf['dihadron tabs'] = READER().load_data_sets('dihadron')  
 
    data = conf['dihadron tabs'].copy()

    return data

def gen_data(wdir,data,ng):

    conf['aux'] = AUX()

    core = CORE()

    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    #--take single replica
    par = replicas[0]

    parman.set_new_params(par,initial=True)

    MELL = MELLIN()
    thy = {}
    #--get Mellin space result
    t1 = time.time()
    for task in MELL.tasks: MELL.process_request(task)
    for task in MELL.tasks: MELL.update(task)
    t2 = time.time()
    t  = t2 - t1
    print('Time to setup Mellin grids: %3.2f'%t)
    thy['mellin'] = {}
    t1 = time.time()
    for idx in conf['dihadron tabs']:
        thy['mellin'][idx],_,_ = MELL.get_asym(idx)
    t2 = time.time()
    t  = t2 - t1
    print('Time to interpolate Mellin results: %3.2f'%t)

    THY = THEORY(ng=ng,predict=False)
    thy['xspace'] = {}
    for idx in data:
        thy['xspace'][idx],_,_ = THY.get_asym(idx)
   
    #--save data for comparison between different values of ng
    filename = 'data/STAR-ng=%d.dat'%(ng)
    checkdir('data')
    save(thy,filename)
    print('saving data to %s'%filename)


def plot_star_RS200(thy,ng,angle=0.3):
    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)

    data = conf['dihadron tabs']

    hand = {}

    #--plot z and Mh
    for idx in data:

        if 'max_open_angle' not in data[idx]: continue
        max_angle = data[idx]['max_open_angle'][0]
        if max_angle != angle: continue

        binned = data[idx]['binned'][0]

        if binned == 'M':   ax = ax11
        if binned == 'PhT': ax = ax12
        if binned == 'eta': ax = ax13

        eta = data[idx]['eta']

        if binned != 'eta' and eta[0] < 0: color = 'darkblue'
        if binned != 'eta' and eta[0] > 0: color = 'firebrick'
        if binned == 'eta':                color = 'firebrick'

        M   = data[idx]['M']
        pT  = data[idx]['PhT']
        eta = data[idx]['eta']
        value = data[idx]['value']
        alpha = data[idx]['stat_u']        

        if binned=='M':
            hand[idx]    = ax.errorbar(M  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
        if binned=='PhT':
            hand[idx]    = ax.errorbar(pT ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
        if binned=='eta':
            hand[idx]    = ax.errorbar(eta,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)

        if idx not in thy['mellin']: continue
        if binned=='M':
            hand['thy'] ,= ax.plot     (M  ,thy['xspace'][idx],        color=color)
            hand['thy'] ,= ax.plot     (M  ,thy['mellin'][idx],        color=color,ls=':')
        if binned=='PhT':
            hand['thy'] ,= ax.plot     (pT ,thy['xspace'][idx],        color=color)
            hand['thy'] ,= ax.plot     (pT ,thy['mellin'][idx],        color=color,ls=':')
        if binned=='eta':
            hand['thy'] ,= ax.plot     (eta,thy['xspace'][idx],        color=color)
            hand['thy'] ,= ax.plot     (eta,thy['mellin'][idx],        color=color,ls=':')


    ax11.set_xlim(0.3,1.3)
    ax11.set_xticks([0.4,0.6,0.8,1.0,1.2])
    minorLocator = MultipleLocator(0.1)
    ax11.xaxis.set_minor_locator(minorLocator)

    ax12.set_xlim(3,11)
    ax12.set_xticks([4,6,8,10])
    minorLocator = MultipleLocator(1)
    ax12.xaxis.set_minor_locator(minorLocator)


    ax13.set_xlim(-0.9,0.9)
    ax13.set_xticks([-0.8,-0.4,0,0.4,0.8])
    minorLocator = MultipleLocator(0.1)
    ax13.xaxis.set_minor_locator(minorLocator)
 
    ax11.set_ylim(-0.1,0.20)
    ax11.set_yticks([-0.05,0,0.05,0.10,0.15])
    minorLocator = MultipleLocator(0.01)
    ax11.yaxis.set_minor_locator(minorLocator)

    ax12.set_ylim(-0.08,0.08)
    ax12.set_yticks([-0.05,0,0.05])
    minorLocator = MultipleLocator(0.01)
    ax12.yaxis.set_minor_locator(minorLocator)

    ax13.set_ylim(-0.01,0.04)
    ax13.set_yticks([0,0.01,0.02,0.03])
    minorLocator = MultipleLocator(0.005)
    ax13.yaxis.set_minor_locator(minorLocator)

    ax11.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=30)
    ax12.set_xlabel(r'\boldmath$P_{hT}~[{\rm GeV}]$',size=30)
    ax13.set_xlabel(r'\boldmath$\eta$',size=30)

    for ax in [ax11,ax12,ax13]:
        ax .tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax .tick_params(axis='both',which='minor',size=4)
        ax .tick_params(axis='both',which='major',size=8)
        ax .axhline(0,0,1,color='black',ls='--',alpha=0.5)


    ax12.text(0.05,0.85,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}' , transform=ax12.transAxes, size=25)
    ax13.text(0.05,0.85,r'\textrm{\textbf{Max opening angle: %s}}'%angle, transform=ax13.transAxes, size=25)
    ax13.text(0.05,0.70,r'\boldmath$\sqrt{s} = 200~{\rm GeV}$' ,      transform=ax13.transAxes, size=25)

    minorLocator = MultipleLocator(0.1)
    #ax11.xaxis.set_minor_locator(minorLocator)

    fs = 30

    handles,labels = [], []
    if angle==0.2:
        handles.append(hand[4000])
        handles.append(hand[4001])
    if angle==0.3:
        handles.append(hand[4100])
        handles.append(hand[4101])
    if angle==0.4:
        handles.append(hand[4200])
        handles.append(hand[4201])
    labels.append(r'\boldmath$\eta<0$')
    labels.append(r'\boldmath$\eta>0$')
    ax11.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.14,left=0.03)
    filename='gallery/STAR/star-RS200-ang%s-ng=%d'%(angle,ng)
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS200_ang07_M(thy,ng):
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax21=py.subplot(nrows,ncols,4)
    ax22=py.subplot(nrows,ncols,5)

    data = conf['dihadron tabs']
    hand = {}

    for idx in data:

        if idx not in [4300,4301]: continue

        binned = data[idx]['binned'][0]
        bins   = np.unique(data[idx]['bin'])

        if data[idx]['etamax'][0] >= 0: color = 'firebrick'
        if data[idx]['etamax'][0] <= 0: color = 'darkblue'
        for i in bins:
            i = int(i)
            if i==1: ax = ax11
            if i==2: ax = ax12
            if i==3: ax = ax13
            if i==4: ax = ax21
            if i==5: ax = ax22
            ind = np.where(data[idx]['bin'] == i)
            M   = data[idx]['M'][ind]
            pT  = data[idx]['PhT'][ind]
            #eta = data[idx]['eta'][ind]
            value = data[idx]['value'][ind]
            stat_u = data[idx]['stat_u'][ind]
            syst_u = data[idx]['syst_u'][ind]
            alpha = np.sqrt(stat_u**2 + syst_u**2)

            if idx==4301: M += 0.05

            hand[idx]    = ax.errorbar(M  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
            ax.text(0.05,0.85,r'\boldmath$\langle P_{hT} \rangle = %3.2f~{\rm GeV}$'%pT[0] , transform=ax.transAxes, size=25)
            if idx not in thy['mellin']: continue
            hand['thy'] ,= ax.plot    (M,thy['xspace'][idx][ind]     ,color=color)
            hand['thy'] ,= ax.plot    (M,thy['mellin'][idx][ind]     ,color=color,ls=':')


    for ax in [ax11,ax12,ax13,ax21,ax22]:

        ax.set_xlim(0.2,2.4)
        ax.set_xticks([0.5,1.0,1.5,2])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.set_ylim(-0.025,0.06)
        ax.set_yticks([-0.02,0,0.02,0.04])
        minorLocator = MultipleLocator(0.005)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
 
    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom = False)

    for ax in [ax12,ax13,ax22]:
        ax.tick_params(labelleft = False)

    for ax in [ax13,ax21,ax22]:
        ax.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=30)

    ax22.text(1.10,0.60,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}' , transform=ax22.transAxes, size=30)
    ax22.text(1.10,0.50,r'\boldmath$\sqrt{s} = 200~{\rm GeV}$' ,      transform=ax22.transAxes, size=30)

    fs = 30

    handles,labels = [], []
    handles.append(hand[4300])
    handles.append(hand[4301])
    labels.append(r'\boldmath$\eta<0$')
    labels.append(r'\boldmath$\eta>0$')
    ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.02,left=0.03)
    filename='gallery/STAR/star-RS200-ang07-M-ng=%d'%(ng)
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS200_ang07_PhT(thy,ng):
    nrows,ncols=2,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)
    ax21=py.subplot(nrows,ncols,4)
    ax22=py.subplot(nrows,ncols,5)

    data = conf['dihadron tabs']
    hand = {}

    for idx in data:

        if idx not in [4310,4311]: continue

        binned = data[idx]['binned'][0]
        bins   = np.unique(data[idx]['bin'])

        if data[idx]['etamax'][0] >= 0: color = 'firebrick'
        if data[idx]['etamax'][0] <= 0: color = 'darkblue'
        for i in bins:
            i = int(i)
            if i==1: ax = ax11
            if i==2: ax = ax12
            if i==3: ax = ax13
            if i==4: ax = ax21
            if i==5: ax = ax22
            ind = np.where(data[idx]['bin'] == i)
            M   = data[idx]['M'][ind]
            pT  = data[idx]['PhT'][ind]
            #eta = data[idx]['eta'][ind]
            value = data[idx]['value'][ind]
            stat_u = data[idx]['stat_u'][ind]
            syst_u = data[idx]['syst_u'][ind]
            alpha = np.sqrt(stat_u**2 + syst_u**2)

            if idx==4311: pT += 0.05

            hand[idx]    = ax.errorbar(pT  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
            ax.text(0.05,0.85,r'\boldmath$\langle M_{h} \rangle = %3.2f~{\rm GeV}$'%M[0] , transform=ax.transAxes, size=25)
            if idx not in thy['mellin']: continue
            hand['thy'] ,= ax.plot    (pT,thy['xspace'][idx][ind]     ,color=color)
            hand['thy'] ,= ax.plot    (pT,thy['mellin'][idx][ind]     ,color=color,ls=':')


    for ax in [ax11,ax12,ax13,ax21,ax22]:

        ax.set_xlim(2,11)
        ax.set_xticks([4,6,8,10])
        minorLocator = MultipleLocator(0.5)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.set_ylim(-0.025,0.06)
        ax.set_yticks([-0.02,0,0.02,0.04])
        minorLocator = MultipleLocator(0.005)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
 
    for ax in [ax11,ax12]:
        ax.tick_params(labelbottom = False)

    for ax in [ax12,ax13,ax22]:
        ax.tick_params(labelleft = False)

    for ax in [ax13,ax21,ax22]:
        ax.set_xlabel(r'\boldmath$P_{hT}~[{\rm GeV}]$',size=30)

    ax22.text(1.10,0.60,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}' , transform=ax22.transAxes, size=30)
    ax22.text(1.10,0.50,r'\boldmath$\sqrt{s} = 200~{\rm GeV}$' ,      transform=ax22.transAxes, size=30)

    fs = 30

    handles,labels = [], []
    handles.append(hand[4310])
    handles.append(hand[4311])
    labels.append(r'\boldmath$\eta<0$')
    labels.append(r'\boldmath$\eta>0$')
    ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.02,left=0.03)
    filename='gallery/STAR/star-RS200-ang07-PhT-ng=%d'%(ng)
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS200_ang07_eta(thy,ng):
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11=py.subplot(nrows,ncols,1)

    data = conf['dihadron tabs']
    hand = {}

    for idx in data:

        if idx not in [4320]: continue
        ax = ax11

        color = 'firebrick'
        M       = data[idx]['M']
        pT      = data[idx]['PhT']
        eta     = data[idx]['eta']
        value   = data[idx]['value']
        stat_u  = data[idx]['stat_u']
        syst_u  = data[idx]['syst_u']
        alpha = np.sqrt(stat_u**2 + syst_u**2)

        hand[idx]    = ax.errorbar(eta,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
        if idx not in thy['mellin']: continue
        hand['thy'] ,= ax.plot    (eta,thy['xspace'][idx]        ,color=color)
        hand['thy'] ,= ax.plot    (eta,thy['mellin'][idx]        ,color=color,ls=':')


    for ax in [ax11]:

        ax.set_xlim(-1,1)
        ax.set_xticks([-0.5,0,0.5])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

        ax.set_ylim(-0.005,0.04)
        ax.set_yticks([0,0.01,0.02,0.03])
        minorLocator = MultipleLocator(0.002)
        ax.yaxis.set_minor_locator(minorLocator)

        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax.tick_params(axis='both',which='minor',size=4)
        ax.tick_params(axis='both',which='major',size=8)
        ax.axhline(0,0,1,color='black',ls='--',alpha=0.5)
        ax.set_xlabel(r'\boldmath$\eta$',size=40)
        ax.xaxis.set_label_coords(0.95,-0.01)

    ax11.text(0.05,0.85,r'\textrm{\textbf{STAR \boldmath$A_{UT}$}}'         ,transform=ax11.transAxes, size=25)
    ax11.text(0.05,0.75,r'\boldmath$\sqrt{s} = 200~{\rm GeV}$'              ,transform=ax11.transAxes, size=20)
    ax11.text(0.05,0.65,r'\boldmath$\langle P_{hT} \rangle = 5~{\rm GeV}$' ,transform=ax11.transAxes, size=20)
    ax11.text(0.05,0.55,r'\boldmath$\langle M_h \rangle = 1 ~ {\rm GeV}$'   ,transform=ax11.transAxes, size=20)

    #handles,labels = [], []
    #handles.append(hand[5020])
    #labels.append(r'\textrm{\textbf{STAR}}')
    #ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)

    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.00,left=0.10)
    filename='gallery/STAR/star-RS200-ang07-eta-ng=%d'%(ng)
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()









 
if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-t',      '--task'   ,type=int,   default=0)
    ap.add_argument('-xsec',   '--xsec'   ,type=str,   default='UT')
    ap.add_argument('-channel','--channel',type=str,   default='QQ,QQ')
    ap.add_argument('-RS' ,    '--nrg',    type=float, default=200)
    ap.add_argument('-eta',    '--eta',    type=float, default=0.1)
    ap.add_argument('-PhT',    '--PhT',    type=float, default=5)
    ap.add_argument('-name',   '--name',   type=str,   default=0)
    args = ap.parse_args()



    task    = args.task
    xsec    = args.xsec
    channel = args.channel
    RS      = args.nrg 
    eta     = args.eta
    PhT     = args.PhT
    name    = args.name

    #--plot against x-space calculation
    if task==0:

        #--wdir to load DiFFs and TPDFs from
        wdir = 'data/step07'

        RS = 200
        ng = 20
        gen_NN = False
        test_NN(wdir,RS,ng=ng,gen_NN=gen_NN)

    #--plot against x-space calculation (real data)
    if task==1:

        #--wdir to load DiFFs and TPDFs from
        wdir = 'data/step07'

        ng = 20
        data = load_data()
        #gen_data(wdir,data,ng)

        filename = 'data/STAR-ng=%d.dat'%(ng)
        thy = load(filename)

        checkdir('gallery/STAR')
        plot_star_RS200          (thy,ng,angle=0.2)
        plot_star_RS200          (thy,ng,angle=0.3)
        plot_star_RS200          (thy,ng,angle=0.4)
        plot_star_RS200_ang07_M  (thy,ng)
        plot_star_RS200_ang07_PhT(thy,ng)
        plot_star_RS200_ang07_eta(thy,ng)





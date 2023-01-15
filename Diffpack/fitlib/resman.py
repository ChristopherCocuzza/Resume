import sys, os
import time
import numpy as np

#--from qcdlib
from qcdlib import diff, tdiff, tpdf
from qcdlib import aux, eweak, alphaS, mellin

#--from fitlib
from fitlib.parman import PARMAN

#--from tools
from tools.tools    import checkdir
from tools.config   import conf,load_config,options
from tools.parallel import PARALLEL


#--from obslib
import obslib.dihadron.mellin
import obslib.dihadron.theory
import obslib.dihadron.residuals
import obslib.dihadron.reader

import obslib.moments.residuals
import obslib.moments.reader

import lhapdf

class RESMAN:

    def __init__(self,nworkers=2,parallel=True,datasets=True,load_lhapdf=True):

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






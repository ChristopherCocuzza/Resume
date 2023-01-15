import sys
from numpy.random import choice, randn
import numpy as np
from .config import conf
from .tools import save
import copy
import time

class _RESIDUALS:

    #--basic routimes to prepare the data sets
  
    def percent_to_absolute(self):

        for k in self.tabs:
            ucorr = [x for x in self.tabs[k] if '_u' in x and '%' in x]
            corr  = [x for x in self.tabs[k] if '_c' in x and '%' in x]

            if  len(ucorr)!=0:
                for name in ucorr:
                    mod_name=name.replace('%','')
                    self.tabs[k][mod_name]=self.tabs[k]['value'] * self.tabs[k][name]/100.0

            if  len(corr)!=0:
                for name in corr:
                    mod_name=name.replace('%','')
                    self.tabs[k][mod_name]=self.tabs[k]['value'] * self.tabs[k][name]/100.0
  
    def add_columns(self):

        for k in self.tabs:
            npts=len(self.tabs[k]['value'])
            self.tabs[k]['thy']=np.zeros(npts)
            self.tabs[k]['N']=np.zeros(npts)
            self.tabs[k]['residuals']=np.zeros(npts)
            self.tabs[k]['r-residuals']=np.zeros(npts)
            self.tabs[k]['Shift']=np.zeros(npts)
  
    def get_alpha(self):

        for k in self.tabs:
            npts=len(self.tabs[k]['value'])
            alpha2=np.zeros(npts) 
            ucorr = [x for x in self.tabs[k] if '_u' in x and '%' not in x]
            for kk in ucorr: alpha2+=self.tabs[k][kk]**2
            self.tabs[k]['alpha']=alpha2**0.5
  
    def retrieve_norm_uncertainty(self):
  
        for k in self.tabs:

            norm  = [_ for _ in self.tabs[k] if '_c' in _ and 'norm' in _ and '%' not in _]

            if  len(norm)>1:
                msg='ERR: more than one normalization found at %s %d'%(self.reaction,k)
                raise ValueError(msg)

            elif len(norm)==1:
  
                if k not in conf['datasets'][self.reaction]['norm']:
                    print('WARNING: %s %s has a normalization uncertainty but no fitted normalization'%(self.reaction,k))

                else:
                    print('%s has norm uncertainty'%k)
                    for i in range(len(self.tabs[k]['value'])):
                        if self.tabs[k]['value'][0]!=0:
                            dN=self.tabs[k][norm[0]][i]/(self.tabs[k]['value'][i])
                            break
                    conf['datasets'][self.reaction]['norm'][k]['dN']=dN

            elif len(norm)==0:
                if k not in conf['datasets'][self.reaction]['norm']: pass
                else:
                    print('Cannot read normalization uncertainty for data set ',self.reaction,k)
                    sys.exit()
  
    def setup_rparams(self):

        if  'rparams' not in conf: 
            conf['rparams']={}

        if  self.reaction not in conf['rparams']: 
            conf['rparams'][self.reaction]={}

        for k in self.tabs:

            if  k not in conf['rparams'][self.reaction]:
                conf['rparams'][self.reaction][k]={}

            corr = [x for x in self.tabs[k] if '_c' in x and '%' not in x and 'norm' not in x]
            for c in corr: 
                conf['rparams'][self.reaction][k][c]={'value':0.0,'registered':False}
  
    def setup_covmat(self):
        self.covmat={}
        for k in self.tabs:
            npts=len(self.tabs[k]['value'])
            self.covmat[k]=np.zeros((npts,npts))
            np.fill_diagonal(self.covmat[k],self.tabs[k]['alpha']**2) #uncorr uncertainty
            corr = [x for x in self.tabs[k] if '_c' in x and '%' not in x]
            for c in corr:
                for i in range(npts):
                    for j in range(npts):
                        self.covmat[k][i,j]+=self.tabs[k][c][i]*self.tabs[k][c][j]


    def resample(self):
        self.tabs=copy.deepcopy(self.original)
        for k in self.tabs:
            SamplingMatrix=np.asmatrix(np.linalg.cholesky(self.covmat[k]))
            npts=len(self.tabs[k]['value'])
            normrands=np.asmatrix(randn(npts)).transpose()
            deviates=np.squeeze(np.array(SamplingMatrix*normrands))
            self.tabs[k]['value']+=deviates  

    def resample_old(self):

        self.tabs=copy.deepcopy(self.original)

        for k in self.tabs:
            npts=len(self.tabs[k]['value'])
            self.tabs[k]['value']+=randn(npts)*self.tabs[k]['alpha']
  
    #--master setup
  
    def setup(self):
        self.percent_to_absolute()
        self.add_columns()
        self.get_alpha()
        self.retrieve_norm_uncertainty()
        self.setup_rparams()
        self.original=copy.deepcopy(self.tabs)
        if 'bootstrap' in conf and conf['bootstrap']:
            self.setup_covmat()
            self.resample()
  
    #--residuals
  
    def _get_residuals(self,k):

        npts=len(self.tabs[k]['value'])
        exp=self.tabs[k]['value']

        if  k in conf['datasets'][self.reaction]['norm']:
            norm=conf['datasets'][self.reaction]['norm'][k]['value']
        else:
            norm=1.0

        thy=self.tabs[k]['thy']/norm
        alpha=self.tabs[k]['alpha']
        corr = [x for x in self.tabs[k] if '_c' in x and '%' not in x and 'norm' not in x]
        N=np.ones(exp.size)
        ncorr=len(corr)

        if  ncorr==0:

            self.tabs[k]['N']=N
            self.tabs[k]['residuals']=(exp-thy)/alpha
            self.tabs[k]['shift']=np.zeros(exp.size)
            self.tabs[k]['prediction']=thy

        else:

            beta=[self.tabs[k][c] * (thy/(exp+1e-100)) for c in corr]
            A=np.diag(np.diag(np.ones((ncorr,ncorr)))) + np.einsum('ki,li,i->kl',beta,beta,1/alpha**2)
            B=np.einsum('ki,i,i->k',beta,exp-thy,1/alpha**2)
            try:
                r=np.einsum('kl,l->k',np.linalg.inv(A),B)
            except:
                r=np.zeros(len(beta))
            shift=np.einsum('k,ki->i',r,beta)
            for i in range(ncorr):
                conf['rparams'][self.reaction][k][corr[i]]['value']=r[i]
            self.tabs[k]['N']=N
            self.tabs[k]['residuals']=(exp-shift-thy)/alpha
            self.tabs[k]['shift']=shift
            self.tabs[k]['prediction']=shift+thy

        return self.tabs[k]['residuals']
  
    def _get_rres(self,k):
        rres=[]
        rparams=conf['rparams'][self.reaction][k]
        for c in rparams:
            rres.append(rparams[c]['value'])
        return np.array(rres)
  
    def _get_nres(self,k):

        if  k not in conf['datasets'][self.reaction]['norm']:
            return 0

        elif  conf['datasets'][self.reaction]['norm'][k]['fixed']:
            return 0

        else:
            norm=conf['datasets'][self.reaction]['norm'][k]
            return (norm['value']-1)/norm['dN']
  
    def get_residuals(self):

        res,rres,nres=[],[],[]

        self.get_theory()

        for k in self.tabs: 
            res=np.append(res,self._get_residuals(k))
            rres=np.append(rres,self._get_rres(k))
            nres=np.append(nres,self._get_nres(k))
        return res,rres,nres
  
    #--mis functions 
 
    def get_npts(self):
        npts=0
        for k in self.tabs: 
            npts+=len(self.tabs[k]['value'])
        return npts

    def get_chi2(self):

      data={self.reaction:{}}

      for idx in self.tabs:
          if len(self.tabs[idx])==0: continue 
          res=self.tabs[idx]['residuals']
          npts=res.size
          chi2=np.sum(res**2)
          data[self.reaction][idx]={'npts':npts,'chi2':chi2}

      return data

    #--for parallelization
  
    def get_state(self):
        return (self.tabs,)

    def set_state(self,state):
        self.tabs=state[0]











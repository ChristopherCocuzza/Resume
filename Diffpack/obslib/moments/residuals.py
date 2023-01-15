#!/usr/bin/env python
import sys
import os
import numpy as np
import time
from tools.residuals import _RESIDUALS
from obslib.moments.reader import READER
from qcdlib.aux import AUX
from tools.config import conf


class RESIDUALS(_RESIDUALS):

    def __init__(self):
        self.reaction = 'moments'
        self.tabs = conf['moments tabs']
        self.tpdf = conf['tpdf-mom']
        self.setup()

    def get_theory(self):

        for idx in self.tabs:

            obs = self.tabs[idx]['obs'][0].strip()
            Q2  = self.tabs[idx]['Q2'][0]
            self.tpdf.evolve([Q2])
            if obs == 'gT(u-d)':
                u  = np.real(self.tpdf.storage[Q2]['u'][0])
                ub = np.real(self.tpdf.storage[Q2]['ub'][0])
                d  = np.real(self.tpdf.storage[Q2]['d'][0])
                db = np.real(self.tpdf.storage[Q2]['db'][0])
                thy = (u-ub)-(d-db)

            self.tabs[idx]['thy'] = thy

    def gen_report(self,verb=1,level=1):
        """
        verb = 0: Do not print on screen. Only return list of strings
        verv = 1: print on screen the report
        level= 0: only the total chi2s
        level= 1: include point by point 
        """
          
        L=[]
  
        if len(self.tabs.keys())!=0:
            L.append('reaction: moments')
            for f in conf['datasets']['moments']['filters']:
                L.append('filters: %s'%f)
  
            L.append('%7s %20s %5s %10s %10s %10s %10s'%('idx','col','npts','chi2','chi2/npts','rchi2','nchi2'))
            for k in self.tabs:
                if len(self.tabs[k])==0: continue 
                res=self.tabs[k]['residuals']
  
                rres=[]
                for c in conf['rparams']['moments'][k]:
                    rres.append(conf['rparams']['moments'][k][c]['value'])
                rres=np.array(rres)
  
                if k in conf['datasets']['moments']['norm']:
                    norm=conf['datasets']['moments']['norm'][k]
                    nres=(norm['value']-1)/norm['dN']
                else:
                    nres=0
  
                chi2=np.sum(res**2)
                rchi2=np.sum(rres**2)
                nchi2=nres**2
                col=self.tabs[k]['col'][0].split()[0]
                npts=res.size
                L.append('%7d %20s %5d %10.2f %10.2f %10.2f %10.2f'%(k,col,npts,chi2,chi2/npts,rchi2,nchi2))
  
            if level==1:
              L.append('-'*100)  
              for k in self.tabs:
                  if len(self.tabs[k]['value'])==0: continue 
                  if k in conf['datasets']['SU23']['norm']:
                      norm=conf['datasets']['SU23']['norm'][k]
                      nres=(norm['value']-1)/norm['dN']
                      norm=norm['value']
                  else:
                      norm=1.0
                      nres=0
                  for i in range(len(self.tabs[k]['value'])):
                      x     = self.tabs[k]['X'][i]
                      Q2    = self.tabs[k]['Q2'][i]
                      res   = self.tabs[k]['residuals'][i]
                      thy   = self.tabs[k]['thy'][i]
                      exp   = self.tabs[k]['value'][i]
                      alpha = self.tabs[k]['alpha'][i]
                      rres  = self.tabs[k]['r-residuals'][i]
                      col   = self.tabs[k]['col'][i]
                      shift = self.tabs[k]['shift'][i]
                      if 'target' in self.tabs[k]: tar   = self.tabs[k]['target'][i]
                      else: tar = '-'
                      msg='%d col=%7s, tar=%5s, x=%10.3e, Q2=%10.3e, exp=%10.3e, alpha=%10.3e, thy=%10.3e, shift=%10.3e, chi2=%10.3e, res=%10.3e, norm=%10.3e, '
                      L.append(msg%(k,col,tar,x,Q2,exp,alpha,thy,shift,res**2,res,norm))
  
        if verb==0:
            return L
        elif verb==1:
            for l in L: print(l)
            return L








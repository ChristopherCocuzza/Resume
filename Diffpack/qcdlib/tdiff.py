#!/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
from tools.tools import checkdir
import pylab as py
from scipy.special import gamma
import numpy as np
import qcdlib.alphaS as alphaS
from qcdlib.aux import AUX
from qcdlib.dglap import DGLAP
from qcdlib.kernels import KERNELS
from qcdlib.mellin import MELLIN
from tools.config import conf
import lhapdf
from scipy.interpolate import RectBivariateSpline 
from scipy.interpolate import interp1d
import time

class TDIFF:

    def __init__(self, mellin=None):

        self.spl='trans'
        self.Q20=conf['Q20']
        self.mc2=conf['aux'].mc2
        self.mb2=conf['aux'].mb2
        if mellin==None:
            self.kernel=KERNELS(conf['mellin'],self.spl)
            if 'mode' in conf: mode=conf['mode']
            else: mode='truncated'
            self.dglap=DGLAP(conf['mellin'],conf['alphaS'],self.kernel,mode,conf['order'])
            self.mellin=conf['mellin']
        else:
            self.kernel=KERNELS(mellin,self.spl)
            if 'mode' in conf: mode=conf['mode']
            else: mode='truncated'
            self.dglap=DGLAP(mellin,conf['alphaS'],self.kernel,mode,conf['order'])
            self.mellin=mellin

        if 'H1 M0' in conf: self.M0 = conf['H1 M0']
        else:
            print('Must specify H1 M0 grid.  Exiting...')      
            sys.exit()

        if 'tdiffpippim_choice' in conf:
            self.choice = conf['tdiffpippim_choice']
        else:
            self.choice = 'threeshapes'

        self.flavs = ['u']

        self.set_params()
        self.setup()

    def set_params(self):
        """
        f(x) = N1 * z**a1 * (1-z)**b1 + N2 * z**a2 * (1-z)**b2 + N3 * z**a3 * (1-z)**b3
        """
        self.params = {}

        M0 = self.M0
        self.FLAV = []

        #--three shapes
        if self.choice=='threeshapes':
            for M in M0['u']: self.params['u %s'%M] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            for M in M0['u']: self.FLAV.append('u %s'%M)
            
            self.PAR  = ['N1','a1','b1','N2','a2','b2','N3','a3','b3']

        #--one shape ++
        if self.choice=='oneshape++':
            for M in M0['u']: self.params['u %s'%M] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            for M in M0['u']: self.FLAV.append('u %s'%M)
            
            self.PAR  = ['N1','a1','b1','c1','d1']

    def setup(self):

        M0 = self.M0

        moms = {}

        u = np.array([self.get_moments('u %s'%M) for M in M0['u']])

        U =  interp1d(M0['u'], u, axis=0, bounds_error=False, fill_value='extrapolate', kind='cubic')

        moms['um']  =  lambda M:  2*U(M)
        moms['dm']  =  lambda M: -2*U(M)

        self.moms0 = moms
        self.get_BC(moms)

        #--we will store all (Q2,M) values that have been precalc
        self.storage={}
        #--we will store all (Q2,M,z) values that have been calculated
        self.storage2={}
        self.storage2['u'] = {}
        self.storage2['s'] = {}
        self.storage2['c'] = {}
        self.storage2['g'] = {}

    def beta(self,a,b):
        return gamma(a)*gamma(b)/gamma(a+b)

    def get_moments(self,flav,N=None):
        """
        if N==None: then parametrization is to be use to compute moments along mellin contour
        else the Nth moment is returned
        """
        if N==None: N=self.mellin.N
        #--three shapes
        if self.choice=='threeshapes':
            M1,a1,b1,M2,a2,b2,M3,a3,b3=self.params[flav]

            norm1= self.beta(1+a1,b1+1)
            norm2= self.beta(1+a2,b2+1)
            norm3= self.beta(1+a3,b3+1)
               
            mom1 = M1*self.beta(N+a1,b1+1)
            mom2 = M2*self.beta(N+a2,b2+1)
            mom3 = M3*self.beta(N+a3,b3+1)
            mom  = mom1/norm1 + mom2/norm2 + mom3/norm3

        #--one shape ++
        if self.choice=='oneshape++':
            M,a,b,c,d=self.params[flav]
            norm=self.beta(1+a,b+1)+c*self.beta(1+a+0.5,b+1)+d*self.beta(1+a+1.0,b+1)
            mom=self.beta(N+a,b+1)+c*self.beta(N+a+0.5,b+1)+d*self.beta(N+a+1.0,b+1)

        return mom

    def _get_BC(self,g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_):
        N=self.mellin.N

        # flav composition
        vm,vp={},{}
        vm[35]= bm + cm + dm + sm - 5*tm + um
        vm[24]= -4*bm + cm + dm + sm + um
        vm[15]= -3*cm + dm + sm + um
        vm[8] = dm - 2*sp + 2*(-sm + sp) + um
        vm[3] = -dm + um
        vm[0] = np.zeros(N.size,dtype=complex)
        vp[0] = np.zeros(N.size,dtype=complex)
        vp[3] = -dp + up
        vp[8] = dp - 2*sp + up
        vp[15]= -3*cp + dp + sp + up
        vp[24]= -4*bp + cp + dp + sp + up
        vp[35]= bp + cp + dp + sp - 5*tp + up
        qs    = bp + cp + dp + sp + tp + up
        qv    = bm + cm + dm + sm + tm + um
        q     = np.zeros((2,N.size),dtype=complex)
        q[0]=np.copy(qs)
        q[1]=np.copy(g)

        BC={}
        BC['vm']=vm
        BC['vp']=vp
        BC['qv']=qv
        BC['q'] =q
        BC['um_'] = um_
        BC['dm_'] = dm_
        return BC

    def get_state(self):
        return (self.BC3,self.BC4,self.BC5)

    def set_state(self,state):
        self.BC3, self.BC4, self.BC5 = state[:]
        self.storage = {}

    def get_BC(self,moms):

        N=self.mellin.N
        zero=np.zeros(N.size,dtype=complex)

        M0 = self.M0
        self.BC3 = {}
        BC4, BC5 = {},{}
        self.BC4, self.BC5 = {}, {}

        for M in M0['u']:
            ###############################################
            # BC for Nf=3
            g   = zero
            up  = zero 
            um  = moms['um'](M)
            dp  = zero 
            dm  = moms['dm'](M)
            sp  = zero 
            sm  = zero
            cp  = zero
            cm  = zero
            bp  = zero
            bm  = zero
            self.BC3[M]=self._get_BC(g,up,um,dp,dm,sp,sm,zero,zero,zero,zero,zero,zero,um,dm)

            ###############################################
            # BC for Nf=4
            BC4[M]=self.dglap.evolve(self.BC3[M],self.Q20,self.mc2,3)
            #--add moment to charm quark
            g =BC4[M]['g']
            up=BC4[M]['up']
            dp=BC4[M]['dp']
            sp=BC4[M]['sp']
            cp=BC4[M]['cp']
            bp=BC4[M]['bp']
            tp=BC4[M]['tp']
            um=BC4[M]['um']
            dm=BC4[M]['dm']
            sm=BC4[M]['sm']
            cm=BC4[M]['cm']
            bm=BC4[M]['bm']
            tm=BC4[M]['tm']
            um_=BC4[M]['um_']
            dm_=BC4[M]['dm_']
            self.BC4[M]=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_)

            ###############################################
            # BC for Nf=5
            BC5[M]=self.dglap.evolve(self.BC4[M],self.mc2,self.mb2,4)
            g =BC5[M]['g']
            up=BC5[M]['up']
            dp=BC5[M]['dp']
            sp=BC5[M]['sp']
            cp=BC5[M]['cp']
            bp=BC5[M]['bp']
            tp=BC5[M]['tp']
            um=BC5[M]['um']
            dm=BC5[M]['dm']
            sm=BC5[M]['sm']
            cm=BC5[M]['cm']
            bm=BC5[M]['bm']
            tm=BC5[M]['tm']
            um_=BC5[M]['um_']
            dm_=BC5[M]['dm_']
            self.BC5[M]=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_)

    #--evolve and interpolate in Mellin space
    def evolve(self,Q2,Mh):

      M0 = self.M0

      for j in range(len(Q2)):

          if Q2[j] not in self.storage:

              self.storage[Q2[j]] = {}
              self.storage[Q2[j]]['real'] = {}
              self.storage[Q2[j]]['imag'] = {}
              real,imag = {},{}
              for M in M0['u']:
                  if self.mb2<Q2[j]:
                      BC = self.dglap.evolve(self.BC5[M],self.mb2,Q2[j],5)
                  elif self.mc2<=Q2[j] and Q2[j]<=self.mb2:
                      BC = self.dglap.evolve(self.BC4[M],self.mc2,Q2[j],4)
                  elif Q2[j]<self.mc2:
                      BC = self.dglap.evolve(self.BC3[M],self.Q20,Q2[j],3)
                  for flav in self.flavs:
                      if flav not in real: real[flav] = []
                      if flav not in imag: imag[flav] = []
                      real[flav].append(np.real(BC[flav]))
                      imag[flav].append(np.imag(BC[flav]))
                        

              for flav in self.flavs:
                  self.storage[Q2[j]]['real'][flav] = interp1d(self.M0['u'],real[flav],axis=0,bounds_error=False,fill_value='extrapolate',kind='cubic')
                  self.storage[Q2[j]]['imag'][flav] = interp1d(self.M0['u'],imag[flav],axis=0,bounds_error=False,fill_value='extrapolate',kind='cubic')


          if Mh[j] not in self.storage[Q2[j]]:
              self.storage[Q2[j]][Mh[j]] = {}
              for flav in self.flavs:
                  self.storage[Q2[j]][Mh[j]][flav] =  self.storage[Q2[j]]['real'][flav](Mh[j]) + 1j * self.storage[Q2[j]]['imag'][flav](Mh[j])

              self.storage[Q2[j]][Mh[j]]['ub'] = -self.storage[Q2[j]][Mh[j]]['u'].copy() 
              self.storage[Q2[j]][Mh[j]]['d']  = -self.storage[Q2[j]][Mh[j]]['u'].copy() 
              self.storage[Q2[j]][Mh[j]]['db'] =  self.storage[Q2[j]][Mh[j]]['u'].copy() 
              self.storage[Q2[j]][Mh[j]]['s']  =  np.zeros(len(conf['mellin'].N),dtype=complex)
              self.storage[Q2[j]][Mh[j]]['sb'] =  np.zeros(len(conf['mellin'].N),dtype=complex)
              self.storage[Q2[j]][Mh[j]]['c']  =  np.zeros(len(conf['mellin'].N),dtype=complex)
              self.storage[Q2[j]][Mh[j]]['cb'] =  np.zeros(len(conf['mellin'].N),dtype=complex)
              self.storage[Q2[j]][Mh[j]]['b']  =  np.zeros(len(conf['mellin'].N),dtype=complex)
              self.storage[Q2[j]][Mh[j]]['bb'] =  np.zeros(len(conf['mellin'].N),dtype=complex)
              self.storage[Q2[j]][Mh[j]]['g']  =  np.zeros(len(conf['mellin'].N),dtype=complex)
 
    def get_H(self,z,M,Q2,flav,evolve=True):
        if evolve: self.evolve(Q2,M)

        #--pi+ pi- symmetry
        if flav   in ['u','db']: flav,sign = 'u',  1.0
        elif flav in ['ub','d']: flav,sign = 'u', -1.0
        elif flav in ['s','sb','c','cb','b','bb','g']:
            return np.zeros(len(Q2))

        for j in range(len(Q2)):
            key = '%s,%s,%s'%(Q2[j],M[j],z[j])
            if key not in self.storage2[flav]:
                self.storage2[flav][key] = self.mellin.invert(z[j],self.storage[Q2[j]][M[j]][flav])
        return  sign*np.array([self.storage2[flav]['%s,%s,%s'%(Q2[j],M[j],z[j])] for j in range(len(Q2))])


    #--|H1| < D1 bound
    def check_positivity(self):

        Q20 = conf['Q20']

        D = conf['diffpippim']

        lenZ = 30
        lenM = 10
        violations = np.zeros(lenZ*lenM)

        Z  = np.linspace(0.20,0.95,lenZ)

        flavs = ['u']
        for z in Z:
            Mmax = np.max(self.M0['u'])
            M = np.linspace(0.28,Mmax,lenM)
            Zgrid, Mgrid = np.meshgrid(Z,M)
            Zgrid = Zgrid.flatten()
            Mgrid = Mgrid.flatten()

            for flav in flavs:
                H1 = self.get_H(Zgrid,Mgrid,Q20*np.ones(len(Zgrid)),flav,evolve=True)
                D1 = D   .get_D(Zgrid,Mgrid,Q20*np.ones(len(Zgrid)),flav,evolve=True)
                plus  = D1 + H1
                minus = D1 - H1
                for i in range(len(Zgrid)):
                    if plus[i]  < 0.0: violations[i] += plus[i]
                    if minus[i] < 0.0: violations[i] += minus[i]

        #--what factor to use here?
        factor = 3
        return violations*factor

    #--put limit on Mmax
    def _check_positivity(self):

        Q20 = conf['Q20']

        D = conf['diffpippim']

        alpha = conf['alpha']
        RS = 10.58

        lenZ = 30
        lenM = 10
        violations = np.zeros(lenZ*lenM)

        Z  = np.linspace(0.25,0.95,lenZ)

        flavs = ['u']
        for z in Z:
            Mmax = alpha*RS/2 * z
            M = np.linspace(0.28,Mmax,lenM)
            Zgrid, Mgrid = np.meshgrid(Z,M)
            Zgrid = Zgrid.flatten()
            Mgrid = Mgrid.flatten()

            for flav in flavs:
                H1 = self.get_H(Zgrid,Mgrid,Q20*np.ones(len(Zgrid)),flav,evolve=True)
                D1 = D   .get_D(Zgrid,Mgrid,Q20*np.ones(len(Zgrid)),flav,evolve=True)
                plus  = D1 + H1
                minus = D1 - H1
                for i in range(len(Zgrid)):
                    if plus[i]  < 0.0: violations[i] += plus[i]
                    if minus[i] < 0.0: violations[i] += minus[i]

        #--what factor to use here?
        factor = 3
        return violations*factor
    
    #--generate report for positivity violations   
    def gen_report(self,verb=1,level=1):
          
        L=[]

        res2 = conf['H1 pos chi2']
 
        L.append('chi2 from H1 positivity: %3.5f'%res2)

        return L


if __name__=='__main__':

  conf['order']='LO'
  conf['Q20'] = 1.00
  conf['aux']=AUX()
  conf['mellin'] = MELLIN()
  conf['alphaS']=alphaS

  TDIFF = TDIFF()
  TDIFF.setup()

  Z = np.linspace(0.2,1.0,50)
  M = 0.8*np.ones(len(Z))
  Q2= 100*np.ones(len(Z))

  t1 = time.time()
  for i in range(275000):
      u = TDIFF.get_H(Z,M,Q2,'u')
  t2 = time.time()



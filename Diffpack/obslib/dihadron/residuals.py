#!/usr/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
from matplotlib.ticker import MultipleLocator
import pylab as py
import numpy as np
from scipy.integrate import cumtrapz, fixed_quad
from obslib.dihadron.reader import READER
from tools.residuals import _RESIDUALS
from tools.config import conf,load_config
from tools.tools import load,save,checkdir,lprint
import time
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
import lhapdf
from obslib.dihadron.aux import AUX

#--SIDIS integration bounds taken from arxiv.org/pdf/2105.08725.pdf
#--questions:
#--should "flag" be added back into integrations?
#--is SIDIS integration correct? (should y actually be integrated over?)

class RESIDUALS(_RESIDUALS):
  
    def __init__(self):

        self.aux = AUX() 
        self.reaction='dihadron'
        self.thy  = conf['dihadron']
        self.tabs=conf['dihadron tabs']
        if 'diffpippim'           in conf: self.diffpippim  = conf['diffpippim']
        if 'tdiffpippim'          in conf: self.tdiffpippim = conf['tdiffpippim']
        if 'tpdf'                 in conf: self.tpdf        = conf['tpdf']
        self.setup()
        self.ratio = {}

        if 'integrate' in conf: self.integrate = conf['integrate']
        else:                   self.integrate = True


        if 'update_unpol' in conf: self.update_unpol = conf['update_unpol']
        else:                      self.update_unpol = True
        self.DEN = {}
        self.DEN['SIDIS'] = {}
        self.DEN['SIA']   = {}


        flavs = ['u','d','s','c','ub','db','sb','cb','g']
        self.PDF   = conf['LHAPDF:PDF']
        self.f1 = {_:{} for _ in flavs}
        if 'use LHAPDF' in conf and conf['use LHAPDF']: 
            self.TPDF = conf['LHAPDF:TPDF'] 
            self.h1 = {_:{} for _ in flavs}
            self.predict = True
        else:
            self.predict = False

        self.ind = {}
        self.ind['u']  =  2
        self.ind['d']  =  1
        self.ind['s']  =  3
        self.ind['c']  =  4
        self.ind['ub'] = -2
        self.ind['db'] = -1
        self.ind['sb'] = -3
        self.ind['cb'] = -4
        self.ind['g']  =  21

    def get_f1(self,x,Q2,flav):

        return self.PDF[0].xfxQ2(self.ind[flav],x,Q2)/x

    def get_h1(self,x,Q2,flav):

        return self.TPDF[0].xfxQ2(self.ind[flav],x,Q2)/x

    #--add option for no integration
    def get_sig(self,idx,flavs=None):
     
        #--3 points is sufficient for this integration
        xg, wg = np.polynomial.legendre.leggauss(3)
 
        RS = self.tabs[idx]['RS'][0]
        if flavs==None:
            #--no bottom quark at Belle energy 
            if   RS == 10.58:  flavs = ['u','d','s','c']
            #--include bottom quark at higher energies
            elif RS > 10.58:   flavs = ['u','d','s','c','b']


        if 'units' in self.tabs[idx]: units = self.tabs[idx]['units'][0].strip()
        else:                         units = 'nb/GeV'

        zdo  = np.array(self.tabs[idx]['zdo'])
        zup  = np.array(self.tabs[idx]['zup'])
        _M   = np.array(self.tabs[idx]['M'])
        _Q2  = np.array(self.tabs[idx]['Q2'])
        val  = np.array(self.tabs[idx]['value'])

        l = len(_M)

        alpha = np.array([conf['eweak'].get_alpha(q2) for q2 in _Q2])

        Nc = 3
        prefactor =  4*np.pi*Nc*alpha**2/3/_Q2


        #--z, M, Q2 are input as arrays
        D = lambda z,M,Q2,flav: self. diffpippim.get_D(z,M,Q2,flav)

        e2  = lambda Q2,flav:      self.aux.get_aX('plus',flav,Q2)
        sig = lambda z,M,Q2,flav:  e2(Q2,flav) * D(z,M,Q2,flav)

        shape = (l,len(xg)) 
        Z   = np.zeros(shape) 
        M   = np.zeros(shape) 
        Q2  = np.zeros(shape)

        for m in range(l):

            #--Gaussian quadrature integration over z 
            _Z     = 0.5*(zup[m]-zdo[m])*xg + 0.5*(zup[m]+zdo[m])
            jac_Z  = 0.5*(zup[m]-zdo[m])
            Zbin   = zup[m]-zdo[m]

            #--integrate over z
            for i in range(len(xg)):
                Z[m][i]  = _Z[i]
                Q2[m][i] = _Q2[m]
                M[m][i]  = _M[m]


        Z   = Z.flatten()
        Q2  = Q2.flatten()
        M   = M.flatten()

        thy = np.zeros(len(Z))
        for flav in flavs:
            #--factor of 2 from antiquark
            thy += 2*sig(Z,M,Q2,flav)

        thy = thy.reshape(shape)
        #--integrate over z
        thy = np.sum(wg*thy,axis=1)

        thy = thy*jac_Z/Zbin

        thy = prefactor*thy
        
        if units=='nb/GeV': thy*= 389.4*10**3

        return thy

    #--add option for no integration (check if it is correct)
    def get_a12R(self,idx):
   
        update = self.update_unpol
        if idx not in self.DEN['SIA']: update = True
 
        if self.integrate: NZ,NM = 4,8
        else:              NZ,NM = 1,1
        #--4 points for z and 8 for M is sufficient
        xgZ, wgZ = np.polynomial.legendre.leggauss(NZ)
        xgM, wgM = np.polynomial.legendre.leggauss(NM)

        #--ignore bottom quark since this is at Belle energy 
        flavs = []
        flavs.append(['u' ,'ub'])
        flavs.append(['d' ,'db'])
        flavs.append(['s' ,'sb'])
        flavs.append(['c' ,'cb'])

        #--charge of positive quarks
        ep = 2/3
        #--charge of negative quarks
        em = -1/3

        _Q2    = np.array(self.tabs[idx]['Q2'])
        val   = np.array(self.tabs[idx]['value'])
        sin2  = np.array(self.tabs[idx]['sin2theta'])
        cos2  = np.array(self.tabs[idx]['cos2theta'])
        RS    = self.tabs[idx]['RS'][0]

        if self.integrate:
            z1min = np.array(self.tabs[idx]['z1min'])
            z1max = np.array(self.tabs[idx]['z1max'])
            M1min = np.array(self.tabs[idx]['M1min'])
            M1max = np.array(self.tabs[idx]['M1max'])
            z2min = np.array(self.tabs[idx]['z2min'])
            z2max = np.array(self.tabs[idx]['z2max'])
            M2min = np.array(self.tabs[idx]['M2min'])
            M2max = np.array(self.tabs[idx]['M2max'])
        else:
            z1 = np.array(self.tabs[idx]['z1'])
            M1 = np.array(self.tabs[idx]['M1'])
            z2 = np.array(self.tabs[idx]['z2'])
            M2 = np.array(self.tabs[idx]['M2'])

        l = len(_Q2)

        #--is a factor of 1/2 needed here?
        #prefactor = 0.5*sin2/(1+cos2)
        prefactor = sin2/(1+cos2)

        H = lambda z,M,Q2,flav: self.tdiffpippim.get_H(z,M,Q2,flav)
        D = lambda z,M,Q2,flav: self. diffpippim.get_D(z,M,Q2,flav)

        num = lambda e,z1,z2,M1,M2,Q2,flav1,flav2: e**2 * H(z1,M1,Q2,flav1) * H(z2,M2,Q2,flav2)
        den = lambda e,z1,z2,M1,M2,Q2,flav1,flav2: e**2 * D(z1,M1,Q2,flav1) * D(z2,M2,Q2,flav2)

        shape = (l,len(xgZ),len(xgM),len(xgZ),len(xgM)) 
        Z1   = np.zeros(shape) 
        Z2   = np.zeros(shape)
        M1   = np.zeros(shape)
        M2   = np.zeros(shape)
        Q2   = np.zeros(shape)
        flag = np.zeros(shape)


        for m in range(l):

            #--Gaussian quadrature integration over z1
            if self.integrate:
                _Z1 = 0.5*(z1max[m]-z1min[m])*xgZ + 0.5*(z1max[m]+z1min[m])
                jac_Z1 = 0.5*(z1max[m]-z1min[m])
                Z1bin = z1max[m]-z1min[m]
            else:
                _Z1 = [z1[m]]
                jac_Z1 = 1.0
                Z1bin = 1.0

            #--Gaussian quadrature integration over z2
            if self.integrate:
                _Z2 = 0.5*(z2max[m]-z2min[m])*xgZ + 0.5*(z2max[m]+z2min[m])
                jac_Z2 = 0.5*(z2max[m]-z2min[m])
                Z2bin = z2max[m]-z2min[m]
            else:
                _Z2 = [z2[m]]
                jac_Z2 = 1.0
                Z2bin = 1.0

            #--integrate over z1
            for i in range(len(xgZ)):

                #--Gaussian quadrature integration over M1
                if self.integrate:
                    M1max1 = M1max[m]
                    M1max2 = RS/2 * _Z1[i]
                    M1MAX = min(M1max1,M1max2)
                    _M1 = 0.5*(M1MAX-M1min[m])*xgM + 0.5*(M1MAX+M1min[m])
                    jac_M1 = 0.5*(M1MAX-M1min[m])
                    M1bin = M1MAX-M1min[m]
                else:
                    _M1 = [M1[m]]
                    jac_M1 = 1.0
                    M1bin = 1.0

                #--integrate over M1
                for j in range(len(xgM)):

                    #--integrate over z2
                    for k in range(len(xgZ)):
                        #--Gaussian quadrature integration over M2
                        if self.integrate:
                            M2max1 = M2max[m]
                            M2max2 = RS/2 * _Z2[k]
                            M2MAX  = min(M2max1,M2max2)
                            _M2 = 0.5*(M2MAX-M2min[m])*xgM + 0.5*(M2MAX+M2min[m])
                            jac_M2 = 0.5*(M2MAX-M2min[m])
                            M2bin = M2MAX-M2min[m]
                        else:
                            _M2 = [M2[m]]
                            jac_M2 = 1.0
                            M2bin = 1.0

                        #--integrate over M2
                        for n in range(len(xgM)):
                            if self.integrate:
                                if   M1MAX < M1min[m]: _flag = 0.0
                                elif M2MAX < M2min[m]: _flag = 0.0
                                else:                  _flag = 1.0
                            else: _flag = 1.0
                            Z1[m][i][j][k][n] = _Z1[i]
                            M1[m][i][j][k][n] = _M1[j]
                            Z2[m][i][j][k][n] = _Z2[k]
                            M2[m][i][j][k][n] = _M2[n]
                            Q2[m][i][j][k][n] = _Q2[m]
                            flag[m][i][j][k][n] = _flag


        Z1   = Z1.flatten()
        Z2   = Z2.flatten()
        M1   = M1.flatten()
        M2   = M2.flatten()
        Q2   = Q2.flatten()
        flag = flag.flatten()

        NUM = np.zeros(len(Z1))
        if update: DEN = np.zeros(len(Z1))

        for flav in flavs:
            if   flav[0] in ['u','c']:   e = ep
            elif flav[0] in ['d','s']:   e = em

            #--factor of 2 from (qb,q) contributions (cancels out anyway)
            NUM            += flag*2*num(e,Z1,Z2,M1,M2,Q2,flav[0],flav[1]) 
            if update: DEN += flag*2*den(e,Z1,Z2,M1,M2,Q2,flav[0],flav[1])

        NUM = NUM.reshape(shape)

        NUM = np.sum(wgM*NUM,axis=4)
        NUM = np.sum(wgZ*NUM,axis=3)
        NUM = np.sum(wgM*NUM,axis=2)
        NUM = np.sum(wgZ*NUM,axis=1)

        NUM = NUM*jac_M2*jac_Z2*jac_M1*jac_Z1/M2bin/Z2bin/M1bin/Z1bin

        if update:
            DEN = DEN.reshape(shape)
            DEN = np.sum(wgM*DEN,axis=4)
            DEN = np.sum(wgZ*DEN,axis=3)
            DEN = np.sum(wgM*DEN,axis=2)
            DEN = np.sum(wgZ*DEN,axis=1)

            DEN = DEN*jac_M2*jac_Z2*jac_M1*jac_Z1/M2bin/Z2bin/M1bin/Z1bin

            self.DEN['SIA'][idx] = DEN

        thy = prefactor*NUM/self.DEN['SIA'][idx]

        return thy

    #--with limits on outer y integration
    #--log scaling on x integration
    def get_A_UT(self,idx):

        update = self.update_unpol
        if idx not in self.DEN['SIDIS']: update = True
 
        col   = self.tabs[idx]['col'][0]

        #--these # of points are sufficient
        if col=='HERMES':
            if self.integrate: NX,NY,NZ,NM = 3,3,3,4
            else:              NX,NY,NZ,NM = 1,1,1,1
            xgX, wgX = np.polynomial.legendre.leggauss(NX)
            xgY, wgY = np.polynomial.legendre.leggauss(NY)
            xgZ, wgZ = np.polynomial.legendre.leggauss(NZ)
            xgM, wgM = np.polynomial.legendre.leggauss(NM)
        #--these # of points are sufficient
        if col=='COMPASS':
            if self.integrate: NX,NY,NZ,NM = 4,3,4,5
            else:              NX,NY,NZ,NM = 1,1,1,1
            xgX, wgX = np.polynomial.legendre.leggauss(NX)
            xgY, wgY = np.polynomial.legendre.leggauss(NY)
            xgZ, wgZ = np.polynomial.legendre.leggauss(NZ)
            xgM, wgM = np.polynomial.legendre.leggauss(NM)


        flavs = ['u','d','s','c','ub','db','sb','cb']

        #--charge of positive quarks
        ep = 2/3
        #--charge of negative quarks
        em = -1/3

        if self.integrate:
            xmin  = np.array(self.tabs[idx]['xdo'])
            xmax  = np.array(self.tabs[idx]['xup'])
            ymin  = np.array(self.tabs[idx]['ymin'])
            ymax  = np.array(self.tabs[idx]['ymax'])
            mmin  = np.array(self.tabs[idx]['Mmin'])
            mmax  = np.array(self.tabs[idx]['Mmax'])
            zmin  = np.array(self.tabs[idx]['zmin'])
            zmax  = np.array(self.tabs[idx]['zmax'])
        else:
            x = self.tabs[idx]['x'] 
            y = self.tabs[idx]['y'] 
            z = self.tabs[idx]['z'] 
            Mh= self.tabs[idx]['M']

        _Q2   = np.array(self.tabs[idx]['Q2'])
        W2min = np.array(self.tabs[idx]['W2min'])
        Q2min = np.array(self.tabs[idx]['Q2min'])
        S     = np.array(self.tabs[idx]['S'])
        tar   = self.tabs[idx]['tar'][0]
        l = len(S)

        M2  = conf['aux'].M2
        mpi = conf['aux'].Mpi
        #--transverse spin transfer coefficient
        Dnn_num = lambda y: (1-y)
        Dnn_den = lambda y: (1-y+y**2/2)

        R    = lambda M: 0.5*M*np.sqrt(1 - 4*mpi**2/M**2)
        R2M2 = lambda M: R(M)**2/M**2

        if col=='HERMES':
            pre_num = lambda y, M: -Dnn_num(y) * 4/(np.pi)
            pre_den = lambda y, M:  Dnn_den(y)
        if col=='COMPASS':
            pre_num = lambda y, M: 1.0
            pre_den = lambda y, M: 1.0

        H = lambda z,M,Q2,flav: self.tdiffpippim.get_H(z,M,Q2,flav)
        D = lambda z,M,Q2,flav: self. diffpippim.get_D(z,M,Q2,flav)
        if self.predict: h = lambda x,Q2,flav:   self.get_h1(x,Q2,flav)
        else:            h = lambda x,Q2,flav:   self.tpdf.get_xF(x,Q2,flav)/x
        f = lambda x,Q2,flav:   self.get_f1(x,Q2,flav)

        num = lambda e2,x,z,M,Q2,y,flav: pre_num(y,M) * e2 * h(x,Q2,flav) * H(z,M,Q2,flav)
        den = lambda e2,x,z,M,Q2,y,flav: pre_den(y,M) * e2 * f(x,Q2,flav) * D(z,M,Q2,flav)

        shape = (l,len(xgY),len(xgX),len(xgZ),len(xgM)) 
        X    = np.zeros(shape) 
        Y    = np.zeros(shape)
        Z    = np.zeros(shape)
        M    = np.zeros(shape)
        Q2   = np.zeros(shape)
        flag = np.zeros(shape)


        for m in range(l):
            #--Gaussian quadrature integration over z
            if self.integrate:
                _Z = 0.5*(zmax[m]-zmin[m])*xgZ + 0.5*(zmax[m]+zmin[m])
                jac_Z = 0.5*(zmax[m]-zmin[m])
                Zbin = zmax[m]-zmin[m]
            else:
                _Z = [z[m]]
                jac_Z = 1.0
                Zbin = 1.0

            #--Gaussian quadrature integration over Y
            if self.integrate:
                ymin1 = ymin[m]
                ymin2 = Q2min[m]/xmax[m]/(S[m]-M2)
                ymin3 = (W2min[m]-M2)/(1-xmin[m])/(S[m]-M2)
                yMIN  = max(ymin1,ymin2,ymin3)
                _Y = 0.5*(ymax[m]-yMIN)*xgY + 0.5*(ymax[m]+yMIN)
                jac_Y = 0.5*(ymax[m]-yMIN)
                Ybin = ymax[m]-yMIN
            else:
                _Y = [y[m]]
                jac_Y = 1.0
                Ybin  = 1.0

            for i in range(len(xgY)):
                #--Gaussian quadrature integration over x
                if self.integrate:
                    xmin1 = xmin[m]
                    xmin2 = Q2min[m]/_Y[i]/(S[m]-M2)
                    xMIN  = np.log(max(xmin1,xmin2))
                    xmax1 = xmax[m]
                    xmax2 = (_Y[i]*(S[m]-M2) - W2min[m] + M2)/_Y[i]/(S[m]-M2)
                    xMAX = np.log(min(xmax1,xmax2))
                    _X = np.exp(0.5*(xMAX-xMIN)*xgX + 0.5*(xMAX+xMIN))
                    jac_X = 0.5*(xMAX-xMIN)
                    Xbin  = xMAX-xMIN
                else:
                    _X = [x[m]]
                    jac_X = 1.0
                    Xbin  = 1.0

                for j in range(len(xgX)):

                    for k in range(len(xgZ)):
                        #--Gaussian quadrature integration over M (upper limit depends on Z and Y)
                        if self.integrate:
                            Mmax1 = mmax[m]
                            Mmax2 = np.sqrt(_Z[k]*_Y[i]*S[m] - M2)
                            Mmax  = min(Mmax1,Mmax2)
                            Mmin  = mmin[m]
                            _M = 0.5*(Mmax-Mmin)*xgM + 0.5*(Mmax+Mmin)
                            jac_M = 0.5*(Mmax-Mmin)
                            Mbin = Mmax-Mmin
                        else:
                            _M = [Mh[m]]
                            jac_M = 1.0
                            Mbin  = 1.0

                        for n in range(len(xgM)):
                            if self.integrate: 
                                if   xMIN > xMAX: _flag = 0.0
                                elif Mmin > Mmax: _flag = 0.0
                                else:             _flag = 1.0
                            else: _flag = 1.0
                            Y [m][i][j][k][n] = _Y[i]
                            X [m][i][j][k][n] = _X[j]
                            Z [m][i][j][k][n] = _Z[k]
                            M [m][i][j][k][n] = _M[n]
                            Q2[m][i][j][k][n] = _X[j]*_Y[i]*(S[m]-M2)
                            flag[m][i][j][k][n] = _flag


        X    = X.flatten()
        Y    = Y.flatten()
        Z    = Z.flatten()
        M    = M.flatten()
        Q2   = Q2.flatten()
        flag = flag.flatten()

        NUM = np.zeros(len(X))
        if update: DEN = np.zeros(len(X))

        for flav in flavs:
            if tar=='p':
                if   flav in ['u','ub','c','cb']:   e2N = ep**2
                elif flav in ['d','db','s','sb']:   e2N = em**2
                if   flav in ['u','ub','c','cb']:   e2D = ep**2
                elif flav in ['d','db','s','sb']:   e2D = em**2
            #--for deuteron: u -> u + d, d -> u + d, s -> 2s, c -> 2c
            elif tar=='d':
                if   flav in ['u','ub','c','cb']:   e2N = ep**2 - em**2
                elif flav in ['d','db','s','sb']:   e2N = -(ep**2 - em**2)
                if   flav in ['u','d','ub','db']:   e2D = ep**2 + em**2
                elif flav in ['c','cb']:            e2D = 2*ep**2
                elif flav in ['s','sb']:            e2D = 2*em**2

            NUM            += flag*num(e2N,X,Z,M,Q2,Y,flav) 
            if update: DEN += flag*den(e2D,X,Z,M,Q2,Y,flav)

        NUM = NUM.reshape(shape)

        NUM = np.sum(wgM*NUM,axis=4)
        NUM = np.sum(wgZ*NUM,axis=3)
        NUM = np.sum(wgX*NUM,axis=2)
        NUM = np.sum(wgY*NUM,axis=1)

        NUM = NUM*jac_Z*jac_M*jac_X*jac_Y/Zbin/Mbin/Xbin/Ybin

        if update:
            DEN = DEN.reshape(shape)
            DEN = np.sum(wgM*DEN,axis=4)
            DEN = np.sum(wgZ*DEN,axis=3)
            DEN = np.sum(wgX*DEN,axis=2)
            DEN = np.sum(wgY*DEN,axis=1)

            DEN = DEN*jac_Z*jac_M*jac_X*jac_Y/Zbin/Mbin/Xbin/Ybin

            self.DEN['SIDIS'][idx] = DEN

        thy = NUM/self.DEN['SIDIS'][idx]

        return thy
    
    def get_theory(self):
    
        for idx in self.tabs:
            process  = self.tabs[idx]['process'][0].strip()
            obs      = self.tabs[idx]['obs'][0].strip()
            hads     = self.tabs[idx]['hadrons'][0].strip()
            col   = self.tabs[idx]['col'][0]

            if hads=='pi+_pi-': hads = 'pi+,pi-'

            if process=='SIA' and hads=='pi+,pi-':  

                if obs=='sig': thy = self.get_sig(idx)

                #--this observable is sigma_q/sigma_tot, tot = u+d+s+c+b
                #--this observable is only from PYTHIA
                if obs=='sig_rat':
                    flav = self.tabs[idx]['channel'][0]
                    num  = self.get_sig(idx,[flav])
                    den  = self.get_sig(idx)
                    thy  = num/den
            
            if process=='SIA'   and hads=='2(pi+,pi-)' and obs=='a12R': thy = self.get_a12R(idx)

            if process=='SIDIS' and hads=='pi+,pi-'    and obs=='A_UT': thy = self.get_A_UT(idx)

            if process=='pp'    and hads=='pi+,pi-'    and obs=='A_UT': thy,_,_ = self.thy.get_asym(idx)

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
            L.append('reaction: dihadron; Q20: %s'%conf['Q20'])
            for f in conf['datasets']['dihadron']['filters']:
                L.append('filters: %s'%f)
  
            L.append('%7s %3s %20s %5s %10s %10s %10s %10s'%('idx','tar','col','npts','chi2','chi2/npts','rchi2','nchi2'))
            for k in self.tabs:
                if len(self.tabs[k])==0: continue 
                res=self.tabs[k]['residuals']
  
                rres=[]
                for c in conf['rparams']['dihadron'][k]:
                    rres.append(conf['rparams']['dihadron'][k][c]['value'])
                rres=np.array(rres)
  
                if k in conf['datasets']['dihadron']['norm']:
                    norm=conf['datasets']['dihadron']['norm'][k]
                    nres=(norm['value']-1)/norm['dN']
                else:
                    nres=0
  
                chi2=np.sum(res**2)
                rchi2=np.sum(rres**2)
                nchi2=nres**2
                if 'target' in self.tabs[k]: tar=self.tabs[k]['target'][0]
                else: tar = '-'
                col=self.tabs[k]['col'][0].split()[0]
                npts=res.size
                L.append('%7d %3s %20s %5d %10.2f %10.2f %10.2f %10.2f'%(k,tar,col,npts,chi2,chi2/npts,rchi2,nchi2))
  
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


    def get_A_UT_old(self,idx):

        update = self.update_unpol
        if idx not in self.DEN['SIDIS']: update = True
 
        col   = self.tabs[idx]['col'][0]

        #--these # of points are sufficient
        if col=='HERMES':
            if self.integrate: NX,NY,NZ,NM = 3,3,3,4
            else:              NX,NY,NZ,NM = 1,1,1,1
            xgX, wgX = np.polynomial.legendre.leggauss(NX)
            xgY, wgY = np.polynomial.legendre.leggauss(NY)
            xgZ, wgZ = np.polynomial.legendre.leggauss(NZ)
            xgM, wgM = np.polynomial.legendre.leggauss(NM)
        #--these # of points are sufficient
        if col=='COMPASS':
            if self.integrate: NX,NY,NZ,NM = 4,3,4,5
            else:              NX,NY,NZ,NM = 1,1,1,1
            xgX, wgX = np.polynomial.legendre.leggauss(NX)
            xgY, wgY = np.polynomial.legendre.leggauss(NY)
            xgZ, wgZ = np.polynomial.legendre.leggauss(NZ)
            xgM, wgM = np.polynomial.legendre.leggauss(NM)


        flavs = ['u','d','s','c','ub','db','sb','cb']

        #--charge of positive quarks
        ep = 2/3
        #--charge of negative quarks
        em = -1/3

        if self.integrate:
            xmin  = np.array(self.tabs[idx]['xdo'])
            xmax  = np.array(self.tabs[idx]['xup'])
            ymin  = np.array(self.tabs[idx]['ymin'])
            ymax  = np.array(self.tabs[idx]['ymax'])
            mmin  = np.array(self.tabs[idx]['Mmin'])
            mmax  = np.array(self.tabs[idx]['Mmax'])
            zmin  = np.array(self.tabs[idx]['zmin'])
            zmax  = np.array(self.tabs[idx]['zmax'])
        else:
            x = self.tabs[idx]['x'] 
            y = self.tabs[idx]['y'] 
            z = self.tabs[idx]['z'] 
            Mh= self.tabs[idx]['M']

        _Q2   = np.array(self.tabs[idx]['Q2'])
        W2min = np.array(self.tabs[idx]['W2min'])
        Q2min = np.array(self.tabs[idx]['Q2min'])
        S     = np.array(self.tabs[idx]['S'])
        tar   = self.tabs[idx]['tar'][0]
        l = len(S)

        M2  = conf['aux'].M2
        mpi = conf['aux'].Mpi
        #--transverse spin transfer coefficient
        Dnn_num = lambda y: (1-y)
        Dnn_den = lambda y: (1-y+y**2/2)

        R    = lambda M: 0.5*M*np.sqrt(1 - 4*mpi**2/M**2)
        R2M2 = lambda M: R(M)**2/M**2

        if col=='HERMES':
            pre_num = lambda y, M: -Dnn_num(y) * 4/(3*np.pi) / (1-R2M2(M))
            pre_den = lambda y, M:  Dnn_den(y) / (3 - 4*R2M2(M))
        if col=='COMPASS':
            pre_num = lambda y, M: 1.0
            pre_den = lambda y, M: 1.0

        H = lambda z,M,Q2,flav: self.tdiffpippim.get_H(z,M,Q2,flav)
        D = lambda z,M,Q2,flav: self. diffpippim.get_D(z,M,Q2,flav)
        if self.predict: h = lambda x,Q2,flav:   self.get_h1(x,Q2,flav)
        else:            h = lambda x,Q2,flav:   self.tpdf.get_xF(x,Q2,flav)/x
        f = lambda x,Q2,flav:   self.get_f1(x,Q2,flav)

        num = lambda e2,x,z,M,Q2,y,flav: pre_num(y,M) * e2 * h(x,Q2,flav) * H(z,M,Q2,flav)
        den = lambda e2,x,z,M,Q2,y,flav: pre_den(y,M) * e2 * f(x,Q2,flav) * D(z,M,Q2,flav)

        shape = (l,len(xgY),len(xgX),len(xgZ),len(xgM)) 
        X    = np.zeros(shape) 
        Y    = np.zeros(shape)
        Z    = np.zeros(shape)
        M    = np.zeros(shape)
        Q2   = np.zeros(shape)
        flag = np.zeros(shape)


        for m in range(l):
            #--Gaussian quadrature integration over z
            if self.integrate:
                _Z = 0.5*(zmax[m]-zmin[m])*xgZ + 0.5*(zmax[m]+zmin[m])
                jac_Z = 0.5*(zmax[m]-zmin[m])
                Zbin = zmax[m]-zmin[m]
            else:
                _Z = [z[m]]
                jac_Z = 1.0
                Zbin = 1.0

            #--Gaussian quadrature integration over Y
            if self.integrate:
                ymin1 = ymin[m]
                ymin2 = Q2min[m]/xmax[m]/(S[m]-M2)
                ymin3 = (W2min[m]-M2)/(1-xmin[m])/(S[m]-M2)
                yMIN  = max(ymin1,ymin2,ymin3)
                _Y = 0.5*(ymax[m]-yMIN)*xgY + 0.5*(ymax[m]+yMIN)
                jac_Y = 0.5*(ymax[m]-yMIN)
                Ybin = ymax[m]-yMIN
            else:
                _Y = [y[m]]
                jac_Y = 1.0
                Ybin  = 1.0

            for i in range(len(xgY)):
                #--Gaussian quadrature integration over x
                if self.integrate:
                    xmin1 = xmin[m]
                    xmin2 = Q2min[m]/_Y[i]/(S[m]-M2)
                    xMIN  = np.log(max(xmin1,xmin2))
                    xmax1 = xmax[m]
                    xmax2 = (_Y[i]*(S[m]-M2) - W2min[m] + M2)/_Y[i]/(S[m]-M2)
                    xMAX = np.log(min(xmax1,xmax2))
                    _X = np.exp(0.5*(xMAX-xMIN)*xgX + 0.5*(xMAX+xMIN))
                    jac_X = 0.5*(xMAX-xMIN)
                    Xbin  = xMAX-xMIN
                else:
                    _X = [x[m]]
                    jac_X = 1.0
                    Xbin  = 1.0

                for j in range(len(xgX)):

                    for k in range(len(xgZ)):
                        #--Gaussian quadrature integration over M (upper limit depends on Z and Y)
                        if self.integrate:
                            Mmax1 = mmax[m]
                            Mmax2 = np.sqrt(_Z[k]*_Y[i]*S[m] - M2)
                            Mmax  = min(Mmax1,Mmax2)
                            Mmin  = mmin[m]
                            _M = 0.5*(Mmax-Mmin)*xgM + 0.5*(Mmax+Mmin)
                            jac_M = 0.5*(Mmax-Mmin)
                            Mbin = Mmax-Mmin
                        else:
                            _M = [Mh[m]]
                            jac_M = 1.0
                            Mbin  = 1.0

                        for n in range(len(xgM)):
                            if self.integrate: 
                                if   xMIN > xMAX: _flag = 0.0
                                elif Mmin > Mmax: _flag = 0.0
                                else:             _flag = 1.0
                            else: _flag = 1.0
                            Y [m][i][j][k][n] = _Y[i]
                            X [m][i][j][k][n] = _X[j]
                            Z [m][i][j][k][n] = _Z[k]
                            M [m][i][j][k][n] = _M[n]
                            Q2[m][i][j][k][n] = _X[j]*_Y[i]*(S[m]-M2)
                            flag[m][i][j][k][n] = _flag


        X    = X.flatten()
        Y    = Y.flatten()
        Z    = Z.flatten()
        M    = M.flatten()
        Q2   = Q2.flatten()
        flag = flag.flatten()

        NUM = np.zeros(len(X))
        if update: DEN = np.zeros(len(X))

        for flav in flavs:
            if tar=='p':
                if   flav in ['u','ub','c','cb']:   e2N = ep**2
                elif flav in ['d','db','s','sb']:   e2N = em**2
                if   flav in ['u','ub','c','cb']:   e2D = ep**2
                elif flav in ['d','db','s','sb']:   e2D = em**2
            #--for deuteron: u -> u + d, d -> u + d, s -> 2s, c -> 2c
            elif tar=='d':
                if   flav in ['u','ub','c','cb']:   e2N = ep**2 - em**2
                elif flav in ['d','db','s','sb']:   e2N = -(ep**2 - em**2)
                if   flav in ['u','d','ub','db']:   e2D = ep**2 + em**2
                elif flav in ['c','cb']:            e2D = 2*ep**2
                elif flav in ['s','sb']:            e2D = 2*em**2

            NUM            += flag*num(e2N,X,Z,M,Q2,Y,flav) 
            if update: DEN += flag*den(e2D,X,Z,M,Q2,Y,flav)

        NUM = NUM.reshape(shape)

        NUM = np.sum(wgM*NUM,axis=4)
        NUM = np.sum(wgZ*NUM,axis=3)
        NUM = np.sum(wgX*NUM,axis=2)
        NUM = np.sum(wgY*NUM,axis=1)

        NUM = NUM*jac_Z*jac_M*jac_X*jac_Y/Zbin/Mbin/Xbin/Ybin

        if update:
            DEN = DEN.reshape(shape)
            DEN = np.sum(wgM*DEN,axis=4)
            DEN = np.sum(wgZ*DEN,axis=3)
            DEN = np.sum(wgX*DEN,axis=2)
            DEN = np.sum(wgY*DEN,axis=1)

            DEN = DEN*jac_Z*jac_M*jac_X*jac_Y/Zbin/Mbin/Xbin/Ybin

            self.DEN['SIDIS'][idx] = DEN

        thy = NUM/self.DEN['SIDIS'][idx]

        return thy

 
if __name__ == "__main__":

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

    def load_data():
    
        conf['datasets'] = {}
        data = {}
    
        conf['datasets']['dihadron']={}
        conf['datasets']['dihadron']['filters']=[]
        #conf['datasets']['dihadron']['filters'].append("Q2>%f"%Q2cut)
        conf['datasets']['dihadron']['xlsx']={}
        conf['datasets']['dihadron']['xlsx'][2000]='dihadron/expdata/2000.xlsx'
        conf['datasets']['dihadron']['xlsx'][2001]='dihadron/expdata/2001.xlsx'
        conf['datasets']['dihadron']['xlsx'][2002]='dihadron/expdata/2002.xlsx'
        #------------------------------------------------------------------------------------------------------------------
        #conf['datasets']['dihadron']['xlsx'][3000]='dihadron/expdata/3000.xlsx'
        #conf['datasets']['dihadron']['xlsx'][3001]='dihadron/expdata/3001.xlsx'
        #conf['datasets']['dihadron']['xlsx'][3002]='dihadron/expdata/3002.xlsx'
        #conf['datasets']['dihadron']['xlsx'][3010]='dihadron/expdata/3010.xlsx'
        #conf['datasets']['dihadron']['xlsx'][3011]='dihadron/expdata/3011.xlsx'
        #conf['datasets']['dihadron']['xlsx'][3012]='dihadron/expdata/3012.xlsx'
        #conf['datasets']['dihadron']['xlsx'][3020]='dihadron/expdata/3020.xlsx'
        #conf['datasets']['dihadron']['xlsx'][3021]='dihadron/expdata/3021.xlsx'
        #conf['datasets']['dihadron']['xlsx'][3022]='dihadron/expdata/3022.xlsx'
        conf['dihadron tabs'] = READER().load_data_sets('dihadron')  
     
        data = conf['dihadron tabs'].copy()
    
        return data

    def gen_data(wdir,DATA):

        conf['aux'] = AUX()
   
        core = CORE()
 
        load_config('%s/input.py'%wdir)

        #conf['lhapdf_pdf'] = 'NNPDF40_nnlo_as_01180'

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
     
        data = {} 
        data['num'] = {}
        data['den'] = {}
        data['asym'] = {}
        for idx in DATA:
            #binning = DATA[idx]['binning'][0]
            data['num'][idx],data['den'][idx],data['asym'][idx] = get_A_UT(DATA[idx])
            #data['num'][idx],data['den'][idx],data['asym'][idx] = get_A_UT_intxMy(DATA[idx])
        print(data)
        
        #--save data for comparison between different values of ng
        filename = 'data/SIDIS.dat'
        checkdir('data')
        save(data,filename)
        print('saving data to %s'%filename)

    def get_A_UT(data): 
  
        xg, wg = np.polynomial.legendre.leggauss(5)
        diffpippim  = conf['diffpippim']
        tdiffpippim = conf['tdiffpippim']
        pdf  = conf['LHAPDF:PDF'][0]
        tpdf = conf['LHAPDF:TPDF'][0]
 
        flavs = ['u','d','s','c','ub','db','sb','cb']

        ind = {}
        ind['u']  =  2
        ind['d']  =  1
        ind['s']  =  3
        ind['c']  =  4
        ind['ub'] = -2
        ind['db'] = -1
        ind['sb'] = -3
        ind['cb'] = -4
        ind['g']  =  21

        #--charge of positive quarks
        ep = 2/3
        #--charge of negative quarks
        em = -1/3

        col   = data['col'][0]

        xdo   = np.array(data['xdo'])
        xup   = np.array(data['xup'])
        Q2    = np.array(data['Q2'])
        ymin  = np.array(data['ymin'])
        ymax  = np.array(data['ymax'])
        mmin  = np.array(data['Mmin'])
        mmax  = np.array(data['Mmax'])
        zmin  = np.array(data['zmin'])
        zmax  = np.array(data['zmax'])
        W2min = np.array(data['W2min'])
        Q2min = np.array(data['Q2min'])
        S     = np.array(data['S'])
        tar   = data['tar'][0]
        l = len(Q2)

        M2  = conf['aux'].M2
        mpi = conf['aux'].Mpi
        #--transverse spin transfer coefficient
        Dnn_num = lambda y: (1-y)
        Dnn_den = lambda y: (1-y+y**2/2)

        R    = lambda M: 0.5*M*np.sqrt(1 - 4*mpi**2/M**2)
        R2M2 = lambda M: R(M)**2/M**2

        if col=='HERMES':
            pre_num = lambda y, M: -Dnn_num(y) * 4/(3*np.pi) / (1-R2M2(M))
            pre_den = lambda y, M:  Dnn_den(y) / (3 - 4*R2M2(M))
        if col=='COMPASS':
            pre_num = lambda y, M: 1.0
            pre_den = lambda y, M: 1.0

        H = lambda z,M,Q2,flav: tdiffpippim.get_H(z,M,Q2,flav)
        D = lambda z,M,Q2,flav:  diffpippim.get_D(z,M,Q2,flav)
        h = lambda x,Q2,flav:   tpdf.xfxQ2(ind[flav],x,Q2)/x
        f = lambda x,Q2,flav:    pdf.xfxQ2(ind[flav],x,Q2)/x

        num = lambda e2,x,z,M,Q2,y,flav: pre_num(y,M) * e2 * h(x,Q2,flav) * H(z,M,Q2,flav)
        den = lambda e2,x,z,M,Q2,y,flav: pre_den(y,M) * e2 * f(x,Q2,flav) * D(z,M,Q2,flav)

        NUM = np.zeros((len(xg),len(xg),len(xg),len(xg),l)) 
        DEN = np.zeros((len(xg),len(xg),len(xg),len(xg),l)) 

        #--Gaussian quadrature integration over z
        Z = 0.5*(zmax-zmin)*xg[:,None] + 0.5*(zmax+zmin)
        jac_Z = 0.5*(zmax-zmin)
        Zbin = zmax-zmin

        #--Gaussian quadrature integration over Y
        Y = 0.5*(ymax-ymin)*xg[:,None] + 0.5*(ymax+ymin)
        jac_Y = 0.5*(ymax-ymin)
        Ybin = ymax-ymin

        for i in range(len(xg)):
            #--Gaussian quadrature integration over x
            #--upper limit depends on W2
            #--lower limit depends on y
            #--at low y, one can have xmin > xmax.
            #--flag will avoid these contributions
            xup1 = xup
            xup2 = Q2/(W2min - M2 + Q2)
            xmax = np.minimum(xup1,xup2)
            xdo1 = xdo
            xdo2 = Q2min/Y[i]/(S-M2)
            xmin = np.maximum(xdo1,xdo2)
            X = 0.5*(xmax-xmin)*xg[:,None] + 0.5*(xmax+xmin)
            jac_X = 0.5*(xmax-xmin)
            Xbin  = xmax-xmin

            flag = np.ones(l)
            for _ in range(l):
                if xmin[_] > xmax[_]: flag[_] = 0.0

            for j in range(len(xg)):
                #print(col)
                #print(flag)
                #print(X[j]*Y[i]*(S-M2))
                #print(Q2/X[j] +M2 - Q2)
                for k in range(len(xg)):
                    #--Gaussian quadrature integration over M (upper limit depends on Z and Y)
                    Mmax1 = mmax
                    Mmax2 = np.sqrt(Z[k]*Y[i]*S[0] - M2)
                    Mmax  = np.minimum(Mmax1,Mmax2)
                    Mmin  = mmin
                    M = 0.5*(Mmax-Mmin)*xg[:,None] + 0.5*(Mmax+Mmin)
                    jac_M = 0.5*(Mmax-Mmin)
                    Mbin = Mmax-Mmin
                    for n in range(len(xg)):
                        for flav in flavs:
                            if tar=='p':
                                if   flav in ['u','ub','c','cb']:   e2N = ep**2
                                elif flav in ['d','db','s','sb']:   e2N = em**2
                                if   flav in ['u','ub','c','cb']:   e2D = ep**2
                                elif flav in ['d','db','s','sb']:   e2D = em**2
                            #--for deuteron: u -> u + d, d -> u + d, s -> 2s, c -> 2c
                            elif tar=='d':
                                if   flav in ['u','ub','c','cb']:   e2N = ep**2 - em**2
                                elif flav in ['d','db','s','sb']:   e2N = -(ep**2 - em**2)
                                if   flav in ['u','d','ub','db']:   e2D = ep**2 + em**2
                                elif flav in ['c','cb']:            e2D = 2*ep**2
                                elif flav in ['s','sb']:            e2D = 2*em**2

                            NUM[i][j][k][n] += flag*num(e2N,X[j],Z[k],M[n],Q2,Y[i],flav) 
                            DEN[i][j][k][n] += flag*den(e2D,X[j],Z[k],M[n],Q2,Y[i],flav)


        NUM = np.sum(wg[:,None]*NUM,axis=3)
        NUM = np.sum(wg[:,None]*NUM,axis=2)
        NUM = np.sum(wg[:,None]*NUM,axis=1)
        NUM = np.sum(wg[:,None]*NUM,axis=0)

        NUM = NUM*jac_Z*jac_M*jac_X*jac_Y

        NUM = NUM/Zbin/Mbin/Xbin/Ybin

        DEN = np.sum(wg[:,None]*DEN,axis=3)
        DEN = np.sum(wg[:,None]*DEN,axis=2)
        DEN = np.sum(wg[:,None]*DEN,axis=1)
        DEN = np.sum(wg[:,None]*DEN,axis=0)

        DEN = DEN*jac_Z*jac_M*jac_X*jac_Y

        DEN = DEN/Zbin/Mbin/Xbin/Ybin

        thy = NUM/DEN

        return NUM,DEN,thy

    def plot_sidis_x(thy):
    
        data = conf['dihadron tabs']
        nrows,ncols=3,2
        fig = py.figure(figsize=(ncols*8,nrows*5))
        ax = {}
        for i in range(6):
            ax[i+1] = py.subplot(nrows,ncols,i+1)
    
        #######################
        #--plot absolute values
        #######################
    
        hand = {}
        #--plot data
        for idx in data:
            if   idx==3000: i,color = 1,'darkgreen'
            elif idx==3001: i,color = 2,'firebrick'
            elif idx==3002: i,color = 2,'darkblue'
            else: continue        
        
            x   = data[idx]['x']
            xdo = data[idx]['xdo']
            xup = data[idx]['xup']
            M = data[idx]['M']
            values = data[idx]['value']
            if idx==3000: alpha  = data[idx]['stat_u']
            if idx==3001: alpha  = np.sqrt(data[idx]['stat_u']**2 + data[idx]['syst_u']**2)
            if idx==3002: alpha  = np.sqrt(data[idx]['stat_u']**2 + data[idx]['syst_u']**2)
            xerr = [x-xdo,xup-x]
    
            hand[idx] = ax[i].errorbar(x,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
     
            hand['thy'] ,= ax[i].plot(x,thy['asym'][idx],color=color)
   
            if   idx==3000: i,color = 3,'darkgreen'
            elif idx==3001: i,color = 4,'firebrick'
            elif idx==3002: i,color = 4,'darkblue'
            else: continue        
             
            hand['thy'] ,= ax[i].plot(x,thy['num'][idx],color=color)
 
            if   idx==3000: i,color = 5,'darkgreen'
            elif idx==3001: i,color = 6,'firebrick'
            elif idx==3002: i,color = 6,'darkblue'
            else: continue        
             
            hand['thy'] ,= ax[i].plot(x,thy['den'][idx],color=color)
    
        for i in [1,3,5]:
            ax[i].set_xlim(0.00,0.19)
            ax[i].set_xticks([0.05,0.10,0.15])
            minorLocator = MultipleLocator(0.01)
            ax[i].xaxis.set_minor_locator(minorLocator)
    
        for i in [2,4,6]:
            ax[i].semilogx()
            ax[i].set_xlim(2e-3,0.40)

        for i in range(6):
            ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
            ax[i+1].xaxis.set_tick_params(which='major',length=6)
            ax[i+1].xaxis.set_tick_params(which='minor',length=3)
            ax[i+1].yaxis.set_tick_params(which='major',length=6)
            ax[i+1].yaxis.set_tick_params(which='minor',length=3)
    
        for i in range(6):
            ax[i+1].tick_params(labelbottom=False)
    
        ax[5].tick_params(labelbottom=True)
        ax[6].tick_params(labelbottom=True)
        ax[5].set_xlabel(r'\boldmath$x$',size=30)
        ax[5].xaxis.set_label_coords(0.95,-0.02)
        ax[5].tick_params(axis='x',which='major',pad=8)
        ax[6].set_xlabel(r'\boldmath$x$',size=30)
        ax[6].xaxis.set_label_coords(0.95,-0.02)
        ax[6].tick_params(axis='x',which='major',pad=8)

        ax[1].set_ylim(-0.02,0.07)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.02)
        ax[1].yaxis.set_minor_locator(minorLocator)
        ax[1].yaxis.set_major_locator(majorLocator)
    
    
        ax[2].set_xlim(2e-3,0.40)
        ax[2].set_ylim(-0.19,0.11)
        ax[2].set_yticks([-0.15,-0.10,-0.05,0.00,0.05,0.10])
        minorLocator = MultipleLocator(0.01)
        ax[2].yaxis.set_minor_locator(minorLocator)
    
    
        ax[1].set_ylabel(r'\boldmath$A_{U \perp}$',size=30)
        ax[3].set_ylabel(r'\boldmath$\Delta \sigma$',size=30)
        ax[5].set_ylabel(r'\boldmath$       \sigma$',size=30)
        #ax[1] .text(0.05, 0.80, r'\boldmath$A_{U \perp}^{\sin(\phi_{R \perp} + \phi_S) \sin(\theta)}$',transform=ax[1].transAxes,size=35)
    
        ax[1].axhline(0,0,1,color='black',ls='--',alpha=0.5)
        ax[2].axhline(0,0,1,color='black',ls='--',alpha=0.5)
    
        handles,labels = [], []
        handles.append(hand[3000])
        handles.append(hand['thy'])
        labels.append(r'\textbf{\textrm{HERMES}}') 
        labels.append(r'\textbf{\textrm{JAM22-3D}}') 
        ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)
    
        handles,labels = [], []
        handles.append(hand[3001])
        handles.append(hand[3002])
        labels.append(r'\textbf{\textrm{COMPASS \boldmath$p$}}') 
        labels.append(r'\textbf{\textrm{COMPASS \boldmath$D$}}') 
        ax[2].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)
    
        py.tight_layout()
        py.subplots_adjust(hspace=0.05,wspace=0.20)
    
    
        checkdir('predictions')
        filename='predictions/SIDIS_x.png'
    
        py.savefig(filename)
        print('Saving SIDIS plot to %s'%filename)

    def plot_sidis_M(thy):
    
        data = conf['dihadron tabs']
        nrows,ncols=3,2
        fig = py.figure(figsize=(ncols*8,nrows*5))
        ax = {}
        for i in range(6):
            ax[i+1] = py.subplot(nrows,ncols,i+1)
    
        #######################
        #--plot absolute values
        #######################
    
        hand = {}
        #--plot data
        for idx in data:
            if   idx==3010: i,color = 1,'darkgreen'
            elif idx==3011: i,color = 2,'firebrick'
            elif idx==3012: i,color = 2,'darkblue'
            else: continue        
        
            xdo = data[idx]['xdo']
            xup = data[idx]['xup']
            M = data[idx]['M']
            values = data[idx]['value']
            if idx==3010: alpha  = data[idx]['stat_u']
            if idx==3011: alpha  = np.sqrt(data[idx]['stat_u']**2 + data[idx]['syst_u']**2)
            if idx==3012: alpha  = np.sqrt(data[idx]['stat_u']**2 + data[idx]['syst_u']**2)
    
            hand[idx] = ax[i].errorbar(M,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
     
            hand['thy'] ,= ax[i].plot(M,thy['asym'][idx],color=color)
   
            if   idx==3010: i,color = 3,'darkgreen'
            elif idx==3011: i,color = 4,'firebrick'
            elif idx==3012: i,color = 4,'darkblue'
            else: continue        
             
            hand['thy'] ,= ax[i].plot(M,thy['num'][idx],color=color)
 
            if   idx==3010: i,color = 5,'darkgreen'
            elif idx==3011: i,color = 6,'firebrick'
            elif idx==3012: i,color = 6,'darkblue'
            else: continue        
            
            hand['thy'] ,= ax[i].plot(M,thy['den'][idx],color=color)
    
        for i in [1,3,5]:
            ax[i].set_xlim(0.00,2.00)
            #ax[i].set_xticks([0.05,0.10,0.15])
            #minorLocator = MultipleLocator(0.01)
            #ax[i].xaxis.set_minor_locator(minorLocator)
    
        for i in [2,4,6]:
            ax[i].set_xlim(0,2.00)

        for i in range(6):
            ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
            ax[i+1].xaxis.set_tick_params(which='major',length=6)
            ax[i+1].xaxis.set_tick_params(which='minor',length=3)
            ax[i+1].yaxis.set_tick_params(which='major',length=6)
            ax[i+1].yaxis.set_tick_params(which='minor',length=3)
    
        for i in range(6):
            ax[i+1].tick_params(labelbottom=False)
    
        ax[5].tick_params(labelbottom=True)
        ax[6].tick_params(labelbottom=True)
        ax[5].set_xlabel(r'\boldmath$M_h$',size=30)
        ax[5].xaxis.set_label_coords(0.95,-0.02)
        ax[5].tick_params(axis='x',which='major',pad=8)
        ax[6].set_xlabel(r'\boldmath$M_h$',size=30)
        ax[6].xaxis.set_label_coords(0.95,-0.02)
        ax[6].tick_params(axis='x',which='major',pad=8)

        ax[1].set_ylim(-0.02,0.07)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.02)
        ax[1].yaxis.set_minor_locator(minorLocator)
        ax[1].yaxis.set_major_locator(majorLocator)
    
    
        ax[2].set_ylim(-0.19,0.11)
        ax[2].set_yticks([-0.15,-0.10,-0.05,0.00,0.05,0.10])
        minorLocator = MultipleLocator(0.01)
        ax[2].yaxis.set_minor_locator(minorLocator)
    
    
        ax[1].set_ylabel(r'\boldmath$A_{U \perp}$',size=30)
        ax[3].set_ylabel(r'\boldmath$\Delta \sigma$',size=30)
        ax[5].set_ylabel(r'\boldmath$       \sigma$',size=30)
        #ax[1] .text(0.05, 0.80, r'\boldmath$A_{U \perp}^{\sin(\phi_{R \perp} + \phi_S) \sin(\theta)}$',transform=ax[1].transAxes,size=35)
    
        ax[1].axhline(0,0,1,color='black',ls='--',alpha=0.5)
        ax[2].axhline(0,0,1,color='black',ls='--',alpha=0.5)
    
        handles,labels = [], []
        handles.append(hand[3010])
        handles.append(hand['thy'])
        labels.append(r'\textbf{\textrm{HERMES}}') 
        labels.append(r'\textbf{\textrm{JAM22-3D}}') 
        ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)
    
        handles,labels = [], []
        handles.append(hand[3011])
        handles.append(hand[3012])
        labels.append(r'\textbf{\textrm{COMPASS \boldmath$p$}}') 
        labels.append(r'\textbf{\textrm{COMPASS \boldmath$D$}}') 
        ax[2].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)
    
        py.tight_layout()
        py.subplots_adjust(hspace=0.05,wspace=0.20)
    
    
        checkdir('predictions')
        filename='predictions/SIDIS_M.png'
    
        py.savefig(filename)
        print('Saving SIDIS plot to %s'%filename)

    def plot_sidis_z(thy):
    
        data = conf['dihadron tabs']
        nrows,ncols=3,2
        fig = py.figure(figsize=(ncols*8,nrows*5))
        ax = {}
        for i in range(6):
            ax[i+1] = py.subplot(nrows,ncols,i+1)
    
        #######################
        #--plot absolute values
        #######################
    
        hand = {}
        #--plot data
        for idx in data:
            if   idx==3020: i,color = 1,'darkgreen'
            elif idx==3021: i,color = 2,'firebrick'
            elif idx==3022: i,color = 2,'darkblue'
            else: continue        
        
            z   = data[idx]['z']
            xdo = data[idx]['xdo']
            xup = data[idx]['xup']
            M = data[idx]['M']
            values = data[idx]['value']
            if idx==3020: alpha  = data[idx]['stat_u']
            if idx==3021: alpha  = np.sqrt(data[idx]['stat_u']**2 + data[idx]['syst_u']**2)
            if idx==3022: alpha  = np.sqrt(data[idx]['stat_u']**2 + data[idx]['syst_u']**2)
    
            hand[idx] = ax[i].errorbar(z,values,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
     
            hand['thy'] ,= ax[i].plot(z,thy['asym'][idx],color=color)
   
            if   idx==3020: i,color = 3,'darkgreen'
            elif idx==3021: i,color = 4,'firebrick'
            elif idx==3022: i,color = 4,'darkblue'
            else: continue        
             
            hand['thy'] ,= ax[i].plot(z,thy['num'][idx],color=color)
 
            if   idx==3020: i,color = 5,'darkgreen'
            elif idx==3021: i,color = 6,'firebrick'
            elif idx==3022: i,color = 6,'darkblue'
            else: continue        
             
            hand['thy'] ,= ax[i].plot(z,thy['den'][idx],color=color)
    
        for i in [1,3,5]:
            ax[i].set_xlim(0.00,1.00)
            #ax[i].set_xticks([0.05,0.10,0.15])
            #minorLocator = MultipleLocator(0.01)
            #ax[i].xaxis.set_minor_locator(minorLocator)
    
        for i in [2,4,6]:
            ax[i].set_xlim(0,1.00)

        for i in range(6):
            ax[i+1].tick_params(axis='both',which='both',top=True,right=True,labelbottom=False,direction='in',labelsize=30)
            ax[i+1].xaxis.set_tick_params(which='major',length=6)
            ax[i+1].xaxis.set_tick_params(which='minor',length=3)
            ax[i+1].yaxis.set_tick_params(which='major',length=6)
            ax[i+1].yaxis.set_tick_params(which='minor',length=3)
    
        for i in range(6):
            ax[i+1].tick_params(labelbottom=False)
    
        ax[5].tick_params(labelbottom=True)
        ax[6].tick_params(labelbottom=True)
        ax[5].set_xlabel(r'\boldmath$z$',size=30)
        ax[5].xaxis.set_label_coords(0.95,-0.02)
        ax[5].tick_params(axis='x',which='major',pad=8)
        ax[6].set_xlabel(r'\boldmath$z$',size=30)
        ax[6].xaxis.set_label_coords(0.95,-0.02)
        ax[6].tick_params(axis='x',which='major',pad=8)

        ax[1].set_ylim(-0.02,0.07)
        minorLocator = MultipleLocator(0.01)
        majorLocator = MultipleLocator(0.02)
        ax[1].yaxis.set_minor_locator(minorLocator)
        ax[1].yaxis.set_major_locator(majorLocator)
    
    
        ax[2].set_ylim(-0.19,0.11)
        ax[2].set_yticks([-0.15,-0.10,-0.05,0.00,0.05,0.10])
        minorLocator = MultipleLocator(0.01)
        ax[2].yaxis.set_minor_locator(minorLocator)
    
    
        ax[1].set_ylabel(r'\boldmath$A_{U \perp}$',size=30)
        ax[3].set_ylabel(r'\boldmath$\Delta \sigma$',size=30)
        ax[5].set_ylabel(r'\boldmath$       \sigma$',size=30)
        #ax[1] .text(0.05, 0.80, r'\boldmath$A_{U \perp}^{\sin(\phi_{R \perp} + \phi_S) \sin(\theta)}$',transform=ax[1].transAxes,size=35)
    
        ax[1].axhline(0,0,1,color='black',ls='--',alpha=0.5)
        ax[2].axhline(0,0,1,color='black',ls='--',alpha=0.5)
    
        handles,labels = [], []
        handles.append(hand[3020])
        handles.append(hand['thy'])
        labels.append(r'\textbf{\textrm{HERMES}}') 
        labels.append(r'\textbf{\textrm{JAM22-3D}}') 
        ax[1].legend(handles,labels,frameon=False,fontsize=22,loc='upper right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)
    
        handles,labels = [], []
        handles.append(hand[3021])
        handles.append(hand[3022])
        labels.append(r'\textbf{\textrm{COMPASS \boldmath$p$}}') 
        labels.append(r'\textbf{\textrm{COMPASS \boldmath$D$}}') 
        ax[2].legend(handles,labels,frameon=False,fontsize=22,loc='lower left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)
    
        py.tight_layout()
        py.subplots_adjust(hspace=0.05,wspace=0.20)
    
    
        checkdir('predictions')
        filename='predictions/SIDIS_z.png'
    
        py.savefig(filename)
        print('Saving SIDIS plot to %s'%filename)

    def test(wdir):

        L = 30
        idxs = [0,1,2,3]
        conf['dihadron tabs'] = {}
        for i in idxs:
            conf['dihadron tabs'][i] = {}
            conf['dihadron tabs'][i]['col']     = ['HERMES']
            conf['dihadron tabs'][i]['process'] = ['SIDIS']
            conf['dihadron tabs'][i]['hadrons'] = ['pi+,pi-']
            conf['dihadron tabs'][i]['tar']     = ['p']
            conf['dihadron tabs'][i]['obs']     = ['A_UT']
            conf['dihadron tabs'][i]['S']       = np.array([52.3]*L)
            conf['dihadron tabs'][i]['Q2']      = np.array([3.0]*L)
       
        #--function of x
        conf['dihadron tabs'][0]['x']     = np.linspace(0.005,0.3,L)
        conf['dihadron tabs'][0]['y']     = np.array([0.4]*L)
        conf['dihadron tabs'][0]['z']     = np.array([0.4]*L)
        conf['dihadron tabs'][0]['M']     = np.array([0.8]*L)
        #--function of z
        conf['dihadron tabs'][1]['z']     = np.linspace(0.2,1.0,L)
        conf['dihadron tabs'][1]['x']     = np.array([0.1]*L)
        conf['dihadron tabs'][1]['y']     = np.array([0.4]*L)
        conf['dihadron tabs'][1]['M']     = np.array([0.8]*L)
        #--function of M
        conf['dihadron tabs'][2]['M']     = np.linspace(0.28,2.0,L)
        conf['dihadron tabs'][2]['x']     = np.array([0.1]*L)
        conf['dihadron tabs'][2]['y']     = np.array([0.4]*L)
        conf['dihadron tabs'][2]['z']     = np.array([0.4]*L)
        #--function of y
        conf['dihadron tabs'][3]['y']     = np.linspace(0.1,0.9,L)
        conf['dihadron tabs'][3]['x']     = np.array([0.1]*L)
        conf['dihadron tabs'][3]['z']     = np.array([0.4]*L)
        conf['dihadron tabs'][3]['M']     = np.array([0.8]*L)
        conf['aux'] = AUX()

        tabs = conf['dihadron tabs']
   
        core = CORE()

        load_config('%s/input.py'%wdir)

        #conf['lhapdf_pdf'] = 'NNPDF40_nnlo_as_01180'

        istep=core.get_istep()
        replicas=core.get_replicas(wdir)
        core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        parman.order=replicas[0]['order'][istep]

        jar=load('%s/data/jar-%d.dat'%(wdir,istep))
        replicas=jar['replicas']

        #--take single replica
        par = replicas[1]

        parman.set_new_params(par,initial=True)
     
        data = {} 
        data['num'] = {}
        data['den'] = {}
        data['thy'] = {}
        for idx in tabs:
            data['num'][idx],data['den'][idx],data['thy'][idx] = get_A_UT_noint(tabs[idx])

        nrows,ncols=3,4
        fig = py.figure(figsize=(ncols*9,nrows*5))
        ax11=py.subplot(nrows,ncols,1)
        ax12=py.subplot(nrows,ncols,2)
        ax13=py.subplot(nrows,ncols,3)
        ax14=py.subplot(nrows,ncols,4)
        ax21=py.subplot(nrows,ncols,5)
        ax22=py.subplot(nrows,ncols,6)
        ax23=py.subplot(nrows,ncols,7)
        ax24=py.subplot(nrows,ncols,8)
        ax31=py.subplot(nrows,ncols,9)
        ax32=py.subplot(nrows,ncols,10)
        ax33=py.subplot(nrows,ncols,11)
        ax34=py.subplot(nrows,ncols,12)

        tabs = conf['dihadron tabs']

        hand = {}

        #--plot z and Mh
        for idx in tabs:

            if idx==0: ax = ax11
            if idx==1: ax = ax12
            if idx==2: ax = ax13
            if idx==3: ax = ax14

            if idx==0:
                x = tabs[idx]['x']
                hand['thy']   ,= ax.plot(x,data['thy'][idx],color='firebrick')
            if idx==1:
                z = tabs[idx]['z']
                hand['thy']   ,= ax.plot(z,data['thy'][idx],color='firebrick')
            if idx==2:
                M = tabs[idx]['M']
                hand['thy']   ,= ax.plot(M,data['thy'][idx],color='firebrick')
            if idx==3:
                y = tabs[idx]['y']
                hand['thy']   ,= ax.plot(y,data['thy'][idx],color='firebrick')

            factor = 1

            if idx==0: ax = ax21
            if idx==1: ax = ax22
            if idx==2: ax = ax23
            if idx==3: ax = ax24

            if idx==0:
                x = tabs[idx]['x']
                hand['thy']   ,= ax.plot(x,data['num'][idx],color='firebrick')
            if idx==1:
                z = tabs[idx]['z']
                hand['thy']   ,= ax.plot(z,data['num'][idx],color='firebrick')
            if idx==2:
                M = tabs[idx]['M']
                hand['thy']   ,= ax.plot(M,data['num'][idx],color='firebrick')
            if idx==3:
                y = tabs[idx]['y']
                hand['thy']   ,= ax.plot(y,data['num'][idx],color='firebrick')

            if idx==0: ax = ax31
            if idx==1: ax = ax32
            if idx==2: ax = ax33
            if idx==3: ax = ax34

            if idx==0:
                x = tabs[idx]['x']
                hand['thy']   ,= ax.plot(x,data['den'][idx],color='firebrick')
            if idx==1:
                z = tabs[idx]['z']
                hand['thy']   ,= ax.plot(z,data['den'][idx],color='firebrick')
            if idx==2:
                M = tabs[idx]['M']
                hand['thy']   ,= ax.plot(M,data['den'][idx],color='firebrick')
            if idx==3:
                y = tabs[idx]['y']
                hand['thy']   ,= ax.plot(y,data['den'][idx],color='firebrick')


        for ax in [ax11,ax21,ax31]: 
            ax.semilogx()
            ax.set_xlim(0.005,0.3)
            #ax.set_xticks([0.2,0.4,0.6,0.8])
            minorLocator = MultipleLocator(0.1)
            ax.xaxis.set_minor_locator(minorLocator)

        for ax in [ax12,ax22,ax32]: 
            ax.set_xlim(0.2,1.0)
            ax.set_xticks([0.4,0.6,0.8])
            minorLocator = MultipleLocator(0.1)
            ax.xaxis.set_minor_locator(minorLocator)

        for ax in [ax13,ax23,ax33]: 
            ax.set_xlim(0.28,2.0)
            ax.set_xticks([0.4,0.8,1.2,1.6])
            minorLocator = MultipleLocator(0.1)
            ax.xaxis.set_minor_locator(minorLocator)

        for ax in [ax14,ax24,ax34]: 
            ax.set_xlim(0.1,0.9)
            ax.set_xticks([0.2,0.4,0.6,0.8])
            minorLocator = MultipleLocator(0.1)
            ax.xaxis.set_minor_locator(minorLocator)

        ax31.set_xlabel(r'\boldmath$x$',size=30)
        ax32.set_xlabel(r'\boldmath$z$',size=30)
        ax33.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=30)
        ax34.set_xlabel(r'\boldmath$y$',size=30)

        ax11.set_ylabel(r'\boldmath$A_{U \perp}$',size=30)
        ax21.set_ylabel(r'\boldmath$\Delta \sigma$',size=30)
        ax31.set_ylabel(r'\boldmath$       \sigma$',size=30)

        for ax in [ax11,ax12,ax13,ax14,ax21,ax22,ax23,ax24,ax31,ax32,ax33,ax34]:
            ax .tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
            ax .tick_params(axis='both',which='minor',size=4)
            ax .tick_params(axis='both',which='major',size=8)
            ax .axhline(0,0,1,color='black',ls='--',alpha=0.5)


        #ax11.text(0.05,0.15,r'\boldmath$\eta=%3.2f$'%tabs[0]['eta'][0] ,transform=ax11.transAxes, size=25)
        #ax11.text(0.05,0.05,r'\boldmath$M_h = %3.2f ~ {\rm GeV}$'%(tabs[0]['M'][0]) ,transform=ax11.transAxes, size=25)
        #ax11.text(0.05,0.05,r'\boldmath$%3.2f < M_h < %3.2f ~ {\rm GeV}$'%(tabs[0]['Mmin'][0],tabs[0]['Mmax'][0]) ,transform=ax11.transAxes, size=25)

        #ax11.text(0.05,0.60,r'\boldmath$\sqrt{s} = %d ~ {\rm GeV}$'%(tabs[0]['RS'][0]) ,transform=ax11.transAxes, size=25)
        #fs = 30

        #handles,labels = [], []
        #handles.append(hand['xspace'])
        #handles.append(hand['mellin'])
        #labels.append(r'\textrm{\textbf{xspace}}')
        #labels.append(r'\textrm{\textbf{mellin+interp}}')
        #ax21.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


        py.tight_layout()
        py.subplots_adjust(wspace=0.14)
        filename='gallery/test_SIDIS'
        filename+='.png'

        checkdir('gallery')
        py.savefig(filename)
        print ('Saving figure to %s'%filename)
        py.clf()

    def get_mean_kin(DATA): 
 
        conf['aux'] = AUX()
   
        core = CORE()
 
        load_config('%s/input.py'%wdir)

        #conf['lhapdf_pdf'] = 'NNPDF40_nnlo_as_01180'

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

 
        xg, wg = np.polynomial.legendre.leggauss(5)
        diffpippim  = conf['diffpippim']
        tdiffpippim = conf['tdiffpippim']
        pdf  = conf['LHAPDF:PDF'][0]
        tpdf = conf['LHAPDF:TPDF'][0]
 
        flavs = ['u','d','s','c','ub','db','sb','cb']

        ind = {}
        ind['u']  =  2
        ind['d']  =  1
        ind['s']  =  3
        ind['c']  =  4
        ind['ub'] = -2
        ind['db'] = -1
        ind['sb'] = -3
        ind['cb'] = -4
        ind['g']  =  21

        #--charge of positive quarks
        ep = 2/3
        #--charge of negative quarks
        em = -1/3

        for idx in DATA:
            data = DATA[idx]

            col   = data['col'][0]

            x     = np.array(data['x'])
            xdo   = np.array(data['xdo'])
            xup   = np.array(data['xup'])
            Q2    = np.array(data['Q2'])
            y     = np.array(data['y'])
            ymin  = np.array(data['ymin'])
            ymax  = np.array(data['ymax'])
            m     = np.array(data['M'])
            mmin  = np.array(data['Mmin'])
            mmax  = np.array(data['Mmax'])
            z     = np.array(data['z'])
            zmin  = np.array(data['zmin'])
            zmax  = np.array(data['zmax'])
            W2min = np.array(data['W2min'])
            Q2min = np.array(data['Q2min'])
            S     = np.array(data['S'])
            tar   = data['tar'][0]
            l = len(Q2)

            M2  = conf['aux'].M2
            mpi = conf['aux'].Mpi
            #--transverse spin transfer coefficient
            Dnn_num = lambda y: (1-y)
            Dnn_den = lambda y: (1-y+y**2/2)

            R    = lambda M: 0.5*M*np.sqrt(1 - 4*mpi**2/M**2)
            R2M2 = lambda M: R(M)**2/M**2

            if col=='HERMES':
                pre_num = lambda y, M: -Dnn_num(y) * 4/(3*np.pi) / (1-R2M2(M))
                pre_den = lambda y, M:  Dnn_den(y) / (3 - 4*R2M2(M))
            if col=='COMPASS':
                pre_num = lambda y, M: 1.0
                pre_den = lambda y, M: 1.0

            H = lambda z,M,Q2,flav: tdiffpippim.get_H(z,M,Q2,flav)
            D = lambda z,M,Q2,flav:  diffpippim.get_D(z,M,Q2,flav)
            h = lambda x,Q2,flav:   tpdf.xfxQ2(ind[flav],x,Q2)/x
            f = lambda x,Q2,flav:    pdf.xfxQ2(ind[flav],x,Q2)/x

            den = lambda e2,x,z,M,Q2,y,flav: pre_den(y,M) * e2 * f(x,Q2,flav) * D(z,M,Q2,flav)

            NUMX = np.zeros((len(xg),len(xg),len(xg),len(xg),l)) 
            NUMY = np.zeros((len(xg),len(xg),len(xg),len(xg),l)) 
            NUMZ = np.zeros((len(xg),len(xg),len(xg),len(xg),l)) 
            NUMM = np.zeros((len(xg),len(xg),len(xg),len(xg),l)) 
            DEN  = np.zeros((len(xg),len(xg),len(xg),len(xg),l)) 

            #--Gaussian quadrature integration over z
            Z = 0.5*(zmax-zmin)*xg[:,None] + 0.5*(zmax+zmin)
            jac_Z = 0.5*(zmax-zmin)
            Zbin = zmax-zmin

            #--Gaussian quadrature integration over Y
            Y = 0.5*(ymax-ymin)*xg[:,None] + 0.5*(ymax+ymin)
            jac_Y = 0.5*(ymax-ymin)
            Ybin = ymax-ymin

            for i in range(len(xg)):
                #--Gaussian quadrature integration over x
                #--upper limit depends on W2
                #--lower limit depends on y
                #--at low y, one can have xmin > xmax.
                #--flag will avoid these contributions
                xup1 = xup
                xup2 = Q2/(W2min - M2 + Q2)
                xmax = np.minimum(xup1,xup2)
                xdo1 = xdo
                xdo2 = Q2min/Y[i]/(S-M2)
                xmin = np.maximum(xdo1,xdo2)
                X = 0.5*(xmax-xmin)*xg[:,None] + 0.5*(xmax+xmin)
                jac_X = 0.5*(xmax-xmin)
                Xbin  = xmax-xmin

                flag = np.ones(l)
                for _ in range(l):
                    if xmin[_] > xmax[_]: flag[_] = 0.0

                for j in range(len(xg)):
                    #print(col)
                    #print(flag)
                    #print(X[j]*Y[i]*(S-M2))
                    #print(Q2/X[j] +M2 - Q2)
                    for k in range(len(xg)):
                        #--Gaussian quadrature integration over M (upper limit depends on Z and Y)
                        Mmax1 = mmax
                        Mmax2 = np.sqrt(Z[k]*Y[i]*S[0] - M2)
                        Mmax  = np.minimum(Mmax1,Mmax2)
                        Mmin  = mmin
                        M = 0.5*(Mmax-Mmin)*xg[:,None] + 0.5*(Mmax+Mmin)
                        jac_M = 0.5*(Mmax-Mmin)
                        Mbin = Mmax-Mmin
                        for n in range(len(xg)):
                            for flav in flavs:
                                if tar=='p':
                                    if   flav in ['u','ub','c','cb']:   e2N = ep**2
                                    elif flav in ['d','db','s','sb']:   e2N = em**2
                                    if   flav in ['u','ub','c','cb']:   e2D = ep**2
                                    elif flav in ['d','db','s','sb']:   e2D = em**2
                                #--for deuteron: u -> u + d, d -> u + d, s -> 2s, c -> 2c
                                elif tar=='d':
                                    if   flav in ['u','ub','c','cb']:   e2N = ep**2 - em**2
                                    elif flav in ['d','db','s','sb']:   e2N = -(ep**2 - em**2)
                                    if   flav in ['u','d','ub','db']:   e2D = ep**2 + em**2
                                    elif flav in ['c','cb']:            e2D = 2*ep**2
                                    elif flav in ['s','sb']:            e2D = 2*em**2

                                NUMX[i][j][k][n] += X[j]*flag*den(e2D,X[j],Z[k],M[n],Q2,Y[i],flav) 
                                NUMY[i][j][k][n] += Y[i]*flag*den(e2D,X[j],Z[k],M[n],Q2,Y[i],flav) 
                                NUMZ[i][j][k][n] += Z[k]*flag*den(e2D,X[j],Z[k],M[n],Q2,Y[i],flav) 
                                NUMM[i][j][k][n] += M[n]*flag*den(e2D,X[j],Z[k],M[n],Q2,Y[i],flav) 
                                DEN[i][j][k][n]  +=      flag*den(e2D,X[j],Z[k],M[n],Q2,Y[i],flav)


            NUMX = np.sum(wg[:,None]*NUMX,axis=3)
            NUMX = np.sum(wg[:,None]*NUMX,axis=2)
            NUMX = np.sum(wg[:,None]*NUMX,axis=1)
            NUMX = np.sum(wg[:,None]*NUMX,axis=0)

            NUMX = NUMX*jac_Z*jac_M*jac_X*jac_Y

            NUMX = NUMX/Zbin/Mbin/Xbin/Ybin

            NUMY = np.sum(wg[:,None]*NUMY,axis=3)
            NUMY = np.sum(wg[:,None]*NUMY,axis=2)
            NUMY = np.sum(wg[:,None]*NUMY,axis=1)
            NUMY = np.sum(wg[:,None]*NUMY,axis=0)

            NUMY = NUMY*jac_Z*jac_M*jac_X*jac_Y

            NUMY = NUMY/Zbin/Mbin/Xbin/Ybin

            NUMZ = np.sum(wg[:,None]*NUMZ,axis=3)
            NUMZ = np.sum(wg[:,None]*NUMZ,axis=2)
            NUMZ = np.sum(wg[:,None]*NUMZ,axis=1)
            NUMZ = np.sum(wg[:,None]*NUMZ,axis=0)

            NUMZ = NUMZ*jac_Z*jac_M*jac_X*jac_Y

            NUMZ = NUMZ/Zbin/Mbin/Xbin/Ybin

            NUMM = np.sum(wg[:,None]*NUMM,axis=3)
            NUMM = np.sum(wg[:,None]*NUMM,axis=2)
            NUMM = np.sum(wg[:,None]*NUMM,axis=1)
            NUMM = np.sum(wg[:,None]*NUMM,axis=0)

            NUMM = NUMM*jac_Z*jac_M*jac_X*jac_Y

            NUMM = NUMM/Zbin/Mbin/Xbin/Ybin


            DEN = np.sum(wg[:,None]*DEN,axis=3)
            DEN = np.sum(wg[:,None]*DEN,axis=2)
            DEN = np.sum(wg[:,None]*DEN,axis=1)
            DEN = np.sum(wg[:,None]*DEN,axis=0)

            DEN = DEN*jac_Z*jac_M*jac_X*jac_Y

            DEN = DEN/Zbin/Mbin/Xbin/Ybin

            AVGX = NUMX/DEN
            AVGY = NUMY/DEN
            AVGZ = NUMZ/DEN
            AVGM = NUMM/DEN

            print(idx)
            print('  X_thy', '  X_exp')
            for i in range(len(AVGX)): 
                print(i, '%3.4f'%AVGX[i], '%3.4f'%x[i])

            print('  Y_thy', '  Y_exp')
            for i in range(len(AVGX)): 
                print(i, '%3.4f'%AVGY[i], '%3.4f'%y[i])

            print('  Z_thy', '  Z_exp')
            for i in range(len(AVGX)): 
                print(i, '%3.4f'%AVGZ[i], '%3.4f'%z[i])

            print('  M_thy', '  M_exp')
            for i in range(len(AVGX)): 
                print(i, '%3.4f'%AVGM[i], '%3.4f'%m[i])

        return

    def plot_sia_kin(data): 


        Z1,M1,Z2,M2,Q2 = {},{},{},{},{}

        for idx in data:
            Z1[idx], M1[idx], Z2[idx], M2[idx], Q2[idx] = [],[],[],[],[]
            tab = data[idx] 
            _Q2    = np.array(tab['Q2'])
            RS    = tab['RS'][0]

            z1min = np.array(tab['z1min'])
            z1max = np.array(tab['z1max'])
            M1min = np.array(tab['M1min'])
            M1max = np.array(tab['M1max'])
            z2min = np.array(tab['z2min'])
            z2max = np.array(tab['z2max'])
            M2min = np.array(tab['M2min'])
            M2max = np.array(tab['M2max'])
            l = len(_Q2)

            N = 5
            xgZ, wgZ = np.polynomial.legendre.leggauss(N)
            xgM, wgM = np.polynomial.legendre.leggauss(N)


            for m in range(l):

                #--Gaussian quadrature integration over z1
                _Z1 = 0.5*(z1max[m]-z1min[m])*xgZ + 0.5*(z1max[m]+z1min[m])
                jac_Z1 = 0.5*(z1max[m]-z1min[m])
                Z1bin = z1max[m]-z1min[m]

                #--Gaussian quadrature integration over z2
                _Z2 = 0.5*(z2max[m]-z2min[m])*xgZ + 0.5*(z2max[m]+z2min[m])
                jac_Z2 = 0.5*(z2max[m]-z2min[m])
                Z2bin = z2max[m]-z2min[m]

                #--integrate over z1
                for i in range(len(xgZ)):

                    #--Gaussian quadrature integration over M1
                    M1max1 = M1max[m]
                    M1max2 = RS/2 * _Z1[i]
                    M1MAX = min(M1max1,M1max2)
                    _M1 = 0.5*(M1MAX-M1min[m])*xgM + 0.5*(M1MAX+M1min[m])
                    jac_M1 = 0.5*(M1MAX-M1min[m])
                    M1bin = M1MAX-M1min[m]

                    #--integrate over M1
                    for j in range(len(xgM)):

                        #--integrate over z2
                        for k in range(len(xgZ)):
                            #--Gaussian quadrature integration over M2
                            M2max1 = M2max[m]
                            M2max2 = RS/2 * _Z2[k]
                            M2MAX  = min(M2max1,M2max2)
                            _M2 = 0.5*(M2MAX-M2min[m])*xgM + 0.5*(M2MAX+M2min[m])
                            jac_M2 = 0.5*(M2MAX-M2min[m])
                            M2bin = M2MAX-M2min[m]

                            #--integrate over M2
                            for n in range(len(xgM)):
                                if   M1MAX < M1min[m]: _flag = 0.0
                                elif M2MAX < M2min[m]: _flag = 0.0
                                else:                  _flag = 1.0 
                                if _flag==0: continue
                                Z1[idx].append(_Z1[i])
                                M1[idx].append(_M1[j])
                                Z2[idx].append(_Z2[k])
                                M2[idx].append(_M2[n])
                                Q2[idx].append(_Q2[m])


        nrows,ncols=1,2
        fig = py.figure(figsize=(ncols*8,nrows*5))
        ax11 = py.subplot(nrows,ncols,1)
        ax12 = py.subplot(nrows,ncols,2)

        hand = {}

        for idx in data:
            if   idx in [2000]: color = 'red'
            elif idx in [2001]: color = 'blue'
            elif idx in [2002]: color = 'green'

            hand[idx] = ax11.scatter(Z1[idx], M1[idx], c=color, s=10, marker='o')
            hand[idx] = ax11.scatter(Z1[idx], M1[idx], c=color, s=10, marker='o')
            hand[idx] = ax12.scatter(Z2[idx], M2[idx], c=color, s=10, marker='o')
            hand[idx] = ax12.scatter(Z2[idx], M2[idx], c=color, s=10, marker='o')


        Z = np.linspace(0.1,1,100)
        Mlim = RS/2 * Z
        hand['Mlim'] ,= ax11.plot(Z,Mlim,color='black')
        hand['Mlim'] ,= ax12.plot(Z,Mlim,color='black')
        
        for ax in [ax11,ax12]:
            ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
            ax.xaxis.set_tick_params(which='major',length=6)
            ax.xaxis.set_tick_params(which='minor',length=3)
            ax.yaxis.set_tick_params(which='major',length=6)
            ax.yaxis.set_tick_params(which='minor',length=3)
            ax.xaxis.set_label_coords(0.90,-0.02)
            ax.tick_params(axis='x',which='major',pad=8)

        ax11.set_xlabel(r'\boldmath$z$',size=30)
        ax12.set_xlabel(r'\boldmath$\bar{z}$',size=30)
   
        ax11.set_xlim(0.15,1.00)
        ax12.set_xlim(0.15,1.00)
        #minorLocator = MultipleLocator(0.01)
        #ax11.xaxis.set_minor_locator(minorLocator)

        ax11.set_ylim(0.20,2.10)
        ax12.set_ylim(0.20,2.10)
        #minorLocator = MultipleLocator(0.01)
        #majorLocator = MultipleLocator(0.02)
        #ax11.yaxis.set_minor_locator(minorLocator)
        #ax11.yaxis.set_major_locator(majorLocator)
    
    
        ax11.set_ylabel(r'\boldmath$M_h~[{\rm GeV}]$'  ,size=30)
        ax12.set_ylabel(r'\boldmath$\overline{M}_h~[{\rm GeV}]$'  ,size=30)
    
        #ax[1].axhline(0,0,1,color='black',ls='--',alpha=0.5)
        #ax[2].axhline(0,0,1,color='black',ls='--',alpha=0.5)
   
        #ax11.text(0.05,0.30,r'\textrm{\textbf{HERMES}}',transform=ax11.transAxes,size=30)
        #ax12.text(0.05,0.30,r'\textrm{\textbf{COMPASS}}',transform=ax12.transAxes,size=30)

        #handles,labels = [], []
        #handles.append(hand['Q2lim'])
        #handles.append(hand['W2lim'])
        #handles.append(hand['ylim'])
        #handles.append(hand['xlim'])
        #labels.append(r'$Q^2 > %d ~ {\rm GeV}^2$'%(Q2limH)) 
        #labels.append(r'$W^2 > %d ~ {\rm GeV}^2$'%(W2minH)) 
        #labels.append(r'$%3.2f < y < %3.2f$'%(yminH,ymaxH)) 
        #labels.append(r'$%3.2f < x < %3.2f$'%(xminH,xmaxH)) 
        #ax11.legend(handles,labels,frameon=False,fontsize=22,loc='upper left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)
    

        py.tight_layout()
        py.subplots_adjust(hspace=0.15,wspace=0.25)
    
    
        checkdir('predictions')
        filename='predictions/SIA_kin.png'
    
        py.savefig(filename)
        print('Saving SIA plot to %s'%filename)

    def plot_sidis_kin(data): 


        X, Y, Z, M, Q2 = {},{},{},{},{}

        for idx in data:
            X[idx], Y[idx], Z[idx], M[idx], Q2[idx] = [],[],[],[],[]
            tab = data[idx] 
            col   = tab['col'][0]
            xmin  = np.array(tab['xdo'])
            xmax  = np.array(tab['xup'])
            Q20   = np.array(tab['Q2'])
            ymin  = np.array(tab['ymin'])
            ymax  = np.array(tab['ymax'])
            mmin  = np.array(tab['Mmin'])
            mmax  = np.array(tab['Mmax'])
            zmin  = np.array(tab['zmin'])
            zmax  = np.array(tab['zmax'])
            W2min = np.array(tab['W2min'])
            Q2min = np.array(tab['Q2min'])
            S     = np.array(tab['S'])
            l = len(S)

            M2  = 0.938**2

            N = 20
            xgX, wgX = np.polynomial.legendre.leggauss(N)
            xgY, wgY = np.polynomial.legendre.leggauss(N)
            xgZ, wgZ = np.polynomial.legendre.leggauss(N)
            xgM, wgM = np.polynomial.legendre.leggauss(N)

            for m in range(l):
                #--Gaussian quadrature integration over z
                _Z = 0.5*(zmax[m]-zmin[m])*xgZ + 0.5*(zmax[m]+zmin[m])
                jac_Z = 0.5*(zmax[m]-zmin[m])
                Zbin = zmax[m]-zmin[m]

                #--Gaussian quadrature integration over Y
                ymin1 = ymin[m]
                ymin2 = Q2min[m]/xmax[m]/(S[m]-M2)
                ymin3 = (W2min[m]-M2)/(1-xmin[m])/(S[m]-M2)
                yMIN  = max(ymin1,ymin2,ymin3)
                _Y = 0.5*(ymax[m]-yMIN)*xgY + 0.5*(ymax[m]+yMIN)
                jac_Y = 0.5*(ymax[m]-yMIN)
                Ybin = ymax[m]-yMIN

                for i in range(len(xgY)):
                    #--Gaussian quadrature integration over x
                    xmin1 = xmin[m]
                    xmin2 = Q2min[m]/_Y[i]/(S[m]-M2)
                    #xMIN  = max(xmin1,xmin2)
                    xMIN  = np.log(max(xmin1,xmin2))
                    xmax1 = xmax[m]
                    xmax2 = (_Y[i]*(S[m]-M2) - W2min[m] + M2)/_Y[i]/(S[m]-M2)
                    #xMAX = min(xmax1,xmax2)
                    xMAX = np.log(min(xmax1,xmax2))
                    #_X = 0.5*(xMAX-xMIN)*xgX + 0.5*(xMAX+xMIN)
                    _X = np.exp(0.5*(xMAX-xMIN)*xgX + 0.5*(xMAX+xMIN))
                    jac_X = 0.5*(xMAX-xMIN)
                    Xbin  = xMAX-xMIN

                    for j in range(len(xgX)):

                        for k in range(len(xgZ)):
                            #--Gaussian quadrature integration over M (upper limit depends on Z and Y)
                            Mmax1 = mmax[m]
                            Mmax2 = np.sqrt(_Z[k]*_Y[i]*S[m] - M2)
                            Mmax  = min(Mmax1,Mmax2)
                            #if Mmax2 < Mmax1: print(idx,Mmax1,Mmax2,_Z[k],_Y[i])
                            Mmin  = mmin[m]
                            _M = 0.5*(Mmax-Mmin)*xgM + 0.5*(Mmax+Mmin)
                            jac_M = 0.5*(Mmax-Mmin)
                            Mbin = Mmax-Mmin

                            for n in range(len(xgM)):
                                if   xMIN > xMAX: _flag = 0.0
                                elif Mmin > Mmax: _flag = 0.0
                                else:             _flag = 1.0
                                if _flag==0: continue
                                _Q2 = _X[j]*_Y[i]*(S[m]-M2)
                                Y[idx] .append(_Y[i])
                                X[idx] .append(_X[j])
                                Z[idx] .append(_Z[k])
                                M[idx] .append(_M[n])
                                Q2[idx].append(_Q2)


        nrows,ncols=2,2
        fig = py.figure(figsize=(ncols*8,nrows*5))
        ax11 = py.subplot(nrows,ncols,1)
        ax12 = py.subplot(nrows,ncols,2)
        ax21 = py.subplot(nrows,ncols,3)
        ax22 = py.subplot(nrows,ncols,4)

        hand = {}

        for idx in data:
            if   idx in [3000,3001,3002]: color = 'red'
            elif idx in [3010,3011,3012]: color = 'blue'
            elif idx in [3020,3021,3022]: color = 'green'

            if idx in [3000,3010,3020]: ax = ax11
            else:                       ax = ax12

            hand[idx] = ax.scatter(X[idx], Q2[idx], c=color, s=10, marker='o')
            hand[idx] = ax.scatter(X[idx], Q2[idx], c=color, s=10, marker='o')

            if idx in [3000,3010,3020]: ax = ax21
            else:                       ax = ax22

            ax.scatter(Z[idx], M[idx], c=color, s=10, marker='o')
            ax.scatter(Z[idx], M[idx], c=color, s=10, marker='o')


        X = np.linspace(0,1,100)
        SH    = data[3000]['S'][0]
        SC    = data[3001]['S'][0]
        ymaxH = data[3000]['ymax'][0]
        ymaxC = data[3001]['ymax'][0]

        ylimH = X*ymaxH*(SH-M2)
        ylimC = X*ymaxC*(SC-M2)

        hand['ylim'] ,= ax11.plot(X,ylimH,color='green')
        hand['ylim'] ,= ax12.plot(X,ylimC,color='green')

        yminH = data[3000]['ymin'][0]
        yminC = data[3001]['ymin'][0]

        ylimH = X*yminH*(SH-M2)
        ylimC = X*yminC*(SC-M2)

        hand['ylim'] ,= ax11.plot(X,ylimH,color='green')
        hand['ylim'] ,= ax12.plot(X,ylimC,color='green')


        Q2limH = data[3000]['Q2min'][0]
        Q2limC = data[3001]['Q2min'][0]

        hand['Q2lim'] ,= ax11.plot(Q2limH*np.ones(len(X)),color='magenta')
        hand['Q2lim'] ,= ax12.plot(Q2limC*np.ones(len(X)),color='magenta')

        W2minH = data[3000]['W2min'][0]
        W2minC = data[3001]['W2min'][0]

        W2limH = X/(1-X)*(W2minH-M2)
        W2limC = X/(1-X)*(W2minC-M2)

        hand['W2lim'] ,= ax11.plot(X,W2limH,color='black')
        hand['W2lim'] ,= ax12.plot(X,W2limC,color='black')

        xminH = data[3010]['xdo'][0]
        xminC = data[3011]['xdo'][0]

        hand['xlim'] = ax11.axvline(xminH,0,0.25,color='black',alpha=0.5,ls=':')
        hand['xlim'] = ax12.axvline(xminC,0,0.25,color='black',alpha=0.5,ls=':')
        
        xmaxH = data[3010]['xup'][0]
        xmaxC = data[3011]['xup'][0]

        hand['xlim'] = ax11.axvline(xmaxH,0,1.00,color='black',alpha=0.5,ls=':')
        hand['xlim'] = ax12.axvline(xmaxC,0,1.00,color='black',alpha=0.5,ls=':')
   
        Z = np.linspace(0.2,1.0,100)
        MlimH = np.sqrt(Z*yminH*SH - M2)
        MlimC = np.sqrt(Z*yminC*SC - M2)

        #hand['Mlim'] ,= ax21.plot(X,MlimH,color='black')
        #hand['Mlim'] ,= ax22.plot(X,MlimC,color='black')
 
        for ax in [ax11,ax12]:
            ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
            ax.xaxis.set_tick_params(which='major',length=6)
            ax.xaxis.set_tick_params(which='minor',length=3)
            ax.yaxis.set_tick_params(which='major',length=6)
            ax.yaxis.set_tick_params(which='minor',length=3)
            ax.set_xlabel(r'\boldmath$x$',size=30)
            ax.xaxis.set_label_coords(0.95,-0.02)
            ax.tick_params(axis='x',which='major',pad=8)
            ax.semilogx()
            ax.semilogy()

        for ax in [ax21,ax22]:
            ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
            ax.xaxis.set_tick_params(which='major',length=6)
            ax.xaxis.set_tick_params(which='minor',length=3)
            ax.yaxis.set_tick_params(which='major',length=6)
            ax.yaxis.set_tick_params(which='minor',length=3)
            ax.set_xlabel(r'\boldmath$z$',size=30)
            ax.xaxis.set_label_coords(0.95,-0.02)
            ax.tick_params(axis='x',which='major',pad=8)

   
        ax11.set_xlim(1e-2,0.50)
        ax12.set_xlim(2e-3,0.80)
        ax21.set_xlim(0.15,1.00)
        ax22.set_xlim(0.15,1.00)
        #minorLocator = MultipleLocator(0.01)
        #ax11.xaxis.set_minor_locator(minorLocator)

 
        #ax[5].tick_params(labelbottom=True)
        #ax[6].tick_params(labelbottom=True)

        ax11.set_ylim(0.9,30)
        ax12.set_ylim(0.9,200)
        ax21.set_ylim(0.20,2.10)
        ax22.set_ylim(0.20,2.10)
        #minorLocator = MultipleLocator(0.01)
        #majorLocator = MultipleLocator(0.02)
        #ax11.yaxis.set_minor_locator(minorLocator)
        #ax11.yaxis.set_major_locator(majorLocator)
    
    
        ax11.set_ylabel(r'\boldmath$Q^2~[{\rm GeV}^2]$',size=30)
        ax21.set_ylabel(r'\boldmath$M_h~[{\rm GeV}]$'  ,size=30)
    
        #ax[1].axhline(0,0,1,color='black',ls='--',alpha=0.5)
        #ax[2].axhline(0,0,1,color='black',ls='--',alpha=0.5)
   
        ax11.text(0.05,0.30,r'\textrm{\textbf{HERMES}}',transform=ax11.transAxes,size=30)
        ax12.text(0.05,0.30,r'\textrm{\textbf{COMPASS}}',transform=ax12.transAxes,size=30)

        handles,labels = [], []
        handles.append(hand['Q2lim'])
        handles.append(hand['W2lim'])
        handles.append(hand['ylim'])
        handles.append(hand['xlim'])
        labels.append(r'$Q^2 > %d ~ {\rm GeV}^2$'%(Q2limH)) 
        labels.append(r'$W^2 > %d ~ {\rm GeV}^2$'%(W2minH)) 
        labels.append(r'$%3.2f < y < %3.2f$'%(yminH,ymaxH)) 
        labels.append(r'$%3.2f < x < %3.2f$'%(xminH,xmaxH)) 
        ax11.legend(handles,labels,frameon=False,fontsize=22,loc='upper left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)
    
        handles,labels = [], []
        handles.append(hand['Q2lim'])
        handles.append(hand['W2lim'])
        handles.append(hand['ylim'])
        handles.append(hand['xlim'])
        labels.append(r'$Q^2 > %d ~ {\rm GeV}^2$'%(Q2limC)) 
        labels.append(r'$W^2 > %d ~ {\rm GeV}^2$'%(W2minC)) 
        labels.append(r'$%3.2f < y < %3.2f$'%(yminC,ymaxC)) 
        labels.append(r'$%4.3f < x < %3.2f$'%(xminC,xmaxC)) 
        ax12.legend(handles,labels,frameon=False,fontsize=22,loc='upper left',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

        #handles,labels = [], []
        #handles.append(hand[3000])
        #handles.append(hand[3010])
        #handles.append(hand[3020])
        #labels.append(r'$x-{\rm bin}$') 
        #labels.append(r'$M-{\rm bin}$') 
        #labels.append(r'$z-{\rm bin}$') 
        #ax12.legend(handles,labels,frameon=False,fontsize=22,loc='lower right',handletextpad = 0.2, handlelength = 1.0, labelspacing=0.7)

    
        py.tight_layout()
        py.subplots_adjust(hspace=0.15,wspace=0.10)
    
    
        checkdir('predictions')
        filename='predictions/SIDIS_kin.png'
    
        py.savefig(filename)
        print('Saving SIDIS plot to %s'%filename)


    from fitlib.resman import RESMAN
    
    #--predictions on SIDIS
    #--load DiFFs D1 and H1 for prediction purposes
    wdir = '/work/JAM/ccocuzza/diffs/results/step06'

    data = load_data()

    plot_sia_kin(data)

    sys.exit()

    gen_data(wdir,data)

    filename = 'data/SIDIS.dat'
    thy = load(filename)

    plot_sidis_x(thy)
    plot_sidis_M(thy)
    plot_sidis_z(thy)





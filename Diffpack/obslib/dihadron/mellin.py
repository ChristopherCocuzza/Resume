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
from termcolor import colored

#--from scipy
from scipy.interpolate import griddata, RegularGridInterpolator, Rbf
from scipy.integrate import fixed_quad
from scipy.special import gamma

#--from tools
from tools.multiproc import MULTIPROC
from tools.tools import load, save, checkdir, lprint
from tools.config import conf
from tools.parallel import PARALLEL 

#--from qcdlib
from qcdlib import aux,mellin,alphaS,eweak
import lhapdf

version = int(sys.version[0])

#--local
from obslib.dihadron.theory   import THEORY
from obslib.dihadron.reader   import READER
from obslib.dihadron.aux      import AUX

  
class MELLIN(THEORY):
 
    def __init__(self,gen=False):
        THEORY.__init__(self)
        self.gen = gen
        self.parts = ['NM','NCM']
        #--M grid
        self.M0 = {}
        self.M0['UU'] = np.array([0.28, 0.40, 0.50, 0.70, 0.75, 0.80, 0.90, 1.00, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00])
        self.M0['UT'] = np.array([0.28, 0.50, 0.70, 0.85, 1.00, 1.20, 1.60, 2.00])

        if 'integrate' in conf: self.integrate = conf['integrate']
        else:                   self.integrate = True

        if 'update_unpol' in conf: self.update_unpol = conf['update_unpol']
        else:                      self.update_unpol = True
        self.DEN = {}

        self.setup()

    def setup(self):
        if self.gen==False:
            self.load_melltab()
            self.mellin = conf['mellin']
            self.setup_interpolation()
          
    def load_melltab(self):
        self.mtab={}
        path2dihadrontab=conf['path2dihadrontab']
        tabs = conf['dihadron tabs']

        #--check for RS values
        RS = []
        for idx in tabs:
            if tabs[idx]['process'][0] != 'pp': continue
            RS.append(tabs[idx]['RS'][0])

        self.RS   = np.unique(RS)

        for rs in self.RS:
            self.mtab[rs]={}
            self.mtab[rs]['UU']={}
            self.mtab[rs]['UT']={}
            directUT   = '%s/RS%s/UT' %(path2dihadrontab,int(rs))
            directUU   = '%s/RS%s/UU' %(path2dihadrontab,int(rs))
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
                    if 'sign_change' in conf and conf['sign_change']:
                        if channel in ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ']: sign = -1
                        else: sign = 1
                    else: sign = 1
                    result += sign * np.real(phase2*np.einsum(ind, lum , W, W, sigma) -\
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

    #--grid generation
    def gen_SIGNM_double(self,N,M,RS,eta,PhT,muF2,channel,flav,msg='NM'):
        pts=len(N)
        SIGNM=np.zeros((pts,pts),dtype=complex)
        #--note: value of Mh does not matter
        Mh = 1.0
        for i in range(pts): 
            for j in range(pts):
                lprint('Generating grids for %s %s %s: [%s/%s]'%(channel,flav,msg,i*pts+j,pts**2)) 
                self.n=N[i]
                self.m=M[j]
                real=self.integrate(RS,PhT,Mh,eta,'UT',ilum='mell-real',channel=channel,flav=flav)
                imag=self.integrate(RS,PhT,Mh,eta,'UT',ilum='mell-imag',channel=channel,flav=flav)
                SIGNM[i,j]=np.complex(real,imag)
        return SIGNM

    def gen_SIGNM_single(self,N,RS,eta,PhT,muF2,channel,flav):
        pts=len(N)
        SIGN=np.zeros(pts,dtype=complex)
        #--note: value of Mh does not matter
        Mh = 1.0
        for i in range(pts): 
            lprint('Generating grids for %s %s: [%s/%s]'%(channel,flav,i,pts)) 
            self.n=N[i]
            real=self.integrate(RS,PhT,Mh,eta,'UU',ilum='mell-real',channel=channel,flav=flav)
            imag=self.integrate(RS,PhT,Mh,eta,'UU',ilum='mell-imag',channel=channel,flav=flav)
            SIGN[i]=np.complex(real,imag)
        return SIGN

    def _gen_melltab(self,xsec,channel,RS,eta,PhT,muF2):
        N =conf['mellin'].N
        NC=np.conjugate(N)
        data={}
        if xsec=='UT':
            if   channel in ['QQ,QQ','QQp,QpQ']:   flavs = ['u','d','s','c','ub','db','sb','cb']
            elif channel in ['QQB,QQB','QQB,QBQ']: flavs = ['u','d','s','c','ub','db','sb','cb']
            elif channel in ['GQ,QG']:             flavs = ['g']
        if xsec=='UU':
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
       
        #--double Mellin transform for UT
        if xsec=='UT': 
            for flav in flavs:
                data[flav] = {}
                data[flav]['NM']  = self.gen_SIGNM_double(N ,N,RS,eta,PhT,muF2,channel,flav,msg='NC')
                data[flav]['NCM'] = self.gen_SIGNM_double(NC,N,RS,eta,PhT,muF2,channel,flav,msg='NCM')
        #--single Mellin transform for UU
        if xsec=='UU': 
            for flav in flavs:
                data[flav]        = self.gen_SIGNM_single(N,   RS,eta,PhT,muF2,channel,flav)
        return data

    def gen_melltab(self,xsec,channel,RS,eta,PhT,name):

        path2dihadrontab='%s/RS%s'%(conf['path2dihadrontab'],int(RS))
        checkdir(path2dihadrontab)
        checkdir('%s/%s'%(path2dihadrontab,xsec))
                
        muF2 = self.muF2(PhT)
        mtab=self._gen_melltab(xsec,channel,RS,eta,PhT,muF2)
        mtab['RS']  = RS
        mtab['eta'] = eta
        mtab['PhT'] = PhT
        fname='%s/%s/%s-%s.melltab'%(path2dihadrontab,xsec,name,channel)
        save(mtab,fname)


  

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

#--test Mellin transform with no interpolation
def test_mellin(wdir,ng=20,gen_xspace=False):

    RS,PhTmin,PhTmax = 200, 2.5, 15
    #RS,PhTmin,PhTmax = 500, 4.0, 19

    ETA = np.linspace(-1 ,1 ,5)
    PHT = np.linspace(PhTmin,PhTmax,5)
    MH  = np.linspace(0.28,2.0,10)

    ETA, PHT, MH = np.meshgrid(ETA,PHT,MH,indexing='ij',sparse=False)

    ETA = ETA.flatten()
    PHT = PHT.flatten()
    MH  = MH.flatten()

    L = len(ETA)

    #--make up data sheets
    idxs = [0]
    conf['dihadron tabs'] = {}
    for i in idxs:
        conf['dihadron tabs'][i] = {}
        conf['dihadron tabs'][i]['col']     = ['test']
        conf['dihadron tabs'][i]['process'] = ['pp']
        conf['dihadron tabs'][i]['hadrons'] = ['pi+,pi-']
        conf['dihadron tabs'][i]['tar']     = ['pp']
        conf['dihadron tabs'][i]['obs']     = ['A_UT']
    #--fixed eta, PhT, and M to match grids
    conf['dihadron tabs'][0]['RS']      = np.array([RS]*L)
    conf['dihadron tabs'][0]['M']       = np.array(MH)
    conf['dihadron tabs'][0]['PhT']     = np.array(PHT)
    conf['dihadron tabs'][0]['eta']  = np.array(ETA)


    conf['aux'] = AUX()

    MELL = MELLIN()

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

    tabs = conf['dihadron tabs']
    #--get Mellin space result
    t1 = time.time()
    for task in MELL.tasks: MELL.process_request(task)
    for task in MELL.tasks: MELL.update(task)
    mell = {}
    mell['thy'] = {}
    mell['num'] = {}
    mell['den'] = {}
    for idx in conf['dihadron tabs']:
        mell['thy'][idx],mell['num'][idx],mell['den'][idx] = MELL.get_asym(idx)
    t2 = time.time()
    t  = t2 - t1
    print('Time to get Mellin results: %3.2f'%t)

    #--get x-space result
    if gen_xspace==True:
        THY = THEORY(ng=ng)
        xspace = {}
        xspace['thy'] = {}
        xspace['num'] = {}
        xspace['den'] = {}
        for idx in conf['dihadron tabs']:
            xspace['thy'][idx],xspace['num'][idx],xspace['den'][idx] = THY.get_asym(idx)

        #--save data for comparison between different values of ng
        filename = 'data/test_mellin-ng=%d.dat'%(ng)
        checkdir('data')
        save(xspace,filename)
        print('saving data to %s'%filename)
    else:
        filename = 'data/test_mellin-ng=%d.dat'%(ng)
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

    factor = 1e8

    #--print results
    for idx in tabs:
        print('idx: %d, RS = %d'%(idx,tabs[idx]['RS'][0]))
        for i in range(len(tabs[idx]['RS'])):
            print('data point: %d'%i)
            print('eta = %3.2f, PhT = %3.2f, M = %3.2f'%(tabs[idx]['eta'][i],tabs[idx]['PhT'][i],tabs[idx]['M'][i]))
            print('xspace result (asym) (ng = %d): %7.6f'%(ng,xspace['thy'][idx][i]))
            print('mellin result (asym) : %7.6f'%(mell['thy'][idx][i]))
            print('percent error (asym) : %10.9f'%(err['thy'][idx][i]))
            print('xspace result (num) (ng = %d): %7.6f'%(ng,xspace['num'][idx][i]*factor))
            print('mellin result (num) : %7.6f'%(mell['num'][idx][i]*factor))
            print('percent error (num) : %10.9f'%(err['num'][idx][i]))
            print('xspace result (den) (ng = %d): %7.6f'%(ng,xspace['den'][idx][i]*factor))
            print('mellin result (den) : %7.6f'%(mell['den'][idx][i]*factor))
            print('percent error (den) : %10.9f'%(err['den'][idx][i]))
            print()

#--plot against x-space calculation
def test_interp(wdir,RS,ng=20,gen_xspace=False):

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


    #--grid points: M = 0.7, PhT = 5, eta = 0.333

    MELL = MELLIN()

    conf['aux'] = AUX()

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

    #--get Mellin space result
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

    #--get x-space result
    if gen_xspace==True:
        t1 = time.time()
        THY = THEORY(ng=ng)
        xspace = {}
        xspace['thy'] = {}
        xspace['num'] = {}
        xspace['den'] = {}
        for idx in conf['dihadron tabs']:
            xspace['thy'][idx],xspace['num'][idx],xspace['den'][idx] = THY.get_asym(idx)
        t2 = time.time()
        t  = t2 - t1
        print('Time to generate x-space results: %3.2f'%t)

        #--save data for comparison between different values of ng
        filename = 'data/test_interp-RS%d-ng=%d.dat'%(RS,ng)
        checkdir('data')
        save(xspace,filename)
        print('saving data to %s'%filename)
    else:
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

    #--plot z and Mh
    for idx in tabs:

        if idx==0: ax = ax11
        if idx==1: ax = ax12
        if idx==2: ax = ax13

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']   ,= ax.plot(M,xspace['thy'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(M,mell  ['thy'][idx],color='darkblue')
        if idx==1:
            PhT = tabs[idx]['PhT']
            hand['xspace']   ,= ax.plot(PhT,xspace['thy'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(PhT,mell  ['thy'][idx],color='darkblue')
        if idx==2:
            eta = tabs[idx]['eta']
            hand['xspace']   ,= ax.plot(eta,xspace['thy'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(eta,mell  ['thy'][idx],color='darkblue')

        factor = 1e4

        if idx==0: ax = ax21
        if idx==1: ax = ax22
        if idx==2: ax = ax23

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']   ,= ax.plot(M,factor*xspace['num'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(M,factor*mell  ['num'][idx],color='darkblue')
        if idx==1:
            PhT = tabs[idx]['PhT']
            hand['xspace']   ,= ax.plot(PhT,factor*xspace['num'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(PhT,factor*mell  ['num'][idx],color='darkblue')
        if idx==2:
            eta = tabs[idx]['eta']
            hand['xspace']   ,= ax.plot(eta,factor*xspace['num'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(eta,factor*mell  ['num'][idx],color='darkblue')

        if idx==0: ax = ax31
        if idx==1: ax = ax32
        if idx==2: ax = ax33

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']   ,= ax.plot(M,factor*xspace['den'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(M,factor*mell  ['den'][idx],color='darkblue')
        if idx==1:
            PhT = tabs[idx]['PhT']
            hand['xspace']   ,= ax.plot(PhT,factor*xspace['den'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(PhT,factor*mell  ['den'][idx],color='darkblue')
        if idx==2:
            eta = tabs[idx]['eta']
            hand['xspace']   ,= ax.plot(eta,factor*xspace['den'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(eta,factor*mell  ['den'][idx],color='darkblue')

    for ax in [ax11,ax21,ax31]:
        ax.set_xlim(0.25,2.1)
        ax.set_xticks([0.4,0.8,1.2,1.6,2.0])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)

    for ax in [ax12,ax22,ax32]: 
        ax.set_xlim(2,22)
        ax.set_xticks([5,10,15,20])
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
        ax11.text(0.15,0.15,r'\boldmath$\eta=%3.2f$'%tabs[0]['eta'][0] ,transform=ax11.transAxes, size=25)
    else:
        ax11.text(0.15,0.15,r'\boldmath$%3.2f < \eta < %3.2f$'%(tabs[0]['etamin'][0],tabs[0]['etamax'][0]) ,transform=ax11.transAxes, size=25)
    if 'PhT' in tabs[0]:
        ax11.text(0.15,0.05,r'\boldmath$P_{hT} = %3.2f ~ {\rm GeV}$'%(tabs[0]['PhT'][0]) ,transform=ax11.transAxes, size=25)
    else:
        ax11.text(0.15,0.05,r'\boldmath$%3.2f < P_{hT} < %3.2f ~ {\rm GeV}$'%(tabs[0]['PhTmin'][0],tabs[0]['PhTmax'][0]) ,transform=ax11.transAxes, size=25)

    if 'eta' in tabs[1]:
        ax12.text(0.05,0.15,r'\boldmath$\eta=%3.2f$'%tabs[1]['eta'][0] ,transform=ax12.transAxes, size=25)
    else:
        ax12.text(0.05,0.15,r'\boldmath$%3.2f < \eta < %3.2f$'%(tabs[1]['etamin'][0],tabs[1]['etamax'][0]) ,transform=ax12.transAxes, size=25)
    if 'M' in tabs[1]:
        ax12.text(0.05,0.05,r'\boldmath$M_h = %3.2f ~ {\rm GeV}$'%(tabs[1]['M'][0]) ,transform=ax12.transAxes, size=25)
    else:
        ax12.text(0.05,0.05,r'\boldmath$%3.2f < M_h < %3.2f ~ {\rm GeV}$'%(tabs[1]['Mmin'][0],tabs[1]['Mmax'][0]) ,transform=ax12.transAxes, size=25)


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
    labels.append(r'\textrm{\textbf{xspace}}')
    labels.append(r'\textrm{\textbf{mellin+interp}}')
    ax23.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.14)
    filename='gallery/test_interp_RS%d'%RS
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

#--test how to make PhT interp easier
def test_PhT_interp(wdir,ng=20,gen_xspace=False):

    #RS,PhTmin,PhTmax = 200, 2.5, 15
    RS,PhTmin,PhTmax = 500, 4.0, 19
    #--make up data sheets
    L = 30
    idxs = [0]
    conf['dihadron tabs'] = {}
    for i in idxs:
        conf['dihadron tabs'][i] = {}
        conf['dihadron tabs'][i]['col']     = ['test']
        conf['dihadron tabs'][i]['process'] = ['pp']
        conf['dihadron tabs'][i]['hadrons'] = ['pi+,pi-']
        conf['dihadron tabs'][i]['tar']     = ['pp']
        conf['dihadron tabs'][i]['obs']     = ['A_UT']
        conf['dihadron tabs'][i]['RS']      = np.array([RS]*L)
    
    #--integrate over eta, fixed M, plot as function of PhT
    conf['dihadron tabs'][0]['PhT']     = np.linspace(PhTmin,PhTmax,L)
    conf['dihadron tabs'][0]['M']       = np.array([0.80]*L)
    conf['dihadron tabs'][0]['etamin']  = np.array([0.0]*L)
    conf['dihadron tabs'][0]['etamax']  = np.array([1.0]*L)

    ##--fixed eta, M, function of PhT
    #conf['dihadron tabs'][0]['PhT']     = np.linspace(PhTmin,PhTmax,L)
    #conf['dihadron tabs'][0]['M']       = np.array([0.700]*L)
    #conf['dihadron tabs'][0]['eta']     = np.array([0.333]*L)


    #--grid points: M = 0.7, PhT = 5, eta = 0.333

    MELL = MELLIN()

    conf['aux'] = AUX()

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

    #--get Mellin space result
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

    #--get x-space result
    if gen_xspace==True:
        THY = THEORY(ng=ng)
        xspace = {}
        xspace['thy'] = {}
        xspace['num'] = {}
        xspace['den'] = {}
        for idx in conf['dihadron tabs']:
            xspace['thy'][idx],xspace['num'][idx],xspace['den'][idx] = THY.get_asym(idx)

        #--save data for comparison between different values of ng
        filename = 'data/test3-ng=%d.dat'%(ng)
        checkdir('data')
        save(xspace,filename)
        print('saving data to %s'%filename)
    else:
        filename = 'data/test3-ng=%d.dat'%(ng)
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
    ax21=py.subplot(nrows,ncols,4)
    ax31=py.subplot(nrows,ncols,7)
    ax12=py.subplot(nrows,ncols,2)
    ax22=py.subplot(nrows,ncols,5)
    ax32=py.subplot(nrows,ncols,8)

    tabs = conf['dihadron tabs']

    hand = {}

    #--plot z and Mh
    for idx in tabs:

        if idx==0: ax = ax11

        if idx==0:
            PhT = tabs[idx]['PhT']
            hand['xspace']   ,= ax.plot(PhT,xspace['thy'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(PhT,mell  ['thy'][idx],color='darkblue')

        factor = 1e8

        if idx==0: ax = ax21

        if idx==0:
            PhT = tabs[idx]['PhT']
            hand['xspace']   ,= ax.plot(PhT,factor*xspace['num'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(PhT,factor*mell  ['num'][idx],color='darkblue')

        if idx==0: ax = ax31

        if idx==0:
            PhT = tabs[idx]['PhT']
            hand['xspace']   ,= ax.plot(PhT,factor*xspace['den'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(PhT,factor*mell  ['den'][idx],color='darkblue')


    PhT = tabs[0]['PhT']

    factor = PhT**8

    ax22.plot(PhT,xspace['num'][0]*factor,color='firebrick')
    ax32.plot(PhT,xspace['den'][0]*factor,color='firebrick')

    for ax in [ax11,ax21,ax31,ax12,ax22,ax32]: 
        ax.set_xlim(2,22)
        ax.set_xticks([5,10,15,20])
        minorLocator = MultipleLocator(1)
        ax.xaxis.set_minor_locator(minorLocator)



    ax31.set_xlabel(r'\boldmath$P_{hT}~[{\rm GeV}]$',size=30)

    ax11.set_ylabel(r'\boldmath$A_{U \perp}$',size=30)
    ax21.set_ylabel(r'\boldmath$\Delta \sigma \times 10^8$',size=30)
    ax31.set_ylabel(r'\boldmath$       \sigma \times 10^8$',size=30)

    for ax in [ax11,ax21,ax31,ax12,ax22,ax32]:
        ax .tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax .tick_params(axis='both',which='minor',size=4)
        ax .tick_params(axis='both',which='major',size=8)
        ax .axhline(0,0,1,color='black',ls='--',alpha=0.5)


    if 'eta' in tabs[0]:
        ax11.text(0.05,0.15,r'\boldmath$\eta=%3.2f$'%tabs[0]['eta'][0] ,transform=ax11.transAxes, size=25)
    else:
        ax11.text(0.05,0.15,r'\boldmath$%3.2f < \eta < %3.2f$'%(tabs[0]['etamin'][0],tabs[0]['etamax'][0]) ,transform=ax11.transAxes, size=25)
    if 'M' in tabs[0]:
        ax11.text(0.05,0.05,r'\boldmath$M_h = %3.2f ~ {\rm GeV}$'%(tabs[0]['M'][0]) ,transform=ax11.transAxes, size=25)
    else:
        ax11.text(0.05,0.05,r'\boldmath$%3.2f < M_h < %3.2f ~ {\rm GeV}$'%(tabs[0]['Mmin'][0],tabs[0]['Mmax'][0]) ,transform=ax11.transAxes, size=25)

    ax11.text(0.05,0.60,r'\boldmath$\sqrt{s} = %d ~ {\rm GeV}$'%(tabs[0]['RS'][0]) ,transform=ax11.transAxes, size=25)
    fs = 30

    handles,labels = [], []
    handles.append(hand['xspace'])
    handles.append(hand['mellin'])
    labels.append(r'\textrm{\textbf{xspace}}')
    labels.append(r'\textrm{\textbf{mellin+interp}}')
    ax21.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.14)
    filename='gallery/test_PhT_interp'
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

#--test how to make Mh interp easier
def test_Mh_interp(wdir,ng=20,gen_xspace=False):

    #RS,PhTmin,PhTmax = 200, 2.5, 15
    RS,PhTmin,PhTmax = 500, 4.0, 19
    #--make up data sheets
    L = 30
    idxs = [0]
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
    conf['dihadron tabs'][0]['M']       = np.array(np.linspace(0.28,2,30))
    conf['dihadron tabs'][0]['PhT']     = np.array([6.00]*L)
    conf['dihadron tabs'][0]['etamin']  = np.array([0.0]*L)
    conf['dihadron tabs'][0]['etamax']  = np.array([1.0]*L)

    MELL = MELLIN()

    conf['aux'] = AUX()

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

    #--get Mellin space result
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

    #--get x-space result
    if gen_xspace==True:
        THY = THEORY(ng=ng)
        xspace = {}
        xspace['thy'] = {}
        xspace['num'] = {}
        xspace['den'] = {}
        for idx in conf['dihadron tabs']:
            xspace['thy'][idx],xspace['num'][idx],xspace['den'][idx] = THY.get_asym(idx)

        #--save data for comparison between different values of ng
        filename = 'data/test_Mh-ng=%d.dat'%(ng)
        checkdir('data')
        save(xspace,filename)
        print('saving data to %s'%filename)
    else:
        filename = 'data/test_Mh-ng=%d.dat'%(ng)
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
    ax21=py.subplot(nrows,ncols,4)
    ax31=py.subplot(nrows,ncols,7)
    ax12=py.subplot(nrows,ncols,2)
    ax22=py.subplot(nrows,ncols,5)
    ax32=py.subplot(nrows,ncols,8)

    tabs = conf['dihadron tabs']

    hand = {}

    for idx in tabs:

        if idx==0: ax = ax11

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']   ,= ax.plot(M,xspace['thy'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(M,mell  ['thy'][idx],color='darkblue')

        factor = 1e4

        if idx==0: ax = ax21

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']   ,= ax.plot(M,factor*xspace['num'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(M,factor*mell  ['num'][idx],color='darkblue')

        if idx==0: ax = ax31

        if idx==0:
            M = tabs[idx]['M']
            hand['xspace']   ,= ax.plot(M,factor*xspace['den'][idx],color='firebrick')
            hand['mellin']   ,= ax.plot(M,factor*mell  ['den'][idx],color='darkblue')


    M = tabs[0]['M']

    factor_num = (M-0.77)/2
    factor_den = (M-0.77)/2

    print(M)
    print(factor_num)

    ax22.plot(M,xspace['num'][0]*factor_num,color='firebrick')
    ax32.plot(M,xspace['den'][0]*factor_den,color='firebrick')
    ax22.plot(M,mell  ['num'][0]*factor_num,color='darkblue')
    ax32.plot(M,mell  ['den'][0]*factor_den,color='darkblue')

    for ax in [ax11,ax21,ax31,ax12,ax22,ax32]: 
        ax.set_xlim(0.28,2)
        ax.set_xticks([0.4,0.8,1.2,1.6])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)



    ax31.set_xlabel(r'\boldmath$M_h~[{\rm GeV}]$',size=30)

    ax11.set_ylabel(r'\boldmath$A_{U \perp}$',size=30)
    ax21.set_ylabel(r'\boldmath$\Delta \sigma \times 10^4$',size=30)
    ax31.set_ylabel(r'\boldmath$       \sigma \times 10^4$',size=30)

    for ax in [ax11,ax21,ax31,ax12,ax22,ax32]:
        ax .tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax .tick_params(axis='both',which='minor',size=4)
        ax .tick_params(axis='both',which='major',size=8)
        ax .axhline(0,0,1,color='black',ls='--',alpha=0.5)


    if 'eta' in tabs[0]:
        ax11.text(0.05,0.15,r'\boldmath$\eta=%3.2f$'%tabs[0]['eta'][0] ,transform=ax11.transAxes, size=25)
    else:
        ax11.text(0.05,0.15,r'\boldmath$%3.2f < \eta < %3.2f$'%(tabs[0]['etamin'][0],tabs[0]['etamax'][0]) ,transform=ax11.transAxes, size=25)
    if 'PhT' in tabs[0]:
        ax11.text(0.05,0.05,r'\boldmath$P_{hT} = %3.2f ~ {\rm GeV}$'%(tabs[0]['M'][0]) ,transform=ax11.transAxes, size=25)
    else:
        ax11.text(0.05,0.05,r'\boldmath$%3.2f < P_{hT} < %3.2f ~ {\rm GeV}$'%(tabs[0]['Mmin'][0],tabs[0]['Mmax'][0]) ,transform=ax11.transAxes, size=25)

    ax11.text(0.05,0.60,r'\boldmath$\sqrt{s} = %d ~ {\rm GeV}$'%(tabs[0]['RS'][0]) ,transform=ax11.transAxes, size=25)
    fs = 30

    handles,labels = [], []
    handles.append(hand['xspace'])
    handles.append(hand['mellin'])
    labels.append(r'\textrm{\textbf{xspace}}')
    labels.append(r'\textrm{\textbf{mellin+interp}}')
    ax21.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.14)
    filename='gallery/test_Mh_interp'
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

#--test Gaussian quadrature
def test_gaussquad(wdir,RS):

    if RS==200: PhTmin,PhTmax = 2.5, 15
    if RS==500: PhTmin,PhTmax = 4.0, 19
    #--make up data sheets
    L = 30
    idxs = [0]
    conf['dihadron tabs'] = {}
    for i in idxs:
        conf['dihadron tabs'][i] = {}
        conf['dihadron tabs'][i]['col']     = ['test']
        conf['dihadron tabs'][i]['process'] = ['pp']
        conf['dihadron tabs'][i]['hadrons'] = ['pi+,pi-']
        conf['dihadron tabs'][i]['tar']     = ['pp']
        conf['dihadron tabs'][i]['obs']     = ['A_UT']
        conf['dihadron tabs'][i]['RS']      = np.array([RS]*L)
    
    #--integrate over M and PhT, plot as function of eta
    conf['dihadron tabs'][0]['eta']     = np.linspace(-1,1,L)
    conf['dihadron tabs'][0]['Mmin']    = np.array([0.28]*L)
    conf['dihadron tabs'][0]['Mmax']    = np.array([2.00]*L)
    conf['dihadron tabs'][0]['PhTmin']  = np.array([PhTmin]*L)
    conf['dihadron tabs'][0]['PhTmax']  = np.array([PhTmax]*L)


    MELL = MELLIN()

    conf['aux'] = AUX()

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

    #--get Mellin space result
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

    nrows,ncols=3,3
    fig = py.figure(figsize=(ncols*9,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax21=py.subplot(nrows,ncols,4)
    ax31=py.subplot(nrows,ncols,7)
    ax12=py.subplot(nrows,ncols,2)
    ax22=py.subplot(nrows,ncols,5)
    ax32=py.subplot(nrows,ncols,8)

    tabs = conf['dihadron tabs']

    hand = {}

    #--plot z and Mh
    for idx in tabs:

        if idx==0: ax = ax11

        if idx==0:
            eta = tabs[idx]['eta']
            hand['mellin']   ,= ax.plot(eta,mell  ['thy'][idx],color='darkblue')

        factor = 1e8

        if idx==0: ax = ax21

        if idx==0:
            eta = tabs[idx]['eta']
            hand['mellin']   ,= ax.plot(eta,factor*mell  ['num'][idx],color='darkblue')

        if idx==0: ax = ax31

        if idx==0:
            eta = tabs[idx]['eta']
            hand['mellin']   ,= ax.plot(eta,factor*mell  ['den'][idx],color='darkblue')


    for ax in [ax11,ax21,ax31]: 
        ax.set_xlim(-1.1,1.1)
        ax.set_xticks([-1,-0.5,0,0.5,1])
        minorLocator = MultipleLocator(0.1)
        ax.xaxis.set_minor_locator(minorLocator)



    ax31.set_xlabel(r'\boldmath$\eta$',size=30)

    ax11.set_ylabel(r'\boldmath$A_{U \perp}$',size=30)
    ax21.set_ylabel(r'\boldmath$\Delta \sigma \times 10^8$',size=30)
    ax31.set_ylabel(r'\boldmath$       \sigma \times 10^8$',size=30)

    for ax in [ax11,ax21,ax31,ax12,ax22,ax32]:
        ax .tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        ax .tick_params(axis='both',which='minor',size=4)
        ax .tick_params(axis='both',which='major',size=8)
        ax .axhline(0,0,1,color='black',ls='--',alpha=0.5)


    ax12.text(0.05,0.80,r'\boldmath$%3.2f < P_{hT} < %3.2f ~ {\rm GeV}$'%(tabs[0]['PhTmin'][0],tabs[0]['PhTmax'][0]) ,transform=ax12.transAxes, size=25)
    ax12.text(0.05,0.70,r'\boldmath$%3.2f < M_h < %3.2f ~ {\rm GeV}$'%(tabs[0]['Mmin'][0],tabs[0]['Mmax'][0]) ,transform=ax12.transAxes, size=25)

    ax11.text(0.05,0.60,r'\boldmath$\sqrt{s} = %d ~ {\rm GeV}$'%(tabs[0]['RS'][0]) ,transform=ax11.transAxes, size=25)
    fs = 30

    handles,labels = [], []
    handles.append(hand['mellin'])
    labels.append(r'\textrm{\textbf{mellin+interp}}')
    ax21.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.14)
    filename='gallery/test_gaussquad'
    filename+='.png'

    checkdir('gallery')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

    from fitlib.resman import RESMAN

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
    conf['datasets']['dihadron']['xlsx'][5000]='dihadron/expdata/5000.xlsx'
    conf['datasets']['dihadron']['xlsx'][5001]='dihadron/expdata/5001.xlsx'
    conf['datasets']['dihadron']['xlsx'][5020]='dihadron/expdata/5020.xlsx'
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
    filename='predictions/star-RS200-ang%s-ng=%d'%(angle,ng)
    filename+='.png'

    checkdir('predictions')
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
    filename='predictions/star-RS200-ang07-M-ng=%d'%(ng)
    filename+='.png'

    checkdir('predictions')
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
    filename='predictions/star-RS200-ang07-PhT-ng=%d'%(ng)
    filename+='.png'

    checkdir('predictions')
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
    filename='predictions/star-RS200-ang07-eta-ng=%d'%(ng)
    filename+='.png'

    checkdir('predictions')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS500(thy,ng):
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

        if idx not in [5000,5001]: continue

        binned = data[idx]['binned'][0]
        bins   = np.unique(data[idx]['bin'])

        if data[idx]['eta'][0] > 0: color = 'firebrick'
        if data[idx]['eta'][0] < 0: color = 'darkblue'
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
            eta = data[idx]['eta'][ind]
            value = data[idx]['value'][ind]
            stat_u  = data[idx]['stat_u'][ind]
            syst1_u = data[idx]['syst1_u'][ind]
            syst2_u = data[idx]['syst2_u'][ind]
            alpha = np.sqrt(stat_u**2 + syst1_u**2 + syst2_u**2)

            if idx==5001: M += 0.05

            hand[idx]    = ax.errorbar(M  ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
            ax.text(0.05,0.85,r'\boldmath$\langle P_{hT} \rangle = %d~{\rm GeV}$'%pT[0] , transform=ax.transAxes, size=25)
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
    ax22.text(1.10,0.50,r'\boldmath$\sqrt{s} = 500~{\rm GeV}$' ,      transform=ax22.transAxes, size=30)

    fs = 30

    handles,labels = [], []
    handles.append(hand[5000])
    handles.append(hand[5001])
    labels.append(r'\boldmath$\eta<0$')
    labels.append(r'\boldmath$\eta>0$')
    ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)


    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.02,left=0.03)
    filename='predictions/star-RS500-ng=%d'%(ng)
    filename+='.png'

    checkdir('predictions')
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()

def plot_star_RS500_eta(thy,ng):
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11=py.subplot(nrows,ncols,1)

    data = conf['dihadron tabs']
    hand = {}

    for idx in data:

        if idx not in [5020]: continue
        ax = ax11

        color = 'firebrick'
        M       = data[idx]['M']
        pT      = data[idx]['PhT']
        eta     = data[idx]['eta']
        value   = data[idx]['value']
        stat_u  = data[idx]['stat_u']
        syst1_u = data[idx]['syst1_u']
        syst2_u = data[idx]['syst2_u']
        alpha = np.sqrt(stat_u**2 + syst1_u**2 + syst2_u**2)

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
    ax11.text(0.05,0.75,r'\boldmath$\sqrt{s} = 500~{\rm GeV}$'              ,transform=ax11.transAxes, size=20)
    ax11.text(0.05,0.65,r'\boldmath$\langle P_{hT} \rangle = 13~{\rm GeV}$' ,transform=ax11.transAxes, size=20)
    ax11.text(0.05,0.55,r'\boldmath$\langle M_h \rangle = 1 ~ {\rm GeV}$'   ,transform=ax11.transAxes, size=20)

    #handles,labels = [], []
    #handles.append(hand[5020])
    #labels.append(r'\textrm{\textbf{STAR}}')
    #ax11.legend(handles,labels,loc='upper right',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)

    py.tight_layout()
    py.subplots_adjust(wspace=0.0,hspace=0.00,left=0.10)
    filename='predictions/star-RS500-eta-ng=%d'%(ng)
    filename+='.png'

    checkdir('predictions')
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

    conf['Q20']   = 1.0**2
    conf['alphaSmode']='backward'
    conf['dglap mode']='truncated'
    conf['order']='LO'
    conf['path2dihadrontab']='%s/grids/dihadron'%os.environ['FITPACK']
    conf['aux']=aux.AUX()
    conf['mellin']=mellin.MELLIN(npts=4)
    conf['alphaS']=alphaS
    conf['eweak']=eweak.EWEAK()


    task    = args.task
    xsec    = args.xsec
    channel = args.channel
    RS      = args.nrg 
    eta     = args.eta
    PhT     = args.PhT
    name    = args.name

    #--generate Mellin grids
    if task==0:
        os.environ['LHAPDF_DATA_PATH'] = '%s/qcdlib/lhapdf'%(os.environ['FITPACK'])
        conf['LHAPDF:PDF'] = lhapdf.mkPDFs('JAM22-PDF_proton_nlo')
        MELLIN(gen=True).gen_melltab(xsec,channel,RS,eta,PhT,name)


    #--test Mellin transform
    if task==1:

        from fitlib.resman import RESMAN
        from tools.config import conf,load_config

        #--wdir to load DiFFs and TPDFs from
        wdir = '/work/JAM/ccocuzza/diffs/step07'

        ng = 20
        gen_xspace = True
        test_mellin(wdir,ng=ng,gen_xspace=gen_xspace)


    #--plot against x-space calculation
    if task==2:

        from fitlib.resman import RESMAN
        from tools.config import conf,load_config


        #--wdir to load DiFFs and TPDFs from
        wdir = '/work/JAM/ccocuzza/diffs/step07'

        RS = 500
        ng = 20
        gen_xspace = False
        test_interp(wdir,RS,ng=ng,gen_xspace=gen_xspace)

    #--test how to make PhT interp easier
    if task==3:

        from fitlib.resman import RESMAN
        from tools.config import conf,load_config

        #--wdir to load DiFFs and TPDFs from
        wdir = '/work/JAM/ccocuzza/diffs/step07'

        ng = 20
        gen_xspace = False
        test_PhT_interp(wdir,ng=ng,gen_xspace=gen_xspace)

    #--test how to make Mh interp easier
    if task==4:

        from fitlib.resman import RESMAN
        from tools.config import conf,load_config

        #--wdir to load DiFFs and TPDFs from
        wdir = '/work/JAM/ccocuzza/diffs/step07'

        ng = 20
        gen_xspace = False
        test_Mh_interp(wdir,ng=ng,gen_xspace=gen_xspace)


    #--test Gaussian quadrature
    if task==5:

        from fitlib.resman import RESMAN
        from tools.config import conf,load_config


        #--wdir to load DiFFs and TPDFs from
        wdir = '/work/JAM/ccocuzza/diffs/step07'

        RS = 200
        test_gaussquad(wdir,RS)


    #--plot against x-space calculation (real data)
    if task==6:

        from fitlib.resman import RESMAN
        from tools.config import conf,load_config

        #--wdir to load DiFFs and TPDFs from
        wdir = '/work/JAM/ccocuzza/diffs/step07'

        ng = 20
        data = load_data()
        gen_data(wdir,data,ng)

        filename = 'data/STAR-ng=%d.dat'%(ng)

        thy = load(filename)

        checkdir('predictions')
        plot_star_RS200          (thy,ng,angle=0.2)
        plot_star_RS200          (thy,ng,angle=0.3)
        plot_star_RS200          (thy,ng,angle=0.4)
        plot_star_RS200_ang07_M  (thy,ng)
        plot_star_RS200_ang07_PhT(thy,ng)
        plot_star_RS200_ang07_eta(thy,ng)
        plot_star_RS500          (thy,ng)
        plot_star_RS500_eta      (thy,ng)




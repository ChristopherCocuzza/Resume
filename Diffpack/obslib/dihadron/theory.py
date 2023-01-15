#!/usr/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
from matplotlib.ticker import MultipleLocator
import pylab as py
import numpy as np
from obslib.dihadron.reader import READER
from tools.config import conf,load_config
from tools.tools import load,save,checkdir,lprint
import time
import lhapdf
from qcdlib import alphaS
from qcdlib.aux import AUX

#--this code is based on: arxiv.org/abs/hep-ph/0409174

#--bottom quark is ignored.  Contribution was tested, found negligible.

#--STAR defines things differently than Pavia
#--switching the sign of eta fixes the difference

class THEORY():
  
    #--ng = 20 for x1, x2 integration is sufficient
    def __init__(self,ng=20,predict=False):

        if 'diffpippim'  in conf: self.D1 = conf['diffpippim']
        if 'tdiffpippim' in conf: self.H1 = conf['tdiffpippim']
        if 'tpdf'        in conf: self.h1 = conf['tpdf']
        #--take the mean
        if 'LHAPDF:PDF'  in conf: self.f1 = conf['LHAPDF:PDF'][0]

        if predict: self.h1 = conf['LHAPDF:TPDF'][0]
        self.predict = predict

        self.ng  = ng
        #--these values are sufficient for the integrations
        self.nge = 3
        self.ngM = 6
        self.ngP = 5
 
        self.xg,  self.wg  = np.polynomial.legendre.leggauss(ng)
        self.xge, self.wge = np.polynomial.legendre.leggauss(self.nge)
        self.xgM, self.wgM = np.polynomial.legendre.leggauss(self.ngM)
        self.xgP, self.wgP = np.polynomial.legendre.leggauss(self.ngP)
        self.flavs = ['u','d','s','c','ub','db','sb','cb','g']
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

        self.f1_storage = {_:{} for _ in self.flavs}
        self.D1_storage = {_:{} for _ in self.flavs}
        self.h1_storage = {_:{} for _ in self.flavs}
        self.H1_storage = {_:{} for _ in self.flavs}

        #--unpolarized channels (all are nonzero)
        self.channels_UU =      ['QQ,QQ','QQp,QpQ','QQp,QQp','QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ']
        self.channels_UU.extend(['QQB,GG','GQ,GQ','GQ,QG','QG,GQ','QG,QG','GG,GG','GG,QQB'])

        #--polarized channels (only five are nonzero)
        self.channels_UT =      ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ','GQ,QG']
 
        #--z must be written in terms of xa, xb, PhT, eta, and S
        self.z  = lambda xa, xb, RS, PhT, eta:  PhT/RS * (xa * np.exp(-eta) + xb * np.exp(eta))/(xa*xb)

        self.s  = lambda xa, xb, RS:            xa*xb*RS**2

        self.t  = lambda xa, xb, RS, PhT, eta: -PhT*RS*np.exp(-eta)*xa/self.z(xa,xb,RS,PhT,eta)
        self.u  = lambda xa, xb, RS, PhT, eta: -PhT*RS*np.exp( eta)*xb/self.z(xa,xb,RS,PhT,eta)

        #--lower limits on xa and xb
        self.xbmin = lambda xa, RS, PhT, eta:  np.exp(-eta)/(RS/PhT - np.exp( eta)/xa)
        self.xamin = lambda     RS, PhT, eta:  np.exp( eta)/(RS/PhT - np.exp(-eta)   )

        self.aS = alphaS

        self.muF2 = lambda PhT: (PhT)**2

        self.setup_lum()
        self.setup_sigUU()
        self.setup_sigUT()
        if 'dihadron tabs' in conf: self.tabs = conf['dihadron tabs']

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

    def setup_sigUU(self):

        aS = lambda PhT: self.aS.get_alphaS(PhT**2,0)

        #--need all 14 channels
        SIGUU = {}

        #--these come directly from arxiv.org/pdf/hep-ph/0409174.pdf
        SIGUU['QQ,QQ']     = lambda PhT, s, t, u: 4*np.pi*aS(PhT)**2/9 * ((s**4 + t**4 + u**4)/(s**2 * t**2 * u**2) - 8/(3*t*u)) 

        SIGUU['QQp,QpQ']   = lambda PhT, s, t, u: 4*np.pi*aS(PhT)**2/9 * ((s**2 + t**2)/(s**2 * u**2)) 

        SIGUU['QQB,QpQBp'] = lambda PhT, s, t, u: 4*np.pi*aS(PhT)**2/9 * ((t**2 + u**2)/(s**4)) 

        SIGUU['QQB,QQB']   = lambda PhT, s, t, u: 4*np.pi*aS(PhT)**2/(9 * s**4 * t**2) * (s**4 + t**4 + u**4 - 8/3*s*t*u**2) 

        SIGUU['QQB,GG']    = lambda PhT, s, t, u: 8*np.pi*aS(PhT)**2/3 * (t**2 + u**2)/(s**2) * (4/(9*t*u) - 1/s**2) 

        SIGUU['GQ,GQ']     = lambda PhT, s, t, u: np.pi*aS(PhT)**2 * (s**2 + t**2)/(s**2) * (1/t**2 - 4/(9*s*u)) 

        SIGUU['GG,GG']     = lambda PhT, s, t, u: 9*np.pi*aS(PhT)**2/8 * (s**4 + t**4 + u**4)*(s**2 + t**2 + u**2)/(s**4 * t**2 * u**2)

        SIGUU['GG,QQB']    = lambda PhT, s, t, u: 3*np.pi*aS(PhT)**2/8 * (t**2 + u**2)/s**2 * (4/(9*t*u) - 1/s**2)

        #--if switching A <-> B, switch t <-> u
        #--if switching C <-> D, switch t <-> u

        SIGUU['QQp,QQp']   = lambda PhT, s, t, u: SIGUU['QQp,QpQ'](PhT,s,u,t) 

        SIGUU['QQB,QBQ']   = lambda PhT, s, t, u: SIGUU['QQB,QQB'](PhT,s,u,t) 
 
        SIGUU['GQ,QG']     = lambda PhT, s, t, u: SIGUU['GQ,GQ'](PhT,s,u,t)
        SIGUU['QG,GQ']     = lambda PhT, s, t, u: SIGUU['GQ,GQ'](PhT,s,u,t)
        SIGUU['QG,QG']     = SIGUU['GQ,GQ'] 

        #--symmetric under t and u
        SIGUU['QQB,QBpQp'] = SIGUU['QQB,QpQBp']

        self.SIGUU = SIGUU

    #--NOTICE: the sign of all QQ terms is opposite than what is in Pavia
    def setup_sigUT(self):

        aS = lambda PhT: self.aS.get_alphaS(PhT**2,0)

        #--need all 14 channels
        SIGUT = {}

        SIGUT['QQ,QQ']     = lambda PhT, s, t, u:  8*np.pi*aS(PhT)**2/(27*s**2) * s * (3*t - u)/u**2

        SIGUT['QQp,QpQ']   = lambda PhT, s, t, u:  8*np.pi*aS(PhT)**2/(9*s**2) * t * s/u**2

        SIGUT['QQB,QQB']   = lambda PhT, s, t, u:  8*np.pi*aS(PhT)**2/(27*s**2)

        SIGUT['QQB,QBQ']   = lambda PhT, s, t, u:  8*np.pi*aS(PhT)**2/(27*s**2) * t * (3*s - u)/u**2

        SIGUT['GQ,QG']     = lambda PhT, s, t, u: -8*np.pi*aS(PhT)**2/(9*s**2) * (1 - 9*t*s/(4*u**2))

        #--all others vanish
        SIGUT['QQp,QQp']   = lambda PhT, s, t, u: 0.0 
        SIGUT['QQB,QBpQp'] = lambda PhT, s, t, u: 0.0 
        SIGUT['QQB,QpQBp'] = lambda PhT, s, t, u: 0.0 
        SIGUT['QG,QG']     = lambda PhT, s, t, u: 0.0 
        SIGUT['QG,GQ']     = lambda PhT, s, t, u: 0.0 
        SIGUT['GQ,GQ']     = lambda PhT, s, t, u: 0.0 
        SIGUT['QQB,GG']    = lambda PhT, s, t, u: 0.0 
        SIGUT['GG,QQB']    = lambda PhT, s, t, u: 0.0 
        SIGUT['GG,GG']     = lambda PhT, s, t, u: 0.0

        self.SIGUT = SIGUT

    def get_lum_UU(self,channel,xa,xb,PhT,eta,M,z):

        muF2 = self.muF2(PhT)

        f1  = self.f1
        D1  = self.D1 
 
        result = 0.0
        lum = self.lum[channel]
        for flavs in lum:
            A,B,C = flavs[0], flavs[1], flavs[2]
            keyA = '%s,%s'%(xa,muF2)
            keyB = '%s,%s'%(xb,muF2)
            if keyA not in self.f1_storage[A]: self.f1_storage[A][keyA] = f1.xfxQ2(self.ind[A],xa,muF2)/xa
            if keyB not in self.f1_storage[B]: self.f1_storage[B][keyB] = f1.xfxQ2(self.ind[B],xb,muF2)/xb
            f1A = self.f1_storage[A][keyA]
            f1B = self.f1_storage[B][keyB]
            keyC = '%s,%s,%s'%(z,M,muF2)
            if keyC not in self.D1_storage[C]: self.D1_storage[C][keyC] = D1.get_D([z],[M],[muF2],C)[0]
            D1C = self.D1_storage[C][keyC]
            result += f1A * f1B * D1C

        return result        

    def get_lum_UT(self,channel,xa,xb,PhT,eta,M,z):

        muF2 = self.muF2(PhT)

        f1  = self.f1
        h1  = self.h1
        H1  = self.H1
       
        result = 0.0
        lum = self.lum[channel]
        for flavs in lum:
            A,B,C = flavs[0], flavs[1], flavs[2]
            #--skip contributions that are zero
            if B not in ['u','d']:           continue
            if C not in ['u','d','ub','db']: continue
            keyA = '%s,%s'%(xa,muF2)
            keyB = '%s,%s'%(xb,muF2)
            if keyA not in self.f1_storage[A]: self.f1_storage[A][keyA] = f1.xfxQ2(self.ind[A],xa,muF2)/xa
            if keyB not in self.h1_storage[B]: 
                if self.predict: self.h1_storage[B][keyB] = h1.xfxQ2(self.ind[B],xb,muF2)/xb
                else:            self.h1_storage[B][keyB] = h1.get_xF([xb],[muF2],B)/xb
            f1A = self.f1_storage[A][keyA]
            h1B = self.h1_storage[B][keyB]
            keyC = '%s,%s,%s'%(z,M,muF2)
            if keyC not in self.H1_storage[C]: self.H1_storage[C][keyC] = H1.get_H([z],[M],[muF2],C)[0]
            H1C = self.H1_storage[C][keyC]
            result += f1A * h1B * H1C

        return result        

    #--Mellin space
    def get_lum_UU_mell(self,channel,xa,xb,z,PhT,flav,ilum):

        split = flav.split(',')
        A = split[0]
        B = split[1]

        muF2 = self.muF2(PhT)

        f1  = self.f1
        keyA = '%s,%s'%(xa,muF2)
        if keyA not in self.f1_storage[A]: self.f1_storage[A][keyA] = f1.xfxQ2(self.ind[A],xa,muF2)/xa
        f1A = self.f1_storage[A][keyA]

        keyB = '%s,%s'%(xb,muF2)
        if keyB not in self.f1_storage[B]: self.f1_storage[B][keyB] = f1.xfxQ2(self.ind[B],xb,muF2)/xb
        f1B = self.f1_storage[B][keyB]
            
        if ilum=='mell-real':
            return f1A * f1B * np.real(z**(-self.n))
        if ilum=='mell-imag':
            return f1A * f1B * np.imag(z**(-self.n))

    def get_lum_UT_mell(self,channel,xa,xb,z,PhT,flav,ilum):

        A = flav

        muF2 = self.muF2(PhT)

        f1  = self.f1
        keyA = '%s,%s'%(xa,muF2)
        if keyA not in self.f1_storage[A]: self.f1_storage[A][keyA] = f1.xfxQ2(self.ind[A],xa,muF2)/xa
        f1A = self.f1_storage[A][keyA]
            
        if ilum=='mell-real':
            return f1A * np.real(xb**(-self.n) * z**(-self.m))
        if ilum=='mell-imag':
            return f1A * np.imag(xb**(-self.n) * z**(-self.m))

    #--calculate numerator in x-space
    def get_asym(self,idx):

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

        l = len(RS)

        mpi = conf['aux'].Mpi
        pre_num = lambda M, PhT: 2*PhT
        pre_den = lambda M, PhT: 2*PhT

        THY = np.zeros(l)
        NUM = np.zeros(l)
        DEN = np.zeros(l)


             
        for m in range(l):

            print(idx,m)

            nge, xge, wge = self.nge, self.xge, self.wge
            ngM, xgM, wgM = self.ngM, self.xgM, self.wgM
            ngP, xgP, wgP = self.ngP, self.xgP, self.wgP


            if int_eta:
                ETA = 0.5*(etamax[m]-etamin[m])*xge + 0.5*(etamax[m]+etamin[m])
                jac_ETA = 0.5*(etamax[m]-etamin[m])
                bin_ETA = etamax[m]-etamin[m]
            else:
                nge = 1
                ETA = [eta[m]]
                jac_ETA = 1.0
                bin_ETA = 1.0

            if int_M:
                M = 0.5*(Mmax[m]-Mmin[m])*xgM + 0.5*(Mmax[m]+Mmin[m])
                jac_M = 0.5*(Mmax[m]-Mmin[m])
                bin_M = Mmax[m]-Mmin[m]
            else:
                ngM = 1
                M = [Mh[m]]
                jac_M = 1.0 
                bin_M = 1.0

            if int_PhT:
                PHT = 0.5*(PhTmax[m]-PhTmin[m])*xgP + 0.5*(PhTmax[m]+PhTmin[m])
                jac_PHT = 0.5*(PhTmax[m]-PhTmin[m])
                bin_PHT = PhTmax[m]-PhTmin[m]
            else:
                ngP = 1
                PHT = [PhT[m]]
                jac_PHT = 1.0 
                bin_PHT = 1.0

            num = np.zeros((nge,ngM,ngP))
            den = np.zeros((nge,ngM,ngP))

            for i in range(nge):
                for j in range(ngM):
                    R = 0.5*M[j]*np.sqrt(1 - 4*mpi**2/M[j]**2)
                    for k in range(ngP):
                        num[i][j][k] = pre_num(M[j],PHT[k])*self.integrate(RS[m],PHT[k],M[j],ETA[i],'UT')*jac_ETA*jac_M*jac_PHT/bin_ETA/bin_M/bin_PHT
                        den[i][j][k] = pre_den(M[j],PHT[k])*self.integrate(RS[m],PHT[k],M[j],ETA[i],'UU')*jac_ETA*jac_M*jac_PHT/bin_ETA/bin_M/bin_PHT

            num = np.sum(wgP*num,axis=2)
            num = np.sum(wgM*num,axis=1)
            num = np.sum(wge*num,axis=0)
            den = np.sum(wgP*den,axis=2)
            den = np.sum(wgM*den,axis=1)
            den = np.sum(wge*den,axis=0)

            THY[m] = num/den
            NUM[m] = num
            DEN[m] = den


        return THY,NUM,DEN

    #--integrate over xa, xb
    def integrate(self,RS,PhT,M,eta,xsec,ilum='xspace',channel='QQ,QQ',flav='u'):

        #--switch signs of eta to match STAR definition
        eta = -eta

        xg,  wg  = self.xg,  self.wg
        xge, wge = self.xge, self.wge
        xamax = 0.99 
        xbmax = 0.99

        if ilum=='xspace':
            if xsec=='UU':
                channels = self.channels_UU
                SIG      = self.SIGUU
                LUM      = lambda channel,xa,xb,z,M: self.get_lum_UU(channel,xa,xb,PhT,eta,M,z)
            if xsec=='UT':
                channels = self.channels_UT
                SIG      = self.SIGUT
                LUM      = lambda channel,xa,xb,z,M: self.get_lum_UT(channel,xa,xb,PhT,eta,M,z)

        else:
            if xsec=='UU':
                SIG      = self.SIGUU
                LUM      = lambda channel,xa,xb,z,M: self.get_lum_UU_mell(channel,xa,xb,z,PhT,flav,ilum)
            channels = [channel]
            if xsec=='UT':
                SIG      = self.SIGUT
                LUM      = lambda channel,xa,xb,z,M: self.get_lum_UT_mell(channel,xa,xb,z,PhT,flav,ilum)


        RESULT = np.zeros((self.ng,self.ng))

        #--Gaussian quadrature integration over xa
        xamin = self.xamin(RS,PhT,eta)
        XA = 0.5*(xamax-xamin)*xg + 0.5*(xamax+xamin)
        jac_XA = 0.5*(xamax-xamin)

        #--loop over XA for Gaussian quadrature
        for j in range(self.ng):

            #--Gaussian quadrature integration over xb
            xbmin = self.xbmin(XA[j],RS,PhT,eta)
            XB = 0.5*(xbmax-xbmin)*xg + 0.5*(xbmax+xbmin)
            jac_XB = 0.5*(xbmax-xbmin)

            #--loop over XB for Gaussian quadrature
            for k in range(self.ng):

                z = self.z(XA[j],XB[k],RS,PhT,eta)
                s = self.s(XA[j],XB[k],RS)
                t = self.t(XA[j],XB[k],RS,PhT,eta)
                u = self.u(XA[j],XB[k],RS,PhT,eta)

                for channel in channels:
                    sig  = SIG[channel](PhT,s,t,u)
                    lum  = LUM(channel,XA[j],XB[k],z,M)
                    result = sig*lum/z
                    RESULT[j][k]  += result*jac_XA*jac_XB

        RESULT = np.sum(wg*RESULT,axis=1)
        RESULT = np.sum(wg*RESULT,axis=0)

        return RESULT

 
if __name__ == "__main__":


    from fitlib.resman import RESMAN

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
   
        THY = THEORY(ng=ng,predict=True)
        thy = {}
        for idx in data:
            thy[idx],_,_ = THY.get_asym(idx)
        print(thy)
        
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
                hand['thy'] ,= ax.plot     (M  ,thy[idx],        color=color)
            if binned=='PhT':
                hand[idx]    = ax.errorbar(pT ,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
                hand['thy'] ,= ax.plot     (pT ,thy[idx],        color=color)
            if binned=='eta':
                hand[idx]    = ax.errorbar(eta,value,yerr=alpha,color=color,fmt='o',ms=2.0,capsize=3.0)
                hand['thy'] ,= ax.plot     (eta,thy[idx],        color=color)
    
            if idx not in thy: continue
            if binned=='M':
                hand['thy'] ,= ax.plot     (M  ,thy[idx],        color=color)
            if binned=='PhT':
                hand['thy'] ,= ax.plot     (pT ,thy[idx],        color=color)
            if binned=='eta':
                hand['thy'] ,= ax.plot     (eta,thy[idx],        color=color)
    
    
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
                if idx not in thy: continue
                hand['thy'] ,= ax.plot    (M,thy[idx][ind]     ,color=color)
    
    
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
                if idx not in thy: continue
                hand['thy'] ,= ax.plot    (pT,thy[idx][ind]     ,color=color)
    
    
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
            if idx not in thy: continue
            hand['thy'] ,= ax.plot    (eta,thy[idx]        ,color=color)
    
    
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
                if idx not in thy: continue
                hand['thy'] ,= ax.plot    (M,thy[idx][ind]     ,color=color)
    
    
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
            if idx not in thy: continue
            hand['thy'] ,= ax.plot    (eta,thy[idx]        ,color=color)
    
    
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



    #--load DiFFs D1 and H1 for prediction purposes
    wdir = '/work/JAM/ccocuzza/diffs/results/step06'
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






 

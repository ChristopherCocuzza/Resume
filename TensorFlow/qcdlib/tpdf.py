#!/bin/env python
import sys,os
import matplotlib
matplotlib.use('Agg')
matplotlib.rc('text',usetex=True)
from tools.tools import checkdir
import pylab as py
from scipy.special import gamma
import numpy as np
import qcdlib.alphaS
from qcdlib.aux import AUX
from qcdlib.dglap import DGLAP
from qcdlib.kernels import KERNELS
from qcdlib.mellin import MELLIN
from tools.config import conf

class TPDF:

    def __init__(self, mellin=None):

        self.spl='trans'
        self.Q20=conf['Q20']
        self.mc2=conf['aux'].mc2
        self.mb2=conf['aux'].mb2

        if 'lhapdf_pdf'   in conf: self.lhapdf_pdf   = conf['lhapdf_pdf']
        else:                      self.lhapdf_pdf   = 'JAM22-PDF_proton_nlo'

        if 'lhapdf_ppdf'  in conf: self.lhapdf_ppdf  = conf['lhapdf_ppdf']
        else:                      self.lhapdf_ppdf  = 'JAM22-PPDF_proton_nlo'


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

        if 'tpdf_choice' in conf:
            self.choice = conf['tpdf_choice']
        else:
            self.choice = 'basic'

        if 'db factor' in conf:
            self.dbfactor = conf['db factor']
        else:
            self.dbfactor = None

        if 'smallx' in conf:
            self.smallx = conf['smallx']
        else:
            self.smallx = False

        self.set_params()
        self.setup()

    def set_params(self):

        if self.choice=='basic':
            """
            f(x) = norm * x**a0 * (1-x)**b0 * (1 + c*sqrt(x) + d*x)
            """
            self.params = {}

            self.params['g']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['u']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['d']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['s']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['c']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['b']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['ub']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['db']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['sb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['cb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['bb']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

            self.FLAV = ['g','u','d','s','c','b','ub','db','sb','cb','bb']
            self.PAR  = ['N', 'a', 'b', 'c', 'd']


        if self.choice=='valence':
            """
            f(x) = norm * x**a0 * (1-x)**b0 * (1 + c*sqrt(x) + d*x)
            """
            self.params = {}

            self.params['g']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['uv']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['dv']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['s']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['c']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['b']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['ub']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['db']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['sb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['cb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['bb']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

            self.FLAV = ['g','uv','dv','s','c','b','ub','db','sb','cb','bb']
            self.PAR  = ['N', 'a', 'b', 'c', 'd']


        if self.choice=='twoshapes':
            """
            f(x) = norm * x**a0 * (1-x)**b0 * (1 + c*sqrt(x) + d*x)
            """
            self.params = {}

            self.params['g']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['u']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['d']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['s']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['c']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['ub']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['db']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['sb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['cb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            self.FLAV = ['g','u','d','s','c','ub','db','sb','cb']
            self.PAR  = ['N0', 'a0', 'b0', 'c0', 'd0', 'N1', 'a1', 'b1', 'c1', 'd1']

        if self.choice=='JAM3D':
            """
            f(x) = norm * x**a0 * (1-x)**b0 * (1 + c*sqrt(x) + d*x)
            """
            self.params = {}

            self.params['g']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['u']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['d']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['s']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['c']    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['ub']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['db']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['sb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.params['cb']   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            self.FLAV = ['g','u','d','s','c','ub','db','sb','cb']
            self.PAR  = ['N0', 'a0', 'b0', 'N1', 'a1', 'b1']

    def setup(self):

        moms = {}

        if self.choice=='valence': 
            u = self.get_moments('uv') + self.get_moments('ub')
            d = self.get_moments('dv') + self.get_moments('db')
        else:
            u = self.get_moments('u')
            d = self.get_moments('d')

        #--have ub and db independent
        if self.dbfactor==None:
            moms['g']  = self.get_moments('g')
            moms['up'] = u                     + self.get_moments('ub')
            moms['dp'] = d                     + self.get_moments('db')
            moms['sp'] = self.get_moments('s') + self.get_moments('sb')

            moms['um'] = u                     - self.get_moments('ub')
            moms['dm'] = self.get_moments('d') - self.get_moments('db')
            moms['sm'] = self.get_moments('s') - self.get_moments('sb')
        #--have db = f*ub
        else:
            f = self.dbfactor
            moms['g']  = self.get_moments('g')
            moms['up'] = u                     +   self.get_moments('ub')
            moms['dp'] = d                     + f*self.get_moments('ub')
            moms['sp'] = self.get_moments('s') +   self.get_moments('sb')

            moms['um'] = u                     -   self.get_moments('ub')
            moms['dm'] = d                     - f*self.get_moments('ub')
            moms['sm'] = self.get_moments('s') -   self.get_moments('sb')

        self.moms0 = moms
        self.get_BC(moms)

        #--we will store all Q2 values that has been precalc
        self.storage={}

    def beta(self,a,b):
        return gamma(a)*gamma(b)/gamma(a+b)

    def get_moments(self,flav,N=None):
        """
        if N==None: then parametrization is to be use to compute moments along mellin contour
        else the Nth moment is returned
        """
        if N==None: N=self.mellin.N
        if self.choice=='basic' or self.choice=='valence':
            M,a,b,c,d=self.params[flav]
            norm = self.beta(1+a,b+1)+c*self.beta(1+a+0.5,b+1)+d*self.beta(1+a+1.0,b+1)
            mom  = self.beta(N+a,b+1)+c*self.beta(N+a+0.5,b+1)+d*self.beta(N+a+1.0,b+1)
            return M*mom/norm

        if self.choice=='twoshapes':
            M0,a0,b0,c0,d0,M1,a1,b1,c1,d1=self.params[flav]
            norm0 = self.beta(1+a0,b0+1)+c0*self.beta(1+a0+0.5,b0+1)+d0*self.beta(1+a0+1.0,b0+1)
            mom0  = self.beta(N+a0,b0+1)+c0*self.beta(N+a0+0.5,b0+1)+d0*self.beta(N+a0+1.0,b0+1)
            norm1 = self.beta(1+a1,b1+1)+c1*self.beta(1+a1+0.5,b1+1)+d1*self.beta(1+a1+1.0,b1+1)
            mom1  = self.beta(N+a1,b1+1)+c1*self.beta(N+a1+0.5,b1+1)+d1*self.beta(N+a1+1.0,b1+1)
            return M0*mom0/norm0 + M1*mom1/norm1


        if self.choice=='JAM3D':
            M0,a0,b0,M1,a1,b1=self.params[flav]
            norm = self.beta(2+a0,b0+1)#+M1*self.beta(a0+a1+2,b0+b1+1)
            mom  = self.beta(N+a0,b0+1)#+M1*self.beta(a0+a1+N,b0+b1+1)
            return M0*mom/norm

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

        ###############################################
        # BC for Nf=3
        g   = moms['g']
        up  = moms['up']
        um  = moms['um']
        dp  = moms['dp']
        dm  = moms['dm']
        sp  = moms['sp']
        sm  = moms['sm']
        cp  = zero
        cm  = zero
        bp  = zero
        bm  = zero
        self.BC3=self._get_BC(g,up,um,dp,dm,sp,sm,zero,zero,zero,zero,zero,zero,um,dm)

        ###############################################
        # BC for Nf=4
        BC4=self.dglap.evolve(self.BC3,self.Q20,self.mc2,3)
        g =BC4['g']
        up=BC4['up']
        dp=BC4['dp']
        sp=BC4['sp']
        cp=BC4['cp']
        bp=BC4['bp']
        tp=BC4['tp']
        um=BC4['um']
        dm=BC4['dm']
        sm=BC4['sm']
        cm=BC4['cm']
        bm=BC4['bm']
        tm=BC4['tm']
        um_=BC4['um_']
        dm_=BC4['dm_']
        self.BC4=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_)

        ###############################################
        # BC for Nf=5
        BC5=self.dglap.evolve(self.BC4,self.mc2,self.mb2,4)
        g =BC5['g']
        up=BC5['up']
        dp=BC5['dp']
        sp=BC5['sp']
        cp=BC5['cp']
        bp=BC5['bp']
        tp=BC5['tp']
        um=BC5['um']
        dm=BC5['dm']
        sm=BC5['sm']
        cm=BC5['cm']
        bm=BC5['bm']
        tm=BC5['tm']
        um_=BC5['um_']
        dm_=BC5['dm_']
        self.BC5=self._get_BC(g,up,um,dp,dm,sp,sm,cp,cm,bp,bm,tp,tm,um_,dm_)

    def evolve(self,Q2array):

      for Q2 in Q2array:
          if Q2 not in self.storage:
              if self.mb2<Q2:
                  self.storage[Q2]=self.dglap.evolve(self.BC5,self.mb2,Q2,5)
              elif self.mc2<=Q2 and Q2<=self.mb2:
                  self.storage[Q2]=self.dglap.evolve(self.BC4,self.mc2,Q2,4)
              elif Q2<self.mc2:
                  self.storage[Q2] =self.dglap.evolve(self.BC3,self.Q20,Q2,3)

    def get_xF(self,x,Q2,flav,evolve=True):
        if evolve: self.evolve(Q2)
        #--skip distributions that are set to zero to save time
        if flav in self.params and self.params[flav][0] == 0:
            return np.zeros(len(x))

        if self.smallx == False:
            return np.array([x[i]*self.mellin.invert(x[i],self.storage[Q2[i]][flav]) for i in range(len(x))])

        if self.smallx == True:
            result = []
            x0 = 0.006
            for i in range(len(x)): 
                #a  = 0.243
                alphaS = conf['alphaS'].get_alphaS(Q2[i],0)
                a = 1 - 2 * np.sqrt(alphaS*3/2/np.pi)
                if x[i] >= x0:
                    result.append(x[i]*self.mellin.invert(x[i],self.storage[Q2[i]][flav]))
                if x[i] < x0:
                    pdf0 = self.mellin.invert(x[i],self.storage[Q2[i]][flav])
                    N = pdf0/x0**a
                    result.append(N*x[i]**(a+1))
            return result

    #--generate report for Soffer Bound violations   
    def gen_report(self,verb=1,level=1):
          
        L=[]

        res2 = conf['SB chi2']
 
        L.append('chi2 from Soffer Bound: %3.5f'%res2)

        return L

        
if  __name__=='__main__':

    conf['order']='LO'
    conf['Q20'] = 4.0
    conf['aux']=AUX()
    conf['mellin']=MELLIN(npts=4)
    conf['alphaS']=alphaS
    #--load all replicas
    os.environ['LHAPDF_DATA_PATH'] = '%s/qcdlib/lhapdf'%(os.environ['FITPACK'])
    conf['LHAPDF:PDF']   = lhapdf.mkPDFs('JAM22-PDF_proton_nlo')
    conf['LHAPDF:PPDF']  = lhapdf.mkPDFs('JAM22-PPDF_proton_nlo')


    tpdf=TPDF()

    X  = tpdf.X
    f1m = tpdf.f1['mean']
    g1m = tpdf.g1['mean']
    f1s = tpdf.f1['std']
    g1s = tpdf.g1['std']

    u = X*0.5*(f1m['u'] + g1m['u'] + f1s['u'] + g1s['u']) 
    d = X*0.5*(f1m['d'] + g1m['d'] + f1s['d'] + g1s['d']) 
    #u_do = -X*0.5*(f1m['u'] + g1m['u'] + f1s['u'] + g1s['u']) 
    #d_do = -X*0.5*(f1m['d'] + g1m['d'] + f1s['d'] + g1s['d']) 

    nrows,ncols = 1,2
    fig = py.figure(figsize=(ncols*6,nrows*3))
    ax11 = py.subplot(nrows,ncols,1)
    ax12 = py.subplot(nrows,ncols,2)

    hand = {}
    hand['u'] ,= ax11.plot(X, u,color='blue',ls=':')
    hand['u'] ,= ax11.plot(X,-u,color='blue',ls=':')
    hand['d'] ,= ax12.plot(X, d,color='blue',ls=':')
    hand['d'] ,= ax12.plot(X,-d,color='blue',ls=':')

    for ax in [ax11,ax12]:

        ax.set_xlim(0.0,1.0)
        ax.tick_params(axis='both', which='major', labelsize=20,direction='in', right = True, left = True)
        ax.set_xlabel(r'\boldmath$x$',size=25)
        ax.xaxis.set_label_coords(0.90,0.00)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.axhline(0,0,1,color='black',ls='-',alpha=0.8)
 
    ax11.set_ylabel(r'\boldmath$\frac{x}{2}(f_1(x) + g_1(x))$',size=25)
    ax11.text(0.60,0.70,r'$Q^2 = %s~{\rm GeV}^2$'%conf['Q20'],transform=ax11.transAxes,size=20)
    ax11.text(0.70,0.20,r'\boldmath$u$',transform=ax11.transAxes,size=30)
    ax12.text(0.70,0.20,r'\boldmath$d$',transform=ax12.transAxes,size=30)

    ax11.set_ylim(-0.6,0.6)
    ax11.set_yticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6])

    ax12.set_ylim(-0.3,0.3)
    ax12.set_yticks([-0.3,-0.2,-0.1,0,0.1,0.2,0.3])


    #handles, labels = [],[]
    #handles.append(u)
    #handles.append(g)
    #labels.append(r'\boldmath$u$')
    #labels.append(r'\boldmath$g$')
    #ax.legend(handles,labels,loc='upper right',fontsize=25,frameon=0,handletextpad=0.3,handlelength=1.0)
    py.tight_layout()

    checkdir('gallery')
    filename = 'gallery/SB.png'
    py.savefig(filename)
    print('saving figure to %s'%filename)
    py.clf()



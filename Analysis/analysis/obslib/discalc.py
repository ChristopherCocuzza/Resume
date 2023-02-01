import sys,os
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import pylab as py
import numpy as np
from scipy.integrate import quad,fixed_quad
from mpmath import fp
import lhapdf
from qcdlib import mellin
from tools.tools import lprint

#--load PDFs and calculate F1, F2, and FL
#--NEED TO FIX
class IDIS:
  
    def __init__(self,fname):

        self.pdf=lhapdf.mkPDFs(fname)
        self.size = len(self.pdf)
        self.mc=self.pdf[0].quarkThreshold(4)
        self.mb=self.pdf[0].quarkThreshold(5)
        self.TR=0.5
        self.CF=4./3.
        self.alfa=1/137.036
        self.M=0.93891897
        self.mpi=0.139
        apU=4.0/9.0
        apD=1.0/9.0
        self.couplings={}
        self.couplings['p']={1:apD,2:apU,3:apD,4:apU,5:apD}
        self.couplings['n']={1:apU,2:apD,3:apD,4:apU,5:apD}
        self.fmap={}

        self.F2={'p':{},'n':{}}
        self.FL={'p':{},'n':{}}

        self.F2['p']['mean']={}
        self.F2['p']['std'] ={}
        self.F2['n']['mean']={}
        self.F2['n']['std'] ={}
        self.FL['p']['mean']={}
        self.FL['p']['std'] ={}
        self.FL['n']['mean']={}
        self.FL['n']['std'] ={}
   
    def integrator(self,f,xmin,xmax,method='gauss',n=100):
        f=np.vectorize(f)
        if method=='quad':
            return quad(f,xmin,xmax)[0]
        elif method=='gauss':
            return fixed_quad(f,xmin,xmax,n=n)[0]
      
    def log_plus(self,z,f,x):
        return np.log(1-z)/(1-z)*(f(x/z)/z-f(x)) + 0.5*np.log(1-x)**2*f(x)/(1-x)
  
    def one_plus(self,z,f,x):
        return 1/(1-z)*(f(x/z)/z-f(x))+ np.log(1-x)*f(x)/(1-x)
  
    def C2q(self,z,f,x):
        return self.CF*(2*self.log_plus(z,f,x)-1.5*self.one_plus(z,f,x)\
          +(-(1+z)*np.log(1-z)-(1+z*z)/(1-z)*np.log(z)+3+2*z)*f(x/z)/z\
          -(np.pi**2/3+4.5)*f(x)/(1-x))
      
    def C2g(self,z,f,x):
        return 0.5*(((1-z)**2+z*z)*np.log((1-z)/z)-8*z*z+8*z-1)*f(x/z)/z
   
    def CLq(self,z,f,x):
        return 2*self.CF*z*f(x/z)/z #<--- note prefactor 2, instead of 4 used by MVV
      
    def CLg(self,z,f,x):
        return 4*z*(1-z)*f(x/z)/z
  
    def qplus(self,rep,x,Q2):
        output=0
        for i in range(1,self.Nf+1):
            output+=self.couplings[self.tar][i]*(self.pdf[rep].xfxQ2(i,x,Q2)/x+self.pdf[rep].xfxQ2(-i,x,Q2)/x)
        return output

    def sumpdfquark(self,x,Q2,tar):
        self.tar=tar
        output=self.qplus(x,Q2)
        return output
  
    def glue(self,rep,x,Q2):
        output=0
        for i in range(1,self.Nf+1):
            output+=2*self.couplings[self.tar][i]
        return output*self.pdf[rep].xfxQ2(21,x,Q2)/x
        
    def integrand_F2(self,rep,x,z,Q2):
        return self.C2q(z,lambda y:self.qplus(rep,y,Q2),x) + self.C2g(z,lambda y:self.glue(rep,y,Q2),x)
      
    def integrand_FL(self,rep,x,z,Q2):
        return self.CLq(z,lambda y:self.qplus(rep,y,Q2),x) + self.CLg(z,lambda y:self.glue(rep,y,Q2),x)
      
    def get_F2(self,x,Q2,tar):
        if (x,Q2) not in self.F2[tar]['mean']:
            f2 = []
            self.tar=tar
            alphaS = self.pdf[0].alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            for rep in range(self.size):
                LO=self.qplus(rep,x,Q2)
                integrand=lambda z:self.integrand_F2(rep,x,z,Q2)
                NLO=self.integrator(integrand,x,1)
                f2.append(x*(LO+alphaS/np.pi/2.0*NLO))
            f2 = np.array(f2)
            self.F2[tar]['mean'][(x,Q2)]=np.mean(f2)
            self.F2[tar]['std'] [(x,Q2)]=np.std (f2)
        return self.F2[tar]['mean'][(x,Q2)],self.F2[tar]['std'][(x,Q2)]
  
    def get_FL(self,x,Q2,tar):
        if (x,Q2) not in self.FL[tar]:
            fl = []
            self.tar=tar
            alphaS = self.pdf[0].alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            for rep in range(self.size):
                integrand=lambda z:self.integrand_FL(rep,x,z,Q2)
                NLO=self.integrator(integrand,x,1)
                fl.append(x*alphaS/np.pi/2.0*NLO)
            fl = np.array(fl)
            self.FL[tar]['mean'][(x,Q2)]=np.mean(fl)
            self.FL[tar]['std'] [(x,Q2)]=np.std (fl)
        return self.FL[tar]['mean'][(x,Q2)],self.FL[tar]['std'][(x,Q2)]
  
    def get_F1(self,x,Q2,tar):
        F2=self.get_F2(x,Q2,tar)
        FL=self.get_FL(x,Q2,tar)
        return ((1+4*self.M**2/Q2*x**2)*F2-FL)/(2*x)
   
    def get_dsigdxdQ2(self,x,y,Q2,target,precalc=False): 
        if precalc==False: 
            return 4*np.pi*self.alfa**2/Q2**2/x*((1-y+y**2/2)*self.get_F2(x,Q2,target)-y**2/2*self.get_FL(x,Q2,target))
        else:
            return self.storage.retrieve([x,y,Q2,target])

    def plot_stf(self,Q2):
    
      nrows,ncols=1,3
      fig = py.figure(figsize=(ncols*7,nrows*4))
      ax11=py.subplot(nrows,ncols,1)
      ax12=py.subplot(nrows,ncols,2)
      ax13=py.subplot(nrows,ncols,3)
    
      filename = 'stf'
   
      X = np.geomspace(0.001,0.1,50)
      X = np.append(X,(np.linspace(0.1,0.99,50)))

      stfs = ['F2']
      STF = {}
      for stf in stfs:
          STF[stf] = {}
          for tar in ['p','n']:
              STF[stf][tar] = {}
              STF[stf][tar]['mean'] = np.zeros(len(X))
              STF[stf][tar]['std']  = np.zeros(len(X))
              for i in range(len(X)):
                  lprint('Calculating %s%s: %s/%s '%(stf,tar,i+1,len(X)))
                  if   stf=='F2': func = self.get_F2(X[i],Q2,tar)
                  elif stf=='FL': func = self.get_FL(X[i],Q2,tar)
                  else: continue
                  STF[stf][tar]['mean'][i],STF[stf][tar]['std'][i] = func
                  STF[stf][tar]['mean'][i] *= X[i]
                  STF[stf][tar]['std'][i]  *= X[i]
              print()
 
      hand = {}
      for stf in stfs:
          for tar in ['p','n']:
    
              if tar=='p': color='red'
              if tar=='n': color='green'
    
              if   stf =='F2': ax = ax11
              elif stf =='FL': ax = ax12
              else: continue
   
              data = STF[stf][tar]
              up = data['mean'] + data['std']
              do = data['mean'] - data['std']

              hand[tar] = ax. fill_between(X,do,up,color=color,alpha=1.0)
    
      for ax in [ax11,ax12,ax13]:
            ax.set_xlim(0,0.9)
              
            ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)
   
      handles = []
      handles.append(hand['p'])
      handles.append(hand['n'])

      labels = []
      labels.append(r'\boldmath$xF_2^p$')
      labels.append(r'\boldmath$xF_2^n$')
      ax11.legend(handles,labels,loc='upper right',fontsize=20,frameon=False)
      labels = []
      labels.append(r'\boldmath$xF_L^p$')
      labels.append(r'\boldmath$xF_L^n$')
      ax12.legend(handles,labels,loc='upper right',fontsize=20,frameon=False)
      labels = []
      labels.append(r'\boldmath$xF_3^p$')
      labels.append(r'\boldmath$xF_3^n$')
      ax13.legend(handles,labels,loc='upper right',fontsize=20,frameon=False)
    
      ax11.set_ylim(0,0.1)      #,ax11.set_yticks([0,0.2,0.4,0.6,0.8])
      ax12.set_ylim(0,0.004)    #,ax12.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
      ax13.set_ylim(0,0.00045)   #,ax13.set_yticks([0,0.2,0.4,0.6])
    
      for ax in [ax11,ax12,ax13]:
          minorLocator = MultipleLocator(0.02)
          majorLocator = MultipleLocator(0.2)
          ax.xaxis.set_minor_locator(minorLocator)
          ax.xaxis.set_major_locator(majorLocator)
          ax.xaxis.set_tick_params(which='major',length=6)
          ax.xaxis.set_tick_params(which='minor',length=3)
          ax.yaxis.set_tick_params(which='major',length=6)
          ax.yaxis.set_tick_params(which='minor',length=3)
          ax.set_xticks([0.2,0.4,0.6,0.8])
          ax.set_xlabel(r'\boldmath$x$' ,size=35)   
          ax.xaxis.set_label_coords(0.97,0)
    
    
      if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
      else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)
    
      py.tight_layout()
    
      filename+='.png'
    
      #checkdir('%s/gallery'%wdir)
      py.savefig(filename)
      print ('Saving figure to %s'%filename)
      py.clf()

#--load PPDFs and calculate g1
class PIDIS:
  
    def __init__(self,fname):

        self.pdf=lhapdf.mkPDFs(fname)
        self.size = len(self.pdf)
        self.mc=self.pdf[0].quarkThreshold(4)
        self.mb=self.pdf[0].quarkThreshold(5)
        self.TR=0.5
        self.CF=4./3.
        self.alfa=1/137.036
        self.M=0.93891897
        self.mpi=0.139
        apU=4.0/9.0
        apD=1.0/9.0
        self.couplings={}
        self.couplings['p']={1:apD,2:apU,3:apD,4:apU,5:apD}
        self.couplings['n']={1:apU,2:apD,3:apD,4:apU,5:apD}
        self.fmap={}

        self.g1={'p':{},'n':{}}
        self.g1['p']['mean'] = {}
        self.g1['p']['std']  = {}
        self.g1['n']['mean'] = {}
        self.g1['n']['std']  = {}
   
    def integrator(self,f,xmin,xmax,method='gauss',n=10):
        f=np.vectorize(f)
        if method=='quad':
            return quad(f,xmin,xmax)[0]
        elif method=='gauss':
            return fixed_quad(f,xmin,xmax,n=n)[0]
      
    def log_plus(self,z,f,x):
        return np.log(1-z)/(1-z)*(f(x/z)/z-f(x)) + 0.5*np.log(1-x)**2*f(x)/(1-x)
  
    def one_plus(self,z,f,x):
        return 1/(1-z)*(f(x/z)/z-f(x))+ np.log(1-x)*f(x)/(1-x)
  
    def PC1q(self,z,f,x):
        zeta2 = fp.zeta(2)
        return self.CF*(4*self.log_plus(z,f,x)-3*self.one_plus(z,f,x)\
        +(-2*(1+z)*np.log(1-z)-2*(1+z*z)*np.log(z)/(1-z)+4+2*z)*f(x/z)/z\
        -(4*zeta2 + 9)*f(x)/(1-x))

    def PC1g(self,z,f,x):
        return 0.5*(4*(2*z-1)*(np.log(1-z)-np.log(z))+4*(3-4*z))*f(x/z)/z
 
    def qplus(self,rep,x,Q2):
        output=0
        for i in range(1,self.Nf+1):
            output+=self.couplings[self.tar][i]*(self.pdf[rep].xfxQ2(i,x,Q2)/x+self.pdf[rep].xfxQ2(-i,x,Q2)/x)
        return output

    def sumpdfquark(self,x,Q2,tar):
        self.tar=tar
        output=self.qplus(x,Q2)
        return output
  
    def glue(self,rep,x,Q2):
        output=0
        for i in range(1,self.Nf+1):
            output+=2*self.couplings[self.tar][i]
        return output*self.pdf[rep].xfxQ2(21,x,Q2)/x
        
    def integrand_g1(self,rep,x,z,Q2):
        return self.PC1q(z,lambda y:self.qplus(rep,y,Q2),x) + self.PC1g(z,lambda y:self.glue(rep,y,Q2),x)
      
    def get_g1(self,x,Q2,tar):
        if (x,Q2) not in self.g1[tar]['mean']:
            G1 = []
            self.tar=tar
            alphaS = self.pdf[0].alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            for rep in range(self.size):
                LO=self.qplus(rep,x,Q2)
                integrand=lambda z:self.integrand_g1(rep,x,z,Q2)
                NLO=self.integrator(integrand,x,1)
                G1.append(0.5*(LO+alphaS/np.pi/4.0*NLO))
            G1 = np.array(G1)
            self.g1[tar]['mean'][(x,Q2)]=np.mean(G1)
            self.g1[tar]['std'] [(x,Q2)]=np.std (G1)
        return self.g1[tar]['mean'][(x,Q2)], self.g1[tar]['std'][(x,Q2)]

    def plot_g1(self,Q2):
    
      nrows,ncols=1,1
      N = nrows*ncols
      fig = py.figure(figsize=(ncols*8,nrows*5))
      axs, axLs = {},{}
      for i in range(N):
          axs[i+1] = py.subplot(nrows,ncols,i+1)
          divider = make_axes_locatable(axs[i+1])
          axLs[i+1] = divider.append_axes("right",size=3.00,pad=0,sharey=axs[i+1])
          axLs[i+1].set_xlim(0.1,0.9)
          axLs[i+1].spines['left'].set_visible(False)
          axLs[i+1].yaxis.set_ticks_position('right')
          py.setp(axLs[i+1].get_xticklabels(),visible=True)
    
          axs[i+1].spines['right'].set_visible(False)
    
      filename = 'g1'
   
      X = np.geomspace(0.001,0.1,50)
      X = np.append(X,(np.linspace(0.1,0.99,50)))

      idx1 = np.nonzero(X <= 0.1)
      idx2 = np.nonzero(X >= 0.1)
   
      xg1 = {'p': {}, 'n': {}}
      for tar in ['p','n']:
          xg1[tar]['mean'] = np.zeros(len(X))
          xg1[tar]['std']  = np.zeros(len(X))
          for i in range(len(X)):
              lprint('Calculating g1%s: %s/%s '%(tar,i+1,len(X)))
              xg1[tar]['mean'][i],xg1[tar]['std'][i] = self.get_g1(X[i],Q2,tar)
              xg1[tar]['mean'][i] *= X[i]
              xg1[tar]['std'][i]  *= X[i]
          print()
 
      hand = {}
      for tar in ['p','n']:
          for stf in ['g1']:
    
              if tar=='p': color='red'
              if tar=='n': color='green'
    
              if stf =='g1': ax,axL = axs[1],axLs[1]
   
              data = xg1[tar]
              up = data['mean'] + data['std']
              do = data['mean'] - data['std']

              hand[tar] = ax. fill_between(X[idx1],do[idx1],up[idx1],color=color,alpha=1.0)
              hand[tar] = axL.fill_between(X[idx2],do[idx2],up[idx2],color=color,alpha=1.0)
        
    
    
      for i in range(N):
            axs[i+1].set_xlim(8e-3,0.1)
            axs[i+1].semilogx()
    
            axs[i+1].tick_params(axis='both', which='both', top=True, direction='in',labelsize=20)
            axs[i+1].set_xticks([0.01,0.1])
            axs[i+1].set_xticklabels([r'$0.01$',r'$0.1$'])
            axs[i+1].axhline(0,0,1,ls='--',color='black',alpha=0.5)
            axs[i+1].axvline(0.1,0,1,ls=':' ,color='black',alpha=0.5)
    
            axLs[i+1].set_xlim(0.1,1.0)
    
            axLs[i+1].tick_params(axis='both', which='both', top=True, right=True, left=False, labelright=False, direction='in',labelsize=20)
            axLs[i+1].set_xticks([0.3,0.5,0.7])
            axLs[i+1].set_xticklabels([r'$0.3$',r'$0.5$',r'$0.7$'])
            axLs[i+1].axhline(0,0,1,ls='--',color='black',alpha=0.5)
            axLs[i+1].axvline(0.1,0,1,ls=':' ,color='black',alpha=0.5)
    
      #axs[1] .tick_params(labelbottom=False)
      #axLs[1].tick_params(labelbottom=False)
      axLs[1].set_xlabel(r'\boldmath$x$' ,size=30)
      axLs[1].xaxis.set_label_coords(0.95,0.00)
    
      axs[1].text(0.10,0.40,r'\boldmath$xg_1$',transform=axs[1].transAxes,size=40)
      axs[1].text(0.10,0.60,r'\boldmath$x~\rm{space}$',transform=axs[1].transAxes,size=40)
      #axs[2].text(0.10,0.25,r'\boldmath$xg_2$',transform=axs[2].transAxes,size=40)
    
      axs[1].set_ylim(-0.025,0.085)
      #axs[2].set_ylim(-0.050,0.015)
    
      axs[1].set_yticks([-0.02,0,0.02,0.04,0.06,0.08])
      #axs[2].set_yticks([-0.04,-0.03,-0.02,-0.01,0,0.01])
    
      if Q2 == 1.27**2: axs[1].text(0.05,0.05,r'$Q^2 = m_c^2$',             transform=axs[1].transAxes,size=30)
      else:             axs[1].text(0.05,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=axs[1].transAxes,size=25)
    
      handles, labels = [],[]
      handles.append(hand['p'])
      handles.append(hand['n'])
      labels.append(r'\textrm{\textbf{p}}')
      labels.append(r'\textrm{\textbf{n}}')
      axs[1].legend(handles,labels,loc='upper left',fontsize=25, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 2, columnspacing = 0.5)
      py.tight_layout()
      py.subplots_adjust(hspace=0)
    
      filename+='.png'
    
      py.savefig(filename)
      print ('Saving figure to %s'%filename)
      py.clf()


















#!/usr/bin/env python
import sys,os
import numpy as np
from numba import jit 
cache=True

@jit(nopython=True, cache=cache)
def jit_invert(x,F,N,W,JAC,phase):
    return np.sum(np.imag(phase * x**(-N) * F)/np.pi * W * JAC)


@jit(nopython=True, cache=cache)
def jit_double_invert(self,x,F,G,N,WN,WM,JACN,JACM,phase):
    H = (F*phase-G)*WM*JACM
    return -np.real(np.sum(x**(-N)*np.einsum('ij->i',H)*WN*JACN))/(2.*np.pi**2)

class MELLIN:

    def __init__(self,npts=8,extended=False,c=None):
 
        #--gen z and w values along coutour
        x,w=np.polynomial.legendre.leggauss(npts)
        znodes=[0,0.1,0.3,0.6,1.0,1.6,2.4,3.5,5,7,10,14,19,25,32,40,50,63]
        #if extended: znodes.extend([70,80,90,100,110,120,130])
        if extended: znodes.extend([70,80,90,100])
  
        Z,W,JAC=[],[],[]
        for i in range(len(znodes)-1):
            a,b=znodes[i],znodes[i+1]
            Z.extend(0.5*(b-a)*x+0.5*(a+b))
            W.extend(w)
            JAC.extend([0.5*(b-a) for j in range(x.size)])
        Z=np.array(Z)
        #--globalize
        self.W=np.array(W)
        self.Z=Z
        self.JAC=np.array(JAC)
        #--gen mellin contour
        if c==None: c=1.9 
        phi=3.0/4.0*np.pi
        self.N=c+Z*np.exp(complex(0,phi)) 
        self.phase= np.exp(complex(0,phi))
          
    def invert(self,x,F):
        return jit_invert(x,F,self.N,self.W,self.JAC,self.phase)

class IMELLIN:

    def __init__(self):
  
        self.N=np.array([1,2,3,4,5,6,7,9],dtype=complex) 

class DMELLIN:

  def __init__(self,nptsN=8,nptsM=8,extN=False,extM=False):

      xN,wN=np.polynomial.legendre.leggauss(nptsN)
      xM,wM=np.polynomial.legendre.leggauss(nptsM)
      znodesN=[0,0.1,0.3,0.6,1.0,1.6,2.4,3.5,5,7,10,14,19,25,32,40,50,63]
      znodesM=[0,0.1,0.3,0.6,1.0,1.6,2.4,3.5,5,7,10,14,19,25,32,40,50,63]
      if extN: znodesN.extend([70,80,90,100])
      if extM: znodesM.extend([70,80,90,100])

      ZN,WN,JACN=[],[],[]
      ZM,WM,JACM=[],[],[]
      for i in range(len(znodesN)-1):
          aN,bN=znodesN[i],znodesN[i+1]
          ZN.extend(0.5*(bN-aN)*xN+0.5*(aN+bN))
          WN.extend(wN)
          JACN.extend([0.5*(bN-aN) for j in range(xN.size)])
      for i in range(len(znodesM)-1):
          aM,bM=znodesM[i],znodesM[i+1]
          ZM.extend(0.5*(bM-aM)*xM+0.5*(aM+bM))
          WM.extend(wM)
          JACM.extend([0.5*(bM-aM) for j in range(xM.size)])
      self.ZN=np.array(ZN)
      self.ZM=np.array(ZM)
      # globalize
      self.WN=np.array(WN)
      self.WM=np.array(WM)
      ZN=self.ZN
      ZM=self.ZM
      self.JACN=np.array(JACN)
      self.JACM=np.array(JACM)
      # gen mellin contour                                                                                       
      cN=1.9
      cM = cN
      phi=3.0/4.0*np.pi
      self.N=cN+ZN*np.exp(complex(0,phi))
      self.M=cM+ZM*np.exp(complex(0,phi))
      self.phase= np.exp(complex(0,2.*phi))

  def invert(self,x,F,G):
      return jit_double_invert(x,F,G,self.N,self.WN,self.WM,self.JACN,self.JACM,self.phase)

if __name__=='__main__':

  mell=MELLIN(8)
  a=-1.8
  b=6.0
  N=mell.N
  
  mom=gamma(N+a)*gamma(b+1)/gamma(N+a+b+1)
  X=10**np.linspace(-5,-1,10)
  f=lambda x: x**a*(1-x)**b
  for x in X:
    print ('x=%10.4e  f=%10.4e  inv=%10.4e'%(x,f(x),mell.invert(x,mom)))


















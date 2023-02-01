#!/usr/bin/env python
import os,sys
import copy

#--helper libraries
import numpy as np

#--from fitpack
from tools.tools import load,checkdir,convert,lprint

#--index conventions
#--0: eta
#--1: PhT
#--2: realN
#--3: imagN
#--4: realT
#--5: imagT

class DMELLIN:

  def __init__(self,nptsN=4,nptsM=4,extN=False,extM=False):

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


def gen_table_UU(channel,flav):

  xsec = 'UU'
  dmellin = DMELLIN()
  N = dmellin.N

  #--use only real part of N
  realN = np.real(N)

  directory = './grids/%s'%xsec
  grids = os.listdir(directory)

  PhT, eta    = [],[]
  realN = []
  realT,imagT = [],[]
  #--single Mellin transform
  num_params = 5
  L = len(N)
  length=len(grids)*len(N)

  i = 0
  for grid in grids:
      data = load('%s/%s'%(directory,grid))
      T = data[channel][flav]
      eta.extend(data['eta']*np.ones(L))
      PhT.extend(data['PhT']*np.ones(L))
      realN.extend(N.flatten().real)
      realT.extend(T.flatten().real)
      imagT.extend(T.flatten().imag)

  DATA = {}
  DATA['data']    = np.zeros((num_params,length))
  DATA['data'][0] = eta 
  DATA['data'][1] = PhT 
  DATA['data'][2] = realN 
  DATA['data'][3] = realT
  DATA['data'][4] = imagT

  #--smoothing factors
  #--add smoothing factor to PhT here: PhT**6 for UU, PhT**4 for UT
  DATA['factor'] = np.ones((num_params,length))
  DATA['factor'][3] = DATA['data'][1]**6.00
  DATA['factor'][4] = DATA['data'][1]**6.00


  #--smooth, then normalize data here
  DATA['data'],DATA['mean'],DATA['std'] = convert(DATA)

  checkdir('modelgrids/%s'%xsec)
  filename='modelgrids/%s/%s-%s.npy'%(xsec,channel,flav)
  np.save(filename, DATA)
  print('Saving table to %s'%filename)


  stats = {}
  stats['mean'] = copy.copy(DATA['mean'])
  stats['std']  = copy.copy(DATA['std'])
  checkdir('modelgrids/stats/%s'%xsec)
  filename='modelgrids/stats/%s/%s-%s.npy'%(xsec,channel,flav)
  np.save(filename, stats)
  print('Saving stats to %s'%filename)


  return

def gen_table_UT(channel,flav):
 
  xsec = 'UT'
  dmellin = DMELLIN()
  N, M = dmellin.N, dmellin.M

  #--only care about real parts
  N = np.real(N)
  M = np.real(M)

  M, N = np.meshgrid(M,N)

  N = N.flatten()
  M = M.flatten()

  directory = './grids/%s'%xsec
  grids = os.listdir(directory)

  parts = ['NM','NCM']

  #--need to handle both NM and NCM
  for part in parts:
      
      PhT, eta    = [],[]
      realN = []
      realM = []
      realT,imagT = [],[]
      #--double Mellin transform
      num_params = 6
      L = len(N)
      length=len(grids)*len(N)

      i = 0
      for idx in range(len(grids)):
          lprint('Generating %s %s'%(part,idx))
          data = load('%s/%s.melltab'%(directory,idx))
          T = data[channel][flav][part]
          eta.extend(data['eta']*np.ones(L))
          PhT.extend(data['PhT']*np.ones(L))
          realN.extend(N)
          realM.extend(M)
          #for i in range(len(N)):
          #    print(N[i],M[i],T.real.flatten()[i])
          #N, M = np.real(dmellin.N), np.real(dmellin.M)
          #for i in range(len(N)):
          #    for j in range(len(M)):
          #        print(N[i],M[j],T.real[i][j])
          realT.extend(T.real.flatten())
          imagT.extend(T.imag.flatten())
          #realT.extend(T.real.flatten('F'))
          #imagT.extend(T.imag.flatten('F'))
          #sys.exit()

      #--consider doubling size of table, reversing N <--> M?

      DATA = {}
      DATA['data']    = np.zeros((num_params,length))
      DATA['data'][0] = eta 
      DATA['data'][1] = PhT 
      DATA['data'][2] = realN 
      DATA['data'][3] = realM
      DATA['data'][4] = realT
      DATA['data'][5] = imagT

      #--smoothing factors
      #--add smoothing factor to PhT here: PhT**6 for UU, PhT**5? for UT
      DATA['factor'] = np.ones((num_params,length))
      DATA['factor'][4] = DATA['data'][1]**6.00
      DATA['factor'][5] = DATA['data'][1]**6.00

      #data = DATA['data']
      #for i in range(len(data[0])):
      #    print(data[0][i],data[1][i],data[2][i],data[3][i],data[4][i],data[5][i])
      #sys.exit()

      #--smooth, then normalize data here
      DATA['data'],DATA['mean'],DATA['std'] = convert(DATA)

      checkdir('modelgrids/%s/%s'%(xsec,part))
      filename='modelgrids/%s/%s/%s-%s.npy'%(xsec,part,channel,flav)
      np.save(filename, DATA)
      print('Saving table to %s'%filename)


      stats = {}
      stats['mean'] = copy.copy(DATA['mean'])
      stats['std']  = copy.copy(DATA['std'])
      checkdir('modelgrids/stats/%s/%s'%(xsec,part))
      filename='modelgrids/stats/%s/%s/%s-%s.npy'%(xsec,part,channel,flav)
      np.save(filename, stats)
      print('Saving stats to %s'%filename)


  return

if __name__ == "__main__":

  xsecs = ['UU','UT']
  xsecs = ['UT']

  channels = {}
  #--unpolarized channels (all are nonzero)
  channels['UU'] =      ['QQ,QQ','QQp,QpQ','QQp,QQp','QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ']
  channels['UU'].extend(['QQB,GG','GQ,GQ','GQ,QG','QG,GQ','QG,QG','GG,GG','GG,QQB'])

  #--polarized channels (only five are nonzero)
  channels['UT'] =      ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ','GQ,QG']
  
 
  for xsec in xsecs: 
      for channel in channels[xsec]:
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
          for flav in flavs:
              if xsec=='UU': gen_table_UU(channel,flav)
              if xsec=='UT': gen_table_UT(channel,flav)





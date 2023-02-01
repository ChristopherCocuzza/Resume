import sys,os,time
import numpy as np
from scipy.special import zeta
from mpmath import fp
#from numba.core import types
#from numba.typed import Dict


#--set_constants 
CA=3
CF=4/3
TR=1/2
TF=1/2
euler=fp.euler 

zeta2 = zeta(2)
zeta3 = zeta(3)
zeta4 = zeta(4)
zeta5 = zeta(5)

#--set_masses
me   = 0.000511
mm   = 0.105658
mt   = 1.77684
mu   = 0.055
md   = 0.055
ms   = 0.2

#--old
mc   = 1.28
mb   = 4.18 

#--mcfm
#mc   = 1.51 
#mb   = 4.92  

mZ   = 91.1876 # compatible with mcfm
mW   = 80.398 # compatible with mcfm
M    = 0.93891897
Mpi  = 0.13803
Mk   = 0.493677
Mdelta = 1.232

me2   = me**2 
mm2   = mm**2 
mt2   = mt**2
mu2   = mu**2  
md2   = md**2  
ms2   = ms**2  
mc2   = mc**2  
mb2   = mb**2  
mZ2   = mZ**2  
mW2   = mW**2  
M2    = M**2  
Mpi2  = Mpi**2  
Mdelta2=Mdelta**2

#--set_ckm #! from mcfm
ckm=np.zeros((3,3))
ckm[0,0]=0.97419 
ckm[0,1]=0.2257 
ckm[0,2]=0.00359
ckm[1,0]=0.2256
ckm[1,1]=0.97334
ckm[1,2]=0.0415

#--set_couplings       
c2w   = mW2/mZ2
s2w   = 1.0-c2w
s2wMZ = 0.23116
c2wMZ = 1.0 - s2wMZ
alfa  = 1/137.036
alphaSMZ = 0.118
GF = 1.1663787e-5   # 1/GeV^2 # compatible with mcfm

#--set_charges
quarkcharges = np.zeros(11)
quarkcharges[0] = 0 # gluon
quarkcharges[1] = 2/3 # u
quarkcharges[2] =-1/3 # d
quarkcharges[3] =-1/3 # s
quarkcharges[4] = 2/3 # c
quarkcharges[5] =-1/3 # b

quarkcharges[-1] =-2/3 # baru
quarkcharges[-2] = 1/3 # bard
quarkcharges[-3] = 1/3 # bars
quarkcharges[-4] =-2/3 # barc
quarkcharges[-5] = 1/3 # barb


#--pQCD setup
mu02_alphaS  = 1.0
mu02_pdf     = mc2
mu02_ff      = mc2

#--RGEs
#--using LO for alphaS
order_alphaS      = 0 #-- 0:LL 1:NLL 2:NNLL 3:NNNLL
#--not sure if these matter?
order_dglap       = 1 #-- 0:LL 1:NLL
order_resummation = 2 #-- 0:LL 1:NLL 2:NNLL 3:NNLL
order_ope         = 1 #-- 0: aS^0 1:LO as^1 1:NLO as^2 










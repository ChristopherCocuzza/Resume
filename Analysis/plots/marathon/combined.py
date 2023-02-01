#!/usr/bin/env python
import sys,os
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerBase
import matplotlib.gridspec as gridspec
import pylab as py

#--from scipy stack 
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from scipy.interpolate import griddata

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
import kmeanconf as kc
from analysis.corelib import core
from analysis.corelib import classifier

cwd = 'plots/marathon/'

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y,box,mode='same')
    return y_smooth

def load_F2nF2p():

  #--load AKP17 and CJ15 data
  F = open('plots/marathon/data/F2nF2p.csv','r')
  L = F.readlines()
  F.close()

  L = [l.strip() for l in L]
  L = [[x for x in l.split()] for l in L]
  L = np.transpose(L)[0]

  X, CJ, AKP = [],[],[]
  for i in range(len(L)):
      if i==0: continue
      X  .append(float(L[i].split(',')[0]))
      AKP.append(float(L[i].split(',')[1]))
      CJ .append(float(L[i].split(',')[2]))

  X,CJ,AKP = np.array(X),np.array(CJ),np.array(AKP)

  x = np.linspace(0.195,0.825,50)

  CJ = smooth(CJ,10)
  AKP = smooth(AKP,10)

  return x,CJ,AKP

def load_F2DF2N(group):

  if group=='AKP17':
      F = open('plots/marathon/data/F2DF2N/AKP.csv','r')
      L = F.readlines()
      F.close()
      
      L = [l.strip() for l in L]
      L = [[x for x in l.split()] for l in L]
      L = np.transpose(L)[0]

      X,upper,lower = [],[],[]

      for i in range(len(L)):
          if i==0: continue
          X    .append(float(L[i].split(',')[0]))
          upper.append(float(L[i].split(',')[1]))
          lower.append(float(L[i].split(',')[2]))

      X,upper,lower=np.array(X),np.array(upper),np.array(lower)
 
  if group=='CJ15':
      filename = 'plots/marathon/data/F2DF2N/CJ15.out'

      F = open(filename,'r')
      L = F.readlines()
      F.close()
      
      L = [l.strip() for l in L]
      L = [[x for x in l.split()] for l in L]
      L = np.transpose(L)
      
      X,thy,err = [],[],[] 
      for i in range(len(L)):
          if len(L[i]) < 9: continue
          if L[i][8]!='test_DN': continue
          if L[i][1] != '10.000': continue
          X  .append(float(L[i][0]))
          thy.append(float(L[i][3]))
          err.append(float(L[i][4]))

      X,thy,err = np.array(X),np.array(thy),np.array(err)

      lower = thy - err
      upper = thy + err

      lower = smooth(lower,10)
      upper = smooth(upper,10)

  return X,lower,upper

if __name__ == "__main__":

  #--top left:     F2D/F2N compared to CJ15 and AKP17
  #--top right:    F2n/F2p compared to no Marathon fit and Marathon + KP Model (from paper)
  #--bottom left:  super ratio compared to KP model
  #--bottom right: d/u ratio compared to no Marathon fit

  nrows,ncols=2,2
  fig  = py.figure(figsize=(ncols*9,nrows*5))
  ax11=py.subplot(nrows,ncols,1)
  ax12=py.subplot(nrows,ncols,2)
  ax21=py.subplot(nrows,ncols,3)
  ax22=py.subplot(nrows,ncols,4)

  filename = '%s/gallery/combined'%cwd

  wdir1 = 'results/marathon/final'
  #wdir1 = 'results/marathon/step30'
  #wdir1 = 'results/marathon/wf/av18'
  #wdir1 = 'results/marathon/wf/cdbonn'
  #wdir1 = 'results/marathon/wf/wjc1'
  #wdir1 = 'results/marathon/wf/wjc2'
  #wdir1 = 'results/marathon/wf/ss'
  #wdir1 = 'results/marathon/more/hanjie'
  #wdir1 = 'results/marathon/GP'
  wdir1 = 'final2'

  wdir2 = 'results/marathon/nomar'
  WDIR = [wdir1,wdir2]

  j = 0
  hand = {}
  JAMcolor = 'red'
  JAMalpha = 0.7
  JAMlw   = 2.0
  for wdir in WDIR:

      load_config('%s/input.py'%wdir)
      istep=core.get_istep()

      #--plot F2D/F2N ratio at Q2 = 10
      Q2 = 10
      data=load('%s/data/stf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))

      cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
      best_cluster=cluster_order[0]

      X  = np.array(data['X'])

      F2p = np.array(data['XF']['p']['F2'])
      F2n = np.array(data['XF']['n']['F2'])
      F2d = np.array(data['XF']['d']['F2'])

      ratDN = 2*F2d/(F2p+F2n)

      meanDN = np.mean(ratDN,axis=0)
      stdDN  = np.std (ratDN,axis=0)

      meanDN = smooth(meanDN,10)
      stdDN  = smooth(stdDN ,10)

      if j==0:
          hand['DNmean'] ,= ax21.plot        (X,meanDN                   ,color=JAMcolor  ,alpha=1.0,zorder=3, lw=JAMlw)
          hand['DNband']  = ax21.fill_between(X,meanDN-stdDN,meanDN+stdDN,color=JAMcolor  ,alpha=JAMalpha,zorder=3)
      if j==1:
          hand['nomar DN']    = ax21.fill_between(X,meanDN-stdDN,meanDN+stdDN,color='gold'  ,alpha=0.5,zorder=1)
          hand['nomar band'] ,= ax21.plot(X,meanDN-stdDN,color='black',alpha=0.5,zorder=1,ls='--')
          ax21                      .plot(X,meanDN+stdDN,color='black',alpha=0.5,zorder=1,ls='--')

      #--plot STF ratios at Marathon kinematics
      data=load('%s/data/stf-marathon-%d.dat'%(wdir,istep))

      cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
      best_cluster=cluster_order[0]

      X  = data['X']

      F2p = np.array(data['XF']['p']['F2'])
      F2n = np.array(data['XF']['n']['F2'])
      F2d = np.array(data['XF']['d']['F2'])
      F2h = np.array(data['XF']['h']['F2'])
      F2t = np.array(data['XF']['t']['F2'])

      ratnp = F2n/F2p
      ratHN = 3*F2h/(2*F2p+F2n)
      ratTN = 3*F2t/(F2p+2*F2n)
      ratHT = ratHN/ratTN

      meannp = np.mean(ratnp,axis=0)
      stdnp  = np.std (ratnp,axis=0)
      meanHT = np.mean(ratHT,axis=0)
      stdHT  = np.std (ratHT,axis=0)

      if j==0:
          ax11.plot        (X,meanHT                   ,color=JAMcolor  ,alpha=1.0 ,zorder=3, lw = JAMlw)
          ax12.plot        (X,meannp                   ,color=JAMcolor  ,alpha=1.0 ,zorder=3, lw = JAMlw)
          ax11.fill_between(X,meanHT-stdHT,meanHT+stdHT,color=JAMcolor  ,alpha=JAMalpha, zorder=3)
          ax12.fill_between(X,meannp-stdnp,meannp+stdnp,color=JAMcolor  ,alpha=JAMalpha ,zorder=3)
      if j==1:
          hand['nomar np'] = ax12.fill_between(X,meannp-stdnp,meannp+stdnp,color='gold'  ,alpha=0.7,zorder=2)
          ax12.plot(X,meannp-stdnp,color='black',alpha=0.5,zorder=1,ls='--')
          ax12.plot(X,meannp+stdnp,color='black',alpha=0.5,zorder=1,ls='--')
          #hand['nomar HT'] = ax11.fill_between(X,meanHT-stdHT,meanHT+stdHT,color='gold'  ,alpha=0.3 ,zorder=1)

      #--plot d/u
      data=load('%s/data/pdf-%d-Q2=10.dat'%(wdir,istep))

      X = np.array(data['X'])
      rat = np.array(data['XF']['d/u'])

      mean = np.mean(rat,axis=0)
      std  = np.std (rat,axis=0)

      if j==0:
          ax22.plot        (X,mean             ,color=JAMcolor  ,alpha=1.0 ,zorder=3, lw = JAMlw)
          ax22.fill_between(X,mean-std,mean+std,color=JAMcolor  ,alpha=JAMalpha ,hatch=None,zorder=3)
      if j==1:
          hand['nomar du'] = ax22.fill_between(X,mean-std,mean+std,color='gold',alpha=0.7,zorder=2)
          ax22.plot(X,mean-std,color='black',alpha=0.5,zorder=1,ls='--')
          ax22.plot(X,mean+std,color='black',alpha=0.5,zorder=1,ls='--')


      j += 1


  #--plot data from Marathon
  X   = np.array([0.195,0.225,0.255,0.285,0.315,0.345,0.375,0.405,0.435,0.465,0.495,0.525,0.555,0.585,0.615,0.645,0.675,0.705,0.735,0.765,0.795,0.825])
  Rht = np.array([0.9989,0.9990,0.9991,0.9993,0.9997,1.0003,1.0010,1.0019,1.0029,1.0039,1.0049,1.0058,1.0067,1.0074,1.0081,1.0087,1.0093,1.0098,1.0104,1.0111,1.0118,1.0125])
  err = np.array([0.0009,0.0009,0.0009,0.0008,0.0009,0.0008,0.0008,0.0008,0.0007,0.0007,0.0007,0.0007,0.0007,0.0008,0.0009,0.0010,0.0013,0.0017,0.0020,0.0024,0.0030,0.0043])
  hand['Marathon HT']     ,= ax11.plot        (X,Rht            ,color='black',alpha=1.0,zorder=5)
  hand['Marathon HT band'] = ax11.fill_between(X,Rht-err,Rht+err,color='black',alpha=0.3,zorder=5)
  Rnp = [0.724,0.701,0.668,0.635,0.647,0.618,0.610,0.547,0.567,0.540,0.528,0.496,0.489,0.489,0.489,0.461,0.466,0.442,0.451,0.436,0.441,0.455]
  err = [0.020,0.019,0.019,0.019,0.019,0.019,0.021,0.020,0.021,0.020,0.020,0.020,0.020,0.020,0.021,0.020,0.022,0.020,0.020,0.020,0.022,0.024]
  hand['Marathon np'] = ax12.errorbar(X,Rnp,yerr=err,fmt='o',ms=4.0,capsize=0.0,color='black',zorder=6,alpha=0.7)

  #--plot n/p data from BONuS
  #X    = np.array([0.1770,0.2250,0.2751,0.3235,0.3719,0.4215,0.4718,0.5205,0.5730,0.6155])
  #Rnp  = np.array([0.7285,0.7143,0.7007,0.6865,0.6574,0.6539,0.6136,0.6157,0.5269,0.4188])
  #stat = np.array([0.0062,0.0051,0.0053,0.0061,0.0073,0.0092,0.0126,0.0172,0.0303,0.0684])
  #syst = np.array([0.0381,0.0375,0.0369,0.0365,0.0354,0.0359,0.0348,0.0363,0.0331,0.0284])
  #norm = 0.07*Rnp
  #err  = np.sqrt(stat**2 + syst**2 + norm**2)
  #hand['BONuS np'] = ax12.errorbar(X,Rnp,yerr=err,fmt='o',ms=4.0,capsize=0.0,color='green',zorder=6,alpha=0.7)


  #--plot CJ15 and AKP17 F2D/F2N
  X,lower,upper = load_F2DF2N('AKP17')
  hand['AKP17 DN']  = ax21.fill_between(X,lower,upper,color='darkcyan',alpha=0.3,zorder=1)

  X,lower,upper = load_F2DF2N('CJ15')
  hand['CJ15 DN'] = ax21.fill_between(X,lower,upper,color='lightgreen',alpha=0.5,zorder=1)

  #--plot CJ15 F2n/F2p
  X,CJ,AKP = load_F2nF2p()

  #hand['CJ15 np'] ,= ax12.plot(X,CJ,color='gold',ls='-',zorder=5,lw=2.0)
 
  #--plot CJ15 d/u 
  #CJ = load('plots/marathon/data/PDFs/CJ15.dat')
  #X     = CJ['X']
  #lower = CJ['XF']['d/u']['xfmin'].T[0]
  #upper = CJ['XF']['d/u']['xfmax'].T[0]
  #ax22.fill_between(X,lower,upper,color='gold',alpha=0.5,zorder=1)


  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)
  ax12.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)
  ax21.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)
  ax22.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)

  ax11.axhline(1,0,1,ls=':',color='black',alpha=0.5)
  ax21.axhline(1,0,1,ls=':',color='black',alpha=0.5)

  ax11.set_ylim(0.94,1.04)
  ax12.set_ylim(0.25,0.80)
  ax21.set_ylim(0.96,1.09)
  ax22.set_ylim(0.01,0.59)

  ax11.text(0.10,0.75,r'\boldmath$\mathcal{R}$'               ,transform=ax11.transAxes,size=65)
  ax12.text(0.67,0.75,r'\boldmath$F_2^n/F_2^p$'               ,transform=ax12.transAxes,size=50)
  ax21.text(0.40,0.80,r'\boldmath$R(D)$'                      ,transform=ax21.transAxes,size=50)
  ax22.text(0.78,0.80,r'\boldmath$d/u$'                       ,transform=ax22.transAxes,size=50)

  ax11.text(0.50,0.07,r'$Q^2 = 14 x ~ \rm{GeV^2}$'      ,transform=ax11.transAxes,size=30)
  ax12.text(0.65,0.60,r'$Q^2 = 14 x ~ \rm{GeV^2}$'      ,transform=ax12.transAxes,size=30)
  ax21.text(0.02,0.05,r'$Q^2 = 10   ~ \rm{GeV^2}$'      ,transform=ax21.transAxes,size=30)
  ax22.text(0.65,0.60,r'$Q^2 = 10   ~ \rm{GeV^2}$'      ,transform=ax22.transAxes,size=30)

  for ax in [ax11,ax12,ax21,ax22]:
      minorLocator = MultipleLocator(0.02)
      majorLocator = MultipleLocator(0.2)
      ax.xaxis.set_minor_locator(minorLocator)
      ax.xaxis.set_major_locator(majorLocator)
      ax.xaxis.set_tick_params(which='major',length=6)
      ax.xaxis.set_tick_params(which='minor',length=3)
      ax.yaxis.set_tick_params(which='major',length=6)
      ax.yaxis.set_tick_params(which='minor',length=3)
      ax.set_xlim(0.15,0.85)
      ax.set_xticks([0.2,0.4,0.6,0.8])

  for ax in [ax21,ax22]:
      ax.set_xlabel(r'\boldmath$x$',size=50)
      ax.xaxis.set_label_coords(0.50,-0.05)

  for ax in [ax11,ax21]:
      minorLocator = MultipleLocator(0.01)
      majorLocator = MultipleLocator(0.05)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  for ax in [ax12,ax22]:
      minorLocator = MultipleLocator(0.05)
      majorLocator = MultipleLocator(0.1)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  ax12.set_yticks([0.3,0.4,0.5,0.6,0.7])

  class AnyObjectHandler(HandlerBase):

      def create_artists(self,legend,orig_handle,x0,y0,width,height,fontsize,trans):
          l1 = py.Line2D([x0,y0+width], [1.0*height,1.0*height], color = 'gray', alpha = 0.5, ls = '--')
          l2 = py.Line2D([x0,y0+width], [0.0*height,0.0*height], color = 'gray', alpha = 0.5, ls = '--')
          return [l1,l2]

  ax11.legend([(object,hand['nomar np'])],['label'],handler_map={object:AnyObjectHandler()})

  handles,labels = [],[]
  handles.append((hand['DNband'],hand['DNmean']))
  handles.append((hand['Marathon HT'],hand['Marathon HT band']))
  labels.append(r'\textrm{\textbf{JAM}}')
  labels.append(r'\textbf{\textrm{KP model}}')
  ax11.legend(handles,labels,frameon=False,loc=(0.00,0.02),fontsize=30, handletextpad = 0.5, handlelength = 1.2, ncol = 1, columnspacing = 0.5)

  handles,labels = [],[]
  handles.append((hand['DNband'],hand['DNmean']))
  handles.append((hand['nomar np'],object))
  handles.append(hand['Marathon np'])
  labels.append(r'\textrm{\textbf{JAM}}')
  labels.append(r'\textbf{\textrm{JAM (no {\huge MARATHON})}}')
  labels.append(r'\textbf{\textrm{{\huge MARATHON} + KP model}}')
  ax12.legend(handles,labels,frameon=False,loc=(0.00,0.02),fontsize=30, handletextpad = 0.5, handlelength = 1.2, ncol = 1, columnspacing = 0.5, handler_map = {object:AnyObjectHandler()})

  handles,labels = [],[]
  handles.append((hand['DNband'],hand['DNmean']))
  handles.append(hand['CJ15 DN'])
  handles.append(hand['AKP17 DN'])
  handles.append((hand['nomar DN'],object))
  labels.append(r'\textrm{\textbf{JAM}}')
  labels.append(r'\textrm{\textbf{CJ15}}')
  labels.append(r'\textrm{\textbf{AKP17}}')
  labels.append(r'\textbf{\textrm{JAM (no {\huge MARATHON})}}')
  ax21.legend(handles,labels,frameon=False,loc=(0.00,0.41),fontsize=30, handletextpad = 0.5, handlelength = 1.2, ncol = 1, columnspacing = 0.5, handler_map = {object:AnyObjectHandler()})

  handles,labels = [],[]
  handles.append((hand['DNband'],hand['DNmean']))
  handles.append((hand['nomar du'],object))
  labels.append(r'\textrm{\textbf{JAM}}')
  labels.append(r'\textbf{\textrm{JAM (no {\huge MARATHON})}}')
  ax22.legend(handles,labels,frameon=False,loc=(0.00,0.02),fontsize=30, handletextpad = 0.5, handlelength = 1.2, ncol = 1, columnspacing = 0.5, handler_map = {object:AnyObjectHandler()})

  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename+='.png'
  #filename+='.pdf'
  ax11.set_rasterized(True)
  ax12.set_rasterized(True)
  ax21.set_rasterized(True)
  ax22.set_rasterized(True)

  checkdir('%s/gallery'%cwd)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()






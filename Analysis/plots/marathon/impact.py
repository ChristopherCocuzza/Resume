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

if __name__ == "__main__":

  #--top left:  Show impact ratios for super ratio, F2n/F2p, F2D/F2N, and d/u

  nrows,ncols=1,2
  fig  = py.figure(figsize=(ncols*9,nrows*5))
  ax11=py.subplot(nrows,ncols,1)

  filename = '%s/gallery/impact'%cwd

  wdir1 = 'step18pos'
  #wdir1 = 'results/marathon/step30'
  #wdir1 = 'results/marathon/wf/av18'
  #wdir1 = 'results/marathon/wf/cdbonn'
  #wdir1 = 'results/marathon/wf/wjc1'
  #wdir1 = 'results/marathon/wf/wjc2'
  #wdir1 = 'results/marathon/wf/ss'
  #wdir1 = 'results/marathon/more/hanjie'

  wdir2 = 'nomar'
  WDIR = [wdir1,wdir2]

  j = 0
  hand = {}

  ratHT,ratDN,ratnp,ratdu = {},{},{},{}

  for wdir in WDIR:

      load_config('%s/input.py'%wdir)
      istep=core.get_istep()

      #--F2D/F2N at Q2 = 10
      Q2 = 10
      data=load('%s/data/stf-%d-Q2=%d.dat'%(wdir,istep,int(Q2)))

      cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
      best_cluster=cluster_order[0]

      XSTF  = np.array(data['X'])

      F2p = np.array(data['XF']['p']['F2'])
      F2n = np.array(data['XF']['n']['F2'])
      F2d = np.array(data['XF']['d']['F2'])

      ratDN[j] = 2*F2d/(F2p+F2n)

      #--STF ratios at Marathon kinematics
      data=load('%s/data/stf-marathon-%d.dat'%(wdir,istep))

      cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
      best_cluster=cluster_order[0]

      XMAR  = data['X']

      F2p = np.array(data['XF']['p']['F2'])
      F2n = np.array(data['XF']['n']['F2'])
      F2d = np.array(data['XF']['d']['F2'])
      F2h = np.array(data['XF']['h']['F2'])
      F2t = np.array(data['XF']['t']['F2'])

      ratnp[j] = F2n/F2p
      ratHN    = 3*F2h/(2*F2p+F2n)
      ratTN    = 3*F2t/(F2p+2*F2n)
      ratHT[j] = ratHN/ratTN


      #--d/u
      data=load('%s/data/pdf-%d-Q2=10.dat'%(wdir,istep))

      XPDF = np.array(data['X'])
      ratdu[j] = np.array(data['XF']['d/u'])


      j += 1

 
  #--calculate impacts
  impactHT  = np.std(ratHT[0],axis=0)/np.std(ratHT[1],axis=0)*np.mean(ratHT[1],axis=0)/np.mean(ratHT[0],axis=0)
  impactDN  = np.std(ratDN[0],axis=0)/np.std(ratDN[1],axis=0)*np.mean(ratDN[1],axis=0)/np.mean(ratDN[0],axis=0)
  impactnp  = np.std(ratnp[0],axis=0)/np.std(ratnp[1],axis=0)*np.mean(ratnp[1],axis=0)/np.mean(ratnp[0],axis=0)
  impactdu  = np.std(ratdu[0],axis=0)/np.std(ratdu[1],axis=0)*np.mean(ratdu[1],axis=0)/np.mean(ratdu[0],axis=0)

  #meanDN = smooth(meanDN,10)
  #stdDN  = smooth(stdDN ,10)

  hand['HT'] ,= ax11.plot(XMAR,impactHT,color='magenta'  ,lw = 2.5)
  hand['DN'] ,= ax11.plot(XSTF,impactDN,color='firebrick',lw = 2.5)
  hand['np'] ,= ax11.plot(XMAR,impactnp,color='darkblue' ,lw = 2.5)
  hand['du'] ,= ax11.plot(XPDF,impactdu,color='darkgreen',lw = 2.5)

  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)

  ax11.axhline(1,0,1,ls=':',color='black',alpha=0.5)

  ax11.set_ylim(0.00,1.30)

  ax11.text(0.05,0.10,r'\boldmath$\sigma_{\rm rel}^{\rm MAR}/\sigma_{\rm rel}$',transform=ax11.transAxes,size=40)

  for ax in [ax11]:
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

  for ax in [ax11]:
      ax.set_xlabel(r'\boldmath$x$',size=50)
      ax.xaxis.set_label_coords(0.50,-0.05)

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.05)
      majorLocator = MultipleLocator(0.2)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  ax11.set_yticks([0.2,0.4,0.6,0.8,1.0,1.2])

  handles,labels = [],[]
  handles.append(hand['HT'])
  handles.append(hand['np'])
  handles.append(hand['DN'])
  handles.append(hand['du'])
  labels.append(r'\boldmath$\mathcal{R}$')
  labels.append(r'\boldmath$F_2^n/F_2^p$')
  labels.append(r'\boldmath$R(D)$')
  labels.append(r'\boldmath$d/u$')
  ax11.legend(handles,labels,frameon=False,loc=(1.02,0.30),fontsize=30, handletextpad = 0.5, handlelength = 1.2, ncol = 1, columnspacing = 0.5)

  ax11.text(1.02,0.15,r'\boldmath$\mathcal{R},F_2^n/F_2^p~{\rm at}~Q^2 = 14x~{\rm GeV}^2$',transform=ax11.transAxes,size=25)
  ax11.text(1.02,0.05,r'\boldmath$R(D),d/u~{\rm at}~Q^2 = 10~{\rm GeV}^2$',transform=ax11.transAxes,size=25)

  py.tight_layout()
  #py.subplots_adjust(hspace=0)

  filename+='.png'
  ax11.set_rasterized(True)

  checkdir('%s/gallery'%cwd)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()






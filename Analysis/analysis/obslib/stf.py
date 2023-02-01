import sys,os
import numpy as np
import time
import argparse
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import pylab as py

#--from scipy stack 
from scipy.integrate import quad
from scipy.integrate import cumtrapz

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.corelib import core
from analysis.corelib import classifier

import kmeanconf as kc

#--unpolarized NC and CC generation
def gen_stf(wdir,Q2=None,tar='p',stf='F2'):
  
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating STF from %s for %s %s at Q2 = %3.5f'%(wdir,stf,tar,Q2))

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    #--setup kinematics for structure functions to be calculated at
    Xgrid = np.geomspace(1e-5,1e-1,20)
    Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    Q2grid = [Q2*0.99,Q2*1.01]
    conf['idis grid'] = {}
    conf['idis grid']['X']  = Xgrid 
    conf['idis grid']['Q2'] = Q2grid 
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
    
    if tar in ['p','n']:
        idis.data[tar] = {}
        idis.data[tar][stf] = np.zeros(idis.X.size)
    else:
        idis.data[tar] = {}
        idis.data['p'][stf] = np.zeros(idis.X.size)
        idis.data['n'][stf] = np.zeros(idis.X.size)
        idis.data[tar][stf] = np.zeros(idis.X.size)
        if stf=='FL':
            idis.data['p']['F2'] = np.zeros(idis.X.size)
            idis.data['n']['F2'] = np.zeros(idis.X.size)
            idis.data[tar]['F2'] = np.zeros(idis.X.size)

    if 'off pdf' in conf:
        idis.data['p']['F2off d'] = np.zeros(idis.X.size)
        idis.data['n']['F2off d'] = np.zeros(idis.X.size)
        idis.data['p']['F2off h'] = np.zeros(idis.X.size)
        idis.data['n']['F2off h'] = np.zeros(idis.X.size)
        idis.data['p']['F2off t'] = np.zeros(idis.X.size)
        idis.data['n']['F2off t'] = np.zeros(idis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics for structure functions to be interpolated to
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))

    #--compute X*STF for all replicas        
    XF=[]
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        pdf.evolve(Q2)
        idis._update()

        xf = X*idis.get_stf(X,Q2,stf=stf,tar=tar)
        XF.append(xf)

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_stf(wdir,Q2=None,mode=0,TAR=['p','n','d','h','t'],STF=['F2','FL','F3']):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,3
  fig = py.figure(figsize=(ncols*7,nrows*4))
  ax11=py.subplot(nrows,ncols,1)
  ax12=py.subplot(nrows,ncols,2)
  ax13=py.subplot(nrows,ncols,3)

  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  hand = {}

  for tar in TAR:
      if tar not in hand: hand[tar] = {}
      for stf in STF:
          filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)
          #--load data if it exists
          try:
              data=load(filename)
          #--generate data and then load it if it does not exist
          except:
              gen_stf(wdir,Q2,tar,stf)
              data=load(filename)
          X    = data['X']
          data = data['XF']
          mean = np.mean(data,axis=0)
          std  = np.std (data,axis=0)

          if tar=='p':   color='red'
          elif tar=='n': color='green'
          elif tar=='d': color='blue'
          elif tar=='h': color='magenta'
          elif tar=='t': color='orange'
          else: continue

          label = None
          if stf =='F2':   ax = ax11
          elif stf =='FL': ax = ax12
          elif stf =='F3': ax = ax13
          else: continue

          #--plot each replica
          if mode==0:
              for i in range(len(data)):
                  hand[tar][stf] ,= ax.plot(X,data[i],color=color,alpha=0.1)
    
          #--plot average and standard deviation
          if mode==1:
              ax.plot(X,mean,color=color)
              hand[tar][stf] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)


  for ax in [ax11,ax12,ax13]:
        ax.set_xlim(0,0.9)
          
        ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)


  ax11.text(0.80,0.70,r'\boldmath$xF_2$',transform=ax11.transAxes,size=40)
  ax12.text(0.80,0.70,r'\boldmath$xF_L$',transform=ax12.transAxes,size=40)
  ax13.text(0.60,0.70,r'\boldmath$xF_3$',transform=ax13.transAxes,size=40)

  handles,labels = [],[]

  handles.append(hand['p']['F2'])
  handles.append(hand['n']['F2'])
  handles.append(hand['d']['F2'])
  handles.append(hand['h']['F2'])
  handles.append(hand['t']['F2'])

  labels.append(r'\boldmath$p$')
  labels.append(r'\boldmath$n$')
  labels.append(r'\boldmath$D$')
  labels.append(r'\boldmath$^3{\rm He}$')
  labels.append(r'\boldmath$^3{\rm H}$')

  ax13.legend(handles,labels,loc='upper right',fontsize=20, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)

  ax11.set_ylim(0,0.1)      
  ax12.set_ylim(0,0.004)    
  ax13.set_ylim(0,0.00045)  

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

  filename = '%s/gallery/stfs-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

def plot_F2_rat(wdir,Q2=None,num='d',den='p',mode=1):

  #--plot F2 ratios
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,1
  fig = py.figure(figsize=(ncols*7,nrows*5))
  ax11=py.subplot(nrows,ncols,1)

  replicas=core.get_replicas(wdir)

  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  stf = 'F2'
  filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,num,stf,Q2)
  #--load data if it exists
  try:
      NUM=load(filename)
  #--generate data and then load it if it does not exist
  except:
      gen_stf(wdir,Q2,num,stf)
      NUM=load(filename)

  filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,den,stf,Q2)
  #--load data if it exists
  try:
      DEN=load(filename)
  #--generate data and then load it if it does not exist
  except:
      gen_stf(wdir,Q2,den,stf)
      DEN=load(filename)


  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  X  = NUM['X']

  F2num = np.array(NUM['XF'])
  F2den = np.array(DEN['XF'])

  rat = F2num/F2den

  mean = np.mean(rat,axis=0)
  std  = np.std (rat,axis=0)

  #--plot each replica
  if mode==0:
      for i in range(len(rat)):
          ax11.plot(X,rat[i],color='red',alpha=0.1)
  
  #--plot average and standard deviation
  if mode==1:
      ax11.fill_between(X,mean-std,mean+std,color='red',alpha=0.7)


  for ax in [ax11,ax11]:
        ax.set_xlim(0,1.0)
        ax.set_xticks([0.2,0.4,0.6,0.8])


  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)

  ax11.set_ylim(0.24,1.05)

  ax11.text(0.05, 0.10, r'\boldmath$F_2^%s/F_2^%s$'%(num,den),transform=ax11.transAxes,size=40)
  ax11.set_xlabel(r'\boldmath$x$'          ,size=30)
  ax11.xaxis.set_label_coords(0.98,0)

  if Q2 == 1.27**2: ax11.text(0.30,0.85,r'$Q^2 = m_c^2$',              transform=ax11.transAxes,size=30)
  else:             ax11.text(0.30,0.85,r'$Q^2 = %s ~ \rm{GeV^2}$'%Q2, transform=ax11.transAxes,size=30)

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.04)
      majorLocator = MultipleLocator(0.2)
      ax.xaxis.set_minor_locator(minorLocator)
      ax.xaxis.set_major_locator(majorLocator)
      ax.xaxis.set_tick_params(which='major',length=6)
      ax.xaxis.set_tick_params(which='minor',length=3)
      ax.yaxis.set_tick_params(which='major',length=6)
      ax.yaxis.set_tick_params(which='minor',length=3)

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.04)
      majorLocator = MultipleLocator(0.2)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  ax.set_xticks([0.2,0.4,0.6,0.8])

  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/stf-F2%s-F2%s-rat-Q2=%3.5f'%(wdir,num,den,Q2)
  if mode==1: filename += '-bands'
  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

def plot_EMC_rat(wdir,Q2=None,mode=0):

  #--plot EMC ratios for F2D, F2H, and F2T.  As well as super ratio for F2H/F2T
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,1
  fig = py.figure(figsize=(ncols*9,nrows*5))
  ax11=py.subplot(nrows,ncols,1)

  replicas=core.get_replicas(wdir)

  hand = {}
  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  stf = 'F2'
  TAR = ['p','n','d','h','t']
  data = {}
  for tar in TAR:
      filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)
      #--load data if it exists
      try:
          data[tar]=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_stf(wdir,Q2,tar,stf)
          data[tar]=load(filename)

  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  X  = data['p']['X']

  F2p = np.array(data['p']['XF'])
  F2n = np.array(data['n']['XF'])
  F2d = np.array(data['d']['XF'])
  F2h = np.array(data['h']['XF'])
  F2t = np.array(data['t']['XF'])

  ratDN = 2*F2d/(F2p+F2n)
  ratHN = 3*F2h/(2*F2p+F2n)
  ratTN = 3*F2t/(F2p+2*F2n)

  ratHT = ratHN/ratTN

  meanDN = np.mean(ratDN,axis=0)
  stdDN  = np.std (ratDN,axis=0)
  meanHN = np.mean(ratHN,axis=0)
  stdHN  = np.std (ratHN,axis=0)
  meanTN = np.mean(ratTN,axis=0)
  stdTN  = np.std (ratTN,axis=0)
  meanHT = np.mean(ratHT,axis=0)
  stdHT  = np.std (ratHT,axis=0)

  #--plot each replica
  if mode==0:
      for i in range(len(ratHN)):
          hand['DN'] ,= ax11.plot(X,ratDN[i],color='firebrick',alpha=0.1)
          hand['HN'] ,= ax11.plot(X,ratHN[i],color='darkgreen',alpha=0.1)
          hand['TN'] ,= ax11.plot(X,ratTN[i],color='blue',alpha=0.1)
          #hand['HT'] ,= ax11.plot(X,ratHT[i],color='magenta',alpha=0.1)
  
  #--plot average and standard deviation
  if mode==1:
      hand['DN'] = ax11.fill_between(X,meanDN-stdDN,meanDN+stdDN,color='firebrick',alpha=0.5,hatch='+')
      hand['HN'] = ax11.fill_between(X,meanHN-stdHN,meanHN+stdHN,color='darkgreen',alpha=0.4,hatch=None)
      hand['TN'] = ax11.fill_between(X,meanTN-stdTN,meanTN+stdTN,color='blue'     ,alpha=0.4,hatch=None)
      #hand['HT'] = ax11.fill_between(X,meanHT-stdHT,meanHT+stdHT,color='magenta'  ,alpha=0.4,hatch=None)
      #hand['DN'] ,= ax11.plot(X,meanDN+stdDN,color='firebrick',alpha=1.0,ls='-')
      #hand['HN'] ,= ax11.plot(X,meanHN+stdHN,color='darkgreen',alpha=1.0,ls='--')
      #hand['TN'] ,= ax11.plot(X,meanTN+stdTN,color='blue'     ,alpha=1.0,ls=':')
      #hand['HT'] ,= ax11.plot(X,meanHT+stdHT,color='magenta'  ,alpha=1.0,ls='-.')
      #ax11              .plot(X,meanDN-stdDN,color='firebrick',alpha=1.0,ls='-')
      #ax11              .plot(X,meanHN-stdHN,color='darkgreen',alpha=1.0,ls='--')
      #ax11              .plot(X,meanTN-stdTN,color='blue'     ,alpha=1.0,ls=':')
      #ax11              .plot(X,meanHT-stdHT,color='magenta'  ,alpha=1.0,ls='-.')


  for ax in [ax11]:
        ax.set_xlim(0,0.9)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        #ax.set_xlim(1e-4,0.9)
        #ax.semilogx()


  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)

  ax11.axhline(1,0,1,ls='--',color='black',alpha=0.5)

  ax11.set_ylim(0.93,1.09)

  ax11.text(0.05, 0.15, r'\textrm{\textbf{EMC Ratios}}',size=30, transform=ax11.transAxes)
  ax11.set_xlabel(r'\boldmath$x$'          ,size=30)
  ax11.xaxis.set_label_coords(0.97,0)

  if Q2 == 1.27**2: ax11.text(0.05,0.05,r'$Q^2 = m_c^2$',              transform=ax11.transAxes,size=30)
  else:             ax11.text(0.05,0.05,r'$Q^2 = %s ~ \rm{GeV^2}$'%Q2, transform=ax11.transAxes,size=30)

  #for ax in [ax11]:

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.01)
      majorLocator = MultipleLocator(0.05)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  handles,labels = [],[]
  handles.append(hand['DN'])
  handles.append(hand['HN'])
  handles.append(hand['TN'])
  #handles.append(hand['HT'])
  labels.append(r'\boldmath$R(D)$')
  labels.append(r'\boldmath$R(^3{\rm He})$')
  labels.append(r'\boldmath$R(^3{\rm H})$')
  #labels.append(r'\boldmath$R(^3{\rm He})/R(^3{\rm H})$')
  #labels.append(r'\boldmath$\mathcal{R}$')
  ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=25, handletextpad = 0.5, handlelength = 1.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/stf-EMC-rat-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'
  filename+='.pdf'

  ax11.set_rasterized(True)

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

#--charged current
def plot_CCstf(wdir,Q2=None,mode=0):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,3
  fig = py.figure(figsize=(ncols*7,nrows*4))
  ax11=py.subplot(nrows,ncols,1)
  ax12=py.subplot(nrows,ncols,2)
  ax13=py.subplot(nrows,ncols,3)

  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  hand = {}

  tar = 'p'
  STF = ['W2+','WL+','W3+','W2-','WL-','W3-']  

  for stf in STF:
      filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)
      #--load data if it exists
      try:
          data=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_stf(wdir,Q2,tar,stf)
          data=load(filename)
      X    = data['X']
      data = data['XF']
      mean = np.mean(data,axis=0)
      std  = np.std (data,axis=0)

      if stf[-1]=='+': color='red'
      if stf[-1]=='-': color='blue'

      label = None
      if stf =='W2+':   ax = ax11
      elif stf =='WL+': ax = ax12
      elif stf =='W3+': ax = ax13
      elif stf =='W2-': ax = ax11
      elif stf =='WL-': ax = ax12
      elif stf =='W3-': ax = ax13
      else: continue

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              hand[stf] ,= ax.plot(X,data[i],color=color,alpha=0.1)
  
      #--plot average and standard deviation
      if mode==1:
          ax.plot(X,mean,color=color)
          hand[stf] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)


  for ax in [ax11,ax12,ax13]:
        ax.set_xlim(2e-4,1)
        ax.semilogx()
        ax.set_xlabel(r'\boldmath$x$'          ,size=30)
        ax.xaxis.set_label_coords(0.90,0)
          
        ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20,pad=7.0)
        ax.set_xticks([0.001,0.01,0.1,1])
        ax.set_xticklabels([r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$1$'])


  ax11.text(0.40,0.70,r'\boldmath$xF_2^p$',transform=ax11.transAxes,size=40)
  ax12.text(0.40,0.70,r'\boldmath$xF_L^p$',transform=ax12.transAxes,size=40)
  ax13.text(0.40,0.70,r'\boldmath$xF_3^p$',transform=ax13.transAxes,size=40)
  
  ax13.axhline(0,0,1,ls='--',color='black',alpha=0.5)

  handles,labels = [],[]

  handles.append(hand['W2+'])
  handles.append(hand['W2-'])

  labels.append(r'\boldmath$W^+$')
  labels.append(r'\boldmath$W^-$')
  ax11.legend(handles,labels,loc='upper left',fontsize=30, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)

  ax11.set_ylim(0,0.4)      #,ax11.set_yticks([0,0.2,0.4,0.6,0.8])
  ax12.set_ylim(0,0.015)    #,ax12.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
  ax13.set_ylim(-1.0,2.0)   #,ax13.set_yticks([0,0.2,0.4,0.6])

  if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
  else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)

  py.tight_layout()

  filename = '%s/gallery/CCstfs-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()




#--comparisons with Marathon (F2 only)
def gen_marathon_stf(wdir,tar='p'):
  
    print('\ngenerating STF at MARATHON kinematics from %s for %s'%(wdir,tar))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    #--setup kinematics for structure functions to be calculated at
    #--marathon kinematics
    Xgrid   = np.array([0.195,0.225,0.255,0.285,0.315,0.345,0.375,0.405,0.435,0.465,0.495,0.525,0.555,0.585,0.615,0.645,0.675,0.705,0.735,0.765,0.795,0.825])
    Q2grid  = 14*Xgrid
    conf['idis grid'] = {}
    conf['idis grid']['X']  = Xgrid 
    conf['idis grid']['Q2'] = Q2grid 
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
   
    if tar in ['p','n']: 
        idis.data[tar]['F2'] = np.zeros(idis.X.size)
    else:
        idis.data[tar]['F2'] = np.zeros(idis.X.size)
        idis.data['p']['F2'] = np.zeros(idis.X.size)
        idis.data['n']['F2'] = np.zeros(idis.X.size)


    if 'off pdf' in conf:
        idis.data['p']['F2off d'] = np.zeros(idis.X.size)
        idis.data['n']['F2off d'] = np.zeros(idis.X.size)
        idis.data['p']['F2off h'] = np.zeros(idis.X.size)
        idis.data['n']['F2off h'] = np.zeros(idis.X.size)
        idis.data['p']['F2off t'] = np.zeros(idis.X.size)
        idis.data['n']['F2off t'] = np.zeros(idis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics for structure functions to be interpolated to
    X   = np.array([0.195,0.225,0.255,0.285,0.315,0.345,0.375,0.405,0.435,0.465,0.495,0.525,0.555,0.585,0.615,0.645,0.675,0.705,0.735,0.765,0.795,0.825])
    Q2  = 14*X

    #--compute X*STF for all replicas        
    XF=[]
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        #pdf.evolve(Q2)
        idis._update()

        xf = X*idis.get_stf(X,Q2,stf='F2',tar=tar)
        XF.append(xf)

    print() 
    checkdir('%s/data'%wdir)
    filename='%s/data/stf-marathon-%d.dat'%(wdir,istep)
    filename ='%s/data/stf-marathon-%s.dat'%(wdir,tar)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)


#--off-shell structure functions (F2 only)
def gen_off_stf(wdir,Q2=None,nucleus='d'):
   
    load_config('%s/input.py'%wdir)
    if Q2==None: Q2 = conf['Q20']
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   
    print('\ngenerating offshell STF from %s for %s at Q2=%3.5f'%(wdir,nucleus,Q2))

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    #--setup kinematics for structure functions to be calculated at
    Xgrid = np.geomspace(1e-5,1e-1,20)
    Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    Q2grid = [Q2*0.99,Q2*1.01]
    conf['idis grid'] = {}
    conf['idis grid']['X']  = Xgrid 
    conf['idis grid']['Q2'] = Q2grid 
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
   
    if 'off pdf' in conf:
        idis.data['p']['F2off d'] = np.zeros(idis.X.size)
        idis.data['n']['F2off d'] = np.zeros(idis.X.size)
        idis.data['p']['F2off h'] = np.zeros(idis.X.size)
        idis.data['n']['F2off h'] = np.zeros(idis.X.size)
        idis.data['p']['F2off t'] = np.zeros(idis.X.size)
        idis.data['n']['F2off t'] = np.zeros(idis.X.size)
    else: return

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    pdf=conf['pdf']
    #--setup kinematics for structure functions to be interpolated to
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.98,100))
    XM , gXM = np.meshgrid(X , idis.gX)
    Q2M, gWM = np.meshgrid(Q2, idis.gW)
    a = XM

    #--compute X*STF for all replicas        
    XF=[]
    cnt=0
    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        pdf.evolve(Q2)
        idis._update()

        if   nucleus=='d': p, n = 1,1
        elif nucleus=='h': p, n = 2,1
        elif nucleus=='t': p, n = 1,2


        #--retrieve smearing class and upper limit of integration 
        if   nucleus=='d': smf, b = idis.dsmf,idis.ymaxD #--deuterium
        elif nucleus=='h': smf, b = idis.hsmf,idis.ymaxH #--helium
        elif nucleus=='t': smf, b = idis.hsmf,idis.ymaxT #--tritium

        switch = False
        #--tritium takes helium and switches p <--> n
        if nucleus=='t': switch = True

        YM   = 0.5*(b-a)*gXM+0.5*(a+b)
        JM   = 0.5*(b-a) 
        XM_YM = XM/YM
       
        if p==n:
            fof22p = smf.get_fXX2('f22','offshell',XM,Q2M,YM)
            fof22n = fof22p
        else:
            fof22p = smf.get_fXX2('f22p','offshell',XM,Q2M,YM)
            fof22n = smf.get_fXX2('f22n','offshell',XM,Q2M,YM)
        if switch: 
            fof22p, fof22n = fof22n[:], fof22p[:]
        F2poff = idis.get_stf(XM_YM,Q2M,stf='F2',tar='p',off=True,nucleus=nucleus)
        F2noff = idis.get_stf(XM_YM,Q2M,stf='F2',tar='n',off=True,nucleus=nucleus)
        integ  = p*fof22p*F2poff + n*fof22n*F2noff

        result = X*np.einsum('ij,ij,ij->j',gWM,JM,integ)/(p+n)
        XF.append(result)

    print() 
    checkdir('%s/data'%wdir)
    filename ='%s/data/off-stf-%s-Q2=%3.5f.dat'%(wdir,nucleus,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_off_stf(wdir,Q2=None,mode=0):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=2,3
  fig = py.figure(figsize=(ncols*7,nrows*4))
  ax11=py.subplot(nrows,ncols,1)
  ax12=py.subplot(nrows,ncols,2)
  ax13=py.subplot(nrows,ncols,3)
  ax21=py.subplot(nrows,ncols,4)
  ax22=py.subplot(nrows,ncols,5)
  ax23=py.subplot(nrows,ncols,6)


  load_config('%s/input.py'%wdir)
  if Q2==None: Q2 = conf['Q20']
  istep=core.get_istep()

  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  TAR = ['d','h','t']
  stf = 'F2'

  for tar in TAR:
      filename ='%s/data/off-stf-%s-Q2=%3.5f.dat'%(wdir,tar,Q2)
      #--load data if it exists
      try:
          data=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_off_stf(wdir,Q2,tar)
          data=load(filename)
      filename ='%s/data/stf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)
      #--load data if it exists
      try:
          STF=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_stf(wdir,Q2,tar,stf)
          STF=load(filename)

      X    = data['X']
      data = data['XF']
      STF  = STF['XF']
      mean = np.mean(data,axis=0)
      std  = np.std (data,axis=0)

      if tar == 'd': ax,color=ax11,'firebrick'
      if tar == 'h': ax,color=ax12,'darkgreen'
      if tar == 't': ax,color=ax13,'darkblue'
      

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              ax.plot(X,data[i],color=color,alpha=0.3)
    
      #--plot average and standard deviation
      if mode==1:
          ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)

      #--plot ratios to onshell structure functions 
      if tar == 'd': ax,color=ax21,'firebrick'
      if tar == 'h': ax,color=ax22,'darkgreen'
      if tar == 't': ax,color=ax23,'darkblue'

      mean = mean/np.mean(STF,axis=0)
      std  = std /np.mean(STF,axis=0)

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              ax.plot(X,data[i]/STF[i],color=color,alpha=0.3)
    
      #--plot average and standard deviation
      if mode==1:
          ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)


  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelbottom=False)
  ax12.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelbottom=False,labelleft=False)
  ax13.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelbottom=False,labelleft=False)
  ax21.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30)
  ax22.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelleft=False)
  ax23.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=30,labelleft=False)

  ax11.text(0.05, 0.08, r'\boldmath$x F_2^{d({\rm off})}$'                        ,size=30, transform=ax11.transAxes)
  ax12.text(0.05, 0.08, r'\boldmath$x F_2^{^3{\rm He}({\rm off})}$'               ,size=30, transform=ax12.transAxes)
  ax13.text(0.05, 0.08, r'\boldmath$x F_2^{^3{\rm H} ({\rm off})}$'               ,size=30, transform=ax13.transAxes)
  ax21.text(0.05, 0.08, r'\boldmath$F_2^{d({\rm off})}/F_2^d$'                    ,size=30, transform=ax21.transAxes)
  ax22.text(0.05, 0.08, r'\boldmath$F_2^{^3{\rm He}({\rm off})}/F_2^{^3{\rm He}}$',size=30, transform=ax22.transAxes)
  ax23.text(0.05, 0.08, r'\boldmath$F_2^{^3{\rm H} ({\rm off})}/F_2^{^3{\rm H}}$' ,size=30, transform=ax23.transAxes)

  #if Q2 == 1.27**2: ax11.text(0.05,0.05,r'$Q^2 = m_c^2$',              transform=ax11.transAxes,size=30)
  #else:             ax11.text(0.05,0.05,r'$Q^2 = %s ~ \rm{GeV^2}$'%Q2, transform=ax11.transAxes,size=30)

  for ax in [ax11,ax12,ax13,ax21,ax22,ax23]:
      ax.set_xlim(0,0.9)
      ax.axhline(0,0,1,ls=':',color='black',alpha=0.5)
      minorLocator = MultipleLocator(0.02)
      majorLocator = MultipleLocator(0.2)
      ax.xaxis.set_minor_locator(minorLocator)
      ax.xaxis.set_major_locator(majorLocator)
      ax.xaxis.set_tick_params(which='major',length=6)
      ax.xaxis.set_tick_params(which='minor',length=3)
      ax.yaxis.set_tick_params(which='major',length=6)
      ax.yaxis.set_tick_params(which='minor',length=3)
      ax.set_xticks([0.2,0.4,0.6,0.8])

  for ax in [ax11,ax12,ax13]:
      ax.set_ylim(-0.005,0.005)
      minorLocator = MultipleLocator(0.0004)
      majorLocator = MultipleLocator(0.002)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  for ax in [ax21,ax22,ax23]:
      ax.set_ylim(-0.19,0.19)
      minorLocator = MultipleLocator(0.02)
      majorLocator = MultipleLocator(0.10)
      ax.yaxis.set_minor_locator(minorLocator)
      ax.yaxis.set_major_locator(majorLocator)

  for ax in [ax21,ax22,ax23]:
      ax.set_xlabel(r'\boldmath$x$'          ,size=30)
      ax.xaxis.set_label_coords(0.97,0)

  handles,labels = [],[]
  #handles.append(hand['HT'])
  #labels.append(r'\boldmath$R(^3{\rm He})/R(^3{\rm H})$')
  #labels.append(r'\boldmath$\mathcal{R}$')
  #ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=25, handletextpad = 0.5, handlelength = 1.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0,wspace=0)

  filename = '%s/gallery/off-stfs-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'
  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()




#--F2 moments as function of Q2
def gen_F2_mom(wdir,tar='p',W2cut=3.5,xmin=10e-9,xmax=0.99):
 
    stf = 'F2' 
    load_config('%s/input.py'%wdir)
    Q20 = conf['Q20']
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'pdf' not in conf['steps'][istep]['active distributions']:
        if 'pdf' not in conf['steps'][istep]['passive distributions']:
                print('pdf is not an active or passive distribution')
                return 

    M2 = 0.9389**2

    #--setup kinematics for structure functions to be calculated at
    Xgrid = np.geomspace(1e-5,1e-1,20)
    Xgrid = np.append(Xgrid,np.linspace(0.1,0.99,20))
    Q2grid = np.linspace(Q20*0.99,8*1.01,10)
    conf['idis grid'] = {}
    conf['idis grid']['X']  = Xgrid 
    conf['idis grid']['Q2'] = Q2grid 
    conf['datasets']['idis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    idis  = resman.idis_thy
    
    if tar in ['p','n']:
        idis.data[tar] = {}
        idis.data[tar][stf] = np.zeros(idis.X.size)
    else:
        idis.data[tar] = {}
        idis.data['p'][stf] = np.zeros(idis.X.size)
        idis.data['n'][stf] = np.zeros(idis.X.size)
        idis.data[tar][stf] = np.zeros(idis.X.size)
        if stf=='FL':
            idis.data['p']['F2'] = np.zeros(idis.X.size)
            idis.data['n']['F2'] = np.zeros(idis.X.size)
            idis.data[tar]['F2'] = np.zeros(idis.X.size)

    if 'off pdf' in conf:
        idis.data['p']['F2off d'] = np.zeros(idis.X.size)
        idis.data['n']['F2off d'] = np.zeros(idis.X.size)
        idis.data['p']['F2off h'] = np.zeros(idis.X.size)
        idis.data['n']['F2off h'] = np.zeros(idis.X.size)
        idis.data['p']['F2off t'] = np.zeros(idis.X.size)
        idis.data['n']['F2off t'] = np.zeros(idis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    ## setup kinematics
    Q2 = np.linspace(Q20,8,5) 
    if xmax==None: print('\ngenerating first moment for F2 %s from %s from %3.9f' % (tar, wdir, xmin))
    else:          print('\ngenerating first moment for F2 %s from %s from %3.9f to %3.2f' % (tar, wdir, xmin, xmax))


    ## compute moments for all replicas
    MOM = []
    cnt = 0

    for par in replicas:
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        idis._update()

        mom = []
        for q2 in Q2:
            if xmax==None: _xmax = q2/(W2cut - M2 + q2)
            else:          _xmax = xmax
            xs = np.geomspace(xmin,0.1,100)
            xs = np.append(xs, np.linspace(0.1, _xmax, 100))
            func = lambda x: idis.get_stf(x,q2,stf='F2',tar=tar)

            function_values = [func(_) for _ in xs]
            moment_temp = cumtrapz(function_values, xs, initial = 0.0)
            moment_temp = np.array(moment_temp)
            moment_max = moment_temp[-1]
            moment = moment_max - moment_temp
            mom.append(moment[0])

        MOM.append(mom)

    print()

    MOM = np.array(MOM)

    checkdir('%s/data'%wdir)
    if xmax==None: filename ='%s/data/stf-F2-%s-mom-xmin-%3.9f.dat'%(wdir,tar,xmin)
    else:          filename ='%s/data/stf-F2-%s-mom-xmin-%3.9f-xmax-%3.5f.dat'%(wdir,tar,xmin,xmax)

    save({'MOM':MOM,'xmin':xmin,'xmax':xmax,'Q2':Q2},filename)
    print ('Saving data to %s'%filename)

def plot_F2_mom(wdir,Q2=None,mode=1,W2cut=3.5,xmin=10e-9,xmax=0.99):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  TAR = ['p','n']
  nrows,ncols=1,1
  fig = py.figure(figsize=(ncols*7,nrows*5))
  ax11=py.subplot(nrows,ncols,1)

  load_config('%s/input.py'%wdir)
  istep=core.get_istep()

  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  hand = {}

  data = {}

  for tar in TAR:
      if xmax==None: filename ='%s/data/stf-F2-%s-mom-xmin-%3.9f.dat'%(wdir,tar,xmin)
      else:          filename ='%s/data/stf-F2-%s-mom-xmin-%3.9f-xmax-%3.5f.dat'%(wdir,tar,xmin,xmax)
      #--load data if it exists
      try:
          data[tar]=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_F2_mom(wdir,tar,)
          data[tar]=load(filename)
      Q2    = data[tar]['Q2']
      value = data[tar]['MOM']
      mean  = np.mean(value,axis=0)
      std   = np.std (value,axis=0)

      if tar=='p':     color='red'
      elif tar=='n':   color='purple'
      else: continue

      ax = ax11

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              hand[tar] ,= ax.plot(Q2,value[i],color=color,alpha=0.1)
    
      #--plot average and standard deviation
      if mode==1:
          #ax.plot(Q2,mean,color=color)
          hand[tar] = ax.fill_between(Q2,mean-std,mean+std,color=color,alpha=0.9)

  #--plot p - n
  value = data['p']['MOM'] - data['n']['MOM']
  mean  = np.mean(value,axis=0)
  std   = np.std (value,axis=0)
  #--plot each replica
  if mode==0:
      for i in range(len(data)):
          hand['p-n'] ,= ax.plot(Q2,data[i],color='blue',alpha=0.1)
  
  #--plot average and standard deviation
  if mode==1:
      #ax.plot(Q2,mean,color=color)
      hand['p-n'] = ax.fill_between(Q2,mean-std,mean+std,color='blue',alpha=0.9)

  for ax in [ax11]:
        ax.set_xlim(0.5,7.8)
          
        ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)


  #ax11.text(0.50,0.70,r'\boldmath$\int_{x_{\rm min}}^{x_{\rm max}}F_2 dx$',transform=ax11.transAxes,size=30)
  if xmax==None: ax11.text(0.20,0.80,r'\boldmath$\int_{%3.9f}^{x_{\rm max}} F_2^N~dx$'%(xmin),transform=ax11.transAxes,size=30)
  else:          ax11.text(0.20,0.80,r'\boldmath$\int_{%3.9f}^{%3.2f} F_2^N~dx$'%(xmin,xmax) ,transform=ax11.transAxes,size=30)

  handles,labels = [],[]

  handles.append(hand['p'])
  handles.append(hand['n'])
  handles.append(hand['p-n'])

  labels.append(r'\boldmath$p$')
  labels.append(r'\boldmath$n$')
  labels.append(r'\boldmath$p-n$')

  ax11.legend(handles,labels,loc='upper right',fontsize=20, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)

  ax11.set_ylim(0,0.3)      
  minorLocator = MultipleLocator(0.01)
  majorLocator = MultipleLocator(0.05)
  ax11.yaxis.set_minor_locator(minorLocator)
  ax11.yaxis.set_major_locator(majorLocator)

  for ax in [ax11]:
      minorLocator = MultipleLocator(0.2)
      majorLocator = MultipleLocator(1.0)
      ax.xaxis.set_minor_locator(minorLocator)
      ax.xaxis.set_major_locator(majorLocator)
      ax.xaxis.set_tick_params(which='major',length=6)
      ax.xaxis.set_tick_params(which='minor',length=3)
      ax.yaxis.set_tick_params(which='major',length=6)
      ax.yaxis.set_tick_params(which='minor',length=3)
      #ax.set_xticks([0.2,0.4,0.6,0.8])
      ax.set_xlabel(r'\boldmath$Q^2~[{\rm GeV}^2]$' ,size=25)   
      #ax.xaxis.set_label_coords(0.97,0)


  #if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
  #else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)

  py.tight_layout()

  if xmax==None: filename ='%s/gallery/stf-F2-mom-xmin-%3.9f.dat'%(wdir,xmin)
  else:          filename ='%s/gallery/stf-F2-mom-xmin-%3.9f-xmax-%3.5f.dat'%(wdir,xmin,xmax)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()





if __name__=="__main__":

    
    ap = argparse.ArgumentParser()

    ap.add_argument('task'                ,type=int                       ,help='0 to generate STFs')
    ap.add_argument('-d'   ,'--directory' ,type=str   ,default='unamed'   ,help='directory name to store results')
    ap.add_argument('-Q2'  ,'--Q2'        ,type=float ,default='unamed'   ,help='Q2 value')
    ap.add_argument('-t'   ,'--tar'       ,type=str   ,default='unamed'   ,help='target')
    ap.add_argument('-s'   ,'--stf'       ,type=str   ,default='unamed'   ,help='structure function')
    args = ap.parse_args()

    if args.task==0:
        gen_stf(args.directory,Q2=args.Q2,tar=args.tar,stf=args.stf)

    if args.task==1:
        gen_marathon_stf(args.directory,tar=args.tar)

    if args.task==2:
        gen_off_stf(args.directory,Q2=args.Q2,nucleus=args.tar)










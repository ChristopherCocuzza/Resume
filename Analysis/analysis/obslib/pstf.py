import sys,os
import numpy as np
import time
import argparse

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

#--polarized

def gen_pstf(wdir,Q2=None,tar='p',stf='g1'):
   
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = True
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = True

    if Q2==None: Q2 = conf['Q20']
    print('\ngenerating PSTF from %s for %s %s at Q2=%3.5f'%(wdir,stf,tar,Q2))

    conf['pidis grid'] = 'prediction'
    conf['datasets']['idis']  = {_:{} for _ in ['xlsx','norm']}
    conf['datasets']['pidis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    conf['idis'] = resman.idis_thy
    resman.setup_pidis()
    pidis = resman.pidis_thy
    if tar in ['p','n']:
        pidis.data[tar] = {}
        pidis.data[tar][stf] = np.zeros(pidis.X.size)
    else:
        pidis.data[tar] = {}
        pidis.data['p']['g1'] = np.zeros(pidis.X.size)
        pidis.data['n']['g1'] = np.zeros(pidis.X.size)
        pidis.data[tar]['g1'] = np.zeros(pidis.X.size)
        pidis.data['p']['g2'] = np.zeros(pidis.X.size)
        pidis.data['n']['g2'] = np.zeros(pidis.X.size)
        pidis.data[tar]['g2'] = np.zeros(pidis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    ppdf=conf['ppdf']
    #--setup kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.99,100))

    #--compute X*STF for all replicas        
    XF=[]
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        ppdf.evolve(Q2)
        pidis._update()

        xf = X*pidis.get_stf(X,Q2,stf=stf,tar=tar)
        XF.append(xf)

    XF = np.array(XF)
    print()
    checkdir('%s/data'%wdir)
    filename ='%s/data/pstf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_pstf(wdir,Q2=None,mode=0,TAR=['p','n','d','h'],STF=['g1','g2']):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=2,1
  N = nrows*ncols
  fig = py.figure(figsize=(ncols*7,nrows*4))
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


  load_config('%s/input.py'%wdir)
  istep=core.get_istep()

  if Q2==None: Q2 = conf['Q20']
      
  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  hand = {}
  for tar in TAR:
      for stf in STF:
          filename ='%s/data/pstf-%s-%s-Q2=%3.5f.dat'%(wdir,tar,stf,Q2)
          #--load data if it exists
          try:
              data=load(filename)
          #--generate data and then load it if it does not exist
          except:
              gen_pstf(wdir,Q2,tar,stf)
              data=load(filename)

          X    = data['X']
          data = data['XF'] 
          mean = np.mean(data,axis=0)
          std  = np.std (data,axis=0)

          if tar=='p': color='red'
          if tar=='n': color='green'
          if tar=='d': color='blue'
          if tar=='h': color='magenta'

          label = None
          if stf =='g1': ax,axL = axs[1],axLs[1]
          if stf =='g2': ax,axL = axs[2],axLs[2]

          #--plot each replica
          if mode==0:
              for i in range(len(data)):
                  hand[tar] ,= ax .plot(X,data[i],color=color,alpha=0.1)
                  hand[tar] ,= axL.plot(X,data[i],color=color,alpha=0.1)
    
          #--plot average and standard deviation
          if mode==1:
              hand[tar] = ax .fill_between(X,(mean-std),(mean+std),color=color,alpha=0.8)
              hand[tar] = axL.fill_between(X,(mean-std),(mean+std),color=color,alpha=0.8)


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
        axs[i+1] .tick_params(axis='both', which='minor', length = 3)
        axs[i+1] .tick_params(axis='both', which='major', length = 6)
        axLs[i+1].tick_params(axis='both', which='minor', length = 3)
        axLs[i+1].tick_params(axis='both', which='major', length = 6)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.2)
        axLs[i+1].xaxis.set_minor_locator(minorLocator)
        axLs[i+1].xaxis.set_major_locator(majorLocator)
        axLs[i+1].set_xticks([0.3,0.5,0.7])
        axLs[i+1].set_xticklabels([r'$0.3$',r'$0.5$',r'$0.7$'])
        axLs[i+1].axhline(0,0,1,ls='--',color='black',alpha=0.5)
        axLs[i+1].axvline(0.1,0,1,ls=':' ,color='black',alpha=0.5)


  minorLocator = MultipleLocator(0.01)
  majorLocator = MultipleLocator(0.02)
  axs[1].yaxis.set_minor_locator(minorLocator)
  axs[1].yaxis.set_major_locator(majorLocator)
  minorLocator = MultipleLocator(0.005)
  majorLocator = MultipleLocator(0.01)
  axs[2].yaxis.set_minor_locator(minorLocator)
  axs[2].yaxis.set_major_locator(majorLocator)

  axs[1] .tick_params(labelbottom=False)
  axLs[1].tick_params(labelbottom=False)
  axLs[2].set_xlabel(r'\boldmath$x$' ,size=30)
  axLs[2].xaxis.set_label_coords(0.95,0.00)

  axs[1].text(0.10,0.40,r'\boldmath$xg_1$',transform=axs[1].transAxes,size=40)
  axs[2].text(0.10,0.25,r'\boldmath$xg_2$',transform=axs[2].transAxes,size=40)

  axs[1].set_ylim(-0.025,0.085)
  axs[2].set_ylim(-0.050,0.019)

  axs[1].set_yticks([-0.02,0,0.02,0.04,0.06,0.08])
  axs[2].set_yticks([-0.04,-0.03,-0.02,-0.01,0,0.01])

  if Q2 == 1.27**2: axs[2].text(0.05,0.05,r'$Q^2 = m_c^2$',             transform=axs[2].transAxes,size=30)
  else:             axs[2].text(0.05,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=axs[2].transAxes,size=25)

  handles, labels = [],[]
  if 'p' in hand: handles.append(hand['p'])
  if 'n' in hand: handles.append(hand['n'])
  if 'd' in hand: handles.append(hand['d'])
  if 'h' in hand: handles.append(hand['h'])
  if 'p' in hand: labels.append(r'\boldmath$p$')
  if 'n' in hand: labels.append(r'\boldmath$n$')
  if 'd' in hand: labels.append(r'\boldmath$D$')
  if 'h' in hand: labels.append(r'\boldmath$^3 {\rm He}$')
  axs[1].legend(handles,labels,loc='upper left',fontsize=25, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/pstfs-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

#--g2-g2WW residuals
def gen_g2res(wdir,Q2=None,tar='p'):
   
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'g2res' not in conf['steps'][istep]['active distributions']:
        if 'g2res' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = True
    
    if Q2==None: Q2 = conf['Q20']
    print('\ngenerating g2res from %s for %s at Q2=%3.5f'%(wdir,tar,Q2))

    #conf['pidis grid'] = 'prediction'
    #conf['datasets']['idis']  = {_:{} for _ in ['xlsx','norm']}
    #conf['datasets']['pidis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    #resman.setup_idis()
    #conf['idis'] = resman.idis_thy
    #resman.setup_pidis()
    #pidis = resman.pidis_thy
    #if tar in ['p','n']:
    #    pidis.data[tar] = {}
    #    pidis.data[tar]['g2'] = np.zeros(pidis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    #--setup kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.99,100))

    g2res = conf['g2res']
    #--compute X*g2res for all replicas        
    XF=[]
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)

        xf = X*g2res.get_g2res(X,Q2,tar,conf['tmc'])
        XF.append(xf)

    XF = np.array(XF)
    print()
    checkdir('%s/data'%wdir)
    filename ='%s/data/g2res-%s-Q2=%3.5f.dat'%(wdir,tar,Q2)

    save({'X':X,'Q2':Q2,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_g2res(wdir,Q2=None,mode=0):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=1,2
  N = nrows*ncols
  fig = py.figure(figsize=(ncols*7,nrows*4))
  ax11 = py.subplot(nrows,ncols,1)
  ax12 = py.subplot(nrows,ncols,2)

  load_config('%s/input.py'%wdir)
  istep=core.get_istep()

  if 'g2res' not in conf['steps'][istep]['active distributions']:
      if 'g2res' not in conf['steps'][istep]['passive distributions']:
          print('g2res not an active or passive distribution')
          return

  if Q2==None: Q2 = conf['Q20']
      
  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  TAR = ['p','n']
  hand = {}
  for tar in TAR:
      filename ='%s/data/g2res-%s-Q2=%3.5f.dat'%(wdir,tar,Q2)
      #--load data if it exists
      try:
          data=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_g2res(wdir,Q2,tar)
          data=load(filename)

      X    = data['X']
      data = data['XF'] 
      mean = np.mean(data,axis=0)
      std  = np.std (data,axis=0)

      if tar=='p': ax,color=ax11,'red'
      if tar=='n': ax,color=ax12,'green'

      label = None

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              hand[tar] ,= ax .plot(X,data[i],color=color,alpha=0.1)
    
      #--plot average and standard deviation
      if mode==1:
          hand[tar] = ax .fill_between(X,(mean-std),(mean+std),color=color,alpha=0.8)


  for ax in [ax11,ax12]:
        ax.set_xlim(0,0.9)

        ax.tick_params(axis='both', which='both', top=True, direction='in',labelsize=20)
        ax.set_xticks([0.2,0.4,0.6,0.8])
        ax.axhline(0,0,1,ls='--',color='black',alpha=0.5)

        ax.tick_params(axis='both', which='both', top=True, right=True, labelright=False, direction='in',labelsize=20)
        ax.tick_params(axis='both', which='minor', length = 3)
        ax.tick_params(axis='both', which='major', length = 6)
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xlabel(r'\boldmath$x$' ,size=30)
        ax.xaxis.set_label_coords(0.95,0.00)
        

  Xmarks = np.linspace(0.001,0.9,5)
  for x in Xmarks:
      ax11.axvline(x,0.45,0.55,ls='-',alpha=0.5,color='black')
      ax12.axvline(x,0.45,0.55,ls='-',alpha=0.5,color='black')

  ax11.set_ylim(-0.025,0.025)
  ax12.set_ylim(-0.060,0.060)

  minorLocator = MultipleLocator(0.01)
  majorLocator = MultipleLocator(0.02)
  ax11.yaxis.set_minor_locator(minorLocator)
  ax11.yaxis.set_major_locator(majorLocator)
  minorLocator = MultipleLocator(0.01)
  majorLocator = MultipleLocator(0.05)
  ax12.yaxis.set_minor_locator(minorLocator)
  ax12.yaxis.set_major_locator(majorLocator)

  #ax.tick_params(labelbottom=False)
  #ax.tick_params(labelbottom=False)

  #ax11.text(0.10,0.10,r'\boldmath$xg_{2,{\rm res}}^p$',transform=ax11.transAxes,size=40)
  #ax12.text(0.10,0.10,r'\boldmath$xg_{2,{\rm res}}^n$',transform=ax12.transAxes,size=40)
  ax11.text(0.05,0.05,r'\boldmath$x(g_{2}^p-g_{2,{\rm WW}}^p)$',transform=ax11.transAxes,size=40)
  ax12.text(0.05,0.05,r'\boldmath$x(g_{2}^n-g_{2,{\rm WW}}^n)$',transform=ax12.transAxes,size=40)


  #ax.set_yticks([-0.02,0,0.02,0.04,0.06,0.08])
  #ax.set_yticks([-0.04,-0.03,-0.02,-0.01,0,0.01])

  #if Q2 == 1.27**2: ax11.text(0.05,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
  #else:             ax11.text(0.05,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)

  handles, labels = [],[]
  #if 'p' in hand: handles.append(hand['p'])
  #if 'n' in hand: handles.append(hand['n'])
  #if 'p' in hand: labels.append(r'\boldmath$p$')
  #if 'n' in hand: labels.append(r'\boldmath$n$')
  #ax11.legend(handles,labels,loc='upper left',fontsize=25, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/g2res-Q2=%3.5f'%(wdir,Q2)
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()


#--polarized moments
def gen_d2(wdir,tar='p'):

    print('\ngenerating d2 from %s for %s'%(wdir,tar))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = true
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = true
    
    conf['pidis grid'] = 'prediction'
    conf['datasets']['idis']  = {_:{} for _ in ['xlsx','norm']}
    conf['datasets']['pidis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    conf['idis'] = resman.idis_thy
    resman.setup_pidis()
    pidis = resman.pidis_thy
    pidis.data[tar]['g1'] = np.zeros(pidis.X.size)
    pidis.data[tar]['g2'] = np.zeros(pidis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    ppdf=conf['ppdf']
    #--setup kinematics
    #--Gaussian quadrature
    npts = 99
    z,w = np.polynomial.legendre.leggauss(npts) 
    jac = 0.5
    x   = 0.5*(z+1)
    q2 = np.linspace(1.27**2,10,20)

    X,Q2 = np.meshgrid(x,q2)

    #--compute d2 for all replicas
    D2=[]
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        pidis._update()

        g1   = X**2*pidis.get_stf(X,Q2,stf='g1',tar=tar)
        g2   = X**2*pidis.get_stf(X,Q2,stf='g2',tar=tar)
        func = 2*g1 + 3*g2

        d2 = [np.sum(w*jac*func[i]) for i in range(len(func))]
        D2.append(d2)

    print()
    checkdir('%s/data'%wdir)
    filename='%s/data/pstf-d2-%s.dat'%(wdir,tar)

    save({'Q2':Q2,'D2':D2},filename)
    print('Saving data to %s'%filename)
        
def plot_d2(wdir,mode=0):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  TAR = ['p','n']

  nrows,ncols=1,1
  fig = py.figure(figsize=(ncols*7,nrows*4))
  ax11 = py.subplot(nrows,ncols,1)

  load_config('%s/input.py'%wdir)
  istep=core.get_istep()


  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  hand = {}
  for tar in TAR:
      filename='%s/data/pstf-d2-%s.dat'%(wdir,tar)
      #--load data if it exists
      try:
         data=load(filename)
      #--generate data and then load it if it does not exist
      except:
          gen_d2(wdir,tar)
          data=load(filename)
     
      Q2  = np.array([data['Q2'][i][0] for i in range(len(data['Q2']))])
      data = data['D2']

      mean = np.mean(data,axis=0)
      std  = np.std (data,axis=0)

      if tar=='p': color='firebrick'
      if tar=='n': color='darkgreen'

      #--plot each replica
      if mode==0:
          for i in range(len(data)):
              hand[tar] ,= ax11.plot(Q2,data[i],color=color,alpha=0.2)
    
      #--plot average and standard deviation
      if mode==1:
          hand[tar] = ax11.fill_between(Q2,(mean-std),(mean+std),color=color,alpha=0.8)


  ax11.set_xlim(0.8,6)
  ax11.set_ylim(-0.006,0.020)
  ax11.set_yticks([-0.005,0.000,0.005,0.010,0.015,0.020])

  ax11.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)
  ax11.set_xticks([1,2,3,4,5])
  ax11.axhline(0,0,1,ls='-',color='black',alpha=0.2)
  ax11.axvline(1.27**2,0,1,ls='--',color='black',alpha=0.5)

  ax11.set_xlabel(r'\boldmath$Q^2 \rm{(GeV}^2 \rm{)}$' ,size=30)
  #ax11.xaxis.set_label_coords(0.95,0.00)

  ax11.text(0.40,0.70,r'\boldmath$d_2$',transform=ax11.transAxes,size=40)
  
  ax11.text(0.10,0.50,r'$Q^2~{\rm cut}$',transform=ax11.transAxes,size=20,rotation=90)

  #ax11.set_ylim(-0.005,0.010)

  #ax11.set_yticks([-0.02,0,0.02,0.04,0.06,0.08])

  handles, labels = [],[]
  handles.append(hand['p'])
  handles.append(hand['n'])
  labels.append(r'\boldmath$p$')
  labels.append(r'\boldmath$n$')
  ax11.legend(handles,labels,loc='upper right',fontsize=25, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 1, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename = '%s/gallery/d2'%wdir
  if mode==1: filename += '-bands'

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

#--g2 sum rule
def gen_BC_SR(wdir,tar='p'):

    print('\ngenerating BC SR from %s for %s'%(wdir,tar))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep   

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return 

    passive=False
    if 'ppdf'  in conf['steps'][istep]['passive distributions']: passive = True
    if 'g2res' in conf['steps'][istep]['passive distributions']: passive = True
    conf['pidis grid'] = 'prediction'
    
    conf['datasets']['idis']  = {_:{} for _ in ['xlsx','norm']}
    conf['datasets']['pidis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_idis()
    conf['idis'] = resman.idis_thy
    resman.setup_pidis()
    pidis = resman.pidis_thy
    pidis.data[tar]['g1'] = np.zeros(pidis.X.size)
    pidis.data[tar]['g2'] = np.zeros(pidis.X.size)

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    ppdf=conf['ppdf']
    #--setup kinematics
    #--Gaussian quadrature
    npts = 99
    z,w = np.polynomial.legendre.leggauss(npts) 
    jac = 0.5
    x   = 0.5*(z+1)
    q2 = np.linspace(1.27**2,10,20)

    X,Q2 = np.meshgrid(x,q2)

    #--compute SR for all replicas
    SR=[]
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        pidis._update()

        sr   = pidis.get_stf(X,Q2,stf='g2',tar=tar)
        func = sr

        sr = [np.sum(w*jac*func[i]) for i in range(len(func))]
        SR.append(sr)

    print()
    checkdir('%s/data'%wdir)
    filename='%s/data/pstf-BC-SR-%s.dat'%(wdir,tar)

    save({'Q2':Q2,'SR':D2},filename)
    print('Saving data to %s'%filename)

#--polarized semi-inclusive
#--plot as function of x with fixed Q2 and z
def gen_sipstf_funcx(wdir,Q2,z=0.4,TAR=['p','n'],HAD=['pi+','K+','pi-','K-'],STF=['g1']):
   
    _STF = {}
    for tar in TAR:
        _STF[tar] = {}
        for had in HAD:
            _STF[tar][had] = []
            for stf in STF:
                _STF[tar][had].append(stf)

    print('\ngenerating semi-inclusive PSTF from %s at Q2=%s, z=%s'%(wdir,Q2,z))
    load_config('%s/input.py'%wdir)
    istep=core.get_istep()
    replicas=core.get_replicas(wdir)
    core.mod_conf(istep,replicas[0]) #--set conf as specified in istep

    if 'ppdf' not in conf['steps'][istep]['active distributions']:
        if 'ppdf' not in conf['steps'][istep]['passive distributions']:
                print('ppdf is not an active or passive distribution')
                return
    if 'pi+' or 'pi-' in HAD: 
        if 'ffpion' not in conf['steps'][istep]['active distributions']:
            if 'ffpion' not in conf['steps'][istep]['passive distributions']:
                    print('ffpion is not an active or passive distribution')
                    return 
    if 'K+' or 'K-' in HAD: 
        if 'ffkaon' not in conf['steps'][istep]['active distributions']:
            if 'ffkaon' not in conf['steps'][istep]['passive distributions']:
                    print('ffkaon is not an active or passive distribution')
                    return 

    passive=False
    if 'ppdf'   in conf['steps'][istep]['passive distributions']: passive = True
    if 'ffkaon' in conf['steps'][istep]['passive distributions']: passive = True
    if 'ffpion' in conf['steps'][istep]['passive distributions']: passive = True
    
    conf['datasets']['sidis']  = {_:{} for _ in ['xlsx','norm']}
    conf['datasets']['psidis'] = {_:{} for _ in ['xlsx','norm']}
    resman=RESMAN(nworkers=1,parallel=False,datasets=False)
    parman=resman.parman
    resman.setup_sidis()
    conf['sidis'] = resman.sidis_thy
    resman.setup_psidis()
    conf['psidis'] = resman.psidis_thy
    psidis = resman.psidis_thy

    parman.order=replicas[0]['order'][istep]

    jar=load('%s/data/jar-%d.dat'%(wdir,istep))
    replicas=jar['replicas']

    ppdf=conf['ppdf']
    if 'ffpion' in conf: ffpion = conf['ffpion']
    if 'ffkaon' in conf: ffkaon = conf['ffkaon']
    #--setup kinematics
    X=10**np.linspace(-4,-1,100)
    X=np.append(X,np.linspace(0.1,0.99,100))

    zlim = np.array([None,None])
    #--compute X*STF for all replicas        
    XF={}
    cnt=0
    for par in replicas:
        if passive: core.mod_conf(istep,core.get_replicas(wdir)[cnt])   
        cnt+=1
        lprint('%d/%d'%(cnt,len(replicas)))

        parman.set_new_params(par,initial=True)
        ppdf.evolve(Q2)
        if 'ffpion' in conf: ffpion.evolve(Q2)
        if 'ffkaon' in conf: ffkaon.evolve(Q2)

        for tar in TAR:
            if tar not in XF: XF[tar] = {}
            for had in HAD:
                if had not in XF[tar]: XF[tar][had] = {}
                for stf in _STF[tar][had]:
                    if stf not in XF[tar][had]:  XF[tar][had][stf]=[]
                    xf = [x*psidis.get_g1(x,z,zlim,Q2,target=tar,hadron=had) for x in X]
                    XF[tar][had][stf].append(xf)

    print()
    checkdir('%s/data'%wdir)
    if Q2==1.27**2: filename='%s/data/sipstf_funcx-%d-z=%s.dat'%(wdir,istep,z)
    else:filename='%s/data/sipstf_funcx-%d-z=%s-Q2=%d.dat'%(wdir,istep,z,int(Q2))

    save({'X':X,'Q2':Q2,'z':z,'XF':XF},filename)
    print ('Saving data to %s'%filename)

def plot_sipstf_funcx(wdir,Q2,kc,z=0.4,mode=1):
  #--mode 0: plot each replica
  #--mode 1: plot average and standard deviation of replicas 

  nrows,ncols=2,1
  N = nrows*ncols
  fig = py.figure(figsize=(ncols*7,nrows*4))
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

  filename = '%s/gallery/sipstfs-z=%s'%(wdir,z)
  if mode==1: filename += '-bands'

  load_config('%s/input.py'%wdir)
  istep=core.get_istep()

  #--load data if it exists
  try:
      if Q2==1.27**2: data=load('%s/data/sipstf_funx-%d-z=%s.dat'%(wdir,istep,z))
      else: data=load('%s/data/sipstf_funcx-%d-z=%s-Q2=%d.dat'%(wdir,istep,z,int(Q2)))
  #--generate data and then load it if it does not exist
  except:
      gen_pstf(wdir,Q2)
      if Q2==1.27**2: data=load('%s/data/sipstf_funcx-%d-z=%s.dat'%(wdir,istep,z))
      else: data=load('%s/data/sipstf_funx-%d-z=%s-Q2=%d.dat'%(wdir,istep,z,int(Q2)))
      
  cluster,colors,nc,cluster_order = classifier.get_clusters(wdir,istep,kc) 
  best_cluster=cluster_order[0]

  X  = data['X']
  idx1 = np.nonzero(X <= 0.1)
  idx2 = np.nonzero(X >= 0.1)

  hand = {}
  for tar in data['XF']:
      for had in data['XF'][tar]:
          for stf in data['XF'][tar][had]:
              mean = np.mean(data['XF'][tar][had][stf],axis=0)
              std = np.std(data['XF'][tar][had][stf],axis=0)

              if tar=='p': color='red'
              if tar=='n': color='green'

              if 'pi' in had: ax,axL = axs[1],axLs[1]
              if 'K'  in had: ax,axL = axs[2],axLs[2]

              hatch = None
              if '-' in had: hatch = '//'

              #--plot each replica
              if mode==0:
                  for i in range(len(data['XF'][tar][had][stf])):
                      ax .plot(X[idx1],data['XF'][tar][had][stf][i][idx1],color=color,alpha=0.1)
                      axL.plot(X[idx2],data['XF'][tar][had][stf][i][idx2],color=color,alpha=0.1)
    
              #--plot average and standard deviation
              if mode==1:
                  hand[tar+','+had] = ax .fill_between(X[idx1],(mean-std)[idx1],(mean+std)[idx1],color=color,alpha=0.5,hatch=hatch)
                  hand[tar+','+had] = axL.fill_between(X[idx2],(mean-std)[idx2],(mean+std)[idx2],color=color,alpha=0.5,hatch=hatch)


  for i in range(N):
        axs[i+1].set_xlim(8e-3,0.1)
        axs[i+1].semilogx()

        axs[i+1].tick_params(axis='both', which='both', top=True, direction='in',labelsize=20)
        axs[i+1].set_xticks([0.01,0.1])
        axs[i+1].set_xticklabels([r'$0.01$',r'$0.1$'])
        axs[i+1].axhline(0,0,1,ls='--',color='black',alpha=0.5)
        axs[i+1].axvline(0.1,0,1,ls=':' ,color='black',alpha=0.5)

        axLs[i+1].set_xlim(0.1,0.9)

        axLs[i+1].tick_params(axis='both', which='both', top=True, right=True, left=False, labelright=False, direction='in',labelsize=20)
        axLs[i+1].set_xticks([0.3,0.5,0.7])
        axLs[i+1].set_xticklabels([r'$0.3$',r'$0.5$',r'$0.7$'])
        axLs[i+1].axhline(0,0,1,ls='--',color='black',alpha=0.5)
        axLs[i+1].axvline(0.1,0,1,ls=':' ,color='black',alpha=0.5)

  axs[1] .tick_params(labelbottom=False)
  axLs[1].tick_params(labelbottom=False)
  axLs[2].set_xlabel(r'\boldmath$x$' ,size=30)
  axLs[2].xaxis.set_label_coords(0.95,0.00)

  axs[1].text(0.10,0.40,r'\boldmath$xg_1$',transform=axs[1].transAxes,size=40)
  #axs[2].text(0.10,0.25,r'\boldmath$xg_2$',transform=axs[2].transAxes,size=40)

  axs[1].set_ylim(-0.035,0.110)
  axs[2].set_ylim(-0.020,0.050)

  #axs[1].set_yticks([-0.02,0,0.02,0.04,0.06,0.08])
  #axs[2].set_yticks([-0.04,-0.03,-0.02,-0.01,0,0.01])

  if Q2 == 1.27**2: axs[2].text(0.05,0.05,r'$Q^2 = m_c^2$',             transform=axs[2].transAxes,size=30)
  else:             axs[2].text(0.05,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=axs[2].transAxes,size=25)

  axs[1].text(0.05,0.05,r'$z=%s$'%z, transform=axs[1].transAxes,size=25)
  
  handles, labels = [],[]
  handles.append(hand['p,pi+'])
  handles.append(hand['p,pi-'])
  handles.append(hand['n,pi+'])
  handles.append(hand['n,pi-'])
  labels.append(r'\boldmath$p,\pi^+$')
  labels.append(r'\boldmath$p,\pi^-$')
  labels.append(r'\boldmath$n,\pi^+$')
  labels.append(r'\boldmath$n,\pi^-$')
  axs[1].legend(handles,labels,loc='upper left',fontsize=20, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 2, columnspacing = 0.5)
  handles, labels = [],[]
  handles.append(hand['p,K+'])
  handles.append(hand['p,K-'])
  handles.append(hand['n,K+'])
  handles.append(hand['n,K-'])
  labels.append(r'\boldmath$p,K^+$')
  labels.append(r'\boldmath$p,K^-$')
  labels.append(r'\boldmath$n,K^+$')
  labels.append(r'\boldmath$n,K^-$')
  axs[2].legend(handles,labels,loc='upper left',fontsize=20, frameon=False, handlelength = 1.0, handletextpad = 0.5, ncol = 2, columnspacing = 0.5)
  py.tight_layout()
  py.subplots_adjust(hspace=0)

  filename+='.png'

  checkdir('%s/gallery'%wdir)
  py.savefig(filename)
  print ('Saving figure to %s'%filename)
  py.clf()

if __name__=="__main__":

    
    ap = argparse.ArgumentParser()

    ap.add_argument('-d'   ,'--directory' ,type=str   ,default='unamed'   ,help='directory name to store results')
    ap.add_argument('-Q2'  ,'--Q2'        ,type=float ,default='unamed'   ,help='Q2 value')
    ap.add_argument('-t'   ,'--tar'       ,type=str   ,default='unamed'   ,help='target')
    ap.add_argument('-s'   ,'--stf'       ,type=str   ,default='unamed'   ,help='structure function')
    args = ap.parse_args()

    gen_pstf(args.directory,Q2=args.Q2,tar=args.tar,stf=args.stf)






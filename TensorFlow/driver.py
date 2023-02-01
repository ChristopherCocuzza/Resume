#!/usr/bin/env python
import sys,os
import numpy as np
import copy

#import gen_table
import gen_model
import gen_table

import tensorflow as tf
from tensorflow import keras

import matplotlib
matplotlib.rc('text',usetex=True)
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py
import matplotlib.pyplot as plt

from tools.tools import load,load2,save,checkdir,deconvert
from qcdlib import mellin

mellin = mellin.DMELLIN()

#--use environment tf-gpu (conda activate tf-gpu)

#--index conventions
#--for UU
#--0: eta
#--1: PhT
#--2: realN
#--3: imagN
#--4: realT
#--5: imagT

#--for UT
#--0: eta
#--1: PhT
#--2: realN
#--3: imagN
#--4: realM
#--5: imagM
#--6: realT
#--7: imagT


xsecs = ['UU','UT']
xsecs = ['UU']

channels = {}
#--unpolarized channels
channels['UU'] =      ['QQ,QQ','QQp,QpQ','QQp,QQp','QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ']
channels['UU'].extend(['QQB,GG','GQ,GQ','GQ,QG','QG,GQ','QG,QG','GG,GG','GG,QQB'])

#--polarized channels (only five are nonzero)
channels['UT'] =      ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ','GQ,QG']

parts = ['NM','NCM']


def plot_prediction_UU(xsec,channel,flav):

    num_inputs = 3
    num_params = 5
    #--get data from grids
    DATA  = np.load('modelgrids/%s/%s-%s.npy'%(xsec,channel,flav),allow_pickle=True).item()
    data0 = DATA['data']
    mean,std,factor = DATA['mean'],DATA['std'],list(DATA['factor'])

    #--get specific kinematics from data
    data  = list(data0)
    ETA = np.unique(data[0])
    PHT = np.unique(data[1])
    eta = ETA[0]
    PhT = PHT[0]
    idx1 = np.where(data[0]==eta)[0]
    idx2 = np.where(data[1]==PhT)[0]
    idx  = [value for value in idx1 if value in idx2]
    for i in range(len(data)):   data[i]   = data[i][idx]
    for i in range(len(factor)): factor[i] = factor[i][idx]

    N = mellin.N
    realN = np.real(N)
    result= np.zeros((num_params,len(N)))
    result[0] = eta*np.ones(len(N))
    result[1] = PhT*np.ones(len(N))
    result[2] = (realN - mean[2])/std[2] #--is this right?  Should it just be realN?

    #--get prediction from model
    model = keras.models.load_model('models/%s/%s-%s'%(xsec,channel,flav))
    IN  = np.array(result[:num_inputs]).T
    predict = model.predict(IN).T
    result[3] = predict[0]
    result[4] = predict[1]

    for i in range(len(data)):
        if i in [3,4]: data[i] = (data[i]*std[i] + mean[i])/PhT**6
        else:          data[i] = (data[i]*std[i] + mean[i])


    for i in range(len(result)):
        if i in [3,4]: result[i] = (result[i]*std[i] + mean[i])/PhT**6
        else:          result[i] = (result[i]*std[i] + mean[i])
    
    #--get N and T
    realN        = data[2]
    realT, imagT = data[3], data[4]

    modelrealT = result[3]
    modelimagT = result[4]


    #--make plot
    nrows,ncols=2,1
    fig = py.figure(figsize=(ncols*10,nrows*5))
    ax11=py.subplot(nrows,ncols,1)
    ax21=py.subplot(nrows,ncols,2)

    hand = {}
    color = 'red'
    marker = 'o'
    s = 60
    hand['grid'] = ax11.scatter(realN,realT,c=color,s=s,marker=marker)
    hand['grid'] = ax21.scatter(realN,imagT,c=color,s=s,marker=marker)
  
    hand['NN']  ,= ax11.plot(realN,modelrealT,color='blue')
    hand['NN']  ,= ax21.plot(realN,modelimagT,color='blue')
 
    for ax in [ax11,ax21]:
        #ax.semilogy()
        #ax.set_xticks([0.00,1.0,2.0,3.0,4.0,5.0,6.0,7.0])
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        #ax.set_ylim(5e-6,9e-3)
        ax.set_xlim(-45,5)
 
    ax11.tick_params(labelbottom=False)

    ax21.set_xlabel(r'\textrm{\textbf{Real(N)}}',size=30)
    ax11.set_ylabel(r'\textrm{\textbf{Real(T)}}',size=30,labelpad=12)
    ax21.set_ylabel(r'\textrm{\textbf{Imag(T)}}',size=30,labelpad=12)

    eta = eta*std[0] + mean[0]
    PhT = PhT*std[1] + mean[1]

    ax21.text(0.05,0.85,r'\boldmath$\eta=%3.2f$'%eta   ,transform=ax21.transAxes, size=30)
    ax21.text(0.05,0.75,r'\boldmath$P_{hT}=%3.2f$'%PhT ,transform=ax21.transAxes, size=30)
 
    handles,labels = [], []
    handles.append(hand['grid'])
    handles.append(hand['NN'])
    labels.append(r'\textrm{grids}')
    labels.append(r'\textrm{NN}')
    ax11.legend(handles,labels,loc='upper left',fontsize=30, frameon=False, handlelength = 1.50, handletextpad = 0.4)



    checkdir('gallery')
    checkdir('gallery/predictions/%s'%xsec)
    py.tight_layout()
    py.subplots_adjust(hspace=0.05,wspace=0.05)
    filename='gallery/predictions/%s/%s-%s'%(xsec,channel,flav)
    filename+='.png'

    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    plt.close()

def plot_prediction_UT(xsec,channel,flav,part):

    #--make plot
    nrows,ncols=2,2
    fig = py.figure(figsize=(ncols*8,nrows*4))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax21=py.subplot(nrows,ncols,3)
    ax22=py.subplot(nrows,ncols,4)


    flav = flav[0]
    num_inputs = 4
    num_params = 6
    #--get data from grids
    DATA  = np.load('modelgrids/%s/%s/%s-%s.npy'%(xsec,part,channel,flav),allow_pickle=True).item()
    data0 = copy.deepcopy(DATA['data'])
    mean,std,factor = DATA['mean'],DATA['std'],list(DATA['factor'])

    #--get specific kinematics from data
    #--fixed M
    data  = list(copy.deepcopy(data0))
    ETA = np.unique(data[0])
    PHT = np.unique(data[1])
    REALM  = np.unique(data[3])
    eta = ETA[4]
    PhT = PHT[0]
    realM = REALM[65]
    idx1 = np.where(data[0]==eta)[0]
    idx2 = np.where(data[1]==PhT)[0]
    idx3 = np.where(data[3]==realM)[0]
    idx  = [value for value in idx1 if value in idx2 and value in idx3]
    for i in range(len(data)):   data[i]   = data[i][idx]

    N = np.real(mellin.N)
    M = np.real(mellin.M)
    
    L = len(N)
    result= np.zeros((num_params,L))
    result[0] = eta*np.ones(L)
    result[1] = PhT*np.ones(L)
    result[2] = data[2]
    result[3] = realM

    #--get prediction from model
    model = keras.models.load_model('models/%s/%s/%s-%s'%(xsec,part,channel,flav))
    IN  = np.array(result[:num_inputs]).T
    predict = model.predict(IN).T
    result[4] = predict[0]
    result[5] = predict[1]



    for i in range(len(data)):
        if i in [4,5]: data[i] = (data[i]*std[i] + mean[i])/PhT**5
        else:          data[i] = (data[i]*std[i] + mean[i])


    for i in range(len(result)):
        if i in [4,5]: result[i] = (result[i]*std[i] + mean[i])/PhT**5
        else:          result[i] = (result[i]*std[i] + mean[i])
   
    #--get N and T
    realN = data[2]
    realT, imagT = data[4], data[5]

    modelrealT = result[4]
    modelimagT = result[5]


    hand = {}
    color = 'red'
    marker = 'o'
    s = 60
    hand['grid'] = ax11.scatter(realN,realT,c=color,s=s,marker=marker)
    hand['grid'] = ax21.scatter(realN,imagT,c=color,s=s,marker=marker)
  
    hand['NN']  ,= ax11.plot(realN,modelrealT,color='blue')
    hand['NN']  ,= ax21.plot(realN,modelimagT,color='blue')

    eta = eta*std[0] + mean[0]
    PhT = PhT*std[1] + mean[1]
    realM = realM*std[3] + mean[3]

    ax21.text(0.05,0.85,r'\boldmath$\eta=%3.2f$'%eta   ,transform=ax21.transAxes, size=30)
    ax21.text(0.05,0.75,r'\boldmath$P_{hT}=%3.2f$'%PhT ,transform=ax21.transAxes, size=30)
    ax21.text(0.05,0.65,r'\boldmath$realM=%3.2f$'%realM,transform=ax21.transAxes, size=30)




    #--get specific kinematics from data
    #--fixed N
    mean,std,factor = DATA['mean'],DATA['std'],list(DATA['factor'])
    data  = list(copy.deepcopy(data0))
    REALN  = np.unique(data[2])
    eta = ETA[4]
    PhT = PHT[0]
    realN = REALN[65]
    idx1 = np.where(data[0]==eta)[0]
    idx2 = np.where(data[1]==PhT)[0]
    idx3 = np.where(data[2]==realN)[0]
    idx  = [value for value in idx1 if value in idx2 and value in idx3]
    for i in range(len(data)):   data[i]   = data[i][idx]


    N = np.real(mellin.N)
    M = np.real(mellin.M)
    
    L = len(M)
    result= np.zeros((num_params,L))
    result[0] = eta*np.ones(L)
    result[1] = PhT*np.ones(L)
    result[2] = realN
    result[3] = data[3]

    #--get prediction from model
    IN  = np.array(result[:num_inputs]).T
    predict = model.predict(IN).T
    result[4] = predict[0]
    result[5] = predict[1]

    #--things look good up until here...
    #print(data[4])
    #print(result[4])


    for i in range(len(data)):
        if i in [4,5]: data[i] = (data[i]*std[i] + mean[i])/PhT**5
        else:          data[i] = (data[i]*std[i] + mean[i])


    for i in range(len(result)):
        if i in [4,5]: result[i] = (result[i]*std[i] + mean[i])/PhT**5
        else:          result[i] = (result[i]*std[i] + mean[i])
   
    print(data[4])
    print(result[4])

 
    #--get N and T
    realM = data[3]
    realT, imagT = data[4], data[5]

    modelrealT = result[4]
    modelimagT = result[5]

    hand = {}
    color = 'red'
    marker = 'o'
    s = 60
    hand['grid'] = ax12.scatter(realM,realT,c=color,s=s,marker=marker)
    hand['grid'] = ax22.scatter(realM,imagT,c=color,s=s,marker=marker)
  
    hand['NN']  ,= ax12.plot(realM,modelrealT,color='blue')
    hand['NN']  ,= ax22.plot(realM,modelimagT,color='blue')

    realN = realN*std[2] + mean[2]

    ax22.text(0.05,0.65,r'\boldmath$realN=%3.2f$'%realN,transform=ax22.transAxes, size=30)

 
    for ax in [ax11,ax12,ax21,ax22]:
        #ax.semilogy()
        #ax.set_xticks([0.00,1.0,2.0,3.0,4.0,5.0,6.0,7.0])
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        #ax.set_ylim(5e-6,9e-3)
        ax.set_xlim(-45,5)
 
    ax11.tick_params(labelbottom=False)

    ax21.set_xlabel(r'\textrm{\textbf{Real(N)}}',size=30)
    ax22.set_xlabel(r'\textrm{\textbf{Real(M)}}',size=30)
    ax11.set_ylabel(r'\textrm{\textbf{Real(T)}}',size=30,labelpad=12)
    ax21.set_ylabel(r'\textrm{\textbf{Imag(T)}}',size=30,labelpad=12)

    ax12.tick_params(labelleft=False,labelright=True)
    ax22.tick_params(labelleft=False,labelright=True)
 
    handles,labels = [], []
    handles.append(hand['grid'])
    handles.append(hand['NN'])
    labels.append(r'\textrm{grids}')
    labels.append(r'\textrm{NN}')
    ax11.legend(handles,labels,loc='upper left',fontsize=30, frameon=False, handlelength = 1.50, handletextpad = 0.4)



    checkdir('gallery/predictions/%s/%s'%(xsec,part))
    py.tight_layout()
    py.subplots_adjust(hspace=0.05,wspace=0.05)
    filename='gallery/predictions/%s/%s/%s-%s'%(xsec,part,channel,flav)
    filename+='.png'

    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    plt.close()



def plot_history(xsec,channel,flav,part=None):

    if xsec=='UU': history = np.load('history/%s/%s-%s.npy'%(xsec,channel,flav),allow_pickle=True).item()
    if xsec=='UT': history = np.load('history/%s/%s/%s-%s.npy'%(xsec,part,channel,flav),allow_pickle=True).item()
    loss, accu = np.array(history['loss']),np.array(history['val_accuracy'])

    epochs = [i for i in range(len(loss))]

    #--make plot
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*10,nrows*5))
    ax11=py.subplot(nrows,ncols,1)

    hand = {}
    hand['loss'] ,= ax11.plot(epochs,loss  ,color='red')
    hand['accu'] ,= ax11.plot(epochs,1-accu,color='blue')
 
    for ax in [ax11]:
        ax.semilogy()
        #ax.set_xticks([0.00,1.0,2.0,3.0,4.0,5.0,6.0,7.0])
        ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)
        #ax.set_ylim(5e-6,9e-3)
        ax.set_xlim(0,len(epochs))
        ax.set_ylim(1e-3,1)
 
    #ax11.tick_params(labelbottom=False)

    ax11.set_xlabel(r'\textrm{\textbf{Epoch}}',size=30)

    #ax21.text(0.05,0.85,r'\boldmath$\eta=%3.2f$'%eta   ,transform=ax21.transAxes, size=30)
    #ax21.text(0.05,0.75,r'\boldmath$P_{hT}=%3.2f$'%PhT ,transform=ax21.transAxes, size=30)
 
    handles,labels = [], []
    handles.append(hand['loss'])
    handles.append(hand['accu'])
    labels.append(r'\textrm{loss}')
    labels.append(r'\textrm{$1~-$ accuracy}')
    ax11.legend(handles,labels,loc='upper right',fontsize=30, frameon=False, handlelength = 0.80, handletextpad = 0.4)



    if xsec=='UU': checkdir('gallery/history/%s'%xsec)
    if xsec=='UT': checkdir('gallery/history/%s/%s'%(xsec,part))
    py.tight_layout()
    py.subplots_adjust(hspace=0.05,wspace=0.05)
    if xsec=='UU': filename='gallery/history/%s/%s-%s'%(xsec,channel,flav)
    if xsec=='UT': filename='gallery/history/%s/%s/%s-%s'%(xsec,part,channel,flav)
    filename+='.png'

    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    plt.close()




if __name__== "__main__":

  #xsec, channel, flav = 'UU', 'QQ,QQ', 'u,u'
  xsec, channel, flav, part = 'UT', 'QQ,QQ', 'u,u', 'NM'

  if xsec=='UU': plot_prediction_UU(xsec,channel,flav) 
  if xsec=='UT': plot_prediction_UT(xsec,channel,flav,part) 
  sys.exit()

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
              if xsec=='UU': 
                  plot_prediction_UU(xsec,channel,flav)
                  plot_history(xsec,channel,flav)
              if xsec=='UT': 
                  for part in parts: plot_prediction_UT(xsec,channel,flav,part)
                  for part in parts: plot_history(xsec,channel,flav,part)










#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys
import argparse

#--TensorFlow and tf.keras
import tensorflow as tf
#print('Num GPUS available: %s'%len(tf.config.experimental.list_physical_devices('GPU')))
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#tf.config.experimental.set_visible_devices([],'GPU')
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
from functools import partial
#from sklearn.preprocessing import MinMaxScalar
from sklearn.model_selection import train_test_split

#--helper libraries
import numpy as np


#--from fitpack
from tools.tools import load,save,checkdir

def get_tname_UU(xsec,channel,flav):

  tname = 'modelgrids/%s/%s-%s.npy'%(xsec,channel,flav)
  print('Loading table for machine learning: %s'%tname)
  return tname

def get_data_UU(tname,test_size):

  #--data is loaded in shape (# params, # points)
  data = np.load(tname,allow_pickle=True).item()
  data = data['data']

  num_params  = len(data)
  num_outputs = 2

  #split into input (x) and output (y)
  x = data[:num_params-num_outputs]
  y = data[num_params-num_outputs:]

  x, y = x.T, y.T

  #--split the data into training and test set
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size, random_state=0)
  
  return x_train, x_test, y_train, y_test

def gen_model_UU(EPOCHS,BATCH_SIZE,lr,test_size,xsec,channel,flav):

  tname=get_tname_UU(xsec,channel,flav)

  x_train, x_test, y_train, y_test = get_data_UU(tname,test_size)

  #--unnormalized x
  #x_unnormal = (x_train.T[0]*std['x'])+mean['x']

  #-------------------------
  #--output is redefined so that loss is weighted by 1/x.  This biases the model towards low x. 
  #y_train = y_train.T

  #y_train[0] = (y_train[0]*std['ReT'])+mean['ReT']
  #y_train[1] = (y_train[1]*std['ImT'])+mean['ImT']

  #y_train[0] = y_train[0]/(x_unnormal)**0.5
  #y_train[1] = y_train[1]/(x_unnormal)**0.5

  #mean0, std0 = np.mean(y_train[0]), np.std(y_train[0])
  #mean1, std1 = np.mean(y_train[1]), np.std(y_train[1])

  #for i in range(len(y_train[0])):
  #  y_train[0][i] = (y_train[0][i] - mean0)/std0
  #  y_train[1][i] = (y_train[1][i] - mean1)/std1

  #y_train = y_train.T
  #-------------------------


  #--input: x, Q2, ReN, ImN
  #--output: ReT, ImT
  input=Input(shape=x_train[0].shape)
  x=Dense(120, activation='relu')(input)
  x=Dropout(0.01)(x)
  x=Dense(120, activation='relu')(x)
  x=Dropout(0.01)(x)
  x=Dense(120, activation='relu')(x)
  x=Dropout(0.01)(x)
  output=Dense(2)(x)
  model=Model(input,output)
  model.summary()
  model_optimizer=Adam(lr=lr, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08, decay=0.00001)
  model.compile(model_optimizer,loss='mean_squared_error',metrics=['accuracy'])
  history=model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS, validation_split=0.2, verbose=1)

  checkdir('models')
  checkdir('models/%s'%xsec)
  filename = 'models/%s/%s-%s'%(xsec,channel,flav) 
  model.save(filename)
  print('Saving model data to %s'%filename)

  history = history.history

  checkdir('history')
  checkdir('history/%s'%xsec)
  filename = 'history/%s/%s-%s.npy'%(xsec,channel,flav) 
  np.save(filename,history)
  #save(history,filename)
  print('Saving history data to %s'%filename)


def get_tname_UT(xsec,channel,flav,part):

  tname = 'modelgrids/%s/%s/%s-%s.npy'%(xsec,part,channel,flav)
  print('Loading table for machine learning: %s'%tname)
  return tname

def get_data_UT(tname,test_size):

  #--data is loaded in shape (# params, # points)
  data = np.load(tname,allow_pickle=True).item()
  data = data['data']

  num_params  = len(data)
  num_outputs = 2

  #split into input (x) and output (y)
  x = data[:num_params-num_outputs]
  y = data[num_params-num_outputs:]

  x, y = x.T, y.T

  #--split the data into training and test set
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size, random_state=0)
  
  return x_train, x_test, y_train, y_test

def gen_model_UT(EPOCHS,BATCH_SIZE,lr,test_size,xsec,channel,flav,part):

  tname=get_tname_UT(xsec,channel,flav,part)

  x_train, x_test, y_train, y_test = get_data_UT(tname,test_size)

  #--unnormalized x
  #x_unnormal = (x_train.T[0]*std['x'])+mean['x']

  #-------------------------
  #--output is redefined so that loss is weighted by 1/x.  This biases the model towards low x. 
  #y_train = y_train.T

  #y_train[0] = (y_train[0]*std['ReT'])+mean['ReT']
  #y_train[1] = (y_train[1]*std['ImT'])+mean['ImT']

  #y_train[0] = y_train[0]/(x_unnormal)**0.5
  #y_train[1] = y_train[1]/(x_unnormal)**0.5

  #mean0, std0 = np.mean(y_train[0]), np.std(y_train[0])
  #mean1, std1 = np.mean(y_train[1]), np.std(y_train[1])

  #for i in range(len(y_train[0])):
  #  y_train[0][i] = (y_train[0][i] - mean0)/std0
  #  y_train[1][i] = (y_train[1][i] - mean1)/std1

  #y_train = y_train.T
  #-------------------------


  #--input: x, Q2, ReN, ImN
  #--output: ReT, ImT
  input=Input(shape=x_train[0].shape)
  x=Dense(120, activation='relu')(input)
  x=Dropout(0.01)(x)
  x=Dense(120, activation='relu')(x)
  x=Dropout(0.01)(x)
  x=Dense(120, activation='relu')(x)
  x=Dropout(0.01)(x)
  output=Dense(2)(x)
  model=Model(input,output)
  model.summary()
  model_optimizer=Adam(lr=lr, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08, decay=0.00001)
  model.compile(model_optimizer,loss='mean_squared_error',metrics=['accuracy'])
  history=model.fit(x_train,y_train,batch_size=BATCH_SIZE,epochs=EPOCHS, validation_split=0.2, verbose=1)

  checkdir('models/%s/%s'%(xsec,part))
  filename = 'models/%s/%s/%s-%s'%(xsec,part,channel,flav) 
  model.save(filename)
  print('Saving model data to %s'%filename)

  history = history.history

  checkdir('history')
  checkdir('history/%s/%s'%(xsec,part))
  filename = 'history/%s/%s/%s-%s.npy'%(xsec,part,channel,flav) 
  np.save(filename,history)
  #save(history,filename)
  print('Saving history data to %s'%filename)


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-xsec',   '--xsec'   ,type=str,   default='UT')
    ap.add_argument('-channel','--channel',type=str,   default='QQ,QQ')
    ap.add_argument('-flav',   '--flav'   ,type=str,   default='u,u')
    ap.add_argument('-part',   '--part'   ,type=str,   default='NM')
    args = ap.parse_args()

    xsec    = args.xsec
    channel = args.channel
    flav    = args.flav
    part    = args.part

    if xsec=='UU':
        EPOCHS = 5000
        BATCH_SIZE = 3200
        lr = 0.0001
        test_size = 0.02
    if xsec=='UT':
        EPOCHS = 1000
        BATCH_SIZE = 3200
        lr = 0.0001
        test_size = 0.02

    #--generate model
    if xsec=='UU': gen_model_UU(EPOCHS,BATCH_SIZE,lr,test_size,xsec,channel,flav)
    if xsec=='UT': gen_model_UT(EPOCHS,BATCH_SIZE,lr,test_size,xsec,channel,flav,part)









# import cPickle 
try:    import cPickle
except: import _pickle as cPickle
import sys, os
import zlib
import numpy as np

def checkdir(path):
  if not os.path.exists(path): 
    os.makedirs(path)

def tex(x):
  return r'$\mathrm{'+x+'}$'

def save(data,name):  
  compressed=zlib.compress(cPickle.dumps(data))
  f=open(name,"wb")
  try:
      f.writelines(compressed)
  except:
      f.write(compressed)
  f.close()

def load(name): 
  compressed=open(name,"rb").read()
  try:    data=cPickle.loads(zlib.decompress(compressed))
  except: data=cPickle.loads(zlib.decompress(compressed, zlib.MAX_WBITS|32))
  return data

def load2(name): 
  compressed=open(name,"rb").read()
  data=cPickle.loads(compressed)
  return data

def isnumeric(value):
  try:
    int(value)
    return True
  except:
    return False

def ERR(msg):
  print(msg)
  sys.exit()

def lprint(msg):
  sys.stdout.write('\r')
  sys.stdout.write(msg)
  sys.stdout.flush()

def convert(DATA):
    data   = DATA['data']
    factor = DATA['factor']
    mean, std = [],[]
    for i in range(len(data)):
        data[i] = factor[i]*data[i]
        mean.append(np.mean(data[i]))
        std .append(np.std (data[i]))
        data[i] = (data[i]-mean[i])/std[i]
    mean,std = np.array(mean), np.array(std)
    return data,mean,std

def deconvert(data,mean,std,factor):
    for i in range(len(data)): data[i] = (data[i]*std[i] + mean[i])/factor[i]
    return data










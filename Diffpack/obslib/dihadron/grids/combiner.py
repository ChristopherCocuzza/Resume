#!/usr/bin/env python
import os,sys,shutil
import argparse
import time
from qcdlib   import aux
from tools.config import conf
from tools.tools import checkdir, load, save
from obslib.dihadron.reader import READER

#--unpolarized channels
channels_UU =      ['QQ,QQ','QQp,QpQ','QQp,QQp','QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ']
channels_UU.extend(['QQB,GG','GQ,GQ','GQ,QG','QG,GQ','QG,QG','GG,GG','GG,QQB'])

#--polarized channels (only five are nonzero)
channels_UT =      ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ','GQ,QG']


class COMBINE():

    def __init__(self,RS,xsec):

        if xsec=='UU': self.channels = channels_UU
        if xsec=='UT': self.channels = channels_UT
        self.xsec = xsec

        self.combine_channels(RS)

    def combine_channels(self,RS):
      conf['aux']=aux.AUX()
      conf['path2dihadrontab']='%s/grids/dihadron'%os.environ['FITPACK']
      path2dihadrontab = conf['path2dihadrontab']
    
      directory  = '%s/RS%s/%s' %(path2dihadrontab,int(RS),self.xsec)
   
      channels = self.channels
 
      #--loop over grid points
      npts = int(len(os.listdir(directory))/len(channels))

      for i in range(npts):
          TAB = {}
          for channel in channels:
              TAB[channel] = {}
              fname = '%s-%s.melltab'%(i,channel)
              tab   =  load('%s/%s' %(directory,fname))
              for key in list(tab):
                  if key in ['eta','RS','PhT']: continue
                  TAB[channel][key] = tab[key]
              TAB['eta']   = tab['eta']
              TAB['RS']    = tab['RS']
              TAB['PhT']   = tab['PhT']
    
          fname  = '%s.melltab'%i
          newname = '%s/%s'%(directory,fname)
   
          print('Combined channels: Saving new grid to %s'%newname)
          save(TAB,newname)

          #--remove split channels
          for channel in channels:
              fname = '%s-%s.melltab'%(i,channel)
              os.remove('%s/%s' %(directory,fname))
    

if __name__ == "__main__":

  RS = [200,500]

  for rs in RS: 
      COMBINE(rs,'UU')
      COMBINE(rs,'UT')










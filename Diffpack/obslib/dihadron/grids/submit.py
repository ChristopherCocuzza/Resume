#!/usr/bin/env python
import os,sys
import argparse
import time
import numpy as np
from qcdlib   import aux
from tools.config import conf
from tools.tools import checkdir
from obslib.dihadron.reader import READER

#--user dependent paths
user    ='ccocuzza'
path    ='/work/JAM/'
fitpack ='%s/ccocuzza/Diffpack'%path
wdir    =os.getcwd()
#--get python version
version = sys.version[0]
python  ='%s/apps/anaconda%s/bin/python'%(path,version)

template="""#!/bin/csh
#SBATCH --account=jam
#SBATCH --nodes 1
#SBATCH --partition=production
#SBATCH --cpus-per-task 1
#SBATCH --mem=1G
#SBATCH --time=96:00:00
#SBATCH --constraint=general
#SBATCH --job-name=<fname>
#SBATCH --output=out/<fname>.out
#SBATCH --error=err/<fname>.err

<python> <fitpack>/obslib/dihadron/mellin.py -xsec <xsec> --channel <channel> -eta <eta> -PhT <PhT> -RS <RS> -name <name> 
"""

#--opening angle = 0.3 for RS = 200 data
#--tables:  col:  tar:   RS:    binned:     Y:             PhT:              npts:
#----------------------------------------------------------------------------------
#--4100:    STAR  pp     200    PhT         -0.84 - 0.00   3.61 - 10.32      5
#--4101:    STAR  pp     200    PhT          0.00 - 0.84   3.61 - 10.32      5
#--4110:    STAR  pp     200    M           -0.84 - 0.00   4.73 - 10.63      5
#--4111:    STAR  pp     200    M            0.00 - 0.84   4.73 - 10.63      5
#--4120:    STAR  pp     200    eta         -0.75 - 0.75   5.99 -  6.04      4
#--5000:    STAR  pp     500    M           -1.00 - 0.00   4.20 - 13.40      34
#--5001:    STAR  pp     500    M            0.00 - 1.00   4.20 - 13.40      34
#--5020:    STAR  pp     500    eta         -0.85 - 0.85   13.00             7

#--Y ranges:
#--1. RS = 200: -0.84 < Y < 0.84
#--2. RS = 500: -1.00 < Y < 1.00

#--for both RS = 200 and RS = 500:
#--eta range: -1   < eta < 1

#--for RS = 200
#--PhT range: 2.5  < PhT < 15
#--for RS = 500
#--PhT range: 4.0  < PhT < 19

#--M does not appear in hard kernels and is not needed

#--unpolarized channels
channels_UU =      ['QQ,QQ','QQp,QpQ','QQp,QQp','QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ']
channels_UU.extend(['QQB,GG','GQ,GQ','GQ,QG','QG,GQ','QG,QG','GG,GG','GG,QQB'])

#--polarized channels (only five are nonzero)
channels_UT =      ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ','GQ,QG']


class GEN_GRID():

    def __init__(self,xsec,RS,force,sfarm,test):

        self.xsec    = xsec
        self.force   = force
        self.sfarm   = sfarm
        self.test    = test
        self.RS      = RS  

        if xsec=='UT': self.channels = channels_UT
        if xsec=='UU': self.channels = channels_UU

        self.gen_grids()

    def gen_script(self,channel,eta,PhT,i):
        fname='%s-%s-%s'%(self.xsec,i,channel)
        print('Submitted %s'%fname)
        script=template[:]
        script=script.replace('<fname>',fname)
        script=script.replace('<xsec>',str(self.xsec))
        script=script.replace('<channel>',str(channel))
        script=script.replace('<eta>',str(eta))
        script=script.replace('<PhT>',str(PhT))
        script=script.replace('<RS>',str(self.RS))
        script=script.replace('<name>',str(i))
        script=script.replace('<python>',str(python))
        script=script.replace('<fitpack>',str(fitpack))

        checkdir('out')
        checkdir('err')
        F=open('current.sbatch','w')
        F.writelines(script)
        F.close()

    def submit(self,channel,eta,PhT,i):

        directory = '%s/grids/dihadron/RS%s/%s'%(os.environ['FITPACK'],int(self.RS),self.xsec)
        checkdir(directory)

        fname = '%s-%s.melltab'%(i,channel)
        grids = os.listdir(directory)
        if self.force==False:
            if fname in grids:
                print('Grid %s %s already exists.  Skipping...'%(directory,fname))
                return

        self.gen_script(channel,eta,PhT,i)
        if self.sfarm: os.system('sbatch current.sbatch')
        else:          os.system('source current.sbatch')
        if self.test: sys.exit()
        time.sleep(0.02)

    def gen_grids(self):

        #--choose grids

        if self.xsec == 'UU':
            if self.RS == 200: 
                ETA = np.linspace(-1 ,1 ,5)
                PHT = np.linspace(2.5,15,5)
            if self.RS == 500: 
                ETA = np.linspace(-1 ,1 ,4)
                PHT = np.linspace(4,19,5)
        if self.xsec == 'UT':
            if self.RS == 200: 
                ETA = np.linspace(-1 ,1 ,5)
                PHT = np.linspace(2.5,15,5)
            if self.RS == 500: 
                ETA = np.linspace(-1 ,1 ,4)
                PHT = np.linspace(4,19,5)

        cnt = 0
        for channel in self.channels:
            for i in range(len(ETA)):
                for j in range(len(PHT)):
                    idx = i*len(PHT) + j
                    self.submit(channel,ETA[i],PHT[j],idx)
                    cnt+=1

        print('Submitted %s jobs for RS = %d'%(cnt,self.RS))

if __name__ == "__main__":

  checkdir('out')
  checkdir('err')

  #--choose if grids should be forced to be regenerated, or to skip grids that have already been generated
  force = False
  #--choose if grids should be sent to the farm or ran locally
  sfarm = True
  #--if testing, only submits a single grid
  test  = False

  RS = [200,500]

  for rs in RS:
      
      GEN_GRID('UT',rs,force,sfarm,test)
      GEN_GRID('UU',rs,force,sfarm,test)








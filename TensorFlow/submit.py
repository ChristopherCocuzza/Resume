#!/usr/bin/env python
import os,sys
import argparse
import time
import numpy as np
from tools.tools import checkdir

#--user dependent paths
user    ='ccocuzza'
path    ='/w/jam-sciwork18/'
cwd     ='%s/%s/Resume/ml'%(path,user)
wdir    =os.getcwd()
#--get python version
version = sys.version[0]
python  ='%s/apps/anaconda%s/bin/python'%(path,version)

template="""#!/bin/csh
#SBATCH --account=jam
#SBATCH --nodes 1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task 1
#SBATCH --mem=2G
#SBATCH --gres=gpu:TitanRTX:1
#SBATCH --time=96:00:00
#SBATCH --job-name=<fname>
#SBATCH --output=out/<fname>.out
#SBATCH --error=err/<fname>.err
conda activate tf-gpu
./gen_model.py -xsec <xsec> --channel <channel> --flav <flav> --part <part>
"""

xsecs = ['UU','UT']
xsecs = ['UT']

channels = {}
#--unpolarized channels
channels['UU'] =      ['QQ,QQ','QQp,QpQ','QQp,QQp','QQB,QpQBp','QQB,QBpQp','QQB,QQB','QQB,QBQ']
channels['UU'].extend(['QQB,GG','GQ,GQ','GQ,QG','QG,GQ','QG,QG','GG,GG','GG,QQB'])

#--polarized channels (only five are nonzero)
channels['UT'] =      ['QQ,QQ','QQp,QpQ','QQB,QQB','QQB,QBQ','GQ,QG']

parts = ['NM','NCM']

class GEN_MODEL():

    def __init__(self,xsec,channel,flav,part,force,sfarm,test):

        self.xsec    = xsec
        self.channel = channel
        self.flav    = flav
        self.part    = part
        self.force   = force
        self.sfarm   = sfarm
        self.test    = test

        self.submit()      

    def gen_script(self):
        if self.xsec=='UU': fname='%s-%s-%s'%(self.xsec,self.channel,self.flav)
        if self.xsec=='UT': fname='%s-%s-%s-%s'%(self.xsec,self.part,self.channel,self.flav)
        print('Submitted %s'%fname)
        script=template[:]
        script=script.replace('<fname>',fname)
        script=script.replace('<xsec>',str(self.xsec))
        script=script.replace('<channel>',str(self.channel))
        script=script.replace('<flav>',str(self.flav))
        script=script.replace('<part>',str(self.part))
        script=script.replace('<python>',str(python))
        script=script.replace('<cwd>',str(cwd))

        checkdir('out')
        checkdir('err')
        F=open('current.sbatch','w')
        F.writelines(script)
        F.close()

    def submit(self):

        if xsec=='UU': directory = './models/%s'%self.xsec
        if xsec=='UT': directory = './models/%s/%s'%(self.xsec,self.part)
        checkdir(directory)

        fname = '%s-%s'%(channel,flav)
        models = os.listdir(directory)
        if self.force==False:
            if fname in models:
                print('Model %s/%s already exists.  Skipping...'%(directory,fname))
                return

        self.gen_script()
        if self.sfarm: os.system('sbatch current.sbatch')
        else:          os.system('source current.sbatch')
        if self.test: sys.exit()
        time.sleep(0.50)


if __name__ == "__main__":

  checkdir('out')
  checkdir('err')

  #--choose if grids should be forced to be regenerated, or to skip grids that have already been generated
  force = False
  #--choose if grids should be sent to the farm or ran locally
  sfarm = True
  #--if testing, only submits a single grid
  test  = False

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
              if xsec=='UU': GEN_MODEL(xsec,channel,flav,None,force,sfarm,test)
              if xsec=='UT':
                  for part in parts: GEN_MODEL(xsec,channel,flav,part,force,sfarm,test)







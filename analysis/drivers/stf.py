#!/usr/bin/env python
import os,sys
import time
import argparse
#--set lhapdf data path
version = int(sys.version[0])
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import kmeanconf as kc

from tools.tools import checkdir

#--from obslib
from analysis.obslib import stf, ht, off

#--user dependent paths
user    ='ccocuzza'
path    ='/work/JAM/'
fitpack ='%s/ccocuzza/fitpack'%path
wdir    =os.getcwd()
#--get python version
version = sys.version[0]
python  ='%s/apps/anaconda%s/bin/python'%(path,version)

analysis = 'analysis-hx'
os.environ["analysis"] = '/work/JAM/ccocuzza/%s/analysis'%analysis
#--account can be chosen as jam or f4thy
template="""#!/bin/csh
#SBATCH --account=jam
#SBATCH --nodes 1
#SBATCH --partition=production
#SBATCH --cpus-per-task 1
#SBATCH --mem=1G
#SBATCH --time=90:00:00
#SBATCH --constraint=general
#SBATCH --job-name="<name>"
#SBATCH --output=/farm_out/ccocuzza/<analysis>/stf/<directory>/out/<name>.out
#SBATCH --error=/farm_out/ccocuzza/<analysis>/stf/<directory>/err/<name>.err

setenv FITPACK <fitpack>
setenv PYTHONPATH <fitpack>:/work/JAM/apps/lhapdf2/lib/python2.7/site-packages/
setenv PYTHONPATH <wdir>:$PYTHONPATH
<python> <wdir>/analysis/obslib/stf.py <task> -d <directory> -Q2 <Q2> -t <tar> -s <stf>

"""

template=template.replace('<python>',python)
template=template.replace('<fitpack>',fitpack)
template=template.replace('<wdir>',wdir)
template=template.replace('<analysis>',analysis)


def gen_script(task,directory,Q2,tar,stf):
    fname='stf-%s-%s-%s-Q2=%3.5f'%(directory,tar,stf,Q2)
    name ='stf-%s-%s-Q2=%3.5f'%(tar,stf,Q2)
    script=template[:]
    script=script.replace('<name>',fname)
    script=script.replace('<task>',str(task))
    script=script.replace('<directory>',directory)
    script=script.replace('<Q2>',str(Q2))
    script=script.replace('<tar>',tar)
    script=script.replace('<stf>',stf)

    print('Submitting %s'%fname)
    F=open('stf.sbatch','w')
    F.writelines(script)
    F.close()

def pexit(msg):
    print(msg)
    sys.exit()

if __name__=='__main__':


    checkdir('/farm_out/ccocuzza/%s/stf'%analysis)

    ap = argparse.ArgumentParser()

    ap.add_argument('task'               ,type=int                       ,help='0 to generate PSTFs')
    ap.add_argument('-d'  ,'--directory' ,type=str   ,default=None       ,help='directory name to store results')
    ap.add_argument('-Q2' ,'--Q2'        ,type=float ,default=10.00      ,help='Q2 value')
    ap.add_argument('-t'  ,'--test'      ,type=bool  ,default=False      ,help='test flag')
    args = ap.parse_args()

    checkdir('/farm_out/ccocuzza/%s/stf/%s'%(analysis,args.directory))
    checkdir('/farm_out/ccocuzza/%s/stf/%s/out'%(analysis,args.directory))
    checkdir('/farm_out/ccocuzza/%s/stf/%s/err'%(analysis,args.directory))
    
    ###################################################
    ##--Plot structure functions and related quantities
    ###################################################
   
 
    #--generate NC and CC structure functions
    if args.task==0:
        TAR = ['p','n','d','h','t']
        STF = ['F2','FL','F3']

        for tar in TAR:
            for _stf in STF: 
                gen_script(args.task,args.directory,args.Q2,tar=tar,stf=_stf)
                if args.test: os.system('source stf.sbatch')
                else:         os.system('sbatch stf.sbatch')
                time.sleep(0.2)

        STF = ['W2+','WL+','W3+','W2-','WL-','W3-']
        for _stf in STF: 
            gen_script(args.task,args.directory,args.Q2,tar='p',stf=_stf)
            if args.test: os.system('source stf.sbatch')
            else:         os.system('sbatch stf.sbatch')
            time.sleep(0.2)
        
        os.remove('stf.sbatch')

    #--generate F2 at MARATHON kinematics
    if args.task==1:
        TAR = ['p','n','d','h','t']
        for tar in TAR:
            gen_script(args.task,args.directory,0.00,tar=tar,stf='F2')
            if args.test: os.system('source stf.sbatch')
            else:         os.system('sbatch stf.sbatch')
            time.sleep(0.2)
        
        os.remove('stf.sbatch')

    #--generate F2 off-shell structure functions
    if args.task==2:
        TAR = ['d','h','t']
        for tar in TAR:
            gen_script(args.task,args.directory,args.Q2,tar=tar,stf='F2')
            if args.test: os.system('source stf.sbatch')
            else:         os.system('sbatch stf.sbatch')
            time.sleep(0.2)
        
        os.remove('stf.sbatch')

    if args.task==3:
        print('Creating STF plots...')
        stf.plot_stf    (args.directory,Q2=args.Q2,mode=0)
        stf.plot_stf    (args.directory,Q2=args.Q2,mode=1)
        stf.plot_CCstf  (args.directory,Q2=args.Q2,mode=0)
        stf.plot_CCstf  (args.directory,Q2=args.Q2,mode=1)
        stf.plot_off_stf(args.directory,Q2=args.Q2,mode=0)
        stf.plot_off_stf(args.directory,Q2=args.Q2,mode=1)
        stf.plot_F2_rat (args.directory,Q2=args.Q2,mode=0)
        stf.plot_F2_rat (args.directory,Q2=args.Q2,mode=1)
        stf.plot_EMC_rat(args.directory,Q2=args.Q2,mode=0)
        stf.plot_EMC_rat(args.directory,Q2=args.Q2,mode=1)
        ht.plot_ht      (args.directory,Q2=1.27**2,mode=0)   
        ht.plot_ht      (args.directory,Q2=1.27**2,mode=1)   
        off.plot_off    (args.directory,Q2=args.Q2,mode=0)
        off.plot_off    (args.directory,Q2=args.Q2,mode=1)

 







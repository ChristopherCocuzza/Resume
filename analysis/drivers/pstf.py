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
from analysis.obslib import pstf, ht, off

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
#SBATCH --output=/farm_out/ccocuzza/<analysis>/pstf/<directory>/out/<name>.out
#SBATCH --error=/farm_out/ccocuzza/<analysis>/pstf/<directory>/err/<name>.err

setenv FITPACK <fitpack>
setenv PYTHONPATH <fitpack>:/work/JAM/apps/lhapdf2/lib/python2.7/site-packages/
setenv PYTHONPATH <wdir>:$PYTHONPATH
<python> <wdir>/analysis/obslib/pstf.py -d <directory> -Q2 <Q2> -t <tar> -s <stf>

"""

template=template.replace('<python>',python)
template=template.replace('<fitpack>',fitpack)
template=template.replace('<wdir>',wdir)
template=template.replace('<analysis>',analysis)


def gen_script(directory,Q2,tar,stf):
    fname='pstf-%s-%s-%s-Q2=%3.5f'%(directory,tar,stf,Q2)
    name ='pstf-%s-%s-Q2=%3.5f'%(tar,stf,Q2)
    script=template[:]
    script=script.replace('<name>',fname)
    script=script.replace('<directory>',directory)
    script=script.replace('<Q2>',str(Q2))
    script=script.replace('<tar>',tar)
    script=script.replace('<stf>',stf)

    print('Submitting %s'%fname)
    F=open('pstf.sbatch','w')
    F.writelines(script)
    F.close()

def pexit(msg):
    print(msg)
    sys.exit()

if __name__=='__main__':


    checkdir('/farm_out/ccocuzza/%s/pstf'%analysis)

    ap = argparse.ArgumentParser()

    ap.add_argument('task'               ,type=int                       ,help='0 to generate PSTFs')
    ap.add_argument('-d'  ,'--directory' ,type=str   ,default='unamed'   ,help='directory name to store results')
    ap.add_argument('-Q2' ,'--Q2'        ,type=float ,default='unamed'   ,help='Q2 value')
    ap.add_argument('-t'  ,'--test'      ,type=bool  ,default=False      ,help='test flag')
    args = ap.parse_args()

    checkdir('/farm_out/ccocuzza/%s/pstf/%s'%(analysis,args.directory))
    checkdir('/farm_out/ccocuzza/%s/pstf/%s/out'%(analysis,args.directory))
    checkdir('/farm_out/ccocuzza/%s/pstf/%s/err'%(analysis,args.directory))
    
    ###################################################
    ##--Plot polarized structure functions and related quantities
    ###################################################
    
    TAR = ['p','n','d','h']
    STF = ['g1','g2']
    if args.task==0:
        print('Submitting jobs to farm...')
        for tar in TAR:
            for stf in STF:
                gen_script(args.directory,args.Q2,tar,stf)
                if args.test: os.system('source pstf.sbatch')
                else:         os.system('sbatch pstf.sbatch')
                time.sleep(0.2)
    
        os.remove('pstf.sbatch')

    if args.task==1:
        print('Creating plots...')
        pstf.plot_pstf(args.directory,Q2=args.Q2,mode=0)
        pstf.plot_pstf(args.directory,Q2=args.Q2,mode=1)
    
    #ht.plot_pol_ht(PLOT,kc,mode=0)
    #ht.plot_pol_ht(PLOT,kc,mode=1)
    #off.plot_pol_off(PLOT,kc,mode=0)
    #off.plot_pol_off(PLOT,kc,mode=1)







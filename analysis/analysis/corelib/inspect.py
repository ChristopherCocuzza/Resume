#!/usr/bin/env python
import os,sys,shutil
import subprocess
import numpy as np

#--matplotlib
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
#matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
matplotlib.rc('text',usetex=True)
import pylab as py

#--from fitlib
from fitlib.resman import RESMAN

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

def get_msr_inspected(wdir,limit=3):

    #--FILT['par'] is a list of tuples: (dist, par, '>' or '<', limit)
    #--dist and par select parameter to filter
    #--choose greater or lesser than limit
    #--i.e. ('pdf','dv1','<',0) filters out any replica with pdf dv1 less than 0

    #--FILT['exp'] is a list of tuples: (reaction,idx,limit)
    #--reaction and idx select parameter to filter
    #--limit chooses chi2/npts limit to remove replicas
    #--i.e. ('idis','10001',3.0) filters out any replicas with DIS experiment 10001 with chi2 greater than 3

    #--FILT['value'] is a list of tuples: (dist,func,'>' or '<', limit,x,Q2)
    #--dist and func select function to filter
    #--choose x and Q2 values
    #--choose greater or lesser than limit
    #--i.e. ('off','F2n','>',1.0,0.3,10) filters out any replicas with offshell F2n > 1.0 at x = 0.3, Q2 = 10

    print('\nget msr inspected (filtered msr files) using %s\n'%wdir)
    load_config('%s/input.py'%wdir)

    if limit==None: limit = 1000000000

    if 'FILT' in conf: 
        FILT = conf['FILT']
        if 'exp' not in FILT:   FILT['exp']   = []
        if 'par' not in FILT:   FILT['par']   = []
        if 'value' not in FILT: FILT['value'] = []
    else:              FILT={_:[] for _ in ['exp','par','value']}

    if len(FILT['exp']) > 0:
        for filt in FILT['exp']:
            print('Filtering %s %s chi2/npts > %s'%(filt[0],filt[1],filt[2]))
    if len(FILT['par']) > 0:
        for filt in FILT['par']:
            print('Filtering %s %s %s %s'%(filt[0],filt[1],filt[2],filt[3]))
    if len(FILT['value']) > 0:
        for filt in FILT['value']:
            print('Filtering %s %s %s %s at x = %s, Q2 = %s'%(filt[0],filt[1],filt[2],filt[3],filt[4],filt[5]))

    if 'positivity' in FILT:
        print('Filtering positivity violations above %3.5f'%FILT['positivity'])

    replicas=sorted(os.listdir('%s/msr'%wdir))
  
    if len(FILT['value']) > 0 or 'positivity' in FILT:
        resman = RESMAN(parallel=False,datasets=False)
        parman = resman.parman

    #--remove directory if it exists, then recreate it
    try:    
        shutil.rmtree('%s/msr-inspected'%wdir)
        checkdir('%s/msr-inspected'%wdir)
    except: checkdir('%s/msr-inspected'%wdir)

    X=[]
    for i in range(len(replicas)):
        lprint('progress: %d/%d'%(i+1,len(replicas)))
        replica=load('%s/msr/%s'%(wdir,replicas[i]))
        istep=sorted(replica['params'].keys())[-1] #--pick last step
        order = replica['order'][istep]
        params = replica['params'][istep]
        flag = False
        #--filter based on parameters
        for filt in FILT['par']:
            dist, par, _, lim = filt[0],filt[1],filt[2],filt[3]
            for j in range(len(order)):
                if order[j][1] == dist:
                    if order[j][2] == par:
                        if _ == '>':
                            if params[j] > lim: flag = True 
                        if _ == '<':
                            if params[j] < lim: flag = True
        if flag: continue
        if params is None: continue
        if istep not in replica['chi2']: data=replica['chi2']
        else:                            data=replica['chi2'][istep]
        chi2,npts=0,0
        for reaction in data: 
            for idx in  data[reaction]:
                chi2+=data[reaction][idx]['chi2']
                npts+=data[reaction][idx]['npts']
                #--filter based on chi2 for a specific experiment
                for filt in FILT['exp']:
                    _reaction, _idx, _limit = filt[0],filt[1],filt[2]
                    if reaction==_reaction:
                        if idx==_idx:
                            chi2_filt=data[reaction][idx]['chi2']
                            npts_filt=data[reaction][idx]['npts']
                            if chi2_filt/npts_filt > _limit: flag = True
        if flag: continue
        if len(FILT['value']) > 0:
            for filt in FILT['value']:
                flag = filter_values(resman,parman,params,order,filt)
                if flag: break
        if flag: continue
        if 'positivity' in FILT:
            flag = filter_positivity(resman,parman,params,order,FILT['positivity'])
        if flag: continue
        if chi2/npts-1<limit:
            X.append(chi2/npts-1)
            cmd=['cp']
            cmd.append('%s/msr/%s'%(wdir,replicas[i]))
            cmd.append('%s/msr-inspected/%s'%(wdir,replicas[i]))
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    print()
    print('original  num. samples :%d'%len(replicas))
    print('inspected num. samples :%d'%len(X))

    nrows=1
    ncols=1
    #print np.mean(X)
    #--plot labeled residuals
    ax=py.subplot(nrows,ncols,1)
    ax.hist(X,bins=50) 
    if istep < 10: title = r'\textrm{\textbf{step0%s $\chi^2$ distribution}}'%istep
    else:          title = r'\textrm{\textbf{step%s $\chi^2$ distribution}}'%istep
    py.title(title,size=20)
    ax.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=20)
    #py.tight_layout()
    checkdir('%s/gallery'%wdir)
    py.savefig('%s/gallery/chi2-dist-inspect.png'%(wdir))
    py.close()

def filter_values(resman,parman,par,order,filt):

    dist,func,lg,lim,x,q2 = filt

    parman.order = order
    parman.set_new_params(par)
    if dist=='pdf' or 'ppdf':
        X = np.array(x)
        Q2= q2
        DIST = conf[dist]
        flav = func
        if flav=='db-ub':
            value = DIST.get_xF(X,Q2,'db') - DIST.get_xF(X,Q2,'ub')
        elif flav=='db/ub':
            value = DIST.get_xF(X,Q2,'db')/DIST.get_xF(X,Q2,'ub')
        elif flav=='d/u':
            value = DIST.get_xF(X,Q2,'d')/DIST.get_xF(X,Q2,'u')
        else:
            value = DIST.get_xF(X,Q2,flav)

    if dist=='off':
        X = np.array(x)
        Q2= np.array(q2)
        DIST = conf['off']
        tar = func[-1]
        stf = func[:2]
        value = DIST.get_offshell(X,Q2,tar,stf)
    if lg == '>':
        if value > lim: return True
    if lg == '<':
        if value < lim: return True
    return False


def filter_positivity(resman,parman,par,order,filt):

    parman.order = order
    parman.set_new_params(par)

    ppdf = conf['ppdf']

    violations = ppdf.check_positivity()

    res2 = np.sum(violations**2)

    if res2 > filt: return True
    else:           return False



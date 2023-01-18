#!/usr/bin/env python
import sys,os
import numpy as np
import copy
from subprocess import Popen, PIPE, STDOUT
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rc('text',usetex=True)
from matplotlib.ticker import MultipleLocator
import pylab  as py
import lhapdf

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitlib
from fitlib.resman import RESMAN

#--from local
from analysis.corelib import core

cwd = os.getcwd()

#--index conventions:
#--1: down
#--2: up
#--3: strange
#--4: charm
#--5: bottom
#--6: top
#--21: gluon
#--negative values for antiquarks

#--900: F2g  (NC F2 gamma channel)
#--901: FLg  (NC FL gamma channel)
#--902: F2gZ (NC F2 gamma/Z channel)
#--903: FLgZ (NC FL gamma/Z channel)
#--904: F3gZ (NC F3 gamma/Z channel)
#--905: F2Z  (NC F2 Z channel)
#--906: FLZ  (NC FL Z channel)
#--907: F3Z  (NC F3 Z channel)
#--908: F2   (NC F2) 
#--909: FL   (NC FL) 
#--910: F3   (NC F3) 
#--930: W2m  (CC F2^{W-}) 
#--931: WLm  (CC FL^{W-})
#--932: W3m  (CC F3^{W-})
#--940: W2p  (CC F2^{W+})
#--941: WLp  (CC FL^{W+})
#--942: W3p  (CC F3^{W+})

#--mode 0: plot each replica
#--mode 1: plot average and standard deviation of replicas

X1=10**np.linspace(-4,-1)
X2=np.linspace(0.101,0.99)
X=np.append(X1,X2)

def write_lhapdf_conf(wdir):
    script = """
    Verbosity: 1
    Interpolator: logcubic
    Extrapolator: continuation
    ForcePositive: 0
    AlphaS_Type: analytic
    MZ: 91.1876
    MUp: 0.002
    MDown: 0.005
    MStrange: 0.10
    MCharm: 1.29
    MBottom: 4.19
    MTop: 172.9
    Pythia6LambdaV5Compat: true"""
    F=open('%s/data/lhapdf.conf'%wdir,'w')
    F.writelines(script)
    F.close()
    
def rename_tables(self,wdir,dirname,newname):

    print('\nrenaming LHAPDF tables for %s to %s'%(dirname,newname))
    old = '%s/data/%s'%(wdir,dirname)
    new = '%s/data/%s'%(wdir,newname)
    
    checkdir(new)
    F=os.listdir(old)

    cnt=0
    for f in F:
        cnt+=1
        lprint('progress %d/%d'%(cnt,len(F)))
        cmd=['cp','%s/%s'%(old,f),'%s/%s'%(new,f.replace(dirname,newname))]
        p=Popen(cmd,  stdout=PIPE, stderr=STDOUT)
        output = p.stdout.read()
    print



class QCF:

    def __init__(self):
        pass

    def gen_cj_grid(self):
    
        Q2=[1.30000E+00,1.50159E+00,1.75516E+00,2.07810E+00\
                ,2.49495E+00,3.04086E+00,3.76715E+00,4.50000E+00\
                ,4.75000E+00,6.23113E+00,8.37423E+00,1.15549E+01\
                ,1.64076E+01,2.40380E+01,3.64361E+01,5.73145E+01\
                ,9.38707E+01,1.60654E+02,2.88438E+02,5.45587E+02\
                ,1.09231E+03,2.32646E+03,5.30043E+03,1.29956E+04\
                ,3.45140E+04,1.00000E+05]
    
        X=[1.00000E-06,1.28121E-06,1.64152E-06,2.10317E-06\
               ,2.69463E-06,3.45242E-06,4.42329E-06,5.66715E-06\
               ,7.26076E-06,9.30241E-06,1.19180E-05,1.52689E-05\
               ,1.95617E-05,2.50609E-05,3.21053E-05,4.11287E-05\
               ,5.26863E-05,6.74889E-05,8.64459E-05,1.10720E-04\
               ,1.41800E-04,1.81585E-04,2.32503E-04,2.97652E-04\
               ,3.80981E-04,4.87518E-04,6.26039E-04,8.00452E-04\
               ,1.02297E-03,1.30657E-03,1.66759E-03,2.12729E-03\
               ,2.71054E-03,3.44865E-03,4.37927E-03,5.54908E-03\
               ,7.01192E-03,8.83064E-03,1.10763E-02,1.38266E-02\
               ,1.71641E-02,2.11717E-02,2.59364E-02,3.15062E-02\
               ,3.79623E-02,4.53425E-02,5.36750E-02,6.29705E-02\
               ,7.32221E-02,8.44039E-02,9.64793E-02,1.09332E-01\
               ,1.23067E-01,1.37507E-01,1.52639E-01,1.68416E-01\
               ,1.84794E-01,2.01731E-01,2.19016E-01,2.36948E-01\
               ,2.55242E-01,2.73927E-01,2.92954E-01,3.12340E-01\
               ,3.32036E-01,3.52019E-01,3.72282E-01,3.92772E-01\
               ,4.13533E-01,4.34326E-01,4.55495E-01,4.76836E-01\
               ,4.98342E-01,5.20006E-01,5.41818E-01,5.63773E-01\
               ,5.85861E-01,6.08077E-01,6.30459E-01,6.52800E-01\
               ,6.75387E-01,6.98063E-01,7.20830E-01,7.43683E-01\
               ,7.66623E-01,7.89636E-01,8.12791E-01,8.35940E-01\
               ,8.59175E-01,8.82485E-01,9.05866E-01,9.29311E-01\
               ,9.52817E-01,9.76387E-01,1.00000E+00]
    
        return X,Q2
    
    def gen_dss_grid(self):
    
        Q2=[1.3e0, 1.5e0, 2.5e0, 
                 4.0e0, 6.4e0, 1.0e1, 1.5e1, 2.5e1, 4.0e1, 6.4e1,
                 1.0e2, 1.8e2, 3.2e2, 5.8e2, 1.0e3, 1.8e3,
                 3.2e3, 5.8e3, 1.0e4, 1.8e4, 3.2e4, 5.8e4, 1.0e5]
        #X=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        #        0.095, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275,
        #        0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475,  0.5, 
        #        0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7,
        #        0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 
        #        0.925, 0.95, 0.975, 1.0] 
        X=np.linspace(0.01,1,200)
    
        return X,Q2
    
    def gen_grid(self,dist):
        if    dist.startswith('pdf') : return self.gen_cj_grid()
        elif  dist.startswith('ppdf'): return self.gen_cj_grid()
        elif  dist.startswith('ff')  : return self.gen_cj_grid()
    
    def _gen_table(self,dist):
    
        X,Q2=self.gen_grid(dist)
        qpd=conf[dist]
    
        nx=len(X)
        nQ2=len(Q2)
    
        #--fill table
        table={iflav:[]  for iflav in [-5,-4,-3,-2,-1,1,2,3,4,5,21]}  
        npts=nQ2*nx
        for iQ2 in range(nQ2):
            for ix in range(nx):
                table[21].append(qpd.get_xF(X[ix],Q2[iQ2],'g'))
                table[ 1].append(qpd.get_xF(X[ix],Q2[iQ2],'d'))
                table[ 2].append(qpd.get_xF(X[ix],Q2[iQ2],'u'))
                table[ 3].append(qpd.get_xF(X[ix],Q2[iQ2],'s'))
                table[-1].append(qpd.get_xF(X[ix],Q2[iQ2],'db'))
                table[-2].append(qpd.get_xF(X[ix],Q2[iQ2],'ub'))
                table[-3].append(qpd.get_xF(X[ix],Q2[iQ2],'sb'))
                if Q2[iQ2] < conf['aux'].mc2: q2 = conf['aux'].mc2+1 
                else:                         q2 = Q2[iQ2]
                table[ 4].append(qpd.get_xF(X[ix],q2,'c'))
                table[-4].append(qpd.get_xF(X[ix],q2,'cb'))
                if Q2[iQ2] < conf['aux'].mb2: q2 = conf['aux'].mb2+1 
                else:                         q2 = Q2[iQ2]
                table[ 5].append(qpd.get_xF(X[ix],q2,'b'))
                table[-5].append(qpd.get_xF(X[ix],q2,'bb'))

        #--remap tables to match with lhapdf format
        for iflav in [-5,-4,-3,-2,-1,1,2,3,4,5,21]: 
            new_list=[]
            for ix in range(nx):
                for inQ2 in range(nQ2):
                    idx=ix+inQ2*nx
                    new_list.append(table[iflav][idx])
            table[iflav]=new_list

        return X,Q2,table
    
    def gen_lhapdf_dat_file(self,X,Q2,table, wdir, file_name,setlabel):
        lines=[]  
        lines.append('PdfType: replica')
        lines.append('Format: lhagrid1')
        lines.append('---')
        line=''
        for _ in X: line+=('%10.5e '%_).upper()
        lines.append(line)
        line=''
        for _ in Q2: line+=('%10.5e '%_**0.5).upper()
        lines.append(line)
        lines.append('-5 -4 -3 -2 -1 1 2 3 4 5 21')
   
        nx=len(X)
        nQ2=len(Q2)
    
        for i in range(nx*nQ2):
            line=''
            for iflav in [-5,-4,-3,-2,-1,1,2,3,4,5,21]:
                line+=('%10.5e '%table[iflav][i]).upper()
            lines.append(line)
        lines.append('---')
        lines=[l+'\n' for l in lines]
        idx=str(setlabel).zfill(4)
        tab=open('%s/data/%s/%s_%s.dat'%(wdir, file_name, file_name,idx),'w')
        tab.writelines(lines)
        tab.close()
    
    def gen_lhapdf_info_file(self,X,Q2,nrep, wdir, file_name,info):
    
        aS=[conf['alphaS'].get_alphaS(_) for _ in Q2]
        mZ=conf['aux'].mZ
        mb=conf['aux'].mb
        mc=conf['aux'].mc
        alphaSMZ=conf['aux'].alphaSMZ
        xmin=X[0]
        xmax=X[-1]
        Qmin=Q2[0]**0.5
        Qmax=Q2[-1]**0.5
    
    
        lines=[]
        lines.append('SetDesc:         "<description>"')
        lines.append('SetIndex:        <index>')
        lines.append('Authors:         <authors>')
        lines.append('Reference:       reference')
        lines.append('Format:          lhagrid1')
        lines.append('DataVersion:     1')
        lines.append('NumMembers:      %d'%nrep)
        lines.append('Particle:        particle')
        lines.append('Flavors:         [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 21]')
        lines.append('OrderQCD:        1')
        lines.append('FlavorScheme:    variable')
        lines.append('NumFlavors:      5')
        lines.append('ErrorType:       replicas')
        lines.append('XMin:            %0.2e'%xmin)
        lines.append('XMax:            %0.2e'%xmax)
        lines.append('QMin:            %0.2e'%Qmin)
        lines.append('QMax:            %0.2e'%Qmax)
        lines.append('MZ:              %f'%mZ)
        lines.append('MUp:             0.0')
        lines.append('MDown:           0.0')
        lines.append('MStrange:        0.0')
        lines.append('MCharm:          %f'%mc)
        lines.append('MBottom:         %f'%mb)
        lines.append('MTop:            180.0')
        lines.append('AlphaS_MZ:       %f'%alphaSMZ)
        lines.append('AlphaS_OrderQCD: 1')
        lines.append('AlphaS_Type:     ipol')
        line='AlphaS_Qs: ['
        for _ in Q2: line+=('%10.5e, '%_**0.5).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        line='AlphaS_Vals: ['
        for _ in aS: line+=('%10.5e, '%_).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        lines.append('AlphaS_Lambda4: 0')
        lines.append('AlphaS_Lambda5: 0')
    
        for i in range(len(lines)):
            for _ in info:
                lines[i]=lines[i].replace(_,info[_])
    
    
        lines=[l+'\n' for l in lines]
        tab=open('%s/data/%s/%s.info'%(wdir, file_name, file_name),'w')
        tab.writelines(lines)
        tab.close()
    
    def gen_tables(self,wdir,dist, file_name,info,info_only=False):
    
        print('\ngenerating QCF LHAPDF tables for %s using %s'%(dist,wdir))
    
        write_lhapdf_conf(wdir)
        load_config('%s/input.py'%wdir)
        istep=core.get_istep()
        core.mod_conf(istep) #--set conf as specified in istep   
    
        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
    
        jar=load('%s/data/jar-%d.dat'%(wdir,istep))
        replicas=jar['replicas']
    
        #--check order consistency
        order=jar['order']
        parman.order = order
    
        #--create output dir
        checkdir(wdir + '/data/')
        checkdir(wdir + '/data/%s/' % file_name)
    
        #--gen_lhapdf_info_file
        X,Q2=self.gen_grid(dist)
        nrep=len(replicas)
        self.gen_lhapdf_info_file(X,Q2,nrep, wdir, file_name,info)

        #--gen lhapdf_data_files
        TABLE = {}
        for iflav in [-5,-4,-3,-2,-1,1,2,3,4,5,21]:
            TABLE[iflav] = []
        if info_only==False:
            cnt=0
            for par in replicas:
                lprint('progress: %d/%d'%(cnt+1,len(replicas)))
                parman.set_new_params(par)
                X,Q2,table = self._gen_table(dist)
                for iflav in [-5,-4,-3,-2,-1,1,2,3,4,5,21]:
                    TABLE[iflav].append(table[iflav])
                self.gen_lhapdf_dat_file(X,Q2,table, wdir, file_name,cnt+1)
                cnt+=1
            print()
       
        #--save mean value with index 0000 
        for iflav in [-5,-4,-3,-2,-1,1,2,3,4,5,21]:
            TABLE[iflav] = np.mean(TABLE[iflav],axis=0)
        self.gen_lhapdf_dat_file(X,Q2,TABLE, wdir, file_name,0)

        print('Saving LHAPDF table to %s/data/%s'%(wdir,file_name))

        #--load and plot the QCF
        if   dist=='pdf':    self.plot_pdfs  (wdir,file_name,Q2=10.0,mode=0)
        elif dist=='ppdf':   self.plot_ppdfs (wdir,file_name,Q2=10.0,mode=0)
        elif dist=='ffpion': self.plot_ffpion(wdir,file_name,Q2=10.0,mode=0)
        elif dist=='ffkaon': self.plot_ffkaon(wdir,file_name,Q2=10.0,mode=0)

        if   dist=='pdf':    self.plot_pdfs  (wdir,file_name,Q2=10.0,mode=1)
        elif dist=='ppdf':   self.plot_ppdfs (wdir,file_name,Q2=10.0,mode=1)
        elif dist=='ffpion': self.plot_ffpion(wdir,file_name,Q2=10.0,mode=1)
        elif dist=='ffkaon': self.plot_ffkaon(wdir,file_name,Q2=10.0,mode=1)
      
    def plot_pdfs(self,wdir,file_name,Q2,mode=0):
    
        nrows,ncols=3,2
        fig = py.figure(figsize=(ncols*7,nrows*4))
        ax11=py.subplot(nrows,ncols,1)
        ax12=py.subplot(nrows,ncols,2)
        ax21=py.subplot(nrows,ncols,3)
        ax22=py.subplot(nrows,ncols,4)
        ax31=py.subplot(nrows,ncols,5)
        ax32=py.subplot(nrows,ncols,6)
    
        hand = {}
   
        os.environ["LHAPDF_DATA_PATH"] = '/w/jam-sciwork18/ccocuzza/analysis-hx/%s/data'%(wdir)
        PDF = lhapdf.mkPDFs(file_name)
        nrep = len(PDF)
    
        flavs = ['uv','dv','g','db+ub','db-ub','s+sb','Rs']
        data = {flav: [] for flav in flavs} 
    
        for i in range(nrep):
            #--skip mean value
            if i==0: continue
            d =  np.array([PDF[i].xfxQ2( 1,x,Q2) for x in X])
            u =  np.array([PDF[i].xfxQ2( 2,x,Q2) for x in X])
            s =  np.array([PDF[i].xfxQ2( 3,x,Q2) for x in X])
            db = np.array([PDF[i].xfxQ2(-1,x,Q2) for x in X])
            ub = np.array([PDF[i].xfxQ2(-2,x,Q2) for x in X])
            sb = np.array([PDF[i].xfxQ2(-3,x,Q2) for x in X])
            g  = np.array([PDF[i].xfxQ2(21,x,Q2) for x in X])
            data['uv'].append(u-ub)
            data['dv'].append(d-db)
            data['g'] .append(g)
            data['db+ub'].append(db+ub)
            data['db-ub'].append(db-ub)
            data['s+sb'].append(s+sb)
            data['Rs'].append((s+sb)/(db+ub))
            
        for flav in data:
    
            if flav=='uv' or flav=='dv': ax = ax11
            elif flav=='g':              ax = ax12
            elif flav=='db+ub':          ax = ax21
            elif flav=='db-ub':          ax = ax22
            elif flav=='s+sb':           ax = ax31
            elif flav=='Rs':             ax = ax32
    
            mean = np.mean(data[flav],axis=0)
            std  = np.std (data[flav],axis=0)
    
            if mode==0:
                for i in range(nrep-1):
    
                    if flav=='g': data[flav][i] /= 10.0
    
                    ax.plot(X,data[flav][i],color='red',alpha=0.1)
     
            #--plot average and standard deviation
            if mode==1:
                if flav=='g':
                    mean /= 10.0
                    std  /= 10.0
    
                where = [1 for i in range(len(X))]
                if flav=='Rs':
                    where = []
                    for x in X:
                        if x < 0.2: where.append(1)
                        if x > 0.2: where.append(0)
    
                ax.fill_between(X,mean-std,mean+std,color='red',alpha=0.9,where=where)
    
    
        for ax in [ax11,ax12,ax21,ax22,ax31,ax32]:
              ax.set_xlim(1e-2,1)
              ax.semilogx()
                
              ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
              ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
              ax.set_xticks([0.01,0.1,1])
              ax.set_xticklabels([r'$0.01$',r'$0.1$',r'$1$'])
    
        ax11.tick_params(axis='both', which='both', labelbottom=False)
        ax12.tick_params(axis='both', which='both', labelbottom=False)
        ax21.tick_params(axis='both', which='both', labelbottom=False)
        ax22.tick_params(axis='both', which='both', labelbottom=False)
    
        ax11.set_ylim(0,0.7)
        ax12.set_ylim(0,0.5)
        ax21.set_ylim(-0.05,0.7)
        ax22.set_ylim(-0.04,0.08)
        ax31.set_ylim(0,0.7)
        ax32.set_ylim(0,1.2)
    
        ax11.set_yticks([0.2,0.4,0.6])
        ax12.set_yticks([0.2,0.4])
        ax21.set_yticks([0,0.2,0.4,0.6])
        ax22.set_yticks([-0.02,0,0.02,0.04,0.06])
        ax31.set_yticks([0.2,0.4,0.6])
        ax32.set_yticks([0.5,1.0])
    
        minorLocator = MultipleLocator(0.05)
        ax11.yaxis.set_minor_locator(minorLocator)
        ax12.yaxis.set_minor_locator(minorLocator)
        ax21.yaxis.set_minor_locator(minorLocator)
        ax31.yaxis.set_minor_locator(minorLocator)
        minorLocator = MultipleLocator(0.005)
        ax22.yaxis.set_minor_locator(minorLocator)
        minorLocator = MultipleLocator(0.1)
        ax32.yaxis.set_minor_locator(minorLocator)
    
        for ax in [ax31,ax32]:
            ax.set_xlabel(r'\boldmath$x$' ,size=30)
            ax.xaxis.set_label_coords(0.80,0.00)
    
        ax11.text(0.85 ,0.50  ,r'\boldmath{$xu_{v}$}'            , transform=ax11.transAxes,size=30)
        ax11.text(0.60 ,0.20  ,r'\boldmath{$xd_{v}$}'            , transform=ax11.transAxes,size=30)
        ax12.text(0.65 ,0.25  ,r'\boldmath{$xg/10$}'             , transform=ax12.transAxes,size=30)
        ax21.text(0.10 ,0.20  ,r'\boldmath{$x(\bar{d}+\bar{u})$}', transform=ax21.transAxes,size=30)
        ax22.text(0.20 ,0.10  ,r'\boldmath{$x(\bar{d}-\bar{u})$}', transform=ax22.transAxes,size=30)
        ax31.text(0.50 ,0.40  ,r'\boldmath{$x(s+\bar{s})$}',       transform=ax31.transAxes,size=30)
        ax32.text(0.05 ,0.05  ,r'\boldmath{$R_s$}',       transform=ax32.transAxes,size=30)
    
        if Q2 == 1.27**2: ax12.text(0.05,0.08,r'$Q^2 = m_c^2$'                                  , transform=ax12.transAxes,size=30)
        else:             ax12.text(0.05,0.08,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax12.transAxes,size=30)
    
        ax21.axhline(0.0,ls='--',color='black',alpha=0.5)
        ax22.axhline(0.0,ls='--',color='black',alpha=0.5)
        ax32.axvline(0.2,ls='--',color='black',alpha=0.5)
    
        py.tight_layout()
        py.subplots_adjust(hspace = 0, wspace = 0.20)
    
        filename = '%s/gallery/lhapdf-pdfs-Q2=%3.5f'%(wdir,Q2)
        if mode==1: filename += '-bands'
        filename+='.png'
    
        py.savefig(filename)
        py.clf()
        print ('Saving figure to %s'%filename)

    def plot_ppdfs(self,wdir,file_name,Q2,mode=0):
    
        nrows,ncols=3,2
        fig = py.figure(figsize=(ncols*7,nrows*4))
        ax11=py.subplot(nrows,ncols,1)
        ax12=py.subplot(nrows,ncols,2)
        ax21=py.subplot(nrows,ncols,3)
        ax22=py.subplot(nrows,ncols,4)
        ax31=py.subplot(nrows,ncols,5)
        ax32=py.subplot(nrows,ncols,6)
    
        hand = {}
   
        os.environ["LHAPDF_DATA_PATH"] = '/w/jam-sciwork18/ccocuzza/analysis-hx/%s/data'%(wdir)
        PPDF = lhapdf.mkPDFs(file_name)
        nrep = len(PPDF)
    
        flavs = ['up','dp','g','ub','db','sp']
        data = {flav: [] for flav in flavs} 
    
        for i in range(nrep):
            #--skip mean value
            if i==0: continue
            d =  np.array([PPDF[i].xfxQ2( 1,x,Q2) for x in X])
            u =  np.array([PPDF[i].xfxQ2( 2,x,Q2) for x in X])
            s =  np.array([PPDF[i].xfxQ2( 3,x,Q2) for x in X])
            db = np.array([PPDF[i].xfxQ2(-1,x,Q2) for x in X])
            ub = np.array([PPDF[i].xfxQ2(-2,x,Q2) for x in X])
            sb = np.array([PPDF[i].xfxQ2(-3,x,Q2) for x in X])
            g  = np.array([PPDF[i].xfxQ2(21,x,Q2) for x in X])
            data['up'].append(u+ub)
            data['dp'].append(d+db)
            data['g'] .append(g)
            data['ub'].append(ub)
            data['db'].append(db)
            data['sp'].append(s+sb)
            
        for flav in data:
    
            if   flav=='up':  ax = ax11
            elif flav=='dp':  ax = ax12
            elif flav=='g':   ax = ax21
            elif flav=='sp':  ax = ax22
            elif flav=='ub':  ax = ax31
            elif flav=='db':  ax = ax32
    
            mean = np.mean(data[flav],axis=0)
            std  = np.std (data[flav],axis=0)
    
            if mode==0:
                for i in range(nrep-1):
                    ax.plot(X,data[flav][i],color='red',alpha=0.1)
     
            #--plot average and standard deviation
            if mode==1:
                ax.fill_between(X,mean-std,mean+std,color='red',alpha=0.9)
    
    
        for ax in [ax11,ax12,ax21,ax22,ax31,ax32]:
              ax.set_xlim(1e-2,1)
              ax.semilogx()
                
              ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
              ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
              ax.set_xticks([0.01,0.1,1])
              ax.set_xticklabels([r'$0.01$',r'$0.1$',r'$1$'])
    
        ax11.tick_params(axis='both', which='both', labelbottom=False)
        ax12.tick_params(axis='both', which='both', labelbottom=False)
        ax21.tick_params(axis='both', which='both', labelbottom=False)
        ax22.tick_params(axis='both', which='both', labelbottom=False)
    
    
        ax11.set_ylim(-0.10,0.50)   
        ax12.set_ylim(-0.20,0.10)   
        ax21.set_ylim(-0.30,0.40)  
        ax22.set_ylim(-0.07,0.15)
        ax31.set_ylim(-0.07,0.15)
        ax32.set_ylim(-0.07,0.15)

        ax11.set_yticks([0,0.2,0.4])
        ax12.set_yticks([-0.15,-0.10,-0.05,0,0.05])
        ax21.set_yticks([-0.2,0.0,0.2])
        ax22.set_yticks([-0.04,0.00,0.04,0.08,0.12])
        ax31.set_yticks([-0.04,0.00,0.04,0.08,0.12])
        ax32.set_yticks([-0.04,0.00,0.04,0.08,0.12])

        for ax in [ax11,ax12,ax21,ax22,ax31,ax32]:
            ax.axhline(0  ,color='k',linestyle='--',alpha=0.5)
            ax.axvline(0.1,color='k',linestyle=':' ,alpha=0.5)

        ax31.set_xlabel(r'\boldmath$x$',size=30)
        ax32.set_xlabel(r'\boldmath$x$',size=30)   
        ax31.xaxis.set_label_coords(0.95,0.00)
        ax32.xaxis.set_label_coords(0.95,0.00)

        ax11.text(0.10,0.80,r'\boldmath{$x \Delta u^+$}',                        transform=ax11.transAxes,size=30)
        ax12.text(0.10,0.80,r'\boldmath{$x \Delta d^+$}',                        transform=ax12.transAxes,size=30)
        ax21.text(0.10,0.85,r'\boldmath{$x \Delta g$}'  ,                        transform=ax21.transAxes,size=30)
        ax22.text(0.10,0.80,r'\boldmath{$x \Delta s^+$}',                        transform=ax22.transAxes,size=30)
        ax31.text(0.10,0.80,r'\boldmath{$x \Delta \bar{u}$}',                    transform=ax31.transAxes,size=30)
        ax32.text(0.10,0.80,r'\boldmath{$x \Delta \bar{d}$}',                    transform=ax32.transAxes,size=30)

        if Q2 == 1.27**2: ax11.text(0.05,0.60,r'$Q^2 = m_c^2$',                            transform=ax11.transAxes,size=20)
        else:             ax11.text(0.05,0.60,r'$Q^2 = %s$~'%Q2 + r'\textrm{GeV}'+r'$^2$', transform=ax11.transAxes,size=20)

        py.tight_layout()
        py.subplots_adjust(hspace = 0, wspace = 0.20)
    
        filename = '%s/gallery/lhapdf-ppdfs-Q2=%3.5f'%(wdir,Q2)
        if mode==1: filename += '-bands'
        filename+='.png'
    
        py.savefig(filename)
        py.clf()
        print ('Saving figure to %s'%filename)

    def plot_ffpion(self,wdir,file_name,Q2,mode=0):
    
        nrows,ncols=1,2
        fig = py.figure(figsize=(ncols*7,nrows*5))
        ax11=py.subplot(nrows,ncols,1)
        ax12=py.subplot(nrows,ncols,2)
    
        hand = {}
   
        os.environ["LHAPDF_DATA_PATH"] = '/w/jam-sciwork18/ccocuzza/analysis-hx/%s/data'%(wdir)
        FFPION = lhapdf.mkPDFs(file_name)
        nrep = len(FFPION)
    
        flavs = ['up','u','ub','sp','g','c','b']
        data = {flav: [] for flav in flavs} 
    
        for i in range(nrep):
            #--skip mean value
            if i==0: continue
            d =  np.array([FFPION[i].xfxQ2( 1,x,Q2) for x in X])
            u =  np.array([FFPION[i].xfxQ2( 2,x,Q2) for x in X])
            s =  np.array([FFPION[i].xfxQ2( 3,x,Q2) for x in X])
            c  = np.array([FFPION[i].xfxQ2( 4,x,Q2) for x in X])
            b  = np.array([FFPION[i].xfxQ2( 5,x,Q2) for x in X])
            db = np.array([FFPION[i].xfxQ2(-1,x,Q2) for x in X])
            ub = np.array([FFPION[i].xfxQ2(-2,x,Q2) for x in X])
            sb = np.array([FFPION[i].xfxQ2(-3,x,Q2) for x in X])
            g  = np.array([FFPION[i].xfxQ2(21,x,Q2) for x in X])
            data['up'].append(u+ub)
            data['u'] .append(u)
            data['ub'].append(ub)
            data['sp'].append(s+sb)
            data['g'] .append(g)
            data['c'] .append(c)
            data['b'] .append(b)

        for flav in data:
    
            if flav=='up': ax,color = ax11,'blue'
            if flav=='ub': ax,color = ax11,'green'
            if flav=='u':  ax,color = ax11,'red'
            if flav=='sp': ax,color = ax11,'magenta'
            if flav=='g':  ax,color = ax12,'blue'
            if flav=='c':  ax,color = ax12,'green'
            if flav=='b':  ax,color = ax12,'red'
    
            mean = np.mean(data[flav],axis=0)
            std  = np.std (data[flav],axis=0)
    
            if mode==0:
                for i in range(nrep-1):
                    hand[flav] ,= ax.plot(X,data[flav][i],color=color,alpha=0.1)
     
            #--plot average and standard deviation
            if mode==1:
                hand[flav] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.9)
    
        for ax in [ax11,ax12]:
              ax.set_xlabel(r'\boldmath$z$'    ,size=40)
              ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
              ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
              ax.xaxis.set_label_coords(0.98,0.00)
              ax.set_xlim(0.2,0.90)
              ax.set_xticks([0.2,0.4,0.6,0.8])
              ax.set_ylim(0,0.8)
              ax.set_yticks([0.2,0.4,0.6,0.8])
              minorLocator = MultipleLocator(0.05)
              ax.xaxis.set_minor_locator(minorLocator)
              minorLocator = MultipleLocator(0.05)
              ax.yaxis.set_minor_locator(minorLocator)


        ax12.text(0.65 ,0.75  ,r'\boldmath$z D^{\pi^+}_q$'   ,transform=ax12.transAxes,size=50)

        if Q2 == 1.27**2: ax12.text(0.55,0.60,r'$Q^2 = m_c^2$'                                  , transform=ax12.transAxes,size=30)
        else:             ax12.text(0.55,0.60,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax12.transAxes,size=30)


        handles,labels = [],[]
        handles.append(hand['up'])
        handles.append(hand['u'])
        handles.append(hand['ub'])
        handles.append(hand['sp'])
        labels.append(r'\boldmath$u^+$')
        labels.append(r'\boldmath$u$')
        labels.append(r'\boldmath$\bar{u}$')
        labels.append(r'\boldmath$s^+$')
        ax11.legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

        handles,labels = [],[]
        handles.append(hand['g'])
        handles.append(hand['c'])
        handles.append(hand['b'])
        labels.append(r'\boldmath$g$')
        labels.append(r'\boldmath$c$')
        labels.append(r'\boldmath$b$')
        ax12.legend(handles,labels,loc='lower right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

        py.tight_layout()

        filename = '%s/gallery/lhapdf-ffpion-Q2=%3.5f'%(wdir,Q2)
        if mode==1: filename += '-bands'
        filename+='.png'
    
        py.savefig(filename)
        py.clf()
        print ('Saving figure to %s'%filename)

    def plot_ffkaon(self,wdir,file_name,Q2,mode=0):
    
        nrows,ncols=1,2
        fig = py.figure(figsize=(ncols*7,nrows*5))
        ax11=py.subplot(nrows,ncols,1)
        ax12=py.subplot(nrows,ncols,2)
    
        hand = {}
   
        os.environ["LHAPDF_DATA_PATH"] = '/w/jam-sciwork18/ccocuzza/analysis-hx/%s/data'%(wdir)
        FFKAON = lhapdf.mkPDFs(file_name)
        nrep = len(FFKAON)
    
        flavs = ['sp','s','up','dp','g','c','b']
        data = {flav: [] for flav in flavs} 
    
        for i in range(nrep):
            #--skip mean value
            if i==0: continue
            d =  np.array([FFKAON[i].xfxQ2( 1,x,Q2) for x in X])
            u =  np.array([FFKAON[i].xfxQ2( 2,x,Q2) for x in X])
            s =  np.array([FFKAON[i].xfxQ2( 3,x,Q2) for x in X])
            c  = np.array([FFKAON[i].xfxQ2( 4,x,Q2) for x in X])
            b  = np.array([FFKAON[i].xfxQ2( 5,x,Q2) for x in X])
            db = np.array([FFKAON[i].xfxQ2(-1,x,Q2) for x in X])
            ub = np.array([FFKAON[i].xfxQ2(-2,x,Q2) for x in X])
            sb = np.array([FFKAON[i].xfxQ2(-3,x,Q2) for x in X])
            g  = np.array([FFKAON[i].xfxQ2(21,x,Q2) for x in X])
            data['sp'].append(s+sb)
            data['up'].append(u+ub)
            data['dp'].append(d+db)
            data['s'] .append(s)
            data['g'] .append(g)
            data['c'] .append(c)
            data['b'] .append(b)
            
        for flav in data:
    
            if flav=='sp': ax,color = ax11,'blue'
            if flav=='s':  ax,color = ax11,'green'
            if flav=='up': ax,color = ax11,'red'
            if flav=='dp': ax,color = ax11,'magenta'
            if flav=='g':  ax,color = ax12,'blue'
            if flav=='c':  ax,color = ax12,'green'
            if flav=='b':  ax,color = ax12,'red'

            mean = np.mean(data[flav],axis=0)
            std  = np.std (data[flav],axis=0)
    
            if mode==0:
                for i in range(nrep-1):
                    hand[flav] ,= ax.plot(X,data[flav][i],color=color,alpha=0.1)
     
            #--plot average and standard deviation
            if mode==1:
                hand[flav] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.9)
    
        for ax in [ax11,ax12]:
              ax.set_xlabel(r'\boldmath$z$'    ,size=40)
              ax.tick_params(axis='both', which='major', top=True, right=True, direction='in',labelsize=30,length=10)
              ax.tick_params(axis='both', which='minor', top=True, right=True, direction='in',labelsize=30,length=5)
              ax.xaxis.set_label_coords(0.98,0.00)
              ax.set_xlim(0.2,0.90)
              ax.set_xticks([0.2,0.4,0.6,0.8])
              ax.set_ylim(0,0.4)
              ax.set_yticks([0.1,0.2,0.3,0.4])
              minorLocator = MultipleLocator(0.05)
              ax.xaxis.set_minor_locator(minorLocator)
              minorLocator = MultipleLocator(0.05)
              ax.yaxis.set_minor_locator(minorLocator)


        ax12.text(0.65 ,0.75  ,r'\boldmath$z D^{K^+}_q$'   ,transform=ax12.transAxes,size=50)

        if Q2 == 1.27**2: ax12.text(0.55,0.60,r'$Q^2 = m_c^2$'                                  , transform=ax12.transAxes,size=30)
        else:             ax12.text(0.55,0.60,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax12.transAxes,size=30)


        handles,labels = [],[]
        handles.append(hand['sp'])
        handles.append(hand['s'])
        handles.append(hand['up'])
        handles.append(hand['dp'])
        labels.append(r'\boldmath$s^+$')
        labels.append(r'\boldmath$s$')
        labels.append(r'\boldmath$u^+$')
        labels.append(r'\boldmath$d^+$')
        ax11.legend(handles,labels,loc='upper right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)

        handles,labels = [],[]
        handles.append(hand['g'])
        handles.append(hand['c'])
        handles.append(hand['b'])
        labels.append(r'\boldmath$g$')
        labels.append(r'\boldmath$c$')
        labels.append(r'\boldmath$b$')
        ax12.legend(handles,labels,loc='lower right', fontsize = 28, frameon = 0, handletextpad = 0.3, handlelength = 1.0)



        py.tight_layout()

        filename = '%s/gallery/lhapdf-ffkaon-Q2=%3.5f'%(wdir,Q2)
        if mode==1: filename += '-bands'

        filename+='.png'
        checkdir('%s/gallery'%wdir)
        py.savefig(filename)
        py.clf()
        print ('Saving figure to %s'%filename)


#--need to update so that mem=0000 is the average 
class STF:

    def __init__(self):

        pass

    def gen_cj_grid(self):
    
        Q2=[1.30000E+00,1.50159E+00,1.75516E+00,2.07810E+00\
                ,2.49495E+00,3.04086E+00,3.76715E+00,4.50000E+00\
                ,4.75000E+00,6.23113E+00,8.37423E+00,1.15549E+01\
                ,1.64076E+01,2.40380E+01,3.64361E+01,5.73145E+01\
                ,9.38707E+01,1.60654E+02,2.88438E+02,5.45587E+02\
                ,1.09231E+03,2.32646E+03,5.30043E+03,1.29956E+04\
                ,3.45140E+04,1.00000E+05]
    
        X=[1.00000E-06,1.28121E-06,1.64152E-06,2.10317E-06\
               ,2.69463E-06,3.45242E-06,4.42329E-06,5.66715E-06\
               ,7.26076E-06,9.30241E-06,1.19180E-05,1.52689E-05\
               ,1.95617E-05,2.50609E-05,3.21053E-05,4.11287E-05\
               ,5.26863E-05,6.74889E-05,8.64459E-05,1.10720E-04\
               ,1.41800E-04,1.81585E-04,2.32503E-04,2.97652E-04\
               ,3.80981E-04,4.87518E-04,6.26039E-04,8.00452E-04\
               ,1.02297E-03,1.30657E-03,1.66759E-03,2.12729E-03\
               ,2.71054E-03,3.44865E-03,4.37927E-03,5.54908E-03\
               ,7.01192E-03,8.83064E-03,1.10763E-02,1.38266E-02\
               ,1.71641E-02,2.11717E-02,2.59364E-02,3.15062E-02\
               ,3.79623E-02,4.53425E-02,5.36750E-02,6.29705E-02\
               ,7.32221E-02,8.44039E-02,9.64793E-02,1.09332E-01\
               ,1.23067E-01,1.37507E-01,1.52639E-01,1.68416E-01\
               ,1.84794E-01,2.01731E-01,2.19016E-01,2.36948E-01\
               ,2.55242E-01,2.73927E-01,2.92954E-01,3.12340E-01\
               ,3.32036E-01,3.52019E-01,3.72282E-01,3.92772E-01\
               ,4.13533E-01,4.34326E-01,4.55495E-01,4.76836E-01\
               ,4.98342E-01,5.20006E-01,5.41818E-01,5.63773E-01\
               ,5.85861E-01,6.08077E-01,6.30459E-01,6.52800E-01\
               ,6.75387E-01,6.98063E-01,7.20830E-01,7.43683E-01\
               ,7.66623E-01,7.89636E-01,8.12791E-01,8.35940E-01\
               ,8.59175E-01,8.82485E-01,9.05866E-01,9.29311E-01\
               ,9.52817E-01,9.76387E-01,1.00000E+00]
    
        return X,Q2
   
    def _gen_table(self,tar):
    
        x,q2=self.gen_cj_grid()

        X,Q2 = [],[]
        for i in range(len(q2)):
            for j in range(len(x)):
                X.append(x[j])
                Q2.append(q2[i])

        idxs = self.idxs
        stfs = self.stfs
   
        STF = conf['idis']
 
        nx=len(x)
        nQ2=len(q2)
 
        #--fill table
        #table={iflav:[]  for iflav in idxs}
        table={}
        npts=nQ2*nx

        for i in range(len(idxs)):
            table[idxs[i]] = STF.get_stf(X,Q2,stfs[i],tar=tar) 

        #--remap tables to match with lhapdf format
        for iflav in idxs: 
            new_list=[]
            for ix in range(nx):
                for inQ2 in range(nQ2):
                    idx=ix+inQ2*nx
                    new_list.append(table[iflav][idx])
            table[iflav]=new_list

        return x,q2,table

    def gen_lhapdf_info_file(self,X,Q2,nrep,wdir,file_name,info):
      
        idxs = self.idxs

        #--get flav string
        flavs=''
        for idx in idxs: flavs+='%d, '%idx
        flavs=flavs.rstrip()
        flavs=flavs.rstrip(',')
   
        #--kinematic limits
        xmin=X[0]
        xmax=X[-1]
        Qmin=Q2[0]**0.5
        Qmax=Q2[-1]**0.5
    
        #--qcd params
        load_config('%s/input.py'%wdir)
        RESMAN(nworkers=1,parallel=False,datasets=False)
        aS=[conf['alphaS'].get_alphaS(_) for _ in Q2]
        mZ=conf['aux'].mZ
        mb=conf['aux'].mb
        mc=conf['aux'].mc
        alphaSMZ=conf['aux'].alphaSMZ
    
        #--begin lhapdf info file
        lines=[]
        lines.append('SetDesc:         "<description>"')
        lines.append('SetIndex:        <index>')
        lines.append('Authors:         <authors>')
        lines.append('Reference:       reference')
        lines.append('Format:          lhagrid1')
        lines.append('DataVersion:     1')
        lines.append('NumMembers:      %d'%nrep)
        lines.append('Particle:        particle')
        lines.append('Flavors:         [%s]'%flavs)
        lines.append('OrderQCD:        1')
        lines.append('FlavorScheme:    variable')
        lines.append('NumFlavors:      %d'%len(idxs))
        lines.append('ErrorType:       replicas')
        lines.append('XMin:            %0.2e'%xmin)
        lines.append('XMax:            %0.2e'%xmax)
        lines.append('QMin:            %0.2e'%Qmin)
        lines.append('QMax:            %0.2e'%Qmax)
        lines.append('MZ:              %f'%mZ)
        lines.append('MUp:             0.0')
        lines.append('MDown:           0.0')
        lines.append('MStrange:        0.0')
        lines.append('MCharm:          %f'%mc)
        lines.append('MBottom:         %f'%mb)
        lines.append('MTop:            180.0')
        lines.append('AlphaS_MZ:       %f'%alphaSMZ)
        lines.append('AlphaS_OrderQCD: 1')
        lines.append('AlphaS_Type:     ipol')
        line='AlphaS_Qs: ['
        for _ in Q2: line+=('%10.5e, '%_**0.5).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        line='AlphaS_Vals: ['
        for _ in aS: line+=('%10.5e, '%_).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        lines.append('AlphaS_Lambda4: 0')
        lines.append('AlphaS_Lambda5: 0')
    
        for i in range(len(lines)):
            for _ in info:
                lines[i]=lines[i].replace(_,info[_])
    
        lines=[l+'\n' for l in lines]
        tab=open('%s/data/%s/%s.info'%(wdir, file_name, file_name),'w')
        tab.writelines(lines)
        tab.close()
    
    def gen_lhapdf_dat_file(self,X,Q2,table, wdir, file_name,setlabel):

        idxs = self.idxs
        #--get flav string
        flavs=''
        for idx in idxs: flavs+='%d '%idx

        lines=[]  
        lines.append('PdfType: replica')
        lines.append('Format: lhagrid1')
        lines.append('---')
        line=''
        for _ in X: line+=('%10.5e '%_).upper()
        lines.append(line)
        line=''
        for _ in Q2: line+=('%10.5e '%_**0.5).upper()
        lines.append(line)
       
        lines.append(flavs)
    
        nx=len(X)
        nQ2=len(Q2)
   
        for i in range(nx*nQ2):
            line=''
            for iflav in idxs:
                line+=('%10.5e '%table[iflav][i]).upper()
            lines.append(line)
        lines.append('---')
        lines=[l+'\n' for l in lines]
        idx=str(setlabel).zfill(4)
        tab=open('%s/data/%s/%s_%s.dat'%(wdir, file_name, file_name,idx),'w')
        tab.writelines(lines)
        tab.close()
    
    def gen_tables(self,wdir,tar,file_name,info,info_only=False,channels=True):
    
        print('\ngenerating STF LHAPDF tables for %s using %s'%(tar,wdir))
    
        #--get unpolarized structure functions and appropriate indices
        #--combined channels
        stfs = ['F2','FL','F3']
        idxs = [908,909,910]

        if channels:
            #--gamma channel
            stfs.extend(['F2g','FLg'])
            idxs.extend([900,901])

            #--gamma/Z channel
            stfs.extend(['F2gZ','FLgZ','F3gZ'])
            idxs.extend([902,903,904])
            
            #--Z channel
            stfs.extend(['F2Z','FLZ','F3Z'])
            idxs.extend([905,906,907])
        
        #--charged current only for proton 
        if tar=='p':  
            #--W-
            stfs.extend(['W2-','WL-','W3-'])
            idxs.extend([930,931,932])
            
            #--W+
            stfs.extend(['W2+','WL+','W3+'])
            idxs.extend([940,941,942])

        self.stfs = stfs
        self.idxs = idxs

        write_lhapdf_conf(wdir)
        load_config('%s/input.py'%wdir)
        istep=core.get_istep()
        core.mod_conf(istep) #--set conf as specified in istep   
        conf['idis grid'] = 'prediction'

        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
        conf['datasets']['idis'] = {_: {} for _ in ['xlsx','norm']}
        resman.setup_idis()
        STF = conf['idis']
        for stf in stfs:
            STF.data[tar][stf] = np.zeros(STF.X.size)
            if tar in ['d','h','t']:
                STF.data['p'][stf] = np.zeros(STF.X.size)
                STF.data['n'][stf] = np.zeros(STF.X.size)
        if 'off pdf' in conf:
            STF.data['p']['F2off d'] = np.zeros(STF.X.size)
            STF.data['n']['F2off d'] = np.zeros(STF.X.size)
            STF.data['p']['F2off h'] = np.zeros(STF.X.size)
            STF.data['n']['F2off h'] = np.zeros(STF.X.size)
            STF.data['p']['F2off t'] = np.zeros(STF.X.size)
            STF.data['n']['F2off t'] = np.zeros(STF.X.size)
    
        jar=load('%s/data/jar-%d.dat'%(wdir,istep))
        replicas=jar['replicas']
    
        #--check order consistency
        order=jar['order']
        parman.order = order
    
        #--create output dir
        checkdir(wdir + '/data/')
        checkdir(wdir + '/data/%s/' % file_name)
    
        #--gen_lhapdf_info_file
        X,Q2=self.gen_cj_grid()
        nrep=len(replicas)
        self.gen_lhapdf_info_file(X,Q2,nrep, wdir, file_name,info)

        #--gen lhapdf_data_files
        if info_only==False:
            cnt=0
            for par in replicas:
                lprint('progress: %d/%d'%(cnt+1,len(replicas)))
                parman.set_new_params(par)
                STF._update()
                X,Q2,table = self._gen_table(tar)
                self.gen_lhapdf_dat_file(X,Q2,table, wdir, file_name,cnt)
                cnt+=1
            print

        #--load and plot the STFs
        self.plot_stf(wdir,file_name,tar,Q2=10.0,mode=0)
        self.plot_stf(wdir,file_name,tar,Q2=10.0,mode=1)

    def plot_stf(self,wdir,file_name,tar,Q2,mode=0):
    
        nrows,ncols=1,3
        fig = py.figure(figsize=(ncols*7,nrows*4))
        ax11=py.subplot(nrows,ncols,1)
        ax12=py.subplot(nrows,ncols,2)
        ax13=py.subplot(nrows,ncols,3)
    
        hand = {}
    
    
        stfs = ['F2','FL','F3','F2g','FLg','F2gZ','FLgZ','F3gZ','F2Z','FLZ','F3Z']
        data = {}
        data[tar] = {stf: [] for stf in stfs} 
    
        os.environ["LHAPDF_DATA_PATH"] = '/w/jam-sciwork18/ccocuzza/analysis-hx/%s/data'%(wdir)
        STF = lhapdf.mkPDFs(file_name)
        nrep = len(STF)
        for i in range(nrep):
            F2 =  np.array([STF[i].xfxQ2(908,x,Q2)*x for x in X])
            FL =  np.array([STF[i].xfxQ2(909,x,Q2)*x for x in X])
            F3 =  np.array([STF[i].xfxQ2(910,x,Q2)*x for x in X])
            data[tar]['F2'].append(F2)
            data[tar]['FL'].append(FL)
            data[tar]['F3'].append(F3)


        #print('deuteron')
        #X,Q2 = 0.35,5.5
        #for i in range(nrep):
        #    F2g  =  STF[i].xfxQ2(900,X,Q2)
        #    FLg  =  STF[i].xfxQ2(901,X,Q2)
        #    F2gZ =  STF[i].xfxQ2(902,X,Q2)
        #    FLgZ =  STF[i].xfxQ2(903,X,Q2)
        #    F3gZ =  STF[i].xfxQ2(904,X,Q2)
        #    F2Z  =  STF[i].xfxQ2(905,X,Q2)
        #    FLZ  =  STF[i].xfxQ2(906,X,Q2)
        #    F3Z  =  STF[i].xfxQ2(907,X,Q2)
        #    F2   =  STF[i].xfxQ2(908,X,Q2)
        #    FL   =  STF[i].xfxQ2(909,X,Q2)
        #    F3   =  STF[i].xfxQ2(910,X,Q2)
        #    data[tar]['F2'].append(F2)
        #    data[tar]['FL'].append(FL)
        #    data[tar]['F3'].append(F3)
        #    data[tar]['F2g'].append(F2g)
        #    data[tar]['FLg'].append(FLg)
        #    data[tar]['F2gZ'].append(F2gZ)
        #    data[tar]['FLgZ'].append(FLgZ)
        #    data[tar]['F3gZ'].append(F3gZ)
        #    data[tar]['F2Z'].append(F2Z)
        #    data[tar]['FLZ'].append(FLZ)
        #    data[tar]['F3Z'].append(F3Z)

        #print('X:',X,'Q2:',Q2,'F2:', np.mean(data[tar]['F2']))
        #print('X:',X,'Q2:',Q2,'FL:', np.mean(data[tar]['FL']))
        #print('X:',X,'Q2:',Q2,'F3:', np.mean(data[tar]['F3']))
        #print('X:',X,'Q2:',Q2,'F2g:', np.mean(data[tar]['F2g']))
        #print('X:',X,'Q2:',Q2,'FLg:', np.mean(data[tar]['FLg']))
        #print('X:',X,'Q2:',Q2,'F2gZ:', np.mean(data[tar]['F2gZ']))
        #print('X:',X,'Q2:',Q2,'FLgZ:', np.mean(data[tar]['FLgZ']))
        #print('X:',X,'Q2:',Q2,'F3gZ:', np.mean(data[tar]['F3gZ']))
        #print('X:',X,'Q2:',Q2,'F2Z:', np.mean(data[tar]['F2Z']))
        #print('X:',X,'Q2:',Q2,'FLZ:', np.mean(data[tar]['FLZ']))
        #print('X:',X,'Q2:',Q2,'F3Z:', np.mean(data[tar]['F3Z']))

        #stfs = ['F2','FL','F3','F2g','FLg','F2gZ','FLgZ','F3gZ','F2Z','FLZ','F3Z']
        #data = {}
        #data[tar] = {stf: [] for stf in stfs} 
        #X,Q2 = 0.73,10.8
        #for i in range(nrep):
        #    F2g  =  STF[i].xfxQ2(900,X,Q2)
        #    FLg  =  STF[i].xfxQ2(901,X,Q2)
        #    F2gZ =  STF[i].xfxQ2(902,X,Q2)
        #    FLgZ =  STF[i].xfxQ2(903,X,Q2)
        #    F3gZ =  STF[i].xfxQ2(904,X,Q2)
        #    F2Z  =  STF[i].xfxQ2(905,X,Q2)
        #    FLZ  =  STF[i].xfxQ2(906,X,Q2)
        #    F3Z  =  STF[i].xfxQ2(907,X,Q2)
        #    F2   =  STF[i].xfxQ2(908,X,Q2)
        #    FL   =  STF[i].xfxQ2(909,X,Q2)
        #    F3   =  STF[i].xfxQ2(910,X,Q2)
        #    data[tar]['F2'].append(F2)
        #    data[tar]['FL'].append(FL)
        #    data[tar]['F3'].append(F3)
        #    data[tar]['F2g'].append(F2g)
        #    data[tar]['FLg'].append(FLg)
        #    data[tar]['F2gZ'].append(F2gZ)
        #    data[tar]['FLgZ'].append(FLgZ)
        #    data[tar]['F3gZ'].append(F3gZ)
        #    data[tar]['F2Z'].append(F2Z)
        #    data[tar]['FLZ'].append(FLZ)
        #    data[tar]['F3Z'].append(F3Z)

        #print('X:',X,'Q2:',Q2,'F2:', np.mean(data[tar]['F2']))
        #print('X:',X,'Q2:',Q2,'FL:', np.mean(data[tar]['FL']))
        #print('X:',X,'Q2:',Q2,'F3:', np.mean(data[tar]['F3']))
        #print('X:',X,'Q2:',Q2,'F2g:', np.mean(data[tar]['F2g']))
        #print('X:',X,'Q2:',Q2,'FLg:', np.mean(data[tar]['FLg']))
        #print('X:',X,'Q2:',Q2,'F2gZ:', np.mean(data[tar]['F2gZ']))
        #print('X:',X,'Q2:',Q2,'FLgZ:', np.mean(data[tar]['FLgZ']))
        #print('X:',X,'Q2:',Q2,'F3gZ:', np.mean(data[tar]['F3gZ']))
        #print('X:',X,'Q2:',Q2,'F2Z:', np.mean(data[tar]['F2Z']))
        #print('X:',X,'Q2:',Q2,'FLZ:', np.mean(data[tar]['FLZ']))
        #print('X:',X,'Q2:',Q2,'F3Z:', np.mean(data[tar]['F3Z']))

        #sys.exit()
 
        for stf in data[tar]:
            mean = np.mean(data[tar][stf],axis=0)
            std  = np.std (data[tar][stf],axis=0)
    
            if tar=='p':   color='red'
            elif tar=='n': color='green'
            elif tar=='d': color='blue'
            elif tar=='h': color='magenta'
            elif tar=='t': color='orange'
    
            if stf =='F2':   ax = ax11
            elif stf =='FL': ax = ax12
            elif stf =='F3': ax = ax13
    
            #--plot each replica
            if mode==0:
                for i in range(nrep):
                    hand[tar] ,= ax.plot(X,data[tar][stf][i],color=color,alpha=0.1)
        
            #--plot average and standard deviation
            if mode==1:
                hand[tar] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.5)
    
    
        for ax in [ax11,ax12,ax13]:
              ax.set_xlim(0,0.9)
                
              ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)


        if tar in ['p','n','d']:
            ax11.text(0.70,0.70,r'\boldmath$xF_2^{%s}$'%tar,transform=ax11.transAxes,size=40)
            ax12.text(0.70,0.70,r'\boldmath$xF_L^{%s}$'%tar,transform=ax12.transAxes,size=40)
            ax13.text(0.70,0.70,r'\boldmath$xF_3^{%s}$'%tar,transform=ax13.transAxes,size=40)
        elif tar=='h':
            ax11.text(0.70,0.70,r'\boldmath$xF_2^{^3{\rm He}}$',transform=ax11.transAxes,size=40)
            ax12.text(0.70,0.70,r'\boldmath$xF_L^{^3{\rm He}}$',transform=ax12.transAxes,size=40)
            ax13.text(0.70,0.70,r'\boldmath$xF_3^{^3{\rm He}}$',transform=ax13.transAxes,size=40)
        elif tar=='t':
            ax11.text(0.70,0.70,r'\boldmath$xF_2^{^3{\rm H}}$',transform=ax11.transAxes,size=40)
            ax12.text(0.70,0.70,r'\boldmath$xF_L^{^3{\rm H}}$',transform=ax12.transAxes,size=40)
            ax13.text(0.70,0.70,r'\boldmath$xF_3^{^3{\rm H}}$',transform=ax13.transAxes,size=40)

        ax11.set_ylim(0,0.1)      
        ax12.set_ylim(0,0.004)    
        ax13.set_ylim(0,0.00045)  

        for ax in [ax11,ax12,ax13]:
            minorLocator = MultipleLocator(0.02)
            majorLocator = MultipleLocator(0.2)
            ax.xaxis.set_minor_locator(minorLocator)
            ax.xaxis.set_major_locator(majorLocator)
            ax.xaxis.set_tick_params(which='major',length=6)
            ax.xaxis.set_tick_params(which='minor',length=3)
            ax.yaxis.set_tick_params(which='major',length=6)
            ax.yaxis.set_tick_params(which='minor',length=3)
            ax.set_xticks([0.2,0.4,0.6,0.8])
            ax.set_xlabel(r'\boldmath$x$' ,size=35)   
            ax.xaxis.set_label_coords(0.97,0)


        if Q2 == 1.27**2: ax11.text(0.10,0.05,r'$Q^2 = m_c^2$',             transform=ax11.transAxes,size=30)
        else:             ax11.text(0.10,0.05,r'$Q^2 = %s~{\rm GeV}^2$'%Q2, transform=ax11.transAxes,size=25)

        py.tight_layout()

        filename = '%s/gallery/lhapdf-stfs-%s-Q2=%3.5f'%(wdir,tar,Q2)
        if mode==1: filename += '-bands'

        filename+='.png'

        checkdir('%s/gallery'%wdir)
        py.savefig(filename)
        print ('Saving figure to %s'%filename)
        py.clf()


#--need to update so that mem=0000 is the average 
class HT:

    def __init__(self):

        pass

    def gen_cj_grid(self):
    
        Q2=[1.30000E+00,1.50159E+00,1.75516E+00,2.07810E+00\
                ,2.49495E+00,3.04086E+00,3.76715E+00,4.50000E+00\
                ,4.75000E+00,6.23113E+00,8.37423E+00,1.15549E+01\
                ,1.64076E+01,2.40380E+01,3.64361E+01,5.73145E+01\
                ,9.38707E+01,1.60654E+02,2.88438E+02,5.45587E+02\
                ,1.09231E+03,2.32646E+03,5.30043E+03,1.29956E+04\
                ,3.45140E+04,1.00000E+05]
    
        X=[1.00000E-06,1.28121E-06,1.64152E-06,2.10317E-06\
               ,2.69463E-06,3.45242E-06,4.42329E-06,5.66715E-06\
               ,7.26076E-06,9.30241E-06,1.19180E-05,1.52689E-05\
               ,1.95617E-05,2.50609E-05,3.21053E-05,4.11287E-05\
               ,5.26863E-05,6.74889E-05,8.64459E-05,1.10720E-04\
               ,1.41800E-04,1.81585E-04,2.32503E-04,2.97652E-04\
               ,3.80981E-04,4.87518E-04,6.26039E-04,8.00452E-04\
               ,1.02297E-03,1.30657E-03,1.66759E-03,2.12729E-03\
               ,2.71054E-03,3.44865E-03,4.37927E-03,5.54908E-03\
               ,7.01192E-03,8.83064E-03,1.10763E-02,1.38266E-02\
               ,1.71641E-02,2.11717E-02,2.59364E-02,3.15062E-02\
               ,3.79623E-02,4.53425E-02,5.36750E-02,6.29705E-02\
               ,7.32221E-02,8.44039E-02,9.64793E-02,1.09332E-01\
               ,1.23067E-01,1.37507E-01,1.52639E-01,1.68416E-01\
               ,1.84794E-01,2.01731E-01,2.19016E-01,2.36948E-01\
               ,2.55242E-01,2.73927E-01,2.92954E-01,3.12340E-01\
               ,3.32036E-01,3.52019E-01,3.72282E-01,3.92772E-01\
               ,4.13533E-01,4.34326E-01,4.55495E-01,4.76836E-01\
               ,4.98342E-01,5.20006E-01,5.41818E-01,5.63773E-01\
               ,5.85861E-01,6.08077E-01,6.30459E-01,6.52800E-01\
               ,6.75387E-01,6.98063E-01,7.20830E-01,7.43683E-01\
               ,7.66623E-01,7.89636E-01,8.12791E-01,8.35940E-01\
               ,8.59175E-01,8.82485E-01,9.05866E-01,9.29311E-01\
               ,9.52817E-01,9.76387E-01,1.00000E+00]
    
        return X,Q2
   
    def _gen_table(self,tar):
    
        X,Q2=self.gen_cj_grid()

        idxs = self.idxs
   
        HT = conf['ht4']
 
        nx=len(X)
        nQ2=len(Q2)
 
        #--fill table
        table={}
        table[idxs[0]] = []
        npts=nQ2*nx

        for iQ2 in range(nQ2):
            for ix in range(nx):
                table[idxs[0]].append(HT.get_ht(X[ix],Q2[iQ2],tar,'F2')*Q2[iQ2])

        #--remap tables to match with lhapdf format
        for iflav in idxs: 
            new_list=[]
            for ix in range(nx):
                for inQ2 in range(nQ2):
                    idx=ix+inQ2*nx
                    new_list.append(table[iflav][idx])
            table[iflav]=new_list

        return X,Q2,table

    def gen_lhapdf_info_file(self,X,Q2,nrep,wdir,file_name,info):
      
        idxs = self.idxs

        #--get flav string
        flavs=''
        for idx in idxs: flavs+='%d, '%idx
        flavs=flavs.rstrip()
        flavs=flavs.rstrip(',')
   
        #--kinematic limits
        xmin=X[0]
        xmax=X[-1]
        Qmin=Q2[0]**0.5
        Qmax=Q2[-1]**0.5
    
        #--qcd params
        load_config('%s/input.py'%wdir)
        RESMAN(nworkers=1,parallel=False,datasets=False)
        aS=[conf['alphaS'].get_alphaS(_) for _ in Q2]
        mZ=conf['aux'].mZ
        mb=conf['aux'].mb
        mc=conf['aux'].mc
        alphaSMZ=conf['aux'].alphaSMZ
    
        #--begin lhapdf info file
        lines=[]
        lines.append('SetDesc:         "<description>"')
        lines.append('SetIndex:        <index>')
        lines.append('Authors:         <authors>')
        lines.append('Reference:       reference')
        lines.append('Format:          lhagrid1')
        lines.append('DataVersion:     1')
        lines.append('NumMembers:      %d'%nrep)
        lines.append('Particle:        particle')
        lines.append('Flavors:         [%s]'%flavs)
        lines.append('OrderQCD:        1')
        lines.append('FlavorScheme:    variable')
        lines.append('NumFlavors:      %d'%len(idxs))
        lines.append('ErrorType:       replicas')
        lines.append('XMin:            %0.2e'%xmin)
        lines.append('XMax:            %0.2e'%xmax)
        lines.append('QMin:            %0.2e'%Qmin)
        lines.append('QMax:            %0.2e'%Qmax)
        lines.append('MZ:              %f'%mZ)
        lines.append('MUp:             0.0')
        lines.append('MDown:           0.0')
        lines.append('MStrange:        0.0')
        lines.append('MCharm:          %f'%mc)
        lines.append('MBottom:         %f'%mb)
        lines.append('MTop:            180.0')
        lines.append('AlphaS_MZ:       %f'%alphaSMZ)
        lines.append('AlphaS_OrderQCD: 1')
        lines.append('AlphaS_Type:     ipol')
        line='AlphaS_Qs: ['
        for _ in Q2: line+=('%10.5e, '%_**0.5).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        line='AlphaS_Vals: ['
        for _ in aS: line+=('%10.5e, '%_).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        lines.append('AlphaS_Lambda4: 0')
        lines.append('AlphaS_Lambda5: 0')
    
        for i in range(len(lines)):
            for _ in info:
                lines[i]=lines[i].replace(_,info[_])
    
        lines=[l+'\n' for l in lines]
        tab=open('%s/data/%s/%s.info'%(wdir, file_name, file_name),'w')
        tab.writelines(lines)
        tab.close()
 
    def gen_lhapdf_dat_file(self,X,Q2,table, wdir, file_name,setlabel):

        idxs = self.idxs
        #--get flav string
        flavs=''
        for idx in idxs: flavs+='%d, '%idx
        flavs=flavs.rstrip()
        flavs=flavs.rstrip(',')

        lines=[]  
        lines.append('PdfType: replica')
        lines.append('Format: lhagrid1')
        lines.append('---')
        line=''
        for _ in X: line+=('%10.5e '%_).upper()
        lines.append(line)
        line=''
        for _ in Q2: line+=('%10.5e '%_**0.5).upper()
        lines.append(line)
        lines.append('%s'%flavs)
    
        nx=len(X)
        nQ2=len(Q2)
   
        for i in range(nx*nQ2):
            line=''
            for iflav in idxs:
                line+=('%10.5e '%table[iflav][i]).upper()
            lines.append(line)
        lines.append('---')
        lines=[l+'\n' for l in lines]
        idx=str(setlabel).zfill(4)
        tab=open('%s/data/%s/%s_%s.dat'%(wdir, file_name, file_name,idx),'w')
        tab.writelines(lines)
        tab.close()
    
    def gen_tables(self,wdir,tar,file_name,info,info_only=False):
    
        print('\ngenerating HT STF LHAPDF tables for %s using %s'%(tar,wdir))
  
        idxs = [908]

        self.idxs = idxs
 
        write_lhapdf_conf(wdir)
        load_config('%s/input.py'%wdir)
        istep=core.get_istep()
        core.mod_conf(istep) #--set conf as specified in istep   

        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
    
        jar=load('%s/data/jar-%d.dat'%(wdir,istep))
        replicas=jar['replicas']
    
        #--check order consistency
        order=jar['order']
        parman.order = order
    
        #--create output dir
        checkdir(wdir + '/data/')
        checkdir(wdir + '/data/%s/' % file_name)
    
        #--gen_lhapdf_info_file
        X,Q2=self.gen_cj_grid()
        nrep=len(replicas)
        self.gen_lhapdf_info_file(X,Q2,nrep, wdir, file_name,info)

        #--gen lhapdf_data_files
        if info_only==False:
            cnt=0
            for par in replicas:
                lprint('progress: %d/%d'%(cnt+1,len(replicas)))
                parman.set_new_params(par)
                X,Q2,table = self._gen_table(tar)
                self.gen_lhapdf_dat_file(X,Q2,table, wdir, file_name,cnt)
                cnt+=1
            print

#--need to update so that mem=0000 is the average 
class OFF:

    def __init__(self):

        pass

    def gen_cj_grid(self):
    
        Q2=[1.30000E+00,1.50159E+00,1.75516E+00,2.07810E+00\
                ,2.49495E+00,3.04086E+00,3.76715E+00,4.50000E+00\
                ,4.75000E+00,6.23113E+00,8.37423E+00,1.15549E+01\
                ,1.64076E+01,2.40380E+01,3.64361E+01,5.73145E+01\
                ,9.38707E+01,1.60654E+02,2.88438E+02,5.45587E+02\
                ,1.09231E+03,2.32646E+03,5.30043E+03,1.29956E+04\
                ,3.45140E+04,1.00000E+05]
    
        X=[1.00000E-06,1.28121E-06,1.64152E-06,2.10317E-06\
               ,2.69463E-06,3.45242E-06,4.42329E-06,5.66715E-06\
               ,7.26076E-06,9.30241E-06,1.19180E-05,1.52689E-05\
               ,1.95617E-05,2.50609E-05,3.21053E-05,4.11287E-05\
               ,5.26863E-05,6.74889E-05,8.64459E-05,1.10720E-04\
               ,1.41800E-04,1.81585E-04,2.32503E-04,2.97652E-04\
               ,3.80981E-04,4.87518E-04,6.26039E-04,8.00452E-04\
               ,1.02297E-03,1.30657E-03,1.66759E-03,2.12729E-03\
               ,2.71054E-03,3.44865E-03,4.37927E-03,5.54908E-03\
               ,7.01192E-03,8.83064E-03,1.10763E-02,1.38266E-02\
               ,1.71641E-02,2.11717E-02,2.59364E-02,3.15062E-02\
               ,3.79623E-02,4.53425E-02,5.36750E-02,6.29705E-02\
               ,7.32221E-02,8.44039E-02,9.64793E-02,1.09332E-01\
               ,1.23067E-01,1.37507E-01,1.52639E-01,1.68416E-01\
               ,1.84794E-01,2.01731E-01,2.19016E-01,2.36948E-01\
               ,2.55242E-01,2.73927E-01,2.92954E-01,3.12340E-01\
               ,3.32036E-01,3.52019E-01,3.72282E-01,3.92772E-01\
               ,4.13533E-01,4.34326E-01,4.55495E-01,4.76836E-01\
               ,4.98342E-01,5.20006E-01,5.41818E-01,5.63773E-01\
               ,5.85861E-01,6.08077E-01,6.30459E-01,6.52800E-01\
               ,6.75387E-01,6.98063E-01,7.20830E-01,7.43683E-01\
               ,7.66623E-01,7.89636E-01,8.12791E-01,8.35940E-01\
               ,8.59175E-01,8.82485E-01,9.05866E-01,9.29311E-01\
               ,9.52817E-01,9.76387E-01,1.00000E+00]
    
        return X,Q2
   
    def _gen_table(self,tar):
    
        X,Q2=self.gen_cj_grid()

        idxs = self.idxs
   
        OFF = conf['off']
 
        nx=len(X)
        nQ2=len(Q2)
 
        #--fill table
        table={}
        table[idxs[0]] = []
        npts=nQ2*nx

        for iQ2 in range(nQ2):
            for ix in range(nx):
                table[idxs[0]].append(OFF.get_offshell(X[ix],np.array([Q2[iQ2]]),tar,'F2'))

        #--remap tables to match with lhapdf format
        for iflav in idxs: 
            new_list=[]
            for ix in range(nx):
                for inQ2 in range(nQ2):
                    idx=ix+inQ2*nx
                    new_list.append(table[iflav][idx])
            table[iflav]=new_list

        return X,Q2,table

    def gen_lhapdf_info_file(self,X,Q2,nrep,wdir,file_name,info):
      
        idxs = self.idxs

        #--get flav string
        flavs=''
        for idx in idxs: flavs+='%d, '%idx
        flavs=flavs.rstrip()
        flavs=flavs.rstrip(',')
   
        #--kinematic limits
        xmin=X[0]
        xmax=X[-1]
        Qmin=Q2[0]**0.5
        Qmax=Q2[-1]**0.5
    
        #--qcd params
        load_config('%s/input.py'%wdir)
        RESMAN(nworkers=1,parallel=False,datasets=False)
        aS=[conf['alphaS'].get_alphaS(_) for _ in Q2]
        mZ=conf['aux'].mZ
        mb=conf['aux'].mb
        mc=conf['aux'].mc
        alphaSMZ=conf['aux'].alphaSMZ
    
        #--begin lhapdf info file
        lines=[]
        lines.append('SetDesc:         "<description>"')
        lines.append('SetIndex:        <index>')
        lines.append('Authors:         <authors>')
        lines.append('Reference:       reference')
        lines.append('Format:          lhagrid1')
        lines.append('DataVersion:     1')
        lines.append('NumMembers:      %d'%nrep)
        lines.append('Particle:        particle')
        lines.append('Flavors:         [%s]'%flavs)
        lines.append('OrderQCD:        1')
        lines.append('FlavorScheme:    variable')
        lines.append('NumFlavors:      %d'%len(idxs))
        lines.append('ErrorType:       replicas')
        lines.append('XMin:            %0.2e'%xmin)
        lines.append('XMax:            %0.2e'%xmax)
        lines.append('QMin:            %0.2e'%Qmin)
        lines.append('QMax:            %0.2e'%Qmax)
        lines.append('MZ:              %f'%mZ)
        lines.append('MUp:             0.0')
        lines.append('MDown:           0.0')
        lines.append('MStrange:        0.0')
        lines.append('MCharm:          %f'%mc)
        lines.append('MBottom:         %f'%mb)
        lines.append('MTop:            180.0')
        lines.append('AlphaS_MZ:       %f'%alphaSMZ)
        lines.append('AlphaS_OrderQCD: 1')
        lines.append('AlphaS_Type:     ipol')
        line='AlphaS_Qs: ['
        for _ in Q2: line+=('%10.5e, '%_**0.5).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        line='AlphaS_Vals: ['
        for _ in aS: line+=('%10.5e, '%_).upper()
        line=line.rstrip(',')+']'
        lines.append(line)
        lines.append('AlphaS_Lambda4: 0')
        lines.append('AlphaS_Lambda5: 0')
    
        for i in range(len(lines)):
            for _ in info:
                lines[i]=lines[i].replace(_,info[_])
    
        lines=[l+'\n' for l in lines]
        tab=open('%s/data/%s/%s.info'%(wdir, file_name, file_name),'w')
        tab.writelines(lines)
        tab.close()
 
    def gen_lhapdf_dat_file(self,X,Q2,table, wdir, file_name,setlabel):

        idxs = self.idxs
        #--get flav string
        flavs=''
        for idx in idxs: flavs+='%d, '%idx
        flavs=flavs.rstrip()
        flavs=flavs.rstrip(',')

        lines=[]  
        lines.append('PdfType: replica')
        lines.append('Format: lhagrid1')
        lines.append('---')
        line=''
        for _ in X: line+=('%10.5e '%_).upper()
        lines.append(line)
        line=''
        for _ in Q2: line+=('%10.5e '%_**0.5).upper()
        lines.append(line)
        lines.append('%s'%flavs)
    
        nx=len(X)
        nQ2=len(Q2)
   
        for i in range(nx*nQ2):
            line=''
            for iflav in idxs:
                line+=('%10.5e '%table[iflav][i]).upper()
            lines.append(line)
        lines.append('---')
        lines=[l+'\n' for l in lines]
        idx=str(setlabel).zfill(4)
        tab=open('%s/data/%s/%s_%s.dat'%(wdir, file_name, file_name,idx),'w')
        tab.writelines(lines)
        tab.close()
    
    def gen_tables(self,wdir,tar,file_name,info,info_only=False):
    
        print('\ngenerating Offshell STF LHAPDF tables for %s using %s'%(tar,wdir))
  
        idxs = [908]

        self.idxs = idxs
 
        write_lhapdf_conf(wdir)
        load_config('%s/input.py'%wdir)
        istep=core.get_istep()
        core.mod_conf(istep) #--set conf as specified in istep   

        resman=RESMAN(nworkers=1,parallel=False,datasets=False)
        parman=resman.parman
    
        jar=load('%s/data/jar-%d.dat'%(wdir,istep))
        replicas=jar['replicas']
    
        #--check order consistency
        order=jar['order']
        parman.order = order
    
        #--create output dir
        checkdir(wdir + '/data/')
        checkdir(wdir + '/data/%s/' % file_name)
    
        #--gen_lhapdf_info_file
        X,Q2=self.gen_cj_grid()
        nrep=len(replicas)
        self.gen_lhapdf_info_file(X,Q2,nrep, wdir, file_name,info)

        #--gen lhapdf_data_files
        if info_only==False:
            cnt=0
            for par in replicas:
                lprint('progress: %d/%d'%(cnt+1,len(replicas)))
                parman.set_new_params(par)
                X,Q2,table = self._gen_table(tar)
                self.gen_lhapdf_dat_file(X,Q2,table, wdir, file_name,cnt)
                cnt+=1
            print



def plot_ht(mode=0):

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1) 

    hand = {}


    #--collect data from different groups
    Q2 = 10
    data = {}
    for tar in ['p','n']:
        data[tar] = []
        if tar == 'p': tabname, color = 'JAM21PDF-HT_proton' , 'firebrick'
        if tar == 'n': tabname, color = 'JAM21PDF-HT_neutron', 'darkgreen'
        HT = lhapdf.mkPDFs(tabname)
        nrep = len(HT)
        for i in range(nrep):
            ht = np.array([HT[i].xfxQ2(908,x,Q2) for x in X])
            data[tar].append(ht)

        if mode == 0:
            for i in range(nrep):
                hand[tar] ,= ax11.plot(X,data[tar][i],color=color,alpha=0.1)

        if mode == 1:
            mean = np.mean(np.array(data[tar]),axis=0)
            std  = np.std (np.array(data[tar]),axis=0)
            hand[tar] = ax11.fill_between(X,mean-std,mean+std,color=color,alpha=0.9)
        

    h0 =-3.2874
    h1 = 1.9274
    h2 =-2.0701
    ht = h0*X**h1*(1+h2*X)
    hand['CJ15'] ,= ax11.plot(X,ht,'b--')

    ax11.set_ylim(-0.4,2)

    ax11.text(0.05,0.25,r'\boldmath$H^N$',transform=ax11.transAxes,size=40)

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    ax11.set_xlim(0,1)
    ax11.set_xlabel(r'\boldmath$x$',size=30)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.axhline(0,0,1,ls='--',color='black',alpha=0.5)

    ax11.text(0.75,0.05,r'\textbf{\textrm{AOT}}',size=30,transform=ax11.transAxes)

    for ax in [ax11]:
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)

    ax11.set_xticks([0,0.2,0.4,0.6,0.8])

    handles,labels = [],[]
    handles.append(hand['p'])
    handles.append(hand['n'])
    handles.append(hand['CJ15'])
    labels.append(r'\textbf{\textrm{JAM21 (p)}}')
    labels.append(r'\textbf{\textrm{JAM21 (n)}}')
    labels.append(r'\textbf{\textrm{CJ15}}')

    ax11.legend(handles,labels,frameon=False,loc=2,fontsize=25, handletextpad = 0.5, handlelength = 1.5)

    py.tight_layout()

    filename = 'gallery/ht'
    if mode == 1: filename += '-bands'
    filename += '.png'
    py.savefig(filename)
    py.clf()
    print('Saving figure to %s'%filename)

def plot_off(Q2,mode=0):

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*8,nrows*5))
    ax11 = py.subplot(nrows,ncols,1)

    hand = {}

    OFF = lhapdf.mkPDFs('JAM21PDF-offshell')
    nrep = len(OFF)

    data = [] 

    for i in range(nrep):
        off =  np.array([OFF[i].xfxQ2(908,x,Q2) for x in X])
        data.append(off)

    if mode==0:
        for i in range(nrep):
            hand['JAM21'] ,= ax11.plot(X,data[i],color='red',alpha=0.1)

    if mode == 1:
        mean = np.mean(np.array(data),axis=0)
        std  = np.std(np.array(data),axis=0)
        hand['JAM21'] = ax11.fill_between(X,mean-std,mean+std,color='red',alpha=0.9,)

    
    #--CJ15 
    C =-3.6735
    x0= 5.7717e-2
    x1=0.36419
    dfcj=C*(X-x0)*(X-x1)*(1+x0-X)
    hand['CJ'] ,= ax11.plot(X,dfcj,'b--')
    #--KP 
    C = 8.10
    x0= 0.448
    x1= 0.05
    dfcj=C*(X-x0)*(X-x1)*(1+x0-X)
    hand['KP'] ,= ax11.plot(X,dfcj,'g--')

    ax11.tick_params(axis='both',which='both',top=True,right=True,direction='in',labelsize=30)

    ax11.text(0.60,0.05,r'$Q^2=%s{\rm~GeV^2}$'%Q2,size=30,transform=ax11.transAxes)

    ax11.set_ylim(-1.2,1.2)
    ax11.set_xlim(0,1)
    ax11.text(0.05,0.05,r'\boldmath$\delta f^0$',transform=ax11.transAxes,size=40)
    ax11.set_xlabel(r'\boldmath$x$'         ,size=30)
    ax11.xaxis.set_label_coords(0.95,0.00)

    ax11.axhline(0,alpha=0.5,color='k',ls='--')

 
    for ax in [ax11]:
        minorLocator = MultipleLocator(0.1)
        majorLocator = MultipleLocator(0.5)
        ax.yaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_major_locator(majorLocator)
        minorLocator = MultipleLocator(0.02)
        majorLocator = MultipleLocator(0.2)
        ax.xaxis.set_minor_locator(minorLocator)
        ax.xaxis.set_major_locator(majorLocator)
        ax.xaxis.set_tick_params(which='major',length=6)
        ax.xaxis.set_tick_params(which='minor',length=3)
        ax.yaxis.set_tick_params(which='major',length=6)
        ax.yaxis.set_tick_params(which='minor',length=3)
        ax.set_xticks([0,0.2,0.4,0.6,0.8])


    handles,labels=[],[]
    handles.append(hand['JAM21'])
    handles.append(hand['CJ'])
    handles.append(hand['KP'])
    labels.append(r'\textbf{\textrm{JAM21}}')
    labels.append(r'\textbf{\textrm{CJ15}}')
    labels.append(r'\textbf{\textrm{KP}}')

    ax11.legend(handles,labels,frameon=False,loc='upper left',fontsize=28, handletextpad = 0.5, handlelength = 1.5, ncol = 1, columnspacing = 0.5)

    py.tight_layout()

    filename = 'gallery/off'
    if mode == 1: filename += '-bands'
    filename += '.png'
    print('Saving figures to %s'%filename)
    py.savefig(filename)
    py.clf()

def plot_CCstf(Q2,mode=0):

    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*7,nrows*4))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)

    hand = {}

    stfs = ['W2+','WL+','W3+','W2-','WL-','W3-']
    data = {stf: [] for stf in stfs} 

    tablename = 'JAM21PDF-STF_proton'
    STF = lhapdf.mkPDFs(tablename)
    nrep = len(STF)

    for i in range(nrep):
        W2m =  np.array([STF[i].xfxQ2(930,x,Q2)*x for x in X])
        WLm =  np.array([STF[i].xfxQ2(931,x,Q2)*x for x in X])
        W3m =  np.array([STF[i].xfxQ2(932,x,Q2)*x for x in X])
        W2p =  np.array([STF[i].xfxQ2(940,x,Q2)*x for x in X])
        WLp =  np.array([STF[i].xfxQ2(941,x,Q2)*x for x in X])
        W3p =  np.array([STF[i].xfxQ2(942,x,Q2)*x for x in X])
        data['W2+'].append(W2p)
        data['WL+'].append(WLp)
        data['W3+'].append(W3p)
        data['W2-'].append(W2m)
        data['WL-'].append(WLm)
        data['W3-'].append(W3m)

    for stf in data:
        mean = np.mean(data[stf],axis=0)
        std = np.std(data[stf],axis=0)

        if stf[-1]=='+': color='firebrick'
        if stf[-1]=='-': color='darkcyan'

        if stf =='W2+':   ax = ax11
        elif stf =='WL+': ax = ax12
        elif stf =='W3+': ax = ax13
        elif stf =='W2-': ax = ax11
        elif stf =='WL-': ax = ax12
        elif stf =='W3-': ax = ax13
        else: continue

        #--plot each replica
        if mode==0:
            for i in range(nrep):
                hand[stf[-1]] ,= ax.plot(X,data[stf][i],color=color,alpha=0.1)
    
        #--plot average and standard deviation
        if mode==1:
            hand[stf[-1]] = ax.fill_between(X,mean-std,mean+std,color=color,alpha=0.9)


    for ax in [ax11,ax12,ax13]:
          ax.set_xlim(1e-4,1)
          ax.semilogx()
            
          ax.tick_params(axis='both', which='both', top=True, right=True, direction='in',labelsize=20)
          ax.set_xticks([0.0001,0.001,0.01,0.1,1])
          ax.set_xticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$',r'$1$'])


    ax13.axhline(0,0,1,ls='--',color='black',alpha=0.5)

    ax11.set_ylim(0,0.4)   
    ax12.set_ylim(0,0.015) 
    ax13.set_ylim(-1.0,2.0)

    ax11.set_xlabel(r'$x$' ,size=35)
    ax12.set_xlabel(r'$x$' ,size=35)   
    ax13.set_xlabel(r'$x$' ,size=35)   

    if Q2 == 1.27**2: ax11.text(0.40,0.85,r'$Q^2 = m_c^2$'                                  , transform=ax11.transAxes,size=30)
    else:             ax11.text(0.40,0.85,r'$Q^2 = %s$'%Q2 + ' ' + r'\textrm{GeV}' + r'$^2$', transform=ax11.transAxes,size=30)

    ax11.text(0.05,0.85,r'\boldmath$xW_2$',transform=ax11.transAxes,size=30)
    ax12.text(0.05,0.85,r'\boldmath$xW_L$',transform=ax12.transAxes,size=30)
    ax13.text(0.05,0.85,r'\boldmath$xW_3$',transform=ax13.transAxes,size=30)

    handles,labels=[],[]
    handles.append(hand['+'])
    handles.append(hand['-'])
    labels.append(r'\boldmath$W^+$')
    labels.append(r'\boldmath$W^-$')
    ax11.legend(handles,labels,frameon=False,loc='lower left',fontsize=28, handletextpad = 0.5, handlelength = 1.5, ncol = 1, columnspacing = 0.5)

    py.tight_layout()

    filename = 'gallery/stfs-CC'
    if mode==1: filename += '-bands'
    filename+='.png'
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
    py.clf()











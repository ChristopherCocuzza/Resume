#!/usr/bin/env python
import os,sys
import argparse
#--set lhapdf data path
#os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/analysis-hx'
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from tools.tools import checkdir

#--from qpdlib
from analysis.qpdlib.tolhapdf import QCF
from analysis.qpdlib.tolhapdf import STF
from analysis.qpdlib.tolhapdf import HT
from analysis.qpdlib.tolhapdf import OFF

QCF = QCF()
STF = STF()
HT  = HT()
OFF = OFF()

description = 'JAM22 NLO global analysis of proton PDFs, PPDFs, pion FFs, and kaon FFs; mem=0 -> average of replicas; mem > 0 -> replicas' 
index       = '1'
authors     = 'JAM Collaboration. C. Cocuzza, W. Melnitchouk, A. Metz, N. Sato'
reference   = 'N/A'


if __name__=='__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument('-d'  ,'--directory' ,type=str   ,default=None       ,help='directory name to store results')
    ap.add_argument('-f'  ,'--function'  ,type=str   ,default='pdf'      ,help='function to generate LHAPDF files for')
    ap.add_argument('-t'  ,'--target'    ,type=str   ,default='p'        ,help='target for structure function')
    args = ap.parse_args()

    checkdir('%s/data'%args.directory)
    checkdir('%s/gallery'%args.directory)

    info = {}
    info['<description>'] = description
    info['<index>']       = index
    info['<authors>']     = authors
    info['reference']     = reference
   
    print('Generating LHAPDF files using directory %s for the function %s with the following information \n'%(args.directory,args.function))
 
    if args.function=='pdf':
        name        = 'JAM22-PDF_proton_nlo'
        particle    = '2212'
        print('Name:        %s'%name)
        print('Description: %s'%description)
        print('Index:       %s'%index)
        print('Authors:     %s'%authors)
        print('Reference:   %s'%reference)
        print('Particle:    %s'%particle)
        info['particle']      = particle
        QCF.gen_tables(args.directory,'pdf',name,info,info_only=False)

    if args.function=='ppdf':
        name        = 'JAM22-PPDF_proton_nlo'
        particle    = '2212'
        print('Name:        %s'%name)
        print('Description: %s'%description)
        print('Index:       %s'%index)
        print('Authors:     %s'%authors)
        print('Reference:   %s'%reference)
        print('Particle:    %s'%particle)
        info['particle']      = particle
        QCF.gen_tables(args.directory,'ppdf',name,info,info_only=False)

    if args.function=='ffpion':
        name        = 'JAM22-FF_pion_nlo'
        particle    = '211'
        print('Name:        %s'%name)
        print('Description: %s'%description)
        print('Index:       %s'%index)
        print('Authors:     %s'%authors)
        print('Reference:   %s'%reference)
        print('Particle:    %s'%particle)
        info['particle']      = particle
        QCF.gen_tables(args.directory,'ffpion',name,info,info_only=False)

    if args.function=='ffkaon':
        name        = 'JAM22-FF_kaon_nlo'
        particle    = '321'
        print('Name:        %s'%name)
        print('Description: %s'%description)
        print('Index:       %s'%index)
        print('Authors:     %s'%authors)
        print('Reference:   %s'%reference)
        print('Particle:    %s'%particle)
        info['particle']      = particle
        QCF.gen_tables(args.directory,'ffkaon',name,info,info_only=False)

    if args.function=='ffhadron':
        name        = 'JAM22-FF_hadron_nlo'
        particle    = 'unidentified hadron'
        print('Name:        %s'%name)
        print('Description: %s'%description)
        print('Index:       %s'%index)
        print('Authors:     %s'%authors)
        print('Reference:   %s'%reference)
        print('Particle:    %s'%particle)
        info['particle']      = particle
        QCF.gen_tables(args.directory,'ffhadron',name,info,info_only=False)


    
    if args.function=='stf':
        if args.target=='p': name        = 'JAM22-STF_proton'
        if args.target=='n': name        = 'JAM22-STF_neutron'
        if args.target=='d': name        = 'JAM22-STF_deuteron'
        if args.target=='h': name        = 'JAM22-STF_helium'
        if args.target=='t': name        = 'JAM22-STF_tritium'
        if args.target=='p': particle    = '2212'
        if args.target=='n': particle    = '2112'
        if args.target=='d': particle    = '1000010020'
        if args.target=='h': particle    = '1000020030'
        if args.target=='t': particle    = '1000010030'
        print('Name:        %s'%name)
        print('Description: %s'%description)
        print('Index:       %s'%index)
        print('Authors:     %s'%authors)
        print('Reference:   %s'%reference)
        print('Particle:    %s'%particle)
        info['particle']      = particle
        STF.gen_tables(args.directory,args.target,name,info,info_only=False)

    sys.exit()
    
    name = 'JAM21PDF-HT_proton'
    #HT.gen_tables(wdir,'p',name,info,info_only=False)
    
    info['particle'] = '2112'
    name = 'JAM21PDF-HT_neutron'
    #HT.gen_tables(wdir,'n',name,info,info_only=False)
    
    info['particle'] = '2212'
    name = 'JAM21PDF-offshell'
    #OFF.gen_tables(wdir,'p',name,info,info_only=False)












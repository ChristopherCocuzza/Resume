#!/usr/bin/env python
import os,sys
#--set lhapdf data path
version = int(sys.version[0])
os.environ["LHAPDF_DATA_PATH"] = '/work/JAM/ccocuzza/lhapdf/python%s/sets'%version
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import kmeanconf as kc

#--from corelib
from analysis.corelib import core, inspect, predict, classifier, optpriors, jar, mlsamples, summary

#--from qpdlib
from analysis.qpdlib import pdf, offpdf, nucpdf, ppdf, ff, tolhapdf, QCFxlsx

#--from obslib
from analysis.obslib import stf, pstf, off, ht, idis, pidis, sia, sidis, psidis, dy, wasym, zrap, wzrv, wpol, jet, pjet, SU23, kin, lattice
from analysis.obslib import discalc, xlsx, reweight

#--from parlib
from analysis.parlib  import params, corr

#--primary working directory
try: wdir=sys.argv[1]
except: wdir = None

Q2 = 10

#summary.print_summary(wdir,kc)
#sidis.plot_hadron(wdir,kc)
#sia.plot_hadron(wdir,kc)
#ff.gen_xf(wdir,'hadron',Q2=Q2)
#ff.plot_xf_hadron(wdir,Q2=Q2,mode=0)
#ff.plot_xf_hadron(wdir,Q2=Q2,mode=1)
#nucpdf.gen_nuclear_pdf(wdir,Q2)
#sys.exit()

######################
##--Initial Processing
######################

inspect.get_msr_inspected(wdir,limit=3.0)
predict.get_predictions(wdir,force=False)
classifier.gen_labels(wdir,kc)
jar.gen_jar_file(wdir,kc)
summary.print_summary(wdir,kc)

#classifier.plot_chi2_dist_per_exp(wdir,kc,'dy',20002)
#classifier.print_chi2_per_exp(wdir,kc)

###################
##--Optimize priors
###################

#optpriors.gen_priors(wdir,kc,10)

###################
#--Plot proton pdfs
###################

#--generate and plot data
pdf.gen_xf(wdir,Q2)
pdf.plot_xf(wdir,Q2,mode=0,sets=False)                
pdf.plot_xf(wdir,Q2,mode=1,sets=False)                

offpdf.gen_xf(wdir,Q2)
offpdf.plot_xf(wdir,Q2,mode=0)
offpdf.plot_xf(wdir,Q2,mode=1)


####################
##--Observable plots
####################

idis. plot_obs (wdir,kc)
dy.   plot_obs (wdir,kc)
dy.   plot_SQ  (wdir,kc,mode=0)
zrap. plot_obs (wdir,kc,mode=0)
wasym.plot_obs (wdir,kc,mode=0)
wzrv. plot_obs (wdir,kc,mode=0)
wzrv. plot_star(wdir,kc,mode=0)
dy.   plot_SQ  (wdir,kc,mode=1)
zrap. plot_obs (wdir,kc,mode=1)
wasym.plot_obs (wdir,kc,mode=1)
wzrv. plot_obs (wdir,kc,mode=1)
wzrv. plot_star(wdir,kc,mode=1)
#jet.   plot_obs(wdir,kc)
dy.   plot_E866_ratio(wdir,kc,mode=0)
dy.   plot_E866_ratio(wdir,kc,mode=1)


##---------------------------------------------------------------
##--Polarized
##---------------------------------------------------------------


########################
#--Polarized proton pdfs
########################
PSETS = []

ppdf.gen_xf(wdir,Q2=Q2)         
ppdf.plot_xf(wdir,Q2=10,mode=0)
ppdf.plot_xf(wdir,Q2=10,mode=1)
ppdf.plot_polarization(wdir,Q2=Q2,mode=0)
ppdf.plot_polarization(wdir,Q2=Q2,mode=1)
ppdf.plot_helicity    (wdir,Q2=Q2,mode=0)
ppdf.plot_helicity    (wdir,Q2=Q2,mode=1)
ppdf.gen_moments_trunc(wdir,Q2=4)


########################
#--polarized structure functions and related quantities
########################
pstf.gen_g2res(wdir,tar='p',Q2=10)
pstf.gen_g2res(wdir,tar='n',Q2=10)
pstf.plot_g2res(wdir,Q2=10,mode=0)
pstf.plot_g2res(wdir,Q2=10,mode=1)
ht.plot_pol_ht(wdir,Q2=Q2,mode=0)
ht.plot_pol_ht(wdir,Q2=Q2,mode=1)



########################
#--polarized observables
########################

pidis .plot_obs(wdir,kc) 
SU23  .plot_obs(wdir,kc,mode=1)
wpol  .plot_obs(wdir,kc,mode=1)
pjet  .plot_obs(wdir,kc)
psidis.plot_obs(wdir,kc,mode=0)
psidis.plot_obs(wdir,kc,mode=1)

###########################
##--Parameter distributions
###########################
hist  = False



##---------------------------------------------------------------
##--Fragmentation functions
##---------------------------------------------------------------

ff.gen_xf(wdir,'pion'  ,Q2=Q2)
ff.gen_xf(wdir,'kaon'  ,Q2=Q2)
ff.gen_xf(wdir,'hadron',Q2=Q2)
ff.plot_xf_pion  (wdir ,Q2=Q2,mode=0)
ff.plot_xf_pion  (wdir ,Q2=Q2,mode=1)
ff.plot_xf_kaon  (wdir ,Q2=Q2,mode=0)
ff.plot_xf_kaon  (wdir ,Q2=Q2,mode=1)
ff.plot_xf_hadron(wdir ,Q2=Q2,mode=0)
ff.plot_xf_hadron(wdir ,Q2=Q2,mode=1)

sia.plot_pion  (wdir,kc)
sia.plot_kaon  (wdir,kc)
sia.plot_hadron(wdir,kc)

sidis.plot_pion  (wdir,kc)
sidis.plot_kaon  (wdir,kc)
sidis.plot_hadron(wdir,kc)


##---------------------------------------------------------------
##--Parameter distributions
##---------------------------------------------------------------
hist=False

params.plot_params(wdir,'pdf',hist)
params.plot_params(wdir,'ht4',hist)
params.plot_params(wdir,'off pdf',hist)
params.plot_params(wdir,'ppdf'   ,hist)
#params.plot_params(wdir,'pol ht4',hist)
#params.plot_params(wdir,'t3ppdf' ,hist)
#params.plot_params(wdir,'pol off',hist)
params.plot_params(wdir,'ffpion',hist)
params.plot_params(wdir,'ffkaon',hist)
params.plot_params(wdir,'ffhadron',hist)


params.plot_norms (wdir,'idis')
params.plot_norms (wdir,'dy'  )
params.plot_norms (wdir,'jet' )
params.plot_norms(wdir,'pidis')
params.plot_norms(wdir,'wzrv')












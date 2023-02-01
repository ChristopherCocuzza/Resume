import os
conf={}

#--fitting setups
conf['bootstrap']=True
conf['flat par']=True

#--setups for DGLAP
conf['dglap mode']='truncated'
conf['alphaSmode']='backward'
conf['order'] = 'NLO'
conf['Q20']   = 1.27**2

#--nuclear smearing
conf['nuc']   = True

#--datasets

conf['datasets']={}

#--lepton-hadron reactions

Q2cut=1.3**2
W2cut=3.0

##--IDIS
conf['datasets']['idis']={}
conf['datasets']['idis']['filters']=[]
conf['datasets']['idis']['filters'].append("Q2>%f"%Q2cut)
conf['datasets']['idis']['filters'].append("W2>%f"%W2cut)
conf['datasets']['idis']['xlsx']={}
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['idis']['xlsx'][10010]='idis/expdata/10010.xlsx' # proton   | F2            | SLAC
conf['datasets']['idis']['xlsx'][10016]='idis/expdata/10016.xlsx' # proton   | F2            | BCDMS
conf['datasets']['idis']['xlsx'][10020]='idis/expdata/10020.xlsx' # proton   | F2            | NMC
conf['datasets']['idis']['xlsx'][10003]='idis/expdata/10003.xlsx' # proton   | sigma red     | JLab Hall C (E00-106)
conf['datasets']['idis']['xlsx'][10026]='idis/expdata/10026.xlsx' # proton   | sigma red     | HERA II NC e+ (1)
conf['datasets']['idis']['xlsx'][10027]='idis/expdata/10027.xlsx' # proton   | sigma red     | HERA II NC e+ (2)
conf['datasets']['idis']['xlsx'][10028]='idis/expdata/10028.xlsx' # proton   | sigma red     | HERA II NC e+ (3)
conf['datasets']['idis']['xlsx'][10029]='idis/expdata/10029.xlsx' # proton   | sigma red     | HERA II NC e+ (4)
conf['datasets']['idis']['xlsx'][10030]='idis/expdata/10030.xlsx' # proton   | sigma red     | HERA II NC e-
conf['datasets']['idis']['xlsx'][10031]='idis/expdata/10031.xlsx' # proton   | sigma red     | HERA II CC e+
conf['datasets']['idis']['xlsx'][10032]='idis/expdata/10032.xlsx' # proton   | sigma red     | HERA II CC e-
conf['datasets']['idis']['xlsx'][10007]='idis/expdata/10007.xlsx' # proton   | sigma red     | HERMES
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['idis']['xlsx'][10011]='idis/expdata/10011.xlsx' # deuteron | F2            | SLAC
conf['datasets']['idis']['xlsx'][10017]='idis/expdata/10017.xlsx' # deuteron | F2            | BCDMS
conf['datasets']['idis']['xlsx'][10021]='idis/expdata/10021.xlsx' # d/p      | F2d/F2p       | NMC
conf['datasets']['idis']['xlsx'][10006]='idis/expdata/10006.xlsx' # deuteron | F2            | HERMES
conf['datasets']['idis']['xlsx'][10002]='idis/expdata/10002.xlsx' # deuteron | F2            | JLab Hall C (E00-106)
conf['datasets']['idis']['xlsx'][10033]='idis/expdata/10033.xlsx' # n/d      | F2n/F2d       | BONUS
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['idis']['xlsx'][10041]='idis/expdata/10041.xlsx' # h/d      | F2h/F2d       | JLab Hall C (E03-103)
conf['datasets']['idis']['xlsx'][10050]='idis/expdata/10050.xlsx' # d/p      | F2d/F2p       | MARATHON
conf['datasets']['idis']['xlsx'][10051]='idis/expdata/10051.xlsx' # h/t      | F2h/F2t       | MARATHON
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['idis']['norm']={}
conf['datasets']['idis']['norm'][10002]={'value':    1.00000e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10003]={'value':    1.00000e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10010]={'value':    1.04352e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10011]={'value':    1.04141e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10016]={'value':    9.89544e-01, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10017]={'value':    1.01306e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10020]={'value':    1.02003e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10021]={'value':    1.00000e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10033]={'value':    1.00000e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10041]={'value':    1.00000e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10050]={'value':    1.00000e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['idis']['norm'][10051]={'value':    1.00000e+00, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}

##--PIDIS
pQ2cut=1.3**2
pW2cut=4.0

conf['datasets']['pidis']={}
conf['datasets']['pidis']['filters']=[]
conf['datasets']['pidis']['filters'].append("Q2>%f"%pQ2cut) 
conf['datasets']['pidis']['filters'].append("W2>%f"%pW2cut) 
conf['datasets']['pidis']['xlsx']={}
conf['datasets']['pidis']['norm']={}
#---------------------------------------------------------------------------------------------------------------------------
conf['datasets']['pidis']['xlsx'][10002]='pidis/expdata/10002.xlsx' # 10002 | proton   | A1   | COMPASS         |          |
conf['datasets']['pidis']['xlsx'][10003]='pidis/expdata/10003.xlsx' # 10003 | proton   | A1   | COMPASS         |          |
conf['datasets']['pidis']['xlsx'][10004]='pidis/expdata/10004.xlsx' # 10004 | proton   | A1   | EMC             |          |
conf['datasets']['pidis']['xlsx'][10005]='pidis/expdata/10005.xlsx' # 10005 | neutron  | A1   | HERMES          |          |
conf['datasets']['pidis']['xlsx'][10007]='pidis/expdata/10007.xlsx' # 10007 | proton   | Apa  | HERMES          |          |
conf['datasets']['pidis']['xlsx'][10008]='pidis/expdata/10008.xlsx' # 10008 | proton   | A2   | HERMES          |          |
conf['datasets']['pidis']['xlsx'][10017]='pidis/expdata/10017.xlsx' # 10017 | proton   | Apa  | JLabHB(EG1DVCS) |          |
conf['datasets']['pidis']['xlsx'][10022]='pidis/expdata/10022.xlsx' # 10022 | proton   | Apa  | SLAC(E143)      |          |
conf['datasets']['pidis']['xlsx'][10023]='pidis/expdata/10023.xlsx' # 10023 | proton   | Ape  | SLAC(E143)      |          |
conf['datasets']['pidis']['xlsx'][10028]='pidis/expdata/10028.xlsx' # 10028 | proton   | Ape  | SLAC(E155)      |          |
conf['datasets']['pidis']['xlsx'][10029]='pidis/expdata/10029.xlsx' # 10029 | proton   | Apa  | SLAC(E155)      |          |
conf['datasets']['pidis']['xlsx'][10031]='pidis/expdata/10031.xlsx' # 10031 | proton   | Atpe | SLAC(E155x)     |          |
conf['datasets']['pidis']['xlsx'][10032]='pidis/expdata/10032.xlsx' # 10032 | proton   | Apa  | SLACE80E130     |          |
conf['datasets']['pidis']['xlsx'][10035]='pidis/expdata/10035.xlsx' # 10035 | proton   | A1   | SMC             |          |
conf['datasets']['pidis']['xlsx'][10036]='pidis/expdata/10036.xlsx' # 10036 | proton   | A1   | SMC             |          |
conf['datasets']['pidis']['xlsx'][10043]='pidis/expdata/10043.xlsx' # 10043 | proton   | Apa  | JLabHB(EG1b)    | E =4 GeV |
conf['datasets']['pidis']['xlsx'][10044]='pidis/expdata/10044.xlsx' # 10044 | proton   | Apa  | JLabHB(EG1b)    | E =5 GeV |
#---------------------------------------------------------------------------------------------------------------------------
conf['datasets']['pidis']['xlsx'][10001]='pidis/expdata/10001.xlsx' # 10001 | deuteron | A1   | COMPASS         |          |
conf['datasets']['pidis']['xlsx'][10006]='pidis/expdata/10006.xlsx' # 10006 | deuteron | Apa  | HERMES          |          |
conf['datasets']['pidis']['xlsx'][10016]='pidis/expdata/10016.xlsx' # 10016 | deuteron | Apa  | JLabHB(EG1DVCS) |          |
conf['datasets']['pidis']['xlsx'][10020]='pidis/expdata/10020.xlsx' # 10020 | deuteron | Ape  | SLAC(E143)      |          |
conf['datasets']['pidis']['xlsx'][10021]='pidis/expdata/10021.xlsx' # 10021 | deuteron | Apa  | SLAC(E143)      |          |
conf['datasets']['pidis']['xlsx'][10026]='pidis/expdata/10026.xlsx' # 10026 | deuteron | Ape  | SLAC(E155)      |          |
conf['datasets']['pidis']['xlsx'][10027]='pidis/expdata/10027.xlsx' # 10027 | deuteron | Apa  | SLAC(E155)      |          |
conf['datasets']['pidis']['xlsx'][10030]='pidis/expdata/10030.xlsx' # 10030 | deuteron | Atpe | SLAC(E155x)     |          |
conf['datasets']['pidis']['xlsx'][10033]='pidis/expdata/10033.xlsx' # 10033 | deuteron | A1   | SMC             |          |
conf['datasets']['pidis']['xlsx'][10034]='pidis/expdata/10034.xlsx' # 10034 | deuteron | A1   | SMC             |          |
conf['datasets']['pidis']['xlsx'][10039]='pidis/expdata/10039.xlsx' # 10039 | deuteron | Apa  | JLabHB(EG1b)    | E =4 GeV |
conf['datasets']['pidis']['xlsx'][10040]='pidis/expdata/10040.xlsx' # 10040 | deuteron | Apa  | JLabHB(EG1b)    | E =5 GeV |
#---------------------------------------------------------------------------------------------------------------------------
conf['datasets']['pidis']['xlsx'][10009]='pidis/expdata/10009.xlsx' # 10009 | helium   | Apa  | JLabHA(E01-012) |          |
conf['datasets']['pidis']['xlsx'][10010]='pidis/expdata/10010.xlsx' # 10010 | helium   | Apa  | JLabHA(E06-014) |          |
conf['datasets']['pidis']['xlsx'][10011]='pidis/expdata/10011.xlsx' # 10011 | helium   | Ape  | JLabHA(E06-014) |          |
conf['datasets']['pidis']['xlsx'][10012]='pidis/expdata/10012.xlsx' # 10012 | helium   | Apa  | JLabHA(E97-103) | wrong?   |
conf['datasets']['pidis']['xlsx'][10013]='pidis/expdata/10013.xlsx' # 10013 | helium   | Ape  | JLabHA(E97-103) | wrong?   |
conf['datasets']['pidis']['xlsx'][10014]='pidis/expdata/10014.xlsx' # 10014 | helium   | Apa  | JLabHA(E99-117) |          |
conf['datasets']['pidis']['xlsx'][10015]='pidis/expdata/10015.xlsx' # 10015 | helium   | Ape  | JLabHA(E99-117) |          |
conf['datasets']['pidis']['xlsx'][10018]='pidis/expdata/10018.xlsx' # 10018 | helium   | A1   | SLAC(E142)      |          |
conf['datasets']['pidis']['xlsx'][10019]='pidis/expdata/10019.xlsx' # 10019 | helium   | A2   | SLAC(E142)      |          |
conf['datasets']['pidis']['xlsx'][10024]='pidis/expdata/10024.xlsx' # 10024 | helium   | Ape  | SLAC(E154)      |          |
conf['datasets']['pidis']['xlsx'][10025]='pidis/expdata/10025.xlsx' # 10025 | helium   | Apa  | SLAC(E154)      |          |
#---------------------------------------------------------------------------------------------------------------------------
conf['datasets']['pidis']['norm'][10002]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10003]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10004]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10005]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10007]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10022]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10023]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10029]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10031]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10032]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
#---------------------------------------------------------------------------------------------------------------------------
conf['datasets']['pidis']['norm'][10006]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10020]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10021]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10001]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10027]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10030]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
#---------------------------------------------------------------------------------------------------------------------------
conf['datasets']['pidis']['norm'][10008]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10019]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10024]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
#---------------------------------------------------------------------------------------------------------------------------
conf['datasets']['pidis']['norm'][10018]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['pidis']['norm'][10025]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}

#--sum rules
conf['datasets']['SU23']={}
conf['datasets']['SU23']['filters']=[]
conf['datasets']['SU23']['xlsx']={}
conf['datasets']['SU23']['norm']={}
#---------------------------------------------------------------------------------------------------------------------------
conf['datasets']['SU23']['xlsx'][10001]='SU23/expdata/10001.xlsx' #  SU2   | nominal 
conf['datasets']['SU23']['xlsx'][10002]='SU23/expdata/10002.xlsx' #  SU3   | nominal 
conf['datasets']['SU23']['xlsx'][20001]='SU23/expdata/20001.xlsx' #  SU2   | JAM17
conf['datasets']['SU23']['xlsx'][20002]='SU23/expdata/20002.xlsx' #  SU3   | JAM17



#--hadron-hadron reactions

##--DY 
conf['datasets']['dy']={}
conf['datasets']['dy']['filters']=[]
conf['datasets']['dy']['filters'].append("Q2>0") 
conf['datasets']['dy']['xlsx']={}
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['dy']['xlsx'][10001]='dy/expdata/10001.xlsx'
conf['datasets']['dy']['xlsx'][20001]='dy/expdata/20001.xlsx'
conf['datasets']['dy']['xlsx'][20002]='dy/expdata/20002.xlsx'
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['dy']['norm']={}
conf['datasets']['dy']['norm'][10001]={'value':    1,'fixed':False,'min':   0.5,'max':    1.5}
conf['datasets']['dy']['norm'][20001]={'value':    1,'fixed':False,'min':   0.9,'max':    1.1}
conf['datasets']['dy']['norm'][20002]={'value':    1,'fixed':False,'min':   0.9,'max':    1.1}
#------------------------------------------------------------------------------------------------------------------

##--charge asymmetry 
conf['datasets']['wzrv']={}
conf['datasets']['wzrv']['filters']=[]
conf['datasets']['wzrv']['xlsx']={}
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['wzrv']['xlsx'][1000]='wzrv/expdata/1000.xlsx'
conf['datasets']['wzrv']['xlsx'][1020]='wzrv/expdata/1020.xlsx'
conf['datasets']['wzrv']['xlsx'][1021]='wzrv/expdata/1021.xlsx'
conf['datasets']['wzrv']['xlsx'][2010]='wzrv/expdata/2010.xlsx'
conf['datasets']['wzrv']['xlsx'][2011]='wzrv/expdata/2011.xlsx'
conf['datasets']['wzrv']['xlsx'][2012]='wzrv/expdata/2012.xlsx'
conf['datasets']['wzrv']['xlsx'][2013]='wzrv/expdata/2013.xlsx'
conf['datasets']['wzrv']['xlsx'][2014]='wzrv/expdata/2014.xlsx'
conf['datasets']['wzrv']['xlsx'][2016]='wzrv/expdata/2016.xlsx'
conf['datasets']['wzrv']['xlsx'][2017]='wzrv/expdata/2017.xlsx'
conf['datasets']['wzrv']['xlsx'][2020]='wzrv/expdata/2020.xlsx'
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['wzrv']['norm']={}
conf['datasets']['wzrv']['norm'][1000]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['wzrv']['norm'][1020]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
conf['datasets']['wzrv']['norm'][1021]={'value':    1, 'min': 8.00000e-01, 'max': 1.20000e+00, 'fixed': False}
#------------------------------------------------------------------------------------------------------------------

##--W asymmetry 
conf['datasets']['wasym']={}
conf['datasets']['wasym']['filters']=[]
conf['datasets']['wasym']['xlsx']={}
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['wasym']['xlsx'][1000]='wasym/expdata/1000.xlsx'
conf['datasets']['wasym']['xlsx'][1001]='wasym/expdata/1001.xlsx'
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['wasym']['norm']={}
#------------------------------------------------------------------------------------------------------------------

##--Z rapidity 
conf['datasets']['zrap']={}
conf['datasets']['zrap']['filters']=[]
conf['datasets']['zrap']['xlsx']={}
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['zrap']['xlsx'][1000]='zrap/expdata/1000.xlsx'
conf['datasets']['zrap']['xlsx'][1001]='zrap/expdata/1001.xlsx'
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['zrap']['norm']={}
conf['datasets']['zrap']['norm'][1000]={'value':    1,'fixed':False,'min':   0.5,'max':    1.5}
#------------------------------------------------------------------------------------------------------------------

#--jets
conf['datasets']['jet'] = {}
conf['datasets']['jet']['filters'] = []
conf['datasets']['jet']['filters'].append("pT>8")
conf['datasets']['jet']['xlsx'] = {}
conf['datasets']['jet']['xlsx'][10001] = 'jets/expdata/10001.xlsx' ## D0 dataset
conf['datasets']['jet']['xlsx'][10002] = 'jets/expdata/10002.xlsx' ## CDF dataset
conf['datasets']['jet']['xlsx'][10003] = 'jets/expdata/10003.xlsx' ## STAR MB dataset
conf['datasets']['jet']['xlsx'][10004] = 'jets/expdata/10004.xlsx' ## STAR HT dataset
conf['datasets']['jet']['norm'] = {}
conf['datasets']['jet']['norm'][10001]={'value':    1,'fixed':False,'min':   0.0,'max':    2.0}
conf['datasets']['jet']['norm'][10002]={'value':    1,'fixed':False,'min':   0.0,'max':    2.0}
conf['datasets']['jet']['norm'][10003]={'value':    1,'fixed':False,'min':   0.0,'max':    2.0}
conf['datasets']['jet']['norm'][10004]={'value':    1,'fixed':False,'min':   0.0,'max':    2.0}
conf['channels'] = {'q_qp': 0, 'q_qbp': 1, 'q_q': 2, 'q_qb': 3, 'q_g': 4, 'g_g': 5}
conf['qr'] = {}
conf['qr']['parameter'] = {'epsilon': 1e-11, 'block_size': 10, 'power_scheme': 1}
conf['jet_qr_fit'] = {'method': 'fixed', 'f_scale': 0.25, 'r_scale': 0.25}
conf['parton_to_hadron'] = True

#--polarized jets
conf['datasets']['pjet'] = {}
conf['datasets']['pjet']['filters'] = []
conf['datasets']['pjet']['filters'].append("pT>8")
conf['datasets']['pjet']['xlsx'] = {}
conf['datasets']['pjet']['xlsx'][20001] = 'pjets/expdata/20001.xlsx' ## STAR 2006 paper on 2003 and 2004 data
conf['datasets']['pjet']['xlsx'][20002] = 'pjets/expdata/20002.xlsx' ## STAR 2012 paper on 2005 data
conf['datasets']['pjet']['xlsx'][20003] = 'pjets/expdata/20003.xlsx' ## STAR 2012 paper on 2006 data
conf['datasets']['pjet']['xlsx'][20004] = 'pjets/expdata/20004.xlsx' ## STAR 2015 paper on 2009 data
conf['datasets']['pjet']['xlsx'][20005] = 'pjets/expdata/20005.xlsx' ## PHENIX 2011 paper on 2005 data
conf['datasets']['pjet']['xlsx'][20006] = 'pjets/expdata/20006.xlsx' ## STAR 2019 paper on 2012 data
conf['datasets']['pjet']['xlsx'][20007] = 'pjets/expdata/20007.xlsx' ## STAR 2021 paper on 2015 data
conf['datasets']['pjet']['xlsx'][20008] = 'pjets/expdata/20008.xlsx' ## STAR 2021 paper on 2013 data
conf['datasets']['pjet']['norm'] = {}
conf['datasets']['pjet']['norm'][20002] = {'value': 1, 'fixed': False, 'min': 0.5, 'max': 1.5}
conf['datasets']['pjet']['norm'][20003] = {'value': 1, 'fixed': False, 'min': 0.5, 'max': 1.5}
conf['datasets']['pjet']['norm'][20004] = {'value': 1, 'fixed': False, 'min': 0.5, 'max': 1.5}
conf['datasets']['pjet']['norm'][20005] = {'value': 1, 'fixed': False, 'min': 0.5, 'max': 1.5}
conf['datasets']['pjet']['norm'][20006] = {'value': 1, 'fixed': False, 'min': 0.5, 'max': 1.5}
conf['datasets']['pjet']['norm'][20007] = {'value': 1, 'fixed': False, 'min': 0.5, 'max': 1.5}
conf['datasets']['pjet']['norm'][20008] = {'value': 1, 'fixed': False, 'min': 0.5, 'max': 1.5}
conf['channels'] = {'q_qp': 0, 'q_qbp': 1, 'q_q': 2, 'q_qb': 3, 'q_g': 4, 'g_g': 5}
conf['qr'] = {}
conf['qr']['parameter'] = {'epsilon': 1e-11, 'block_size': 10, 'power_scheme': 1}
conf['pjet_qr_fit'] = {'method': 'fixed', 'f_scale': 0.25, 'r_scale': 0.25}

##--PSIDIS 
conf['datasets']['psidis']={}
conf['datasets']['psidis']['filters']=[]
conf['datasets']['psidis']['filters'].append("Q2>%f"%pQ2cut) 
conf['datasets']['psidis']['filters'].append("W2>%f"%pW2cut) 
conf['datasets']['psidis']['filters'].append('Z>0.2 and Z<0.8')
conf['datasets']['psidis']['xlsx']={}
conf['datasets']['psidis']['xlsx'][20004]='psidis/expdata/20004.xlsx' # 20004 | proton   | A1pi+  | HERMES  
conf['datasets']['psidis']['xlsx'][20005]='psidis/expdata/20005.xlsx' # 20005 | proton   | A1pi-  | HERMES  
conf['datasets']['psidis']['xlsx'][20008]='psidis/expdata/20008.xlsx' # 20008 | deuteron | A1pi+  | HERMES  
conf['datasets']['psidis']['xlsx'][20009]='psidis/expdata/20009.xlsx' # 20009 | deuteron | A1pi-  | HERMES  
conf['datasets']['psidis']['xlsx'][20012]='psidis/expdata/20012.xlsx' # 20012 | deuteron | A1K+   | HERMES  
conf['datasets']['psidis']['xlsx'][20013]='psidis/expdata/20013.xlsx' # 20013 | deuteron | A1K-   | HERMES  
conf['datasets']['psidis']['xlsx'][20014]='psidis/expdata/20014.xlsx' # 20014 | deuteron | A1Ksum | HERMES  
conf['datasets']['psidis']['xlsx'][20017]='psidis/expdata/20017.xlsx' # 20017 | proton   | A1pi+  | COMPASS 
conf['datasets']['psidis']['xlsx'][20018]='psidis/expdata/20018.xlsx' # 20018 | proton   | A1pi-  | COMPASS 
conf['datasets']['psidis']['xlsx'][20019]='psidis/expdata/20019.xlsx' # 20019 | proton   | A1K+   | COMPASS 
conf['datasets']['psidis']['xlsx'][20020]='psidis/expdata/20020.xlsx' # 20020 | proton   | A1K-   | COMPASS 
conf['datasets']['psidis']['xlsx'][20021]='psidis/expdata/20021.xlsx' # 20021 | deuteron | A1pi+  | COMPASS 
conf['datasets']['psidis']['xlsx'][20022]='psidis/expdata/20022.xlsx' # 20022 | deuteron | A1pi-  | COMPASS 
conf['datasets']['psidis']['xlsx'][20025]='psidis/expdata/20025.xlsx' # 20025 | deuteron | A1K+   | COMPASS 
conf['datasets']['psidis']['xlsx'][20026]='psidis/expdata/20026.xlsx' # 20026 | deuteron | A1K-   | COMPASS 
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['psidis']['xlsx'][20000]='psidis/expdata/20000.xlsx' # 20000 | proton   | A1h+   | SMC 
conf['datasets']['psidis']['xlsx'][20001]='psidis/expdata/20001.xlsx' # 20001 | proton   | A1h-   | SMC 
conf['datasets']['psidis']['xlsx'][20002]='psidis/expdata/20002.xlsx' # 20002 | deuteron | A1h+   | SMC 
conf['datasets']['psidis']['xlsx'][20003]='psidis/expdata/20003.xlsx' # 20003 | deuteron | A1h-   | SMC 
conf['datasets']['psidis']['xlsx'][20006]='psidis/expdata/20006.xlsx' # 20006 | proton   | A1h+   | HERMES 
conf['datasets']['psidis']['xlsx'][20007]='psidis/expdata/20007.xlsx' # 20007 | proton   | A1h-   | HERMES 
conf['datasets']['psidis']['xlsx'][20010]='psidis/expdata/20010.xlsx' # 20010 | deuteron | A1h+   | HERMES 
conf['datasets']['psidis']['xlsx'][20011]='psidis/expdata/20011.xlsx' # 20011 | deuteron | A1h-   | HERMES 
conf['datasets']['psidis']['xlsx'][20015]='psidis/expdata/20015.xlsx' # 20015 | helium   | A1h+   | HERMES 
conf['datasets']['psidis']['xlsx'][20016]='psidis/expdata/20016.xlsx' # 20016 | helium   | A1h-   | HERMES 
conf['datasets']['psidis']['xlsx'][20023]='psidis/expdata/20023.xlsx' # 20023 | deuteron | A1h+   | COMPASS 
conf['datasets']['psidis']['xlsx'][20024]='psidis/expdata/20024.xlsx' # 20024 | deuteron | A1h-   | COMPASS 
#------------------------------------------------------------------------------------------------------------------
conf['datasets']['psidis']['norm']={}

#--lepton-lepton reactions

##--SIA pion

conf['datasets']['sia']={}
conf['datasets']['sia']['filters']=[]
conf['datasets']['sia']['filters'].append('z>0.2 and z<0.9') 
conf['datasets']['sia']['xlsx']={}
conf['datasets']['sia']['xlsx'][1001]='sia/expdata/1001.xlsx'  # hadron: pion exp: TASSO
conf['datasets']['sia']['xlsx'][1002]='sia/expdata/1002.xlsx'  # hadron: pion exp: TASSO
conf['datasets']['sia']['xlsx'][1003]='sia/expdata/1003.xlsx'  # hadron: pion exp: TASSO
conf['datasets']['sia']['xlsx'][1004]='sia/expdata/1004.xlsx'  # hadron: pion exp: TASSO
conf['datasets']['sia']['xlsx'][1005]='sia/expdata/1005.xlsx'  # hadron: pion exp: TASSO
conf['datasets']['sia']['xlsx'][1006]='sia/expdata/1006.xlsx'  # hadron: pion exp: TASSO
conf['datasets']['sia']['xlsx'][1007]='sia/expdata/1007.xlsx'  # hadron: pion exp: TPC
conf['datasets']['sia']['xlsx'][1008]='sia/expdata/1008.xlsx'  # hadron: pion exp: TPC
conf['datasets']['sia']['xlsx'][1012]='sia/expdata/1012.xlsx'  # hadron: pion exp: HRS
conf['datasets']['sia']['xlsx'][1013]='sia/expdata/1013.xlsx'  # hadron: pion exp: TOPAZ
conf['datasets']['sia']['xlsx'][1014]='sia/expdata/1014.xlsx'  # hadron: pion exp: SLD
conf['datasets']['sia']['xlsx'][1018]='sia/expdata/1018.xlsx'  # hadron: pion exp: ALEPH
conf['datasets']['sia']['xlsx'][1019]='sia/expdata/1019.xlsx'  # hadron: pion exp: OPAL
conf['datasets']['sia']['xlsx'][1025]='sia/expdata/1025.xlsx'  # hadron: pion exp: DELPHI
conf['datasets']['sia']['xlsx'][1028]='sia/expdata/1028.xlsx'  # hadron: pion exp: BABAR
conf['datasets']['sia']['xlsx'][1029]='sia/expdata/1029.xlsx'  # hadron: pion exp: BELL
conf['datasets']['sia']['xlsx'][1030]='sia/expdata/1030.xlsx'  # hadron: pion exp: ARGUS
conf['datasets']['sia']['norm']={}
conf['datasets']['sia']['norm'][1001]={'value':    1.10478e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1002]={'value':    9.82581e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1003]={'value':    1.03054e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1005]={'value':    1.03419e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1006]={'value':    9.79162e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1014]={'value':    9.97770e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1018]={'value':    1.02378e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1028]={'value':    9.76001e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1029]={'value':    8.68358e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1030]={'value':    1.01509e+00,'fixed':False,'min':0.5,'max':1.5}

##--SIA pion HQ
conf['datasets']['sia']['xlsx'][1010]='sia/expdata/1010.xlsx'  # hadron: pion exp: TPC(c)
conf['datasets']['sia']['xlsx'][1011]='sia/expdata/1011.xlsx'  # hadron: pion exp: TPC(b)
conf['datasets']['sia']['xlsx'][1016]='sia/expdata/1016.xlsx'  # hadron: pion exp: SLD(c)
conf['datasets']['sia']['xlsx'][1017]='sia/expdata/1017.xlsx'  # hadron: pion exp: SLD(b)
conf['datasets']['sia']['xlsx'][1023]='sia/expdata/1023.xlsx'  # hadron: pion exp: OPAL(c)
conf['datasets']['sia']['xlsx'][1024]='sia/expdata/1024.xlsx'  # hadron: pion exp: OPAL(b)
conf['datasets']['sia']['xlsx'][1027]='sia/expdata/1027.xlsx'  # hadron: pion exp: DELPHI(b)
conf['datasets']['sia']['norm'][1016]={'value':    1.18920e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1017]={'value':    1.00345e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1023]={'value':    1.33434e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][1024]={'value':    1.19151e+00,'fixed':False,'min':0.5,'max':1.5}

##--SIA kaon 
conf['datasets']['sia']['xlsx'][2030]='sia/expdata/2030.xlsx'  # hadron: kaon exp: ARGUS
conf['datasets']['sia']['xlsx'][2028]='sia/expdata/2028.xlsx'  # hadron: kaon exp: BABAR
conf['datasets']['sia']['xlsx'][2029]='sia/expdata/2029.xlsx'  # hadron: kaon exp: BELL
conf['datasets']['sia']['xlsx'][2001]='sia/expdata/2001.xlsx'  # hadron: kaon exp: TASSO
conf['datasets']['sia']['xlsx'][2002]='sia/expdata/2002.xlsx'  # hadron: kaon exp: TASSO
conf['datasets']['sia']['xlsx'][2003]='sia/expdata/2003.xlsx'  # hadron: kaon exp: TASSO
conf['datasets']['sia']['xlsx'][2004]='sia/expdata/2004.xlsx'  # hadron: kaon exp: TASSO
conf['datasets']['sia']['xlsx'][2005]='sia/expdata/2005.xlsx'  # hadron: kaon exp: TASSO
conf['datasets']['sia']['xlsx'][2006]='sia/expdata/2006.xlsx'  # hadron: kaon exp: TASSO
conf['datasets']['sia']['xlsx'][2007]='sia/expdata/2007.xlsx'  # hadron: kaon exp: TPC
conf['datasets']['sia']['xlsx'][2008]='sia/expdata/2008.xlsx'  # hadron: kaon exp: TPC
conf['datasets']['sia']['xlsx'][2012]='sia/expdata/2012.xlsx'  # hadron: kaon exp: HRS
conf['datasets']['sia']['xlsx'][2013]='sia/expdata/2013.xlsx'  # hadron: kaon exp: TOPAZ
conf['datasets']['sia']['xlsx'][2014]='sia/expdata/2014.xlsx'  # hadron: kaon exp: SLD
conf['datasets']['sia']['xlsx'][2018]='sia/expdata/2018.xlsx'  # hadron: kaon exp: ALEPH
conf['datasets']['sia']['xlsx'][2019]='sia/expdata/2019.xlsx'  # hadron: kaon exp: OPAL
conf['datasets']['sia']['xlsx'][2025]='sia/expdata/2025.xlsx'  # hadron: kaon exp: DELPHI
conf['datasets']['sia']['xlsx'][2031]='sia/expdata/2031.xlsx'  # hadron: kaon exp: DELPHI
conf['datasets']['sia']['norm'][2030]={'value':    1.00482e+00,'fixed':False,'min':0.5,'max':1.5,'dN':0.1}
conf['datasets']['sia']['norm'][2028]={'value':    9.97435e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2029]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2002]={'value':    9.83394e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2003]={'value':    9.94421e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2005]={'value':    9.92876e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2014]={'value':    9.12186e-01,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2018]={'value':    8.88453e-01,'fixed':False,'min':0.5,'max':1.5}

##--SIA kaon HQ
conf['datasets']['sia']['xlsx'][2016]='sia/expdata/2016.xlsx'  # hadron: kaon exp: SLD(c)
conf['datasets']['sia']['xlsx'][2017]='sia/expdata/2017.xlsx'  # hadron: kaon exp: SLD(b)
conf['datasets']['sia']['xlsx'][2023]='sia/expdata/2023.xlsx'  # hadron: kaon exp: OPAL(c)
conf['datasets']['sia']['xlsx'][2024]='sia/expdata/2024.xlsx'  # hadron: kaon exp: OPAL(b)
conf['datasets']['sia']['xlsx'][2027]='sia/expdata/2027.xlsx'  # hadron: kaon exp: DELPHI(b)
conf['datasets']['sia']['norm'][2016]={'value':    1.03996e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2017]={'value':    1.00015e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2023]={'value':    1.35922e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][2024]={'value':    1.38163e+00,'fixed':False,'min':0.5,'max':1.5}

##--SIA hadrons
conf['datasets']['sia']['xlsx'][4000]='sia/expdata/4000.xlsx'  # hadron: hadrons exp: ALEPH
conf['datasets']['sia']['xlsx'][4001]='sia/expdata/4001.xlsx'  # hadron: hadrons exp: DELPHI
conf['datasets']['sia']['xlsx'][4013]='sia/expdata/4013.xlsx'  # hadron: hadrons exp: DELPHI(b)
conf['datasets']['sia']['xlsx'][4002]='sia/expdata/4002.xlsx'  # hadron: hadrons exp: SLD
conf['datasets']['sia']['xlsx'][4014]='sia/expdata/4014.xlsx'  # hadron: hadrons exp: SLD(c)
conf['datasets']['sia']['xlsx'][4015]='sia/expdata/4015.xlsx'  # hadron: hadrons exp: SLD(b)
conf['datasets']['sia']['xlsx'][4003]='sia/expdata/4003.xlsx'  # hadron: hadrons exp: TASSO
conf['datasets']['sia']['xlsx'][4008]='sia/expdata/4008.xlsx'  # hadron: hadrons exp: TASSO
conf['datasets']['sia']['xlsx'][4009]='sia/expdata/4009.xlsx'  # hadron: hadrons exp: TASSO
conf['datasets']['sia']['xlsx'][4010]='sia/expdata/4010.xlsx'  # hadron: hadrons exp: TASSO
conf['datasets']['sia']['xlsx'][4011]='sia/expdata/4011.xlsx'  # hadron: hadrons exp: TASSO
conf['datasets']['sia']['xlsx'][4012]='sia/expdata/4012.xlsx'  # hadron: hadrons exp: TASSO
conf['datasets']['sia']['xlsx'][4004]='sia/expdata/4004.xlsx'  # hadron: hadrons exp: TPC
conf['datasets']['sia']['xlsx'][4005]='sia/expdata/4005.xlsx'  # hadron: hadrons exp: OPAL(b)
conf['datasets']['sia']['xlsx'][4006]='sia/expdata/4006.xlsx'  # hadron: hadrons exp: OPAL(c)
conf['datasets']['sia']['xlsx'][4007]='sia/expdata/4007.xlsx'  # hadron: hadrons exp: OPAL
conf['datasets']['sia']['norm'][4000]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4002]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4003]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4008]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4009]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4010]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4011]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4012]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4014]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sia']['norm'][4015]={'value':    1.00000e+00,'fixed':False,'min':0.5,'max':1.5}




##--SIDIS 
conf['datasets']['sidis']={}
conf['datasets']['sidis']['filters']=[]
conf['datasets']['sidis']['filters'].append("Q2>%f"%Q2cut) 
conf['datasets']['sidis']['filters'].append("W2>%f"%W2cut) 
conf['datasets']['sidis']['filters'].append('Z>0.2 and Z<0.8')
conf['datasets']['sidis']['xlsx']={}
conf['datasets']['sidis']['xlsx'][1005]='sidis/expdata/1005.xlsx' # deuteron , mult , pi+ , COMPASS
conf['datasets']['sidis']['xlsx'][1006]='sidis/expdata/1006.xlsx' # deuteron , mult , pi- , COMPASS
conf['datasets']['sidis']['xlsx'][2005]='sidis/expdata/2005.xlsx' # deuteron , mult , K+  , COMPASS
conf['datasets']['sidis']['xlsx'][2006]='sidis/expdata/2006.xlsx' # deuteron , mult , K-  , COMPASS
conf['datasets']['sidis']['xlsx'][3000]='sidis/expdata/3000.xlsx' # deuteron , mult , h+  , COMPASS
conf['datasets']['sidis']['xlsx'][3001]='sidis/expdata/3001.xlsx' # deuteron , mult , h-  , COMPASS
conf['datasets']['sidis']['norm']={}
conf['datasets']['sidis']['norm'][1005]={'value':    1.0000000,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sidis']['norm'][1006]={'value':    1.0000000,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sidis']['norm'][2005]={'value':    1.0000000,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sidis']['norm'][2006]={'value':    1.0000000,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sidis']['norm'][3000]={'value':    1.0000000,'fixed':False,'min':0.5,'max':1.5}
conf['datasets']['sidis']['norm'][3001]={'value':    1.0000000,'fixed':False,'min':0.5,'max':1.5}

#--parameters
conf['params']={}

#--pdf parameters
conf['params']['pdf']={}

conf['params']['pdf']['g1 N']    ={'value':    3.87592e-01   , 'min':  None, 'max':  None, 'fixed': True }
conf['params']['pdf']['g1 a']    ={'value':   -6.23068169e-01, 'min':  -1.9, 'max':     1, 'fixed': False}
conf['params']['pdf']['g1 b']    ={'value':    9.25741583e+00, 'min':     0, 'max':    20, 'fixed': False}
conf['params']['pdf']['g1 c']    ={'value':    0.00000000e+00, 'min':  -100, 'max':    100, 'fixed': False}
conf['params']['pdf']['g1 d']    ={'value':    0.00000000e+00, 'min':  -100, 'max':    100, 'fixed': False}

conf['params']['pdf']['uv1 N']   ={'value':    3.47549e-01   , 'min':  None, 'max':  None, 'fixed': True }
conf['params']['pdf']['uv1 a']   ={'value':   -1.21835956e-01, 'min':  -0.9, 'max':     1, 'fixed': False}
conf['params']['pdf']['uv1 b']   ={'value':    3.20766744e+00, 'min':     0, 'max':    10, 'fixed': False}
conf['params']['pdf']['uv1 c']   ={'value':    0.00000000e+00, 'min':   -100, 'max':    100, 'fixed': False}
conf['params']['pdf']['uv1 d']   ={'value':    0.00000000e+00, 'min':   -100, 'max':    100, 'fixed': False}

conf['params']['pdf']['dv1 N']   ={'value':    1.52089e-01   , 'min':  None, 'max':  None, 'fixed': True }
conf['params']['pdf']['dv1 a']   ={'value':   -2.39874967e-01, 'min':  -0.9, 'max':     1, 'fixed': False}
conf['params']['pdf']['dv1 b']   ={'value':    3.83902620e+00, 'min':     0, 'max':    10, 'fixed': False}
conf['params']['pdf']['dv1 c']   ={'value':    0.00000000e+00, 'min':   -100, 'max':    100, 'fixed': False}
conf['params']['pdf']['dv1 d']   ={'value':    0.00000000e+00, 'min':   -100, 'max':    100, 'fixed': False}

conf['params']['pdf']['mix N']    ={'value':    0.00000000e+00   , 'min':  -0.1, 'max': 5.0, 'fixed': False}
conf['params']['pdf']['mix a']    ={'value':    2.00000000e+00   , 'min':   1.0, 'max':10.0, 'fixed': False}

conf['params']['pdf']['db1 N']   ={'value':    3.67609928e-02, 'min':     0, 'max':     1, 'fixed': False}
conf['params']['pdf']['db1 a']   ={'value':   -8.41360631e-01, 'min':    -1, 'max':     1, 'fixed': False}
conf['params']['pdf']['db1 b']   ={'value':    5.31285539e+00, 'min':     0, 'max':    50, 'fixed': False}
conf['params']['pdf']['db1 c']   ={'value':    0.00000000e+00, 'min':   -100, 'max':    100, 'fixed': False}
conf['params']['pdf']['db1 d']   ={'value':    0.00000000e+00, 'min':   -100, 'max':    100, 'fixed': False}

conf['params']['pdf']['ub1 N']   ={'value':    1.95464789e-02, 'min':     0, 'max':     1, 'fixed': False}
conf['params']['pdf']['ub1 a']   ={'value':   -9.93659187e-01, 'min':    -1, 'max':     1, 'fixed': False}
conf['params']['pdf']['ub1 b']   ={'value':    8.38905814e+00, 'min':     0, 'max':    20, 'fixed': False}
conf['params']['pdf']['ub1 c']   ={'value':    0.00000000e+00, 'min':   -100, 'max':    100, 'fixed': False}
conf['params']['pdf']['ub1 d']   ={'value':    0.00000000e+00, 'min':   -100, 'max':    100, 'fixed': False}

conf['params']['pdf']['s1 N']    ={'value':       0.00000e+00, 'min':     0, 'max':     1, 'fixed': True }
conf['params']['pdf']['s1 a']    ={'value':    1.34706224e-01, 'min':  -0.5, 'max':     1, 'fixed': False}
conf['params']['pdf']['s1 b']    ={'value':    6.00759596e+00, 'min':     0, 'max':   100, 'fixed': False}

conf['params']['pdf']['sb1 N']   ={'value':    7.46109845e-07, 'min':     0, 'max':     1, 'fixed': False}
conf['params']['pdf']['sb1 a']   ={'value':    3.83495317e-01, 'min':  -0.9, 'max':     1, 'fixed': False}
conf['params']['pdf']['sb1 b']   ={'value':    4.61209808e+00, 'min':     0, 'max':    10, 'fixed': False}

conf['params']['pdf']['sea1 N']  ={'value':    5.71081196e-03, 'min':     0, 'max':     1, 'fixed': False}
conf['params']['pdf']['sea1 a']  ={'value':   -1.36329697e+00, 'min':  -1.9, 'max':    -1, 'fixed': False}
conf['params']['pdf']['sea1 b']  ={'value':    4.74721050e+00, 'min':     0, 'max':   100, 'fixed': False}

conf['params']['pdf']['sea2 N']  ={'value':       2.08640e-02, 'min':     0, 'max':     1, 'fixed': False}
conf['params']['pdf']['sea2 a']  ={'value':      -1.500000000, 'min':  -1.9, 'max':    -1, 'fixed': False}
conf['params']['pdf']['sea2 b']  ={'value':       1.00000e+01, 'min':     0, 'max':   100, 'fixed': False}

#--ppdf parameters
conf['ppdf_choice'] = 'valence'
conf['params']['ppdf'] = {}

conf['params']['ppdf']['g1 N']    = {'value':    1.54709e+00, 'min':   -10, 'max':    30, 'fixed': False}
conf['params']['ppdf']['g1 a']    = {'value':   -7.13518e-01, 'min': -0.99, 'max':     5, 'fixed': False}
conf['params']['ppdf']['g1 b']    = {'value':    9.45902e-01, 'min':     0, 'max':    30, 'fixed': False}
conf['params']['ppdf']['g1 d']    = {'value':    0.00000e+00, 'min': -1000, 'max':    10, 'fixed': False}

conf['params']['ppdf']['uv1 N']   = {'value':    3.47549e-01, 'min':   -10, 'max':    10, 'fixed': False}
conf['params']['ppdf']['uv1 a']   = {'value':    2.92830e-01, 'min': -0.99, 'max':     5, 'fixed': False}
conf['params']['ppdf']['uv1 b']   = {'value':    2.95841e+00, 'min':     0, 'max':    30, 'fixed': False}
conf['params']['ppdf']['uv1 d']   = {'value':    0.000000000, 'min':   -10, 'max':    10, 'fixed': False}

conf['params']['ppdf']['dv1 N']   = {'value':    1.52089e-01, 'min':   -10, 'max':    10, 'fixed': False}
conf['params']['ppdf']['dv1 a']   = {'value':    5.34011e-01, 'min': -0.99, 'max':     5, 'fixed': False}
conf['params']['ppdf']['dv1 b']   = {'value':    2.01125e+00, 'min':     0, 'max':    30, 'fixed': False}
conf['params']['ppdf']['dv1 d']   = {'value':    0.000000000, 'min':   -10, 'max':    10, 'fixed': False}

conf['params']['ppdf']['ub1 N']   = {'value':   -1.21705e-01, 'min':   -10, 'max':    10, 'fixed': False}
conf['params']['ppdf']['ub1 a']   = {'value':    1.36941e+00, 'min': -0.99, 'max':     5, 'fixed': False}
conf['params']['ppdf']['ub1 b']   = {'value':    9.99976e+00, 'min':     0, 'max':    30, 'fixed': False}
conf['params']['ppdf']['ub1 d']   = {'value':    0.000000000, 'min':   -10, 'max':    10, 'fixed': False}

conf['params']['ppdf']['db1 N']   = {'value':   -1.21705e-01, 'min':   -10, 'max':    10, 'fixed': False}
conf['params']['ppdf']['db1 a']   = {'value':    1.36941e+00, 'min': -0.99, 'max':     5, 'fixed': False}
conf['params']['ppdf']['db1 b']   = {'value':    9.99976e+00, 'min':     0, 'max':    30, 'fixed': False}
conf['params']['ppdf']['db1 d']   = {'value':    0.000000000, 'min':   -10, 'max':    10, 'fixed': False}

conf['params']['ppdf']['s1 N']    = {'value':   -1.21705e-01, 'min':   -10, 'max':    10, 'fixed': False}
conf['params']['ppdf']['s1 a']    = {'value':    1.36941e+00, 'min': -0.99, 'max':     5, 'fixed': False}
conf['params']['ppdf']['s1 b']    = {'value':    9.99976e+00, 'min':     0, 'max':    30, 'fixed': False}
conf['params']['ppdf']['s1 d']    = {'value':    0.000000000, 'min':   -10, 'max':    10, 'fixed': False}

conf['params']['ppdf']['sb1 N']   = {'value':   -1.21705e-01, 'min':   -10, 'max':    10, 'fixed': False}
conf['params']['ppdf']['sb1 a']   = {'value':    1.36941e+00, 'min': -0.99, 'max':     5, 'fixed': False}
conf['params']['ppdf']['sb1 b']   = {'value':    9.99976e+00, 'min':     0, 'max':    30, 'fixed': False}
conf['params']['ppdf']['sb1 d']   = {'value':    0.000000000, 'min':   -10, 'max':    10, 'fixed': False}

conf['params']['ppdf']['sea1 N']  = {'value':   -1.21705e-01, 'min':   -10, 'max':    10, 'fixed': False}
conf['params']['ppdf']['sea1 a']  = {'value':    1.36941e+00, 'min': -0.99, 'max':     5, 'fixed': False}
conf['params']['ppdf']['sea1 b']  = {'value':    9.99976e+00, 'min':     0, 'max':    30, 'fixed': False}
conf['params']['ppdf']['sea1 d']  = {'value':    0.000000000, 'min':   -10, 'max':    10, 'fixed': False}


#--ht params (NOTE: What's being fitted here is the (M^2/Q^2) and (m^2/Q^2) correction H. To get the correction to FX, take H/Q^2)

conf['params']['ht4']={}

conf['params']['ht4']['F2p N']  ={'value':   0.0,   'min': -30.0,  'max': 30.0, 'fixed':False}
conf['params']['ht4']['F2p a']  ={'value':   1.5,   'min':  0.0,   'max': 4,    'fixed':False}
conf['params']['ht4']['F2p b']  ={'value':   0.0,   'min':  None,  'max': None, 'fixed':True}
conf['params']['ht4']['F2p c']  ={'value':   0.0,   'min': -10.0,  'max': 10,   'fixed':True}
conf['params']['ht4']['F2p d']  ={'value':  -5.0,   'min': -10.0,  'max': -1.0, 'fixed':False}

conf['params']['ht4']['FLp N']  ={'value':   0.0,   'min': -30.0,  'max': 30.0, 'fixed':'F2p N'}
conf['params']['ht4']['FLp a']  ={'value':   1.5,   'min':  0.0,   'max': 4,    'fixed':'F2p a'}
conf['params']['ht4']['FLp b']  ={'value':   0.0,   'min':  None,  'max': None, 'fixed':'F2p b'}
conf['params']['ht4']['FLp c']  ={'value':   0.0,   'min': -10.0,  'max': 10,   'fixed':'F2p c'}
conf['params']['ht4']['FLp d']  ={'value':  -5.0,   'min': -10.0,  'max': -1.0, 'fixed':'F2p d'}

conf['params']['ht4']['F3p N']  ={'value':   0.0,   'min': -30.0,  'max': 30.0, 'fixed':'F2p N'}
conf['params']['ht4']['F3p a']  ={'value':   1.5,   'min':  0.0,   'max': 4,    'fixed':'F2p a'}
conf['params']['ht4']['F3p b']  ={'value':   0.0,   'min':  None,  'max': None, 'fixed':'F2p b'}
conf['params']['ht4']['F3p c']  ={'value':   0.0,   'min': -10.0,  'max': 10,   'fixed':'F2p c'}
conf['params']['ht4']['F3p d']  ={'value':  -5.0,   'min': -10.0,  'max': -1.0, 'fixed':'F2p d'}

#--for p != n
conf['params']['ht4']['F2n N']  ={'value':   0.0,   'min': -30.0,  'max': 30.0, 'fixed':False}
conf['params']['ht4']['F2n a']  ={'value':   1.5,   'min':  0.0,   'max': 4,    'fixed':False}
conf['params']['ht4']['F2n b']  ={'value':   0.0,   'min':  None,  'max': None, 'fixed':True}
conf['params']['ht4']['F2n c']  ={'value':   0.0,   'min': -10.0,  'max': 10,   'fixed':True}
conf['params']['ht4']['F2n d']  ={'value':  -5.0,   'min':  -10.0, 'max': -1.0, 'fixed':False}

conf['params']['ht4']['FLn N']  ={'value':   0.0,   'min': -30.0,  'max': 30.0, 'fixed':'F2n N'}
conf['params']['ht4']['FLn a']  ={'value':   1.5,   'min':  0.0,   'max': 4,    'fixed':'F2n a'}
conf['params']['ht4']['FLn b']  ={'value':   0.0,   'min':  None,  'max': None, 'fixed':'F2n b'}
conf['params']['ht4']['FLn c']  ={'value':   0.0,   'min': -10.0,  'max': 10,   'fixed':'F2n c'}
conf['params']['ht4']['FLn d']  ={'value':  -5.0,   'min':  -10.0, 'max': -1.0, 'fixed':'F2n d'}

#--offshell paramaterization
conf['params']['off pdf']={}

conf['params']['off pdf']['uv1 N']   ={'value':   0.0, 'min': -10.0, 'max':  10.0, 'fixed': False}
conf['params']['off pdf']['uv1 a']   ={'value':   0.0, 'min':  -0.9, 'max':     5, 'fixed': False}
conf['params']['off pdf']['uv1 b']   ={'value':   3.0, 'min':   0.0, 'max':    20, 'fixed': False}
conf['params']['off pdf']['uv1 c']   ={'value':   0.0, 'min': -10.0, 'max':  10.0, 'fixed': True}
conf['params']['off pdf']['uv1 d']   ={'value':   0.0, 'min':  None, 'max':  None, 'fixed': True}

conf['params']['off pdf']['dv1 N']   ={'value':   0.0, 'min': -10.0, 'max':  10.0, 'fixed': False}
conf['params']['off pdf']['dv1 a']   ={'value':   0.0, 'min':  -0.9, 'max':     5, 'fixed': False}
conf['params']['off pdf']['dv1 b']   ={'value':   3.0, 'min':   0.0, 'max':    20, 'fixed': False}
conf['params']['off pdf']['dv1 c']   ={'value':   0.0, 'min': -10.0, 'max':  10.0, 'fixed': True}
conf['params']['off pdf']['dv1 d']   ={'value':   0.0, 'min':  None, 'max':  None, 'fixed': True}

#--pion fragmentation
conf['ffpion parametrization']=0
conf['params']['ffpion']={}

conf['params']['ffpion']['g1 N']  ={'value':    2.95437e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffpion']['g1 a']  ={'value':    1.00469e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffpion']['g1 b']  ={'value':    6.85766e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                
conf['params']['ffpion']['u1 N']  ={'value':    2.67821e-02, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffpion']['u1 a']  ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffpion']['u1 b']  ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffpion']['d1 N']  ={'value':    2.99974e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffpion']['d1 a']  ={'value':   -6.89477e-01, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffpion']['d1 b']  ={'value':    4.79992e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffpion']['s1 N']  ={'value':    1.54863e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffpion']['s1 a']  ={'value':    3.00305e-01, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffpion']['s1 b']  ={'value':    1.83178e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                  
conf['params']['ffpion']['c1 N']  ={'value':    1.84550e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffpion']['c1 a']  ={'value':   -5.05798e-02, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffpion']['c1 b']  ={'value':    3.19952e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffpion']['b1 N']  ={'value':    3.74125e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffpion']['b1 a']  ={'value':   -1.59541e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffpion']['b1 b']  ={'value':    4.50102e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffpion']['ub1 N'] ={'value':    2.99974e-01, 'min':  0.0 , 'max':     1,'fixed':'d1 N'}
conf['params']['ffpion']['ub1 a'] ={'value':   -6.89477e-01, 'min': -1.8 , 'max':    10,'fixed':'d1 a'}
conf['params']['ffpion']['ub1 b'] ={'value':    4.79992e+00, 'min':  0   , 'max':    30,'fixed':'d1 b'}
                                                                                                  
conf['params']['ffpion']['db1 N'] ={'value':    2.67821e-02, 'min':  0.0 , 'max':     1,'fixed':'u1 N'}
conf['params']['ffpion']['db1 a'] ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':'u1 a'}
conf['params']['ffpion']['db1 b'] ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':'u1 b'}
                                                                                                  
conf['params']['ffpion']['sb1 N'] ={'value':    1.54863e-01, 'min':  0.0 , 'max':     1,'fixed':'s1 N'}
conf['params']['ffpion']['sb1 a'] ={'value':    3.00305e-01, 'min': -1.8 , 'max':    10,'fixed':'s1 a'}
conf['params']['ffpion']['sb1 b'] ={'value':    1.83178e+00, 'min':  0   , 'max':    30,'fixed':'s1 b'}
                                                                                                  
conf['params']['ffpion']['cb1 N'] ={'value':    1.84550e-01, 'min':  0.0 , 'max':     1,'fixed':'c1 N'}
conf['params']['ffpion']['cb1 a'] ={'value':   -5.05798e-02, 'min': -1.8 , 'max':    10,'fixed':'c1 a'}
conf['params']['ffpion']['cb1 b'] ={'value':    3.19952e+00, 'min':  0   , 'max':    30,'fixed':'c1 b'}
                                                                                                  
conf['params']['ffpion']['bb1 N'] ={'value':    3.74125e-01, 'min':  0.0 , 'max':     1,'fixed':'b1 N'}
conf['params']['ffpion']['bb1 a'] ={'value':   -1.59541e+00, 'min': -1.8 , 'max':    10,'fixed':'b1 a'}
conf['params']['ffpion']['bb1 b'] ={'value':    4.50102e+00, 'min':  0   , 'max':    30,'fixed':'b1 b'}
                                   
conf['params']['ffpion']['u2 N']  ={'value':    0.00000e+00, 'min': -0.01, 'max':    10,'fixed':False}
conf['params']['ffpion']['u2 a']  ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffpion']['u2 b']  ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                
conf['params']['ffpion']['d2 N']  ={'value':    0.00000e+00, 'min': -0.01, 'max':    10,'fixed':False}
conf['params']['ffpion']['d2 a']  ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffpion']['d2 b']  ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffpion']['ub2 N'] ={'value':    0.00000e+00, 'min': -0.01, 'max':    10,'fixed':'d2 N'}
conf['params']['ffpion']['ub2 a'] ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':'d2 a'}
conf['params']['ffpion']['ub2 b'] ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':'d2 b'}
                                                                       
conf['params']['ffpion']['db2 N'] ={'value':    0.00000e+00, 'min': -0.01, 'max':    10,'fixed':'u2 N'}
conf['params']['ffpion']['db2 a'] ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':'u2 a'}
conf['params']['ffpion']['db2 b'] ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':'u2 b'}
                                                                                           

#--kaon fragmentation
conf['params']['ffkaon']={}
conf['ffkaon parametrization']=0

conf['params']['ffkaon']['g1 N']  ={'value':   2.33320e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffkaon']['g1 a']  ={'value':   1.48737e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['g1 b']  ={'value':   9.62755e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                
conf['params']['ffkaon']['u1 N']  ={'value':   4.03672e-02, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffkaon']['u1 a']  ={'value':   1.26356e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['u1 b']  ={'value':   1.62596e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffkaon']['d1 N']  ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffkaon']['d1 a']  ={'value':  -1.43444e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['d1 b']  ={'value':   5.65143e+00, 'min':  5   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffkaon']['s1 N']  ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffkaon']['s1 a']  ={'value':  -1.43444e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['s1 b']  ={'value':   5.65143e+00, 'min':  5   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffkaon']['c1 N']  ={'value':   7.76923e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffkaon']['c1 a']  ={'value':  -1.80000e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['c1 b']  ={'value':   2.50452e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffkaon']['b1 N']  ={'value':   5.66971e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffkaon']['b1 a']  ={'value':  -1.80000e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['b1 b']  ={'value':   3.41727e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                   
conf['params']['ffkaon']['ub1 N'] ={'value':   4.03672e-02, 'min':  0.0 , 'max':     1,'fixed':'d1 N'}
conf['params']['ffkaon']['ub1 a'] ={'value':   2.26356e+00, 'min': -1.8 , 'max':    10,'fixed':'d1 a'}
conf['params']['ffkaon']['ub1 b'] ={'value':   1.62596e+00, 'min':  5   , 'max':    30,'fixed':'d1 b'}
                                                                                               
conf['params']['ffkaon']['db1 N'] ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':'d1 N'}
conf['params']['ffkaon']['db1 a'] ={'value':  -1.43444e+00, 'min': -1.8 , 'max':    10,'fixed':'d1 a'}
conf['params']['ffkaon']['db1 b'] ={'value':   1.65143e+00, 'min':  5   , 'max':    30,'fixed':'d1 b'}
                                                                                               
conf['params']['ffkaon']['sb1 N'] ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffkaon']['sb1 a'] ={'value':  -1.43444e+00, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['sb1 b'] ={'value':   1.65143e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                  
conf['params']['ffkaon']['cb1 N'] ={'value':   7.76923e-01, 'min':  0.0 , 'max':     1,'fixed':'c1 N'}
conf['params']['ffkaon']['cb1 a'] ={'value':  -1.80000e+00, 'min': -1.8 , 'max':    10,'fixed':'c1 a'}
conf['params']['ffkaon']['cb1 b'] ={'value':   2.50452e+00, 'min':  0   , 'max':    30,'fixed':'c1 b'}
                                                                                                  
conf['params']['ffkaon']['bb1 N'] ={'value':   5.66971e-01, 'min':  0.0 , 'max':     1,'fixed':'b1 N'}
conf['params']['ffkaon']['bb1 a'] ={'value':  -1.80000e+00, 'min': -1.8 , 'max':    10,'fixed':'b1 a'}
conf['params']['ffkaon']['bb1 b'] ={'value':   3.41727e+00, 'min':  0   , 'max':    30,'fixed':'b1 b'}

conf['params']['ffkaon']['u2 N']  ={'value':    0.00000e+00, 'min': -0.01, 'max':    10,'fixed':False}
conf['params']['ffkaon']['u2 a']  ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['u2 b']  ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                              
conf['params']['ffkaon']['d2 N']  ={'value':    0.00000e+00, 'min': -0.01, 'max':    10,'fixed':False}
conf['params']['ffkaon']['d2 a']  ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':False}
conf['params']['ffkaon']['d2 b']  ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':False}
                                                                                                 
conf['params']['ffkaon']['ub2 N'] ={'value':    0.00000e+00, 'min': -0.01, 'max':    10,'fixed':'d2 N'}
conf['params']['ffkaon']['ub2 a'] ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':'d2 a'}
conf['params']['ffkaon']['ub2 b'] ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':'d2 b'}
                                                                     
conf['params']['ffkaon']['db2 N'] ={'value':    0.00000e+00, 'min': -0.01, 'max':    10,'fixed':'d2 N'}
conf['params']['ffkaon']['db2 a'] ={'value':    1.76877e-01, 'min': -1.8 , 'max':    10,'fixed':'d2 a'}
conf['params']['ffkaon']['db2 b'] ={'value':    4.81521e+00, 'min':  0   , 'max':    30,'fixed':'d2 b'}

#--unidentified hadrons
conf['params']['ffhadron']={}
conf['ffhadron parametrization']='sum2'

conf['params']['ffhadron']['g1 N']  ={'value':   2.33320e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffhadron']['g1 a']  ={'value':   1.48737e+00, 'min': -1.8 , 'max':     2,'fixed':False}
conf['params']['ffhadron']['g1 b']  ={'value':   9.62755e+00, 'min':  0   , 'max':    10,'fixed':False}

conf['params']['ffhadron']['u1 N']  ={'value':   4.03672e-02, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffhadron']['u1 a']  ={'value':   1.26356e+00, 'min': -1.8 , 'max':     2,'fixed':False}
conf['params']['ffhadron']['u1 b']  ={'value':   1.62596e+00, 'min':  0   , 'max':    10,'fixed':False}
conf['params']['ffhadron']['u1 c']  ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':False}
conf['params']['ffhadron']['u1 d']  ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':False}

conf['params']['ffhadron']['d1 N']  ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':'u1 N'}
conf['params']['ffhadron']['d1 a']  ={'value':  -1.43444e+00, 'min': -1.8 , 'max':     2,'fixed':'u1 a'}
conf['params']['ffhadron']['d1 b']  ={'value':   1.65143e+00, 'min':  0   , 'max':    10,'fixed':'u1 b'}
conf['params']['ffhadron']['d1 c']  ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 c'}
conf['params']['ffhadron']['d1 d']  ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 d'}

conf['params']['ffhadron']['s1 N']  ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':'u1 N'}
conf['params']['ffhadron']['s1 a']  ={'value':  -1.43444e+00, 'min': -1.8 , 'max':     2,'fixed':'u1 a'}
conf['params']['ffhadron']['s1 b']  ={'value':   1.65143e+00, 'min':  0   , 'max':    10,'fixed':'u1 b'}
conf['params']['ffhadron']['s1 c']  ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 c'}
conf['params']['ffhadron']['s1 d']  ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 d'}

conf['params']['ffhadron']['c1 N']  ={'value':   7.76923e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffhadron']['c1 a']  ={'value':  -1.80000e+00, 'min': -1.8 , 'max':     2,'fixed':False}
conf['params']['ffhadron']['c1 b']  ={'value':   2.50452e+00, 'min':  0   , 'max':    10,'fixed':False}

conf['params']['ffhadron']['b1 N']  ={'value':   5.66971e-01, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffhadron']['b1 a']  ={'value':  -1.80000e+00, 'min': -1.8 , 'max':     2,'fixed':False}
conf['params']['ffhadron']['b1 b']  ={'value':   3.41727e+00, 'min':  0   , 'max':    10,'fixed':False}

conf['params']['ffhadron']['ub1 N'] ={'value':   4.03672e-02, 'min':  0.0 , 'max':     1,'fixed':False}
conf['params']['ffhadron']['ub1 a'] ={'value':   2.26356e+00, 'min': -1.8 , 'max':     2,'fixed':'u1 a'}
conf['params']['ffhadron']['ub1 b'] ={'value':   1.62596e+00, 'min':  0   , 'max':    10,'fixed':False}
conf['params']['ffhadron']['ub1 c'] ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 c'}
conf['params']['ffhadron']['ub1 d'] ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 d'}

conf['params']['ffhadron']['db1 N'] ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':'ub1 N'}
conf['params']['ffhadron']['db1 a'] ={'value':  -1.43444e+00, 'min': -1.8 , 'max':     2,'fixed':'u1 a'}
conf['params']['ffhadron']['db1 b'] ={'value':   1.65143e+00, 'min':  0   , 'max':    10,'fixed':'ub1 b'}
conf['params']['ffhadron']['db1 c'] ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 c'}
conf['params']['ffhadron']['db1 d'] ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 d'}

conf['params']['ffhadron']['sb1 N'] ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':'ub1 N'}
conf['params']['ffhadron']['sb1 a'] ={'value':  -1.43444e+00, 'min': -1.8 , 'max':     2,'fixed':'u1 a'}
conf['params']['ffhadron']['sb1 b'] ={'value':   1.65143e+00, 'min':  0   , 'max':    10,'fixed':'ub1 b'}
conf['params']['ffhadron']['sb1 c'] ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 c'}
conf['params']['ffhadron']['sb1 d'] ={'value':   0.00000e+00, 'min': -10  , 'max':    10,'fixed':'u1 d'}

conf['params']['ffhadron']['cb1 N'] ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':'c1 N'}
conf['params']['ffhadron']['cb1 a'] ={'value':  -1.43444e+00, 'min': -1.8 , 'max':     2,'fixed':'c1 a'}
conf['params']['ffhadron']['cb1 b'] ={'value':   1.65143e+00, 'min':  0   , 'max':    10,'fixed':'c1 b'}

conf['params']['ffhadron']['bb1 N'] ={'value':   2.51671e-01, 'min':  0.0 , 'max':     1,'fixed':'b1 N'}
conf['params']['ffhadron']['bb1 a'] ={'value':  -1.43444e+00, 'min': -1.8 , 'max':     2,'fixed':'b1 a'}
conf['params']['ffhadron']['bb1 b'] ={'value':   1.65143e+00, 'min':  0   , 'max':    10,'fixed':'b1 b'}



#--steps
conf['steps']={}

istep=41
#--lower W2 cut (PDIS W2 > 4, Q2 > mc2 (w/ smearing, valence parameterization, OS+ for all, g2 = 0))
conf['ftol']=1e-6
conf['tmc']     = 'AOT'
conf['ht']      = True
conf['ht type'] = 'mult'
conf['offpdf']  = True
conf['pidis nuc']      = True
conf['pidis tmc']      = False
conf['pidis ht']       = False
conf['pidis twist3']   = False
conf['pidis ht type']  = 'add'
conf['ww']  = False
conf['steps'][istep]={}
conf['steps'][istep]['dep']=[19,39,40]
conf['steps'][istep]['active distributions']=['ppdf']
conf['steps'][istep]['passive distributions']=['pdf','ht4','off pdf','ffpion','ffkaon','ffhadron']
#------------------------------------------------------------------------------------------------------------------
conf['steps'][istep]['datasets']={}
conf['steps'][istep]['datasets']['pidis']=[]
conf['steps'][istep]['datasets']['pidis'].append(10002) # 10002 | proton   | A1   | COMPASS         |          |
conf['steps'][istep]['datasets']['pidis'].append(10003) # 10003 | proton   | A1   | COMPASS         |          |
conf['steps'][istep]['datasets']['pidis'].append(10004) # 10004 | proton   | A1   | EMC             |          |
conf['steps'][istep]['datasets']['pidis'].append(10007) # 10007 | proton   | Apa  | HERMES          |          |
conf['steps'][istep]['datasets']['pidis'].append(10022) # 10022 | proton   | Apa  | SLAC(E143)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10029) # 10029 | proton   | Apa  | SLAC(E155)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10032) # 10032 | proton   | Apa  | SLACE80E130     |          |
conf['steps'][istep]['datasets']['pidis'].append(10035) # 10035 | proton   | A1   | SMC             |          |
conf['steps'][istep]['datasets']['pidis'].append(10036) # 10036 | proton   | A1   | SMC             |          |
conf['steps'][istep]['datasets']['pidis'].append(10005) # 10005 | neutron  | A1   | HERMES          |          |
conf['steps'][istep]['datasets']['pidis'].append(10001) # 10001 | deuteron | A1   | COMPASS         |          |
conf['steps'][istep]['datasets']['pidis'].append(10006) # 10006 | deuteron | Apa  | HERMES          |          |
conf['steps'][istep]['datasets']['pidis'].append(10021) # 10021 | deuteron | Apa  | SLAC(E143)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10027) # 10027 | deuteron | Apa  | SLAC(E155)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10033) # 10033 | deuteron | A1   | SMC             |          |
conf['steps'][istep]['datasets']['pidis'].append(10034) # 10034 | deuteron | A1   | SMC             |          |
conf['steps'][istep]['datasets']['pidis'].append(10016) # 10016 | deuteron | Apa  | JLabHB(EG1DVCS) |          |
conf['steps'][istep]['datasets']['pidis'].append(10017) # 10017 | proton   | Apa  | JLabHB(EG1DVCS) |          |
conf['steps'][istep]['datasets']['pidis'].append(10039) # 10039 | deuteron | Apa  | JLabHB(EG1b)    | E =4 GeV |
conf['steps'][istep]['datasets']['pidis'].append(10040) # 10040 | deuteron | Apa  | JLabHB(EG1b)    | E =5 GeV |
conf['steps'][istep]['datasets']['pidis'].append(10043) # 10043 | proton   | Apa  | JLabHB(EG1b)    | E =4 GeV |
conf['steps'][istep]['datasets']['pidis'].append(10044) # 10044 | proton   | Apa  | JLabHB(EG1b)    | E =5 GeV |
conf['steps'][istep]['datasets']['pidis'].append(10010) # 10010 | helium   | Apa  | JLabHA(E06-014) |          |
conf['steps'][istep]['datasets']['pidis'].append(10014) # 10014 | helium   | Apa  | JLabHA(E99-117) |          |
conf['steps'][istep]['datasets']['pidis'].append(10018) # 10018 | helium   | A1   | SLAC(E142)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10025) # 10025 | helium   | Apa  | SLAC(E154)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10008) # 10008 | proton   | A2   | HERMES          |          |
conf['steps'][istep]['datasets']['pidis'].append(10023) # 10023 | proton   | Ape  | SLAC(E143)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10028) # 10028 | proton   | Ape  | SLAC(E155)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10031) # 10031 | proton   | Atpe | SLAC(E155x)     |          |
conf['steps'][istep]['datasets']['pidis'].append(10020) # 10020 | deuteron | Ape  | SLAC(E143)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10026) # 10026 | deuteron | Ape  | SLAC(E155)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10030) # 10030 | deuteron | Atpe | SLAC(E155x)     |          |
conf['steps'][istep]['datasets']['pidis'].append(10011) # 10011 | helium   | Ape  | JLabHA(E06-014) |          |
conf['steps'][istep]['datasets']['pidis'].append(10015) # 10015 | helium   | Ape  | JLabHA(E99-117) |          |
conf['steps'][istep]['datasets']['pidis'].append(10019) # 10019 | helium   | A2   | SLAC(E142)      |          |
conf['steps'][istep]['datasets']['pidis'].append(10024) # 10024 | helium   | Ape  | SLAC(E154)      |          |
conf['steps'][istep]['datasets']['pjet'] = []
conf['steps'][istep]['datasets']['pjet'].append(20001) ## STAR 2006 paper on 2003 and 2004 data
conf['steps'][istep]['datasets']['pjet'].append(20002) ## STAR 2012 paper on 2005 data
conf['steps'][istep]['datasets']['pjet'].append(20003) ## STAR 2012 paper on 2006 data
conf['steps'][istep]['datasets']['pjet'].append(20004) ## STAR 2015 paper on 2009 data
conf['steps'][istep]['datasets']['pjet'].append(20005) ## PHENIX 2011 paper on 2005 data
conf['steps'][istep]['datasets']['pjet'].append(20006) ## STAR 2019 paper on 2012 data
conf['steps'][istep]['datasets']['pjet'].append(20007) ## STAR 2021 paper on 2015 data
conf['steps'][istep]['datasets']['pjet'].append(20008) ## STAR 2021 paper on 2013 data
conf['steps'][istep]['datasets']['wzrv'] = []
conf['steps'][istep]['datasets']['wzrv'].append(1000) 
conf['steps'][istep]['datasets']['wzrv'].append(1020) 
conf['steps'][istep]['datasets']['wzrv'].append(1021) 
#conf['steps'][istep]['datasets']['sia']=[]
#conf['steps'][istep]['datasets']['sia'].append(1001)  # hadron: pion exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(1002)  # hadron: pion exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(1003)  # hadron: pion exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(1004)  # hadron: pion exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(1005)  # hadron: pion exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(1006)  # hadron: pion exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(1007)  # hadron: pion exp: TPC
#conf['steps'][istep]['datasets']['sia'].append(1008)  # hadron: pion exp: TPC
#conf['steps'][istep]['datasets']['sia'].append(1012)  # hadron: pion exp: HRS
#conf['steps'][istep]['datasets']['sia'].append(1013)  # hadron: pion exp: TOPAZ
#conf['steps'][istep]['datasets']['sia'].append(1014)  # hadron: pion exp: SLD
#conf['steps'][istep]['datasets']['sia'].append(1018)  # hadron: pion exp: ALEPH
#conf['steps'][istep]['datasets']['sia'].append(1019)  # hadron: pion exp: OPAL
#conf['steps'][istep]['datasets']['sia'].append(1025)  # hadron: pion exp: DELPHI
#conf['steps'][istep]['datasets']['sia'].append(1028)  # hadron: pion exp: BABAR
#conf['steps'][istep]['datasets']['sia'].append(1029)  # hadron: pion exp: BELL
#conf['steps'][istep]['datasets']['sia'].append(1030)  # hadron: pion exp: ARGUS
#conf['steps'][istep]['datasets']['sia'].append(1010)  # hadron: pion exp: TPC(c)
#conf['steps'][istep]['datasets']['sia'].append(1011)  # hadron: pion exp: TPC(b)
#conf['steps'][istep]['datasets']['sia'].append(1016)  # hadron: pion exp: SLD(c)
#conf['steps'][istep]['datasets']['sia'].append(1017)  # hadron: pion exp: SLD(b)
#conf['steps'][istep]['datasets']['sia'].append(1023)  # hadron: pion exp: OPAL(c)
#conf['steps'][istep]['datasets']['sia'].append(1024)  # hadron: pion exp: OPAL(b)
#conf['steps'][istep]['datasets']['sia'].append(1027)  # hadron: pion exp: DELPHI(b)
#conf['steps'][istep]['datasets']['sia'].append(2001)  # hadron: kaon exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(2002)  # hadron: kaon exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(2003)  # hadron: kaon exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(2004)  # hadron: kaon exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(2005)  # hadron: kaon exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(2006)  # hadron: kaon exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(2007)  # hadron: kaon exp: TPC
#conf['steps'][istep]['datasets']['sia'].append(2008)  # hadron: kaon exp: TPC
#conf['steps'][istep]['datasets']['sia'].append(2012)  # hadron: kaon exp: HRS
#conf['steps'][istep]['datasets']['sia'].append(2013)  # hadron: kaon exp: TOPAZ
#conf['steps'][istep]['datasets']['sia'].append(2014)  # hadron: kaon exp: SLD
#conf['steps'][istep]['datasets']['sia'].append(2016)  # hadron: kaon exp: SLD(c)
#conf['steps'][istep]['datasets']['sia'].append(2017)  # hadron: kaon exp: SLD(b)
#conf['steps'][istep]['datasets']['sia'].append(2018)  # hadron: kaon exp: ALEPH
#conf['steps'][istep]['datasets']['sia'].append(2019)  # hadron: kaon exp: OPAL
#conf['steps'][istep]['datasets']['sia'].append(2023)  # hadron: kaon exp: OPAL(c)
#conf['steps'][istep]['datasets']['sia'].append(2024)  # hadron: kaon exp: OPAL(b)
#conf['steps'][istep]['datasets']['sia'].append(2025)  # hadron: kaon exp: DELPHI
#conf['steps'][istep]['datasets']['sia'].append(2027)  # hadron: kaon exp: DELPHI(b)
#conf['steps'][istep]['datasets']['sia'].append(2028)  # hadron: kaon exp: BABAR
#conf['steps'][istep]['datasets']['sia'].append(2029)  # hadron: kaon exp: BELL
#conf['steps'][istep]['datasets']['sia'].append(2030)  # hadron: kaon exp: ARGUS
#conf['steps'][istep]['datasets']['sia'].append(2031)  # hadron: kaon exp: DELPHI
#conf['steps'][istep]['datasets']['sia'].append(4000)  # hadron: hadrons exp: ALEPH
#conf['steps'][istep]['datasets']['sia'].append(4001)  # hadron: hadrons exp: DELPHI
#conf['steps'][istep]['datasets']['sia'].append(4002)  # hadron: hadrons exp: SLD
#conf['steps'][istep]['datasets']['sia'].append(4004)  # hadron: hadrons exp: TPC
#conf['steps'][istep]['datasets']['sia'].append(4005)  # hadron: hadrons exp: OPAL(b)
#conf['steps'][istep]['datasets']['sia'].append(4006)  # hadron: hadrons exp: OPAL(c)
#conf['steps'][istep]['datasets']['sia'].append(4007)  # hadron: hadrons exp: OPAL
#conf['steps'][istep]['datasets']['sia'].append(4011)  # hadron: hadrons exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(4012)  # hadron: hadrons exp: TASSO
#conf['steps'][istep]['datasets']['sia'].append(4013)  # hadron: hadrons exp: DELPHI(b)
#conf['steps'][istep]['datasets']['sia'].append(4014)  # hadron: hadrons exp: SLD(c)
#conf['steps'][istep]['datasets']['sia'].append(4015)  # hadron: hadrons exp: SLD(b)
conf['steps'][istep]['datasets']['psidis']=[]
conf['steps'][istep]['datasets']['psidis'].append(20004) # 20004 | proton   | A1pi+  | HERMES  
conf['steps'][istep]['datasets']['psidis'].append(20005) # 20005 | proton   | A1pi-  | HERMES  
conf['steps'][istep]['datasets']['psidis'].append(20008) # 20008 | deuteron | A1pi+  | HERMES  
conf['steps'][istep]['datasets']['psidis'].append(20009) # 20009 | deuteron | A1pi-  | HERMES  
conf['steps'][istep]['datasets']['psidis'].append(20017) # 20017 | proton   | A1pi+  | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20018) # 20018 | proton   | A1pi-  | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20021) # 20021 | deuteron | A1pi+  | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20022) # 20022 | deuteron | A1pi-  | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20012) # 20012 | deuteron | A1K+   | HERMES  
conf['steps'][istep]['datasets']['psidis'].append(20013) # 20013 | deuteron | A1K-   | HERMES  
conf['steps'][istep]['datasets']['psidis'].append(20014) # 20014 | deuteron | A1Ksum | HERMES  
conf['steps'][istep]['datasets']['psidis'].append(20019) # 20019 | proton   | A1K+   | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20020) # 20020 | proton   | A1K-   | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20025) # 20025 | deuteron | A1K+   | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20026) # 20026 | deuteron | A1K-   | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20000) # 20000 | proton   | A1h+   | SMC 
conf['steps'][istep]['datasets']['psidis'].append(20001) # 20001 | proton   | A1h-   | SMC 
conf['steps'][istep]['datasets']['psidis'].append(20002) # 20002 | deuteron | A1h+   | SMC 
conf['steps'][istep]['datasets']['psidis'].append(20003) # 20003 | deuteron | A1h-   | SMC 
conf['steps'][istep]['datasets']['psidis'].append(20006) # 20006 | proton   | A1h+   | HERMES 
conf['steps'][istep]['datasets']['psidis'].append(20007) # 20007 | proton   | A1h-   | HERMES 
conf['steps'][istep]['datasets']['psidis'].append(20010) # 20010 | deuteron | A1h+   | HERMES 
conf['steps'][istep]['datasets']['psidis'].append(20011) # 20011 | deuteron | A1h-   | HERMES 
conf['steps'][istep]['datasets']['psidis'].append(20015) # 20015 | helium   | A1h+   | HERMES 
conf['steps'][istep]['datasets']['psidis'].append(20016) # 20016 | helium   | A1h-   | HERMES 
conf['steps'][istep]['datasets']['psidis'].append(20023) # 20023 | deuteron | A1h+   | COMPASS 
conf['steps'][istep]['datasets']['psidis'].append(20024) # 20024 | deuteron | A1h-   | COMPASS 
#conf['steps'][istep]['datasets']['sidis']=[]
#conf['steps'][istep]['datasets']['sidis'].append(1005) # deuteron , mult , pi+ , COMPASS
#conf['steps'][istep]['datasets']['sidis'].append(1006) # deuteron , mult , pi- , COMPASS
#conf['steps'][istep]['datasets']['sidis'].append(2005) # deuteron , mult , K+  , COMPASS
#conf['steps'][istep]['datasets']['sidis'].append(2006) # deuteron , mult , K-  , COMPASS
#conf['steps'][istep]['datasets']['sidis'].append(3000) # deuteron , mult , h+  , COMPASS
#conf['steps'][istep]['datasets']['sidis'].append(3001) # deuteron , mult , h-  , COMPASS

conf['FILT'] = {_:[] for _ in ['exp','par','value']}

conf['FILT']['exp'].append(('sidis',1005,10))
conf['FILT']['exp'].append(('sidis',1006,10))
conf['FILT']['exp'].append(('sidis',2005,10))
conf['FILT']['exp'].append(('sidis',2006,10))
conf['FILT']['exp'].append(('sidis',3000,10))
conf['FILT']['exp'].append(('sidis',3001,10))
conf['FILT']['exp'].append(('psidis',20004,10))
conf['FILT']['exp'].append(('psidis',20005,10))
conf['FILT']['exp'].append(('psidis',20008,10))
conf['FILT']['exp'].append(('psidis',20009,10))
conf['FILT']['exp'].append(('psidis',20017,10))
conf['FILT']['exp'].append(('psidis',20018,10))
conf['FILT']['exp'].append(('psidis',20021,10))
conf['FILT']['exp'].append(('psidis',20012,10))
conf['FILT']['exp'].append(('psidis',20013,10))
conf['FILT']['exp'].append(('psidis',20014,10))
conf['FILT']['exp'].append(('psidis',20019,10))
conf['FILT']['exp'].append(('psidis',20020,10))
conf['FILT']['exp'].append(('psidis',20025,10))
conf['FILT']['exp'].append(('psidis',20026,10))
conf['FILT']['exp'].append(('psidis',20000,10))
conf['FILT']['exp'].append(('psidis',20001,10))
conf['FILT']['exp'].append(('psidis',20002,10))
conf['FILT']['exp'].append(('psidis',20003,10))
conf['FILT']['exp'].append(('psidis',20006,10))
conf['FILT']['exp'].append(('psidis',20007,10))
conf['FILT']['exp'].append(('psidis',20010,10))
conf['FILT']['exp'].append(('psidis',20011,10))
conf['FILT']['exp'].append(('psidis',20015,10))
conf['FILT']['exp'].append(('psidis',20016,10))
conf['FILT']['exp'].append(('psidis',20023,10))
conf['FILT']['exp'].append(('psidis',20024,10))
conf['FILT']['exp'].append(('pjet',20001,10))
conf['FILT']['exp'].append(('pjet',20002,10))
conf['FILT']['exp'].append(('pjet',20003,10))
conf['FILT']['exp'].append(('pjet',20004,10))
conf['FILT']['exp'].append(('pjet',20005,10))
conf['FILT']['exp'].append(('pjet',20006,10))
conf['FILT']['exp'].append(('pjet',20007,10))
conf['FILT']['exp'].append(('pjet',20008,10))
conf['FILT']['exp'].append(('wzrv',1000,10))
conf['FILT']['exp'].append(('wzrv',1020,10))
conf['FILT']['exp'].append(('wzrv',1021,10))
conf['FILT']['exp'].append(('sia ',4000,10))
conf['FILT']['exp'].append(('sia' ,4001,10))
conf['FILT']['exp'].append(('sia' ,4002,10))
conf['FILT']['exp'].append(('sia' ,4004,10))
conf['FILT']['exp'].append(('sia' ,4005,10))
conf['FILT']['exp'].append(('sia' ,4006,10))
conf['FILT']['exp'].append(('sia' ,4007,10))
conf['FILT']['exp'].append(('sia' ,4011,10))
conf['FILT']['exp'].append(('sia' ,4012,10))
conf['FILT']['exp'].append(('sia' ,4013,10))
conf['FILT']['exp'].append(('sia' ,4014,10))
conf['FILT']['exp'].append(('sia' ,4015,10))






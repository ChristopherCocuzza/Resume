#!/usr/bin/env python

import sys,os
import numpy as np
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

#--matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import pylab as py

#--from local
from analysis.corelib import core,classifier

#--from tools
from tools.tools     import load,save,checkdir,lprint
from tools.config    import conf,load_config

#--from fitpack
from obslib.idis.reader    import READER as idisREAD
from obslib.pidis.reader   import READER as pidisREAD
from obslib.dy.reader      import READER as dyREAD
from obslib.wasym.reader   import READER as wasymREAD
from obslib.zrap.reader    import READER as zrapREAD
from obslib.wzrv.reader    import READER as wzrvREAD
from obslib.jets.reader    import READER as jetREAD
from obslib.pjets.reader   import READER as pjetREAD
from obslib.sidis.reader   import READER as sidisREAD
from obslib.psidis.reader  import READER as psidisREAD
from obslib.sia.reader     import READER as siaREAD
from qcdlib import aux

conf['aux']=aux.AUX()

cwd = 'plots/star/'
#--make kinematic plot

def get_kin_X(exp,data):
    #--get X, Q2
    if exp == 'idis' or exp=='pidis' or exp=='sidis' or exp=='psidis':
        X  = data['X']
        Q2 = data['Q2']
    elif exp == 'dy' or exp == 'wasym' or exp == 'zrap':
        Y = data['Y']
        if exp == 'dy':
            Q2  = data['Q2']
            tau = data['tau']
        if exp == 'wasym':
            Q2  = 80.398**2*np.ones(len(Y))
            S   = data['S']
            tau = Q2/S
        if exp == 'zrap':
            Q2  = 91.1876**2*np.ones(len(Y))
            S   = data['S']
            tau = Q2/S
        X = np.sqrt(tau)*np.cosh(Y)
    elif exp == 'wzrv':
        if 'eta' in data:
            eta = data['eta']
        else:
            eta = (data['eta_min'] + data['eta_max'])/2.0
        Q2  = 80.398**2*np.ones(len(eta))
        S   = data['cms']**2
        tau = Q2/S
        X = np.sqrt(tau)*np.cosh(eta)
    elif exp == 'pjet' or exp=='jet':
        pT   = (data['pt-max'] + data['pt-min'])/2.0
        Q2   = pT**2
        S    = data['S']
        tau  = Q2/S
        if 'eta-max' in data:
            etamin,etamax = data['eta-max'],data['eta-min']
            Xmin = 2*np.sqrt(tau)*np.cosh(etamin)
            Xmax = 2*np.sqrt(tau)*np.cosh(etamax)
            X = (Xmin+Xmax)/2.0
        else:
            eta = (data['eta-abs-max']+data['eta-abs-min'])/2.0
            X = 2*np.sqrt(tau)*np.cosh(eta)

    return X,Q2

def get_kin_Z(exp,data):
    #--get Z, Q2
    if  exp=='sidis' or exp=='psidis':
        Z  = data['Z']
        Q2 = data['Q2']

    elif exp=='sia':
        Z  = data['z']
        Q2 = data['RS']**2

    return Z,Q2

def load_data(W2cut = 3.0, pW2cut=10):

    conf['datasets'] = {}
    data = {}

    ##--IDIS
    Q2cut=1.3**2
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
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['idis']['xlsx'][10011]='idis/expdata/10011.xlsx' # deuteron | F2            | SLAC
    conf['datasets']['idis']['xlsx'][10017]='idis/expdata/10017.xlsx' # deuteron | F2            | BCDMS
    conf['datasets']['idis']['xlsx'][10021]='idis/expdata/10021.xlsx' # d/p      | F2d/F2p       | NMC
    conf['datasets']['idis']['xlsx'][10002]='idis/expdata/10002.xlsx' # deuteron | F2            | JLab Hall C (E00-106)
    conf['datasets']['idis']['xlsx'][10033]='idis/expdata/10033.xlsx' # n/d      | F2n/F2d       | BONUS
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['idis']['xlsx'][10050]='idis/expdata/10050.xlsx' # d/p      | F2d/F2p       | MARATHON
    conf['datasets']['idis']['xlsx'][10051]='idis/expdata/10051.xlsx' # h/t      | F2h/F2t       | MARATHON
    #------------------------------------------------------------------------------------------------------------------
    data['idis'] = idisREAD().load_data_sets('idis')  
 
    ##--DY 
    conf['datasets']['dy']={}
    conf['datasets']['dy']['filters']=[]
    conf['datasets']['dy']['filters'].append("Q2>36") 
    conf['datasets']['dy']['xlsx']={}
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['dy']['xlsx'][10001]='dy/expdata/10001.xlsx'
    conf['datasets']['dy']['xlsx'][10002]='dy/expdata/10002.xlsx'
    #------------------------------------------------------------------------------------------------------------------
    data['dy'] = dyREAD().load_data_sets('dy')  
    
    ##--charge asymmetry 
    conf['datasets']['wzrv']={}
    conf['datasets']['wzrv']['filters']=[]
    conf['datasets']['wzrv']['xlsx']={}
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['wzrv']['xlsx'][2010]='wzrv/expdata/2010.xlsx'
    conf['datasets']['wzrv']['xlsx'][2011]='wzrv/expdata/2011.xlsx'
    conf['datasets']['wzrv']['xlsx'][2012]='wzrv/expdata/2012.xlsx'
    conf['datasets']['wzrv']['xlsx'][2013]='wzrv/expdata/2013.xlsx'
    conf['datasets']['wzrv']['xlsx'][2014]='wzrv/expdata/2014.xlsx'
    conf['datasets']['wzrv']['xlsx'][2016]='wzrv/expdata/2016.xlsx'
    conf['datasets']['wzrv']['xlsx'][2017]='wzrv/expdata/2017.xlsx'
    conf['datasets']['wzrv']['xlsx'][2020]='wzrv/expdata/2020.xlsx'
    #------------------------------------------------------------------------------------------------------------------
    
    ##--W asymmetry 
    conf['datasets']['wasym']={}
    conf['datasets']['wasym']['filters']=[]
    conf['datasets']['wasym']['xlsx']={}
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['wasym']['xlsx'][1000]='wasym/expdata/1000.xlsx'
    conf['datasets']['wasym']['xlsx'][1001]='wasym/expdata/1001.xlsx'
    #------------------------------------------------------------------------------------------------------------------
    data['wasym'] = wasymREAD().load_data_sets('wasym')  
    
    ##--Z rapidity 
    conf['datasets']['zrap']={}
    conf['datasets']['zrap']['filters']=[]
    conf['datasets']['zrap']['xlsx']={}
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['zrap']['xlsx'][1000]='zrap/expdata/1000.xlsx'
    conf['datasets']['zrap']['xlsx'][1001]='zrap/expdata/1001.xlsx'
    #------------------------------------------------------------------------------------------------------------------
    data['zrap'] = zrapREAD().load_data_sets('zrap')

    #--Jets
    conf['datasets']['jet'] = {}
    conf['datasets']['jet']['filters'] = []
    conf['datasets']['jet']['filters'].append("pT>10")
    conf['datasets']['jet']['xlsx'] = {}
    conf['datasets']['jet']['xlsx'][10001] = 'jets/expdata/10001.xlsx' ## D0 dataset
    conf['datasets']['jet']['xlsx'][10002] = 'jets/expdata/10002.xlsx' ## CDF dataset
    conf['datasets']['jet']['xlsx'][10003] = 'jets/expdata/10003.xlsx' ## STAR MB dataset
    conf['datasets']['jet']['xlsx'][10004] = 'jets/expdata/10004.xlsx' ## STAR HT dataset
    data['jet'] = jetREAD().load_data_sets('jet')

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
    data['sidis'] = sidisREAD().load_data_sets('sidis')

    #--PIDIS
    conf['datasets']['pidis']={}
    pQ2cut=1.3**2
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['pidis']['filters']=[]
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['pidis']['filters'].append("Q2>%f"%pQ2cut) 
    conf['datasets']['pidis']['filters'].append("W2>%f"%pW2cut) 
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['pidis']['xlsx']={}
    #---------------------------------------------------------------------------------------------------------------------------
    conf['datasets']['pidis']['xlsx'][10002]='pidis/expdata/10002.xlsx' # 10002 | proton   | A1   | COMPASS         |          |
    conf['datasets']['pidis']['xlsx'][10003]='pidis/expdata/10003.xlsx' # 10003 | proton   | A1   | COMPASS         |          |
    conf['datasets']['pidis']['xlsx'][10004]='pidis/expdata/10004.xlsx' # 10004 | proton   | A1   | EMC             |          |
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
    conf['datasets']['pidis']['xlsx'][10041]='pidis/expdata/10041.xlsx' # 10041 | proton   | Apa  | JLabHB(EG1b)    | E =1 GeV |
    conf['datasets']['pidis']['xlsx'][10042]='pidis/expdata/10042.xlsx' # 10042 | proton   | Apa  | JLabHB(EG1b)    | E =2 GeV |
    conf['datasets']['pidis']['xlsx'][10043]='pidis/expdata/10043.xlsx' # 10043 | proton   | Apa  | JLabHB(EG1b)    | E =4 GeV |
    conf['datasets']['pidis']['xlsx'][10044]='pidis/expdata/10044.xlsx' # 10044 | proton   | Apa  | JLabHB(EG1b)    | E =5 GeV |
    conf['datasets']['pidis']['xlsx'][10005]='pidis/expdata/10005.xlsx' # 10005 | neutron  | A1   | HERMES          |          |
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
    conf['datasets']['pidis']['xlsx'][10037]='pidis/expdata/10037.xlsx' # 10037 | deuteron | Apa  | JLabHB(EG1b)    | E =1 GeV |
    conf['datasets']['pidis']['xlsx'][10038]='pidis/expdata/10038.xlsx' # 10038 | deuteron | Apa  | JLabHB(EG1b)    | E =2 GeV |
    conf['datasets']['pidis']['xlsx'][10039]='pidis/expdata/10039.xlsx' # 10039 | deuteron | Apa  | JLabHB(EG1b)    | E =4 GeV |
    conf['datasets']['pidis']['xlsx'][10040]='pidis/expdata/10040.xlsx' # 10040 | deuteron | Apa  | JLabHB(EG1b)    | E =5 GeV |
    #---------------------------------------------------------------------------------------------------------------------------
    conf['datasets']['pidis']['xlsx'][10009]='pidis/expdata/10009.xlsx' # 10009 | helium   | Apa  | JLabHA(E01-012) | < cuts  |
    conf['datasets']['pidis']['xlsx'][10010]='pidis/expdata/10010.xlsx' # 10010 | helium   | Apa  | JLabHA(E06-014) |          |
    conf['datasets']['pidis']['xlsx'][10011]='pidis/expdata/10011.xlsx' # 10011 | helium   | Ape  | JLabHA(E06-014) |          |
    conf['datasets']['pidis']['xlsx'][10012]='pidis/expdata/10012.xlsx' # 10012 | helium   | Apa  | JLabHA(E97-103) | < cuts  |
    conf['datasets']['pidis']['xlsx'][10013]='pidis/expdata/10013.xlsx' # 10013 | helium   | Ape  | JLabHA(E97-103) | < cuts  |
    conf['datasets']['pidis']['xlsx'][10014]='pidis/expdata/10014.xlsx' # 10014 | helium   | Apa  | JLabHA(E99-117) |          |
    conf['datasets']['pidis']['xlsx'][10015]='pidis/expdata/10015.xlsx' # 10015 | helium   | Ape  | JLabHA(E99-117) |          |
    conf['datasets']['pidis']['xlsx'][10018]='pidis/expdata/10018.xlsx' # 10018 | helium   | A1   | SLAC(E142)      |          |
    conf['datasets']['pidis']['xlsx'][10019]='pidis/expdata/10019.xlsx' # 10019 | helium   | A2   | SLAC(E142)      |          |
    conf['datasets']['pidis']['xlsx'][10024]='pidis/expdata/10024.xlsx' # 10024 | helium   | Ape  | SLAC(E154)      |          |
    conf['datasets']['pidis']['xlsx'][10025]='pidis/expdata/10025.xlsx' # 10025 | helium   | Apa  | SLAC(E154)      |          |
    #---------------------------------------------------------------------------------------------------------------------------
    data['pidis'] = pidisREAD().load_data_sets('pidis')

    ##--charge asymmetry 
    #------------------------------------------------------------------------------------------------------------------
    conf['datasets']['wzrv']['xlsx'][1000]='wzrv/expdata/1000.xlsx'
    conf['datasets']['wzrv']['xlsx'][1020]='wzrv/expdata/1020.xlsx'
    conf['datasets']['wzrv']['xlsx'][1021]='wzrv/expdata/1021.xlsx'
    #------------------------------------------------------------------------------------------------------------------
    data['wzrv'] = wzrvREAD().load_data_sets('wzrv')  

    ##--PJET
    conf['datasets']['pjet'] = {}
    conf['datasets']['pjet']['xlsx'] = {}
    conf['datasets']['pjet']['xlsx'][20001] = 'pjets/expdata/20001.xlsx' ## STAR 2006 paper on 2003 and 2004 data
    conf['datasets']['pjet']['xlsx'][20002] = 'pjets/expdata/20002.xlsx' ## STAR 2012 paper on 2005 data
    conf['datasets']['pjet']['xlsx'][20003] = 'pjets/expdata/20003.xlsx' ## STAR 2012 paper on 2006 data
    conf['datasets']['pjet']['xlsx'][20004] = 'pjets/expdata/20004.xlsx' ## STAR 2015 paper on 2009 data
    conf['datasets']['pjet']['xlsx'][20005] = 'pjets/expdata/20005.xlsx' ## PHENIX 2011 paper on 2005 data
    conf['datasets']['pjet']['xlsx'][20006] = 'pjets/expdata/20006.xlsx' ## STAR 2019 paper on 2012 data
    conf['datasets']['pjet']['xlsx'][20007] = 'pjets/expdata/20007.xlsx' ## STAR 2021 paper on 2015 data
    data['pjet'] = pjetREAD().load_data_sets('pjet')

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
    data['psidis'] = psidisREAD().load_data_sets('psidis')
    
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
    conf['datasets']['sia']['norm'][2029]={'value':    1.00000e+00,'fixed':True,'min':0.5,'max':1.5}
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
    data['sia'] = siaREAD().load_data_sets('sia')

    return data

def plot_kin(data):

    nrows,ncols=1,3
    fig = py.figure(figsize=(ncols*14,nrows*8))
    ax11=py.subplot(nrows,ncols,1)
    ax12=py.subplot(nrows,ncols,2)
    ax13=py.subplot(nrows,ncols,3)

    divider = make_axes_locatable(ax11)
    ax11L = divider.append_axes("right",size=6,pad=0,sharey=ax11)
    ax11L.set_xlim(0.1,0.85)
    ax11L.spines['left'].set_visible(False)
    ax11L.yaxis.set_ticks_position('right')
    py.setp(ax11L.get_xticklabels(),visible=True)

    ax11.spines['right'].set_visible(False)

    divider = make_axes_locatable(ax12)
    ax12L = divider.append_axes("right",size=6,pad=0,sharey=ax12)
    ax12L.set_xlim(0.1,0.85)
    ax12L.spines['left'].set_visible(False)
    ax12L.yaxis.set_ticks_position('right')
    py.setp(ax12L.get_xticklabels(),visible=True)

    ax12.spines['right'].set_visible(False)

    hand = {}

    #--plot x and Q2 (unpolarized)
    for exp in data:
        if exp=='sia': continue
        if exp=='pidis': continue
        if exp=='pjet':  continue
        if exp=='psidis':  continue
        hand[exp] = {}
        for idx in data[exp]:
            X,Q2 = get_kin_X(exp,data[exp][idx])
            label = None
            if exp == 'idis':
                marker,color,label = 'o','cyan','DIS'
                s=35

            elif exp == 'sidis':
                marker,color,label = 's','orange','SIDIS'
                s=35

            elif exp == 'dy':
                marker,color,label  = 'D','purple','DY'
                s=35

            elif exp == 'wasym':
                marker,color,label = '*', 'red', 'W/Z'
                s=80

            elif exp == 'zrap':
                marker,color,label = '*', 'red', 'W/Z'
                s=80

            elif exp == 'wzrv' and idx not in [1000,1020,1021]:
                marker,color,label = '*', 'red', 'W/Z'
                s=80

            elif exp == 'jet':
                marker,color,label = 'v', 'blue', 'jets'
                s=40

            else: continue

            s, zorder,edgecolors,linewidths = 35,1,'face',1.5
            ax11               .scatter(X,Q2,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)
            hand[label] = ax11L.scatter(X,Q2,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)


    #--plot x and Q2 (polarized)
    for exp in data:
        if exp=='sia': continue
        if exp=='idis': continue
        if exp=='sidis': continue
        if exp=='dy': continue
        if exp=='zrap': continue
        if exp=='wasym': continue
        if exp=='jet': continue
        hand[exp] = {}
        for idx in data[exp]:
            X,Q2 = get_kin_X(exp,data[exp][idx])
            label = None
            if exp == 'pidis':
                marker,color,label = 'o','cyan','DIS'
                s=35

            elif exp == 'psidis':
                marker,color,label = 's','green','PSIDIS'
                s=35

            elif exp=='wzrv' and idx in [1000,1020,1021]:
                marker,color,label = '*', 'red', 'RHIC W/Z'
                s=80

            elif exp=='pjet':
                marker,color,label = 'v', 'blue', 'jets'
                s=40

            else: continue

            s, zorder,edgecolors,linewidths = 35,1,'face',1.5
            ax12               .scatter(X,Q2,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)
            hand[label] = ax12L.scatter(X,Q2,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)

    #--plot z and Q2
    for exp in data:
        if exp=='idis':  continue
        if exp=='dy':    continue
        if exp=='wasym': continue
        if exp=='zrap':  continue
        if exp=='jet':   continue
        if exp=='pidis': continue
        if exp=='wzrv':  continue
        if exp=='pjet':  continue
        hand[exp] = {}
        for idx in data[exp]:
            Z,Q2 = get_kin_Z(exp,data[exp][idx])
            label = None
            if exp == 'sidis':
                marker,color,label = 's','orange','SIDIS'
                s=35

            elif exp == 'psidis':
                marker,color,label = 's','green','PSIDIS'
                s=35

            elif exp=='sia':
                marker,color,label = '^', 'magenta', 'SIA'
                s=80

            else: continue

            s, zorder,edgecolors,linewidths = 35,1,'face',1.5
            hand[label] = ax13.scatter(Z,Q2,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)

    #--Plot cuts
    x = np.linspace(0.1,0.9,100)
    W2cut3  = (3.0 - 0.938**2)*x/(1-x)
    W2cut10 = (10.0 - 0.938**2)*x/(1-x)

    hand['W2=3']  ,= ax11L.plot(x,W2cut3 ,'k--',zorder=2)
    hand['W2=10'] ,= ax12L.plot(x,W2cut10,'k--',zorder=2)

    for ax in [ax11, ax11L, ax12, ax12L, ax13]:
        ax.set_yscale('log')
        ax.set_ylim(1,6e5)
 

    ax11 .tick_params(axis='both',which='both',top=True,right=False,direction='in',labelsize=30)
    ax11L.tick_params(axis='both',which='both',top=True,right=True,labelright=False,direction='in',labelsize=30)

    ax11 .tick_params(axis='x',which='major',pad=8)
    ax11L.tick_params(axis='x',which='major',pad=8)

    ax11 .tick_params(axis='both',which='minor',size=4)
    ax11 .tick_params(axis='both',which='major',size=8)
    ax11L.tick_params(axis='both',which='minor',size=4)
    ax11L.tick_params(axis='both',which='major',size=8)

    ax11.set_xscale('log')

    ax11.set_xlim(2e-5,0.1)
    ax11. set_xticks([1e-4,1e-3,1e-2])
    ax11. set_xticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'])
    ax11L.set_xticks([0.1,0.3,0.5,0.7])

    ax11L.set_xlabel(r'\boldmath$x$',size=50)
    ax11L.xaxis.set_label_coords(0.95,0.00)
    ax11.set_ylabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    minorLocator = MultipleLocator(0.1)
    ax11L.xaxis.set_minor_locator(minorLocator)

    ax12 .tick_params(axis='both',which='both',labelleft=False,top=True,right=False,direction='in',labelsize=30)
    ax12L.tick_params(axis='both',which='both',labelleft=False,top=True,right=True,labelright=False,direction='in',labelsize=30)

    ax12 .tick_params(axis='x',which='major',pad=8)
    ax12L.tick_params(axis='x',which='major',pad=8)

    ax12 .tick_params(axis='both',which='minor',size=4)
    ax12 .tick_params(axis='both',which='major',size=8)
    ax12L.tick_params(axis='both',which='minor',size=4)
    ax12L.tick_params(axis='both',which='major',size=8)

    ax12.set_xscale('log')

    ax12. set_xlim(4e-3,0.1)
    ax12. set_xticks([1e-2])
    ax12. set_xticklabels([r'$10^{-2}$'])
    ax12L.set_xticks([0.1,0.3,0.5,0.7])

    ax12L.set_xlabel(r'\boldmath$x$',size=50)
    ax12L.xaxis.set_label_coords(0.95,0.00)

    minorLocator = MultipleLocator(0.1)
    ax12L.xaxis.set_minor_locator(minorLocator)

    ax13 .tick_params(axis='both',which='both',top=True,right=False,labelleft=False,direction='in',labelsize=30)

    ax13 .tick_params(axis='x',which='major',pad=8)

    ax13 .tick_params(axis='both',which='minor',size=4)
    ax13 .tick_params(axis='both',which='major',size=8)

    ax13.set_xlim(0.1,1.0)
    ax13. set_xticks([0.2,0.4,0.6,0.8])

    ax13.set_xlabel(r'\boldmath$z_h$',size=50)
    ax13.xaxis.set_label_coords(0.95,0.00)

    minorLocator = MultipleLocator(0.1)
    ax13.xaxis.set_minor_locator(minorLocator)


    ax11L.text(-0.40,0.90,r'\textrm{\textbf{Spin-Averaged PDFs}}',transform = ax11L.transAxes, size=30)
    ax12L.text(-0.40,0.90,r'\textrm{\textbf{Helicity PDFs}}'     ,transform = ax12L.transAxes, size=30)
    ax13. text( 0.50,0.90,r'\textrm{\textbf{FFs}}'               ,transform = ax13 .transAxes, size=30)


    hand['blank'] ,= ax11.plot(0,0,alpha=0)

    fs = 25

    handles,labels = [], []
    handles.append(hand['DIS'])
    handles.append(hand['SIDIS'])
    handles.append(hand['DY'])
    handles.append(hand['jets'])
    handles.append(hand['W/Z'])
    handles.append(hand['W2=3'])
    labels.append(r'\textbf{\textrm{DIS (\boldmath$p,D,^3$He,\boldmath$^3$H)}}')
    labels.append(r'\textbf{\textrm{SIDIS (\boldmath$\pi, K, h$)}}')
    labels.append(r'\textbf{\textrm{Drell-Yan}}')
    labels.append(r'\textbf{\textrm{RHIC/Tevatron jets}}')
    labels.append(r'\textbf{\textrm{RHIC \boldmath$W/Z$}}')
    labels.append(r'\boldmath$W^2 = 3~{\rm GeV}^2$')
    ax11.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol = 1, columnspacing = 1.0)



    handles,labels = [],[]
    handles.append(hand['DIS'])
    handles.append(hand['PSIDIS'])
    handles.append(hand['jets'])
    handles.append(hand['RHIC W/Z'])
    handles.append(hand['W2=10'])
    labels.append(r'\textbf{\textrm{DIS (\boldmath$p,D,^3$He)}}')
    labels.append(r'\textbf{\textrm{SIDIS (\boldmath$\pi, K, h$)}}')
    labels.append(r'\textbf{\textrm{RHIC jets}}')
    labels.append(r'\textbf{\textrm{RHIC \boldmath$W/Z$}}')
    labels.append(r'\boldmath$W^2 = 10~{\rm GeV}^2$')
    ax12.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1)

    handles,labels = [],[]
    handles.append(hand['SIDIS'])
    handles.append(hand['PSIDIS'])
    handles.append(hand['SIA'])
    labels.append(r'\textbf{\textrm{SIDIS (\boldmath$\pi, K, h$)}}')
    labels.append(r'\textbf{\textrm{Pol. SIDIS (\boldmath$\pi, K, h$)}}')
    labels.append(r'\textbf{\textrm{SIA (\boldmath$\pi, K, h$)}}')
    labels.append(r'\boldmath$W^2 = 10~{\rm GeV}^2$')
    ax13.legend(handles,labels,loc='upper left',fontsize=fs,frameon=False, handlelength = 1.0, handletextpad = 0.1, ncol=1, columnspacing = 1.0)



    py.tight_layout()
    filename='%s/gallery/kin'%cwd
    filename+='.png'

    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)

def plot_kin_pol(data):
    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*14,nrows*8))
    ax=py.subplot(nrows,ncols,1)

    divider = make_axes_locatable(ax)
    axL = divider.append_axes("right",size=6,pad=0,sharey=ax)
    axL.set_xlim(0.1,0.85)
    axL.spines['left'].set_visible(False)
    axL.yaxis.set_ticks_position('right')
    py.setp(axL.get_xticklabels(),visible=True)

    ax.spines['right'].set_visible(False)

    hand = {}
    for exp in data:
        if exp=='sia': continue
        if exp=='idis': continue
        if exp=='sidis': continue
        if exp=='dy': continue
        if exp=='zrap': continue
        if exp=='wasym': continue
        if exp=='jet': continue
        hand[exp] = {}
        for idx in data[exp]:
            X,Q2 = get_kin_X(exp,data[exp][idx])
            label = None
            if exp == 'pidis':
                if idx==10001:   marker,color,label = 'o','green', 'COMPASS'    
                elif idx==10002: marker,color       = 'o','green'    
                elif idx==10003: marker,color       = 'o','green'    
                elif idx==10004: marker,color,label = 's','gray',  'EMC'
                elif idx==10033: marker,color,label = '*','magenta',  'SMC'
                elif idx==10034: marker,color       = '*','magenta'
                elif idx==10035: marker,color       = '*','magenta'
                elif idx==10036: marker,color       = '*','magenta'
                elif idx==10006: marker,color,label = 'v','purple', 'HERMES'
                elif idx==10007: marker,color       = 'v','purple'
                elif idx==10008: marker,color       = 'v','purple'
                elif idx==10018: marker,color,label = '^','goldenrod', 'SLAC'
                elif idx==10019: marker,color       = '^','goldenrod'
                elif idx==10020: marker,color       = '^','goldenrod'
                elif idx==10021: marker,color       = '^','goldenrod'
                elif idx==10022: marker,color       = '^','goldenrod'
                elif idx==10023: marker,color       = '^','goldenrod'
                elif idx==10024: marker,color       = '^','goldenrod'
                elif idx==10025: marker,color       = '^','goldenrod'
                elif idx==10026: marker,color       = '^','goldenrod'
                elif idx==10027: marker,color       = '^','goldenrod'
                elif idx==10028: marker,color       = '^','goldenrod'
                elif idx==10029: marker,color       = '^','goldenrod'
                elif idx==10030: marker,color       = '^','goldenrod'
                elif idx==10031: marker,color       = '^','goldenrod'
                elif idx==10032: marker,color       = '^','goldenrod'
                elif idx==10010: marker,color,label = 'o','red', 'JLab'
                elif idx==10011: marker,color       = 'o','red'
                elif idx==10014: marker,color       = 'o','red'
                elif idx==10015: marker,color       = 'o','red'
                elif idx==10016: marker,color       = 'o','red'
                elif idx==10017: marker,color       = 'o','red'
                elif idx==10037: marker,color       = 'o','red'
                elif idx==10038: marker,color       = 'o','red'
                elif idx==10039: marker,color       = 'o','red'
                elif idx==10040: marker,color       = 'o','red'
                elif idx==10041: marker,color       = 'o','red'
                elif idx==10042: marker,color       = 'o','red'
                elif idx==10043: marker,color       = 'o','red'
                elif idx==10044: marker,color       = 'o','red'
                else: continue
                s=35

            elif exp=='wzrv' and idx in [1000,1020,1021]:
                if   idx==1000:    marker,color,label = '*', 'darkcyan', 'STAR W'
                elif idx==1020:    marker,color,label = 'o', 'red', 'PHENIX W/Z'
                elif idx==1021:    marker,color       = 'o', 'red'
                s=80

            elif exp=='pjet':
                if   idx==20001:   marker,color,label = 's', 'black', 'STAR jets'
                elif idx==20002:   marker,color       = 's', 'black' 
                elif idx==20003:   marker,color       = 's', 'black' 
                elif idx==20004:   marker,color       = 's', 'black' 
                elif idx==20006:   marker,color       = 's', 'black' 
                elif idx==20005:   marker,color,label = 'v', 'blue', 'PHENIX jets' 
                s=40

            else: continue

            s, zorder,edgecolors,linewidths = 35,1,'face',1.5
            ax               .scatter(X,Q2,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)
            hand[label] = axL.scatter(X,Q2,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)


    #--Plot cuts
    x = np.linspace(0.1,0.9,100)
    Q2cut=np.ones(len(x))*1.3**2
    W2cut10 = (10.0 - 0.938**2)*x/(1-x)

    hand['W2=10'] ,= axL.plot(x,W2cut10,'k--',zorder=2)

    ax.axvline(0.1,color='black',ls=':',alpha=0.5)

    ax .tick_params(axis='both',which='both',top=True,right=False,direction='in',labelsize=30)
    axL.tick_params(axis='both',which='both',top=True,right=True,labelright=False,direction='in',labelsize=30)

    ax .tick_params(axis='x',which='major',pad=8)
    axL.tick_params(axis='x',which='major',pad=8)

    ax .tick_params(axis='both',which='minor',size=4)
    ax .tick_params(axis='both',which='major',size=8)
    axL.tick_params(axis='both',which='minor',size=4)
    axL.tick_params(axis='both',which='major',size=8)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(5e-3,0.1)
    ax.set_ylim(1,2e4)
    ax. set_xticks([1e-2])
    ax. set_xticklabels([r'$10^{-2}$'])
    axL.set_xticks([0.1,0.3,0.5,0.7])

    axL.set_xlabel(r'\boldmath$x$',size=50)
    axL.xaxis.set_label_coords(0.95,0.00)
    ax.set_ylabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    minorLocator = MultipleLocator(0.1)
    axL.xaxis.set_minor_locator(minorLocator)

    hand['blank'] ,= ax.plot(0,0,alpha=0)

    handles,labels = [],[]
    handles.append(hand['EMC'])
    handles.append(hand['SMC'])
    handles.append(hand['COMPASS'])
    handles.append(hand['HERMES'])
    handles.append(hand['SLAC'])
    handles.append(hand['STAR jets'])
    handles.append(hand['PHENIX jets'])
    handles.append(hand['STAR W'])
    handles.append(hand['PHENIX W/Z'])
    handles.append(hand['W2=10'])
    labels.append(r'\textbf{\textrm{EMC}}')
    labels.append(r'\textbf{\textrm{SMC}}')
    labels.append(r'\textbf{\textrm{COMPASS}}')
    labels.append(r'\textbf{\textrm{HERMES}}')
    labels.append(r'\textbf{\textrm{SLAC}}')
    labels.append(r'\textbf{\textrm{STAR jets}}')
    labels.append(r'\textbf{\textrm{PHENIX jets}}')
    labels.append(r'\textbf{\textrm{STAR W}}')
    labels.append(r'\textbf{\textrm{PHENIX W/Z}}')
    labels.append(r'\boldmath$W^2 = 10~{\rm GeV}^2$')
    ax.legend(handles,labels,loc='upper left',fontsize=25,frameon=False, handlelength = 1.0, handletextpad = 0.1)

    py.tight_layout()
    filename='%s/gallery/kin_pol'%cwd
    filename+='.png'

    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)


if __name__ == "__main__":

    data = load_data()

    plot_kin_pol(data)
    plot_kin(data)






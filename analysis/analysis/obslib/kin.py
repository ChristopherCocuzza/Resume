#!/usr/bin/env python

import sys,os
import numpy as np
import copy

#--matplotlib
import matplotlib
matplotlib.use('Agg')
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
from obslib.pjets.reader   import READER as pjetREAD
from qcdlib import aux

conf['aux']=aux.AUX()

#--make kinematic plots

def plot(wdir,pol,W2cut=10):

    data = load_data(pol,W2cut)
    if pol: plot_kin_pol(wdir,data,W2cut)
    else:   plot_kin_upol(wdir,data)

def get_kin(exp,data):
    #--get X, Q2
    if exp == 'idis' or exp=='pidis':
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
    elif exp == 'pjet':
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

def load_data(pol=False,W2cut=10):

    conf['datasets'] = {}
    data = {}
    if pol==False:
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
        #conf['datasets']['idis']['xlsx'][10007]='idis/expdata/10007.xlsx' # proton   | sigma red     | HERMES
        #------------------------------------------------------------------------------------------------------------------
        conf['datasets']['idis']['xlsx'][10011]='idis/expdata/10011.xlsx' # deuteron | F2            | SLAC
        conf['datasets']['idis']['xlsx'][10017]='idis/expdata/10017.xlsx' # deuteron | F2            | BCDMS
        conf['datasets']['idis']['xlsx'][10021]='idis/expdata/10021.xlsx' # d/p      | F2d/F2p       | NMC
        #conf['datasets']['idis']['xlsx'][10006]='idis/expdata/10006.xlsx' # deuteron | F2            | HERMES
        conf['datasets']['idis']['xlsx'][10002]='idis/expdata/10002.xlsx' # deuteron | F2            | JLab Hall C (E00-106)
        conf['datasets']['idis']['xlsx'][10033]='idis/expdata/10033.xlsx' # n/d      | F2n/F2d       | BONUS
        #------------------------------------------------------------------------------------------------------------------
        conf['datasets']['idis']['xlsx'][10050]='idis/expdata/10050.xlsx' # d/p      | F2d/F2p       | MARATHON
        conf['datasets']['idis']['xlsx'][10051]='idis/expdata/10051.xlsx' # h/d      | F2h/F2d       | MARATHON
        conf['datasets']['idis']['xlsx'][10052]='idis/expdata/10052.xlsx' # t/d      | F2t/F2d       | MARATHON
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
        conf['datasets']['wzrv']['xlsx'][2000]='wzrv/expdata/2000.xlsx'
        conf['datasets']['wzrv']['xlsx'][2003]='wzrv/expdata/2003.xlsx'
        conf['datasets']['wzrv']['xlsx'][2006]='wzrv/expdata/2006.xlsx'
        conf['datasets']['wzrv']['xlsx'][2007]='wzrv/expdata/2007.xlsx'
        #conf['datasets']['wzrv']['xlsx'][2008]='wzrv/expdata/2008.xlsx'  #--ATLAS 2011 w/ correlated uncertainties
        conf['datasets']['wzrv']['xlsx'][2009]='wzrv/expdata/2009.xlsx'
        conf['datasets']['wzrv']['xlsx'][2010]='wzrv/expdata/2010.xlsx'
        conf['datasets']['wzrv']['xlsx'][2011]='wzrv/expdata/2011.xlsx'
        conf['datasets']['wzrv']['xlsx'][2012]='wzrv/expdata/2012.xlsx'
        conf['datasets']['wzrv']['xlsx'][2013]='wzrv/expdata/2013.xlsx'
        conf['datasets']['wzrv']['xlsx'][2014]='wzrv/expdata/2014.xlsx'
        conf['datasets']['wzrv']['xlsx'][2015]='wzrv/expdata/2015.xlsx'  #--ATLAS 2011 w/ uncorrelated uncertainties
        #------------------------------------------------------------------------------------------------------------------
        data['wzrv'] = wzrvREAD().load_data_sets('wzrv')  
        
        ##--W asymmetry 
        conf['datasets']['wasym']={}
        conf['datasets']['wasym']['filters']=[]
        conf['datasets']['wasym']['xlsx']={}
        #------------------------------------------------------------------------------------------------------------------
        conf['datasets']['wasym']['xlsx'][1000]='wasym/expdata/1000.xlsx'
        conf['datasets']['wasym']['xlsx'][1001]='wasym/expdata/1001.xlsx'
        #------------------------------------------------------------------------------------------------------------------
        data['wasym'] = wasymREAD().load_data_sets('wasym')  
        
        ##--W asymmetry 
        conf['datasets']['zrap']={}
        conf['datasets']['zrap']['filters']=[]
        conf['datasets']['zrap']['xlsx']={}
        #------------------------------------------------------------------------------------------------------------------
        conf['datasets']['zrap']['xlsx'][1000]='zrap/expdata/1000.xlsx'
        conf['datasets']['zrap']['xlsx'][1001]='zrap/expdata/1001.xlsx'
        #------------------------------------------------------------------------------------------------------------------
        data['zrap'] = zrapREAD().load_data_sets('zrap')

    if pol==True:
        conf['datasets']['pidis']={}
        Q2cut=1.3**2
        #------------------------------------------------------------------------------------------------------------------
        conf['datasets']['pidis']['filters']=[]
        #------------------------------------------------------------------------------------------------------------------
        conf['datasets']['pidis']['filters'].append("Q2>%f"%Q2cut) 
        conf['datasets']['pidis']['filters'].append("W2>%f"%W2cut) 
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
        conf['datasets']['wzrv']={}
        conf['datasets']['wzrv']['filters']=[]
        conf['datasets']['wzrv']['xlsx']={}
        #------------------------------------------------------------------------------------------------------------------
        conf['datasets']['wzrv']['xlsx'][1000]='wzrv/expdata/1000.xlsx'
        conf['datasets']['wzrv']['xlsx'][1001]='wzrv/expdata/1001.xlsx'
        #------------------------------------------------------------------------------------------------------------------
        data['wzrv'] = wzrvREAD().load_data_sets('wzrv')  

        ##--PJET
        conf['datasets']['pjet'] = {}
        conf['datasets']['pjet']['filters'] = []
        conf['datasets']['pjet']['filters'].append("pT>10.0")
        conf['datasets']['pjet']['xlsx'] = {}
        conf['datasets']['pjet']['xlsx'][20001] = 'pjets/expdata/20001.xlsx' ## STAR 2006 paper on 2003 and 2004 data
        conf['datasets']['pjet']['xlsx'][20002] = 'pjets/expdata/20002.xlsx' ## STAR 2012 paper on 2005 data
        conf['datasets']['pjet']['xlsx'][20003] = 'pjets/expdata/20003.xlsx' ## STAR 2012 paper on 2006 data
        conf['datasets']['pjet']['xlsx'][20004] = 'pjets/expdata/20004.xlsx' ## STAR 2015 paper on 2009 data
        conf['datasets']['pjet']['xlsx'][20005] = 'pjets/expdata/20005.xlsx' ## PHENIX 2011 paper on 2005 data
        conf['datasets']['pjet']['xlsx'][20006] = 'pjets/expdata/20006.xlsx' ## STAR 2019 paper on 2012 data
        conf['pjet_qr_fit'] = {'method': 'fixed', 'f_scale': 1.0, 'r_scale': 1.0}
        data['pjet'] = pjetREAD().load_data_sets('pjet')  


    return data

def plot_kin_upol(wdir,data):
    s = 35

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*14,nrows*8))
    ax=py.subplot(nrows,ncols,1)

    divider = make_axes_locatable(ax)
    axL = divider.append_axes("right",size=6,pad=0,sharey=ax)
    axL.set_xlim(0.1,0.9)
    axL.spines['left'].set_visible(False)
    axL.yaxis.set_ticks_position('right')
    py.setp(axL.get_xticklabels(),visible=True)

    ax.spines['right'].set_visible(False)

    hand = {}
    for exp in data:
        hand[exp] = {}
        for idx in data[exp]:
            X,Q2 = get_kin(exp,data[exp][idx])
            X1,  X2  = [], []
            Q21, Q22 = [], []
            for i in range(len(X)):
                if X[i] < 0.1:
                    X1.append(X[i])
                    Q21.append(Q2[i])
                else:
                    X2.append(X[i])
                    Q22.append(Q2[i])
            label = None
            if exp == 'idis':
                if idx==10016:   marker,color,label  = '^','black',    'BCDMS'
                elif idx==10017: marker,color        = '^','black'
                #elif idx==10050: marker,color,label  = '*','deeppink', 'MARATHON'
                #elif idx==10051: marker,color        = '*','deeppink'
                #elif idx==10052: marker,color        = '*','deeppink'
                elif idx==10020: marker,color,label  = '+','goldenrod','NMC'
                elif idx==10021: marker,color        = '+','goldenrod'
                elif idx==10010: marker,color,label  = 'v','blue',     'SLAC'
                elif idx==10011: marker,color        = 'v','blue'
                elif idx==10026: marker,color,label  = 'o','green',    'HERA'
                elif idx==10027: marker,color        = 'o','green'
                elif idx==10028: marker,color        = 'o','green'
                elif idx==10029: marker,color        = 'o','green'
                elif idx==10030: marker,color        = 'o','green'
                elif idx==10031: marker,color        = 'o','green'
                elif idx==10032: marker,color        = 'o','green'
                elif idx==10033: marker,color,label  = 's','orange',   'JLab BONuS'
                elif idx==10002: marker,color,label  = 'x','red',      'JLab Hall C'
                elif idx==10003: marker,color        = 'x','red'
                else: continue
            if exp == 'dy':
                if idx == 10001: marker,color,label  = 'D','magenta','FNAL E866'
            if exp == 'wasym':
                if idx == 1000:  marker,color,label  = 'p','maroon','CDF/D0'
            if exp == 'zrap':
                marker,color,label = 'p', 'maroon', None
            if exp == 'wzrv':
                if idx == 2000: marker,color       = 'p','maroon'
                if idx == 2003: marker,color       = 'p','maroon'
                if idx == 2006: marker,color       = 'p','maroon'
                if idx == 2007: marker,color,label = '*','darkcyan','ATLAS/CMS'
                if idx == 2009: marker,color       = '*','darkcyan'
                if idx == 2010: marker,color       = '*','darkcyan'
                if idx == 2011: marker,color       = '*','darkcyan'
                if idx == 2012: marker,color       = '*','darkcyan'
                if idx == 2013: marker,color       = '*','darkcyan'
                if idx == 2014: marker,color       = '*','darkcyan'
                if idx == 2015: marker,color       = '*','darkcyan'

            ax .scatter(X1,Q21,c=color,label=label,s=s,marker=marker)
            hand[label] = axL.scatter(X2,Q22,c=color,s=s,marker=marker)


    #--Plot cuts
    x = np.linspace(0.1,0.9,100)
    W2cut10_p=np.zeros(len(x))
    W2cut10_d=np.zeros(len(x))
    W2cut3_p=np.zeros(len(x))
    W2cut3_d=np.zeros(len(x))
    Q2cut=np.ones(len(x))*1.3**2

    for i in range(len(x)):
        W2cut10_p[i]=(10.0-(0.938)**2)*(x[i]/(1-x[i]))
        W2cut10_d[i]=(10.0-(1.8756)**2)*(x[i]/(1-x[i]))
        W2cut3_p[i]=(3.0-(0.938)**2)*(x[i]/(1-x[i]))
        W2cut3_d[i]=(3.0-(1.8756)**2)*(x[i]/(1-x[i]))

    hand['W2=10'] ,= axL.plot(x,W2cut10_p,'k--')
    hand['W2=3']  ,= axL.plot(x,W2cut3_p,c='k')

    ax.axvline(0.1,color='black',ls=':',alpha=0.5)

    ax .tick_params(axis='both',which='both',top=True,right=False,direction='in',labelsize=30)
    axL.tick_params(axis='both',which='both',top=True,right=True,labelright=False,direction='in',labelsize=30)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(2e-5,0.1)
    ax.set_ylim(1,10e4)
    ax. set_xticks([1e-4,1e-3,1e-2])
    ax. set_xticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'])
    axL.set_xticks([0.1,0.3,0.5,0.7])

    axL.set_xlabel(r'\boldmath$x$',size=40)
    axL.xaxis.set_label_coords(0.95,0.00)
    ax.set_ylabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    handles,labels = [], []
    handles.append(hand['BCDMS'])
    handles.append(hand['NMC'])
    handles.append(hand['SLAC'])
    handles.append(hand['JLab BONuS'])
    handles.append(hand['JLab Hall C'])
    handles.append(hand['HERA'])
    handles.append(hand['FNAL E866'])
    handles.append(hand['CDF/D0'])
    handles.append(hand['ATLAS/CMS'])
    #handles.append(hand['MARATHON'])
    handles.append(hand['W2=10'])
    handles.append(hand['W2=3'])
    labels.append(r'\textbf{\textrm{BCDMS}}')
    labels.append(r'\textbf{\textrm{NMC}}')
    labels.append(r'\textbf{\textrm{SLAC}}')
    labels.append(r'\textbf{\textrm{JLab BONuS}}')
    labels.append(r'\textbf{\textrm{JLab Hall C}}')
    labels.append(r'\textbf{\textrm{HERA}}')
    labels.append(r'\textbf{\textrm{FNAL E866}}')
    labels.append(r'\textbf{\textrm{CDF/D0}}')
    labels.append(r'\textbf{\textrm{ATLAS/CMS}}')
    #labels.append(r'\textbf{\textrm{MARATHON}}')
    labels.append(r'\boldmath$W^2 = 10$' + ' ' + r'\textbf{\textrm{GeV}}' + r'\boldmath$^2$')
    labels.append(r'\boldmath$W^2 = 3$' + '  ' + r'\textbf{\textrm{GeV}}' + r'\boldmath$^2$')
    ax.legend(handles,labels,loc='upper left',fontsize=20,frameon=False, handlelength = 1.0, handletextpad = 0.1)

    py.tight_layout()
    filename='%s/gallery/kinematics-upol'%wdir
    filename+='.png'

    py.savefig(filename)
    print ('Saving figure to %s'%filename)
 
def plot_kin_pol(wdir,data,W2cut):

    nrows,ncols=1,1
    fig = py.figure(figsize=(ncols*14,nrows*8))
    ax=py.subplot(nrows,ncols,1)

    divider = make_axes_locatable(ax)
    axL = divider.append_axes("right",size=6,pad=0,sharey=ax)
    if W2cut==10: axL.set_xlim(0.1,0.7)
    if W2cut==4:  axL.set_xlim(0.1,0.9)
    axL.spines['left'].set_visible(False)
    axL.yaxis.set_ticks_position('right')
    py.setp(axL.get_xticklabels(),visible=True)

    ax.spines['right'].set_visible(False)

    hand = {}
    for exp in data:
        hand[exp] = {}
        for idx in data[exp]:
            X,Q2 = get_kin(exp,data[exp][idx])
            X1,  X2  = [], []
            Q21, Q22 = [], []
            for i in range(len(X)):
                if X[i] < 0.1:
                    X1.append(X[i])
                    Q21.append(Q2[i])
                else:
                    X2.append(X[i])
                    Q22.append(Q2[i])
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

            if exp=='wzrv':
                if   idx==1000:    marker,color,label = '*', 'darkcyan', 'STAR W'
                elif idx==1001:    marker,color       = '*', 'darkcyan'
                s=80

            if exp=='pjet':
                if   idx==20001:   marker,color,label = 's', 'black', 'STAR jets'
                elif idx==20002:   marker,color       = 's', 'black' 
                elif idx==20003:   marker,color       = 's', 'black' 
                elif idx==20004:   marker,color       = 's', 'black' 
                elif idx==20006:   marker,color       = 's', 'black' 
                elif idx==20005:   marker,color,label = 'v', 'blue', 'PHENIX jets' 
                s=40

            ax .scatter(X1,Q21,c=color,label=label,s=s,marker=marker)
            hand[label] = axL.scatter(X2,Q22,c=color,s=s,marker=marker)

    #--Plot cuts
    x = np.linspace(0.1,0.9,100)
    W2cut10=np.zeros(len(x))
    W2cut4=np.zeros(len(x))
    Q2cut=np.ones(len(x))*1.3**2

    for i in range(len(x)):
        W2cut10[i]=(10.0-(0.938)**2)*(x[i]/(1-x[i]))
        W2cut4[i]=(4.0-(0.938)**2)*(x[i]/(1-x[i]))

    if W2cut == 10.0: axL.plot(x,W2cut10,'k--')
    if W2cut == 4.0:  
        hand['W2=4']  ,= axL.plot(x,W2cut4,c='k')
        hand['W2=10'] ,=axL.plot(x,W2cut10,'k--')

    ax.axvline(0.1,color='black',ls=':',alpha=0.5)

    ax .tick_params(axis='both',which='both',top=True,right=False,direction='in',labelsize=30)
    axL.tick_params(axis='both',which='both',top=True,right=True,labelright=False,direction='in',labelsize=30)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1.0,1.5e4)
    ax.set_xlim(5e-3,0.1)
    ax. set_xticks([1e-2])

    if W2cut==4:
        axL.set_xticks([0.1,0.3,0.5,0.7])
    if W2cut==10:
        axL.set_xticks([0.1,0.3,0.5])

    axL.set_xlabel(r'\boldmath$x$',size=40)
    axL.xaxis.set_label_coords(0.95,0.00)
    ax.set_ylabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    handles,labels = [],[]
    handles.append(hand['EMC'])
    handles.append(hand['SMC'])
    handles.append(hand['COMPASS'])
    handles.append(hand['HERMES'])
    handles.append(hand['SLAC'])
    if W2cut==4.0: handles.append(hand['JLab'])
    handles.append(hand['STAR jets'])
    handles.append(hand['PHENIX jets'])
    handles.append(hand['STAR W'])
    handles.append(hand['W2=10'])
    if 'W2=4' in hand: handles.append(hand['W2=4'])
    labels.append(r'\textbf{\textrm{EMC}}')
    labels.append(r'\textbf{\textrm{SMC}}')
    labels.append(r'\textbf{\textrm{COMPASS}}')
    labels.append(r'\textbf{\textrm{HERMES}}')
    labels.append(r'\textbf{\textrm{SLAC}}')
    if W2cut==4.0: labels.append(r'\textbf{\textrm{JLab}}')
    labels.append(r'\textbf{\textrm{STAR jets}}')
    labels.append(r'\textbf{\textrm{PHENIX jets}}')
    labels.append(r'\textbf{\textrm{STAR W$^{\pm}$}}')
    labels.append(r'\boldmath$W^2 = 10~{\rm GeV}^2$')
    if 'W2=4' in hand: labels.append(r'\boldmath$W^2 = 4~{\rm GeV}^2$')
    ax.legend(handles,labels,loc='upper left',fontsize=25,frameon=False, handlelength = 1.0, handletextpad = 0.1)

    py.tight_layout()
    filename='%s/gallery/kinematics-pol'%wdir
    filename+='.png'

    py.savefig(filename)
    print ('Saving figure to %s'%filename)









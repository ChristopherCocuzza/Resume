#!/usr/bin/env python

import sys,os
import numpy as np
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

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
from obslib.jets.reader    import READER as jetREAD
from obslib.pjets.reader   import READER as pjetREAD
from qcdlib import aux

conf['aux']=aux.AUX()

cwd = 'plots/marathon/'
#--make kinematic plot

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

def load_data(W2cut=3):

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

    return data

if __name__ == "__main__":

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

    data = load_data()

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
                elif idx==10050: marker,color,label  = 'o','deeppink', 'MARATHON'
                elif idx==10051: marker,color        = 'o','deeppink'
                else: continue
            if exp == 'dy':
                if idx == 10001: marker,color,label  = 'D','purple','FNAL E866'
            if exp == 'wasym':
                if idx == 1000:  marker,color,label  = 'p','maroon','CDF/D0 W/Z'
            if exp == 'zrap':
                marker,color,label = 'p', 'maroon', None
            if exp == 'wzrv':
                if idx == 2000: marker,color       = 'p','maroon'
                if idx == 2003: marker,color       = 'p','maroon'
                if idx == 2006: marker,color       = 'p','maroon'
                if idx == 2007: marker,color,label = '*','darkcyan','ATLAS/CMS W'
                if idx == 2009: marker,color       = '*','darkcyan'
                if idx == 2010: marker,color       = '*','darkcyan'
                if idx == 2011: marker,color       = '*','darkcyan'
                if idx == 2012: marker,color       = '*','darkcyan'
                if idx == 2013: marker,color       = '*','darkcyan'
                if idx == 2014: marker,color       = '*','darkcyan'
                if idx == 2015: marker,color       = '*','darkcyan'
            if exp == 'jet':
                if idx== 10001: marker,color,label = '+','gray','CDF/D0 Jets'
                if idx== 10002: marker,color       = '+','gray'
                if idx== 10003: marker,color,label = 'x','blue','STAR Jets'
                if idx== 10004: marker,color       = 'x','blue'


            s, zorder,edgecolors,linewidths = 35,1,'face',1.5
            if exp=='idis' and idx in [10050,10051]: s,zorder,edgecolors,linewidths = 200, 3, 'black',2.0
            ax               .scatter(X1,Q21,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)
            hand[label] = axL.scatter(X2,Q22,c=color,s=s,marker=marker,zorder=zorder,edgecolors=edgecolors,linewidths=linewidths)


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

    hand['W2=10'] ,= axL.plot(x,W2cut10_p,'k--',zorder=2)
    hand['W2=3']  ,= axL.plot(x,W2cut3_p, c='k',zorder=2)

    ax.axvline(0.1,color='black',ls=':',alpha=0.5)

    ax .tick_params(axis='both',which='both',top=True,right=False,direction='in',labelsize=30)
    axL.tick_params(axis='both',which='both',top=True,right=True,labelright=False,direction='in',labelsize=30)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(2e-5,0.1)
    ax.set_ylim(1,6e5)
    ax. set_xticks([1e-4,1e-3,1e-2])
    ax. set_xticklabels([r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'])
    axL.set_xticks([0.1,0.3,0.5,0.7])

    axL.set_xlabel(r'\boldmath$x$',size=40)
    axL.xaxis.set_label_coords(0.95,0.00)
    ax.set_ylabel(r'\boldmath$Q^2$' + '  ' + r'\textbf{\textrm{(GeV}}' + r'\boldmath$^2)$', size=40)

    hand['blank'] ,= ax.plot(0,0,alpha=0)

    handles,labels = [], []
    handles.append(hand['MARATHON'])
    handles.append(hand['BCDMS'])
    handles.append(hand['NMC'])
    handles.append(hand['SLAC'])
    handles.append(hand['JLab BONuS'])
    handles.append(hand['JLab Hall C'])
    handles.append(hand['HERA'])
    handles.append(hand['FNAL E866'])
    handles.append(hand['CDF/D0 W/Z'])
    handles.append(hand['ATLAS/CMS W'])
    handles.append(hand['CDF/D0 Jets'])
    handles.append(hand['STAR Jets'])
    handles.append(hand['W2=10'])
    handles.append(hand['W2=3'])
    handles.append(hand['blank'])
    handles.append(hand['blank'])
    handles.append(hand['blank'])
    handles.append(hand['blank'])
    handles.append(hand['blank'])
    labels.append(r'\textbf{\textrm{MARATHON}}')
    labels.append(r'\textbf{\textrm{BCDMS}}')
    labels.append(r'\textbf{\textrm{NMC}}')
    labels.append(r'\textbf{\textrm{SLAC}}')
    labels.append(r'\textbf{\textrm{JLab BONuS}}')
    labels.append(r'\textbf{\textrm{JLab Hall C}}')
    labels.append(r'\textbf{\textrm{HERA}}')
    labels.append(r'\textbf{\textrm{FNAL E866}}')
    labels.append(r'\textbf{\textrm{CDF/D0 W/Z}}')
    labels.append(r'\textbf{\textrm{ATLAS/CMS W}}')
    labels.append(r'\textbf{\textrm{CDF/D0 Jets}}')
    labels.append(r'\textbf{\textrm{STAR Jets}}')
    labels.append(r'\boldmath$W^2 = 10$' + ' ' + r'\textbf{\textrm{GeV}}' + r'\boldmath$^2$')
    labels.append(r'\boldmath$W^2 = 3$' + '  ' + r'\textbf{\textrm{GeV}}' + r'\boldmath$^2$')
    labels.append(r'')
    labels.append(r'')
    labels.append(r'')
    labels.append(r'')
    labels.append(r'')
    ax.legend(handles,labels,loc='upper left',fontsize=20,frameon=False, handlelength = 1.2, handletextpad = 0.1, ncol = 2, columnspacing = 1.0)

    py.tight_layout()
    filename='%s/gallery/kin'%cwd
    filename+='.png'

    checkdir('%s/gallery'%cwd)
    py.savefig(filename)
    print ('Saving figure to %s'%filename)
 








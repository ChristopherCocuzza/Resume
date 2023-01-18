#!/usr/bin/env python
import sys,os
import numpy as np
import pylab as py
try: import lhapdf
except: pass

class QPDCALC:

    def __init__(self,name,central_only=False,ismc=False):
        print ('loading ',name)
        self.name=name
        self.ismc=ismc
        self.central_only=central_only
        if central_only==True:
            self.central=lhapdf.mkPDF(name,0)
        else:
            self.SETS=lhapdf.mkPDFs(name)
  
    def _get_xpdf(self,Set,flav,x,Q2):
        if   flav=='g':     xpdf= Set.xfxQ2(21,x,Q2)
        elif flav=='uv':    xpdf= Set.xfxQ2(2,x,Q2)-Set.xfxQ2(-2,x,Q2)
        elif flav=='dv':    xpdf= Set.xfxQ2(1,x,Q2)-Set.xfxQ2(-1,x,Q2)
        elif flav=='d/u':
            if x > 0.95: xpdf = 0   
            else:        xpdf= Set.xfxQ2(1,x,Q2)/Set.xfxQ2(2,x,Q2)
        elif flav=='db/ub':
            if x > 0.95: xpdf = 0   
            else:        xpdf= Set.xfxQ2(-1,x,Q2)/Set.xfxQ2(-2,x,Q2)
        elif flav=='ub':    xpdf= Set.xfxQ2(-2,x,Q2)
        elif flav=='db':    xpdf= Set.xfxQ2(-1,x,Q2)
        elif flav=='sb':    xpdf= Set.xfxQ2(-3,x,Q2)
        elif flav=='u':     xpdf= Set.xfxQ2(2,x,Q2)
        elif flav=='d':     xpdf= Set.xfxQ2(1,x,Q2)
        elif flav=='s':     xpdf= Set.xfxQ2(3,x,Q2)
        elif flav=='c':     xpdf= Set.xfxQ2(4,x,Q2)
        elif flav=='b':     xpdf= Set.xfxQ2(5,x,Q2)
        elif flav=='db+ub': xpdf= Set.xfxQ2(-2,x,Q2)+Set.xfxQ2(-1,x,Q2)
        elif flav=='db-ub': xpdf= Set.xfxQ2(-1,x,Q2)-Set.xfxQ2(-2,x,Q2)
        elif flav=='ub-db': xpdf= Set.xfxQ2(-2,x,Q2)-Set.xfxQ2(-1,x,Q2)
        elif flav=='s-sb':  xpdf= Set.xfxQ2(3,x,Q2)-Set.xfxQ2(-3,x,Q2)
        elif flav=='s+sb':  xpdf= Set.xfxQ2(3,x,Q2)+Set.xfxQ2(-3,x,Q2)
        elif flav=='dp':    xpdf= Set.xfxQ2(1,x,Q2)+Set.xfxQ2(-1,x,Q2)
        elif flav=='up':    xpdf= Set.xfxQ2(2,x,Q2)+Set.xfxQ2(-2,x,Q2)
        elif flav=='sp':    xpdf= Set.xfxQ2(3,x,Q2)+Set.xfxQ2(-3,x,Q2)
        elif flav=='Rs':
            num=Set.xfxQ2(3,x,Q2)+Set.xfxQ2(-3,x,Q2)
            den=Set.xfxQ2(-2,x,Q2)+Set.xfxQ2(-1,x,Q2)
            if den==0: xpdf=np.nan
            else:      xpdf=num/den
        elif flav=='rs':
            num=Set.xfxQ2(3,x,Q2)+Set.xfxQ2(-3,x,Q2)
            den=2*Set.xfxQ2(-1,x,Q2)
            if den==0: xpdf=np.nan
            else:      xpdf=num/den
        elif flav=='d/u': 
            xpdf= Set.xfxQ2(1,x,Q2)/Set.xfxQ2(2,x,Q2)
        elif flav=='db/ub':
            num= Set.xfxQ2(-1,x,Q2)
            den= Set.xfxQ2(-2,x,Q2)
            if den==0: xpdf=np.nan
            else:      xpdf=num/den
        else:
            print ('ERR: %s not available'%flav)
            sys.exit()
        if np.isnan(xpdf): xpdf=0
        return xpdf
  
    def _get_symmetric_errors(self,OBS):
        n=len(OBS)-1
        feven=np.array([OBS[2*i] for i in range(1,int(n/2))])
        fodd=np.array([OBS[2*i-1] for i in range(1,int(n/2))])
        df=np.zeros(feven[0].size)
        for i in range(int(n/2)-1):
          df+=(fodd[i]-feven[i])**2
        return df**0.5/2
  
    def _get_asymmetric_errors(self,OBS):
        n=len(OBS)-1
        f0=np.array(OBS[0])
        feven=np.array([OBS[2*i] for i in range(1,int(n/2))])
        fodd=np.array([OBS[2*i-1] for i in range(1,int(n/2))])
        dfeven=feven-f0
        dfodd=fodd-f0
        zeros=np.zeros(f0.size)
        dfP=np.zeros(f0.size)
        dfM=np.zeros(f0.size)
        for i in range(int(n/2)-1):
          dfP+=np.amax([dfodd[i],dfeven[i],zeros],0)**2
          dfM+=np.amax([-dfodd[i],-dfeven[i],zeros],0)**2
        return dfP**0.5,dfM**0.5
  
    def get_xpdf(self,flav,X,Q2):
        D={}
        D['X']=X
        D['Q2']=Q2
        if  self.central_only:
            D['xf0']=np.array([self._get_xpdf(self.central,flav,x,Q2) for x in X])
            D['dxf']=np.zeros(X.size)
            D['dxf+']=np.zeros(X.size)
            D['dxf-']=np.zeros(X.size)
        else:
            PDFS=[[self._get_xpdf(Set,flav,x,Q2) for x in X] for Set in self.SETS]
            if  self.ismc==False:
                D['xf0']=np.array(PDFS[0])
                D['dxf']=self._get_symmetric_errors(PDFS)
                D['dxf+'],D['dxf-']=self._get_asymmetric_errors(PDFS)
            else:
                D['xf0']=np.mean(PDFS,axis=0)
                D['dxf']=np.var(PDFS,axis=0)**0.5
        D['xfmin']=D['xf0']-D['dxf']
        D['xfmax']=D['xf0']+D['dxf']
        return D

if __name__=="__main__":


    pdf=QPDCALC('JAM19PDF_proton_nlo',ismc=True)

    print (pdf.get_xpdf('u',[0.1,0.2],10.))






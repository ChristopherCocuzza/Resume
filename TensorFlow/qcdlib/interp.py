import sys,os
import numpy as np
from numba import jit


@jit(nopython=True)
def locate(xx,x):
    """
    Finds an index iLoc such that xx(iLoc) <= x < xx(iLoc+1)
    Assumes the sequence is ordered in ascending order.
    """
    if x<np.amin(xx): return 0
    else:
        return np.where((x>=xx))[0][-1]

@jit(nopython=True)
def get_irange(xx,x,n):
    i=locate(xx,x) 
    ini=i-int(n/2)
    if ini<0: ini=0
    fin=ini+n
    if fin>len(xx): 
        fin=len(xx)
        ini=fin-n
    return ini,fin

@jit(nopython=True)
def polint(xa,ya,x):
    NMAX=10 #--Largest anticipated value of n.
    c=np.zeros(NMAX)
    d=np.zeros(NMAX)
    if len(xa)>10:
        print('ERR: len of xa>NMAX')
    n=len(xa)
    ns=0
    dif=abs(x-xa[0])

    for i in range(n): #--Here we find the index ns of the closest table entry, 
        dift=abs(x-xa[i])
        if (dift<dif):
            ns=i
            dif=dift 
        c[i]=ya[i] #--and initialize the tableau of c’s and d’s.
        d[i]=ya[i] 
    
    y=ya[ns] #--This is the initial approximation to y.
    ns=ns-1
    for m in range(1,n):
        for i in range(1,n-m+1):
            #print(m,i)
            ho=xa[i-1]-x
            hp=xa[i+m-1]-x
            w=c[i+1-1]-d[i-1]
            den=ho-hp
            if(den==0.): continue #-- ’failure in polint’
            den=w/den
            d[i-1]=hp*den
            c[i-1]=ho*den 
        if (2*ns<n-m): 
            dy=c[ns+1]
        else: 
            dy=d[ns]
            ns=ns-1
        #print(dy)
        y=y+dy 
    return y

@jit(nopython=True)
def interpolate1d(xx,yy,x,n):
    ini,fin=get_irange(xx,x,n)
    return polint(xx[ini:fin],yy[ini:fin],x)

@jit(nopython=True)
def polin2(x1a,x2a,ya,x1,x2):
    m=len(x1a)
    n=len(x2a)
    ymtmp=np.zeros(m)
    yntmp=np.zeros(n)
    for j in range(m):
        for k in range(n): 
            yntmp[k]=ya[j,k]
        ymtmp[j]=polint(x2a,yntmp,x2) 
    return polint(x1a,ymtmp,x1) 
    
@jit(nopython=True)
def interpolate2d(xx,yy,zz,x,y,nx,ny):
    inix,finx=get_irange(xx,x,nx)
    iniy,finy=get_irange(yy,y,ny)
    return polin2(xx[inix:finx],yy[iniy:finy],zz[inix:finx,iniy:finy],x,y)

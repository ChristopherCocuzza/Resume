import sys,os,time
import numpy as np
from numba import jit
cache=True

#--mpmath
from mpmath import fp

import qcdlib.params as par

@jit(nopython=True, cache=cache)
def get_Nf(mu2):
    Nf=3
    if mu2>=(par.mc2): Nf+=1
    if mu2>=(par.mb2): Nf+=1
    return Nf

#--setup the beta function
beta=np.zeros((7,4))
for Nf in range(3,7): 
    beta[Nf,0]=11-2/3*Nf 
    beta[Nf,1]=102-38/3*Nf 
    beta[Nf,2]=2857/2-5033/18*Nf+325/54*Nf**2 
    beta[Nf,3]=2857/54 * par.CA**3 + (2*par.CF**2 - 205/9 * par.CF*par.CA - 1415/27 * par.CA**2) * par.TR*Nf +\
                (44/9 * par.CF + 158/27 * par.CA) * par.TR**2 * Nf**2


@jit(nopython=True, cache=cache)
def beta_func(a,Nf,order):
    betaf = -beta[Nf,0]
    if order>=1: betaf+=-a**1*beta[Nf,1]
    if order>=2: betaf+=-a**2*beta[Nf,2]
    if order>=3: betaf+=-a**3*beta[Nf,3]
    return betaf*a**2

@jit(nopython=True, cache=cache)
def evolve_a(Q20,a,Q2,Nf,order):
    # Runge-Kutta 
    LR = np.log(Q2/Q20)/20
    for k in range(20):
        XK0 = LR * beta_func(a,Nf,order)
        XK1 = LR * beta_func(a + 0.5 * XK0,Nf,order)
        XK2 = LR * beta_func(a + 0.5 * XK1,Nf,order)
        XK3 = LR * beta_func(a + XK2,Nf,order)
        a+= (XK0 + 2* XK1 + 2* XK2 + XK3) * 0.166666666666666
    return a

global a0,ac,ab
a0=0; ac=0; ab=0


@jit(nopython=True, cache=cache)
def get_a(Q2,order):
    if par.mb2<=Q2:
        return evolve_a(par.mb2,ab,Q2,5,order)
    elif par.mc2<=Q2 and Q2<par.mb2: 
        return evolve_a(par.mc2,ac,Q2,4,order)
    elif Q2<par.mc2:
        return evolve_a(par.mu02_alphaS,a0,Q2,3,order)


#- returns alpha_S(Q2)
@jit(nopython=True, cache=cache)
def get_alphaS(Q2,order):
    return get_a(Q2,order)*4*np.pi


def setup(order):
    global a0,ac,ab
    aZ=par.alphaSMZ/(4*np.pi)
    ab=evolve_a(par.mZ2,aZ,par.mb2,5,order)
    ac=evolve_a(par.mb2,ab,par.mc2,4,order)
    a0=evolve_a(par.mc2,ac,par.mu02_alphaS,3,order)

setup(par.order_alphaS)

if __name__=='__main__':
    
    setup(par.order_alphaS)
    print(get_alphaS(10.0))


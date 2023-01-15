from mpmath import fp
import numpy as np
from tools.config import conf

class AUX:

  def __init__(self):

    pass

  #--Q2 is input as an array
  def get_aX(self,aX,flav,Q2):

    ############################################
    ############################################

    #--which one of these is correct?
    if flav in ['u','c']:        eq =  2.0/3.0
    if flav in ['d','s','b']:    eq = -1.0/3.0
    if flav in ['ub','cb']:      eq = -2.0/3.0
    if flav in ['db','sb','bb']: eq =  1.0/3.0

    s = Q2
    alpha = np.array([conf['eweak'].get_alpha(q2) for q2 in Q2])
    sin2w = np.array([conf['eweak'].get_sin2w(q2) for q2 in Q2])

    if flav in ['u','c','ub','cb']:          aq =  1 - 8/3 * sin2w
    if flav in ['d','s','b','db','sb','bb']: aq = -1 + 4/3 * sin2w

    if flav in ['u','c','ub','cb']:          bq = -1
    if flav in ['d','s','b','db','sb','bb']: bq =  1

    aZ = -1 + 4*sin2w

    GF  = conf['aux'].GF
    mZ2 = conf['aux'].mZ2

    LZ = 2.4952

    prop = (s-mZ2)**2 + LZ**2 * mZ2

    Fgg = 1.0
    FZZ = GF**2 * mZ2**2 * s**2 / (128 * np.pi**2 * prop) / alpha**2
    FgZ = GF * mZ2 * s * (s - mZ2) / (4 * np.sqrt(2) * np.pi * prop) / alpha

    if aX=='plus':
        F1qp = eq**2 * Fgg + (1 + aZ**2)*(aq**2 + bq**2)*FZZ - aZ * eq * aq * FgZ
        return F1qp
    if aX=='minus':
        F1qm = eq**2 * Fgg + (1 + aZ**2)*(aq**2 - bq**2)*FZZ - aZ * eq * aq * FgZ
        return F1qm

    






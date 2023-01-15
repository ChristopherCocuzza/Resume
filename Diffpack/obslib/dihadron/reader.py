import sys
import pandas as pd
from qcdlib.aux import AUX
from tools.reader import _READER
from tools.config import conf

class READER(_READER):
  
    def __init__(self):
        pass
  
    def get_idx(self,tab):
        tab['idx']=pd.Series(tab.index,index=tab.index)
        return tab

    def get_limit(self,tab):
        for _ in tab['process']:
            if _ != 'SIA': return tab
        for _ in tab['obs']:
            if _ != 'sig' and _ != 'sig_rat': return tab
        cols=tab.columns.values
        if any([c=='limit' for c in cols])==False:
            tab['limit']            = tab['RS']/2 * tab['zup']
            tab['lim_ratio']        = tab['M']/tab['limit']
            tab['limit_BELLE']      = 10.58/2 * tab['zup']
            tab['lim_ratio_BELLE']  = tab['M']/tab['limit_BELLE']

        return tab
  
    def modify_table(self,tab):
        tab=self.get_limit(tab)
        tab=self.apply_cuts(tab)
        return tab
  

import sys,os
import numpy as np
import pandas as pd
try: #--for py3
    from   tools.config import conf
    from   tools.tools  import isnumeric
except: #--for py2
    from   config import conf
    from   tools  import isnumeric

class _READER:

  def apply_cuts(self,tab,idx=None):
      
      #--filter all datasets
      if  'filters' in conf['datasets'][self.reaction]:
          for f in conf['datasets'][self.reaction]['filters']:
              try: tab=tab.query(f)
              except: pass

      if idx==None: return tab
      #--filter individual datasets
      if  'idxfilters' in conf['datasets'][self.reaction]:
          for k in conf['datasets'][self.reaction]['idxfilters']:
              if idx != k: continue
              for f in conf['datasets'][self.reaction]['idxfilters'][k]:
                  try: tab=tab.query(f)
                  except: pass

      return tab

  def load_data_sets(self,reaction,verb=True):
      self.reaction=reaction
      if reaction not in conf['datasets']: return None
      XLSX=conf['datasets'][reaction]['xlsx']
      TAB={}
      for k in XLSX: 
          if verb: print('loading %s data sets %s'%(reaction,k))
          fname=conf['datasets'][reaction]['xlsx'][k]
          if  fname.startswith('./'):
              tab=pd.read_excel(fname)
          else:
              tab=pd.read_excel('%s/database/%s'%(os.environ['FITPACK'],fname))
          tab=self.modify_table(tab)
          tab=self.apply_cuts(tab,k)
          npts=tab.index.size
          if npts==0: continue
          TAB[k]=tab.to_dict(orient='list')
          for kk in TAB[k]: 
              if  isnumeric(TAB[k][kk][0]):
                if kk == 'idx':
                  TAB[k][kk]=np.array(TAB[k][kk])
                else:
                  TAB[k][kk]=np.array(TAB[k][kk]).astype(np.float64)
     
      return TAB





import os
from .tools import load, checkdir, save
from .parallel import PARALLEL
import time

class STORAGE(object):

    def __init__(self,path,fname):
        checkdir(path)
        self.fname=path+'/'+fname
        self.load_storage(self.fname)
        self.requests=[]
  
    def load_storage(self,fname):
  
        #--create storage if not created
        if not os.path.exists(fname):
            print(fname+' is been created !')
            self.storage={}
        else:
            self.storage=load(fname)
  
    def gen_key(self,x,Q2):
        #return 'x='+str(x)+',Q2='+str(Q2)
        return 'x=%10.10e,Q2=%10.10e'%(x,Q2)

    def query(self,x,Q2):
        key=self.gen_key(x,Q2)
        return any([key==k for k in self.storage.keys()])
  
    def register(self,func,x,Q2,name):
  
        if self.query(x,Q2)==False:
            key=self.gen_key(x,Q2)
            #self.storage[key]=func(x,Q2)
            data={'key':key,'x':x,'Q2':Q2,'func':func}
            self.requests.append(data)

    def process_request(self,nworkers=20):

        print('processing %s'%self.fname)

        def task(data):
            x=data['x']
            Q2=data['Q2']
            key=data['key']
            func=data['func']
            return {'key':key,'value':func(x,Q2)}

        parallel=PARALLEL(verb=True)
        parallel.task=task

        parallel.setup_master()
        parallel.setup_workers(nworkers)

        t=time.time()
        results=parallel.send_tasks(self.requests)

        for result in results:
            key=result['key']
            value=result['value']
            self.storage[key]=value

        print(time.time()-t)
        parallel.stop_workers()
  
    def Save(self):
        save(self.storage,self.fname)
  
    def retrieve(self,x,Q2):
        key=self.gen_key(x,Q2)
        return self.storage[key]
  


 

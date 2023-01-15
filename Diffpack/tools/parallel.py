#!/usr/bin/env python
import sys,os
import time
import random
import zmq
import multiprocessing
import numpy as np
import cProfile
import distutils.util
try:
  import dill as pickle
except:
  try:
      import cPickle as pickle
  except:
      import _pickle as pickle

class PARALLEL:

    def __init__(self,verb=False):
        self.verb=verb

    def lprint(self,msg):
        sys.stdout.write('\r')
        sys.stdout.write(msg)
        sys.stdout.flush()

    def setup_master(self):

        context = zmq.Context()
        self.sock = context.socket(zmq.REP)
        self.port=self.sock.bind_to_random_port("tcp://*", min_port=49152, max_port=65536, max_tries=100)

    def setup_workers(self,nworkers):
        
        for idx in range(nworkers): 
            if distutils.util.strtobool(os.getenv('PARALLEL2_PROFILE_WORKERS','False')):
                process = multiprocessing.Process(target=self.profile,name='w'+str(idx),args=(idx,))
            else:
                process = multiprocessing.Process(target=self.worker,name='w'+str(idx),args=(idx,))
            #process.daemon=True
            process.start()
            time.sleep(0.1)

        #--set nworkers as data member
        self.nworkers=nworkers

    def send(self,sock,data):
        #sock.send_json(data)
        #sock.send(pickle.dumps(data,pickle.HIGHEST_PROTOCOL))
        sock.send(pickle.dumps(data))

    def recv(self,sock):
        #return sock.recv_json()
        return pickle.loads(sock.recv())

    def profile(self,idx):
        cProfile.runctx('self.worker(idx)',globals(),locals(),'worker-%d.prof' %idx)
 
    def worker(self,idx):

        #print 'worker %d is ready'%idx  
        context = zmq.Context()
        sock = context.socket(zmq.REQ)
        sock.connect("tcp://localhost:%d"%self.port) #--IP of master

        while True:
            self.send(sock,{ "msg": "available",'worker':idx})
            work = self.recv(sock)

            if 'task0' in work: 
                #print "running taks %d " % (work['task'])
                result=self.task(work['task0'])
                self.send(sock,{ "msg": "result", "result": result})
                sock.recv()

            elif 'task1' in work: 
                #print "running taks %d " % (work['task'])
                result=self.task1(work['task1'])
                self.send(sock,{ "msg": "result", "result": result})
                sock.recv()

            elif 'state' in work:
                self.set_state(work['state'])
                self.send(sock,{ "msg": "updated",'idx':idx})
                sock.recv()

            elif 'end' in work:
                #print 'worker %d out'%idx
                break 

    def update_workers(self,state):

        sent=0
        worker_states=[0 for _ in range(self.nworkers)]
        
        while True:
            #print worker_states
            recv = self.recv(self.sock)

            if recv['msg'] == "available":

                if sum(worker_states)<self.nworkers:
                    self.send(self.sock,{'state':state})

            elif recv['msg'] == "updated":
                idx=recv['idx']
                worker_states[idx]=1
                self.send(self.sock,{})
                if sum(worker_states)==self.nworkers: break

        #print 'workers UPDATED'
        #print recv
        #print worker_states

    def send_tasks(self,requests,itask=0):

        t=time.time()
        sent=0
        received=0
        ntasks=len(requests)
        results=[]

        while True:

            if self.verb: self.lprint('%d/%d'%(received,ntasks))

            try:
                recv = self.recv(self.sock)
            except zmq.ZMQError:
                self.send(self.sock,{})
                continue 

            if recv['msg'] == "available":

                if sent<ntasks:
               
                    if   itask==0: self.send(self.sock,{'task0':requests[sent]})
                    elif itask==1: self.send(self.sock,{'task1':requests[sent]})
                    sent+=1

            elif recv['msg'] == "result":

                received+=1
                result=recv['result']
                results.append(result)
                self.send(self.sock,{})

                if received==ntasks: break

        #print 'time elapsed:',time.time()-t
        #print "all the task completed"

        return results

    def stop_workers(self):

        nworkers=self.nworkers
        sent=0

        while True:

            try:
                recv = self.recv(self.sock)
            except zmq.ZMQError:
                self.send(self.sock,{}) 
                continue 

            if recv['msg'] == "available":

              self.send(self.sock,{'end':True})
              sent+=1

            if sent==nworkers: break


        #--close the master socket
        time.sleep(1)
        self.sock.close()

def example1():
    '''
    In this example we show how use the basic distribution 
    of task among the workers
    '''
  
    #--lets define dummy task to be executed in parallel
    ntasks=100
    requests=[None for _ in range(ntasks)]

    ntasks=10
    requests1=[None for _ in range(ntasks)]
  
    def task(data):
        #--data ==  entries of request
        time.sleep(0.1)
        return random.random()

    def task1(data):
        #--data ==  entries of request
        return True

    #--here None could be replace by anything that 
    #--task function can interprete

    #--now we setup the parallel class
    nworkers=3

    parallel=PARALLEL()
    parallel.task=task
    parallel.task1=task1

    parallel.setup_master()
    parallel.setup_workers(nworkers)

    t=time.time()
    results=parallel.send_tasks(requests,0)
    print(results)
    print(time.time()-t)
    results=parallel.send_tasks(requests1,1)
    print(results)
    print(time.time()-t)
    parallel.stop_workers()

def example2():
    '''
    In this example we show how to update the  "state" of the 
    workers. The logic is as follow:

    1. define a set of tasks that depends on a given "state"
    2. create workers 
    3. update the workers state 
    4. distribute the task 
    5. loop over 1-4 by changing the state

    '''

    #--lets define a simple state based on integers
    global state

    state=0
    def set_state(arg): #--external function
        global state
        state=arg

    #-- define the dummy taks that depend on the state
    ntasks=10
    requests=[None for _ in range(ntasks)]

    def task(data):
        #--data ==  entries of request
        time.sleep(0.1) 
        return state   #<--this is how the task depends on the state


    #--now we setup the parallel class
    nworkers=20

    parallel=PARALLEL()
    parallel.task=task
    parallel.set_state=set_state

    parallel.setup_master()
    parallel.setup_workers(nworkers)

    t=time.time()
    #--loop over states 
    for _ in range(10):  

        #--state can be anything
        state=_
        #state={_:'ABC'}
        #state=random.random()
        #state={0:{0:{0:1}}}

        parallel.update_workers(state)
        results=parallel.send_tasks(requests)

        print(results)

    print(time.time()-t)
    parallel.stop_workers()

def example3():
    '''
    a more realistic example
    '''

    class PDF:

        def __init__(self):

            self.params=[1,-0.5,3]

        def get_pdf(self,x):
            N,a,b=self.params
            return N*x**a*(1-x)**b

        def get_state(self):
            return (self.params) #--add more elements as needed   

        def set_state(self,state):
            self.params=state

    global pdf
    pdf=PDF()

    def set_state(state): 
        global pdf
        pdf.set_state(state['pdf'])

    ntasks=1000
    requests=[{'x':_} for _ in np.linspace(0.1,0.9,ntasks)]

    def task(data):
        #--data ==  entries of request
        global pdf
        data['thy']=pdf.get_pdf(data['x'])
        return data

    #--construct states
    nstates=10
    states=[]
    for _ in range(nstates):
        N=np.random.uniform( 0.0,1.0,1)[0]
        a=np.random.uniform(-0.5,0.5,1)[0]
        b=np.random.uniform( 3.0,10.,1)[0]
        states.append({'pdf':[N,a,b]})

    #--now we setup the parallel class
    nworkers=20

    parallel=PARALLEL()
    parallel.print_slave_state=False
    parallel.task=task
    parallel.set_state=set_state

    parallel.setup_master()
    parallel.setup_workers(nworkers)


    t=time.time()
    #--loop over states 
    for state in states:  

        parallel.update_workers(state)
        results=parallel.send_tasks(requests)
        print(results[0])

    #print time.time()-t
    parallel.stop_workers()

def example4():
  
    #--lets define dummy task to be executed in parallel
    ntasks=100
    requests=[{'f':lambda : 1} for _ in range(ntasks)]

    def task(data):
        #--data ==  entries of request
        time.sleep(0.1)
        return data['f']()

    #--here None could be replace by anything that 
    #--task function can interprete

    #--now we setup the parallel class
    nworkers=20

    parallel=PARALLEL()
    parallel.task=task

    parallel.setup_master()
    parallel.setup_workers(nworkers)

    t=time.time()
    results=parallel.send_tasks(requests)
    print(results)
    print(time.time()-t)
    parallel.stop_workers()


if __name__ == "__main__":

    example1()
    #example2()
    #example3()
    #example4()







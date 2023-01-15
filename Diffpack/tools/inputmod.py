#!/usr/bin/env python

class INPUTMOD:

    def __init__(self,path2inputfile):
        self.path2inputfile=path2inputfile
        self.L=open(path2inputfile).readlines()

    def mod_par(self,kind,par,item,value):
        L=self.L[:]
        for i in range(len(L)):
            if ('params' in L[i]) and (kind in L[i]) and ('\'%s\''%par in L[i]):
                head,entry=L[i].split('=')
                entry=entry.strip().replace('{','').replace('}','').split(',')
                for j in range(len(entry)): 
                    if item in entry[j]:
                        key,old_value = entry[j].split(':')
                        entry[j]='%s:%15.5e'%(key,value)
                new_entry=''
                for _ in entry: new_entry+=_+','
                L[i]='%s={%s}\n'%(head,new_entry.rstrip(','))
        self.L=L[:]
    
    def mod_norm(self,reaction,idx,item,value):
        L=self.L[:]
        for i in range(len(L)):
            if ('datasets' in L[i]) and ('norm' in L[i]) and (str(idx) in L[i]):
                #print reaction,L[i].split('][')[1].strip("'")
                if reaction!=L[i].split('][')[1].strip("'"): continue
                head,entry=L[i].split('=')
                entry=entry.strip().replace('{','').replace('}','').split(',')
                for j in range(len(entry)): 
                    if item in entry[j]:
                        key,old_value = entry[j].split(':')
                        entry[j]='%s:%15.5e'%(key,value)
                new_entry=''
                for _ in entry: new_entry+=_+','      
                L[i]='%s={%s}\n'%(head,new_entry.rstrip(','))
        self.L=L[:]
        
    def gen_input(self,path2new_inputfile=None):
        if path2new_inputfile==None: 
            path2new_inputfile=self.path2inputfile
        #print('saveing results at ',path2new_inputfile)
        #F=open(path2new_inputfile,'w')
        #for l in self.L: print l
        #F.writelines(self.L)
        #F.close()

if __name__=='__main__':

    inputmod=INPUTMOD('input.py')
    inputmod.mod_par('pdf','uv a','min',-0.5)
    inputmod.mod_norm('dis',10010,'min',0.99)
    inputmod.gen_input()








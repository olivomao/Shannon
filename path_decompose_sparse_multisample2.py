# -*- coding: utf-8 -*-
"""
modified based on path_decompose_sparse_multisample
- wrap related stuff to matrix_flow class
- add path info P
"""

import numpy as np
import pdb
from cvxopt import solvers, matrix
import copy
from numpy import linalg as LA
import time
import subprocess
import re

'''
Functions - general vector / matrix
'''
def is_zero(v):

    v = np.asarray(v)

    return (v==0).all()


def is_le(v1, v2):

    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    return (v1<=v2).all()
    
def is_eq(v1, v2):

    v1 = np.asarray(v1)
    v2 = np.asarray(v2)

    return (v1==v2).all()


'''
Functions - path_decompose_multisample
'''

def get_dim(in_edges, out_edges):    
    
    M = len(in_edges)
    if M != len(out_edges):
        print('path_decompose_multisample: imbalanced number of samples')
        
    I = len(in_edges[0])
    J = len(out_edges[0])
    
    return [M, I, J]
    
def get_constraints(M, I, J, in_edges, out_edges):
    
    #constraints of G, h
    #f_ij(m)>=0, f_ij(m)<=g_ij
    nG = J*I*(2*M)
    nX = J*I*(M+1)

    G = matrix(0.,(nG, nX))
    h = matrix(0.,(nG, 1))

    for j in range(J):
        for i in range(I):
            for m in range(M):
                #f_ij(m)>=0
                r_idx1 = j*(I*2*M) + i*(2*M) + m
                c_idx1 = j*(I*(M+1)) + i*(M+1) + m
                G[r_idx1, c_idx1] = -1.0

                #f_ij(m)(e.g. m=0~M-1) <=g_ij (e.g. f_ij(M))
                r_idx2 = j*(I*2*M) + i*(2*M) + M + m
                c_idx2_1 = j*(I*(M+1)) + i*(M+1) + m
                c_idx2_2 = j*(I*(M+1)) + i*(M+1) + M
                G[r_idx2, c_idx2_1] = 1.0
                G[r_idx2, c_idx2_2] = -1.0
    

    #constraints of A, b
    nA = M*(J+I-1) #J+I constraints, but if we know w(0)...w(I-2),w'(0)...w'(J-1), we can know w(I-1)

    A = matrix(0., (nA, nX))
    b = matrix(0., (nA, 1))

    for m in range(M):
        for j in range(J):
            r_idx = m*(J+I-1)+j
            b[r_idx, 0] = out_edges[m][j] #E_j(m)
            for i in range(I):
                c_idx = j*I*(M+1) + i*(M+1) + m #f_ij(m) sum through i
                A[r_idx, c_idx] = 1.0

        for i in range(I-1): # we consider w(0) till w(I-2)
            r_idx = m*(J+I-1) + J + i
            b[r_idx, 0] = in_edges[m][i] #E_i(m)
            for j in range(J):
                c_idx = j*I*(M+1) + i*(M+1) + m
                A[r_idx, c_idx] = 1.0
    
    return [A, b, G, h]

def get_coeff(M, I, J, P, c_ref=[]):

    #pdb.set_trace()
    
    c=matrix(np.zeros((J*I*(M+1))))
    
    if c_ref != []:
        for j in range(J):
            for i in range(I):
                
                if P[i,j]==1:#there is known path, no need to estimate g_ij
                    for m in range(M+1):#also mask f_ij(m)
                        idx = j*I*(M+1) + i*(M+1) + m
                        c[idx] = 0 
                else:
                    for m in range(M+1):
                        idx = j*I*(M+1) + i*(M+1) + m
                        c[idx] = c_ref[idx]
    else:
        c=matrix(abs(np.random.normal(0,1,(J*I*(M+1),1))))
        #keep coefficients only corresponding to g_ij (e.g. f_ij(M)) and unknown path (P[i,j]==0)
        for j in range(J):
            for i in range(I):
                
                if P[i,j]==1:#there is known path, no need to estimate g_ij
                    idx = j*I*(M+1) + i*(M+1) + M
                    c[idx] = 0
    
                for m in range(M):#also mask f_ij(m)
                    idx = j*I*(M+1) + i*(M+1) + m
                    c[idx] = 0          
    return c
    
def describe_coeff_list(c_list): #debug purpose
    #pdb.set_trace()
    
    st = ''
    N_Iter = len(c_list)
    for it in range(N_Iter):
        [R,C] = np.shape(c_list[it]) #R = J*I*(M+1), C=1
        st = st + 'iter %2d: '%it
        for r in range(R):
            st = st + '%e\t'%c_list[it][r,0]
        st = st + '\n'
    #pdb.set_trace()
    return st
    
def get_f(x, M, I, J, P=None):

    #convert x (J*I*(M+1) by 1 to f=M by (I by J) and g=I by J)

    f = np.zeros((M, I, J))
    g = np.zeros((I, J))

    for i in range(I):
        for j in range(J):
            for m in range(M):
                idx = j*I*(M+1) + i*(M+1) + m #f_ij(m)
                f[m][i,j] = max(x[idx], 0) #flow is nonzero

            if P is not None and P[i,j]==1:
                #print('check g_ij when path ij known (if g_ij is bounded or unbounded)')
                #pdb.set_trace()
                g[i,j] = max([f[m][i,j] for m in range(M)])
            else:
                #print('check g_ij when path ij is unknown')
                #pdb.set_trace()
                idx = j*I*(M+1) + i*(M+1) + M #g_ij (e.g. f_ij(M))
                g[i][j] = max(x[idx], 0)
            
    f_est = FlowMatrix(f=f,g=g,P=P)

    return f_est
    
def threshold_f(f_est, in_edges, out_edges):
    
    #parameters
    sparsity_factor = 0.1
    
    #threshold f_est here
    f_est_thr = copy.deepcopy(f_est)
    
    g_in_edges  = np.sum(f_est.g, axis=1).tolist()
    g_out_edges = np.sum(f_est.g, axis=0).tolist()
    
    for i in range(f_est.I):
        for j in range(f_est.J):
            if f_est.g[i,j] < sparsity_factor*min(g_in_edges[i], g_out_edges[j]) and f_est.P[i,j]==0: 
                #Currently we keep f[m][i,j] for known path ij, even f[m][i,j] can be small
                f_est_thr.set_zero(i,j)       
    
    #print('check threshold_f')
    #pdb.set_trace()
    return f_est_thr
    
def path_decompose_multisample(in_edges, out_edges, P, sparse_limit, c_list_ref=[], dump_c_list=False, n_iter=-1):
    
    #dimension
    [M, I, J] = get_dim(in_edges, out_edges)
    
    #trivial cases
    
    #make in/out edge flow equal
    
    #constraints for lp problem
    [A, b, G, h] = get_constraints(M, I, J, in_edges, out_edges)

    #trials for random algorithm
    n_trials = 100
    if c_list_ref != []:
        n_trials = len(c_list_ref)

    solvers.options['show_progress'] = False
    
    #track params:
    #pdb.set_trace()
    msr = MinSparsityRes(I*J+1)
    
    #pdb.set_trace()
    msr.set_tolerance(in_edges)    
    
    c_list = []#debug purpose
    for i_trial in range(n_trials):

        #if i_trial==97:#debug purpose
        #    pdb.set_trace()
        
        if c_list_ref != []:
            c = get_coeff(M, I, J, P, c_list_ref[i_trial])
        else:
            c = get_coeff(M, I, J, P)
            if dump_c_list==True:
                c_list.append(c)
        
        sol = solvers.lp(c =c ,G=G,h=h,A=A,b=b)
        x = np.array(sol['x'])

        est_f = get_f(x, M, I, J, P) # [est_f, g] = get_f(x, M, I, J); f_est: flow matrix

        #thresholding
        #pdb.set_trace()
        est_f_thr = threshold_f(est_f, in_edges, out_edges) 
        
        #pdb.set_trace()
        f_before = msr.min_sparsity_f

        flag = msr.update(f=est_f_thr, trial_idx=i_trial)

        if False: #n_iter == 45 and flag!='' and i_trial>0:#debug, avoid the 1st update
            #pdb.set_trace()
            st = syncstr(i_trial)
            LogObj.log_f1.write(f_before.description('i_trial=%d before update'%i_trial))
            LogObj.log_f1.write('%s\n\n'%st)# to align comparison view in meld software
            LogObj.log_f2.write(msr.min_sparsity_f.description('i_trial=%d after update [%s]'%(i_trial, flag)))
            LogObj.log_f2.write('%s\n\n'%st)
        #    #pdb.set_trace()

    #sparse limit
    #msr.postprocess(P)
    #candidate_sols = msr.postprocess2(msr.min_sparsity_f)#reduce X or rotate Z and reduce X
    #for candidate_sol in candidate_sols:
    #    msr.update2(f=candidate_sol, g=None, P=None, trial_idx=i_trial)

    #pdb.set_trace()
    return [msr.min_sparsity_f, c_list]
    
'''
Class definitions
'''
class FlowMatrix:# an object to keep track of f
        
    def __init__(self, f, g=None, P=None): #f:MxIxJ flow info; P:IxJ path info
    
        [self.M, self.I, self.J] = np.shape(f)
        
        self.f = copy.copy(f)
        
        if P is not None:
            self.P = copy.copy(P)
        else:
            self.P = np.zeros((self.I, self.J)) #all entries are unknown paths
            
        if g is not None:
            self.g = copy.copy(g)
        else:
            self.g = np.zeros((self.I, self.J))
            for i in range(self.I):
                for j in range(self.J):
                    self.g[i,j] = max([self.f[m][i,j] for m in range(self.M)])
    
    def set_zero(self, i, j):
        
        self.g[i, j] = 0
        
        for m in range(self.M):
            self.f[m][i,j] = 0
            
    def get_sparsity_on_unknown_paths(self):        
        #pdb.set_trace()
        sparsity_unknown_paths = np.count_nonzero(self.g*(1-self.P)) #count P[i,j]==0 (unknown path) and g[i,j]>0
        return sparsity_unknown_paths
        
    def calc_flow_on_paths(self):#calc_tot_flow_on_unknown_paths(self):    
        #pdb.set_trace()        
        tot_flow_unknown = 0
        tot_flow_known = 0
            
        for m in range(self.M):            
            tot_flow_unknown += sum(sum(self.f[m]*(1-self.P)))
            tot_flow_known += sum(sum(self.f[m]*self.P))
        
        tot_flow = tot_flow_unknown + tot_flow_known
        
        return [tot_flow_unknown, tot_flow_known, tot_flow]
        
    def diff(self, f2):
        #difference (distance) of two flow matrice
        return LA.norm(self.f-f2.f)
        
    def find_cycle(self):
        #X repeat or general cycle
        return
        
    def find_alt_sparsity_pattern(self):
        #no Z pattern that can be confusable
        #Z pattern that can be confusable
        return
        
    def description(self, tag=''):
        return describe_fmatrix(tag, self.f, self.P)
        
    def is_close(self, f_est):    
        #pdb.set_trace()
        
        res = True
        
        for i in range(self.I):
            for j in range(self.J):
                if self.g[i,j]==0 and f_est.g[i,j]!=0:
                    res = False
                    break
            if res==False:
                break
            
        #pdb.set_trace()
        diff_f = FlowMatrix(f=abs(self.f-f_est.f), g=None, P=self.P)
        [tot_flow_unknown0, tot_flow_known0, tot_flow0] = diff_f.calc_flow_on_paths()
        [tot_flow_unknown1, tot_flow_known1, tot_flow1] = self.calc_flow_on_paths()
        rat = float(tot_flow0)/tot_flow1
        
        close_T = 0.1 # sparsity_factor 0.01 (some false alarms)        
        
        if res == True: #same sparsity condition, check flow
            if rat > close_T:
                res = False
        else:
            rat = -rat
        
        #res: True         : same pattern, smaller diff
        #res: False, rat>0 : same pattern, larger diff
        #res: False, rat<0 : diff pattern
        return [res, rat] #rat==-1: res= false because of diff sparsity

class MinSparsityRes:# an object to keep track of the sparsest flow decomposition result
        
    def __init__(self, init_sparsity):
        
        self.min_sparsity_unknown_paths = init_sparsity #initially I*J+1
        self.min_sparsity_f = None #FlowMatrix
        #self.min_sparsity_g = None
        self.min_sparsity_f_trial_idx = -1
        self.min_sparsity_multiplicity = 0
        self.min_sparsity_tot_flow_unknown_paths = 0 
        self.min_sparsity_tot_flow_all_paths = 0
        self.tol = 0 #tolerance value for update of est_f & est_g        
    
    def set_tolerance(self, in_edges):
        
        [M, I] = np.shape(in_edges)
        
        weight = np.mean([LA.norm(in_edges[m]) for m in range(M)])
        eps = 0.001
        self.tol = eps * weight
        
        #pdb.set_trace()
        return
    
    def update(self, f, trial_idx=-1):

        flag = '' # 0, 1, 2
        
        s = f.get_sparsity_on_unknown_paths() #sparsity_unknown_paths
        
        if s < self.min_sparsity_unknown_paths:

            #pdb.set_trace()
            
            self.min_sparsity_unknown_paths = s
            self.min_sparsity_f = f
            
            if trial_idx != -1:
                self.min_sparsity_f_trial_idx = trial_idx
                
            self.min_sparsity_multiplicity = 0
            
            [tot_flow_unknown, tot_flow_known, tot_flow] = f.calc_flow_on_paths()
            self.min_sparsity_tot_flow_unknown_paths = tot_flow_unknown
            self.min_sparsity_tot_flow_all_paths = tot_flow
            #print('HERE')
            #pdb.set_trace()

            flag += ' smaller sp '
        
        elif s == self.min_sparsity_unknown_paths:
            
            #pdb.set_trace()
            
            dist = self.min_sparsity_f.diff(f)
            
            if dist > self.tol:
                self.min_sparsity_multiplicity += 1
                
            cond1 = self.min_sparsity_f==None
            if cond1:
                flag += ' cond1 '
            
            [tot_flow_unknown, tot_flow_known, tot_flow] = f.calc_flow_on_paths()
            new_tot_flow_unk_paths = tot_flow_unknown
            new_tot_flow_all_paths = tot_flow
            
            #total flow difference is below a threshold AND less flow is going down unknown paths.
            cond2 = abs(new_tot_flow_all_paths-self.min_sparsity_tot_flow_all_paths)<self.tol and \
                    new_tot_flow_unk_paths < self.min_sparsity_tot_flow_unknown_paths
            if cond2:
                flag += ' cond2[tot_new-tot_old = %e unk_new=%e < unk_old=%e] ' \
                        %(abs(new_tot_flow_all_paths-self.min_sparsity_tot_flow_all_paths), \
                          new_tot_flow_unk_paths, \
                          self.min_sparsity_tot_flow_unknown_paths)
            
            #total flow is greater than curr_ans total flow
            cond3 = new_tot_flow_all_paths > self.min_sparsity_tot_flow_all_paths
            if cond3:
                flag += ' cond3[tot_new=%e > tot_old=%e] '%(new_tot_flow_all_paths, \
                                                            self.min_sparsity_tot_flow_all_paths)
            
            if cond1 or cond2 or cond3:

                #pdb.set_trace()
                
                self.min_sparsity_f = f
                
                if trial_idx != -1:
                    self.min_sparsity_f_trial_idx = trial_idx
                    
                self.min_sparsity_tot_flow_unknown_paths = new_tot_flow_unk_paths
                self.min_sparsity_tot_flow_all_paths = new_tot_flow_all_paths
        
        return flag #debug purpose

class Log:# usef for debug

    def __init__(self):

        self.log_f1 = None
        self.log_f2 = None

    def init(self, log_loc):

        cmd = 'mkdir -p %s'%log_loc
        subprocess.call(cmd, shell=True)

        cmd = 'rm %s/*.*'%log_loc
        subprocess.call(cmd, shell=True)

        self.log_f1 = open(log_loc+'/log_f1.txt', 'a')
        self.log_f2 = open(log_loc+'/log_f2.txt', 'a')

        return

    def close(self):

        if self.log_f1 != None:
            self.log_f1.close()

        if self.log_f2 != None:
            self.log_f2.close()

        return

def syncstr(i_trial):

    
    state = np.random.get_state()

    st= str(np.random.random((5,10,10)))

    np.random.set_state(state)

    return st


'''
Test - description
'''

def DebugPrint(msg, file_tag=''):
    fi_obj = open('test_path_decompose_sparse_multisample2/debug/DebugPrint%s.txt'%file_tag, 'w')
    fi_obj.write(msg)
    fi_obj.close()
    return

def Print(msg, fo=None):
    if fo==None:
        print(msg)
    else:
        fo.write(msg)
    return

def describe_one_sample(f, P, I, J):

    # f is I by J
    res_str = ''

    col_sum = f.sum(axis=0)
    row_sum = f.sum(axis=1)
    tot_weight = sum(col_sum)
    #if tot_weight != sum(row_sum):
    #    res_str = res_str + 'inbalanced in/out flows\n'
    #    #pdb.set_trace()

    res_str = res_str + '\t\t'
    for j in range(J):
        res_str = res_str + 'col(%02d)\t\t'%j
    res_str = res_str + '|row_sum\n'

    for i in range(I):
        res_str = res_str + 'row(%02d)\t'%i
        for j in range(J):
            if P[i,j]==0:
                res_str = res_str + '%e\t'%f[i,j]
            else:
                res_str = res_str + '[%e]\t'%f[i,j] #[.] indicates known path
        res_str = res_str + '|%.2e\n'%row_sum[i]

    res_str = res_str + '\ncol_sum\t'
    for j in range(J):
        res_str = res_str + '%.2e\t'%col_sum[j]
    res_str = res_str + '\n'

    res_str = res_str + '\ntot weight=%.2e\n'%tot_weight

    res_str = res_str + '\nsparsity=%d\n\n'%np.count_nonzero(f)

    return res_str

def describe_fmatrix(tag, f, P):
    
    #f is M by I by J

    res_str = '-'*20 + 'describe_fmatrix start' + '-'*20 + '\n\n'

    res_str = res_str + tag + '\n\n'

    sp = np.shape(f) #shape of f, can be M*I*J or I*J

    if len(sp)==2:

        I = sp[0]
        J = sp[1]
        str_one_sample = describe_one_sample(f, P, I, J)
        res_str = res_str + str_one_sample + '\n'

    elif len(sp)==3:

        M = sp[0]
        I = sp[1]
        J = sp[2]

        for m in range(M):
            str_one_sample = describe_one_sample(f[m], P, I, J)
            res_str = res_str + '%d-th sample\n'%m
            res_str = res_str + str_one_sample + '\n'

    else:
        print('unknown fmatrix shape')
        #pdb.set_trace()

    res_str = res_str + '-'*20 + 'describe_fmatrix end  ' + '-'*20 + '\n\n'

    #print(res_str)

    return res_str

def describe_dim(dim):
    st = ''
    if len(dim)==3:
        st = st + 'Number of Samples = %d, Number of In-edges = %d, Number of Out-edges = %d' \
            %(dim[0], dim[1], dim[2])
    #pdb.set_trace()
    return st

def describe_P(max_known_paths):
    return 'number of known paths is limited by %d (and f matrix)'%max_known_paths

'''
Test 
'''

def force_no_Z(f):
    
    [M, I, J] = np.shape(f)
    
    I_idx = np.random.permutation(I)
    J_idx = np.random.permutation(J)
    
    f_mask = np.ones((I,J))
    
    for i1 in np.arange(0, I):
        for i2 in np.arange(i1+1,I):
            for j1 in np.arange(0, J):
                for j2 in np.arange(j1+1,J):
                    
                    ii1 = I_idx[i1]
                    ii2 = I_idx[i2]
                    jj1 = J_idx[j1]
                    jj2 = J_idx[j2]
                    
                    #no Z repeat
                    if  f_mask[ii1,jj1] != 0 and f_mask[ii2,jj2] != 0:
                        
                        f_mask[ii1,jj2] = 0
                        for m in range(M):
                            f[m][ii1,jj2] = 0
                            
                        f_mask[ii2,jj1] = 0
                        for m in range(M):
                            f[m][ii2,jj1] = 0
                            
                    elif f_mask[ii1,jj2] != 0 and f_mask[ii2,jj1] != 0:
                        
                        f_mask[ii1,jj1] = 0
                        for m in range(M):
                            f[m][ii1,jj1] = 0
                            
                        f_mask[ii2,jj2] = 0
                        for m in range(M):
                            f[m][ii2,jj2] = 0
                            
                    else:
                        continue                    
    return f

def force_Z_repeat(f):
    
    [M, I, J] = np.shape(f)
    
    I_idx = np.random.permutation(I)
    J_idx = np.random.permutation(J)
    
    f_mask = np.ones((I,J))
    
    for i1 in np.arange(0, I):
        for i2 in np.arange(i1+1,I):
            for j1 in np.arange(0, J):
                for j2 in np.arange(j1+1,J):
                    
                    ii1 = I_idx[i1]
                    ii2 = I_idx[i2]
                    jj1 = J_idx[j1]
                    jj2 = J_idx[j2]
                    
                    #no X repeat
                    if f_mask[ii1,jj1] != 0 and f_mask[ii1,jj2] != 0 \
                       and f_mask[ii2,jj1] != 0 and f_mask[ii2,jj2] != 0:
                           
                        f_mask[ii1,jj1] = 0
                        
                        for m in range(M):
                            f[m][ii1,jj1] = 0
                            
                    else:
                        continue
                    
    return f
    
def get_P(f, max_known_paths=0):
    
    #pdb.set_trace()
    
    P = np.zeros((f.I, f.J))
    
    items = []
    for i in range(f.I):
        for j in range(f.J):
            if f.g[i,j] != 0:
                items.append([i,j])
    
    num_known_paths = min(len(items), max_known_paths)
    
    if num_known_paths>0:
        idxes = np.random.permutation(len(items))
        for i in range(num_known_paths):
            P[items[idxes[i]][0],items[idxes[i]][1]]=1
    
    #pdb.set_trace()
    
    return P
    

def generate_f(M, I, J, repeat_option, max_known_paths=0):
    
    f = np.random.random((M,I,J))
    
    if repeat_option == 'no_Z':
        #pdb.set_trace()
        f = force_no_Z(f)
    elif repeat_option == 'Z_no_X':
        f = force_Z_repeat(f)
    elif repeat_option == 'X':
        #pdb.set_trace()
        f = f #pdb.set_trace()
    else:
        print('unknown repeat option')
        pdb.set_trace()
        
    f0 = FlowMatrix(f) # f0.P is all 0
        
    P = get_P(f0, max_known_paths) # if P[i,j]=1, there is known path; 0: there is no known path (expected for further processing)
    
    f1 = FlowMatrix(f=f, g=None, P=P)
    
    in_edges  = [np.sum(f[m], axis=1).tolist() for m in range(M)]
    out_edges = [np.sum(f[m], axis=0).tolist() for m in range(M)]    
    
    sparse_limit = M*I*J
    
    return [f1, in_edges, out_edges, P, sparse_limit]
   
def test_path_decompose_sparse_multisample(M, I, J, repeat_option, max_known_paths=0, dump_c_list=False, n=-1):
    #M, I, J, repeat_option, max_known_paths, dump_c_list)
    
    #pdb.set_trace()
    
    [f, in_edges, out_edges, P, sparse_limit] = generate_f(M=M, I=I, J=J, repeat_option=repeat_option, max_known_paths=max_known_paths)
    
    #pdb.set_trace()
    
    time_start = time.time()
    
    [f_est, c_list] = path_decompose_multisample(in_edges, out_edges, P, sparse_limit, \
                                                 c_list_ref=[], dump_c_list=dump_c_list, n_iter=n)
    
    est_time = time.time() - time_start
    return [f, f_est, est_time, c_list]
    
def get_g(f):
    
    [M, I, J] = np.shape(f)
    
    g = np.zeros((I,J))
    
    for i in range(I):
        for j in range(J):
            g[i,j] = max([f[m][i,j] for m in range(M)])
            
    return g

def test_N(M, I, J, N, test_loc, test_tag, mode, repeat_option, max_known_paths=0, dump_c_list=False):
    
    fi_obj = None
    fi_obj_est = None
    fi_obj_c = None
    if test_loc != '':
        fi_obj = open(test_loc + '/log_M%d_I%d_J%d%s_true.txt'%(M, I, J, test_tag), mode)
        fi_obj_est = open(test_loc + '/log_M%d_I%d_J%d%s_est.txt'%(M, I, J, test_tag), mode)
        fi_obj_c = open(test_loc + '/log_M%d_I%d_J%d%s_c.txt'%(M, I, J, test_tag), mode)
        
    #np.random.seed(0)
    num_est_error = 0
    avg_est_time = 0
    
    cnt = 0
    for n in range(N):
        
        cnt += 1
        if cnt > N/10:
            print('10% processed')
            cnt = 0
        
        nth_msg = '='*30 + '%d-th sim starts'%n + '='*30 + '\n'
        
        [f, f_est, est_time, c_list] = test_path_decompose_sparse_multisample(M, I, J, repeat_option, max_known_paths, dump_c_list, n)
        #c_list
        #f, f_est flow matrice here
        #pdb.set_trace()

        [est_res, dummy] = f.is_close(f_est)
        if est_res == False:
            #pdb.set_trace()
            num_est_error += 1
            
        avg_est_time += est_time
        
        nth_msg1 = nth_msg + f.description('true f') #describe_fmatrix('true f', f)

        nth_msg2 = nth_msg + f_est.description('est  f [correct estimation? %s]'%est_res) #describe_fmatrix('est  f [correct estimation? %s]'%est_res, f_est)
        
        nth_msg3 = nth_msg + describe_coeff_list(c_list)
        
        nth_msg1 = nth_msg1 + '='*30 + '%d-th sim ends  '%n + '='*30 + '\n'

        nth_msg2 = nth_msg2 + '='*30 + '%d-th sim ends  '%n + '='*30 + '\n'
        
        nth_msg3 = nth_msg3 + '='*30 + '%d-th sim ends  '%n + '='*30 + '\n'
        
        Print(nth_msg1, fi_obj)

        Print(nth_msg2, fi_obj_est)
        
        Print(nth_msg3, fi_obj_c)
    
    avg_est_time = float(avg_est_time)/N

    if fi_obj != None:
        fi_obj.close()

    if fi_obj_est != None:
        fi_obj_est.close()
        
    if fi_obj_c != None:
        fi_obj_c.close()
        
    return [num_est_error, avg_est_time]

LogObj = Log()

def test():

    np.random.seed(0)
    
    test_tag = '_I4J4_Z_no_X_P3_NoPostProcess_0519' #'_no_Z' #'_X' #'_Z_no_X' #'_example_tag'
    
    repeat_option = 'Z_no_X' #'no_Z' #'X' #'Z_no_X' #'no_Z', 'X'
    max_known_paths = 3 #can be further constrained by f matrix

    #'''
    k = 4
    ofst = 1
    rg = 10 #M=0+ofst,1+ofst,...,rg-1+ofst, with dim k by k
    dim_list = [[i+ofst,k,k] for i in range(rg)]
    #'''
    '''
    dim_list = [[5,2,2]]
    '''
    
    N = 100
    
    dump_c_list = False
    
    '''
    not to be configured below
    '''
    
    test_loc = 'test_path_decompose_sparse_multisample2/%s/'%test_tag

    LogObj.init(test_loc+'/log/')

    cmd = 'mkdir -p %s'%test_loc
    subprocess.call(cmd, shell=True)    
    
    fi_obj = None
    if test_loc!='':
        fi_obj = open(test_loc+'/summary%s.txt'%test_tag, 'w')    
    
    test_msg = ''
    
    for dim in dim_list:
        print(describe_dim(dim))
        print(describe_P(max_known_paths))
        
        #pdb.set_trace()
        [num_est_error, avg_est_time] = test_N(M=dim[0], I=dim[1], J=dim[2], \
                                            N=N, test_loc=test_loc, test_tag=test_tag, mode='w', \
                                            repeat_option=repeat_option, max_known_paths=max_known_paths, \
                                            dump_c_list=dump_c_list)
        test_msg = test_msg + describe_dim(dim) + '\tmax known paths=%d'%max_known_paths + '\n'
        tmp_msg = '\t average estimation error: %.02e (%d out of %d trials)\n\n'%(float(num_est_error)/N, num_est_error, N)
        print(tmp_msg)        
        test_msg = test_msg + tmp_msg
        tmp_msg = '\t average estimation time: %.02e sec\n\n'%(avg_est_time)
        print(tmp_msg)
        test_msg = test_msg + tmp_msg
        #print(test_msg)
    
    Print(test_msg, fi_obj)

    if fi_obj != None:
        fi_obj.close()

    LogObj.close()

    return

'''
Debug
'''
'''
def debug_extract_f(M, I, J, cases, file_path):
    
    f_dic = {} # case: f
    
    with open(file_path,'rU') as fi:
        in_block = False
        case = -1
        m = -1
        f = np.zeros((M,I,J))
        row_pattern = 'row.([0-9]+).'        
        for j in range(J):
            row_pattern = row_pattern + '\t(.+)'
        row_pattern = row_pattern + '\t(.+)'
        for line in fi:
            if in_block == False:
                x = re.findall('([0-9]+)-th sim starts', line)
                if len(x)>0 and int(x[0]) in cases:
                    case = int(x[0])
                    in_block = True
                    #pdb.set_trace()
                    #print('in block: %d'%case)
            elif len(re.findall('sim ends', line))>0:
                #print('out block: %d'%case)
                in_block = False
                case = -1
                f = np.zeros((M,I,J))
                #pdb.set_trace()                
            else:
                x = re.findall('([0-9]+)-th sample', line)
                if len(x) > 0:
                    m = int(x[0])
                    #print('%d-th sample'%m)
                
                x = re.findall(row_pattern, line)
                if len(x) > 0:
                    #pdb.set_trace()
                    #print('in row pattern')
                    i = int(x[0][0])
                    for j in range(J):
                        f[m][i, j] = float(x[0][1+j])
                    #pdb.set_trace()
                    if i==I-1:
                        f_dic[case] = f
    
    return f_dic
    
def debug_extract_c(M, I, J, cases, file_path):
    
    f_dic = {} # case: 100 iter*((M+1)*I*J coeff per iter)
    
    with open(file_path,'rU') as fi:
        in_block = False
        case = -1
        c_list = []
        for line in fi:
            if in_block == False:
                x = re.findall('([0-9]+)-th sim starts', line)
                if len(x)>0 and int(x[0]) in cases:
                    case = int(x[0])
                    in_block = True
                    #pdb.set_trace()
                    #print('in block: %d'%case)
            elif len(re.findall('sim ends', line))>0:
                #print('out block: %d'%case)
                in_block = False
                case = -1
                c_list = []
                #pdb.set_trace()                
            elif line.split()[0]=='iter':
                vals = line.split()[2:]
                c_list.append([float(v) for v in vals])
                if int(line.split()[1][:-1])==99:
                    f_dic[case] = c_list
                    #pdb.set_trace()
    
    return f_dic
    
def debug_err_cases():
    
    np.random.seed(0)
    
    test_tag = '_Z_no_X' #'_no_Z' #'_X' #'_Z_no_X' #'_example_tag'
    #repeat_option = 'Z_no_X' #'no_Z' #'X' #'Z_no_X' #'no_Z', 'X'

    #[M, I, J] = [6, 3, 3]
    #err_cases = [13, 30, 55, 87] # previous false alarm
    #err_cases = [33, 51, 56, 86]
    #err_cases = [79]

    #[M, I, J] = [7, 3, 3]
    #err_cases = [4, 40, 42, 50, 55, 65, 95, 96] #is_close false alarm ~ 1% to 5% (case 96: 6%)
    #err_cases = [7,8,9,10,11,12,13,14] #correct cases ~ 1e-7
    #err_cases = [0, 6, 15, 26, 27, 44, 63, 69, 72, 79, 86, 90, 93, 97] #less sparser sol 10+% (7+%~) (case 15, 27, 93)
    #err_cases = [0]
    #err_cases = [43, 62, 82, 89] #compatible Z

    #[M, I, J] = [8,3,3]
    #err_cases = [16, 39, 50, 64] #16, 39, 64
    #err_cases = np.arange(0,100).tolist()

    #[M, I, J] = [9,3,3]
    #err_cases = [17, 69]

    #[M, I, J] = [10,3,3]
    #err_cases = [18, 62] #18

    [M, I, J] = [5,4,4]
    err_cases = np.arange(0,100).tolist()
    
    #test_loc = 'test_path_decompose_sparse_multisample/%s/I3J3_ZnoX_0510_rand_coeff/'%test_tag  
    test_loc = 'test_path_decompose_sparse_multisample/%s/I4J4_ZnoX_0511/'%test_tag
    
    
    #not to be configured below
       
    file_true = test_loc + '/log_M%d_I%d_J%d%s_true.txt'%(M, I, J, test_tag)
    file_est  = test_loc + '/log_M%d_I%d_J%d%s_est.txt'%(M, I, J, test_tag)
    file_c  = test_loc + '/log_M%d_I%d_J%d%s_c.txt'%(M, I, J, test_tag)
    
    f_true_dic = debug_extract_f(M, I, J, err_cases, file_true)
    f_est_dic = debug_extract_f(M, I, J, err_cases, file_est)
    f_c_dic = debug_extract_c(M, I, J, err_cases, file_c)
    #pdb.set_trace()
    
    fi_obj = None
    fi_obj_est = None
    fi_obj_c = None
    if test_loc != '':
        fi_obj = open('test_path_decompose_sparse_multisample/debug/log_M%d_I%d_J%d%s_true.txt'%(M, I, J, test_tag), 'w')
        fi_obj_est = open('test_path_decompose_sparse_multisample/debug/log_M%d_I%d_J%d%s_est.txt'%(M, I, J, test_tag), 'w')
        fi_obj_c = open('test_path_decompose_sparse_multisample/debug/log_M%d_I%d_J%d%s_c.txt'%(M, I, J, test_tag), 'w')
    
    err_cnt = 0
    for case in err_cases:
        #print('M=%d, I=%d, J=%d, case=%d'%(M, I, J, case))
        
        f_true = f_true_dic[case]
        f_est_old_ref = f_est_dic[case]
        c_list = f_c_dic.setdefault(case,[])
        
        in_edges  = [np.sum(f_true[m], axis=1).tolist() for m in range(M)]
        out_edges = [np.sum(f_true[m], axis=0).tolist() for m in range(M)]    
        P = np.zeros((I, J)) # if P[i,j]=1, there is known path; 0: there is no known path (expected for further processing)    
        sparse_limit = M*I*J
        
        [f_est_old, dummy] = path_decompose_multisample(in_edges, out_edges, P, sparse_limit, c_list)
        #[f_est, dummy] = path_decompose_multisample(in_edges, out_edges, P, sparse_limit)
        
        #est_res_old_ref = is_close(f_true, f_est_old_ref)
        [est_res_old, est_res_old_rat] = is_close(f_true, f_est_old)
        if est_res_old==False:
            err_cnt += 1
        #est_res_new = is_close(f_true, f_est)
        
        #if est_res_old == False and find_compatible_Z_repeat(f_true)==[] and find_X_repeat(f_true)==[]:
        #    print('Here')
        #    pdb.set_trace()

        #print
        print('M=%d, I=%d, J=%d, case=%d'%(M, I, J, case))
        
        nth_msg = '='*30 + '%d-th sim starts'%case + '='*30 + '\n'
        
        nth_msg1 = nth_msg + describe_fmatrix('true f', f_true)
        
        #nth_msg1 = nth_msg1 + describe_fmatrix('true f', f_true)
        
        #nth_msg1 = nth_msg1 + describe_fmatrix('true f', f_true)

    
        #nth_msg2 = nth_msg + describe_fmatrix('est  f old ref [correct estimation? %s]'%est_res_old_ref, f_est_old_ref)
        
        nth_msg2 = nth_msg + describe_fmatrix('est  f old [correct estimation? %s]'%est_res_old, f_est_old)
        print('est  f old [correct estimation? %s (rat=%e) %d errors so far]'%(est_res_old, est_res_old_rat, err_cnt))
        
        #nth_msg2 = nth_msg2 + describe_fmatrix('est  f mod [correct estimation? %s]'%est_res_new, f_est)
        
        #nth_msg3 = nth_msg + describe_coeff_list(c_list)
        
        nth_msg1 = nth_msg1 + '='*30 + '%d-th sim ends  '%case + '='*30 + '\n'

        nth_msg2 = nth_msg2 + '='*30 + '%d-th sim ends  '%case + '='*30 + '\n'
        
        #nth_msg3 = nth_msg3 + '='*30 + '%d-th sim ends  '%n + '='*30 + '\n'
        
        Print(nth_msg1, fi_obj)

        Print(nth_msg2, fi_obj_est)
        
        #Print(nth_msg3, fi_obj_c)
        
        #pdb.set_trace()
    
    if fi_obj != None:
        fi_obj.close()
        
    if fi_obj_est != None:
        fi_obj_est.close()
        
    if fi_obj_c != None:
        fi_obj_c.close()
    
    return

'''

if __name__ == "__main__":
    test()
    #debug_err_cases()

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 05:43:37 2016

@author: olivo
"""

import numpy as np
import pdb
from cvxopt import solvers, matrix

def generate_f_1():
    
    f = [[[0, 1], [3, 2]],#1st sample
         [[0, 4], [1, 5]]]#2nd sample

    f = np.random.random((2,2,2))
    f[0][0][0]=0
    f[1][0][0]=0
    
    f = np.asanyarray(f)#sample/row/col
    
    #pdb.set_trace()
    M = len(f)
    I = len(f[0])
    J = len(f[0][0])
    
    in_edges  = [np.sum(f[m], axis=1).tolist() for m in range(M)]
    out_edges = [np.sum(f[m], axis=0).tolist() for m in range(M)]
    
    P = np.zeros((I, J)) # if P[i,j]=1, there is known path; 0: there is no known path (expected for further processing)
    
    sparse_limit = M*I*J
    
    return [f, in_edges, out_edges, P, sparse_limit]
    
def get_dim(in_edges, out_edges):    
    
    M = len(in_edges)
    if M != len(out_edges):
        print('path_decompose_multisample: imbalanced number of samples')
        pdb.set_trace()
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

def get_coeff(M, I, J, P):

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

def get_f(x, M, I, J):

    #convert x (J*I*(M+1) by 1 to f=M by (I by J) and g=I by J)

    f = np.zeros((M, I, J))
    g = np.zeros((I, J))

    for i in range(I):
        for j in range(J):
            for m in range(M):
                idx = j*I*(M+1) + i*(M+1) + m #f_ij(m)
                f[m][i][j] = x[idx]

            idx = j*I*(M+1) + i*(M+1) + M #g_ij (e.g. f_ij(M))
            g[i][j] = x[idx]

    return [f, g]

def describe_one_sample(f, I, J):

    # f is I by J
    res_str = ''

    col_sum = f.sum(axis=0)
    row_sum = f.sum(axis=1)
    tot_weight = sum(col_sum)
    if tot_weight != sum(row_sum):
        res_str = res_str + 'inbalanced in/out flows\n'
        #pdb.set_trace()

    res_str = res_str + '\t\t'
    for j in range(J):
        res_str = res_str + 'col(%02d)\t\t'%j
    res_str = res_str + '|row_sum\n'

    for i in range(I):
        res_str = res_str + 'row(%02d)\t'%i
        for j in range(J):
            res_str = res_str + '%e\t'%f[i,j]
        res_str = res_str + '|%.2e\n'%row_sum[i]

    res_str = res_str + '\ncol_sum\t'
    for j in range(J):
        res_str = res_str + '%.2e\t'%col_sum[j]
    res_str = res_str + '\n'

    res_str = res_str + '\ntot weight=%.2e\n'%tot_weight

    res_str = res_str + '\nsparsity=%d\n\n'%np.count_nonzero(f)

    return res_str

def describe_fmatrix(tag, f):

    res_str = '-'*20 + 'describe_fmatrix start' + '-'*20 + '\n\n'

    res_str = res_str + tag + '\n\n'

    sp = np.shape(f) #shape of f, can be M*I*J or I*J

    if len(sp)==2:

        I = sp[0]
        J = sp[1]
        str_one_sample = describe_one_sample(f, I, J)
        res_str = res_str + str_one_sample + '\n'

    elif len(sp)==3:

        M = sp[0]
        I = sp[1]
        J = sp[2]

        for m in range(M):
            str_one_sample = describe_one_sample(f[m], I, J)
            res_str = res_str + '%d-th sample\n'%m
            res_str = res_str + str_one_sample + '\n'

    else:
        print('unknown fmatrix shape')
        pdb.set_trace()

    res_str = res_str + '-'*20 + 'describe_fmatrix end  ' + '-'*20 + '\n\n'

    #print(res_str)

    return res_str

def path_decompose_multisample(in_edges, out_edges, P, sparse_limit):
    
    #dimension
    [M, I, J] = get_dim(in_edges, out_edges)
    
    #trivial cases
    
    #make in/out edge flow equal
    
    #constraints for lp problem
    [A, b, G, h] = get_constraints(M, I, J, in_edges, out_edges)

    #trials for random algorithm
    n_trials = 100

    solvers.options['show_progress'] = False
    for i_trial in range(n_trials):

        c = get_coeff(M, I, J, P)
        sol = solvers.lp(c =c ,G=G,h=h,A=A,b=b)
        x = np.array(sol['x'])

        [est_f, g] = get_f(x, M, I, J)

        #thresholding
        pdb.set_trace()

        print(describe_fmatrix('%d-th trial, est f'%i_trial, est_f))
        print(describe_fmatrix('%d-th trial, g'%i_trial, g))

        #update result with min sparsity
        pdb.set_trace()

    
    return []


# dim=2x2 n_sample=2 n_iter=1
def test_1():

    pdb.set_trace()

    np.random.seed(0)
    
    [f, in_edges, out_edges, P, sparse_limit] = generate_f_1()
    
    print(describe_fmatrix('true f', f))
    
    est_f = path_decompose_multisample(in_edges, out_edges, P, sparse_limit)
    
    pdb.set_trace()
    
    return    
    

if __name__ == "__main__":
    test_1()
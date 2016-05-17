import pdb, heapq
import numpy as np

def local_dot(a,b):
    ## This function takes the dot product between two vectors.
    n = len(a)
    sum=0
    for i in range(n):
        sum += a[i]*b[i]
    return sum


def nth_largest(n, iter):
    return heapq.nlargest(n, iter)[-1]

def reshape(arr, num_row, num_col):
    res = np.zeros((num_row, num_col))
    for i in range(num_row):
        for j in range(num_col):
            res[i,j]=arr[j*num_row+i]
    return res

def path_decompose(a,b,a_true,b_true,overwrite_norm,P,use_GLPK, sparsity= False):
	'''This function takes in a node's information in the attempt to decompose it into the lowest number of paths that
	accounts for the flow constraints.
	Thhe algorithm uses many trials of a randomizaed optimization model and takes the best result seen.
	**Do not use over_write_norm becuase a_true and b_true surrently have normalization instead of copycount.
	a: a vector of the copytcounts of the in-edges for the node.
	b: a vector of the copycounts of the out-edges for the node.
	a_true: a vector that should have the copycount values for the in-edges but does not.  (Don't Use)
	b_true: a vector that should have the copycount values for the out-edges but does not.  (Don't Use)
	decides whether or not to  use a_true and b_true.
	This is a matrix of in-edges versus out-edges that has a 0 if there is a known path and a 1 otherwise.
	'''

    #mb_check = 1 #if this parameter is set to 1, if the data can be set using MB, then it
	from cvxopt import matrix, solvers, spmatrix
	import numpy, copy
	from numpy import linalg as LA
	from numpy import array
	from numpy import zeros
	import operator
	#solvers.options['msg_lev'] = 'GLP_MSG_OFF'
	solvers.options['show_progress'] = False

	m = len(a)  ## a is a vector of the current in edge copycount values.
	n = len(b)  ## b is a vector of the current out edge copycount values.
	sa = sum(a); sb = sum(b)

	## Trivial case
	if m==0 or n==0:
		return [[],0]
	if m==1:
                answer = array(matrix(b,(m,n)))
                return [answer,0]
        elif n==1:
                answer = array(matrix(a,(m,n)))
                return [answer,0]

	if sa<=0 or sb<=0:
		answer = array(matrix(0,(m,n)))
		return [answer,0]


	## Make all in flow values non-zero.
	'''for i in range(m):
		a[i]=max(a[i],1e-10)
	for j in range(n):
		b[j]=max(b[j],1e-10)'''



	## If the flow in does not equal the flow out, make them equal.
	if sa>sb:
		const = (sa-sb)
		b = [k+const*k/sb for k in b ]
	else:
		const = (sb-sa)
		a = [k+const*k/sa for k in a ]




	A = matrix(0.,(m+n-1,m*n))  ## A is used to enforce flow constraint on decomposition.
	p=matrix(0.,(m*n,1))  ## This vector tells whether there is a known path for the pair of nodes or not.

	for i in range(m):
		for j in range(n):
			A[i,j*m+i] = 1.;  #check indexing
			p[j*m+i] = 1-P[i,j];#p denotes if known path is *ABSENT*

	for j in range(n-1):  #Must be range(n) to get full mattrix
		for i in range(m):
			A[m+j,j*m+i] = 1.;
	z = a +b




	rhs = matrix(z)
	rhs = rhs[0:n+m-1,0]  ## this vector is used to enforce flow constraints

	weight = LA.norm(a,1) ## (Not used)
	eps = 0.001
	tol = eps*weight  #test for significance.  Used for various purposes
	sparsity_factor = 0.1  #very aggressive curently - revert to 0.1 later
	removal_factor = 0.4;
	scale = max(max(rhs),1e-100) * 0.01;
	#print(rhs)
	trials = int(round(min(2*m*n*max(m,n),100))) ## Number of randomized trials.
	curr_min = m*n +1  ## The lowest number of non known paths seen so far.
	curr_ans = [];     ## The best solution seen so far.
	curr_ans_raw = []
	curr_err = 0;
	curr_mult = 0;         ## multiplicity of current solution
	curr_on_unknown = 0;   ## amount of flow on non "known paths".
	for ctr in range(trials):  ## randomize the the coefficients for the known non=known path flow values.
		c=matrix(abs(numpy.random.normal(0,1,(m*n,1))))
		for i in range(m*n):
			c[i] = c[i]*p[i]

		## G and h are used to make sure that the flow values are non-negative.
		G = spmatrix(-1.,range(m*n),range(m*n))
		h = matrix(0.,(m*n,1))
		if use_GLPK:
			sol = solvers.lp(c =c ,G=G,h=h,A=A,b=rhs/scale,solver='glpk')
		else:
                        sol = solvers.lp(c =c ,G=G,h=h,A=A,b=rhs/scale)
		temp_sol = array(sol['x'])*scale
		temp_sol_raw = copy.deepcopy(temp_sol)
		another_sol = copy.deepcopy(temp_sol)
		if overwrite_norm: ## Do not use right now because a_true and b_true are wrong.
			#the true values of copy count are used to decide thresholding behavior
			a = a_true[:]
			b = b_true[:]

		## This loop basically sets the flow values equal to 0 under certain conditions.
		for i in range(m):
			for j in range(n):
				if another_sol[j*m+i]<sparsity_factor*min(a[i],b[j]):
					another_sol[j*m+i]=0

				if temp_sol[j*m+i]<removal_factor*min(a[i],b[j]) or temp_sol[j*m+i]<tol: # temporarily disabled
					temp_sol[j*m+i]=0
					another_sol[j*m+i]=0

				if another_sol[j*m+i]<0:
					another_sol[j*m+i]=0
					temp_sol[j*m+i]=0


		s=0 ## s equals how many non-zero flows we are sending down non "known paths" in the temp solution
		for i in range(m*n):
			if p[i] > 0:  #Only couont the paths that are not supported
				s=s+numpy.count_nonzero(another_sol[i])

				## if the temporary solution is less than the current minimum solution, replace current minimum solution with
				## the temporary solution.
                if s<curr_min:
                        curr_min = s  ## current minimum value of non "known paths" in solution
                        curr_ans = temp_sol[:]  ## answer that attains it.
                        curr_ans_raw = temp_sol_raw[:]
                        curr_mult = 0  ## This is how many times a solution with this many non-zero non-known paths is seen.
                        curr_on_unknown = local_dot(array(p),temp_sol)  ## This says how much flow we are sending down non "known paths"

                else:
                        if s==curr_min:
                                if LA.norm(curr_ans-temp_sol) > tol:  ## Determines whethee to classify the solutions as different.
                                        curr_mult = curr_mult +1
                                if curr_ans==[] or ( abs(sum(sum(temp_sol))-sum(sum(curr_ans)))<tol and (local_dot(array(p),temp_sol) < curr_on_unknown) ) or sum(sum(temp_sol)) > sum(sum(curr_ans)):
										## These are a few conditions that make it the temporary solution:
										## 1: curr_sol is empty.  2: total flow difference is below a threshold AND less flow is going down unknown paths.  3: total flow is greater than curr_ans total flow.
                                        curr_ans = temp_sol[:]
                                        curr_ans_raw = temp_sol_raw[:]
                                        curr_on_unknown = local_dot(array(p),temp_sol)




	answer = matrix(0.,(m,n))
	if len(curr_ans) < m*n:
		pdb.set_trace()
	for i in range(m):
		for j in range(n):
			answer[i,j]=float(curr_ans[j*m+i])

    # Uniqueness consideration
	answer = array(answer)
	non_unique = 0
	if curr_mult > 1:
		non_unique =1

    # This makes the flow solution more sparse
	if sparsity != False:
		if m*n > sparsity:
			tmp_dict = {}
			for i in range(m):
				for j in range(n):
					tmp_dict[(i,j)]=answer[i, j]
			sorted_tmp = sorted(tmp_dict.items(), key=operator.itemgetter(1))[::-1]
			sorted_tmp = sorted_tmp[:sparsity]

			new_ans = zeros((m, n))
			for ind in sorted_tmp:
				new_ans[ind[0][0], ind[0][1]] = ind[1]
			answer = new_ans
	#return [answer,non_unique]
	return [reshape(curr_ans_raw,m,n), answer, non_unique]


def print_fmatrix(tag, f, file_path=''):

    res_str = '-'*20 + 'print_fmatrix start' + '-'*20 + '\n\n'

    res_str = res_str + tag + '\n'

    I = len(f)
    J = len(f[0])

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

    res_str = res_str + '-'*20 + 'print_fmatrix end  ' + '-'*20 + '\n\n'

    #print(res_str)

    return res_str

'''
Detailed Evaluation Below
''' 

def generate_f(I,J):
    #f: generic, and if f_ii' neq 0, f_jj' neq 0 then f_ij'=0, f_i'j=0
    #INPUT:
    #  I: # of in edges
    #  J: # of out edges
    #OUTPUT:
    #  f: array, I by J
    #  in_edges: list, I floats
    #  out_edges: list, J floats

    #  P: array, I by J, 1: has known path, 0: has no known path
    #  sparse limit: max num of non-zeros an est f should have

    f = np.random.random((I, J))
    I_idx = np.random.permutation(I)
    J_idx = np.random.permutation(J)

    #'''
    for i1 in np.arange(0, I):
        for i2 in np.arange(i1+1,I):
            for j1 in np.arange(0, J):
                for j2 in np.arange(j1+1,J):
                    
                    ii1 = I_idx[i1]
                    ii2 = I_idx[i2]
                    jj1 = J_idx[j1]
                    jj2 = J_idx[j2]
                        
                    if f[ii1,jj1] != 0 and f[ii2,jj2] != 0:
                        f[ii1,jj2] = 0
                        f[ii2,jj1] = 0
                    elif f[ii1,jj2] != 0 and f[ii2,jj1] != 0:
                        f[ii1,jj1] = 0
                        f[ii2,jj2] = 0
                    else:
                        continue
    #'''
    #pdb.set_trace()
    
    in_edges = np.sum(f, axis=1).tolist()
    out_edges = np.sum(f, axis=0).tolist()
                        
    P = np.zeros((I,J))
    
    sparse_limit = np.count_nonzero(f)

    return [f, in_edges, out_edges, P, sparse_limit]

def get_g(f):
    
    [M, I, J] = np.shape(f)
    
    g = np.zeros((I,J))
    
    for i in range(I):
        for j in range(J):
            g[i,j] = max([f[m][i,j] for m in range(M)])
            
    return g
    
def calc_tot_flow_on_all_paths(f):
    
    #pdb.set_trace()
    
    tot_flow = 0
    
    [M, I, J] = np.shape(f)
    
    for m in range(M):        
        tot_flow += sum(sum(f[m]))
    
    return tot_flow

def is_close(f, f_est):

    #pdb.set_trace()
    
    res = True
    
    [M, I, J] = np.shape(f)
    
    g = get_g(f)
    g_est = get_g(f_est)
    
    for i in range(I):
        for j in range(J):
            if g[i,j]==0 and g_est[i,j]!=0:
                res = False
                break
        if res==False:
            break
    
    rat = -1
    if res == True: #same sparsity condition, check flow
        rats = []
        for m in range(M):
            f_diff = calc_tot_flow_on_all_paths([abs(f[m]-f_est[m])])
            f_tot = calc_tot_flow_on_all_paths([f[m]])
            rats.append(float(f_diff)/f_tot)
    
        close_T = 0.1 # sparsity_factor 0.01 (some false alarms)
        rat = np.mean(rats)
        
        if rat >= close_T:
            res = False
    
    return [res, rat] #rat==-1: res= false because of diff sparsity

def test_path_decompose(I, J):
    
    #pdb.set_trace()
    
    [f, a, b, P, sparse_limit] = generate_f(I, J)
    
    #pdb.set_trace()
    
    [est_f_raw, est_f, is_non_uniqe] = path_decompose(a,b,a,b,False,P,False,sparse_limit)
    
    [close_to_true_f, ratio] = is_close([f], [est_f]) #is_close originally used for multisample, so add [] outside f, f_est
    
    return [f, est_f,est_f_raw, is_non_uniqe, close_to_true_f, ratio]
    
def Print(msg, fo=None):
    if fo==None:
        print(msg)
    else:
        fo.write(msg)
    return

def test_N(I, J, N, dir='', test_tag='', mode=''):

    fi_obj = None
    fi_obj_est = None
    if dir!='':
        fi_obj = open(dir+'/log_I%d_J%d_%s_true.txt'%(I,J, test_tag),mode)
        fi_obj_est = open(dir+'/log_I%d_J%d_%s_est.txt'%(I,J, test_tag),mode)
    
    np.random.seed(0)
    
    num_non_unique = 0
    num_error = 0
    acc_sparsity = 0
    
    for n in range(N):
        
        nth_msg = '='*30 + '%d-th sim starts'%n + '='*30 + '\n'
        
        [f, f_est,est_f_raw, is_non_unique, close_to_true_f, ratio] = test_path_decompose(I, J)
        
        if is_non_unique==1:
            num_non_unique = num_non_unique + 1
            
        if close_to_true_f==False:
            num_error = num_error+1
            
        acc_sparsity = acc_sparsity + np.count_nonzero(f)
        
        nth_msg1 = nth_msg + print_fmatrix('true f', f)
        
        nth_msg2 = nth_msg + print_fmatrix( \
                            'est  f [not unique? %s close to true f? %s rat=%e]' \
                            %(is_non_unique, close_to_true_f, ratio), f_est)
                            
        if close_to_true_f == False:
            
            nth_msg1 = nth_msg1 + print_fmatrix('true f', f)
        
            nth_msg2 = nth_msg2 + print_fmatrix('est_f_raw', est_f_raw)
        #nth_msg = nth_msg + print_fmatrix('est  f raw (before thresholding etc)', est_f_raw)
        #nth_msg = nth_msg + 'is not unique? %d\n'%is_non_unique
        
        nth_msg1 = nth_msg1 + '='*30 + '%d-th sim ends  '%n + '='*30 + '\n'
        
        nth_msg2 = nth_msg2 + '='*30 + '%d-th sim ends  '%n + '='*30 + '\n'
        
        Print(nth_msg1, fi_obj)
        
        Print(nth_msg2, fi_obj_est)

    avg_sparsity = float(acc_sparsity)/N

    if dir!='':
        fi_obj.close()
        fi_obj_est.close()
    
    return [avg_sparsity, num_non_unique, num_error]

def test():

    dir = 'test_path_decompose_sparse/'
    test_tag = ''
    #test_tag = 'no_sparsity'#no entry modified as 0 in f

    fi_obj = None
    if dir!='':
        fi_obj = open(dir+'/summary_%s.txt'%test_tag, 'w')

    dim_list = [[2,2], [3,3], [4,4], [5,5]]
    #dim_list = [[2,3], [4,3], [5,3]]
    N = 100

    dim_msg = ''

    for dim in dim_list:
        print('dim=%s'%str(dim))
        [avg_sparsity, num_non_unique, num_error] = test_N(I=dim[0], J=dim[1], N=N, dir=dir, test_tag=test_tag, mode='w')
        dim_msg = dim_msg + 'dim: I=%d, J=%d\n'%(dim[0], dim[1])
        dim_msg = dim_msg + '\t average sparsity: %.02e\n'%avg_sparsity
        dim_msg = dim_msg + '\t rate of non-unique decomposition: %.02e (%d out of %d trials)\n\n'%(float(num_non_unique)/N, num_non_unique, N)
        dim_msg = dim_msg + '\t rate of error in   decomposition: %.02e (%d out of %d trials)\n\n'%(float(num_error)/N, num_error, N)

    Print(dim_msg, fi_obj)

    if fi_obj != None:
        fi_obj.close()

if __name__ == "__main__":
    test()


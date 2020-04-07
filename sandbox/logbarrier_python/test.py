# experimentation, taken from assignment
import cvxpy as cp
import numpy as np
import matplotlib.pylab as pl
import scipy.sparse as sps
import seaborn as sns
from scipy.io import loadmat
from os.path import dirname, join as pjoin
import numpy.linalg as linalg
from scipy import linalg as scipy_linalg

def objective(t, A, B, x, c, b, s):
    obj = x/(c-x)
    return np.sum(obj)

def grad(t, A, B, x, c, b, s):
    
    gradient = (t * (1.0/(c-x) + x/((c-x)**2))
                + 1.0/(-x+c)
                - 1.0/x
                - 1.0 * (((B[0,:].T)/(B[0,:].dot(x)-b[0,0]) + 
                          (B[1,:].T)/(B[1,:].dot(x)-b[1,0]) +
                          (B[2,:].T)/(B[2,:].dot(x)-b[2,0]) )[..., np.newaxis]))

    assert(gradient.shape==(x.size,1))
    
    return gradient

def residual_prim(t, A, B, x, c, b, s, v):
    return A.dot(x)-s

def residual_dual(t, A, B, x, c, b, s, v):
    return grad(t, A, B, x, c, b, s) + (A.T).dot(v)

def kkt_rhs(t, A, B, x, c, b, s):
    
    L = x.size
    r = np.zeros((L+A.shape[0], 1))

    gradient = grad(t, A, B, x, c, b, s)

    r[0:x.size,:] = -1.0 * gradient
    r[x.size:,:] = s - A.dot(x)

    return r

def init_point(A, B, x, c, b, s):
    
    #manually init x to be feasible
    x[:,] = 0.1

    return x

def hessian(t, A, B, x, c, b, s):

    m = np.zeros((x.size, x.size))

    m[np.diag_indices_from(m)] = ( t * (2.0/((c-x)**2) + 2.0*x/((c-x)**3))
                                   + 1.0/np.power(x-c,2)
                                   + 1.0/np.power(x,2) )[:,0]

    denom = np.power(B.dot(x)-b, 2)
    
    for i in range(0, x.size):
        for j in range(0, x.size):
            m[i,j] +=  1.0 * ( (B[0,i]*B[0,j])/denom[0] +
                               (B[1,i]*B[1,j])/denom[1] +
                               (B[2,i]*B[2,j])/denom[2] )
    return m
    
def kkt_matrix(t, A, B, x, c, b, s):
    
    m = np.zeros((x.size + A.shape[0], x.size + A.shape[0]))

    m[0:x.size, 0:x.size] = hessian(t, A, B, x, c, b, s)
    
    m[x.size:x.size+A.shape[0], 0:A.shape[1]] = A
    m[0:A.shape[1], x.size:x.size+A.shape[0]] = A.T

    return m
    
def formulate():

    N = 8
    L = 13
    
    A = np.zeros((N-1,L))
    B = np.zeros((3,L))
    x = np.zeros((L,1)) #to be initialized in phase1
    c = np.zeros((L,1))
    b = np.zeros((3,1))
    s = np.zeros((N-1,1))

    #node1, links: out: 1,2,3, in:
    A[0,0] = 1.
    A[0,1] = 1.
    A[0,2] = 1.

    #node2, links: out: 4,6, in: 1
    A[1,3] = 1.
    A[1,5] = 1.
    A[1,0] = -1.

    #node3, links: out: 5,8, in: 3
    A[2,4] = 1.
    A[2,7] = 1.
    A[2,2] = -1.

    #node4, links: out: 7, in: 2,4,5
    A[3,6] = 1.
    A[3,1] = -1.
    A[3,3] = -1.
    A[3,4] = -1.

    #node5, links: out: 9,10,12, in: 7
    A[4,8] = 1.
    A[4,9] = 1.
    A[4,11] = 1.
    A[4,6] = -1.

    #node6, links: out: 11, in: 6,9
    A[5,10] = 1.
    A[5,5] = -1.
    A[5,8] = -1.

    #node7, links: out: 13, in: 8,10
    A[6,12] = 1.
    A[6,7] = -1.
    A[6,9] = -1.
    
    B[0,3] = 1.
    B[0,5] = 1.

    B[1,4] = 1.
    B[1,7] = 1.

    B[2,8] = 1.
    B[2,9] = 1.
    B[2,11] = 1.

    c[:,0] = 1.
    
    b[:,0] = 1.
    
    s[0,0] = 1.2
    s[1,0] = 0.6
    s[2,0] = 0.6
    s[3:,0] = 0.

    return [A, B, x, c, b, s]

def solve_kkt(t, A, B, x, c, b, s):
    kkt_m = kkt_matrix(t, A, B, x, c, b, s)
    res = kkt_rhs(t, A, B, x, c, b, s)
    # ax = sns.heatmap(kkt_m)
    # pl.show()  
    return scipy_linalg.solve(kkt_m, res, assume_a='sym')
        
def solve_inner(t, A, B, x, c, b, s, v,
                loop_outer, loop_inner_accum,
                record_residual, outer_residual):

    eps1 = 1e-12
    eps2 = 1e-12

    first_iter = False

    loop_inner = 0

    max_iter = 50

    y_prev = None
    
    while True:
        loop_inner += 1
        print("loop inner", loop_inner) 
        y = solve_kkt(t, A, B, x, c, b, s)

        if y_prev is not None and linalg.norm(y-y_prev) < 1e-15:
            break
        
        y_prev = y
        
        delta_x = y[0:x.size,:]
        v_new = y[x.size:,:]
        if v is None:
            first_iter = True
            v = v_new
            print("first_iter")
        delta_v = v_new - v
        
        #backtracking line search
        beta = 0.95
        alpha = 0.4
        h = 1.0
        
        loopc = 0
        res_inner = None
        
        while True:
            loopc+= 1
            assert(v.shape == delta_v.shape)

            res_prim_next = residual_prim(t, A, B, x+h*delta_x,
                                          c, b, s, v+h*delta_v)
            res_prim_cur = residual_prim(t, A, B, x, c, b, s, v)

            res_dual_phase1_next = residual_dual(t, A, B,
                                                 x+h*delta_x,
                                                 c, b, s,
                                                 v+h*delta_v)
            res_dual_phase1_cur = residual_dual(t, A, B, x, c,
                                                b, s, v)

            r1 = np.concatenate((res_prim_next,res_dual_phase1_next), axis=0)
            r2 = np.concatenate((res_prim_cur,res_dual_phase1_cur), axis=0)
            
            res_inner = linalg.norm(r1)
            
            if res_inner <= (1-alpha*h)*linalg.norm(r2):
                print("obj: ", objective(t, A, B, x+h*delta_x, c, b, s))
                print("res: ", res_inner)
                print("h: ", h)
                break

            h = beta * h

        x = x + h * delta_x
        v = v + h * delta_v

        record_residual[0].append(loop_inner_accum + loop_inner)
        record_residual[1].append(res_inner+outer_residual)
        
        res_prim_cur = residual_prim(t, A, B, x, c, b, s, v)
        res_dual_cur = residual_dual(t, A, B, x, c, b, s, v)

        if linalg.norm(res_prim_cur,2) <= eps1 and np.linalg.norm(res_dual_cur,2) <= eps2:
            break
        if loop_inner > max_iter:
            break
    
    print("loop: outer: ", loop_outer, ", inner: ", loop_inner)

    return x, v, objective(t, A, B, x, c, b, s), loop_inner_accum + loop_inner, record_residual
        
def solve(A, B, x, c, b, s):

    rec_res = ([],[])
    record_delay_vs_outer_iter = ([],[])
    
    x = init_point(A, B, x, c, b, s)

    #sanity check
    assert(np.all(B.dot(x) <= b))
    assert(np.all(x<=c))
    assert(np.all(x>=0))

    t = 1.0
    mu = 2.0

    m = A.shape[0]
    eps = 1e-12
    v = None

    loop_outer = 0
    loop_inner_accum = 0
    
    while True:
        loop_outer += 1
        x, v, obj, loop_inner_accum, rec_res = solve_inner(t, A, B, x, c,
                                                           b, s, v, loop_outer,
                                                           loop_inner_accum, rec_res, m/t )
        if m/t <= eps:
            break
        t = mu * t
        
        record_delay_vs_outer_iter[0].append(loop_outer)
        record_delay_vs_outer_iter[1].append(obj)
        
        if loop_inner_accum > 5000:
            break

    print("loop_outer:", loop_outer)
            
    return x, obj, record_delay_vs_outer_iter, rec_res

prob = formulate()
A, B, x, c, b, s = prob
solution, objective, record1, record2 = solve(*prob)

print("objective achieved: ", objective)
print("solution: ", solution)

#sanity check
assert(np.all(np.abs(A.dot(solution)-s)<1e-14))
assert(np.all(B.dot(solution) <= b))
assert(np.all(solution<=c))
assert(np.all(solution>=0))

# print(A.dot(solution))
# print(B.dot(solution))

#delay vs outer iterations
pl.plot(record1[0],record1[1])
pl.ylabel('delay')
pl.xlabel('outer iterations')
pl.show()
# pl.savefig('imgs/delay_vs_outer.png')

#residual (sum of norm of residual of infeasible start
#newton method and outer iteration residual): ||r||+m/t
res = np.array(record2[1])
assert(np.all(res>=0))

log_vals = np.log10(res)

pl.plot(record2[0], log_vals)
pl.ylabel('log10 (residual(outer+inner))')
pl.xlabel('cumulative inner iterations')
pl.show()
# pl.savefig('imgs/res_vs_inner_iter.png')


# interior point, log barrier implementation
# solves: min f(x) s.t. f_ineq(x) <= 0, Ax = b, where f and f_ineq are convex

import autograd as ag
import autograd.numpy as np

#system: kkt_matrix * [delta_x, delta_v].T = hht_rhs


def kkt_matrix(f, A, b, x, v):

    d = x.shape[0] + A.shape[0]
    m = np.zeros((d, d))
    m[0:x.shape[0], 0:x.shape[0]] = np.squeeze(ag.hessian(f)(x))
    m[x.shape[0]:x.shape[0] + A.shape[0], 0:A.shape[1]] = A
    m[0:A.shape[1], x.shape[0]:x.shape[0] + A.shape[0]] = A.T

    return m


def kkt_rhs(f, A, b, x, v):

    r = np.zeros((x.shape[0] + A.shape[0], 1))
    g = ag.grad(f)(x) + np.dot(A.T, v)
    r[0:x.shape[0], :] = -g
    r[x.shape[0]:, :] = b - A.dot(x)

    return r


def solve_kkt(f, A, b, x, v):

    kkt_m = kkt_matrix(f, A, b, x, v)
    res = kkt_rhs(f, A, b, x, v)

    return np.linalg.solve(kkt_m, res)


def residual(f, A, b, x, v):
    return np.concatenate(
        [ag.grad(f)(x) + np.dot(A.T, v),
         np.dot(A, x) - b])


def line_search(f, A, b, x, v, delta_x, delta_v, iter_max=50):

    beta = 0.97
    alpha = 0.4
    t = 1.

    r = residual(f, A, b, x + t * delta_x, v + t * delta_v)
    r0 = residual(f, A, b, x, v)

    i = 0
    while np.dot(r.T, r) > (1. - alpha * t) * \
            np.dot(r0.T, r0) and i < iter_max:
        t = t * beta
        r = residual(f, A, b, x + t * delta_x, v + t * delta_v)
        i += 1

    return t


def solve_inner(f, A, b, x, v, it=20, eps1=1e-9, eps2=1e-9):

    i = 0
    while i < it:

        delta = solve_kkt(f, A, b, x, v)

        delta_x = delta[0:x.shape[0], :]
        delta_v = delta[x.shape[0]:, :]

        t = line_search(f, A, b, x, v, delta_x, delta_v)

        x = x + t * delta_x
        v = v + t * delta_v

        r = residual(f, A, b, x, v)

        feasibility = np.abs(np.dot(A, x) - b) < eps2

        if np.dot(r.T, r) < eps1 and np.all(feasibility):
            break

        i += 1

    return x, v


def f_augment(f_obj, f_ineq, t):
    def f_new(x):
        return t * f_obj(x) - np.sum(np.log(-f_ineq(x)))
    return f_new


def solve(f_obj, f_ineq, A, b, x0, it=20, eps=1e-9):
    """
    solves min_x f(x) s.t. f_ineq(x) <= 0, A(x) = b
    """
    t = 1.0
    mu = 1.5
    m = A.shape[0]
    eps = 1e-9

    x = x0

    # check f_ineq constraints
    if not np.all(f_ineq(x) <= 0):
        # solve for min a s.t. f_ineq(x) <= a
        # check if a <=0
        def f_obj_aux(x):
            return x[0, 0]

        def f_ineq_aux(x):
            return x[1:, 0] - x[0, 0]

        v_max = np.amax(x) + 1.
        xx = np.concatenate([[[v_max]], x])
        A_aux = np.zeros((A.shape[0], A.shape[1] + 1))
        A_aux[:, 1:] = A
        b_aux = b
        xx_soln, _, _, _ = solve(f_obj_aux, f_ineq_aux, A_aux, b_aux, xx)
        if xx[0, 0] > 0:
            return None
        else:
            x = xx[1:, 0]  # set initial starting point

    v = np.zeros((A.shape[0], 1))  # dual variable

    while True:
        f_aug = f_augment(f_obj, f_ineq, t)
        x, v = solve_inner(f_aug, A, b, x, v, it, eps, eps)
        if m / t <= eps:
            break
        t = mu * t

    r = residual(f_aug, A, b, x, v)

    return x, v, np.dot(r.T, r)[0, 0], np.max(np.dot(A, x) - b)


# tests ---
if __name__ == "__main__":

    # objective
    def my_f_obj(x):
        return x[0, 0] * x[0, 0] + x[1, 0] * x[1, 0] + x[2, 0] * x[2, 0] * 4.

    # inequality constraint: Ix <= 10
    def my_f_ineq(x):
        return x - 10.

    # equality constraint: Ax = b
    A = np.array([[1., 1., 0.], [0., 1., 1.]])
    b = np.array([[1.], [2.]])
    x0 = 2. * np.ones((3, 1))

    ret = solve(my_f_obj, my_f_ineq, A, b, x0)

    if ret is not None:
        (soln, dual, res, feas_err) = ret
        print("objective achieved: ", my_f_obj(soln))
        print("solution: ")
        print(soln)
        print("dual: ")
        print(dual)
        print("residual: ", res)
        print("equality constraint error: ", feas_err)
        assert(np.all(soln <= 10.))
        np.all(np.abs(np.dot(A, soln) - b) < 1e-7)
    else:
        print("infeasible")

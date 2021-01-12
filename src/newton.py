# newton's method, unconstrained

import autograd as ag
import autograd.numpy as np


def line_search(f, df, x, d, iter_max=50):
    # backtracking search, eg: wolfe condition
    beta, alpha = 0.97, 0.4
    s = 1.
    i = 0.
    df_x, f_x = df(x), f(x)

    while f(x + s * d) >= f_x + alpha * s * np.dot(df_x.T, d) and i < iter_max:
        s *= beta
        i += 1
    return s


def solve(f, x0, eps=1e-8, it_max=200):

    x = x0
    it = 0
    df, dff = ag.grad(f), ag.hessian(f)

    d_f_eval_prev = None

    while it < it_max:
        # augment hessian to be PD
        hess = np.squeeze(dff(x))
        eig_min = np.amin(np.linalg.eig(hess)[0])
        hess += max(0., -eig_min + 0.001) * np.eye(hess.shape[0])

        direction = np.dot(np.linalg.inv(hess), -df(x))
        s = line_search(f, df, x, direction)
        xx = x + s * direction
        f_eval_xx, f_eval_x = f(xx), f(x)
        d_f_eval = f_eval_xx - f_eval_x

        x = xx
        it += 1

        if d_f_eval_prev is not None and abs(d_f_eval / d_f_eval_prev) < eps:
            break

        d_f_eval_prev = d_f_eval

    return x, f_eval_xx, it


# test
if __name__ == "__main__":

    def myfunc(x):
        return np.squeeze(np.dot(x.T, x)) + 10.

    x0 = 1000. * np.random.rand(5, 1)

    print(f"x0: {x0}, f(x0): {myfunc(x0)}")

    ret, f, _ = solve(myfunc, x0)

    print(f"ret: {ret}, f_x_star: {f}")

    assert abs(f - 10.) < 1e-9

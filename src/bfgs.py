import autograd as ag
import autograd.numpy as np


def line_search(f, df, x, direc, iter_max=100):

    beta, alpha = 0.95, 0.4
    s = 1.0
    i = 0
    f_x, df_x = f(x), df(x)

    while f(x + s * direc) > f_x + alpha * s * df_x.T @ direc and i < iter_max:
        s *= beta
        i += 1

    return s


def solve(f, x0, eps=1e-8, hess_approx=None, it_max=200):

    x = x0

    H = np.eye(x0.size) if hess_approx is None else np.inv(hess_approx)

    df = ag.grad(f)

    it = 0

    while it < it_max:

        if np.linalg.norm(df(x)) < eps:

            break

        p = -H @ df(x)

        alpha = line_search(f, df, x, p)

        s = alpha * p
        x_next = x + s

        y = df(x_next) - df(x)
        rho = 1 / (y.T @ s)
        a = np.eye(x0.size) - rho * s @ y.T
        b = np.eye(x0.size) - rho * y @ s.T
        H = a @ H @ b + rho * s @ s.T
        x = x_next

        it += 1

    return x, f(x), it


if __name__ == "__main__":

    def myfunc(x):
        return np.squeeze(x.T @ x) + 10.0

    x0 = 1000.0 * np.random.rand(5, 1)

    print(f"x0: {x0}, f(x0): {myfunc(x0)}")

    ret, f, it = solve(myfunc, x0)

    print(f"ret: {ret}, f_x_star: {f}, iterations: {it}")

    assert abs(f - 10.0) < 1e-9

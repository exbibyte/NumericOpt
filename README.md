# NumericOpt

Playground for optimization routines. See src/ folder for implementations. More to come.


ip_log_barrier: Solves: min f(x) s.t. f_ineq(x) <= 0, Ax = b, where f and f_ineq are convex. Input parameters: f, f_ineq, A, b, x0 (initial guess), inner max iterations, epsilon stopping threshold. Returns None if no feasible point is found; otherwise return primal, dual, and residuals.


# NumericOpt

Playground for optimization routines. See src/ folder for implementations. More to come.


ip_log_barrier: Solves: min f(x) s.t. f_ineq(x) <= 0, Ax = b, where f and f_ineq are convex. Input parameters: f, f_ineq, A, b, x0 (initial guess). Returns None if constraints are not feasible; otherwise primal, dual, and residuals.


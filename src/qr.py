import autograd as ag
import autograd.numpy as np
import numpy

# require A be a matrix of dimension m x n, m >= n
#
# QR factorization with Householder reflection
def factor_qr(A):
    m = A.shape[0]
    n = A.shape[1]
    qs = []
    for i in range(0,n): # iterate over columns
        column = A[i:,i].T # partial column

        l2_norm = np.sqrt(np.sum(column.T @ column))
        alpha = -l2_norm if A[i,i] >= 0 else l2_norm
        e = np.zeros(column.shape)
        e[0] = 1
        u = column - alpha * e
        v = u / np.sqrt(np.sum(u.T @ u))
        v = v.reshape((v.size, 1))
        Q = np.eye(v.size) - 2 * v @ v.T
        Q_adjusted = np.eye(m)
        Q_adjusted[i:,i:] = Q
        qs.append(Q_adjusted)
        A = Q_adjusted @ A
        
        # check column values are 0 except for 1 element
        b = A[i+1:, i]
        abs_sum = np.sum(np.abs(b))

        assert abs_sum < 1e-7

    # here Q is explicitly calculated
    # R = (Q_t @ .. @ Q_1) @ A
    # (Q_t @ .. @ Q_1)^T R = A, by orthogonality
    # Q = (Q_t @ .. @ Q_1)^T
    # Q = Q_1^T @ .. @ Q_t^T
    # Q = Q_1 @ .. @ Q_t, by construction of Q_i = Q_i^T
    Q = np.eye(m)
    for i in qs: 
        Q = Q @ i 
        
    R = A
    
    return Q, R

if __name__ == "__main__":

    # test
    
    np.set_printoptions(precision=3, suppress=True)

    A = np.random.rand(4,3)
    mm = min(A.shape[0], A.shape[1])
    A[range(0,mm), range(0,mm)] = 1.1 + np.abs(np.random.rand(mm))
    
    print(f"{A=}")
    
    Q, R = factor_qr(A)

    print(f"{Q=}")
    print(f"{R=}")

    numpy.testing.assert_allclose(Q @ Q.T, np.eye(Q.shape[0]), atol=1e-7)

    x = np.random.rand(A.shape[1], 1)

    # oracle check
    b = A @ x
    
    # R x = Q^-1 b
    # R x = Q^T b, by orthogonality of Q
    # back substitution to get answer

    # numpy.linalg.inv requires input to be square matrix
    if R.shape[0] == R.shape[1]:
        x_solve = np.linalg.inv(R) @ Q.T @ b
        numpy.testing.assert_allclose(x_solve, x)

    # manual back substitution
    c = Q.T @ b
    x_solve_2 = np.zeros((R.shape[1], 1))
    for row in reversed(range(0, min(R.shape[0], R.shape[1]))):
        for col in range(row+1, R.shape[1]):
            c[row] -= R[row, col] * x_solve_2[col]
        x_solve_2[row, 0] = c[row] / R[row, row]

    print(f"{x_solve_2=}")

    numpy.testing.assert_allclose(x_solve_2, x)

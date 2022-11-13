# LU factorization with partial pivot in rows

import numpy as np

def extract_lu(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n,n))
    for col in range(n):
        L[col, :col] = A[col, :col]
        U[col, col:] = A[col, col:]
    return L, U

def lu(A):

    n = A.shape[0]
    S = 0 # row swap count
    P = np.arange(n, dtype=np.int32) # row permutation
                  
    for col in range(n):

        # find row to pivot
        row_swap = col
        for row in range(col+1, n):
            row_swap = row if A[row, col] > A[row_swap, col] else row_swap
            
        p_temp = P[col]
        P[col] = P[row_swap]
        P[row_swap] = p_temp
        
        S = S + 1 if row_swap != col else 0

        temp = np.copy(A[row_swap, :])
        A[row_swap, :] = A[col, :]
        A[col, :] = temp
        
        if A[col, col] == 0:
            raise Exception("degenerate matrix detected")

        for row in range(col+1, n):
            ratio = A[row, col] / A[col, col]
            A[row, col] = ratio # save value for L in A
            
            # perform gaussian elimination to get U

            # for c in range(col+1, n):
            #     A[row, c] = A[row, c] - ratio * A[col, c]            
            # simplify above:
            A[row, col+1:n] = A[row, col+1:n] - ratio * A[col, col+1:n]

    return A, P, S        

def det(L, U, S):
    ret = pow(-1, S)
    n = L.shape[0]
    for i in range(n):
        ret *= L[i,i]
        ret *= U[i,i]
    return ret

# test
if __name__ == "__main__":

    A = np.array([[1, 1, 1], [4, 3, -1], [3, 5, 3]], dtype=np.float64)
    b = np.array([1, 6, 4], dtype=np.float64)
    x_actual = np.array([1, 0.5, -0.5], dtype=np.float64)
    
    A, P, S = lu(A)

    d_actual = np.linalg.det(A)
        
    print(f"{A=}, {P=}, {S=}")

    # LU = PA
    # Ax = b
    # PAx = Pb
    # LUx = Pb
    # y = Ux
    # solve y: Ly = Pb
    # solve x: Ux = y

    L, U = extract_lu(A)

    print(f"{L=}")
    print(f"{U=}")
    
    b_permuted = b[P]

    n = A.shape[0]
        
    # forward substitution:
    
    # y = b_permuted
    # for i in range(n):
    #     for j in range(i):
    #         y[i] -= L[i, j] * y[j]
    # simplify above
    y = np.zeros((n,))
    for i in range(n):
        y[i] = b_permuted[i] - np.sum(L[i, 0:i] * y[0:i])

    # backward substitution:

    # x_solve = y    
    # for i in reversed(range(n)):
    #     for j in range(i+1,n):
    #         x_solve[i] -= U[i, j] * x_solve[j]
    #     x_solve[i] /= U[i, i]
    # simplify above
    x_solve = np.zeros((n,))
    for i in reversed(range(n)):
        x_solve[i] = (y[i] - np.sum(U[i, i+1:n] * x_solve[i+1:n])) / U[i, i]

    print(f"{x_solve=}")
    print(f"{x_actual=}")
    
    np.testing.assert_allclose(x_solve, x_actual)

    d = det(L, U, S)

    # print(f"det: {d}")
    # print(f"det actual: {d_actual}")
    
    np.testing.assert_allclose(d, d_actual, rtol=1e-2, atol=2e-2)
    
    
    

import autograd as ag
import autograd.numpy as np
import numpy

from qr import factor_qr

# require A be a matrix of dimension m x n, m >= n
#
# QR factorization with Householder reflection
def qr_iter(A):
    A_k = A
    Qs = numpy.eye(A.shape[0])
    for k in range(0,100):
        Q, R = factor_qr(A_k)
        A_k = R @ Q
        Qs = Qs @ Q
    return A_k, Qs

# AKA Rayleigh Quotient Iteration
# not: does not handle complex eigenvalues at the moment
def qr_iter_shift(A):
    n = A.shape[0]
    A_k = A
    Qs = np.eye(n)
    j = n - 1

    for k in range(0,1000):
        print(k)
        if j < 0:
            break

        if j == 0 or numpy.all(numpy.abs(A_k[j,:j] < 1e-15)):
            j -= 1
            
            continue

        shift = A_k[j, j]
        A_k -= np.eye(n) * shift
        # Q, R = np.linalg.qr(A_k[:j+1, :j+1])
        Q, R = factor_qr(A_k[:j+1, :j+1])
        A_k[:j+1, :j+1] = R @ Q
        A_k += np.eye(n) * shift

        temp = numpy.eye(n)
        temp[:j+1, :j+1] = Q[:j+1, :j+1]
        Qs = Qs @ temp

    e_vals_arr = numpy.diag(A_k, k=0)
    
    eigenvalues_sort_indices = numpy.argsort(e_vals_arr)[::-1]
    e_vals_sorted = e_vals_arr[eigenvalues_sort_indices]
    
    eigenvecs_sorted = Qs[:, eigenvalues_sort_indices]

    return e_vals_sorted, eigenvecs_sorted

if __name__ == "__main__":
    # test
    
    np.set_printoptions(precision=3, suppress=True)

    A = np.random.rand(4,4)
    # A = A.T @ A
    mm = min(A.shape[0], A.shape[1])
    A[range(0,mm), range(0,mm)] = np.abs(np.random.rand(mm))
    
    print(f"{A=}")

    w, v = numpy.linalg.eig(A)

    A_copy = numpy.copy(A)
    
    eigenvalues, Qs = qr_iter_shift(A)

    print(f"iter_shift: evals :{eigenvalues=}")
    print(f"iter_shit: evecs :{Qs=}")

    # print("---")
    # eigenvalues, Qs = qr_iter(A_copy)
    # print(f"{eigenvalues=}")
    # print(f"{Qs=}")

    print("---")
    print("reference:")
    print(f"{w=}")
    print(f"{v=}")


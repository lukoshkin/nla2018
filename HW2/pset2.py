# encoding: utf-8
# pset2.py

import numpy as np
from scipy.sparse import csr_matrix
# don't forget import packages, e.g. scipy
# but make sure you didn't put unnecessary stuff in here

# INPUT : diag_broadcast - list of diagonals value to broadcast,length equal to 3 or 5; n - integer, band matrix shape.
# OUTPUT : L - 2D np.ndarray, L.shape[0] depends on bandwidth, L.shape[1] = n-1, do not store main diagonal, where all ones;                  add zeros to the right side of rows to handle with changing length of diagonals.
#          U - 2D np.ndarray, U.shape[0] = n, U.shape[1] depends on bandwidth;
#              add zeros to the bottom of columns to handle with changing length of diagonals.
def band_lu(diag_broadcast, n): # 5 pts
    # enter your code here
    k_diags = len(diag_broadcast)
    
    if (k_diags == 3):
        U = np.zeros((n,2))
        L = np.zeros((1,n-1))

        a,b,c = diag_broadcast
        for i in range(n-1):
            U[i,0] = b - L[0,i-1] * c
            L[0,i] = a / U[i,0]
        U[n-1,0] = b - L[0,n-2] * c
        U[:-1,1] = [c] * (n-1)

    if (k_diags == 5):
        U = np.zeros((n,3))
        L = np.zeros((2,n-1))

        a,b,c,d,e = diag_broadcast
        for i in range(n-2):
            U[i,0] = c - L[1,i-1] * U[i-1,1] - L[0,i-2] * e
            U[i,1] = d - L[1,i-1] * e
            L[1,i] = (b - L[0,i-1] * U[i-1,1]) / U[i,0]
            L[0,i] = a / U[i,0]
        U[n-2,0] = c - L[1,n-3] * U[n-3,1] - L[0,n-4] * e
        U[n-2,1] = d - L[1,n-3] * e
        L[1,n-2] = (b - L[0,n-3] * U[n-3,1]) / U[n-2,0]
        U[n-1,0] = c - L[1,n-2] * U[n-2,1] - L[0,n-3] * e
        U[:-2,2] = [e] * (n - 2)

    return L, U


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def gram_schmidt_qr(A,check=True,eps=1e-8): # 5 pts
    # your code is here
    m,n = A.shape
    Q = []
    num_orts = min(m,n)
    
    # check if A is rank-deficient
    if check and np.linalg.matrix_rank(A[:,:num_orts]) < num_orts:
        B = A[:,:num_orts] + eps * np.random.rand(num_orts,num_orts)
    else:
        B = np.copy(A[:,:num_orts])
    #-------------------------------
    
    for v in B.T:
        gather = np.copy(v)
        for u in Q:
            proj = v @ u / (u @ u) * u
            gather -= proj
        gather /= np.linalg.norm(gather)
        Q.append(gather)
    Q = np.array(Q).T
    R = Q.T @ A
    
    return Q, R

# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def modified_gram_schmidt_qr(A): # 5 pts
    # your code is here
    num_orts = min(A.shape)
    Q = np.empty((A.shape[0],num_orts))
    R = np.zeros((num_orts,A.shape[1]))
    
    np.copyto(Q,A[:,:num_orts])
    for i in range(num_orts):
        R[i,i] = np.linalg.norm(Q.T[i])
        q = Q.T[i] / R[i,i]
        for j in range(i+1,num_orts):
            R[i,j] = q @ Q.T[j]
            Q.T[j] -= R[i,j] * q
    Q /= np.linalg.norm(Q,axis=0)
    R[:,num_orts:] = Q.T @ A[:,num_orts:]

    return Q, R


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A=QR
def householder_qr(A): # 7 pts
    # your code is here
    m, n = np.shape(A)
    Q = np.identity(m)
    R = np.copy(A)

    for k in range(m - 1):
        u = np.copy(R[k:, k])
        u[0] += np.copysign(np.linalg.norm(u), R[k, k])
        u /= np.linalg.norm(u)
        
        uu = 2 * np.outer(u,u)
        R[k:,:] -= uu @ R[k:,:]
        Q[k:,:] -= uu @ Q[k:,:]
        
    return Q, R


# INPUT:  G - np.ndarray
# OUTPUT: A - np.ndarray (of size G.shape)
def pagerank_matrix(A): # 5 pts
    # enter your code here
    G = csr_matrix(A)
    _,cols = G.nonzero()
    left_sums = np.array(G.sum(0))[0]
    data = A.data / left_sums[cols]
    return csr_matrix((data,G.indices,G.indptr),G.shape)


# INPUT:  A - np.ndarray (2D), x0 - np.ndarray (1D), num_iter - integer (positive) 
# OUTPUT: x - np.ndarray (of size x0), l - float, res - np.ndarray (of size num_iter + 1 [include initial guess])
def power_method(A, x0, num_iter): # 5 pts
    # enter your code here
    assert np.any(x0 != np.zeros_like(x0))
    assert type(x0) is np.ndarray
    x = x0
    res = []
    for _ in range(num_iter):
        s = np.inner(A @ x, x)
        res.append(np.linalg.norm(A @ x - s * x))
        if np.linalg.norm(A @ x) == 0:
            print("zero is encountered:",np.linalg.norm(A @ x))
        x = A @ x / np.linalg.norm(A @ x)
    return x, s, res


# INPUT:  A - np.ndarray (2D), d - float (from 0.0 to 1.0), x - np.ndarray (1D, size of A.shape[0/1])
# OUTPUT: y - np.ndarray (1D, size of x)
def pagerank_matvec(A, d, x): # 2 pts
    # enter your code here
    N = G.shape[0]
    pos = (G.sum(0) == 0).nonzero()[1]
    y = d * G @ x 
    y += d / N * np.sum(x[pos]) 
    y += (1 - d) / N * np.sum(x)
    return y


def return_words():
    # insert the (word, cosine_similarity) tuples
    # for the words 'numerical', 'linear', 'algebra' words from the notebook
    # into the corresponding lists below
    # words_and_cossim = [('word1', 'cossim1'), ...]
    
    numerical_words_and_cossim = [('computation', '0.547'),
 ('mathematical', '0.532'),
 ('calculations', '0.499'),
 ('polynomial', '0.485'),
 ('calculation', '0.473'),
 ('practical', '0.460'),
 ('statistical', '0.456'),
 ('symbolic', '0.455'),
 ('geometric', '0.441'),
 ('simplest', '0.438')]

    linear_words_and_cossim = [('differential', '0.759'),
 ('equations', '0.724'),
 ('equation', '0.682'),
 ('continuous', '0.674'),
 ('multiplication', '0.674'),
 ('integral', '0.672'),
 ('algebraic', '0.667'),
 ('vector', '0.654'),
 ('algebra', '0.630'),
 ('inverse', '0.622')]

    algebra_words_and_cossim = [('geometry', '0.795'),
 ('calculus', '0.730'),
 ('algebraic', '0.716'),
 ('differential', '0.687'),
 ('equations', '0.665'),
 ('equation', '0.648'),
 ('theorem', '0.647'),
 ('topology', '0.634'),
 ('linear', '0.630'),
 ('integral', '0.618')]
    
    return numerical_words_and_cossim, linear_words_and_cossim, algebra_words_and_cossim
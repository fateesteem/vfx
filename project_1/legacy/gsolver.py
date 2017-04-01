import numpy as np

def gsolver(Z, B, l, w):
    n = 256
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, Z.shape[0] + n), dtype = np.float32)
    b = np.zeros((A.shape[0], 1), dtype = np.float32)

    k = 1
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            w_ij = w[Z[i, j]]
            A[k, Z[i,j]] = w_ij
            A[k, n + i] = -w_ij
            b[k, 0] = w_ij * B[j]
            k = k + 1

    A[k, 128] = 1
    k = k + 1

    for i in range(n - 2):
        A[k, i] = l * w[i + 1]
        A[k, i + 1] = -2 * l * w[i + 1]
        A[k, i + 2] = l * w[i + 1]
        k = k + 1
    x = np.matmul(np.linalg.pinv(A), b)

    g = x[:n]
    ln_E = x[n + 1:]

    return np.reshape(g, (n, )), ln_E

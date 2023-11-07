import numpy as np
from scipy.spatial import distance_matrix

def distance_matrix_(fea_a, fea_b):
    nSmp_a, nFea = fea_a.shape
    nSmp_b, nFea = fea_b.shape

    aa = np.sum(fea_a * fea_a, axis=1)
    bb = np.sum(fea_b * fea_b, axis=1)
    ab = np.dot(fea_a, fea_b.T)

    aa = np.full((nSmp_a, ), aa)
    bb = np.full((nSmp_b,), bb)
    ab = np.full((nSmp_a, nSmp_b), ab)

    a = np.tile(aa.reshape(-1, 1), (1, nSmp_a))
    b = np.tile(bb, (nSmp_b, 1))
    D = a + b - 2 * ab
    D = np.abs(D)
    return D

def gaussianKernel(A, B, sigma):
    D = distance_matrix_(A, B)
    sig = np.max(D) / sigma
    K = np.exp(-D / sig)
    return K

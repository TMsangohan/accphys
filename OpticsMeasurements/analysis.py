import numpy as np

def getbeta(x,px):
    """
    Get beta, alpha and ex from turn by turn data (x,px)
    """
    U, S, v = np.linalg.svd([x,px]) # SVD
    N = np.dot(U,np.diag(S))
    theta = np.arctan(-N[0,1]/N[0,0]) # rotation angle of matrix
    c = np.cos(theta)
    s = np.sin(theta)
    R = [[c,s],[-s,c]]
    X = np.dot(N,R) # Floquet up to 1/det(USR)
    betx = np.abs(X[0,0]/X[1,1])
    alfx = X[1,0] / X[1,1]
    ex = S[0] * S[1] / (len(x)/2.) # emit = det(S)/(n/2)
    return betx, alfx, ex


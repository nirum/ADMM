"""
nuclear norm minimization via ADMM
author: Niru Maheswaranathan
10:12 PM May 5, 2014
"""
import numpy as np
from scipy.linalg import svd, norm, cho_factor, cho_solve

def admm(y,X,shape,options,penalty):
    """
    ADMM for nuclear norm minimization
    """

    # dimensions
    ds = shape[0]
    dt = shape[1]
    n  = ds*dt
    m  = float(y.size)
    sq = lambda k: k.reshape(ds,dt)

    # initialize variables
    k = np.zeros(n)
    z = np.zeros(n)
    u = k - z
    resid = np.zeros(options['maxiter'])

    # linear system
    P = X.T.dot(X) / m + penalty['rho']*np.eye(n)
    print('Condition number of P: %5.4f' % np.linalg.cond(P))
    L = cho_factor(P)
    xty = X.T.dot(y) / m

    # loop until convergence or maxiter is reached
    for idx in range(1,options['maxiter']):

        # minimize l2 error
        #k = solve(P, X.T.dot(y)/m + penalty['rho']*(z-u))
        k = cho_solve(L, xty + penalty['rho']*(z-u))

        # singular value thresholding
        U,S,V = svd(sq(k+u), full_matrices=False)
        z = (U.dot(np.diag(np.maximum(S-penalty['rank'],0))).dot(V)).ravel()

        # dual update
        u += k-z

        # stopping criterion
        print('Resid.\t\tError\tNuc. Norm')
        resid[idx] = norm(u)
        if (resid[idx] <= options['tol']): #| (np.abs(resid[idx]-resid[idx-1]) <= options['tol']):
            print('Converged after %i iterations.' % idx)
            break
        else:
            print('%5.4f\t\t%5.2f\t%5.4f' % (resid[idx], norm(X.dot(k)-y)/m, norm(S,1)))

    A = sq(0.5*(k+z))
    return A, k, z, u, resid

if __name__ == "__main__":

    # problem size
    ds = 20
    dt = 10
    r  = 3
    m  = 100
    n  = ds*dt

    # generate data
    A = np.random.randn(ds,r).dot(np.random.randn(r,dt))
    X = np.random.randn(m,n)
    y = X.dot(A.ravel())

    # options
    options = {'maxiter': 1000, 'tol': 1e-4}
    penalty = {'rho': 0.01, 'rank': 0.5}

    # run ADMM
    Ahat, k, z, u, resid = admm(y,X,A.shape,options,penalty)

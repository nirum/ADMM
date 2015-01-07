"""
ADMM python implementation
author: Niru Maheswaranathan
01:20 PM Aug 12, 2014
"""
import numpy as np
import proxops as po
from functools import partial

class ADMM(object):
    def __init__(self, lmbda):
        self.objectives = list()  # prox. operators for objectives
        self.lmbda = lmbda  # prox. op. trade-off parameter
        self.rho = 1.0 / lmbda  # inverse of trade-off parameter

    def add_operator(self, proxfun, **kwargs):
        # add proximal operator to the list
        proxop = partial(proxfun, lmbda=self.lmbda, **kwargs)
        self.objectives.append(proxop)

def lowrank_approx_demo():
    """
    solve a low-rank matrix approximation problem
    """

    # parameters
    n = 50  # dimension
    k = 3  # rank
    eta = 0.01  # noise strength
    gamma = 0.1  # low-rank penalty
    lmbda = 100.0  # ADMM parameter
    num_batches = 25

    # reproducible
    np.random.seed(1234)

    # build data matrix
    A_star = np.random.randn(n, k).dot(np.random.randn(k, n))
    data = [A_star + eta * np.random.randn(n, n) for j in range(num_batches)]

    # define objective and gradient (fro-norm)
    f = lambda x, d: 0.5 * np.sum((x.reshape(d.shape) - d) ** 2)
    fgrad = lambda x, d: (x.reshape(d.shape) - d).ravel()

    # initialize proximal operators and ADMM object
    lowrank = ADMM(lmbda)

    from sfo.sfo import SFO
    def f_df(x, d):
        return f(x,d), fgrad(x,d)

     # optimizer = SFO(f_df, 0.1*np.random.randn(n*n), data, display=1, admm_lambda=lmbda)

    ## set up SFO for ADMM iteration
    # def sfo_admm(v, lmbda):
    #     optimizer.set_theta(v)
    #     optimizer.theta_admm_prev = optimizer.theta_original_to_flat(v)
    #     return optimizer.optimize(num_steps=5)

    # lowrank.add(po.bfgs, f=f, fgrad=fgrad)
    lowrank.add_operator(po.sfo, f=f, fgrad=fgrad, data=data)
    # lowrank.add(sfo_admm)

    # theta_init = [0.1 * np.random.randn(n*n) for dummy in range(2)]
    # from sfo.sfo import SFO
    # optimizer = SFO(po.get_f_df(theta_init, lmbda, f, fgrad, data), theta_init, data, display=1)
    # lowrank.add(po.sfo_persist, optimizer=optimizer, f=f, fgrad=fgrad, data=data)

    lowrank.add_operator(po.nucnorm, gamma=gamma, array_shape=A_star.shape)

    # optimize
    A_hat = lowrank.optimize((n, n), maxiter=20)[0]
    print('\nLow-rank matrix approximation\n----------')
    print('Final Error: %4.4f' % np.linalg.norm(A_hat - A_star))
    print('')

    return A_hat, A_star


def lasso_demo():
    """
    solve a LASSO problem via ADMM
    """

    # generate problem instance
    n = 150
    p = 500
    A = np.random.randn(n, p)
    x_star = np.random.randn(p) * (np.random.rand(p) < 0.05)
    b = A.dot(x_star)

    # parameters
    sparsity = 0.8  # sparsity penalty
    lmbda = 2  # ADMM parameter

    # initialize prox operators and problem instance
    lasso = ADMM(lmbda)
    lasso.add_operator(po.linsys, P=A.T.dot(A), q=A.T.dot(b))
    lasso.add_operator(po.sparse, gamma=sparsity)

    # optimize
    x_hat = lasso.optimize(x_star.shape, maxiter=50)[0]
    print('\nLasso\n----------')
    print('Final Error: %4.4f' % np.sum((x_hat - x_star) ** 2))
    print('')

    return x_hat, x_star


if __name__ == "__main__":
    # x_hat, x_star = lasso_demo()
    A_hat, A_star = lowrank_approx_demo()

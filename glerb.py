import numpy as np
import torch as torch
import torch.nn as nn
import torch.distributions as dist
from sklearn.base import BaseEstimator
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_kernels
from scipy.optimize import root_scalar

class ExpFunc(nn.Module):
    def __init__(self,):
        super().__init__()
    def forward(self, X):
        return torch.exp(X)

def cut(X):
    up, low = 1-1e-4, 1e-4
    return torch.clip(X, low, up)

class GLERB(BaseEstimator):
    '''
    Parameters::
        K: int, default=8
            Number of nearest neighbors.
        lam: float, default=50
            Strength of logical labels.
        verbose: int, default=0 
            How many intermediate ELBO values will be printed during training.
        lr: float, default=1e-3
            Learning rate of Adam.
        max_iter: int, default=200
            Maximum iterations of Adam.
    --------------------------------------------------------------
    Attributes::
        label_distribution_: ndarray of shape (n_samples, n_labels)
            Recovered label distributions.
    --------------------------------------------------------------
    Methods::
        fit(X, Y): training the model with feature matrix X and logical label matrix Y.
    --------------------------------------------------------------
    Examples::
        >>> Drec = GLEMR().fit(X, L).label_distribution_
        >>> evaluate(Drec, ground_truth)
    '''
    def __init__(self, K=8, lam=50, verbose=0, max_iter=200, lr=1e-3, random_state=123):
        self.K = K
        self.lam = lam
        self.max_iter = max_iter
        self.lr = lr
        self.verbose = verbose
        self.random_state = random_state

    def fix_random_state(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
    
    def fit(self, X, Y):
        self.fix_random_state()
        n, m = Y.shape
        c = X.shape[1]
        if self.K > 1:
            NSm = kneighbors_graph(X, n_neighbors=self.K, include_self=True).T.toarray()
            NSm[np.arange(n), np.arange(n)] = 1
            NSm /= NSm.sum(1, keepdims=True)
        else:
            NSm = np.eye(n)
        # make the Graph Laplacian for mean function
        A = pairwise_kernels(Y.T, metric=lambda x, y: x[x * y == 1].size / x[x + y != 0].size)
        Dv = np.power(A.sum(1), -.5)
        L = A * Dv.reshape(-1, 1) * Dv.reshape(1, -1)
        eigvals, col_eigvec_mat = np.linalg.eig(L)
        col_eigvec_mat = col_eigvec_mat.T[:,:,None]
        basis = col_eigvec_mat @ col_eigvec_mat.swapaxes(1,2)
        eigvals = eigvals[:,None,None]
        right, scaler = 0.9999, []
        def lpf(y):
            def func(alpha):
                vec = ((1-alpha) * np.linalg.inv(np.eye(m) - alpha * L) @ y.reshape(-1,1)).flatten()
                return np.min(vec[y==1].reshape(-1,1) - vec[y==0].reshape(1,-1))
            return func
        scaler = np.ones(n)
        for y in np.unique(Y, axis=0):
            upper = 1 if lpf(y)(right) > 0 else root_scalar(lpf(y), bracket=[0.5, right]).root
            scaler[(Y == y[None,:]).all(axis=1)] = upper
        scaler = scaler[:, None]
        # torch-Tensorize
        L, eigvals, basis, Y, X, NSm, scaler = [torch.FloatTensor(_x) for _x in [L, eigvals, basis, Y, X, NSm, scaler]]
        X = (X - X.min(0, keepdims=True)[0]) / (X.max(0, keepdims=True)[0] - X.min(0, keepdims=True)[0])

        h = nn.Linear(c, 1)
        mufn = nn.Linear(m, c)
        lamfn = nn.Sequential(nn.Linear(c, m), ExpFunc())
        params = list(h.parameters()) + list(mufn.parameters()) + list(lamfn.parameters())

        def train(init):
            for p in params:
                nn.init.constant_(p, val=init)
            optimizer = torch.optim.Adam(params, lr=self.lr)
            
            for epoch in range(self.max_iter+1):
                optimizer.zero_grad()
                epsilon = cut(torch.rand(n, m))
                alpha = (torch.sigmoid(h(X)) * scaler).unsqueeze(-1).unsqueeze(-1)
                Phi = (basis * ( (1-alpha) / (1 - eigvals * alpha) )).sum(1)
                self.Kuma_median = (Y.unsqueeze(1) @ Phi).squeeze(1) # shape=(n,m)
                B = lamfn(X)
                A = cut( cut(1-2**(-1/B)).log() / cut(self.Kuma_median).log() )  # shape=(n,m)
                KLloss = ( A.log() + B.log() + ((1 - A) * torch.digamma(B) + 1/B + np.euler_gamma) / A ).sum()
                Z = cut((1 - epsilon.pow(1/B)).pow(1/A))   # shape=(n,m)
                xrec = dist.Normal(mufn(Z), 1).log_prob(X).sum()
                yrec = (Y * (NSm @ Z.log()) + (1 - Y) * (NSm @ (1 - Z).log())).sum() * self.lam
                elbo = xrec + yrec - KLloss
                loss = -elbo
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if (self.verbose > 0) and (epoch % (self.max_iter // self.verbose) == 0):
                        print("* epoch %4d, elbo: %.3f" % (epoch, (xrec+yrec/self.lam-KLloss)/n))
        try: train(1) 
        except: train(0)
        with torch.no_grad():
            self.label_distribution_ = (cut(self.Kuma_median) / cut(self.Kuma_median).sum(1, keepdims=True)).numpy()
        return self


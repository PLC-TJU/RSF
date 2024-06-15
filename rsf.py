# Riemannian Geometry-Based Spatial Filtering (RSF)
# Author: Pan Lincong
# Edition date: 5 Mar 2024

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize
import scipy.linalg as la
from scipy.linalg import eigh, eig
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.spatialfilters import CSP
from pyriemann.utils.distance import distance
# from numba import jit, cuda

def optimizeRiemann_old(P1, P2, W0=None, N=8, maxiter=5000,):
    M = P1.shape[0]
    
    if W0 is None:
        W0 = np.random.randn(M, N)
        W0 = np.linalg.qr(W0)[0]  # Orthogonalize W0
    else:
        N = W0.shape[1]
    
    W0_flat = W0.flatten()
    def objFunc(W_flat):
        W = W_flat.reshape(M, -1)
        eigvals = la.eigvals(W.T @ P1 @ W, W.T @ P2 @ W)
        eigvals = np.real(eigvals)
        return -np.sum(np.log(eigvals)**2)
    
    result = minimize(objFunc, W0_flat, method='trust-constr', options={'maxiter': maxiter})
    W_opt = result.x.reshape(M, -1)
    d0 = np.sum(np.log(la.eigvals(W0.T @ P1 @ W0, W0.T @ P2 @ W0))**2)
    d1 = np.sum(np.log(la.eigvals(W_opt.T @ P1 @ W_opt, W_opt.T @ P2 @ W_opt))**2)
    return W0 if d0 > d1 else W_opt

def optimizeRiemann(P1, P2, W0=None, N=8, maxiter=1000, collect_obj_values=False, 
                    solver='trust-constr', tolerance=1e-8, verbose=0):
    M = P1.shape[0]
    obj_values = []
    W = []

    if P1.shape[0] != P2.shape[0] or P1.shape[1] != P2.shape[1]:
        raise ValueError("The input data must have the same number of samples")
    if P1.shape[0] < N:
        raise ValueError("The number of samples is less than the number of filters")    

    if W0 is None:
        W0 = np.random.randn(M, N)
        W0, _ = np.linalg.qr(W0) 
    W0_flat = W0.ravel()
    
    def objFunc(W_flat):
        W = W_flat.reshape(M, -1)
        eigvals = eigh(W.T @ P1 @ W, W.T @ P2 @ W, eigvals_only=True)
        eigvals = np.clip(eigvals, a_min=1e-10, a_max=None)  # Avoid log(0) issues
        return -np.sum(np.log(eigvals)**2)
    
    def callback(xk,state=None):
        W.append(xk.reshape(M, -1))
        obj_values.append(-state.fun) if solver == 'trust-constr' else obj_values.append(-objFunc(xk))
    
    result = minimize(objFunc, W0_flat, method = solver, 
                      options={'maxiter': maxiter, 'gtol': tolerance, 'verbose': verbose}, 
                      callback=callback if collect_obj_values else None)
        
    W_opt = result.x.reshape(M, -1)
    d0 = -objFunc(W0_flat) 
    d1 = -objFunc(result.x) 
    return (W0 if d0 > d1 else W_opt), (obj_values if collect_obj_values else None), (W if collect_obj_values else None)

def rsf_demo(traindata, trainlabel, testdata=None, dim=4, method='default'):
    """Riemannian geometry-based spatial filter

    Args:
        traindata (ndarray): train samples. shape (n_trials, n_channels, n_times)
        trainlabel (ndarray): train labels. shape (n_trials,)
        testdata (ndarray, optional): test samples. shape (n_trials, n_channels, n_times). Defaults to None.
        dim (int, optional): spatial filters. Defaults to 4.
        method (str, optional): _description_. Defaults to 'default'.

    Returns:
        trainData: train data after RSF filtering
        testData: test data after RSF filtering
        dd: objective function value
        W: spatial filter
    """
    
    if method.lower() != 'none':
        labeltype = np.unique(trainlabel)
        traincov = covariances(traindata,'cov')
        covm1 = mean_covariance(traincov[trainlabel == labeltype[0]], metric='riemann')
        covm2 = mean_covariance(traincov[trainlabel == labeltype[1]], metric='riemann')
        
    else:
        trainData = traindata
        W = np.eye(traindata.shape[1])
        if testdata is None:
            testData = None
        else:
            testData = testdata
        dd = np.nan
        return trainData, testData, dd, W

    if method.lower() == 'csp':
        scaler = CSP(nfilter=dim, metric='euclid')
        CSPmodel = scaler.fit(traincov,trainlabel)
        W0 = CSPmodel.filters_.T
        W = optimizeRiemann_old(covm1, covm2, W0=W0)
    elif method.lower() == 'riemann-csp':
        scaler = CSP(nfilter=dim, metric='riemann')
        CSPmodel = scaler.fit(traincov,trainlabel)
        W0 = CSPmodel.filters_.T
        W = optimizeRiemann_old(covm1, covm2, W0=W0)
    else:
        try:
            W = optimizeRiemann_old(covm1, covm2, N=dim)
        except:
            _, _, _, W = rsf_demo(traindata, trainlabel, testdata, dim, method)

    trainData = np.zeros((traindata.shape[0], dim, traindata.shape[2]))
    for i in range(trainData.shape[0]):
        trainData[i, :, :] = W.T @ traindata[i, :, :]

    if testdata is None:
        testData = None
    else:
        testData = np.zeros((testdata.shape[0], dim, testdata.shape[2]))
        for i in range(testData.shape[0]):
            testData[i, :, :] = W.T @ testdata[i, :, :]

    dd = distance(W.T @ covm1 @ W, W.T @ covm2 @ W, metric='riemann')
    return trainData, testData, dd, W

class RSF(BaseEstimator, TransformerMixin):
    def __init__(self, dim=8, method='default', solver='trust-constr', flag=False):
        """
        Initialize the RSF Transformer.

        Parameters:
        - dim (int, optional): Number of spatial filters to compute (default: 4).
        - method (str, optional): Filtering method ('default', 'csp', or 'riemann-csp').  
        - solver (str, optional): Optimization solver ('trust-constr', 'bfgs',etc.).
        - flag (bool, optional): Whether to collect objective function values during optimization (default: False).
        """
        self.dim = dim
        self.method = method.lower() if method is not None else 'none'
        self.flag = flag
        self.solver = solver
        self.W = None
        self.obj_values = None
        self.W_history = None

    def fit(self, X, y):
        """
        Fit the RSF Transformer to the data.

        Parameters:
        - X (array-like, shape [n_trials, n_channels, n_times]): EEG data.
        - y (array-like, shape [n_trials]): Class labels.

        Returns:
        - self: Fitted RSF Transformer instance.
        """
        if self.method != 'none':
            labeltype = np.unique(y)
            traincov = covariances(X, estimator='lwf')
            if self.method != 'cspf':
                covm1 = mean_covariance(traincov[y == labeltype[0]], metric='riemann')
                covm2 = mean_covariance(traincov[y == labeltype[1]], metric='riemann')
        else:
            self.W = np.eye(X.shape[1])
            return self

        if self.method == 'rsf-csp':
            scaler = CSP(nfilter=self.dim, metric='euclid')
            CSPmodel = scaler.fit(traincov,y)
            W0 = CSPmodel.filters_.T
            self.W, self.obj_values, self.W_history = optimizeRiemann(covm1, covm2, W0=W0, 
                                                      collect_obj_values=self.flag,
                                                      solver=self.solver)    
            
        elif self.method == 'rsf-rcsp': 
            scaler = CSP(nfilter=self.dim, metric='riemann')
            CSPmodel = scaler.fit(traincov,y)
            W0 = CSPmodel.filters_.T
            self.W, self.obj_values, self.W_history = optimizeRiemann(covm1, covm2, W0=W0, 
                                                      collect_obj_values=self.flag,
                                                      solver=self.solver)    
        elif self.method == 'cspf':
            scaler = CSP(nfilter=self.dim)
            CSPmodel = scaler.fit(traincov,y)
            self.W = CSPmodel.filters_.T     
            
        else:
            self.W, self.obj_values, self.W_history = optimizeRiemann(covm1, covm2, N=self.dim, 
                                                      collect_obj_values=self.flag,
                                                      solver=self.solver)

        return self

    def transform(self, X):
        """
        Transform the input EEG data using the learned RSF spatial filters.

        Parameters:
        - X (array-like, shape [n_trials, n_channels, n_times]): EEG data.

        Returns:
        - transformed_data (array-like, shape [n_trials, dim, n_times]):
        Transformed EEG data after applying RSF spatial filters.
        """
        if self.method != 'none':
            # Apply spatial filters using vectorized operations
            transformed_data = np.einsum('ij,kjl->kil', self.W.T, X)
        else:
            transformed_data = X

        return transformed_data

    
# import pymanopt
# from pymanopt.manifolds import Stiefel
# from pymanopt.manifolds import SymmetricPositiveDefinite as SPD
# from pymanopt import Problem
# import pymanopt.optimizers
# import autograd.numpy as anp
# def optimizeRiemann2x(P1, P2, W0=None, N=8, maxiter=5000, **kwargs):
    
#     M = P1.shape[0]
    
#     #Ensure the input data is valid
#     if P1.shape[0] != P2.shape[0] or P1.shape[1] != P2.shape[1]:
#         raise ValueError("The input data must have the same number of samples")
#     if P1.shape[0] < N:
#         raise ValueError("The number of samples is less than the number of filters")    

#     if W0 is not None:
#         N = W0.shape[1]
        
#     # 定义流形
#     manifold = Stiefel(M, N)
    
#     # 定义目标函数
#     @pymanopt.function.autograd(manifold)
#     def cost(W):
#         # 计算特征值并确保它们是正的

#         # eigvals, _ = anp.linalg.eig(W.T @ P1 @ W, W.T @ P2 @ W)
#         A = W.T @ P1 @ W
#         B = W.T @ P2 @ W
#         # eigvals, _ = anp.linalg.eig(anp.linalg.inv(B) @ A)
#         eigvals, _ = eigh(A, B)
        
#         # 确保特征值大于1e-10
#         # eigvals = np.clip(eigvals, a_min=1e-10, a_max=None)  # Avoid log(0) issues
#         # 计算成本
#         return -np.sum(np.log(eigvals)**2)

#     # 定义问题
#     problem = Problem(manifold=manifold, cost=cost)

#     # 选择优化器并优化
#     # optimizer = pymanopt.optimizers.SteepestDescent(max_iterations=maxiter)
#     optimizer = pymanopt.optimizers.TrustRegions(max_iterations=maxiter)
#     result = optimizer.run(problem, initial_point=W0)
#     W_opt = result.point

#     return W_opt


# from pymanopt.tools.multi import (
#     multiexpm,
#     multihconj,
#     multilogm,
# )
# def optimizeRiemann2(P1, P2, W0=None, N=8, maxiter=5000, **kwargs):
    
#     def dist(point_a, point_b):
        
#         c = np.linalg.cholesky(point_a)
#         c_inv = np.linalg.inv(c)
#         logm = multilogm(
#             c_inv @ point_b @ multihconj(c_inv),
#             positive_definite=True,
#         )
#         return np.real(np.linalg.norm(logm))
    
#     def extract_value(W):
#         # 检测 W 是否有 _value 属性，如果有，则将 W 设置为 W._value
#         while hasattr(W, '_value'):
#             W = W._value
#         # 返回 W 的真实数据值
#         return W
    
#     M = P1.shape[0]
    
#     #Ensure the input data is valid
#     if P1.shape[0] != P2.shape[0] or P1.shape[1] != P2.shape[1]:
#         raise ValueError("The input data must have the same number of samples")
#     if P1.shape[0] < N:
#         raise ValueError("The number of samples is less than the number of filters")    

#     if W0 is not None:
#         N = W0.shape[1]
        
#     # 定义流形
#     manifold = Stiefel(M, N)
#     # manifold = SPD(N)
    
#     # 定义目标函数
#     @pymanopt.function.autograd(manifold)
#     def cost(W):
#         W = extract_value(W)
        
#         A = W.T @ P1 @ W
#         B = W.T @ P2 @ W

#         # 计算成本
#         return -dist(A, B)

#     # 定义问题
#     problem = Problem(manifold=manifold, cost=cost)

#     # 选择优化器并优化
#     # optimizer = pymanopt.optimizers.SteepestDescent(max_iterations=maxiter)
#     optimizer = pymanopt.optimizers.TrustRegions(max_iterations=maxiter)
#     result = optimizer.run(problem, initial_point=W0)
#     W_opt = result.point

#     return W_opt

        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tung doan
"""
import numpy as np

from tslearn.clustering import TimeSeriesKMeans
from math import exp
from dtw import dtw_grad

epsilon = 1e-6 # small value to avoid avoid division by zero
psi = 5e-2 # parameter control random noise for initialization

def tmf(X,k,l,lambda_1,lambda_2,lambda_3,sigma,eta,o_max,i_max):
    
    print('start...')
    
    """ Initialization """
    D = gen_D(l)
    F, G = init(X, l, k)
    
    F_list = list(); F_list.append(F)
    G_list = list(); G_list.append(G)
    
    """ Outer loop """
    for i in range(o_max):
        
        print('outer iteration', i+1, '.')
        
        """ Centroid similarity matrix """
        H = compute_H(F, k, sigma)
        
        """ Gradient w.r.t FG """
        Z = compute_Z(X, F, G, l)
        
        """ Update F """
        for j in range(i_max):
            Y = compute_Y(H,F,k,l,sigma)
            F = F - eta * ( psi * np.matmul(G.T,Z) + lambda_1 * Y  + lambda_2 * np.linalg.multi_dot([F,D,D.T]) )
        
        F_list.append(F)
            
        """ Update G """
        G_grad = compute_M(F,G,Z,k,lambda_3)
        G = np.multiply(G, G_grad)
        # Normalize G to avoid its arbitrary volumes
        G_sum = np.sum(G,axis=1)
        G = np.divide(G,G_sum[:, np.newaxis])
        
        G_list.append(G)
        
    print('finish.')
        
    #return F, G
    return F_list, G_list    
      
def gen_D(l):
    D = np.zeros((l,l-2))
    I = np.eye(l-2)
    D[0:l-2,:] = D[0:l-2,:] - I
    D[1:l-1,:] = D[1:l-1,:] + 2 * I
    D[2:l,:] = D[2:l,:] - I
    return D

def init(X,l,k):
    # Good initial start improve to convergence 
    seed = 0
    sdtw_km = TimeSeriesKMeans(n_clusters=k, metric="euclidean", max_iter=10, random_state=seed)
    sdtw_km.fit(X)
    G_init = np.zeros((sdtw_km.labels_.size, sdtw_km.labels_.max()+1)) 
    G_init[np.arange(sdtw_km.labels_.size),sdtw_km.labels_] = 1
    G_init = G_init + np.random.rand(G_init.shape[0],G_init.shape[1])
    F_init = sdtw_km.cluster_centers_[:,:,0] + 2 ** psi * np.random.rand(k,l)
    return F_init, G_init 
        
def compute_H(F,k,sigma):
    H = np.zeros((k,k))
    for i in range(k):
        for j in range(i+1,k):
            H[i,j] = H[j,i] = exp( - (np.linalg.norm(F[i] - F[j]) ** 2) / sigma)
    return H

def dismat(x,y):
    m = len(x)
    n = len(y)
    theta = np.ndarray(shape=(m,n), dtype=float)
    for i in range(m):
        for j in range(n):
            theta[i,j] = (x[i] - y[j]) ** 2
    return theta    

def compute_Z(X,F,G,l):
    Z = np.zeros((X.shape[0],l))
    GF = np.matmul(G, F)
    for i in range(X.shape[0]):
        theta = dismat(X[i],GF[i])
        (v, grad, Q, E) = dtw_grad(theta,'softmax')
        Z[i] = 2 * ( np.multiply(np.sum(grad, axis=0), GF[i]) - np.matmul(X[i], grad) )
    return(Z)    

def compute_Y(H,F,k,l,sigma):
    Y = np.zeros((k,l))
    for i in range(k):
        for j in range(k):
            Y[i] = Y[i] + H[i,j] ** 2 * (F[j] - F[i])
    return 4 / sigma * Y 

def compute_S(G,k):
    G_sign = np.sign(G)
    G_norm = np.linalg.norm(G,ord=1,axis=1)
    S = np.multiply(G_sign, G_norm[:, np.newaxis])
    return S          
    
def compute_M(F,G,Z,k,lambda_3):
    S = compute_S(G,k)
    M = np.matmul(Z, F.T) + lambda_3 * S
    M_absolute = np.absolute(M)
    M_positive = M_absolute + M + epsilon
    M_negative = M_absolute - M + epsilon  
    M = np.divide(M_positive,M_negative)
    return np.sqrt(M)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tung doan
"""
import numpy as np
import matplotlib.pyplot as plt

from tslearn.datasets import UCR_UEA_datasets
from tmf import tmf
from scipy.io import loadmat

""" load data """
data_loader = UCR_UEA_datasets()
X_tr, y_tr, X_te, y_te = data_loader.load_dataset('Coffee')
X = X_tr[:,::2,0] #reduce length a factor of 2 for fast demo
y = y_tr
# Ground truth indicator matrix
grd = np.zeros((y.size, y.max()+1)) 
grd[np.arange(y.size),y] = 1

""" run temporal matrix factorization """
k = y.max()+1; l = X.shape[1]; lambda_1 = lambda_2 = 1e-2; lambda_3 = 10; sigma = 0.05 ** 2; eta = 1e-2; o_max = 15; i_max = 50;
F_list, G_list = tmf(X, k, l, lambda_1, lambda_2, lambda_3, sigma, eta, o_max, i_max)

#ddict = loadmat('results.mat')
#F_list = ddict['F'] 
#G_list = ddict['G']

""" plot """
plt.style.use(style='ggplot')
colors = ['tab:blue','tab:red','tab:green','tab:black','tab:cyan']
plt.figure(1)

# Plot initial centroid
plt.title('Initial centroids')
for i in range(k):
    plt.plot(F_list[0][i],color=colors[i],label='Centroid '+str(i+1),linewidth=2)
plt.legend()

# Plot resulted centroid
plt.figure(2)
plt.title('Resulted centroids')
for i in range(k):
    plt.plot(F_list[-1][i],color=colors[i],label='Centroid '+str(i+1),linewidth=2)
plt.legend()

# Plot indicator matrix 
plt.style.use(style='classic')
fig, axs = plt.subplots(2,1,figsize=(100,50))
## Plot ground truth indicator matrix
axs[0].set_title('Ground truth indicators',pad=20)
axs[0].matshow(grd.T, cmap=plt.cm.Blues)
for i in range(y.shape[0]):
    for j in range(k):
        c = format(grd[i,j], '.1f') 
        axs[0].text(i, j, c, va='center', ha='center')
axs[0].set_xticks(np.arange(y.shape[0]))
axs[0].xaxis.set_ticks_position('bottom')        
## Plot resulted indicator matrix 
axs[1].set_title('Resulted indicators',pad=20)
axs[1].matshow(G_list[-1].T, cmap=plt.cm.Blues)
for i in range(X.shape[0]):
    for j in range(k):
        c = format(G_list[-1][i,j], '.1f') 
        axs[1].text(i, j, c, va='center', ha='center')
axs[1].set_xticks(np.arange(X.shape[0]))
axs[1].xaxis.set_ticks_position('bottom')

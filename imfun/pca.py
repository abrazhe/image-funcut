### Simple PCA in python
### Based on:
### (c) Jan Erik Solem
### link: http://www.janeriksolem.net/2009/01/pca-for-images-using-python.html

import numpy as np

def pca(X):
  # Principal Component Analysis
  # input: X, matrix with training data as flattened arrays in rows
  # return: projection matrix (with important dimensions first),
  # variance and mean

  #get dimensions
  num_data,dim = X.shape

  #center data
  mean_X = X.mean(axis=0)
  for i in range(num_data):
      X[i] -= mean_X

  if dim>100:
      print 'PCA - compact trick used'
      M = np.dot(X,X.T) #covariance matrix
      e,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
      kei = np.argsort(e)
      tmp = np.dot(X.T,EV).T #this is the compact trick
      V = tmp[kei] # sort by eigenvalues
      S = np.sqrt(e)[kei] #sort by eigenvalues
  else:
      print 'PCA - SVD used'
      U,S,V = np.linalg.svd(X)
      V = V[:num_data] #only makes sense to return the first num_data

  #return the projection matrix, the variance and the mean
  return V,S,mean_X

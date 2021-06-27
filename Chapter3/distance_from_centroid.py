import pandas as pd
import numpy as np
import scipy.linalg as la
from sklearn.preprocessing import MinMaxScaler


def distance_from_centroid_scores(data):
    
    """
    
    This function computes the outlier scores of the items that haves historical 
    transactions. We used the algorithm presented in page 78 of the Outlier 
    Analysis. This function takes one argument:
    
    data: a data-frame containing the observations to be score. Missing values are 
    not allowed.
    
    """
    
    temp_data = data
    
    ## Standardizing the data
    scaler = MinMaxScaler().fit(temp_data)
    temp_data = scaler.transform(temp_data)

    ## Computing the covariance matrix
    sigma = np.cov(temp_data, rowvar = False)
    
    ## Computing eigenvalues and eigenvectos of the covariance matrix
    eigvals, eigvecs = la.eig(sigma)
    
    ## Defining D and P (for PCA outlier score algorithm form Outlier 
    ## Analysis book)
    D = temp_data
    P = eigvecs

    ## Computing D'
    D_prime = np.matmul(D, P)

    ## Standardizing (dividing each column by it standard deviation)
    for i in range(0, D_prime.shape[1]):
        
        D_prime[:, i] = D_prime[:, i] / D_prime[:, i].std(ddof = 1)
    
    ## Computing the centroid
    centroid = D_prime.mean(axis = 0)
    
    ## Declaring list to store Euclidean distances
    distances = []
    
    ## Finding the number of rows in data
    n = D_prime.shape[0]
    
    for i in range(0, n):
        
        ## Selecting the i-th row
        temp = D_prime[i, :]
        
        ## Computing the Euclidean distance
        distances.append(np.sqrt(np.sum((temp - centroid)**2)))
    
    return distances
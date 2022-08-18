import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn
import pandas as pd
import random
import math
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal

class GMM:
    
    def __init__(self, C, n_runs):
        self.C = C # number of Guassians/clusters
        self.n_runs = n_runs
        
    
    def get_params(self):
        return (self.mu, self.pi, self.sigma)
    
    
    def calculate_mean_covariance(self, X):
        d = X.shape[1]
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros(self.C)

        cov_ = np.cov(X.astype(float).T)
        identity = np.eye(len(self.train[0]), dtype=int)
        for c in range(self.C):
            initial = cov_
            self.initial_cov[c] = initial + 0.0001*identity
            self.initial_means[c] = np.mean(X, axis = 0)
            self.initial_pi[c] = random.random()


        self.initial_pi = self.initial_pi/np.sum(self.initial_pi)
        
        return (self.initial_means, self.initial_cov, self.initial_pi)
    
    
    
    def _initialise_parameters(self, X):
    
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X)    
        
        return (self._initial_means, self._initial_cov, self._initial_pi)
    
    def likelihood(self, X, mi, sigmai):
       
        P = multivariate_normal.pdf(X, mean=mi, cov=sigmai)
       
        return np.array(P) 
        
    def _e_step(self, X, pi, mu, sigma):
    
        N = X.shape[0] 
        self.gamma = np.zeros((N, self.C))
        
        
        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for c in range(self.C):
        
             
            identity = np.eye(len(self.train[0]), dtype=int)
          
            sigma_ = self.sigma[c]
            self.sigma[c] =  sigma_ + 0.0001*identity
        
            self.gamma[:,c] = self.pi[c] * self.likelihood(X, self.mu[c,:], self.sigma[c]) 
           
        gamma_norm = np.sum(self.gamma, axis=1)+ 0.0001#[:,np.newaxis] + 0.0001
        gamma_norm = np.reshape(gamma_norm, (gamma_norm.shape[0], 1))
        
        self.gamma /= gamma_norm
       
        return self.gamma
    
    
    def _m_step(self, X, gamma):
        N = X.shape[0] # number of objects
        C = self.gamma.shape[1] # number of clusters
        d = X.shape[1] # dimension of each object

    
        self.pi = np.mean(self.gamma, axis = 0)
       
        self.mu = np.dot(self.gamma.T, X) / (np.sum(self.gamma, axis = 0)[:,np.newaxis]+0.0001)
    
       
        for c in range(C):
            x = X - self.mu[c, :] # (N x d)
            
            gamma_diag = np.diag(self.gamma[:,c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)

            
            sigma_c = x_mu.T * gamma_diag * x_mu
            
            self.sigma[c,:,:]=(sigma_c) / (np.sum(self.gamma, axis = 0)[:,np.newaxis][c]+0.0001)
            
        #print("Parameters: ", self.pi, self.mu, self.sigma)
        return self.pi, self.mu, self.sigma
    
    
    def _compute_loss_function(self, X, pi, mu, sigma):
       
        N = X.shape[0]
        C = self.gamma.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            identity = np.eye(len(self.train[0]), dtype=int)
            self.sigma[c] = self.sigma[c]+ 0.0001*identity
            dist = mvn(self.mu[c], self.sigma[c],allow_singular=True)
            self.loss[:,c] = self.gamma[:,c] * (np.log(self.pi[c]+0.00001)+dist.logpdf(X)-np.log(self.gamma[:,c]+0.000001))
        self.loss = np.sum(self.loss)
        return self.loss
    
    
    
    def fit(self, X):
        self.train = X
        d = X.shape[1] 
        self.mu, self.sigma, self.pi =  self._initialise_parameters(X)
        
    
        try:
            for run in range(self.n_runs):  
                
                identity = np.eye(len(self.train[0]), dtype=int)
               
                self.sigma = self.sigma+ 0.0001*identity
                
                self.gamma  = self._e_step(X, self.mu, self.pi, self.sigma)
                
                self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)
                identity = np.eye(len(self.train[0]), dtype=int)
                self.sigma = self.sigma+ 0.0001*identity
        
        
        except Exception as e:
            print(e)
        
        return (self.mu, self.sigma, self.pi)
    
    
    def predict(self, X):
    
        labels = np.zeros((X.shape[0], self.C))
        
        for c in range(self.C):
            labels [:,c] = self.pi[c] * self.likelihood(X, self.mu[c,:], self.sigma[c])
        labels  = labels .argmax(1)
    
        return labels 
    
    def predict_proba_x(self, x, mu, sigma, pi):
        
        l = len(x)
        post_proba = np.zeros((1, self.C))
        
        for c in range(self.C):
            identity = np.eye(len(self.train[0]), dtype=int)
            
            sigmai = sigma[c]+ 0.001*identity
            #Posterior Distribution using Bayes Rule, try and vectorise
            mi = mu[c]
            #P = (1.0/(pow(2*math.pi, float(l)/2) * pow(det(sigmai), 1.0/2) )) * math.exp(-1.0/2 * np.dot(np.dot((x - mi).T, inv(sigmai)), x - mi))
            P = multivariate_normal.pdf(x, mean=mi, cov=sigmai)
          
            post_proba[:,c] = pi[c] * P
    
        return post_proba

    def accuracy(self, actual, predicted, n_classes):
        correct = 0
        confusion_matrix = np.zeros((n_classes,n_classes), int)    
        for i in range(len(actual)):
            actual[i] = int(actual[i])
            confusion_matrix[actual[i]][predicted[i]] = confusion_matrix[actual[i]][predicted[i]] + 1
            if actual[i] == predicted[i]:
                correct += 1
        
        acc = correct / float(len(actual)) * 100.0
        return confusion_matrix, acc
  
    

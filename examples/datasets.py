import pdb

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
from sklearn.datasets import fetch_openml
import scipy
from scipy.spatial.distance import cdist
import pandas as pd
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
import os
import scipy.misc
import matplotlib.image as mpimg
from scipy.stats import truncnorm

def read_img(fpath, grayscale=False, bbox=None):
    if grayscale:
        img = ImageOps.grayscale(Image.open(fpath))
    else:
        img = Image.open(fpath)
    if bbox is not None:
        return np.asarray(img.crop(bbox).reduce(2))
    else:
        return np.asarray(img.reduce(2))
    
def do_pca(X, n_pca):
    print('Applying PCA')
    pca = PCA(n_components=n_pca, random_state=42)
    pca.fit(X)
    #print('explained_variance_ratio:', pca.explained_variance_ratio_)
    print('sum(explained_variance_ratio):', np.sum(pca.explained_variance_ratio_))
    #print('singular_values:', pca.singular_values_)
    X = pca.fit_transform(X)
    return X

class Datasets:
    def __init__(self):
        pass
    
    def linesegment(self, RES=100, noise=0):
        np.random.seed(42)
        L = 1
        xv = np.linspace(0, L, RES)[:,None]
        yv = noise*np.random.randn(xv.shape[0],1)
        X = np.concatenate([xv,yv],axis=1)
        labelsMat = X[:,0][:,None]
        ddX = np.minimum(X,L-X).flatten()
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def helicical_tree(self, RES=100, noise=0):
        return None
    
    def circle(self, RES=100, noise=0):
        np.random.seed(42)
        theta = np.linspace(0, 2*np.pi, RES)[:-1]
        xv = np.cos(theta)[:,None]
        yv = np.sin(theta)[:,None]
        X = np.concatenate([xv,yv], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def figure_eight2d(self, RES=100, noise=0, noise_type='normal', seed=42):
        np.random.seed(seed)        
        theta = np.linspace(0, 2*np.pi, RES)[:-1]
        xv = np.sin(2*theta)[:,None]
        yv = np.cos(theta)[:,None]
        X = np.concatenate([xv,yv], axis=1)
        if 'normal' in noise_type:
            noise = noise*np.random.normal(0,1,X.shape)
        elif 'uniform' in noise_type:
            noise = noise*np.random.uniform(-1,1,X.shape)
        else:
            noise = noise*(0.01 + 0.49*(1+np.cos(2*theta))/2)[:,None]*np.random.normal(0,1,X.shape)

        if 'ortho' in noise_type:
            dx = 2*np.cos(2*theta)[:,None]
            dy = -np.sin(theta)[:,None]
            normal_dir = np.concatenate([-dy,dx], axis=1)
            normal_dir = normal_dir/np.linalg.norm(normal_dir, axis=1)[:,None]
            X = X + noise[:,0:1]*normal_dir
        else:
            X = X+noise
        labelsMat = theta[:,None]
        print('X.shape = ', X.shape)
        return X, labelsMat, None

    def figure_eight(self, RES=100, noise=0, noise_type='normal', seed=42):
        np.random.seed(seed)        
        theta = np.linspace(0, 2*np.pi, RES)[:-1]
        xv = np.sin(2*theta)[:,None]
        yv = np.cos(theta)[:,None]
        X = np.concatenate([xv,yv,xv*0], axis=1)
        if noise_type=='normal':
            noise = noise*np.random.normal(0,1,X.shape)
        else:
            noise = noise*(0.01 + 0.49*(1+np.cos(2*theta))/2)[:,None]*np.random.normal(0,1,X.shape)
        X = X+noise
        labelsMat = theta[:,None]
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def almost_figure_eight(self, RES=100, noise=0, noise_type='normal', seed=42, z_scale=0.125):
        np.random.seed(seed)        
        theta = np.linspace(0, 1.75*np.pi, RES)[:-1]
        xv = np.sin(2*theta)[:,None]
        yv = np.cos(theta)[:,None]
        X = np.concatenate([xv,yv,z_scale*theta[:,None]], axis=1)
        if noise_type=='normal':
            noise = noise*np.random.normal(0,1,X.shape)
        else:
            noise = noise*(0.01 + 0.49*(1+np.cos(2*theta))/2)[:,None]*np.random.normal(0,1,X.shape)
        X = X+noise
        labelsMat = np.column_stack([theta, np.linalg.norm(noise, axis=1)])
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def checkered_board(self, m, n, RES=100, noise=0):
        np.random.seed(42)
        X_ = np.random.uniform(0,1,(m+n+2,RES))
        X = np.zeros((RES*(m+n+2), 2))
        for i in range(m+1):
            X[i*RES:(i+1)*RES,0] = X_[i,:]
            X[i*RES:(i+1)*RES,1] = i/m
        for i in range(n+1):
            X[(i+m+1)*RES:(i+1+m+1)*RES,1] = X_[i+m+1,:]
            X[(i+m+1)*RES:(i+1+m+1)*RES,0] = i/n
        return X, X, None
    
    def rose(self, ang_freq_n=2, ang_freq_d=1, radius=1, RES=1000, noise=0, n_cycles=None):
        ang_freq = ang_freq_n/ang_freq_d
        if np.mod(ang_freq_n, 2):
            n_petals = ang_freq_n
            n_cycles_ = ang_freq_d
        else:
            n_petals = 2*ang_freq_n
            n_cycles_ = 2*ang_freq_d
        if n_cycles is None:
            n_cycles = n_cycles_
        theta = np.linspace(0, n_cycles*np.pi, RES)[:-1]
        xv = radius*(np.cos(ang_freq*theta)*np.cos(theta))[:,None]
        yv = radius*(np.cos(ang_freq*theta)*np.sin(theta))[:,None]
        X = np.concatenate([xv,yv], axis=1)
        labelsMat = theta[:,None]
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def closedcurve_1(self, n=1000, noise=0):
        t = 2.01*np.pi*np.random.uniform(0,1,n)
        x = np.cos(t)
        y = np.sin(2*t)
        z = np.sin(3*t)
        X = np.vstack([x,y,z])
        X = np.transpose(X)
        X += noise*np.random.randn(X.shape[0],X.shape[1])
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, None

    def generate_random_curve(n_data=5000, n_freq=8, noise=0.1, noise_type='uniform', trunc_min=0, trunc_max=1, seed=42):
        theta = np.linspace(trunc_min,trunc_max,n_data)
        
        np.random.seed(seed)
        x_coeffs = np.random.normal(0, 1, (n_freq, 2))
        y_coeffs = np.random.normal(0, 1, (n_freq, 2))

        x = 0
        y = 0
        x_prime = 0
        y_prime = 0
        for i in range(n_freq):
            x = x + x_coeffs[i,0]*np.sin(2*np.pi*theta*i) + x_coeffs[i,1]*np.cos(2*np.pi*theta*i)
            y = y + y_coeffs[i,0]*np.sin(2*np.pi*theta*i) + y_coeffs[i,1]*np.cos(2*np.pi*theta*i)
        
            x_prime = x_prime + 2*np.pi*i*(x_coeffs[i,0]*np.cos(2*np.pi*theta*i) - x_coeffs[i,1]*np.sin(2*np.pi*theta*i))
            y_prime = y_prime + 2*np.pi*i*(y_coeffs[i,0]*np.cos(2*np.pi*theta*i) - y_coeffs[i,1]*np.sin(2*np.pi*theta*i))

        X0 = np.column_stack([x, y])
        N = np.column_stack([-y_prime,x_prime])
        N = N/np.linalg.norm(N, axis=1)[:,None]
        
        if noise:
            if noise_type == 'normal':
                X = X0 + N * noise*np.random.normal(0,1,n_data)[:,None]
            else:
                X = X0 + N * noise*np.random.uniform(-1,1,n_data)[:,None]

        return X, X0, theta
    
    def rectanglegrid(self, ar=16, RES=100, noise=0, noise_type='normal', seed=42, sideLx=None, sideLy=None):
        if sideLx is None:
            sideLx = np.sqrt(ar)
        if sideLy is None:
            sideLy = 1/sideLx
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)
        y = np.linspace(0, sideLy, RESy)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,None]
        yv = yv.flatten('F')[:,None]
        X = np.concatenate([xv,yv], axis=1)
        if noise:
            np.random.seed(seed)
            n = xv.shape[0]
            if noise_type == 'normal':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
            
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = np.zeros(n)
        for k in range(n):
            ddXx = np.min([X[k,0], sideLx-X[k,0]])
            ddXy = np.min([X[k,1], sideLy-X[k,1]])
            ddX[k] = np.min([ddXx, ddXy])
            
        return X, labelsMat, ddX
    
    def rectangle(self, ar=16, n=5000, noise=0, noise_type='normal', seed=42):
        sideLx = np.sqrt(ar)
        sideLy = 1/sideLx
        np.random.seed(seed)
        xv = np.random.uniform(0, sideLx, n)
        yv = np.random.uniform(0, sideLy, n)
        xv = xv.flatten('F')[:,None]
        yv = yv.flatten('F')[:,None]
        X = np.concatenate([xv,yv], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        if noise:
            n = xv.shape[0]
            if noise_type == 'normal':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        
        n = X.shape[0]
        ddX = np.zeros(n)
        for k in range(n):
            ddXx = np.min([X[k,0], sideLx-X[k,0]])
            ddXy = np.min([X[k,1], sideLy-X[k,1]])
            ddX[k] = np.min([ddXx, ddXy])
            
        return X, labelsMat, ddX
    
    def rectanglegrid_mog(self, ar=16, RES=10, n=100, sigma=0.015, noise=0, noise_type='uniform'):
        sideLx = np.sqrt(ar)
        sideLy = 1/sideLx
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)
        y = np.linspace(0, sideLy, RESy)
        xv_, yv_ = np.meshgrid(x, y)
        xv_ = xv_.flatten('F')
        yv_ = yv_.flatten('F')
        
        xv = []
        yv = []
        cluster_label = []
        np.random.seed(42)
        for i in range(xv_.shape[0]):
            X_r = np.random.multivariate_normal([xv_[i],yv_[i]], [[sigma,0],[0,sigma]], [n])
            #X_r[:,0] = np.clip(X_r[:,0], 0, sideLx)
            #X_r[:,1] = np.clip(X_r[:,1], 0, sideLy)
            xv += X_r[:,0].tolist()
            yv += X_r[:,1].tolist()
            cluster_label += [i]*n
        
        xv = np.array(xv)[:,None]
        yv = np.array(yv)[:,None]
        cluster_label = np.array(cluster_label)[:,None]
        
        X = np.concatenate([xv,yv], axis=1)
        if noise:
            np.random.seed(42)
            n = xv.shape[0]
            if noise_type == 'normal':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
            
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = np.zeros(n)
        for k in range(n):
            ddXx = np.min([X[k,0], sideLx-X[k,0]])
            ddXy = np.min([X[k,1], sideLy-X[k,1]])
            ddX[k] = np.min([ddXx, ddXy])
            
        return X, labelsMat, ddX
    
    def circular_disk(self, RES=100, noise=0, noise_type='uniform'):
        sideLx = 2
        sideLy = 2
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)-sideLx/2
        y = np.linspace(0, sideLy, RESy)-sideLy/2
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')
        yv = yv.flatten('F')
        mask = (xv**2 + yv**2) < 1
        xv = xv[mask][:,None]
        yv = yv[mask][:,None]
        X = np.concatenate([xv,yv], axis=1)
        if noise:
            np.random.seed(42)
            n = xv.shape[0]
            if noise_type == 'normal':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = 1-np.sqrt(X[:,0]**2+X[:,1]**2)
        return X, labelsMat, ddX
    
    def circular_disk_uniform(self, n=10000, noise=0, noise_type='uniform', seed=42):
        np.random.seed(seed)
        xv = np.random.uniform(-1, 1, n)
        yv = np.random.uniform(-1, 1, n)
        mask = (xv**2 + yv**2) < 1
        xv = xv[mask][:,None]
        yv = yv[mask][:,None]
        X = np.concatenate([xv,yv], axis=1)
        if noise:
            np.random.seed(42)
            n = xv.shape[0]
            if noise_type == 'gaussian':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X[:,-1] = X[:,-1] + noise*np.random.normal(0,1,(n))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = 1-np.sqrt(X[:,0]**2+X[:,1]**2)
        return X, labelsMat, ddX
    
    def annulus_non_uniform(self, n=1000, inner_r=0.3, seed=42, noise=0, noise_type='uniform'):
        np.random.seed(seed)
        sigma = 0.5*np.pi
        theta = np.mod(sigma*np.random.normal(0, 1, n), 2*np.pi)
        X = np.zeros((n,2))
        X[:,0] = np.cos(theta)
        X[:,1] = np.sin(theta)
        r = np.random.uniform(inner_r,1,n)
        ddX = np.minimum(1 - r, r - 0.3)
        X= X*r[:,None]
        q_true = np.exp(-(np.minimum(theta, 2*np.pi-theta)**2)/(2*sigma**2))/r
        labelsMat = X.copy()
        if noise:
            np.random.seed(42)
            if noise_type == 'gaussian':
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X[:,-1] = X[:,-1] + noise*np.random.normal(0,1,(n))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        return X, labelsMat, ddX, q_true 
    
    def annulus(self, n=1000, inner_r=0.3, outer_r=1, seed=42, noise=0, noise_type='uniform'):
        X = []
        labelsMat = []
        ddX = []
        cur_n = 0
        while cur_n < n:
            X_, labelsMat_, ddX_ = self.circular_disk_uniform(n=n, noise=0, noise_type=noise_type, seed=cur_n)
            r = np.linalg.norm(X_, axis=1)
            max_r = np.max(r)
            X_ = outer_r*X_/max_r
            r = outer_r*r/max_r

            mask = r > inner_r
            X_ = X_[mask,:]
            labelsMat_ = labelsMat_[mask,:]
            r = r[mask]
            ddX_ = np.minimum(outer_r - r, r - inner_r)
            X.append(X_)
            labelsMat.append(labelsMat_)
            ddX.append(ddX_)

            cur_n += X_.shape[0]

        X = np.concatenate(X, axis=0)[:n,:]
        labelsMat = np.concatenate(labelsMat, axis=0)[:n,:]
        ddX = np.concatenate(ddX)[:n]
        print(X.shape)

        if noise:
            np.random.seed(42)
            if noise_type == 'gaussian':
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X[:,-1] = X[:,-1] + noise*np.random.normal(0,1,(n))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        return X, labelsMat, ddX 
    
    def barbell(self, RES=100):
        A1 = 0.425
        Rmax = np.sqrt(A1/np.pi)
        sideL1x = 1.5
        sideL1y = (1-2*A1)/sideL1x

        sideLx = sideL1x+4*Rmax
        sideLy = 2*Rmax

        RESx = int(np.ceil(sideLx*RES)+1)
        RESy = int(np.ceil(sideLy*RES)+1)
        x1 = np.linspace(0,sideLx,RESx)
        y1 = np.linspace(0,sideLy,RESy)
        x1v, y1v = np.meshgrid(x1,y1);
        x1v = x1v.flatten('F')[:,None]
        y1v = y1v.flatten('F')[:,None]
        x2v = np.copy(x1v)
        y2v = np.copy(y1v)
        
        mask1 = (((x1v-Rmax)**2+(y1v-Rmax)**2) < Rmax**2)|(((x1v-3*Rmax-sideL1x)**2+(y1v-Rmax)**2)<Rmax**2)
        mask2 = (x2v>=(2*Rmax))&(x2v<=(2*Rmax+sideL1x))&(y2v>(Rmax-sideL1y/2))&(y2v<(Rmax+sideL1y/2))
        x1v = x1v[mask1][:,None]
        y1v = y1v[mask1][:,None]
        x2v = x2v[mask2][:,None]
        y2v = y2v[mask2][:,None]
        xv = np.concatenate([x1v,x2v],axis=0)
        yv = np.concatenate([y1v,y2v],axis=0)
        X = np.concatenate([xv,yv],axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        
        n = X.shape[0]
        ddX = np.zeros(n)
        for k in range(n):
            x_ = X[k,0]
            y_ = X[k,1]
            if (x_<=2*Rmax) or (x_>=(2*Rmax+sideL1x)):
                if x_>=(2*Rmax+sideL1x):
                    x_=x_-2*Rmax-sideL1x
                    x_=2*Rmax-x_
                if (x_>=Rmax) and (y_>=Rmax) and (y_<=Rmax+sideL1y*(x_-Rmax)/(2*Rmax)):
                    ddX[k]=np.sqrt((x_-2*Rmax)**2+(y_-(Rmax+sideL1y/2))**2)
                elif (x_>Rmax) and (y_<=Rmax) and (y_>=Rmax-sideL1y*(x_-Rmax)/(2*Rmax)):
                    ddX[k]=np.sqrt((x_-2*Rmax)**2+(y_-(Rmax-sideL1y/2))**2)
                else:
                    ddX[k]=Rmax-np.sqrt((x_-Rmax)**2+(y_-Rmax)**2)
            else:
                ddX[k]=np.min([y_-(Rmax-sideL1y/2),Rmax+sideL1y/2-y_])
        ddX[ddX<1e-2] = 0
        return X, labelsMat, ddX
    
    def squarewithtwoholesgrid(self, RES=100, noise=0, noise_type='normal', seed=42):
        sideLx = 1
        sideLy = 1
        RESx = sideLx*RES+1
        RESy = sideLy*RES+1
        x = np.linspace(0,sideLx,RESx);
        y = np.linspace(0,sideLy,RESy);
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,None]
        yv = yv.flatten('F')[:,None]
        X = np.concatenate([xv,yv], axis=1)
        hole1 = np.sqrt((X[:,0] - 0.5*np.sqrt(2))**2 + (X[:,1]-0.5*np.sqrt(2))**2) < 0.1*np.sqrt(2)
        hole2 = np.abs(X[:,0] - 0.2*np.sqrt(2)) + np.abs(X[:,1]-0.2*np.sqrt(2)) < 0.1*np.sqrt(2)
        
        Xhole1 = X[hole1,:]
        Xhole2 = X[hole2,:]
        ddX1 = np.min(cdist(X,Xhole1),axis=1)
        ddX1[ddX1<1e-2*1.2] = 0
        ddX2 = np.min(cdist(X,Xhole2),axis=1)
        ddX2[ddX2<1e-2*1.2] = 0
        ddXx = np.minimum(X[:,0],sideLx-X[:,0])
        ddXy = np.minimum(X[:,1],sideLy-X[:,1])
        ddX = np.minimum(ddXx,ddXy)
        ddX = np.minimum(ddX,ddX1)
        ddX = np.minimum(ddX,ddX2)
        
        X = X[~hole1 & ~hole2,:]
        ddX = ddX[~hole1 & ~hole2]
        n=X.shape[0]
        if noise:
            np.random.seed(seed)
            if noise_type == 'normal':
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def squarewithtwoholes(self, n=5000, noise=0, noise_type='normal', seed=42):
        sideLx = 1
        sideLy = 1
        np.random.seed(seed)
        xv = np.random.uniform(0,sideLx,n)
        yv = np.random.uniform(0,sideLy,n)
        xv = xv.flatten('F')[:,None]
        yv = yv.flatten('F')[:,None]
        X = np.concatenate([xv,yv], axis=1)
        hole1 = np.sqrt((X[:,0] - 0.5*np.sqrt(2))**2 + (X[:,1]-0.5*np.sqrt(2))**2) < 0.1*np.sqrt(2)
        hole2 = np.abs(X[:,0] - 0.2*np.sqrt(2)) + np.abs(X[:,1]-0.2*np.sqrt(2)) < 0.1*np.sqrt(2)
        
        Xhole1 = X[hole1,:]
        Xhole2 = X[hole2,:]
        ddX1 = np.min(cdist(X,Xhole1),axis=1)
        ddX1[ddX1<1e-2*1.2] = 0
        ddX2 = np.min(cdist(X,Xhole2),axis=1)
        ddX2[ddX2<1e-2*1.2] = 0
        ddXx = np.minimum(X[:,0],sideLx-X[:,0])
        ddXy = np.minimum(X[:,1],sideLy-X[:,1])
        ddX = np.minimum(ddXx,ddXy)
        ddX = np.minimum(ddX,ddX1)
        ddX = np.minimum(ddX,ddX2)
        
        X = X[~hole1 & ~hole2,:]
        ddX = ddX[~hole1 & ~hole2]
        n=X.shape[0]
        if noise:
            np.random.seed(seed)
            if noise_type == 'normal':
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)
        
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def spherewithhole(self, n=10000, hole_frac=1/6):
        Rmax = np.sqrt(1/(4*np.pi))
        indices = np.arange(n)
        indices = indices+0.5
        phiv = np.arccos(1 - 2*indices/n)[:,None]
        thetav = (np.pi*(1 + np.sqrt(5))*indices)[:,None]
        X = np.concatenate([np.sin(phiv)*np.cos(thetav), np.sin(phiv)*np.sin(thetav), np.cos(phiv)], axis=1)
        X = X*Rmax
        z0 = np.max(X[:,2])
        R_hole = hole_frac*Rmax
        hole = (X[:,0]**2+X[:,1]**2+(X[:,2]-z0)**2)<R_hole**2
        
        Xhole = X[hole,:]
        ddX = np.min(cdist(X,Xhole), axis=1)
        ddX[ddX<1e-2*1.2] = 0
        
        X = X[~hole,:]
        ddX = ddX[~hole]
        thetav = thetav[~hole][:,None]
        phiv = phiv[~hole][:,None]
        labelsMat = np.concatenate([np.mod(thetav,2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def swissrollwithhole(self, n=5000, seed=42):
        theta0 = 3*np.pi/2
        nturns = 2
        rmax = 2*1e-2
        sideL1 = integrate.quad(lambda x: rmax*np.sqrt(1+x**2), theta0, theta0*(1+nturns))[0]
        sideL2 = 1/sideL1
        np.random.seed(seed)
        tdistv = np.random.uniform(0, sideL1, n)
        tv = []
        for tdist in tdistv.tolist():
            tt = fsolve(lambda x: (0.5*rmax*(x*np.sqrt(1+x**2)+np.arcsinh(x)))-\
                                   0.5*rmax*(theta0*np.sqrt(1+theta0**2)+np.arcsinh(theta0))-\
                                   tdist,theta0*(1+nturns/2))
            tv.append(tt)
        tv = np.array(tv)    
        heightv = np.random.uniform(0,sideL2,n)[:,None]
        heightv = heightv.flatten('F')[:,None]
        tv = tv[:,None]
        X=np.concatenate([rmax*tv*np.cos(tv), heightv, rmax*tv*np.sin(tv)], axis=1)
        
        ddX11 = np.minimum(heightv, sideL2-heightv).flatten()
        ddX12 = np.tile(tdistv[:,None], RESh).flatten()
        ddX12 = np.minimum(ddX12, sideL1-ddX12)
        ddX1 = np.minimum(ddX11, ddX12)

        y_mid = sideL2*0.5
        t_min = np.min(tv)
        t_max = np.max(tv)
        t_range = t_max-t_min
        t_mid = t_min + t_range/2
        x_mid = rmax*t_mid*np.cos(t_mid)
        z_mid = rmax*t_mid*np.sin(t_mid)
        hole = np.sqrt((X[:,0]-x_mid)**2+(X[:,1]-y_mid)**2+(X[:,2]-z_mid)**2)<0.1

        Xhole = X[hole,:]
        ddX2 = np.min(cdist(X,Xhole), axis=1)
        ddX2[ddX2<1e-2*1.2] = 0
        
        X = X[~hole,:]
        ddX = np.minimum(ddX1[~hole], ddX2[~hole])
        tv = tv[~hole]
        labelsMat = np.concatenate([tv, X[:,[1]]], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def swissrollwithholegrid(self, RES=100, theta0 = 3*np.pi/2, nturns = 2):
        rmax = 2*1e-2
        sideL1 = integrate.quad(lambda x: rmax*np.sqrt(1+x**2), theta0, theta0*(1+nturns))[0]
        sideL2 = 1/sideL1
        RESt = int(np.ceil(sideL1*RES+1))
        tdistv = np.linspace(0,sideL1,RESt)
        tv = []
        for tdist in tdistv.tolist():
            tt = fsolve(lambda x: (0.5*rmax*(x*np.sqrt(1+x**2)+np.arcsinh(x)))-\
                                   0.5*rmax*(theta0*np.sqrt(1+theta0**2)+np.arcsinh(theta0))-\
                                   tdist,theta0*(1+nturns/2))
            tv.append(tt)
        tv = np.array(tv)    
        RESh = int(np.ceil(sideL2*RES+1))
        heightv = np.linspace(0,sideL2,RESh)[:,None]
        heightv = np.tile(heightv,[RESt,1])
        heightv = heightv.flatten('F')[:,None]
        tv = np.repeat(tv,RESh)[:,None]
        X=np.concatenate([rmax*tv*np.cos(tv), heightv, rmax*tv*np.sin(tv)], axis=1)
        
        ddX11 = np.minimum(heightv, sideL2-heightv).flatten()
        ddX12 = np.tile(tdistv[:,None], RESh).flatten()
        ddX12 = np.minimum(ddX12, sideL1-ddX12)
        ddX1 = np.minimum(ddX11, ddX12)

        y_mid = sideL2*0.5
        t_min = np.min(tv)
        t_max = np.max(tv)
        t_range = t_max-t_min
        t_mid = t_min + t_range/2
        x_mid = rmax*t_mid*np.cos(t_mid)
        z_mid = rmax*t_mid*np.sin(t_mid)
        hole = np.sqrt((X[:,0]-x_mid)**2+(X[:,1]-y_mid)**2+(X[:,2]-z_mid)**2)<0.1

        Xhole = X[hole,:]
        ddX2 = np.min(cdist(X,Xhole), axis=1)
        ddX2[ddX2<1e-2*1.2] = 0
        
        X = X[~hole,:]
        ddX = np.minimum(ddX1[~hole], ddX2[~hole])
        tv = tv[~hole]
        labelsMat = np.concatenate([tv, X[:,[1]]], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, ddX
    
    def noisyswissroll(self, RES=100, noise=0.01, noise_type = 'ortho',
                       theta0=3*np.pi/2, nturns=2, rmax=2*1e-2, seed=42):
        sideL1 = integrate.quad(lambda x: rmax*np.sqrt(1+x**2), theta0, theta0*(1+nturns))[0]
        sideL2 = 1/sideL1
        RESt = int(np.ceil(sideL1*RES+1))
        tdistv = np.linspace(0,sideL1,RESt)
        tv = []
        for tdist in tdistv.tolist():
            tt = fsolve(lambda x: (0.5*rmax*(x*np.sqrt(1+x**2)+np.arcsinh(x)))-\
                                   0.5*rmax*(theta0*np.sqrt(1+theta0**2)+np.arcsinh(theta0))-\
                                   tdist,theta0*(1+nturns/2))
            tv.append(tt)
        tv = np.array(tv)    
        RESh = int(np.ceil(sideL2*RES+1))
        heightv = np.linspace(0,sideL2,RESh)[:,None]
        heightv = np.tile(heightv,[RESt,1])
        heightv = heightv.flatten('F')[:,None]
        tv = np.repeat(tv,RESh)[:,None]
        X=np.concatenate([rmax*tv*np.cos(tv), heightv, rmax*tv*np.sin(tv)], axis=1)
        np.random.seed(seed)
        if noise_type == 'normal':
            X = X+noise*np.random.normal(0,1,[X.shape[0],3])
        elif noise_type == 'uniform':
            X = X+noise*np.random.uniform(0,1,[X.shape[0],3])
        elif 'ortho' in noise_type:
            # the swiss roll rolls around y axis
            temp = X.copy()
            temp[:,1] = 0
            temp = temp/np.linalg.norm(temp, axis=1)[:,None]
            if noise_type == 'ortho-uniform':
                X = X + noise*np.random.uniform(-1,1,(X.shape[0],1))*temp
            else:
                X = X + noise*np.random.normal(0,1,(X.shape[0],1))*temp
        labelsMat = np.concatenate([tv, X[:,[1]]], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
        
    def sphere(self, n=10000, noise = 0, seed=42, R=None):
        if R is None:
            R = np.sqrt(1/(4*np.pi))
        indices = np.arange(n)+0.5
        phiv = np.arccos(1 - 2*indices/n)
        phiv = phiv[:,None]
        thetav = np.pi*(1 + np.sqrt(5))*indices
        thetav = thetav[:,None]
        X = np.concatenate([np.sin(phiv)*np.cos(thetav),
                            np.sin(phiv)*np.sin(thetav),
                            np.cos(phiv)], axis=1)
        X = X*R;
        np.random.seed(seed)
        X = X*(1+noise*np.random.uniform(-1,1,(X.shape[0],1)))
        labelsMat = np.concatenate([np.mod(thetav,2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def RP2(self, n=10000, noise = 0):
        X_, labelsMat, _ = self.sphere(n, noise)
        mask = (X_[:,2] > 0) | ((X_[:,0] > 0) & (X_[:,2]==0))
        X_ = X_[mask,:]
        labelsMat = labelsMat[mask,:]
        X = np.zeros((X_.shape[0],4))
        X[:,0] = X_[:,0]*X_[:,1]
        X[:,1] = X_[:,0]*X_[:,2]
        X[:,2] = X_[:,1]**2 - X_[:,2]**2
        X[:,3] = 2*X_[:,1]*X_[:,2]
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def RP1(self, RES=100, noise = 0):
        X_, labelsMat, _ = self.circle(RES, noise)
        mask = (X_[:,1] > 0) | ((X_[:,0] > 0) & (X_[:,1]==0))
        X_ = X_[mask,:]
        labelsMat = labelsMat[mask,:]
        X = np.zeros((X_.shape[0],2))
        X[:,0] = X_[:,0]*X_[:,1]
        X[:,1] = X_[:,0]**2 - X_[:,1]**2
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def wave_on_circle(self, RES=25, R_in=2, R_out=3, r=1/2, f=8):
        sideLx = int(np.ceil(2*(R_out+r)))
        sideLy = int(np.ceil(2*(R_out+r)))
        RESx = sideLx*RES+1
        RESy = sideLy*RES+1
        x = np.linspace(-sideLx/2,sideLx/2,RESx);
        y = np.linspace(-sideLy/2,sideLy/2,RESy);
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')
        yv = yv.flatten('F')
        
        theta = np.arctan2(yv, xv)
        radius = np.sqrt(xv**2 + yv**2)
        R_out_at_theta = R_out+r*np.sin(f*theta)
        R_in_at_theta = R_in+r*np.sin(f*theta)
        mask = (radius <= R_out_at_theta) & (radius >= R_in_at_theta)
        
        xv = xv[mask][:,None]
        yv = yv[mask][:,None]
        X = np.concatenate([xv,yv], axis=1)
        
        theta = theta[mask][:,None]
        radius = radius[mask][:,None]
        
        labelsMat = np.concatenate([radius, theta], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def sphere3d(self, n=10000, noise = 0, seed=42):
        R = np.power(1/(4*np.pi), 0.5)
        np.random.seed(seed)
        X = np.random.normal(0,1,(n,3))
        X = X/np.linalg.norm(X,axis=1)[:,None]
        X = X*R;
        X = X*(1+noise*np.random.uniform(-1,1,(X.shape[0],1)))
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def sphere3(self, n=10000, noise = 0, seed=42):
        R = np.power(2/(np.pi**2), 0.25)
        np.random.seed(seed)
        X = np.random.normal(0,1,(n,4))
        X = X/np.linalg.norm(X,axis=1)[:,None]
        X = X*R;
        np.random.seed(seed)
        X = X*(1+noise*np.random.uniform(-1,1,(X.shape[0],1)))
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def sphere_and_swissroll(self, n=5000, RES=70, noise1 = 0.01, noise2=0.015, sep=1):
        s1, l1, _ = self.sphere(n, noise=noise1)
        s2, l2, _ = self.noisyswissroll(RES=RES, noise=noise2)
        x_max = np.max(s1[:,0])
        x_min = np.min(s2[:,0])
        s2 = s2 + np.array([x_max-x_min,0,0]).reshape((1,3)) + sep
        X = np.concatenate([s1, s2], axis=0)
        labelsMat = np.concatenate([l1, l2], axis=0)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def cuboid_and_swissroll(self, n=5000, RES=70, noise1 = 0.01, noise2=0.015, sep=0):
        s1, l1, _ = self.solidcuboid3d(RES=15)
        s2, l2, _ = self.noisyswissroll(RES=RES, noise=0)
        x_max = np.max(s1[:,0])
        x_min = np.min(s2[:,0])
        s2 = s2 + np.array([x_max-x_min,0,0]).reshape((1,3)) + sep
        X = np.concatenate([s1, s2], axis=0)
        labelsMat = np.concatenate([l1[:,:2], l2], axis=0)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def multi_spheres(self, m=3, n=3000, noise = 0, sep=1):
        X = []
        labelsMat = []
        offset = 0
        for i in range(m):
            s1, l1, _ = self.sphere(n, noise)
            if i > 0:
                s1[:,0] += 1.25*offset - np.min(s1[:,0])
            offset = np.max(s1[:,0])
            X.append(s1)
            labelsMat.append(l1)
            
        
        X = np.concatenate(X, axis=0)
        labelsMat = np.concatenate(labelsMat, axis=0)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def sphere_mog(self, k=10, n=1000, sigma=0.1, noise = 0):
        R = np.sqrt(1/(4*np.pi))
        indices = np.arange(k)+0.5
        phiv_ = np.arccos(1 - 2*indices/k)
        thetav_ = np.mod(np.pi*(1 + np.sqrt(5))*indices, 2*np.pi)
        
        phiv = []
        thetav = []
        cluster_label = []
        np.random.seed(42)
        for i in range(k):
            X_r = np.random.multivariate_normal([phiv_[i],thetav_[i]], [[sigma,0],[0,sigma]], [n])
            phiv += X_r[:,0].tolist()
            thetav += X_r[:,1].tolist()
            cluster_label += [i]*n
        
        phiv = np.array(phiv)[:,None]
        thetav = np.array(thetav)[:,None]
        cluster_label = np.array(cluster_label)[:,None]
        
        X = np.concatenate([np.sin(phiv)*np.cos(thetav),
                            np.sin(phiv)*np.sin(thetav),
                            np.cos(phiv)], axis=1)
        X = X*R;
        np.random.seed(2)
        X = X*(1+noise*np.random.uniform(-1,1,(X.shape[0],1)))
        labelsMat = np.concatenate([cluster_label,np.mod(thetav,2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def spherewithanomaly(self, n=10000, epsilon=0.05, noise=0.05):
        R = np.sqrt(1/(4*np.pi))
        indices = np.arange(n)+0.5
        phiv = np.arccos(1 - 2*indices/n)
        phiv = phiv[:,None]
        thetav = np.pi*(1 + np.sqrt(5))*indices
        thetav = thetav[:,None]
        X = np.concatenate([np.sin(phiv)*np.cos(thetav),
                            np.sin(phiv)*np.sin(thetav),
                            np.cos(phiv)], axis=1)
        X = X*R;
        np.random.seed(2)
        X = X*(1+noise*np.random.uniform(-1,1,(X.shape[0],1)))
        labelsMat = np.concatenate([np.mod(thetav,2*np.pi), phiv], axis=1)
        
        # add anomaly at north pole
        k = np.argmin(np.abs(phiv))
        d_k_kp = np.sqrt(np.sum((X - X[k,:][None,:])**2, axis=1))
        mask = (d_k_kp < epsilon)
        n_ = np.sum(mask)
        np.random.seed(42)
        X[mask,2:3] = X[mask,2:3] + np.random.normal(0,1,(n_,1))*noise
        
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def flattorus_3d_in_6d(self, radii=None, RES=25):
        if radii is None:
            radii = [0.1, 0.1, 0.1]

        rx, ry, rz = radii
        lx = 2*np.pi*rx
        ly = 2*np.pi*ry
        lz = 2*np.pi*rz
        RESx=int(lx*RES+1)
        RESy=int(ly*RES+1)
        RESz=int(lz*RES+1)
        
        x=np.linspace(0,2*np.pi,RESx)[:-1] # remove 2pi
        y=np.linspace(0,2*np.pi,RESy)[:-1] # remove 2pi
        z=np.linspace(0,2*np.pi,RESz)[:-1] # remove 2pi
        xv, yv, zv = np.meshgrid(x, y, z)
        xv = xv.flatten('F')[:,None]
        yv = yv.flatten('F')[:,None]
        zv = zv.flatten('F')[:,None]
        X=np.concatenate([rx*np.cos(xv), rx*np.sin(xv), ry*np.cos(yv), ry*np.sin(yv), rz*np.cos(zv), rz*np.sin(zv)], axis=1)
        labelsMat = np.concatenate([xv, yv, zv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None

    def flattorus4d(self, ar=4, RES=100):
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rout = sideLx/(2*np.pi)
        Rin = sideLy/(2*np.pi)
        RESx=int(sideLx*RES+1)
        RESy=int(sideLy*RES+1)
        x=np.linspace(0,sideLx,RESx)[:-1] # remove 2pi
        y=np.linspace(0,sideLy,RESy)[:-1] # remove 2pi
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,None]/Rout
        yv = yv.flatten('F')[:,None]/Rin
        X=np.concatenate([Rout*np.cos(xv), Rout*np.sin(xv), Rin*np.cos(yv), Rin*np.sin(yv)], axis=1)
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
#     def S1xS1xS1(self, n=10000, seed=42):
#         np.random.seed(42)
#         xv=np.linspace(0,2*np.pi,n)[:-1][:,None] # remove 2pi
#         yv=np.linspace(0,2*np.pi,n)[:-1][:,None] # remove 2pi
#         zv=np.linspace(0,2*np.pi,n)[:-1][:,None] # remove 2pi
        
#         X=np.concatenate([np.cos(xv), np.sin(xv), np.cos(yv),
#                           np.sin(yv), np.cos(zv), np.sin(zv)], axis=1)
#         labelsMat = np.concatenate([xv, yv, zv], axis=1)
#         print('X.shape = ', X.shape)
#         return X, labelsMat, None
    
    def T3(self, n=10000, noise=0, Rmax=0.25, seed=42):
        rmax=1/(4*(np.pi**2)*Rmax);
        X = []
        thetav = []
        phiv = []
        np.random.seed(seed)
        k = 0
        while k < n:
            rU = np.random.uniform(0,1,3)
            theta = 2*np.pi*rU[0]
            phi = 2*np.pi*rU[1]
            if rU[2] <= (Rmax + rmax*np.cos(theta))/(Rmax + rmax):
                thetav.append(theta)
                phiv.append(phi)
                k = k + 1
        
        thetav = np.array(thetav)[:,None]
        phiv = np.array(phiv)[:,None]
        psiv = np.random.uniform(0, 2*np.pi, (n,1))
        np.random.seed(42)
        noise = noise*np.random.uniform(-1,1,(phiv.shape[0],1))
        X = np.concatenate([(Rmax+(1+noise)*rmax*np.cos(thetav))*np.cos(phiv),
                             (Rmax+(1+noise)*rmax*np.cos(thetav))*np.sin(phiv),
                             (Rmax+(1+noise)*rmax*np.sin(thetav))*np.cos(psiv),
                             (Rmax+(1+noise)*rmax*np.sin(thetav))*np.sin(psiv)], axis=1)
        labelsMat = np.concatenate([thetav, phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def wrinkled_sheet(
            self,
            length = 4,         # Length of the sheet in x-direction
            width = 2,          # Width of the sheet in y-direction
            amplitude = 0.75,     # Amplitude of the wrinkles
            freq_x = 4,          # Number of wrinkles along x
            freq_y = 2,          # Number of wrinkles along y
            n = 10000,
            noise = 0,
            noise_type='uniform',
            noise_freq_x = 3,
            noise_freq_y = 3,
            seed=42
        ):

        def wrinkle_height(x, y):
            z = amplitude * np.sin(2 * np.pi * freq_x * x / length) * np.sin(2 * np.pi * freq_y * y / width)
            return z
        
        def dz_dx_(x, y):
            return amplitude * (2 * np.pi * freq_x / length) * np.cos(2 * np.pi * freq_x * x / length) * np.sin(2 * np.pi * freq_y * y / width)

        def dz_dy_(x, y):
            return amplitude * (2 * np.pi * freq_y / width) * np.sin(2 * np.pi * freq_x * x / length) * np.cos(2 * np.pi * freq_y * y / width)

        def surface_area_density(x, y):
            dz_dx = dz_dx_(x,y)
            dz_dy = dz_dy_(x,y)
            return np.sqrt(1 + dz_dx**2 + dz_dy**2)

        # Rejection sampling
        X = []
        max_density = 1 + (2 * np.pi * max(freq_x / length, freq_y / width) * amplitude)**2  # rough upper bound

        while len(X) < n:
            x_rand = np.random.uniform(0, length, n)
            y_rand = np.random.uniform(0, width, n)
            z_rand = wrinkle_height(x_rand, y_rand)

            area_density = surface_area_density(x_rand, y_rand)
            u = np.random.uniform(0, max_density, n)
            mask = u < area_density

            new_points = np.stack([x_rand[mask], y_rand[mask], z_rand[mask]], axis=1)
            if len(X) == 0:
                X = new_points[:n,:]
            else:
                X = np.concatenate([X, new_points], axis=0)[:n,:]

        ddX = np.zeros(X.shape[0])
        for i in range(len(ddX)):
            ddX[i] = min(min(X[i,0], length-X[i,0]), min(X[i,1], width-X[i,1]))
        labelsMat = X[:,:2]

        X_noisy = X.copy()
        if noise:
            np.random.seed(seed)
            if noise_type=='uniform':
                noise = noise*np.random.uniform(-1, 1, X.shape[0])
            elif noise_type=='non-uniform':
                noise_tube = np.sin(2 * np.pi * noise_freq_x * X[:,0] / length) * np.sin(2 * np.pi * noise_freq_y * X[:,1] / width)
                noise = noise*np.random.uniform(0, noise_tube, X.shape[0])
            X_noisy[:,2] = X[:,2] + noise

        X_theta = np.column_stack([np.ones(n), np.zeros(n), dz_dx_(X[:,0], X[:,1])])
        X_phi = np.column_stack([np.zeros(n), np.ones(n), dz_dy_(X[:,0], X[:,1])])
        return X_noisy, labelsMat, ddX, X, X_theta, X_phi

    
    def curvedtorus3d(self, n=10000, noise=0, noise_type='uniform',
                       Rmax=0.25, seed=42, freq=4, density='uniform', rmax=None):
        if rmax is None:
            rmax=1/(4*(np.pi**2)*Rmax)
        X = []
        thetav = []
        phiv = []
        np.random.seed(seed)
        k = 0
        sigma = 0.75*np.pi
        while k < n:
            rU = np.random.uniform(0,1,3)
            if density != 'uniform':
                #theta = np.mod(sigma*np.random.normal(0, 1), 2*np.pi)
                theta = truncnorm.rvs(a=-np.pi/sigma,b=np.pi/sigma,scale=sigma)
            else:
                theta = 2*np.pi*rU[0]
            phi = 2*np.pi*rU[1]
            
            if rU[2] <= (Rmax + rmax*np.cos(theta))/(Rmax + rmax):
                thetav.append(theta)
                phiv.append(phi)
                k = k + 1
        
        thetav = np.array(thetav)[:,None]
        phiv = np.array(phiv)[:,None]

        if density != 'uniform':
            #q_true = np.exp(-(np.minimum(thetav, 2*np.pi-thetav)**2)/(2*sigma**2))
            q_true = truncnorm.pdf(thetav, a=-np.pi/sigma,b=np.pi/sigma,scale=sigma)
            #q_true = q_true/(Rmax+rmax*np.cos(thetav))
        else:
            q_true = 1
        dX = None

        np.random.seed(42)
        if noise_type == 'uniform':
            noise = noise*np.random.uniform(-1,1,(phiv.shape[0],1))
        elif noise_type == 'gaussian':
            noise = noise*np.random.normal(0,1,(phiv.shape[0],1))
        else:
            noise_u = 0.01 + 0.3*(1+np.cos(freq*phiv))/2
            noise_u = np.random.uniform(-noise_u,noise_u)
            noise = noise*noise_u
        X = np.concatenate([(Rmax+(1+noise)*rmax*np.cos(thetav))*np.cos(phiv),
                             (Rmax+(1+noise)*rmax*np.cos(thetav))*np.sin(phiv),
                             (1+noise)*rmax*np.sin(thetav)], axis=1)
        labelsMat = np.concatenate([np.mod(thetav, 2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, dX, q_true
    
    def curvedtorus3d_with_normal_dir(self, n=10000, density='uniform', seed=42):
        X_noisy, X, labelsMat, dX, (X_theta, X_phi, normal_dir) = self.wave_on_curvedtorus3d(
            n=n, noise=0, noise_type='ortho',
            seed=seed, density=density,
            wave_amp_r=0, wave_freq_r=0, wave_amp_R=0,
            wave_freq_R=0, rmax=None)
        return X, labelsMat, normal_dir


    def wave_on_curvedtorus3d(self, n=10000, noise=0, noise_type='ortho',
                       Rmax=0.25, seed=42, freq=4, density='uniform',
                       wave_amp_r=0.2, wave_freq_r=5, wave_amp_R=0.1, wave_freq_R=3, rmax=None):
        if rmax is None:
            rmax=1/(4*(np.pi**2)*Rmax)

        theta = np.pi
        phi = (3*np.pi)/(2*wave_freq_r + 1e-12)
        r_ = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phi)
        R_ = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * theta)

        r_prime_ = wave_freq_r * wave_amp_r * np.cos(wave_freq_r * phi)
        R_prime_ = wave_freq_R * wave_amp_R * np.cos(wave_freq_R * theta)
        
        X_theta_ = np.array([(R_prime_ - r_*np.sin(theta))*np.cos(phi),
                                (R_prime_ - r_*np.sin(theta))*np.sin(phi),
                                r_*np.cos(theta)])
        X_phi_ = np.array([r_prime_*np.cos(theta)*np.cos(phi) - (R_+r_*np.cos(theta))*np.sin(phi),
                            r_prime_*np.cos(theta)*np.sin(phi) + (R_+r_*np.cos(theta))*np.cos(phi),
                            r_prime_*np.sin(theta)])
        X_theta_cross_X_phi_max = np.linalg.norm(np.cross(X_theta_, X_phi_))

        X = []
        thetav = []
        phiv = []
        np.random.seed(seed)
        k = 0
        sigma = 0.75*np.pi
        while k < n:
            rU = np.random.uniform(0,1,3)
            if density != 'uniform':
                #theta = np.mod(sigma*np.random.normal(0, 1), 2*np.pi)
                theta = truncnorm.rvs(a=-np.pi/sigma,b=np.pi/sigma,scale=sigma)
            else:
                theta = 2*np.pi*rU[0]
            phi = 2*np.pi*rU[1]

            r_ = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phi)
            R_ = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * theta)

            r_prime_ = wave_freq_r * wave_amp_r * np.cos(wave_freq_r * phi)
            R_prime_ = wave_freq_R * wave_amp_R * np.cos(wave_freq_R * theta)
            
            X_theta_ = np.array([(R_prime_ - r_*np.sin(theta))*np.cos(phi),
                                 (R_prime_ - r_*np.sin(theta))*np.sin(phi),
                                  r_*np.cos(theta)])
            X_phi_ = np.array([r_prime_*np.cos(theta)*np.cos(phi) - (R_+r_*np.cos(theta))*np.sin(phi),
                               r_prime_*np.cos(theta)*np.sin(phi) + (R_+r_*np.cos(theta))*np.cos(phi),
                               r_prime_*np.sin(theta)])
            X_theta_cross_X_phi = np.linalg.norm(np.cross(X_theta_, X_phi_))
            X_theta_cross_X_phi = X_theta_cross_X_phi/X_theta_cross_X_phi_max

            #if rU[2] <= (Rmax + rmax*np.cos(theta))/(Rmax + rmax):
            #if rU[2] <= (R_ + r_*np.cos(theta))/(R_ + r_):
            if rU[2] <= X_theta_cross_X_phi:
                thetav.append(theta)
                phiv.append(phi)
                k = k + 1
        
        thetav = np.array(thetav)[:,None]
        phiv = np.array(phiv)[:,None]
        dX = None

        r = rmax + wave_amp_r * rmax * np.sin(wave_freq_r * phiv)
        R = Rmax + wave_amp_R * Rmax * np.sin(wave_freq_R * thetav)
        X = np.concatenate([(R+r*np.cos(thetav))*np.cos(phiv),
                             (R+r*np.cos(thetav))*np.sin(phiv),
                              r*np.sin(thetav)], axis=1)
        
        np.random.seed(42)
        if 'uniform' in noise_type:
            noise = noise*np.random.uniform(-1,1,(X.shape[0],1))
        elif 'gaussian' in noise_type:
            noise = noise*np.random.normal(0,1,(X.shape[0],1))
        else:
            #noise_u = 0.01 + 0.3*(1+np.cos(freq*phiv))/2
            noise_u = np.cos(freq*phiv)**2
            noise_u = np.random.uniform(-noise_u,noise_u)
            noise = noise*noise_u

        r_prime = wave_freq_r * wave_amp_r * rmax * np.cos(wave_freq_r * phiv)
        R_prime = wave_freq_R * wave_amp_R * Rmax * np.cos(wave_freq_R * thetav)
        X_theta = np.concatenate([(R_prime - r*np.sin(thetav))*np.cos(phiv),
                                (R_prime - r*np.sin(thetav))*np.sin(phiv),
                                   r*np.cos(thetav)], axis=1)
        X_phi = np.concatenate([r_prime*np.cos(thetav)*np.cos(phiv) - (R+r*np.cos(thetav))*np.sin(phiv),
                                r_prime*np.cos(thetav)*np.sin(phiv) + (R+r*np.cos(thetav))*np.cos(phiv),
                                r_prime*np.sin(thetav)], axis=1)
        normal_dir = np.cross(X_theta, X_phi)
        normal_dir = normal_dir/np.linalg.norm(normal_dir, axis=1)[:,None]
        X_noisy = X + noise * normal_dir

        X_theta = X_theta/np.linalg.norm(X_theta,axis=1)[:,None]
        X_phi = X_phi/np.linalg.norm(X_phi,axis=1)[:,None]

        labelsMat = np.concatenate([np.mod(thetav, 2*np.pi), phiv], axis=1)
        print('X.shape = ', X.shape)
        return X_noisy, X, labelsMat, dX, (X_theta, X_phi, normal_dir)
    
    def curvedtorus3d_grid_old(self, RES=50, noise=0, Rmax=0.25):
        rmax=1/(4*(np.pi**2)*Rmax)
        
        sideLx = 2*np.pi
        sideLy = 2*np.pi
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)[:-1] # remove 2pi
        y = np.linspace(0, sideLy, RESy)[:-1] # remove 2pi
        xv, yv = np.meshgrid(x, y)
        thetav = xv.flatten('F')[:,None]
        phiv = yv.flatten('F')[:,None]
        noise = noise*np.random.uniform(-1,1,(phiv.shape[0],1))
        X = np.concatenate([(Rmax+(1+noise)*rmax*np.cos(thetav))*np.cos(phiv),
                             (Rmax+(1+noise)*rmax*np.cos(thetav))*np.sin(phiv),
                             (1+noise)*rmax*np.sin(thetav)], axis=1)
        labelsMat = np.concatenate([thetav, phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def curvedtorus3d_grid(self, RES=50, noise=0, Rmax=0.25, rmax=None):
        if rmax is None:
            rmax=1/(4*(np.pi**2)*Rmax)
        
        ds = 2*np.pi*(Rmax-rmax)/RES
        #RES2 = 2*np.pi*rmax/ds
        RES2 = (RES*rmax)/(Rmax-rmax)
        print('rmax:', rmax)
        print('ds:', ds)
        print('RES2:', RES2)
        RES2 = int(np.round(RES2))
        print('RES2:', RES2)
        theta = np.linspace(0, 2*np.pi, RES2+1)[:-1]
        phiv = []
        thetav = []
        for i in range(RES2):
            RES3 = int(np.round((RES*(Rmax+rmax*np.cos(theta[i])))/(Rmax-rmax)))
            phi = np.linspace(0, 2*np.pi, RES3+1)[:-1]
            thetav += [theta[i]]*RES3
            phiv += phi.tolist()
        
        thetav = np.array(thetav)[:,None]
        phiv = np.array(phiv)[:,None]
        labelsMat = np.concatenate([thetav, phiv], axis=1)
            
        
        noise = noise*np.random.uniform(-1,1,(phiv.shape[0],1))
        X = np.concatenate([(Rmax+(1+noise)*rmax*np.cos(thetav))*np.cos(phiv),
                             (Rmax+(1+noise)*rmax*np.cos(thetav))*np.sin(phiv),
                             (1+noise)*rmax*np.sin(thetav)], axis=1)
        labelsMat = np.concatenate([thetav, phiv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def kleinbottle(self, ar=4, RES=100, noise=0, seed=42):
        np.random.seed(seed)
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rout = sideLx/(2*np.pi)
        Rin = sideLy/(2*np.pi)
        RESx=int(sideLx*RES+1)
        RESy=int(sideLy*RES+1)
        x=np.linspace(0,sideLx,RESx)[:-1] # remove 2pi
        y=np.linspace(0,sideLy,RESy)[:-1] # remove 2pi
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,None]/Rout
        yv = yv.flatten('F')[:,None]/Rin
        noise = np.random.uniform(-1,1,(yv.shape[0],1))*noise
        X=np.concatenate([(Rout+(1+noise)*Rin*np.cos(yv))*np.cos(xv), (Rout+(1+noise)*Rin*np.cos(yv))*np.sin(xv),
                          (1+noise)*Rin*np.sin(yv)*np.cos(xv/2), (1+noise)*Rin*np.sin(yv)*np.sin(xv/2)], axis=1)
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def kleinbottle4d(self, ar=4, n=10000, noise=0, seed=42):
        np.random.seed(seed)
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rout = sideLx/(2*np.pi)
        Rin = sideLy/(2*np.pi)
        xv = np.random.uniform(0, sideLx, n)
        yv = np.random.uniform(0, sideLy, n)
        xv = xv.flatten('F')[:,None]/Rout
        yv = yv.flatten('F')[:,None]/Rin
        noise = np.random.uniform(-1,1,(yv.shape[0],1))*noise
        X=np.concatenate([(Rout+(1+noise)*Rin*np.cos(yv))*np.cos(xv), (Rout+(1+noise)*Rin*np.cos(yv))*np.sin(xv),
                          (1+noise)*Rin*np.sin(yv)*np.cos(xv/2), (1+noise)*Rin*np.sin(yv)*np.sin(xv/2)], axis=1)
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def mobiusstrip3d(self, ar=4, RES=90):
        sideLx=np.sqrt(ar)
        sideLy=1/sideLx
        Rmax = sideLx/(2*np.pi)
        RESx=int(sideLx*RES+1)
        RESy=int(sideLy*RES+1)
        x=np.linspace(0,sideLx,RESx)[:-1] #remove 2pi
        y=np.linspace(-sideLy/2,sideLy/2,RESy)
        xv, yv = np.meshgrid(x, y)
        xv = xv.flatten('F')[:,None]/Rmax
        yv = yv.flatten('F')[:,None]
        X=np.concatenate([(Rmax+0.5*yv*np.cos(0.5*xv))*np.cos(xv),
                         (Rmax+0.5*yv*np.cos(0.5*xv))*np.sin(xv),
                         0.5*yv*np.sin(0.5*xv)], axis=1)   
        labelsMat = np.concatenate([xv, yv], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def non_uniform_trapezoid(self, ar=2, RES=100, r0=0.98, r1=0.75, noise=0, noise_type='uniform'):
        sideLx = np.sqrt(ar)
        sideLy = 1/sideLx
        RESx = int(sideLx*RES+1)
        RESy = int(sideLy*RES+1)
        x = np.linspace(0, sideLx, RESx)
        dx = x[1]-x[0]
        x = np.concatenate([[0],dx*np.power(r0,np.arange(RESx-1))])
        x = sideLx * np.cumsum(x)/np.sum(x)
        y = np.linspace(0, sideLy, RESy)
        rx = np.linspace(r0, 1, RESx)
        rSideLx = np.linspace(1, r1, RESy)
        xv_, yv = np.meshgrid(x, y)
        _, rSideLxv = np.meshgrid(x, rSideLx)
        xv = rSideLxv * xv_
        xv_ = xv_.flatten('F')[:,None]
        xv = xv.flatten('F')[:,None]
        yv = yv.flatten('F')[:,None]
        X = np.concatenate([xv,yv], axis=1)
        X_ = np.concatenate([xv_,yv], axis=1)
        if noise:
            np.random.seed(42)
            n = xv.shape[0]
            if noise_type == 'normal':
                n = xv.shape[0]
                X = np.concatenate([X,np.zeros((n,1))], axis=1)
                X = X + noise*np.random.normal(0,1,(n,3))
            elif noise_type == 'uniform':
                X = np.concatenate([X,noise*np.random.uniform(-1,1,(n,1))], axis=1)

        labelsMat = X
        print('X.shape = ', X.shape)

        n = X.shape[0]
        ddX = np.zeros(n)
        for k in range(n):
            ddXx = np.min([X_[k,0], sideLx-X_[k,0]])
            ddXy = np.min([X_[k,1], sideLy-X_[k,1]])
            ddX[k] = np.min([ddXx, ddXy]) > 0
            
        return X, labelsMat, ddX
    
    def twinpeaks(self, n=10000, noise=0, ar=4, seed=42):
        np.random.seed(seed)
        s_ = 2
        t_ = 2*ar
        t = np.random.uniform(-t_/2,t_/2,(n,1))
        s = np.random.uniform(-s_/2,s_/2,(n,1))
        h = 0.3*(1-t**2)*np.exp(-t**2-(s+1)**2)-\
            (0.2*t-np.power(t,3)-np.power(s,5))*np.exp(-t**2-s**2)-\
            0.1*np.exp(-(t+1)**2-s**2)
        
        eta = noise * np.random.normal(0,1,(n,1))
        X = np.concatenate([t,s,h+eta],axis=1)
        labelsMat = np.concatenate([t,s], axis=1)
        print('X.shape = ', X.shape)
        return X, labelsMat, None

    def twinpeaks2(self, n=1000, param=1, noise=0, seed=42):
        np.random.seed(seed)
        xy = 1 - 2 * np.random.rand(2, n)
        X = np.array([xy[1, :], xy[0, :], param * np.sin(np.pi * xy[0, :]) * np.tanh(3 * xy[1, :])]).T
        eta = noise * np.random.normal(0,1,n)
        X[:,-1] = X[:,-1] + eta
        labelsMat = X.copy()
        print('X.shape = ', X.shape)
        return X, labelsMat, None
        
    
    def floor(self, fpath, noise=0.01, n_transmitters = 42, eps = 1):
        data = scipy.io.loadmat(fpath)
        X = data['X']
        np.random.seed(42)
        X = X + np.random.uniform(0, 1, X.shape)*noise
        t_inds = np.random.permutation(range(X.shape[0]))
        t_inds = t_inds[:n_transmitters]
        t_locs = X[t_inds,:]
        
        mask = np.ones(X.shape[0])
        mask[t_inds] = 0
        X = X[mask==1,:]
        labelsMat = X.copy()
        
        dist_bw_x_and_t = cdist(X, t_locs)
        X = np.exp(-(dist_bw_x_and_t**2)/(eps**2))
        print('X.shape = ', X.shape)
        return X, labelsMat, None
        
    
    def solidcuboid3d(self, l=0.5, w=0.5, RES=20):
        sideLx = l
        sideLy = w
        sideLz = 1/(l*w)
        RESx=int(sideLx*RES+1)
        RESy=int(sideLy*RES+1)
        RESz=int(sideLz*RES+1)
        x=np.linspace(0,sideLx,RESx)
        y=np.linspace(0,sideLy,RESy)
        z=np.linspace(0,sideLz,RESz)
        xv, yv, zv = np.meshgrid(x, y, z)
        xv = xv.flatten('F')[:,None]
        yv = yv.flatten('F')[:,None]
        zv = zv.flatten('F')[:,None]
        X = np.concatenate([xv,yv,zv], axis=1)
        labelsMat = X
        print('X.shape = ', X.shape)
        return X, labelsMat, None
    
    def mnist(self, digits, n, n_pca=25, scale=True):
        X0, y0 = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
        X = []
        y = []
        for digit in digits:
            X_ = X0[y0 == str(digit),:]
            X_= X_[:n,:]
            X.append(X_)
            y.append(np.zeros(n)+digit)
            
        X = np.concatenate(X, axis=0)
        if scale:
            X = X/np.max(np.abs(X))
        y = np.concatenate(y, axis=0)
        labelsMat = y[:,None]
        
        if n_pca:
            X_new = do_pca(X,n_pca)
        else:
            X_new = X
        
        print('X_new.shape = ', X_new.shape)
        return X_new, labelsMat, X, [28,28] 
    
    def face_data(self, fpath, pc=False, n_pca=0):
        data = scipy.io.loadmat(fpath)
        if pc:
            X = data['image_pcs'].transpose()
        else:
            X = data['images'].transpose()
        labelsMat = np.concatenate([data['lights'].transpose(), data['poses'].transpose()], axis=1)
        
        if n_pca:
            X_new = do_pca(X,n_pca)
        else:
            X_new = X
        
        print('X.shape = ', X_new.shape)
        
        min_pose = np.min(labelsMat[:,1])
        min_light = np.min(labelsMat[:,0])
        max_pose = np.max(labelsMat[:,1])
        max_light = np.max(labelsMat[:,0])
        N = X.shape[0]
        ddX = np.zeros(N)
        for k in range(N):
            ddX1 = np.min([labelsMat[k,0]-min_light, max_light-labelsMat[k,0]])
            ddX2 = np.min([labelsMat[k,1]-min_pose, max_pose-labelsMat[k,1]])
            ddX[k] = np.min([ddX1, ddX2])
        
        return X_new, labelsMat, X, [64,64], ddX
    
    def puppets_data(self, dirpath, prefix='s1', n=None, bbox=None,
                     grayscale=False, normalize = False, n_pca=100):
        X = []
        labels = []
        fnames = []
        for fname in sorted(os.listdir(dirpath)):
            if prefix in fname:
                fnames.append(fname)
        
        if n is not None:
            fnames = fnames[:n]
            
        for fname in fnames:
            X_k = read_img(dirpath+'/'+fname, bbox=bbox, grayscale=grayscale)
            X.append(X_k.flatten())
            labels.append(int(fname.split('.')[0].split('_')[1])-100000)
        
        img_shape = X_k.shape
        X = np.array(X)
        labels = np.array(labels)[:,None]-1
        labelsMat = np.concatenate([labels,labels], axis=1)
        if normalize:
            X = X - np.mean(X,axis=0)[None,:]
            X = X / (np.std(X,axis=0)[None,:] + 1e-12)
            
        if n_pca:
            X_new = do_pca(X, n_pca)
        else:
            X_new = X
        
        X_new = X_new / np.max(np.abs(X_new))
        print('X.shape = ', X_new.shape)
        return X_new, labelsMat, X, img_shape
        
    
    def soils88(self, labels_path, X_path):
        df2 = pd.read_csv(X_path, sep='\t')
        df2 = df2.sort_values(by='Unnamed: 0').reset_index(drop=True)
        sample_names = df2['Unnamed: 0'].tolist()
        X = df2.to_numpy()[:,1:]
        
        df1 = pd.read_csv(labels_path, sep='\t')
        mask = df1['sample_name'].apply(lambda x: x in sample_names)
        df1 = df1[mask].reset_index(drop=True)
        df1 = df1.sort_values(by='sample_name').reset_index(drop=True)
        labelsMat = df1.to_numpy()[:,1:]
        
        print('X.shape = ', X.shape)
        return X, labelsMat, None
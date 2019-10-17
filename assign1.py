#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:02:36 2019

@author: Jiaxin Tan
"""

import numpy as np
import matplotlib.pyplot as plt

#ignore Scientific notation like e+02
np.set_printoptions(suppress=True)

#load matrix
matrix = np.loadtxt('/Users/apple/Desktop/airfoil_self_noise.dat', delimiter= None)

#get the No. of rows
n= matrix.shape[0]

#print(n)

#get attributes' sum
a= np.sum(matrix, axis=0)

# mean vector

mv= a/ n
print("The mean vector :")
print(mv)

print("-------------------------------------------")

#the squared magnitude of the mean
    #mv_magnitude= np.linalg.norm(mv)

mv2= np.linalg.norm(mv)**2

i=0
total =0
while i<n : 
  r= (np.linalg.norm(matrix[i,:]))**2
  total= total + r 
  i= i+1

total_variance = (1/n)*total-mv2


print("Total Variance: ")
print(total_variance)

print("-------------------------------------------")


# b. covariance matrix
#Mul_mv= np.repeat(mv,n, axis=1)


Mul_mv= np.tile(mv,(n,1))

centered_matrix = matrix- Mul_mv

#inner_p = (1/n)*(centered_matrix)*(centered_matrix.T)
#inner products
inner_p= (1/n)*np.matmul(centered_matrix.T,centered_matrix)

print("sample covariance matrix via inner product formula: ")
print(inner_p)


#outer products

sum = 0
#r =0

#while r < n :

	#x = np.outer(matrix[r,:].T, matrix[r,:])
#	sum = sum + np.outer(matrix[r,:].T, matrix[r,:])
#	r = r +1 
 
for r in range(0,n):
	sum = sum + np.outer(centered_matrix[r,:].T,centered_matrix[r,:])
   


outer_p =  sum/n

print("sample covariance matrix via outer product formula:")
print (outer_p)


print("-------------------------------------------")

#c. Correlation matrix as pair-wise cosines

mv_trans = mv.reshape((1,6))
X1= matrix[:,0].reshape(n,1)
X2= matrix[:,1].reshape(n,1)
X3= matrix[:,2].reshape(n,1)
X4= matrix[:,3].reshape(n,1)
X5= matrix[:,4].reshape(n,1)
X6= matrix[:,5].reshape(n,1)

m1 =np.tile(mv_trans[:,0],(n,1))
m2 =np.tile(mv_trans[:,1],(n,1)) 
m3 =np.tile(mv_trans[:,2],(n,1))
m4 =np.tile(mv_trans[:,3],(n,1))
m5 =np.tile(mv_trans[:,4],(n,1))
m6 =np.tile(mv_trans[:,5],(n,1))

Z1= X1- m1
Z2= X2- m2
Z3= X3- m3
Z4= X4- m4
Z5= X5- m5
Z6= X6- m6

Z1_nor=Z1/np.linalg.norm(Z1, axis=0, keepdims=True)
Z2_nor=Z2/np.linalg.norm(Z2, axis=0, keepdims=True)
Z3_nor=Z3/np.linalg.norm(Z3, axis=0, keepdims=True)
Z4_nor=Z4/np.linalg.norm(Z4, axis=0, keepdims=True)
Z5_nor=Z5/np.linalg.norm(Z5, axis=0, keepdims=True)
Z6_nor=Z6/np.linalg.norm(Z6, axis=0, keepdims=True)

#print(Z1_nor)

cos12= np.dot(Z1_nor.T,Z2_nor)
cos13= np.dot(Z1_nor.T,Z3_nor)
cos14= np.dot(Z1_nor.T,Z4_nor)
cos15= np.dot(Z1_nor.T,Z5_nor)
cos16= np.dot(Z1_nor.T,Z6_nor)
cos23= np.dot(Z2_nor.T,Z3_nor)
cos24= np.dot(Z2_nor.T,Z4_nor)
cos25= np.dot(Z2_nor.T,Z5_nor)
cos26= np.dot(Z2_nor.T,Z6_nor)
cos34= np.dot(Z3_nor.T,Z4_nor)
cos35= np.dot(Z3_nor.T,Z5_nor)
cos36= np.dot(Z3_nor.T,Z6_nor)
cos45= np.dot(Z4_nor.T,Z5_nor)
cos46= np.dot(Z4_nor.T,Z6_nor)
cos56= np.dot(Z5_nor.T,Z6_nor)

#25 most-correlated
plt.title("Most correlated")
plt.scatter(matrix[:,1],matrix[:,4])
plt.show()
#23 most anti-correlated
plt.title("Most anti-correlated")
plt.scatter(matrix[:,1],matrix[:,2])
plt.show()

#34 least correlated
plt.title("Least Correlated")
plt.scatter(matrix[:,2],matrix[:,3])
plt.show()


print("-------------------------------------------")


# cov is the target covariance function to modify 

cov=inner_p

# eig_computation with iterative method 
def my_eig(cov=None, max_iter=1000, epsilon=0.0001):
    d=cov.shape[0]
    np.random.seed(1)
    X0_initial=np.random.rand(d,2)+0.1
    length_0=np.sqrt(np.sum(X0_initial**2,axis=0))
    X0=X0_initial/length_0

    X=[None]*int(max_iter)
    X[0]=X0
    Xi=np.matmul(cov,X0)
    X[1]=Xi
    count=0

    while((np.linalg.norm(X[count+1] - X[count]) > epsilon) & (count < max_iter)):
        a=Xi[:,0].reshape(-1,1)
        b=Xi[:,1].reshape(-1,1)
        b=b-(np.matmul(b.T,a)/np.matmul(a.T,a))*a
        Xi_initial=np.concatenate((a,b),axis=1)
        length_i=np.sqrt(np.sum(Xi_initial**2,axis=0))
        Xi=Xi_initial/length_i
        X[count+2]=Xi
        Xi=np.matmul(cov,Xi)
        count+=1
    return X[count+1]


eigvector=my_eig(cov)
eigvalue=np.matmul(np.matmul(eigvector.T,cov),eigvector).diagonal()


print("eigenvector: ")
print(eigvector)

plt.title("Projected Points in the two new dimensions")
plt.scatter(np.dot(matrix,eigvector[:,0]),np.dot(matrix,eigvector[:,1]))

print("eigenvalues: ")
print(eigvalue[0],eigvalue[1])


#plt.show(block=False)
plt.show() 




























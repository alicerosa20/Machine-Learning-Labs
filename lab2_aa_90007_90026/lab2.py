# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:35:41 2020

@author: Alice
"""
import numpy as np
import matplotlib.pyplot as plt
import math


# Functions

#quad1

def quad1(threshold, maxiter, anim, x_o=-9, a=1, eta=0.1):
        
    it=0
    f=(a*x_o**2)/2
    x_n=x_o
    grad=a*x_o
    f_total=f
    x_total=x_n
        
    while f>threshold:
            if it>maxiter:
                print(">1000")
                break
            
            x_n1=x_n-eta*grad
                
            f_old=f
            f=(a*x_n1**2)/2
                
            grad=a*x_n1
                
            x_n=x_n1
                
            if it!=0 and (f_old==f or f>1e+6):
                print("Diverged")
                break
                
            it=it+1
            
            if anim:
                plt.plot(x_n,f,'r.-')
                plt.pause(0.2)
            else:
                f_total=np.c_[(f_total, f)]
                x_total=np.c_[(x_total, x_n)]
        
    if ~anim:
        plt.plot(x_total,f_total,'r.-')
            
        
        return it,x_n,f
 
#quad2       
                 
def quad2(threshold, maxiter, anim, x_o=np.array([-9,9]), a=2, eta=0.1):
    
    it=0
    f=(a*x_o[0]**2+x_o[1]**2)/2
    x_n=x_o
    grad=np.array([a*x_o[0], x_o[1]])
    x_total=x_n
    
    while f>threshold:
            if it>maxiter:
                print(">1000")
                break
            
            x_n1=np.asarray(x_n-eta*grad)
                
            f_old=f
            f=(a*x_n1[0]**2+x_n1[1]**2)/2
                
            grad=np.array([a*x_n1[0], x_n1[1]])
                
            x_n=x_n1
                
            if it!=0 and (f_old==f or abs(f)>1e+6):
                print("Diverged")
                break
                
            it=it+1
            
            if anim:
                plt.plot(x_n,'r.-')
                plt.pause(0.2)
            else:
                x_total=np.c_[(x_total, x_n)]
        
    if ~anim:
        plt.plot(x_total,'r.-')
            

    return it,x_n,f
#%% 2.1 Minimization of function of one variable

threshold=0.01
maxiter=1000
anim=0
x_o=-9
a=0.5
eta=0.3

(it,x_min,f_min)=quad1(threshold, maxiter, anim, x_o, a, eta)

#%% 2.2 Minimization of function of more than one variable

threshold=0.01
maxiter=1000
x_o=np.array([-9,9])
anim=0
a=20
eta=0.1


(it,x_min,f)=quad2(threshold, maxiter, anim, x_o, a, eta)

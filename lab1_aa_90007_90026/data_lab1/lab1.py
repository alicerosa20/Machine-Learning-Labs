"""
1st Lab: Linear Regression

@author: Alice Rosa 90007 
         Aprígio Malveiro 90026

"""
#%% Parte 2.1

#Funções

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

#Estimativa dos parâmetros Beta
def estimateBeta(X, y):
    
    X_t=X.transpose()
    
    A=inv(np.matmul(X_t, X))
    
    Beta=np.matmul(np.matmul(A,X_t), y)
    
    return Beta

#Construção da matrix X de grau P
def computeX(x, P):
    
    n=x.shape[0]
    
    X=np.ones((n,1))
    
    for i in range (1, P+1):
        X=np.c_[(X, np.power(x,i))]
    return X

#Cálculo do SSE
def computeSSE(y,y_est):
    
    e=y-y_est
    
    SSE=np.matmul(e.transpose(),e)
    
    return SSE

#Identificação dos outliers num determinado dataset
def computeOutliers(y,y_est):
    
    e=y-y_est
    
    return np.nonzero(e > 1), e[e>1]

#Método que subtrai a cada coluna o seu valor médio, de forma a normalizá-los
def pre_processing(x):

    x_new = np.copy(x)
    if x.mean() == 0:
        return (x)
    else :
        c = x.shape[1]
        x_av = x.mean(0)
        for i in range(0,c):
            x_new[:,i] = x[:,i]- x_av[i]
        return x_new,x_av
#%% Exercicío 2.1.3

x=np.load('data1_x.npy')
y=np.load('data1_y.npy')

P=1
X=computeX(x, P)

beta=estimateBeta(X, y)

y_est=np.matmul(X, beta)

sse=computeSSE(y, y_est)

#Plot
plt.figure()
plt.scatter(x,y,color='red')
plt.plot(x,y_est)
plt.title('2.1.3: Least Squares')
plt.legend(['Fit', 'Training Set'])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Resultados
print("2.1.3. (b)")
print("Beta_0 =", beta[0,0])
print("Beta_1 =", beta[1,0])
print("SSE =", sse[0,0])

#%% Exercicío 2.1.4

import operator

x=np.load('data2_x.npy')
y=np.load('data2_y.npy')


P=2
X=computeX(x, P)

beta=estimateBeta(X, y)

y_est=np.matmul(X, beta)

sse=computeSSE(y, y_est)

#Plot
plt.figure()
plt.scatter(x,y,color='red')

#Ordenação dos dados
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_est), key = sort_axis)
x, y_est = zip(*sorted_zip)


plt.plot(x,y_est)

#Legenda e título
plt.title('2.1.4: Second-degree polynomial')
plt.legend(['Fit', 'Training Set'])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Resultados
print("2.1.4. (b)")
print("Beta_0 =", beta[0,0])
print("Beta_1 =", beta[1,0])
print("Beta_2 =", beta[2,0])
print("SSE =", sse[0,0])


#%% Exercício 2.1.5

import operator

x=np.load('data2a_x.npy')
y=np.load('data2a_y.npy')
y=y.reshape(-1,1)

P=2
X=computeX(x, P)

beta=estimateBeta(X, y)

y_est=np.matmul(X, beta)

#Calculos com outliers
sse_out=computeSSE(y, y_est)

(indices, outliers)=computeOutliers(y, y_est)

#Calculos sem outliers

y_inl=np.delete(y,[indices[0][0], indices[0][1]])
y_est_inl=np.delete(y_est,[indices[0][0], indices[0][1]])
sse_inl=computeSSE(y_inl, y_est_inl)

# Plot
plt.figure()
plt.scatter(x,y,color='red')

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x, y_est), key = sort_axis)
x, y_est = zip(*sorted_zip)


plt.plot(x,y_est)

#Legenda e título
plt.title('2.1.5: Second-degree polynomial with outliers')
plt.legend(['Fit', 'Training Set'])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Resultados
print("2.1.5. (b)")
print("Beta_0 =", beta[0,0])
print("Beta_1 =", beta[1,0])
print("Beta_2 =", beta[2,0])
print("SSE with outliers =", sse_out[0,0])
print("SSE without outliers =", sse_inl)

#%% Parte 2.2.

# Exercício 2.2.4

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


X=np.load('data3_x.npy')
y=np.load('data3_y.npy')

#Pré-processamento dos dados
(X_av,Xb) = pre_processing(X)
(y_av,yb) = pre_processing(y)

#Estimativa dos coeficientes para o método LS
beta = estimateBeta(X_av, y_av)
beta = beta.transpose()

#Vector de alphas
start = 10**(-3)
stop = 10
step = 0.01
alphas = np.arange(start, stop, step)


for a in alphas:
    
    #fit de Ridge
    rr = Ridge(alpha=a,max_iter=(10000))
    rr.fit(X,y)
    
    #fit de Lasso
    lasso = Lasso(alpha=a,max_iter=(10000))
    lasso.fit(X,y)
    
    #Não é necessário utilizar o x e y processados neste caso, uma vez que tanto 
    # o Ridge como o Lasso já realizam esse processo dentro das suas funções

    #Cálculo dos coeficientes de Lasso e Ridge
    if a == start:
            rr_coefs = rr.coef_
            Lasso_coefs = lasso.coef_
            betas =  beta

    else:
        rr_coefs = np.r_[rr_coefs, rr.coef_]
        Lasso_coefs =  np.c_[Lasso_coefs , lasso.coef_]
        betas =   np.r_[betas, beta]
        
#Reposição das dimensões originais
beta = beta.transpose()
Lasso_coefs =Lasso_coefs.transpose()


#%% Exercício 2.2.5

#Plot Ridge
plt.figure()
plt.plot(alphas, betas, linestyle='--')
plt.plot(alphas, rr_coefs)
plt.legend(['LS \u03B21', 'LS \u03B22', 'LS \u03B23', 'Ridge \u03B21', 'Ridge \u03B22', 'Ridge \u03B23'])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('2.2.5: Ridge coefficients')
plt.axis('tight')
plt.show()



#Plot Lasso
plt.figure()
plt.plot(alphas, betas, linestyle='--')
plt.plot(alphas, Lasso_coefs)
plt.legend(['LS \u03B21', 'LS \u03B22', 'LS \u03B23', 'Lasso \u03B21', 'Lasso \u03B22', 'Lasso \u03B23'])
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('2.2.5: Lasso coefficients')
plt.axis('tight')
plt.show()



#%% 2.2.7

# Não correr antes de correr o 2.2.4

# Limpar variáveis 
y_est_LS = None
sse_LS = None
y_est_lasso = None
sse_lasso = None

#Escolha do Alpha
beta_lasso = Lasso_coefs[4,:]
beta_lasso = beta_lasso.reshape(3,1)

xx = np.linspace(0,50)

#Cálculo do y estimado e do SSE para o método LS
y_est_LS=np.matmul(X_av, beta)
sse_LS=computeSSE(y_av, y_est_LS)

#Cálculo do y estimado e do SSE para método Lasso
y_est_lasso=np.matmul(X_av, beta_lasso)
sse_lasso=computeSSE(y_av, y_est_lasso)

#Plot 
plt.figure()
plt.plot(xx,y_est_LS)
plt.plot(xx,y_est_lasso,color='green')
plt.scatter(xx,y_av,marker='o',color='red' )
plt.legend(['LS', 'Lasso', 'Data'])
plt.title('2.2.7: LS vs Lasso')
plt.xlabel('Position')
plt.ylabel('y')

#Resultados
print("2.2.7")
print("Alpha =", alphas[4])
print("SSE LS =",sse_LS[0][0])
print("SSE Lasso =",sse_lasso[0][0])
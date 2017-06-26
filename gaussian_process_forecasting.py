

# coding: utf-8

# In[132]:

import matplotlib.pyplot as plt
import os, os.path
import numpy as np
import GPflow
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
get_ipython().magic('matplotlib inline')

#Obtener los archivos de datos.
DIR = 'database/'
files=([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
n=len(files)
D=np.array([])
for i in range(0,len(files)):    
    source=DIR+files[i]
    file = open(source, "r")
    d=[]
    f=[]
    for line in file: 
        if len(line.split())>2:
            f.append(line.split()[2])        
        if len(line.split())==9:
            d.append(line.split()[7])
        elif len(line.split())==5:
            d.append(line.split()[3])
        else:
            d.append(0)
    d.pop(0)
    myarray = np.asarray(d)
    file.close()    
    D = np.append(D, np.transpose(myarray), axis=0)
    
D=D.reshape(n,len(D)/n)
D=D.astype(np.float)

#Borar datos defectuosos/no adquiridos.
ind=[]
for j in range(0,len(D)):
    for i in range(0,len(D[0])):
        if D[j,i]==0:
            ind.append(i)
a=list(set(ind))
a=np.asarray(a)

DO=np.delete(D[0],ind)
EC=np.delete(D[1],ind)
RD=np.delete(D[2],ind)
T=np.delete(D[3],ind)
PH=np.delete(D[4],ind)

#Separa los conjuntos en Entrenamiento y Test.
PE=.75
PT=.25
n=len(DO)
DO_train=DO[0:np.round(n*PE)]
DO_train.shape=(len(DO_train),1)
DO_test=DO[np.round(n*PE)+1:-1]
DO_test.shape=(len(DO_test),1)


EC_train=EC[0:np.round(n*PE)]
EC_train.shape=(len(EC_train),1)
EC_test=EC[np.round(n*PE)+1:-1]
EC_test.shape=(len(EC_test),1)


RD_train=RD[0:np.round(n*PE)]
RD_train.shape=(len(RD_train),1)
RD_test=RD[np.round(n*PE)+1:-1]
RD_test.shape=(len(RD_test),1)


T_train=T[0:np.round(n*PE)]
T_train.shape=(len(T_train),1)
T_test=T[np.round(n*PE)+1:-1]
T_test.shape=(len(T_test),1)


PH_train=PH[0:np.round(n*PE)]
PH_train.shape=(len(PH_train),1)
PH_test=PH[np.round(n*PE)+1:-1]
PH_test.shape=(len(PH_test),1)
#Graficar los valores
p=range(0,len(DO))

plt.plot(p[0:len(DO_train)],DO_train,'b')
plt.plot(p[len(DO_train)+1:-1],DO_test,'r')
#lt.show()

#Métricas a utilizar.
def RMSE(obs,med):
    r=np.power(np.subtract(obs,med),2)
    return np.sqrt((1/len(obs)*np.sum(r)))

def NS(obs,med):
    mean=np.mean(obs)
    num=np.sum(np.power(np.subtract(obs,med),2))
    den=np.sum(np.power(np.subtract(obs,np.mean(obs)),2))
    return 1-num/den

def MARE(obs,med):
    num=np.subtract(obs,med)
    return (1/len(obs))*100*np.sum(np.absolute(np.divide(num,med)))

def R(obs,med):
    num=np.sum(np.multiply(np.subtract(obs,np.mean(obs)),np.subtract(med,np.mean(med))))
    den=np.multiply(np.sum(np.power(np.subtract(obs,np.mean(obs)),2)),np.sum(np.power(np.subtract(med,np.mean(med)),2)))
    return num/np.sqrt(den)

def Norm(x,X):
    return np.divide(np.subtract(x,np.mean(X)),np.sqrt(np.var(X)))
def DeNorm(x,X):
    return np.multiply(np.add(x,np.mean(X)),np.sqrt(np.var(X)))


# # Gaussian Kernel

# In[44]:

#Utilizando sólo la T como entrada
Train=np.squeeze(np.dstack((T_train)),1)
k = GPflow.kernels.RBF(input_dim=1,ARD=True)
m = GPflow.gpr.GPR(np.transpose(Train), DO_train, kern=k)
print(m)
# Optimización de los parámetros
m.optimize()
print(m)


# In[87]:

print(m.likelihood)
# Predicción punto a punto
Test=np.squeeze(np.dstack((T_test)),1)
mean, var = m.predict_y(np.transpose(Test))

# Métricas del paper
print("RMSE: ",RMSE(mean,DO_test))
print("MARE: ",MARE(mean,DO_test))
print("NS :",NS(mean,DO_test))
print("R :",R(mean,DO_test))

inf=mean[:,0] - 2*np.sqrt(var[:,0])
sup=mean[:,0] + 2*np.sqrt(var[:,0])
plt.plot(mean)
plt.plot(DO_test)
plt.plot(inf,color='blue',alpha=0.2)
plt.plot(sup,color='blue',alpha=0.2)
#plt.fill_between(np.transpose(inf),np.transpose(sup) ,color='blue', alpha=0.2)
plt.show()


# In[107]:

#Utilizando la T y PH
Train=np.squeeze(np.dstack((T_train,PH_train)),1)
k = GPflow.kernels.RBF(input_dim=2,ARD=True)
m = GPflow.gpr.GPR(Train, DO_train, kern=k)
print(m)
m.optimize()
print(m)


# In[113]:

print(m.likelihood)
# Predicción punto a punto
Test=np.squeeze(np.dstack((T_test,PH_test)),1)
mean, var = m.predict_y(Test)

# Métricas del paper
print("RMSE: ",RMSE(mean,DO_test))
print("MARE: ",MARE(mean,DO_test))
print("NS :",NS(mean,DO_test))
print("R :",R(mean,DO_test))

inf=mean[:,0] - 2*np.sqrt(var[:,0])
sup=mean[:,0] + 2*np.sqrt(var[:,0])
plt.plot(mean)
plt.plot(DO_test)
plt.plot(inf,color='blue',alpha=0.2)
plt.plot(sup,color='blue',alpha=0.2)
#plt.fill_between(np.transpose(inf),np.transpose(sup) ,color='blue', alpha=0.2)
plt.show()


# In[134]:

#Utilizando sólo la T como entrada normalizada
Train=np.squeeze(np.dstack((Norm(T_train,T))),1)
k = GPflow.kernels.RBF(input_dim=1,ARD=True)
m = GPflow.gpr.GPR(np.transpose(Train), Norm(DO_train,DO), kern=k)
print(m)
# Optimización de los parámetros
m.optimize()
print(m)


# In[139]:

print(m.likelihood)
# Predicción punto a punto
Test=np.squeeze(np.dstack((Norm(T_test,T))),1)
mean, var = m.predict_y(np.transpose(Test))

# Métricas del paper
print("RMSE: ",RMSE(mean,Norm(DO_test,DO)))
print("MARE: ",MARE(mean,Norm(DO_test,DO)))
print("NS :",NS(mean,Norm(DO_test,DO)))
print("R :",R(mean,Norm(DO_test,DO)))

inf=mean[:,0] - 2*np.sqrt(var[:,0])
sup=mean[:,0] + 2*np.sqrt(var[:,0])
plt.plot(mean)
plt.plot(DO_test_n)
plt.plot(inf,color='blue',alpha=0.2)
plt.plot(sup,color='blue',alpha=0.2)
#plt.fill_between(np.transpose(inf),np.transpose(sup) ,color='blue', alpha=0.2)
plt.show()


# In[143]:

#Utilizando la T y PH
Train=np.squeeze(np.dstack(( Norm(T_train,T),Norm(RD_train,RD),Norm(PH_train,PH) )),1)
k = GPflow.kernels.RBF(input_dim=3,ARD=True)
m = GPflow.gpr.GPR(Train, Norm(DO_train,DO), kern=k)
print(m)
m.optimize()
print(m)


# In[144]:

print(m.likelihood)
# Predicción punto a punto
Test=np.squeeze(np.dstack(( Norm(T_test,T),Norm(RD_test,RD),Norm(PH_test,PH) )),1)
mean, var = m.predict_y(Test)

# Métricas del paper
print("RMSE: ",RMSE(mean,Norm(DO_test,DO) ))
print("MARE: ",MARE(mean,Norm(DO_test,DO)))
print("NS :",NS(mean,Norm(DO_test,DO)))
print("R :",R(mean,Norm(DO_test,DO)))

inf=mean[:,0] - 2*np.sqrt(var[:,0])
sup=mean[:,0] + 2*np.sqrt(var[:,0])
plt.plot(mean)
plt.plot(DO_test)
plt.plot(inf,color='blue',alpha=0.2)
plt.plot(sup,color='blue',alpha=0.2)
#plt.fill_between(np.transpose(inf),np.transpose(sup) ,color='blue', alpha=0.2)
plt.show()

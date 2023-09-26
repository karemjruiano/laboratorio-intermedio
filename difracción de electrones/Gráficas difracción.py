# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 17:38:33 2023

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
import xlsxwriter
import scipy.stats as stats


#lectura del archivo.
dif=pd.read_csv("Datos difracción.csv",sep=";")


#extracción de datos del archivo. 
vol= dif["U (V)"].iloc[:]
D1= dif["D1 (m)"].iloc[:]
D2= dif["D2 (m)"].iloc[:]


#cambio de las comas por puntos
vol=np.array([float(valor.replace(',', '.')) if isinstance(valor, str) else valor for valor in vol])
D1=np.array([float(valor.replace(',', '.')) if isinstance(valor, str) else valor for valor in D1])
D2=np.array([float(valor.replace(',', '.')) if isinstance(valor, str) else valor for valor in D2])


#creación de arrays
vol= np.array(vol)
D1_y= np.array(D1)
D2_y= np.array(D2)


#Calculo de v^(-1/2)
V=[]
for cada in vol:
    v_new= cada**(-1/2)
    V.append(v_new)

V_x= np.array(V)


#Regresión lineal
lim_x=[0]
for cada in V:
    lim_x.append(cada)
    
lim_x=np.array(lim_x)

lim_p1=[0]
for cada in D1_y:
    lim_p1.append(cada)
    
lim_p1=np.array(lim_p1)

lim_p2=[0]
for cada in D2_y:
    lim_p2.append(cada)
    
lim_p2=np.array(lim_p2)

lim_x=lim_x.reshape((-1,1))

modelo1=LinearRegression()
modelo1.fit(lim_x,lim_p1)
pred_y1= modelo1.predict(lim_x)

modelo2=LinearRegression()
modelo2.fit(lim_x,lim_p2)
pred_y2= modelo2.predict(lim_x)

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(V_x,D1_y)
error_m1 = std_err1
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(V_x,D1_y)
error_m2 = std_err2


#Ecuaciones de la recta
m1=modelo1.coef_[0]
b1=modelo1.intercept_
ecuacion1 = f'y = {m1:.2f}x - {abs(b1):.2f}'

m2=modelo2.coef_[0]
b2=modelo2.intercept_
ecuacion2 = f'y = {m2:.2f}x + {b2:.2f}'

#Gráfica con regresión líneal
plt.scatter(V_x, D1, color="black", label="D1")
plt.scatter(V_x, D2, color="blue", label="D1")
plt.plot(lim_x, lim_p1, linestyle='--', color="black", label="Ajuste D1")
plt.plot(lim_x, lim_p2, linestyle='--', color="blue", label="Ajuste D2")
plt.xlabel(r'$V^-$'+'$^1$'+'$^/$'+'$^2$'+' (V)',fontfamily='Times New Roman',fontsize=14)
plt.ylabel('Diametro (m)',fontfamily='Times New Roman',fontsize=14)
plt.title('Cambio del diametro de difracción\ncon respecto a un voltaje',fontfamily='Times New Roman',fontsize=18)
plt.ylim(0,0.06)
plt.xlim(0,0.019)
plt.text(0.0118,0.01,ecuacion1, fontsize=12, bbox=dict(boxstyle='square,pad=0.5', facecolor='white'))
plt.text(0.0064,0.04,ecuacion2, fontsize=12, bbox=dict(boxstyle='square,pad=0.5', facecolor='white'))
plt.legend()
plt.grid()
plt.show()
plt.savefig("Diametros")



#Calculo y gráfica de residuales
res1=D1-pred_y1[1:]
res2=D2-pred_y2[1:]

fig, axs = plt.subplots(2, 1, figsize=(8, 12))

axs[0].scatter(V_x, res1, color="black", label="residuales")
axs[0].axhline(y=0, linestyle='--')
axs[0].set_xlabel('Voltaje (V)',fontfamily='Times New Roman',fontsize=14)
axs[0].set_ylabel('Residuales (-)',fontfamily='Times New Roman',fontsize=14)
axs[0].set_title('Residuales de diametro 1 (pequeño)',fontfamily='Times New Roman',fontsize=18)
axs[0].grid(True)

axs[1].scatter(V_x, res2, color="black", label="residuales")
axs[1].axhline(y=0, linestyle='--')
axs[1].set_xlabel('Voltaje (V)',fontfamily='Times New Roman',fontsize=14)
axs[1].set_ylabel('Residuales (-)',fontfamily='Times New Roman',fontsize=14)
axs[1].set_title('Residuales de diametro 2 (grande)',fontfamily='Times New Roman',fontsize=18)
axs[1].grid(True)

plt.savefig("Residuales")
plt.show()


#errores diametros
error_D1= dif["Ancho D1 (m)"].iloc[:]
error_D2= dif["Ancho D2 (m) "].iloc[:]

error_D1=np.array([float(valor.replace(',', '.')) if isinstance(valor, str) else valor for valor in error_D1])
error_D2=np.array([float(valor.replace(',', '.')) if isinstance(valor, str) else valor for valor in error_D2])

#Longitud de onda de Broglie
def de_broglie(v):
    e=1.6e-19
    h=6.62e-34
    me=9.1e-31
    lamda=h/(np.sqrt(2*e*v*me))
    return lamda

def bragg(d,D):
    L=0.135
    lamda=d*(D/(2*L))
    return lamda

l_broglie=[]
for i in (vol):
    l_broglie.append(de_broglie(i))
    
    
#Propagación de error lamda de Broglie
def errorbroglie(v):
    e=1.6e-19
    h=6.62e-34
    me=9.1e-31
    delta=((2*e*me*h)/(2*v*e*me)**(-3/2))*0.1
    return delta

error_broglie=[]
for i in range(len(vol)):
    error_broglie.append(errorbroglie(vol[i]))

#Longitud de onda de Bragg    
l_braggD1=[]
l_braggD2=[]
dp=213e-12
dg=123e-12
for i in range(len(D1_y)):
    l_braggD1.append(bragg(dp,D1_y[i]))
    l_braggD2.append(bragg(dg,D2_y[i]))


#Propagación de error lamda de Broglie
def errorbragg(d,error_D):
    L=0.135
    delta=(d/(2*L))*error_D
    return delta

error_braggD1=[]
error_braggD2=[]
for i in range(len(error_D1)):
    error_braggD1.append(errorbragg(dp,error_D1[i]))
    error_braggD2.append(errorbragg(dg,error_D2[i]))


#Creación de tablas de datos
t_broglie=[]
for i in range(len(error_broglie)):
    valor=[l_broglie[i],error_broglie[i]]
    t_broglie.append(valor)

t_bragg=[]    
for i in range(len(l_braggD1)):
    dupla=[l_braggD1[i],error_braggD1[i],l_braggD2[i],error_braggD2[i]]
    t_bragg.append(dupla)
    
header=["lamda De Broglie","Incertidumbre (+/-)"]
lamda_broglie=tabulate(t_broglie,header,tablefmt="grid")
print(lamda_broglie)
data_broglie=pd.DataFrame(t_broglie,columns=header)

headers=["Lamda con D1","Incertidumbre (+/-)","Lamda con D2","Incertidumbre (+/-)"]
lamda_bragg=tabulate(t_bragg,headers,tablefmt="grid")
print(lamda_bragg)
data_bragg=pd.DataFrame(t_bragg,columns=headers)


#Calculo de distancias interplanares
def distancias(lamda,D):
    L=0.135
    d=(lamda*2*L)/D
    return d

d1=[]
d2=[]
for i in range(len(D1_y)):
    d1.append(distancias(l_broglie[i],D1_y[i]))
    d2.append(distancias(l_broglie[i],D2_y[i]))
    

#Propagación de error distancias interplanares
def error_dist(lamda,error_lamda,D,error_D):
    L=0.135
    d_lamda=(2*L)/D
    d_D=(lamda*2*L)/(D**2)
    delta=np.sqrt((d_lamda*error_lamda)**2+(d_D*error_D)**2)
    return delta

error_d1=[]
error_d2=[]
for i in range(len(D1_y)):
    error_d1.append(error_dist(l_broglie[i],error_broglie[i],D1_y[i],error_D1[i]))
    error_d2.append(error_dist(l_broglie[i],error_broglie[i],D2_y[i],error_D2[i]))


#Creación de tablas
t_dist=[]
for i in range(len(d1)):
    dupla=[d1[i],error_d1[i],d2[i],error_d2[i]]
    t_dist.append(dupla)
    
head=["d1","Incertidumbre (+/-)","d2","Incertidumbre (+/-)"]
dist=tabulate(t_dist,head,tablefmt="grid")
print(dist)
data_dist=pd.DataFrame(t_dist,columns=head)

#Calculo Constante de Planck
def planck(v,D,d):
    L=0.135
    e=1.6e-19
    h=6.62e-34
    me=9.1e-31
    h=np.sqrt(2*e*v*me)*d*(D/(2*L))
    return h

h_D1=[]
h_D2=[]
for i in range(len(D1_y)):
    h_D1.append(planck(vol[i],D1_y[i],d1[i]))
    h_D2.append(planck(vol[i],D2_y[i],d2[i]))


#Propagación de error constante de planck
def error_h(v,d,error_d,D,error_D):
    L=0.135
    e=1.6e-19
    me=9.1e-31
    d_v=((d*D)/(2*L))*((me*e)/(2*v*me*e)**(1/2))
    d_d=np.sqrt(2*v*me*e)*(D/(2*L))
    d_D=np.sqrt(2*v*me*e)*(d/(2*L))
    delta=np.sqrt((d_v*0.1)**2+(d_d*error_d)**2+(d_D*error_D)**2)
    return delta

error_hD1=[]
error_hD2=[]
for i in range(len(D1_y)):
    error_hD1.append(error_h(vol[i],d1[i],error_d1[i],D1_y[i],error_D1[i]))
    error_hD2.append(error_h(vol[i],d2[i],error_d2[i],D2_y[i],error_D2[i]))    


#Creación de tablas    
t_h=[]
for i in range(len(h_D1)):
    dupla=[h_D1[i],error_hD1[i],h_D2[i],error_hD2[i]]
    t_h.append(dupla)
    
title=["h con D1","Incertidumbre (+/-)","h con D2","Incertidumbre (+/-)"]
cons_h=tabulate(t_h,title,tablefmt="grid")
print(cons_h)
data_h=pd.DataFrame(t_h,columns=title)



#Calculo Constante de Planck
def planck2(d,m):
    L=0.135
    e=1.6e-19
    me=9.1e-31
    h= (d*m*np.sqrt(2*me*e))/(2*L)
    return h

h_D12=planck2(2.13e-10,m1)
h_D22=planck2(1.23e-10,m2)


#Propagación de error constante de planck
def error_h2(m,error_m,d):
    L=0.135
    e=1.6e-19
    me=9.1e-31
    d_m=(np.sqrt(2*me*e)*d)/(2*L)
    delta=d_m*error_m
    return delta

error_hD12=error_h2(m1,error_m1,2.13e-10)
error_hD22= error_h2(m2,error_m2,1.23e-10)

print(h_D12,error_hD12)
print(h_D22,error_hD22)


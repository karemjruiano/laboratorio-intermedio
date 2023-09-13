# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 22:52:01 2023

@author: USUARIO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import sympy as sp

niquel=pd.read_csv("niquel.csv",sep=";")
hierro=pd.read_csv("hierro.csv",sep=";")

#datos niquel

max_niquel=niquel["n(max)"].iloc[:]
i_niquel=niquel["amperios"].iloc[:]

max_niquel=np.array([float(valor.replace(',', '.')) if isinstance(valor, str) else valor for valor in max_niquel])
i_niquel=np.array([float(valor.replace(',', '.')) if isinstance(valor, str) else valor for valor in i_niquel])

#datos hierro

max_hierro=hierro["n(max)"].iloc[:4]
i_hierro=hierro["amperios"].iloc[:4]

max_hierro=np.array(max_hierro)
i_hierro=np.array(i_hierro)

#datos campo

campo=hierro["campo (mT)"].iloc[:]
i_campo=hierro["corriente (A)"].iloc[:]

campo_y=np.array(campo)
i_x=np.array(i_campo)

#regresión lineal de gráfica de campo magnético

i_x=i_x.reshape((-1,1))
modelo=LinearRegression()
modelo.fit(i_x,campo_y)
pred_y= modelo.predict(i_x)

#pendiente e intercepto

m=modelo.coef_[0]
b=modelo.intercept_
ecuacion = f'y = {m:.2f}x + {b:.2f}'

#gráfica campo magnético

plt.scatter(i_campo, campo, color="black", label="Datos")
plt.plot(i_campo, pred_y, linestyle='--', color="black", label="Ajuste")
plt.xlabel('Corriente (A)',fontfamily='Times New Roman',fontsize=14)
plt.ylabel('Campo magnético (mT)',fontfamily='Times New Roman',fontsize=14)
plt.title('Gráfica campo magnético de la bobina',fontfamily='Times New Roman',fontsize=18)
plt.text(3,27,ecuacion, fontsize=12, bbox=dict(boxstyle='square,pad=0.5', facecolor='white'))
plt.legend()
plt.grid()
plt.savefig("Gráfica campo")
plt.show()

#calculo residuales

residuales= campo - pred_y

plt.scatter(i_campo, residuales, color="black", label="residuales")
plt.axhline(y=0, linestyle='--')
plt.xlabel('Corriente (A)',fontfamily='Times New Roman',fontsize=14)
plt.ylabel('Residuales (-)',fontfamily='Times New Roman',fontsize=14)
plt.title('Gráfica residuales del campo magnético',fontfamily='Times New Roman',fontsize=18)
plt.legend()
plt.grid()
plt.savefig("Gráfica residuales")
plt.show()

#cálculo de campo para cada elemento

x= sp.symbols('x')
ec=m*x+b

campo_niquel=[]
campo_hierro=[]

for cada in i_niquel:
    res=ec.subs(x,cada)
    campo_niquel.append(res)
    
for cada in i_hierro:
    res=ec.subs(x,cada)
    campo_hierro.append(res)

campo_niquel=np.array(campo_niquel)
campo_hierro=np.array(campo_hierro)
    
#calculo de delta l/l

l=1.4e8
lamda=633
ec2=(x*lamda)/2

delta_niquel=[]
delta_hierro=[]

for cada in max_niquel:
    deltal=(ec2.subs(x,cada))/l
    delta_niquel.append(deltal)
    
for cada in max_hierro:
    deltal=(ec2.subs(x,cada))/l
    delta_hierro.append(deltal)
    
delta_niquel=np.array(delta_niquel)
delta_hierro=np.array(delta_hierro)

#errores

error_n= []
error_h= []

for cada in range(0,14):
    error_n.append(5.7e-7)
    if cada<4:
        error_h.append(5.7e-7)
        
error_n= np.array(error_n)
error_h= np.array(error_h)

#Gráficas del Niquel y el Hierro

plt.scatter(campo_niquel, delta_niquel, color="black")
plt.errorbar(campo_niquel, delta_niquel, yerr=error_n, fmt='o', capsize=5, color="black")
plt.title('Gráfico del comportamiento del Niquel',fontfamily='Times New Roman',fontsize=18)
plt.xlabel('Campo magnético (mT)',fontfamily='Times New Roman',fontsize=14)
plt.ylabel('(\u0394l)/l (-)',fontfamily='Times New Roman',fontsize=14)
plt.grid()
plt.savefig("Gráfica Niquel")
plt.show()

plt.scatter(campo_hierro, delta_hierro, color="black")
plt.errorbar(campo_hierro, delta_hierro, yerr=error_h, fmt='o', capsize=5, color="black")
plt.title('Gráfico del comportamiento del Hierro',fontfamily='Times New Roman',fontsize=18)
plt.xlabel('Campo magnético (mT)',fontfamily='Times New Roman',fontsize=14)
plt.ylabel('(\u0394l)/l (-)',fontfamily='Times New Roman',fontsize=14)
plt.grid()
plt.savefig("Gráfica Hierro")
plt.show()


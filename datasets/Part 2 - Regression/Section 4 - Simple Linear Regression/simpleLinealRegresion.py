# -*- coding: utf-8 *-
"""
Created on Mon May 30 12:16:13 2022

@author: julian.barbosa
"""

# Regresion Lineal Simple

#Como importa librerias 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataSet

dataSet = pd.read_csv('Salary_Data.csv')
X = dataSet.iloc[:,:-1].values #Vector de caracteristicas 
y = dataSet.iloc[:,1].values #Variable dependiente, Vector

# Dividir el set enn conjunto de entrenamiento con conjunto de testing
from sklearn.model_selection import train_test_split   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0 )

#Crear de modelo de regresion Lineal simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
y_pred = regression.predict(X_test)

#Visualizar los resultados de entrenamiento
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regression.predict(X_train), color='blue')
#plt.plot(X_train, y_pred, color='purple')
plt.title('Sueldo vs Años de Experiencia (Conjunto de Entrenamiento)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Sueldo en Dolares $')
plt.show()

#Visualizar los resultados de test
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regression.predict(X_train), color='blue')
plt.title('Sueldo vs Años de Experiencia (Conjunto de Testing)')
plt.xlabel('Años de Experiencia')
plt.ylabel('Sueldo en Dolares $')
plt.show()


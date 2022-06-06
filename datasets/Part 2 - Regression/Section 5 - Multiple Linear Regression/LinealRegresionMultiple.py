# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 19:38:54 2022

@author: julian.barbosa
"""

#Regrsión Lineal Multiple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset

dataSet = pd.read_csv('50_Startups.csv')
X = dataSet.iloc[:,:-1].values #Matriz de caracteristicas 
y = dataSet.iloc[:,4].values #Variable dependiente, Vector

#Codificar datos categoricos a numericos
from sklearn import preprocessing
lab_x = preprocessing.LabelEncoder() #Codificador de datos
#Calcula los datos a codificar lab_x.fit_transform(X[:,0]) , luego se le modifica la columna X[:,0] 
X[:,3] = lab_x.fit_transform(X[:,3]) 

##Esto lo que hace es el oneHotEncoder para la matriz X
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer( 
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[3])], 
    remainder ='passthrough'
    )

X = np.array(ct.fit_transform(X), dtype=np.float64())

#Tengo que elimar una columna Dummy para no caer en variables ficticias (multicolinealidad)
X = X[:, 1:]

#Dividir el DataSet en entrenamienot y testing
from sklearn.model_selection import train_test_split   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

#Crear de modelo de regresion Lineal simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test
y_pred = regression.predict(X_test)


#Construir el modelo optimo de RLM utilizando hacia atras 
import statsmodels.api as sm 
X = np.append(arr = np.ones([50,1]).astype(int),values = X, axis = 1)

#Creacion del Nivel de Significancia 
SL = 0.05

"""
#Eliminación hacia atras
X_opt = X[:, [0,1,2,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regression_OLS.summary())

X_opt = X[:,[0,1,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regression_OLS.summary())

X_opt = X[:, [0,3,4,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regression_OLS.summary())

X_opt = X[:, [0,3,5]].tolist()
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#print(regression_OLS.summary())

X_opt = X[:, [0,3]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regression_OLS.summary())
max_valu = max(regression_OLS.pvalues).astype(float)
"""

#-----FUNCIÓN HACIA ATRAS -----#
def backwardElimination (x, sl):
    numVariables = len(x[0])
    for i in range(0, numVariables):
        regression_OLS = sm.OLS(endog = y, exog = x).fit()
        max_value = max(regression_OLS.pvalues).astype(float)
        if max_value > sl:
            for j in range(0, numVariables - i):
                if (regression_OLS.pvalues[j].astype(float) == max_value):
                    x = np.delete(x,j,1)
    regression_OLS.summary()
    return x

X_opt = X[:, [0,1,2,3,4,5]]
X_modeling = backwardElimination(X_opt, SL)



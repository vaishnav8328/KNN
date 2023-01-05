# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 12:30:07 2022

@author: vaishnav
"""

#===================================================================================================================
#importing the data

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


df = pd.read_csv(r"C:\anaconda\New folder (2)\glass.csv")
df


df.describe()

df.info()

#================================================================================================================
#visualization
import seaborn as sns
import matplotlib.pyplot as plt


# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))


# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()



# Scatter plot of two features, and pairwise plot
sns.scatterplot(df['RI'],df['Na'],hue=df['Type'])



#pairwise plot of all the features
sns.pairplot(df,hue='Type')
plt.show()


#==============================================================================================================================================================
#splitting the data into x and y

x = df.iloc[:,0:9]

y = df.iloc[:,9]

#====================================================================================================================================
#data transformation

from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()
mm_x = mm.fit_transform(x)



#===================================================================================================================================
#data partition

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(mm_x,y,test_size=0.33,random_state=(24))

#==============================================================================================================================

#KNeighbour calssifier

train_accuracy=[]
test_accuracy=[]

for i in range(3,24,2):
    knn = KNeighborsClassifier(n_neighbors=i,p=2)
    knn.fit(x_train,y_train)
    y_pred_train = knn.predict(x_train)
    y_pred_test = knn.predict(x_test)
    train_accuracy.append(accuracy_score(y_train,y_pred_train).round(2))
    test_accuracy.append(accuracy_score(y_test,y_pred_test).round(2))
    

import numpy as np

np.mean(train_accuracy).round(2)
np.mean(test_accuracy).round(2)




#======================================================================================================================================================================
#models

models = []
models.append(('LogisticRegression', LogisticRegression()))
models.append(('SVM', SVC()))
models.append(('Decision Tree Classifier', DecisionTreeClassifier()))
models.append(('Random Forest Classifier', RandomForestClassifier(max_depth=0.7)))

for title, modelname in models:
    modelname.fit(x_train, y_train)

    y_pred = modelname.predict(x_test)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print(title,"Accuracy: %.2f%%" % (accuracy * 100.0))


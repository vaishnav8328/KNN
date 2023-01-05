# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 11:19:05 2022

@author: vaishnav
"""

#importing the data
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("C:\\anaconda\\New folder (2)\\Zoo.csv")
df
df.columns


df.rename({'animal name':'animal_name'},axis=1,inplace=True)#renaming the column name
df

df.info()

df.describe()
df.isnull().sum()

df["animal_name"].value_counts()
#==============================================================================================================================
#check if there are duplicates in animal_name

duplicates = df['animal_name'].value_counts()
duplicates[duplicates > 1]


frog = df[(df["animal_name"] == 'frog')] 
frog
#======================================================================================================================
# observation: find that one frog is venomous and another one is not 
# change the venomous one into venomfrog to seperate 2 kinds of frog 
df['animal_name'][(df['venomous'] == 1 )& (df['animal_name'] == 'frog')] ='venomfrog'

#===================================================================================================================
df['venomous'].value_counts()

# finding Unique value of hair
colorlist = [("black" if i == 1 else "blue" if i == 0 else "orange" ) for i in df.hair]
uniquecolor = list(set(colorlist))
uniquecolor

#=========================================================================================================================================     
#visualization
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="hair", data=df)
plt.xlabel("Hair")
plt.ylabel("Count")
plt.show()

df.loc[:,'hair'].value_counts()


# Lets see  how many animals are domestic or not
plt.figure(figsize=(10,8));
df['domestic'].value_counts().plot(kind="pie");
plt.xlabel('Is Domestic');
plt.ylabel("Count");
plt.plot()
     
df.loc[:,'domestic'].value_counts()     

pd.crosstab(df['type'], df['domestic']).plot(kind="bar", figsize=(10, 8), title="Domestic & Non-Domestic Count");
plt.plot();
     
t1=pd.crosstab(df['type'], df['milk'])
t1.plot(kind='bar')
plt.title('animals which provide milk')
plt.show()
     
t2=pd.crosstab(df['type'], df['aquatic'])
t2.plot(kind='bar')
plt.title('animals which are aquatic')
plt.show()

sns.factorplot('type', data=df, kind="count",size = 5,aspect = 2)



df.type.unique()#unique values in type

zoo_df_temp = df.drop(['legs'], axis=1)
zoo_df_temp = zoo_df_temp.groupby(by='type').mean()
plt.rcParams['figure.figsize'] = (10,8) 
sns.heatmap(zoo_df_temp, annot=True, cmap="inferno")
ax = plt.gca()
ax.set_title("HeatMap of Features for the Classes")

#=============================================================================================================================================

#split the data into x and y

x = df.iloc[:,1:17]

y = df.iloc[:,17]

#===========================================================================================================================================
#split the data into train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

#=======================================================================================================================
#fit the model


knn = KNeighborsClassifier(n_neighbors=5,p=2) #k=5,p=2--> Eucledian distance
knn.fit(x_train,y_train)

#prediction
y_pred_train = knn.predict(x_train)
y_pred_test = knn.predict(x_test)


from sklearn.metrics import accuracy_score

ac1 = accuracy_score(y_train,y_pred_train)
print(ac1.round(2))


ac2 = accuracy_score(y_test,y_pred_test)
print(ac2.round(2))
#==============================================================================================================================================
#KNeighbour calssifier

train_accuracy=[]
test_accuracy=[]

for i in range(5,20,2):
    knn = KNeighborsClassifier(n_neighbors=i,p=2)
    knn.fit(x_train,y_train)
    y_pred_train = knn.predict(x_train)
    y_pred_test = knn.predict(x_test)
    train_accuracy.append(accuracy_score(y_train,y_pred_train).round(2))
    test_accuracy.append(accuracy_score(y_test,y_pred_test).round(2))
    

import numpy as np

np.mean(train_accuracy).round(2)
np.mean(test_accuracy).round(2)

#==========================================================================================================================================
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



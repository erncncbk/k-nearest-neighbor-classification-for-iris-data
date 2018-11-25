# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:38:59 2018

@author: Erencan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

df = df.drop(["Id"],axis=1)
#%%

x = df.iloc[:,2:4].values
y = df.iloc[:,-1:].values

#%%     Categorical Data
#   Label Encoding

from sklearn.preprocessing import LabelEncoder 
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

#%%     Train Test Split
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#%%     Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

#   Prediction

y_pred = knn.predict(x_test)

#   Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print("accuracy: " ,knn.score(x_test,y_test))

#%%
score_list = []
for each in range(1,50,2):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,50,2),score_list)
plt.xlabel("k_values")
plt.ylabel("accuracy")
plt.show()
#%%
# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1 , stop = X_set[:, 0].max()+1 , step = 0.1),
                     np.arange(start = X_set[:, 1].min()-1 , stop = X_set[:, 1].max()+1 , step = 0.1))
plt.contourf(X1, X2, knn.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.1, cmap = ListedColormap(('b', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('r', 'black','y'))(i), label = j)
         
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()

#%%     Pairplot
import seaborn as sns

sns.pairplot(df, hue = "Species", size=3, markers=["o", "s", "D"])
plt.show()

#%%     Boxplot

df.boxplot(by="Species", figsize=(15, 10))
plt.show()




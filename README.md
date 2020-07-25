# Iris-decision-tree
#Implementation of Decision tree on Iris dataset using Python 


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv(r'C:\Users\YESHPAL SINGH\Downloads\Iris.csv')
df.head(10)

df.info()
df.describe()

sns.pairplot(df, hue='Species')

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
Y = df['Species'].values

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, train_size=0.7, random_state=1)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

Y_pred = dtc.predict(X_test)
comparison = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

dtc.score(X_test, Y_test)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtc, 
                   feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],  
                   class_names=['Setosa', 'Versicolor', 'Virginica'],
                   filled=True)
fig.savefig("decistion_tree.png")

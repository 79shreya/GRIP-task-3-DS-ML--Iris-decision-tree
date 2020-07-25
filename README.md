# Iris-decision-tree
#Implementation of Decision tree on Iris dataset using Python 

### Importing libraries and reading the dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv(r'C:\Users\YESHPAL SINGH\Downloads\Iris.csv')
df.head(10)

### Viewing the column info and a summary of the data
df.info()
df.describe()

### Checking the relationship between the columns
sns.pairplot(df, hue='Species')

### Splitting the dataset
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
Y = df['Species'].values

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, train_size=0.7, random_state=1)

### Training the classsifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)

### Predicting the test set results
Y_pred = dtc.predict(X_test)
comparison = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})

### Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

### Calculating accuracy
dtc.score(X_test, Y_test)

### Visualizing the Decision Tree
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtc, 
                   feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],  
                   class_names=['Setosa', 'Versicolor', 'Virginica'],
                   filled=True)
fig.savefig("decistion_tree.png")

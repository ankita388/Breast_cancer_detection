import numpy as np
import pandas as pd 
from sklearn.datasets import load_breast_cancer 
d=load_breast_cancer()
d.data
d.feature_names
d.target
d.target_names
df = pd.DataFrame(np.c_[d.data, d.target], columns=list(d.feature_names) + ['target'])
print(df.head())
df
df.tail()
df.shape
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=2020)


print('shape of x_train=',x_train.shape) 
print('shape of y_train=',y_train.shape)
print('shape of x_test=',x_test.shape) 
print('shape of y_test=',y_test.shape)

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='gini')
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)

classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)

from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
sc.fit(x_train)
x_train_sc=sc.transform(x_train)
x_test_sc=sc.transform(x_test)

classifier_sc=DecisionTreeClassifier(criterion='gini')
classifier_sc.fit(x_train_sc,y_train)
classifier_sc.score(x_test_sc,y_test)

classifier_sc=DecisionTreeClassifier(criterion='entropy')
classifier_sc.fit(x_train_sc,y_train)
classifier_sc.score(x_test_sc,y_test)

patient1=[17.9,10.38,122.8,1001.0,.1184,.2776,.3001,.1471,.2419,.07871,1.095,.9053,8.589,
          153.4,.006399,.04904,.05373,.01587,.03003,.006193,25.38,17.33,184.6,2019.0,.1622,.6656,.7119,.2654,.4601,.1189]
patient1=np.array([patient1])
patient1

pred=classifier_sc.predict(patient1)
if pred[0]==0:
    print("cancer")
else:
    print("not cancer")

x_train    

x_train_sc
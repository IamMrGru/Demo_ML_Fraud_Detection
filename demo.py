import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
df=pd.read_csv(r'C:\Users\ANH HIEN\Downloads\1\PS_20174392719_1491204439457_log.csv')
###print(df.head())

### Let get more info about data
###df.info()

### Eliminate unecessary features or unique data
df = df.drop(['oldbalanceOrg' , 'newbalanceOrig' , 'oldbalanceDest' , 'newbalanceDest','nameOrig', 'nameDest','step'],axis=1)
###print(df.shape)

### Numerical change of categorical data
data = { "CASH_OUT":0,
            "PAYMENT":1,
            "CASH_IN":2,
            "TRANSFER":3,
            "DEBIT":4,}
df['type'] = df['type'].map(data)
###print(df.head())
cols = list(df.columns)
print(df.info())
print(df.head())

x  = df.drop('isFraud', axis='columns')  #features columns
y = df['isFraud']                        #target column
###print(df)

print('Imbalance Dataset')
print(y.value_counts())

### Data needs to be balanced

from imblearn.over_sampling import SMOTE
smt = SMOTE(sampling_strategy = 'minority')
x_smt, y_smt = smt.fit_resample(x,y)
print('Balance Dataset')
print(y_smt.value_counts())

# split train test data into 70/30.

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x_smt, y_smt, test_size=0.3, train_size=0.7,random_state=0)

# Fit training data into model

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2) ### Su dung phep Euclidean
clf.fit(x_train, y_train)
y_pred = clf.predict(x_valid)
test=pd.DataFrame({'y':y_valid,'y_preds':y_pred })
test.to_csv(r'C:\Users\ANH HIEN\Desktop\Ecom\File Name.csv',index=False, header=True)
###print(test)

### Evaluation

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
print("Accuracy of KNN Classifer:",accuracy_score(y_valid, y_pred))
print("Precision of KNN Classifer:",precision_score(y_valid, y_pred))















###print(x_train.shape)
###print(x_valid.shape)
##print(x.shape)
### Column Transformation (Categorical to Numerical)
"""from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
transformer1 = ColumnTransformer(transformers=[
    ('trs1',OneHotEncoder(),['type','amount','isFlaggedFraud'])
],remainder='passthrough')
transformer1
x = transformer1.fit_transform(x)
print(y.value_counts())
print(x)
print(y)"""

###df.isnull().sum()

###print(df.describe())

###print(df["isFraud"].value_counts())

###print(df["type"].value_counts())



"""plt.figure(figsize=(10,8));
df.type.value_counts().plot(kind="bar");
plt.xlabel('Type');
plt.ylabel("Count");
plt.show();"""

"""import seaborn as sns
plt.figure(figsize=(8,8))
sns.countplot(x="type", data=df,hue="isFraud" , palette="Set2")
plt.show()"""

"""
import seaborn as sns
plt.figure(figsize=(15,8))
sns.countplot(x="type", data=df,hue="isFraud" , palette="Set2")
plt.show()"""

"""
import seaborn as sns
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()"""

"""
df1.head()
X=df1.iloc[:,:-1]##independent/ Crucial features
y=df1.iloc[:,-1]## dependent/ Target

print(X.head())
print('Va')
print(y.head())"""









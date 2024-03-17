#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from warnings import filterwarnings
filterwarnings(action='ignore')


# In[2]:


df=pd.read_csv("iris.csv")
print(df)


# In[3]:


print(df.shape)


# In[4]:


print(df.describe())


# In[5]:


#Checking for null values
print(df.isna().sum())
print(df.describe())


# In[6]:


df.head()


# In[7]:


df.head(150)


# In[8]:


df.tail(100)


# In[12]:


n = len(df[df['Species'] == 'versicolor'])
print("No of Versicolor in Dataset:",n)


# In[13]:


n1 = len(df[df['Species'] == 'virginica'])
print("No of Virginica in Dataset:",n1)


# In[14]:


n2 = len(df[df['Species'] == 'setosa'])
print("No of Setosa in Dataset:",n2)


# In[15]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# In[17]:


#Checking for outliars
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([df['Sepal.Length']])
plt.figure(2)
plt.boxplot([df['Sepal.Width']])
plt.show()


# In[18]:


df.hist()
plt.show()


# In[19]:


df.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)


# In[20]:


df.plot(kind ='box',subplots = True, layout =(2,5),sharex = False)


# In[22]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='Petal.Length',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='Petal.Width',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='Sepal.Length',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='Sepal.Width',data=df)


# In[23]:


sns.pairplot(df,hue='Species');


# In[25]:


X = df['Sepal.Length'].values.reshape(-1,1)
print(X)


# In[29]:


plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='b')
plt.show()


# In[28]:


Y = df['Sepal.Width'].values.reshape(-1,1)
print(Y)


# In[30]:


#Correlation 
corr_mat = df.corr()
print(corr_mat)


# In[31]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[32]:


train, test = train_test_split(df, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[33]:


train_X = train[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
train_y = train.Species

test_X = test[['Sepal.Length', 'Sepal.Width', 'Petal.Length',
                 'Petal.Width']]
test_y = test.Species


# In[34]:


train_X.head()


# In[35]:


test_y.head()


# In[36]:


test_y.head()


# In[37]:


#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[38]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))


# In[39]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(train_X,train_y)

pred_y = model1.predict(test_X)

from sklearn.metrics import accuracy_score
print("Acc=",accuracy_score(test_y,pred_y))


# In[40]:


#Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(train_X,train_y)
y_pred2 = model2.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred2))


# In[41]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model4 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model4.fit(train_X,train_y)
y_pred4 = model4.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred4))


# In[42]:


#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(train_X,train_y)
y_pred3 = model3.predict(test_X)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(test_y,y_pred3))


# In[44]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [1.0,1.0,0.973,0.97,0.94]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:





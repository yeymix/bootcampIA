#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score


data = load_iris()
X, y = data.data, data.target


model = RandomForestClassifier()


k = 5

kf = KFold(n_splits=k, shuffle=True, random_state=42)


accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
precision_scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(precision_score, average='weighted'))
recall_scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(recall_score, average='weighted'))
f1_scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(f1_score, average='weighted'))

print(f"Exactitud promedio: {accuracy_scores.mean()}")
print(f"Precisión promedio: {precision_scores.mean()}")
print(f"Sensibilidad promedio: {recall_scores.mean()}")
print(f"F1 Score promedio: {f1_scores.mean()}")


# In[5]:


from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

model = RandomForestClassifier()


loo = LeaveOneOut()


precisions, recalls, f1_scores = [], [], []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

print(f"Precisión promedio: {sum(precisions) / len(precisions)}")
print(f"Sensibilidad promedio: {sum(recalls) / len(recalls)}")
print(f"F1 Score promedio: {sum(f1_scores) / len(f1_scores)}")


# In[6]:


from sklearn.model_selection import LeavePOut
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings("ignore")

p = 2


model = RandomForestClassifier()


lpo = LeavePOut(p=p)


precisions, recalls, f1_scores = [], [], []

for train_index, test_index in lpo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

print(f"Precisión promedio: {sum(precisions) / len(precisions)}")
print(f"Sensibilidad promedio: {sum(recalls) / len(recalls)}")
print(f"F1 Score promedio: {sum(f1_scores) / len(f1_scores)}")


# In[7]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings

warnings.filterwarnings("ignore")
model = RandomForestClassifier()


k = 5


skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


precisions, recalls, f1_scores = [], [], []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    precisions.append(precision_score(y_test, y_pred, average='weighted'))
    recalls.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

print(f"Precisión promedio: {sum(precisions) / len(precisions)}")
print(f"Sensibilidad promedio: {sum(recalls) / len(recalls)}")
print(f"F1 Score promedio: {sum(f1_scores) / len(f1_scores)}")


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


data = {
    'Edad': ['Menor de 30', '30-50', 'Mayor de 50', 'Menor de 30', '30-50', 'Mayor de 50', '30-50', 'Menor de 30'],
    'Ingreso': ['Bajo', 'Medio', 'Alto', 'Medio', 'Bajo', 'Bajo', 'Alto', 'Alto'],
    'Historial de Compras': ['Bajo', 'Alto', 'Bajo', 'Alto', 'Bajo', 'Alto', 'Bajo', 'Alto'],
    'Comprador': ['No', 'Sí', 'Sí', 'Sí', 'No', 'No', 'Sí', 'Sí']
}
df = pd.DataFrame(data)


df_encoded = pd.get_dummies(df, columns=['Edad', 'Ingreso', 'Historial de Compras'], drop_first=True)


X = df_encoded.drop('Comprador', axis=1)
y = df_encoded['Comprador']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=3)  
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Sí')
recall = recall_score(y_test, y_pred, pos_label='Sí')
f1 = f1_score(y_test, y_pred, pos_label='Sí')


print(f'Precisión (Accuracy): {accuracy:.2f}')
print(f'Precisión (Precision): {precision:.2f}')
print(f'Sensibilidad (Recall): {recall:.2f}')
print(f'F1-Score: {f1:.2f}')


plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Sí'], filled=True)
plt.show()



# In[ ]:





# In[ ]:





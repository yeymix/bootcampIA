#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Cargamos los datos, en este caso un conjunto de datos simulado 
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definimos la curva de aprendizaje
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=5, scoring='accuracy', 
                                                           train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Promedio de los scores de entrenamiento y validación
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    # Graficamos
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Error en Entrenamiento")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Error en Validación")
    plt.xlabel("Número de ejemplos de entrenamiento")
    plt.ylabel("Precisión")
    plt.title("Curva de Aprendizaje")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Aplicamos el modelo de Regresión Logística
modelo = LogisticRegression()
plot_learning_curve(modelo, X_train, y_train)


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Cargamos los datos, en este caso un conjunto de datos simulado
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para graficar las curvas de aprendizaje
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=5, scoring='accuracy', 
                                                           train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Promedio de los scores de entrenamiento y validación
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    # Graficamos
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Error en Entrenamiento")
   
    plt.xlabel("Número de ejemplos de entrenamiento")
    plt.ylabel("Precisión")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Aplicamos el modelo de Regresión Logística
modelo = LogisticRegression()
plot_learning_curve(modelo, X_train, y_train)


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Cargamos los datos, en este caso un conjunto de datos simulado
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Función para graficar las curvas de aprendizaje
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=5, scoring='accuracy', 
                                                           train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Promedio de los scores de entrenamiento y validación
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    # Graficamos
  
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Error en Validación")
    plt.xlabel("Número de ejemplos de entrenamiento")
    plt.ylabel("Precisión")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Aplicamos el modelo de Regresión Logística
modelo = LogisticRegression()
plot_learning_curve(modelo, X_train, y_train)


# In[ ]:





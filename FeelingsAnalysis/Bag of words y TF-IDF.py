#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importamos el CountVectorizer para Bag of Words
from sklearn.feature_extraction.text import CountVectorizer


documentos = [
    "El gato duerme en el sofá",
    "El perro también duerme en el sofá",
    "El gato y el perro no se llevan bien"
]

# Creamos el vectorizador para Bag of Words
vectorizer = CountVectorizer()

# Ajustamos el vectorizador y transformamos los documentos
X_bow = vectorizer.fit_transform(documentos)

# Mostramos el vocabulario generado
print("Vocabulario:", vectorizer.vocabulary_)

# Mostramos la representación en forma de array
print("Representación Bag of Words:")
print(X_bow.toarray())


# In[2]:


# Importamos el TfidfVectorizer para TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Creamos el vectorizador TF-IDF
tfidf_vectorizer = TfidfVectorizer()

# Ajustamos el vectorizador y transformamos los documentos
X_tfidf = tfidf_vectorizer.fit_transform(documentos)

# Mostramos el vocabulario generado
print("Vocabulario:", tfidf_vectorizer.vocabulary_)

# Mostramos la representación TF-IDF en forma de array
print("Representación TF-IDF:")
print(X_tfidf.toarray())


# In[ ]:





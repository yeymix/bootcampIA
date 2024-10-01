#!/usr/bin/env python
# coding: utf-8

# In[12]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Cargar y preparar los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # Redimensionar para incluir el canal
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Crear el modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Capa de convolución
model.add(MaxPooling2D(pool_size=(2, 2)))  # Capa de pooling
model.add(Conv2D(64, (3, 3), activation='relu'))  # Otra capa de convolución
model.add(MaxPooling2D(pool_size=(2, 2)))  # Otra capa de pooling
model.add(Flatten())  # Aplanar la salida
model.add(Dense(128, activation='relu'))  # Capa densa
model.add(Dense(10, activation='softmax'))  # Capa de salida

# 3. Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entrenar el modelo
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# 5. Evaluar el modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Exactitud en el conjunto de prueba: {test_accuracy:.2f}")



# In[9]:


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Cargar y preparar los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # Redimensionar y normalizar
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Crear el modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))  # Capa convolucional
model.add(MaxPooling2D((2, 2)))  # Capa de pooling
model.add(Conv2D(64, (3, 3), activation='relu'))  # Otra capa convolucional
model.add(MaxPooling2D((2, 2)))  # Otra capa de pooling
model.add(Flatten())  # Aplanar las salidas
model.add(Dense(64, activation='relu'))  # Capa densa
model.add(Dense(10, activation='softmax'))  # Capa de salida

# 3. Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Entrenar el modelo y guardar el historial
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# 5. Evaluar el modelo
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_accuracy:.2f}")

# 6. Visualización de imágenes del conjunto de datos
def plot_sample_images(x, y, n=10):
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.subplot(2, 5, i+1)
        plt.imshow(x[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {y[i].argmax()}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Mostrar algunas imágenes del conjunto de entrenamiento
plot_sample_images(x_train, y_train)

# 7. Graficar la precisión y pérdida
plt.figure(figsize=(12, 4))



plt.tight_layout()
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Cargar el conjunto de datos MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Seleccionar una imagen para el ejemplo (por ejemplo, la primera imagen del conjunto de entrenamiento)
image = train_images[0]
plt.imshow(image, cmap='gray')
plt.title(f'Imagen Original - Etiqueta: {train_labels[0]}')
plt.show()


# In[4]:


# Filtro de detección de bordes (Sobel en la dirección horizontal)
edge_filter = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

# Filtro de suavizado (Promedio)
smooth_filter = np.ones((3, 3)) / 9

# Filtro de detección de esquinas
corner_filter = np.array([[1, 0, -1],
                          [0, 0, 0],
                          [-1, 0, 1]])


# In[5]:


def apply_convolution(image, kernel):
    # Obtener las dimensiones de la imagen y el kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calcular las dimensiones de la imagen resultante después de la convolución
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    
    # Crear una imagen vacía para almacenar el resultado de la convolución
    convolved_image = np.zeros((output_height, output_width))
    
    # Aplicar la convolución
    for i in range(output_height):
        for j in range(output_width):
            # Extraer la región de la imagen sobre la cual aplicaremos el filtro
            image_region = image[i:i + kernel_height, j:j + kernel_width]
            
            # Realizar la operación de convolución (producto punto y suma)
            convolved_image[i, j] = np.sum(image_region * kernel)
    
    return convolved_image


# In[6]:


# Aplicar el filtro de detección de bordes
convolved_edge = apply_convolution(image, edge_filter)
plt.imshow(convolved_edge, cmap='gray')
plt.title('Imagen con Filtro de Detección de Bordes')
plt.show()

# Aplicar el filtro de suavizado
convolved_smooth = apply_convolution(image, smooth_filter)
plt.imshow(convolved_smooth, cmap='gray')
plt.title('Imagen Suavizada')
plt.show()

# Aplicar el filtro de detección de esquinas
convolved_corner = apply_convolution(image, corner_filter)
plt.imshow(convolved_corner, cmap='gray')
plt.title('Imagen con Filtro de Detección de Esquinas')
plt.show()


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Tomar una sola imagen para el ejemplo
image = x_train[0]

# Normalizar la imagen (escala de grises)
image = image / 255.0

# Mostrar la imagen original
plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Definir una función para aplicar Max Pooling
def max_pooling(image, pool_size=2):
    # Calcular las dimensiones de la nueva imagen
    new_shape = (image.shape[0] // pool_size, image.shape[1] // pool_size)
    pooled_image = np.zeros(new_shape)

    # Aplicar max pooling
    for i in range(0, image.shape[0], pool_size):
        for j in range(0, image.shape[1], pool_size):
            pooled_image[i // pool_size, j // pool_size] = np.max(image[i:i + pool_size, j:j + pool_size])

    return pooled_image

# Aplicar max pooling a la imagen
pooled_image = max_pooling(image)

# Mostrar la imagen después del pooling
plt.subplot(1, 2, 2)
plt.imshow(pooled_image, cmap='gray')
plt.title('Imagen Después de Max Pooling')
plt.axis('off')

plt.show()


# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Cargar el conjunto de datos MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Tomar una sola imagen para el ejemplo
image = x_train[0]

# Normalizar la imagen (escala de grises)
image = image / 255.0

# Mostrar la imagen original
plt.figure(figsize=(6, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

# Definir una función para aplicar Average Pooling
def average_pooling(image, pool_size=2):
    # Calcular las dimensiones de la nueva imagen
    new_shape = (image.shape[0] // pool_size, image.shape[1] // pool_size)
    pooled_image = np.zeros(new_shape)

    # Aplicar average pooling
    for i in range(0, image.shape[0], pool_size):
        for j in range(0, image.shape[1], pool_size):
            pooled_image[i // pool_size, j // pool_size] = np.mean(image[i:i + pool_size, j:j + pool_size])

    return pooled_image

# Aplicar average pooling a la imagen
pooled_image = average_pooling(image)

# Mostrar la imagen después del pooling
plt.subplot(1, 2, 2)
plt.imshow(pooled_image, cmap='gray')
plt.title('Imagen Después de Average Pooling')
plt.axis('off')

plt.show()


# In[ ]:





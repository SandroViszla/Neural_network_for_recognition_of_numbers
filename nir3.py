import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras.optimizers
import tensorflow as tf
from tensorflow import keras as k
from keras.datasets import mnist   # библиотека базы выборок Mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = k.utils.to_categorical(y_train, 10)
y_test_cat = k.utils.to_categorical(y_test, 10)

model = k.Sequential([
    k.layers.Flatten(input_shape=(28, 28, 1)),
    k.layers.Dense(512, activation='relu'),
    k.layers.Dense(10, activation='softmax')
])

print(model.summary())      # вывод структуры НС в консоль



model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train_cat, batch_size=10, epochs=10, validation_split=0.2)
print("Модель обучена")
model.evaluate(x_test, y_test_cat)
model.save('mnist.h5')
print("Модель сохранена как mnist.h5")


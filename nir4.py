from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageOps
from keras.models import load_model

model = load_model('mnist.h5')

# Загрузка изображения и изменение размера до 28x28 пикселей
image_path = "3.jpeg"  # Замените на путь к вашему изображению
image = Image.open(image_path).convert("L")
image = image.resize((28, 28))
image = ImageOps.invert(image)
image_array = np.array(image)

# Подготовка данных для передачи в модель
x = image_array.reshape(1, 28, 28) / 255.0

# Выполнение распозонвания модели
res = model.predict(x)
print(res)
print("Ваше число это: ", np.argmax(res))

plt.imshow(image_array, cmap='binary')
plt.show()

import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import os
import cv2
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#load dataset
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Shape X_train: ", X_train.shape)
print("Shape y_train: ", y_train.shape)
print("Shape X_test: ", X_test.shape)
print("Shape y_test: ", y_test.shape)

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

#build model CNN in sub data

modelStrong = Sequential(
    [
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')        
    ]
)
modelStrong.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelStrong.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=10)

modelStrong.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)

# Test on my data with strong model
images = []
predictions = []
true_labels = []
index = 0

# Thu thập tất cả ảnh và dự đoán
while os.path.exists(f'test3/{index}.png'):
    img = cv2.imread(f'test3/{index}.png', cv2.IMREAD_GRAYSCALE)
    img = np.invert(np.array(img))
    img_reshaped = img.reshape(1, 28, 28, 1)
    y_pred = modelStrong.predict(img_reshaped, verbose=0)
    
    images.append(img)
    predictions.append(np.argmax(y_pred))
    true_labels.append(index)
    index += 1

# Hiển thị tất cả ảnh cùng lúc
plt.figure(figsize=(15, 6))
for i in range(len(images)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(f'Dự đoán: {predictions[i]}\nThực tế: {true_labels[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Tính độ chính xác
count = sum(1 for i in range(len(predictions)) if predictions[i] == true_labels[i])
print(f"\nKết quả dự đoán: {predictions}")
print(f"Nhãn thực tế: {true_labels}")
print(f"Bạn đã dự đoán đúng {count} / {len(images)} chữ số")
print(f"Độ chính xác: {count/len(images)*100:.1f}%")
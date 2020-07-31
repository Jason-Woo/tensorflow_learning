import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(train_img, train_label), (test_img, test_label) = tf.keras.datasets.fashion_mnist.load_data()
'''
train_img 60,000,28,28
train_label 60,000
test_img 10,000,28,28
test_img 10,000

train_label: 0~9
'''

train_img = train_img / 255
test_img = test_img / 255

train_label_onehot = tf.keras.utils.to_categorical(train_label)
test_label_onehot = tf.keras.utils.to_categorical(test_label)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Flatten the img into vector so we can use Dense
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # Output possibility of being each categories

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy')  # Change learning rate
'''
数字编码（顺序表现）使用sparse_categorical_crossentropy
独热编码使用categorical_crossentropy
'''
history = model.fit(train_img, train_label_onehot, epochs=5, validation_data=(test_img, test_label_onehot))
model.evaluate(test_img, test_label_onehot)
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('dataset/credit.csv', header=None)
x = data.iloc[:, :-1]
y = data.iloc[:, -1].replace(-1, 0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 4)                 64        
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 20        
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 5         
=================================================================
Total params: 89
Trainable params: 89
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x, y, epochs=100)
plt.plot(history.epoch, history.history.get('loss'))
plt.plot(history.epoch, history.history.get('acc'))

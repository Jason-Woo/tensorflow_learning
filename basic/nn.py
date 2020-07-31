import tensorflow as tf
import pandas as pd

data = pd.read_csv('dataset/adver.csv')
x = data.iloc[:, 1: -1]
y = data.iloc[:, -1]

model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                             tf.keras.layers.Dense(1)
                             ])
# model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 10)                40        
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11        
=================================================================
Total params: 51
Trainable params: 51
Non-trainable params: 0
_________________________________________________________________
'''
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100)
n_y = model.predict(x)
print(n_y)
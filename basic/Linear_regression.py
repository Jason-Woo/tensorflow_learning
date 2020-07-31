# f(x)=ax+b

import tensorflow as tf
import pandas as pd

data = pd.read_csv('dataset/income.csv')
x = data.Education
y = data.Income

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))

# model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 1)                 2         
=================================================================
Total params: 2
Trainable params: 2
Non-trainable params: 0
_________________________________________________________________
'''
model.compile(optimizer='adam', loss='mse')
history = model.fit(x, y, epochs=500)
n_y = model.predict(x)
print(n_y)
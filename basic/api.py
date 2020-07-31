# 函数式api

from tensorflow import keras
(train_img, train_label), (test_img, test_label) = keras.datasets.fashion_mnist.load_data()
'''
train_img 60,000,28,28
train_label 60,000
test_img 10,000,28,28
test_img 10,000

train_label: 0~9
'''

train_img = train_img / 255
test_img = test_img / 255

input = keras.Input(shape=(28, 28))
x = keras.layers.Flatten()(input)
x = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
output = keras.layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=input, outputs=output)
# model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28)]          0         
_________________________________________________________________
flatten (Flatten)            (None, 784)               0         
_________________________________________________________________
dense (Dense)                (None, 32)                25120     
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                2112      
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650       
=================================================================
Total params: 27,882
Trainable params: 27,882
Non-trainable params: 0
_________________________________________________________________
'''
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(train_img, train_label, epochs=10)



input1 = keras.Input(shape=(28, 28))
input2 = keras.Input(shape=(28, 28))
x1 = keras.layers.Flatten()(input1)
x2 = keras.layers.Flatten()(input2)
x = keras.layers.concatenate([x1, x2])
x = keras.layers.Dense(32, activation='relu')(x)
output = keras.layers.Dense(10, activation='sigmoid')(x)
model = keras.Model(inputs=[input1, input2], outputs=output)

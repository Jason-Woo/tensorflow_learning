import tensorflow as tf


def load_MNIST():
    (train_img, train_label), (test_img, test_label) = tf.keras.datasets.mnist.load_data()
    return train_img, train_label, test_img, test_label


def Model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Reshape((28, 28, 1), input_shape=(28,28)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))  # Filter 32, Kernel_size 3x3
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


if __name__ == '__main__':
    (train_img, train_label, test_img, test_label) = load_MNIST()

    model = Model()

    model.summary()
    '''
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 26, 26, 32)        320 = (3 * 3 * 1+ 1) * 32       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
    13 = (26 - 2) / 2 + 1         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496 = (3 * 3 * 32 + 1) * 64  
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
    5 = (11 - 2) / 2 + 1 (舍弃1)        
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928 = (3 * 3 * 64 + 1) * 64    
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0
    576 = 64 * 3 * 3         
    _________________________________________________________________
    dense (Dense)                (None, 64)                36928 = (576 + 1) * 64     
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650 = (64 + 1) * 10       
    =================================================================
    Total params: 93,322
    Trainable params: 93,322
    Non-trainable params: 0
    _________________________________________________________________
    '''

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_img, train_label, batch_size=128, epochs=5)

    test_loss, test_acc = model.evaluate(test_img, test_label)

    print(test_acc)
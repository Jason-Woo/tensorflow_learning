import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(sc):
    train_data = pd.read_csv('./dataset/NSE-TATAGLOBAL.csv').iloc[:, 1:2].values
    train_data_scaled = sc.fit_transform(train_data)
    # 归一化

    train_x, train_y = [], []
    for i in range(60, 2035):
        train_x.append(train_data_scaled[i - 60:i, 0])
        train_y.append(train_data_scaled[i, 0])
    # 创建长度为60的时间切片序列

    test_data = pd.read_csv('./dataset/tatatest.csv').iloc[:, 1:2].values
    data = np.concatenate((train_data, test_data), axis=0)
    data_test = data[len(data) - len(test_data) - 60:]
    data_test_scaled = sc.transform(data_test)
    test_x = []
    for i in range(60, 76):
        test_x.append(data_test_scaled[i - 60:i, 0])
    return data, train_x, train_y, test_x


def model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((60, 1), input_shape=(60,)))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=1))
    return model


def visualization(real_price, predict_price):
    x_axis1 = [i for i in range(2051)]
    x_axis2 = [i for i in range(2035, 2051)]
    plt.plot(x_axis1, real_price, color='black', label='Real Price')
    plt.plot(x_axis2, predict_price, color='red', label='Predict Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    sc = MinMaxScaler(feature_range=(0, 1))
    data, train_x, train_y, test_x = load_data(sc)
    lstm = model()
    lstm.compile(optimizer='adam', loss='mean_squared_error')
    lstm.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=32)
    prediction = lstm.predict(np.array(test_x))
    prediction = sc.inverse_transform(prediction)
    visualization(data, prediction)





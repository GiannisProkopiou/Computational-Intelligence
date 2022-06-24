import keras
from keras.regularizers import l2
from keras import layers
import tensorflow as tf


# make function to get keras model with one hidden layer and outputs of 20 classes
def get_model(input_shape, output_shape, lr, m, L2):

    model = keras.Sequential()

    model.add(keras.Input(shape=(input_shape,)))
    model.add(layers.Dense(units=8192, activation='relu', kernel_regularizer=l2(L2)))
    model.add(layers.Dense(units=4096, activation='relu', kernel_regularizer=l2(L2)))
    model.add(layers.Dense(units=output_shape, activation='sigmoid'))

    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=m)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy', "categorical_accuracy",
                                                                      "categorical_crossentropy", "mean_squared_error"])

    print(model.summary())

    return model


def get_embedding_model(maxlen_train_embedding, input_dim, output_shape, lr, m):

    model = keras.Sequential()
    model.add(keras.Input(shape=(maxlen_train_embedding,)))
    model.add(keras.layers.Embedding(input_dim=8520, output_dim=64))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(64))
    model.add(layers.Dense(units=output_shape, activation='sigmoid'))

    opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=m)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['accuracy', "categorical_accuracy",
                                                                      "categorical_crossentropy", "mean_squared_error"])
    print(model.summary())

    return model

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label],
                                         color=color))


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    print('\n y_train: ', y_train[0])
    print(x_train.shape)
    print(type(x_train))
    print(type(y_train))

    model = keras.Sequential([Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
                              MaxPooling2D((2, 2), strides=2),
                              Conv2D(64, (3, 3), padding='same', activation='relu'),
                              MaxPooling2D((2, 2), strides=2),
                              Flatten(),
                              Dense(128, activation='relu'),
                              Dense(10, activation='softmax')
                              ])

    print(model.summary())

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=2)
    print('\nTest accuracy:', test_acc)

    probability_model = keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
    predictions = probability_model.predict(x_test)

    print('\nPredictions 1:\n', predictions[0])

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], y_test, x_test)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], y_test)
        plot_value_array(i, predictions[i], y_test)

    plt.tight_layout()
    plt.show()

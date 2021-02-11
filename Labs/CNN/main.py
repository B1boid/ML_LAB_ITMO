import tensorflow as tf
import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense

models = [
    Sequential([  # 1
        Conv2D(3, kernel_size=5),
        MaxPool2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ]),
    Sequential([  # 2
        MaxPool2D(),
        Conv2D(3, kernel_size=5),
        Flatten(),
        Dense(10, activation='softmax')
    ]),
    Sequential([  # 3
        MaxPool2D(),
        MaxPool2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ]),
    Sequential([  # 4
        Conv2D(8, kernel_size=5),
        Conv2D(16, kernel_size=5),
        Flatten(),
        Dense(10, activation='softmax')
    ]),
    Sequential([  # 5
        Conv2D(8, kernel_size=5),
        MaxPool2D(),
        Conv2D(16, kernel_size=5),
        Flatten(),
        Dense(10, activation='softmax')
    ]),
    Sequential([  # 6
        Conv2D(4, kernel_size=5),
        Conv2D(8, kernel_size=5),
        MaxPool2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ]),
    Sequential([  # 7
        Conv2D(8, kernel_size=5),
        MaxPool2D(),
        Conv2D(16, kernel_size=5),
        MaxPool2D(),
        Flatten(),
        Dense(10, activation='softmax')
    ])
]

fashion_classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def check_architecture(model, train_images, train_labels, test_images, test_labels):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    _, test_acc = model.evaluate(test_images, test_labels)
    return model, test_acc


def process_data(train_images, test_images):
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    img_h = train_images[0].shape[0]  # 28
    img_w = train_images[0].shape[1]  # 28
    train_images = train_images.reshape(train_images.shape[0], img_h, img_w, 1)  # (60000, 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], img_h, img_w, 1)  # (10000, 28, 28, 1)
    return train_images, test_images


def find_best_architecture(train_images, train_labels, test_images, test_labels):
    train_images, test_images = process_data(train_images, test_images)

    best_accuracy, best_model_ind, cnt = 0, 0, 0
    for model in models:
        cnt += 1
        _, cur_accuracy = check_architecture(model, train_images, train_labels, test_images, test_labels)
        print("Model:" + str(cnt) + "  accuracy =", cur_accuracy)
        if cur_accuracy > best_accuracy:
            best_accuracy = cur_accuracy
            best_model_ind = cnt
    print("\nBest architecture model:", best_model_ind, "  with MNIST accuracy =", best_accuracy)
    return best_model_ind


def test_data(ind, train_images, train_labels, test_images, test_labels):
    train_images, test_images = process_data(train_images, test_images)
    model, cur_accuracy = check_architecture(models[ind - 1], train_images, train_labels, test_images, test_labels)
    print("\nFashion-MNIST with Model:" + str(ind) + " accuracy =", cur_accuracy)

    full_predictions = model.predict(test_images)
    predictions = [np.argmax(full_predictions[i]) for i in range(len(full_predictions))]
    confusion_matrix = tf.math.confusion_matrix(test_labels, predictions)

    print("\nConfusion matrix")
    print(tabulate(confusion_matrix))

    images_matrix = [[[0, 0] for _ in range(10)] for _ in range(10)]
    for i in range(len(full_predictions)):
        k = test_labels[i]
        for j in range(10):
            if full_predictions[i][j] > images_matrix[k][j][0]:
                images_matrix[k][j] = [full_predictions[i][j], i]

    print("\nPrediction matrix")
    print(tabulate(images_matrix))
    show_data(images_matrix, test_images, test_labels)


def show_data(images_matrix, test_images, test_labels):
    plt.figure(figsize=(11, 11))
    for i in range(11):
        for j in range(10):
            if i == 0:
                plt.subplot(11, 10, i * 10 + j + 1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow([[1]])
                plt.xlabel(fashion_classes[j])
                continue
            plt.subplot(11, 10, (i * 10 + j + 1))
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(test_images[images_matrix[i - 1][j][1]], cmap=plt.cm.binary)
            plt.xlabel(("%.2f" % (100 * images_matrix[i - 1][j][0])) + "% "
                       + fashion_classes[test_labels[images_matrix[i - 1][j][1]]])
    plt.show()


def main():
    mnist = keras.datasets.mnist
    (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
    best_arch_ind = find_best_architecture(mnist_train_images, mnist_train_labels, mnist_test_images, mnist_test_labels)

    fashion_mnist = keras.datasets.fashion_mnist
    (fashion_train_images, fashion_train_labels), (fashion_test_images, fashion_test_labels) = fashion_mnist.load_data()
    test_data(best_arch_ind, fashion_train_images, fashion_train_labels, fashion_test_images, fashion_test_labels)


main()

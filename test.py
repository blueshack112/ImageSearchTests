from numpy import load
from PIL import Image
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

datasetPath = "Datasets/"
tempPath = datasetPath+"temp/"
print (y_train.shape)

for i in range(0, len(y_train)-1):
    print (y_train[i])


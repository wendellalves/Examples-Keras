from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical
import numpy as np
import cv2
from os import listdir
from matplotlib import pyplot as plt

class Dataset:
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.dimData = None
        self.labelsDict = {
            'if': 0,
            'direita': 1,
            'tras': 2,
            'loop': 3,
            'frente': 4,
            'fechar': 5,
            'esquerda': 6
        }

        self.trainData, self.trainLabels = self.loadDataset('train/')
        self.testData, self.testLabels = self.loadDataset('test/')

    def loadDataset(self, which_one):
        list_of_images = []
        labels = []

        for folder in listdir(self.path + which_one):
            for img in listdir(self.path + which_one + folder):
                image = cv2.imread(self.path + which_one + folder + "/" + img, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (100, 100))

                list_of_images.append(image)
                labels.append(self.labelsDict[folder])

        data = np.array(list_of_images)
        self.dimData = np.prod(data.shape[1:])

        data = data.reshape(data.shape[0], self.dimData)
        data = data.astype('float32')
        data /= 255

        labels = np.array(labels)
        labels = to_categorical(labels)

        return data, labels

if __name__ == '__main__':
    # Carrega o dataset usando a classe 
    dataset = Dataset('/home/del/Documents/testesMlp/dataset/')

    # Constroi o modelo My Little Pony
    model = Sequential()
    model.add(Dense(512, activation='tanh', input_shape=(dataset.dimData,)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.45))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.35))
    model.add(Dense(7, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    epochs = 5

    history = model.fit(
        dataset.trainData,
        dataset.trainLabels,
        batch_size = 1,
        epochs = epochs,
        verbose = 1,
        validation_data = (dataset.testData, dataset.testLabels)
    )

    model.save('mlp_{}epochs.h5'.format(epochs))

    # Plot the acc and loss curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

    plt.show()


import numpy as np 
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.datasets import mnist

# Gets the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def get_data_info():
    print('Training data shape : ', train_images.shape, train_labels.shape)

    print('Testing data shape : ', test_images.shape, test_labels.shape)

    # Find the unique numbers from the train labels
    classes = np.unique(train_labels)
    nClasses = len(classes)
    print('Total number of outputs : ', nClasses)
    print('Output classes : ', classes)

    plt.figure(figsize=[10,5])

    # Display the first image in training data
    plt.subplot(121)
    plt.imshow(train_images[0,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(train_labels[0]))

    # Display the first image in testing data
    plt.subplot(122)
    plt.imshow(test_images[0,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(test_labels[0]))

    plt.show()

# Change from matrix to array of dimension 28x28 to array of dimention 784
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

# Change to float datatype
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

# Display the change for category label using one-hot encoding
#print('Original label 0 : ', train_labels[0])
#print('After conversion to categorical ( one-hot ) : ', train_labels_one_hot[0])

# MLP model
from keras.models import Sequential
from keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(512, activation='tanh', input_shape=(dimData,)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.45))
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.4))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.35))
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adadelta',
    loss='categorical_crossentropy',
    metrics=['acc']
)

epochs = 10

print(train_data.shape)
print(type(train_data))

exit()
# Traning the model
history = model.fit(
    train_data,
    train_labels_one_hot,
    batch_size = 128,
    epochs = epochs,
    verbose = 1,
    validation_data = (test_data, test_labels_one_hot)
)

model.save('results/models/mlp_{}e.h5'.format(epochs))

# Plot the acc and loss curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)

plt.savefig('results/img/loss_{}_adadelta.png'.format(epochs))

plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

plt.savefig('results/img/acc_{}e_adadelta.png'.format(epochs))

plt.show()

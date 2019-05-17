from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import SGD, Adadelta
from keras.metrics import categorical_accuracy

# Definição do modelo
classifier = Sequential()

# Passo 1 - Convolução
# "same" results in padding the input such that the output has the same length as the original input
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 1), padding='same', activation = 'elu'))
classifier.add(Conv2D(32, (3, 3), activation='elu'))

# Passo 2 - Pooling 
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Dropout
classifier.add(Dropout(0.25))

# Segunda camada de convolução 
classifier.add(Conv2D(64, (3, 3),  padding='same', activation = 'elu'))
classifier.add(Conv2D(64, (3, 3), activation='elu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.25))

# Terceira camada de convolução
classifier.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
classifier.add(Conv2D(64, (3, 3), activation='elu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Flattening
classifier.add(Flatten())

# Camda totalmente conectada 
classifier.add(Dense(512, activation = 'tanh'))
classifier.add(Dropout(0.5))
classifier.add(Dense(128, activation = 'tanh'))
classifier.add(Dropout(0.45))
classifier.add(Dense(64, activation = 'tanh'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 7, activation = 'softmax'))

gld = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compilando a CNN
classifier.compile(
    optimizer = 'adadelta', 
    loss = 'categorical_crossentropy', 
    metrics = ['accuracy']
    #metrics=['accuracy', categorical_accuracy]
)

# 'Ajustando a CNN as imagens 
from keras.preprocessing.image import ImageDataGenerator

# Augmentation configuration
train_data = ImageDataGenerator(
    rescale = 1./255,
    #rotation_range=180.,
    #shear_range = 0.2,
    #zoom_range = 0.2,
    #horizontal_flip = True
)

test_data = ImageDataGenerator(
    rescale = 1./255
)

training_set = train_data.flow_from_directory(
    'dataset/train',
    target_size = (128, 128),
    batch_size = 1,
    class_mode = 'categorical',
    color_mode = "grayscale"
)

test_set = test_data.flow_from_directory(
    'dataset/test',
    target_size = (128, 128),
    batch_size = 1,
    class_mode = 'categorical',
    color_mode = "grayscale"
)

# Fit the model
history = classifier.fit_generator(
    training_set,
    steps_per_epoch = len(training_set), #4233
    epochs = 6,
    validation_data = test_set,
    validation_steps = len(test_set) #1060
)

# Salvar o modelo
classifier.save('gestosCNN_100e.h5')


import matplotlib.pyplot as plt

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)

plt.show()


from os import system
import numpy as np 
from keras.preprocessing import image
from keras.models import load_model 

# Carrega o classificador 
classifier = load_model('gestosCNN_100e.h5')

system('clear')

for i in range(0, 4): 
    img = 'gesto_{}.jpg'.format(i)
    
    # Carrega a imagem 
    test_image = image.load_img(
        'dataset/single_prediction/{}'.format(img),
        target_size = (128, 128),
        grayscale = True
    )

    # Transforma a imagem em um array
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)

    # Classifica 
    res = classifier.predict(test_image)
   
    print(res)

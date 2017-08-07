from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np
from PIL import Image
import h5py

model = VGG16(weights='imagenet')
# грузим предварительно обученные веса

img = image.load_img('kurich.jpeg', target_size=(224,224))
x = image.img_to_array(img)
#размерность
x = np.expand_dims(x, axis=0)
# предварительная обработка изображений
x = preprocess_input(x)

preds = model.predict(x)
print('Image Recognition Results',decode_predictions(preds,top = 5)[0])
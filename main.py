from model import unet 
from data import trainGenerator,testGenerator,saveResult
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

pretrained_weights = None
input_size = (256,256,1)
model = unet(pretrained_weights, input_size)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=10,epochs=1,callbacks=[model_checkpoint])



testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,30,verbose=1)
saveResult("data/membrane/test",results)


import pickle 
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

from keras.preprocessing import image
from PIL import Image
import numpy as np
img_width, img_height = 256, 256
test_image = image.load_img('data/membrane/test/1.jpg', target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = Image.open('data/membrane/test/1.jpg')
test_image = test_image.resize((256,256))
test_image = test_image / 255.0
test_image = test_image.reshape(256,256)
result = model.predict(test_image, batch_size=1)
print(result)
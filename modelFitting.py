import tensorflow as tf
from tensorflow import keras
from keras.applications import mobilenet_v2
from os import listdir,mkdir
from os.path import isfile, join, exists
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

## settings ----------------------------------
valid_test=0.1
batch_size = 16
lr = 1e-3
num_epochs = 100
input_shape=(500,500,3) #(width,height)
output_num=8

background_shape=(500,500) #(width,height)
plate_shape=(180,45) #(width,height)
plate_corner=np.array([[0,0],[0,plate_shape[0]],[plate_shape[1],0],[plate_shape[1],plate_shape[0]]])

outpath='dataset/syntheticData/'
modelPath='logs/'

model_path = modelPath+'model.h5'
history_path = modelPath+'history.pickle'
log_path = outpath+'log.csv'
checkpoint_path=modelPath+'weights-{epoch:03d}-{val_loss:.4f}.hdf5'
## building model ----------------------------
def defineModel():
    inputs = keras.layers.Input(input_shape)

    backbone = mobilenet_v2.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        alpha=1.0
    )

    backbone.trainable = False
 
    x = backbone.output
    x = keras.layers.Conv2D(256, kernel_size=1, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(output_num, activation="sigmoid")(x)


    model = keras.Model(inputs, x)
    return model

def parse(imgPath, corner):
    image_string = tf.io.read_file(imgPath)
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image=tf.image.resize(image,(input_shape[0],input_shape[1]))
    image=(image-127.5)/127.5
    return image,corner

def tf_dataset(images, corners, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((images, corners))
    ds = ds.map(parse).batch(batch).prefetch(10)
    return ds

def loadDataset():
    images=[]
    corners=[]

    dirs=[d for d in listdir(outpath) if ~isfile(join(outpath,d))]
    for d in range(1,31):
        imgPath=outpath+str(d)+'/image/'
        infoPath=outpath+str(d)+'/info/'
        imgs=[f for f in listdir(imgPath) if isfile(join(imgPath,f))]
        for i in range(len(imgs)):
            images.append(imgPath+imgs[i])
            name_parts=imgs[i].split('.jpg')
            corner=np.loadtxt(infoPath+name_parts[0]+'.txt')
            corner[:,0]=corner[:,0]/input_shape[0]
            corner[:,1]=corner[:,1]/input_shape[1]
            corner=np.reshape(corner,output_num)
            corners.append(corner)
    
    print(len(images))
    train_x, valid_x = train_test_split(images, test_size=0.2, random_state=42)
    train_y, valid_y = train_test_split(corners, test_size=0.2, random_state=42)

    valid_x, test_x = train_test_split(valid_x, test_size=0.5, random_state=42)
    valid_y, test_y = train_test_split(valid_y, test_size=0.5, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def train(train_x, train_y, valid_x, valid_y):
    model=defineModel()
    print(model.summary())

    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)
    optim = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss="mse",optimizer=optim)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto',
                            min_delta=0, patience=2, verbose=0, restore_best_weights=True)

    saveRes=keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    tensorboard_callback=keras.callbacks.TensorBoard(log_dir=modelPath, histogram_freq=1)

    model.fit(train_ds, epochs=num_epochs, batch_size=batch_size, validation_data=valid_ds,
    callbacks=[tensorboard_callback,early_stopping,saveRes], verbose=2)

    model.save(model_path)

# apply inverse transformation to the background image containing plate
def inverse_transformation(image,corner):
    corner=corner.astype(int)
    min_row=np.min(corner[:,0])
    min_col=np.min(corner[:,1])
    max_row=np.max(corner[:,0])
    max_col=np.max(corner[:,1])

    croped_img=image[min_row:max_row,min_col:max_col,:]
    input_pts = np.float32([[corner[0,0]-min_row,corner[0,1]-min_col], 
    [corner[1,0]-min_row,corner[1,1]-min_col],
                             [corner[2,0]-min_row,corner[2,1]-min_col]])
    output_pts = np.float32([[plate_corner[0,0],plate_corner[0,1]], 
                             [plate_corner[1,0],plate_corner[1,1]],
                             [plate_corner[2,0],plate_corner[2,1]]])
    
    inv_affine=cv2.getAffineTransform(output_pts,input_pts)
    transformed_image=cv2.warpAffine(croped_img,inv_affine,(2*croped_img.shape[1],2*croped_img.shape[0]))

    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(image)
    plt.subplot(3,1,2)
    plt.imshow(croped_img)
    plt.subplot(3,1,3)
    plt.imshow(transformed_image)
    plt.show()

def inference(test_x, test_y):
    model=keras.models.load_model(model_path)

    predictions=[]
    for i in range(len(test_x)):
        image = cv2.imread(test_x[i])
        image=cv2.resize(image,(input_shape[1],input_shape[0]))
        main_img=np.copy(image)
        
        image=image.astype(np.float32)
        image=(image-127.5)/127.5
        #image=image/255
        image=np.expand_dims(image,axis=0)
        corner=model.predict(image)
        corner=corner.reshape((int(output_num/2),2))
        corner[:,0]=corner[:,0]*input_shape[0]
        corner[:,1]=corner[:,1]*input_shape[1]
        corner=corner.astype(np.int16)
        inverse_transformation(main_img,corner)


## Fit model ----------------------------------------------------------------------------
with open(history_path, "rb") as handel:
    h = pickle.load(handel)
    plt.figure()
    plt.plot(h['history']['loss'])
    plt.plot(h['history']['val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training loss','validation loss'])
    plt.show()

(train_x, train_y),(valid_x, valid_y),(test_x, test_y)=loadDataset()

inference(test_x, test_y)








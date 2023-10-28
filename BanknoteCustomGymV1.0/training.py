#Custom Dataset
import os
#Environment Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from tensorflow.python.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
import random
from keras.optimizers import RMSprop

import cv2
import numpy as np
import matplotlib.pyplot as plt

import random

# Define a function to predict using tflite model
def predict_with_tflite(model_path, input_data):
    interpreter = tf.lite.Interpreter(model_path=model_path+".tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def Switch(val,data_type):
    if data_type == 1:
        x = np.array(val[0])
        y = np.argmax(x)
        print(y)
    if data_type == 3:
        y = np.argmax(val)
    
    if(y == 0):
        return "100THB"
    elif(y == 1):
        return "1000THB"
    elif(y == 2):
        return "20THB"
    elif(y == 3):
        return "50THB"
    elif(y == 4):
        return "500THB"
    else:return print("Fake")
    
    if data_type == 2:
        if (x[0] >= 1).all():
            return "Banknote!"
        else:
            return "Fake!"
    else:return print("fake")

def Analizer(model_path,format):
    if format == "tflite":
        tf.lite.experimental.Analyzer.analyze(model_path=model_path)
    elif format == "keras":
        model_path.summary()

def Alice(_get_training,
          _get_test,
          __color_mode,
          training_dir,
          validation_dir,
          dataset_save_path,
          tflite_saved_path,
          data_type,
          sample_datas,
          isKeras,
          getsave,
          img_Height,
          img_Width,
          batch_size,
          color_mode,
          class_mode,
          loss,
          optimizer,
          epoch):

    print(_get_training,_get_test)
    #Named Value

    print("Configuring Model Architecture...")
    train = ImageDataGenerator(rescale = 1./255)
    validation = ImageDataGenerator(rescale = 1./255)

    #Train/Validation Datasets
    train_dataset = train.flow_from_directory(
                                              training_dir,
                                              target_size=(img_Height,img_Width),
                                              batch_size= batch_size,
                                              class_mode=class_mode,
                                              color_mode=color_mode,
                                              shuffle=True,
                                              seed= 101)
    validation_dataset = validation.flow_from_directory(
                                            validation_dir,
                                            target_size=(img_Height,img_Width),
                                            batch_size= batch_size,
                                            class_mode=class_mode,
                                            color_mode=color_mode,
                                            shuffle=True,
                                            seed= 151)
    
    #Custom Architecture
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(192,kernel_size=(7,7),strides=2,activation= 'relu',
                                        input_shape = (img_Height,img_Width,3),data_format="channels_last"),
                                        tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2),
                                        #
                                        tf.keras.layers.Conv2D(256,kernel_size=(5,5),strides=1,padding="VALID",activation= 'relu'),
                                        tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2),
                                        #
                                        tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding="SAME",activation= 'relu'),
                                        tf.keras.layers.Conv2D(512,kernel_size=(3,3),padding="SAME",activation= 'relu'),
                                        tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding="SAME",activation= 'relu'),
                                        tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2),
                                        #
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(8192,activation= 'relu'),
                                        ##
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(4096,activation= 'relu'),
                                        ##
                                        tf.keras.layers.Dense(5,activation= 'softmax')])
    
    model.summary()
    print(train_dataset.class_indices)
    print("Compiling Model...")
    model.compile(loss = loss,
                  optimizer=optimizer,
                  metrics = ['accuracy'])

    if _get_training == True:
        print("Begin Fit...")
        model_fit = model.fit(train_dataset,
                              epochs=epoch,
                              validation_data=validation_dataset)

        #Save Model
        print("Saving Model...")
        model.save(dataset_save_path)
        print("Saved Model!!")
    if getsave:
        model = keras.models.load_model(dataset_save_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        # Save the model.
        with open(tflite_saved_path +'.tflite', 'wb') as f:
            f.write(tflite_model)
    
    if _get_test == True:
        if isKeras == True:
        #Load Model
            print("Loading Model...")
            model = keras.models.load_model(dataset_save_path)
            print("Loaded Model!!")
            print("Loading Sample...")
                #RGB Mode
            if __color_mode == "RGB":
                print("Comparing...")
                for i in os.listdir(sample_datas):
                    img = image.load_img(sample_datas+i,target_size=(img_Height,img_Width))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x,axis=0)
                    images = np.vstack([x])
                    print(x.shape)
                    val = model.predict(images)
                    print(val)
                    print(Switch(val,data_type))

                    plt.imshow(img)
                    plt.show()
                print(train_dataset.class_indices)
            elif __color_mode == "Grayscale":
                #Grayscale Mode
                print("Comparing...")
                for i in os.listdir(sample_datas):
                    img = image.load_img(sample_datas+i,color_mode='grayscale',target_size=(img_Height,img_Width))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x,axis=0)
                    images = np.vstack([x])
                    print(x.shape)
                    val = model.predict(images)
                    print(val)
                    print(Switch(val,data_type))

                    plt.imshow(np.squeeze(x), cmap='gray')
                    plt.show()
                print(train_dataset.class_indices)
            else:print("End")
        else: 
            print("Comparing...")
            for i in os.listdir(sample_datas):
                img = image.load_img(sample_datas+i,target_size=(img_Height,img_Width))
                x = image.img_to_array(img)
                x = np.expand_dims(x,axis=0)
                images = np.vstack([x])
                print(x.shape)
                val = predict_with_tflite(tflite_saved_path,images)
                print(val)
                print(Switch(val,data_type))
                plt.imshow(img)
                plt.show()
            print(train_dataset.class_indices)
            
        
       
    
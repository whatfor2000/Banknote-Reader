import tensorflow as tf
import os
from subprocess import call
from training import Alice,Analizer

#Alice is Training assistance Develop by Predpa
Alice(_get_training = False,#True when need to train
      _get_test = True,#False when no need to test a model
      data_type = 1, #type = 1 ;type of banknote,type = 2 ;check  is this banknote
      __color_mode ="RGB", #RGB,Grayscale
      dataset_save_dir = "Model/banknote_1",#Do not forget to change model name everytime before training to avoid replacing model
      tflite_saved_dir = "Model/banknote_1",#This is path of .tflite format to load the model up
      sample_datas = "Sample/",#This is Directory folder stored sample for examine
      isKeras= False,#False when work on tflite
      getsave=False,#as TFLITE
      #Training & Examining Config
      training_dir="datasets/training/",
      validation_dir="datasets/validation/",
      img_Height=200,
      img_Width=200,
      batch_size=3,
      class_mode='spares',
      color_mode='rgb',
      epoch=50,
      loss ='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.RMSprop(learning_rate= 1e-4),
      )

#Analize a TFlite Model
#Analizer(model_path="PATH_TO_MODEL_.tflite")

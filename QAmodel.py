import os
import string
import glob
from tensorflow.keras.applications import MobileNet
import tensorflow.keras.applications.mobilenet  
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3
from preprocessing import *
from utills import *


from tqdm import tqdm
import tensorflow.keras.preprocessing.image
import pickle
from time import time
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Embedding, 
    TimeDistributed, Dense, RepeatVector, 
    Activation, Flatten, Reshape, concatenate,  
    Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model

from tensorflow.keras.layers import add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
class Encodetext():
  def __init__(self,vocab_size,stoi,path=path,embed_dim = 200):
    self.embed_dim = embed_dim
    self.stoi = stoi
    self.vocab_size = vocab_size
    self.glove_dir = os.path.join(path,'Glove6b')
    self.embeds_index = {} 
    f = open(os.path.join(self.glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        self.embeds_index[word] = coefs

    f.close()
  def forward(self):
    
    # Get 200-dim dense vector for each of the 10000 words in out vocabulary
    embed_matrix = np.zeros((self.vocab_size, self.embed_dim))

    for word, i in self.stoi.items():
        #if i < max_words:
        embed_vector = self.embeds_index.get(word)
        if embed_vector is not None:
            # Words not found in the embed index will be all zeros
            embed_matrix[i] = embed_vector
    return self.embed_dim,embed_matrix
class Encodeimage():
    def __init__(self,path=path,USE_INCEPTION=True,w=299,h=299,df = df):
      self.path = path 
      self.df = df
      if USE_INCEPTION:
        self.encode_model = InceptionV3(weights='imagenet')
        self.encode_model = Model(self.encode_model.input, self.encode_model.layers[-2].output)
        self.WIDTH = w
        self.HEIGHT = h
        self.OUTPUT_DIM = 2048
        self.preprocess_input = \
          tensorflow.keras.applications.inception_v3.preprocess_input
      else:
        self.encode_model = MobileNet(weights='imagenet',include_top=False)
        self.WIDTH = w
        self.HEIGHT = h
        self.OUTPUT_DIM = 50176
        self.preprocess_input = tensorflow.keras.applications.mobilenet.preprocess_input
    def encodeImage(self,img):
      # Resize all images to a standard size (specified bythe image 
      # encoding network)
      img = img.resize((self.WIDTH, self.HEIGHT), Image.ANTIALIAS)
      # Convert a PIL image to a numpy array
      x = tensorflow.keras.preprocessing.image.img_to_array(img)
      # Expand to 2D array
      x = np.expand_dims(x, axis=0)
      # Perform any preprocessing needed by InceptionV3 or others
      x = self.preprocess_input(x)
      # Call InceptionV3 (or other) to extract the smaller feature set for 
      # the image.
      x = self.encode_model.predict(x) # Get the encoding vector for the image
      # Shape to correct form to be accepted by LSTM QA network.
      x = np.reshape(x, self.OUTPUT_DIM )
      return x
    def forward(self,current_path ="train",df= df):
      train_path = os.path.join(self.path,"train_pickles_files",f'train{self.OUTPUT_DIM}.pkl')
      if not os.path.exists(train_path):
        start = time()
        encoded_images = {}
        hms_obj =Utills() 
        imgs_path = os.path.join(self.path,current_path,"png")
        
        for id in tqdm(os.listdir(imgs_path)):
          image_path = os.path.join(imgs_path, id)
        
          if id in self.df.imgname.values:
            img = tensorflow.keras.preprocessing.image.load_img(image_path, \
                    target_size=(self.HEIGHT, self.WIDTH))
            
            encoded_images[id] = self.encodeImage(img)
        with open(train_path, "wb") as fp:
          pickle.dump(encoded_images, fp)
        print(f"\nGenerating training set took: {hms_obj.hms_string(time()-start)}")
      else:
        with open(train_path, "rb") as fp:
          encoded_images = pickle.load(fp)
      return encoded_images,self.OUTPUT_DIM
class QAModel():
  def __init__(self,max_l,vocab_size,embeded_dim,OUTPUT_DIM):
    self.max_l = max_l+2
    self.OUTPUT_DIM  = OUTPUT_DIM
    self.embeded_dim = embeded_dim
    self.vocab_size = vocab_size
    self.in1 = Input(shape=(self.OUTPUT_DIM,))
    self.in2 = inputs2 = Input(shape=(self.max_l,))
    self.dropout1 = Dropout(0.5)
    self.linear1 =  Dense(256, activation='relu')
    self.dropout2 = Dropout(0.5)
    self.linear2 =  Dense(256, activation='relu')
    self.lstm = LSTM(256)
    self.embed_l = Embedding(self.vocab_size,self.embeded_dim, mask_zero=True)
    self.out =  Dense(self.vocab_size, activation='softmax')
  def forward(self):
    fe1  = self.dropout1(self.in1)
    fe2 = self.linear1(fe1)
    se1 = self.embed_l(self.in2)
    se2 = self.dropout2(se1)
    se3 = self.lstm(se2)
    d1 = add([fe2,se3])
    d2 = self.linear2(d1)
    out = self.out(d2)
    model = Model(inputs=[self.in1,self.in2],outputs=out)
     # fe1 = Dropout(0.5)(inputs1)
    # fe2 = Dense(256, activation='relu')(fe1)
    # inputs2 = Input(shape=(max_length,))
    # se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    # se2 = Dropout(0.5)(se1)
    # se3 = LSTM(256)(se2)
    # decoder1 = add([fe2, se3])
    # decoder2 = Dense(256, activation='relu')(decoder1)
    # outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # QA_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

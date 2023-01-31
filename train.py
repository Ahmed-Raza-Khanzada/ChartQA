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
from QAmodel import *

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


# QA_model.layers[2].set_weights([embedding_matrix])
# QA_model.layers[2].trainable = False
# QA_model.compile(loss='categorical_crossentropy', optimizer='adam')
class Train():
    def __init__(self,epochs =7,number_pics_per_bath = 6,path = path,df = df):
      self.start = time()
      self.number_pics_per_bath = number_pics_per_bath
      self.EPOCHS = epochs
      self.path = path
      self.hmsobj  = Utills()
      self.number_pics_per_bath = number_pics_per_bath
      self.sg = Swipe_Garb(df)
      self.max_l,self.lookup,self.lex,max_id = self.sg.polish() 
      self.steps =  len(df)//number_pics_per_bath#len(self.lookup)
      self.dg = Data_Gamb()
      self.vocab,self.vocabsize,self.stoi,self.itos = self.dg.preprocess(data1=self.lookup)   
      self.ten = Encodetext(self.vocabsize,self.stoi,self.path)
      self.embed_dim,self.embed_matrix=self.ten.forward()
      self.imgen = Encodeimage(self.path,df = df)
      self.train_encodings,self.OUTPUT_DIM = self.imgen.forward() 
      self.m_obj = QAModel(self.max_l,self.vocabsize,self.embed_dim,self.OUTPUT_DIM)
      self.model = self.m_obj.forward()
      self.model.layers[2].set_weights([self.embed_matrix])
      self.model.layers[2].trainable = False
      self.model.compile(loss='categorical_crossentropy', optimizer='adam')
      print(self.model.summary())
    def train(self,trail=0,path1 = None):
      model_path =os.path.join(f".\model","model"+"e-"+str(self.EPOCHS)+"npb-"+str(self.number_pics_per_bath)+".hdf5")  if path1!=None else os.path.join(self.path,f"model",f'model.hdf5')
      if not os.path.exists(model_path):
        for i in tqdm(range(self.EPOCHS*2)):
            # datagen(self,descriptions, photos, wordtoidx, \
            #           max_length, num_photos_per_batch,vocab_size,stoi):
            # print(list(self.train_encodings.keys())[0],"=====",list(self.train_encodings.values())[0],"*******************",len(self.train_encoding))
            generator = self.dg.datagen(self.lookup, 
                                        self.train_encodings, 
                          self.stoi, self.max_l, self.number_pics_per_bath,
                          vocab_size = self.vocabsize,stoi = self.stoi)
            self.model.fit(generator, epochs=1,
                          steps_per_epoch=self.steps, verbose=2)

        # self.model.optimizer.lr = 1e-4
        # number_pics_per_bath = 6
        # steps = len(train_descriptions)//number_pics_per_bath

        # for i in range(EPOCHS):
        #     generator = self.dg.datagen(train_descriptions, encoding_train, 
        #                   wordtoidx, max_length, number_pics_per_bath)
        #     self.model.fit_generator(generator, epochs=1, 
        #                           steps_per_epoch=steps, verbose=1)  
        self.model.save_weights(model_path)
        print(f"\Training took: {self.hmsobj.hms_string(time()-self.start)}")
      else:
        plot_model(self.model,  show_shapes=True, show_layer_names=True)
        self.model.load_weights(model_path)
        return self.model
  
  
  
# if __name__ == "__main__":
#     train = Train(epochs=36,number_pics_per_bath=10)
#     model = train.train(path1=True)
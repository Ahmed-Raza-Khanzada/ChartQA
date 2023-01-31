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
from train import Train

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

def giveanswer(img_path,q):
    sg  = Swipe_Garb(df)
    # model =  model.load_weights(os.path.join(path,f"model",f'model.hdf5'))   
    train = Train(epochs=36,number_pics_per_bath=10) 
    model = train.model
    imgen = Encodeimage(path,df = df)
    img = tensorflow.keras.preprocessing.image.load_img(img_path, \
                    target_size=(299,299))
    img2 = tensorflow.keras.preprocessing.image.load_img(img_path)
    photo = imgen.encodeImage(img).reshape((1,2048))
    max_l,lookup,lex,max_id = sg.polish()
    max_l +=2 
    dg = Data_Gamb()
    vocab,vocabsize,stoi,itos = dg.preprocess(data1=lookup)  
    in_text = "sos "+q.lower()
    print("Question :",in_text[4:])
    for i in range(max_l):
        seq = [stoi[w] for w in in_text.split(" ") if w in stoi.keys()]
        seq = pad_sequences([seq], maxlen=max_l)
        # print(photo.shape,seq.shape)
        yhat = model.predict([photo,seq],verbose=0)
        yhat = np.argmax(yhat)
        # print(yhat,itos[yhat])
        word = itos[yhat]

        if in_text.split(" ")[-1] != word:
          in_text += ' ' + word
        if word == "eos":
            in_text = " ".join(in_text.split()[:-1])
            break
    final = in_text.split()[len(q.split(" "))+1:]
    final = ' '.join(final)
    p =False
    w1= ""
    for w in q.split(" "):
      if w not in stoi.keys():
        w1+=" "+w
        p =True
    
    if p: print("These words are not in my database\n",w1) 
    
    return final,img2

def images_QA(imgs,q,path = path):
  for z in range(len(imgs)): # set higher to see more examples
    ans,img = giveanswer(os.path.join(path,"train","png",imgs[z]),q[z])
    print("Answer: ",ans)
    plt.imshow(img)
    plt.show()
    print("_____________________________________")
    
    
images_QA(["74.png"],["what is percentage of first qurter"])
    
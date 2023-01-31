import os
import string
import glob
from tensorflow.keras.applications import MobileNet
import tensorflow.keras.applications.mobilenet  
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications.inception_v3 import InceptionV3
import tensorflow.keras.applications.inception_v3
import pandas as pd

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


path = "E:/figureqa/ChartQA-main/ChartQA Dataset/"
df = pd.read_json(os.path.join(os.path.join(path,"train"),"train_human.json"))


class Swipe_Garb():
  def __init__(self,df):
    self.df = df
    self.null_punct = str.maketrans('', '', string.punctuation)
    self.lookup = dict()
    self.max_l = 0
    self.id = 0
  def apply_cl_up(self,desc):
    desc = desc.split(" ")
    # desc[0],desc[-1] = START,STOP
    desc = [word.lower() for word in desc]
    desc = [w.translate(self.null_punct) for w in desc]
    return " ".join(desc)
  def polish(self):
    for pos,(id,q,desc) in enumerate(zip(self.df["imgname"].values,self.df["query"].values,self.df["label"].values)):
        q=self.apply_cl_up(q)
        desc = self.apply_cl_up(desc)
        if len(desc.split(" "))+len(q.split(" "))>self.max_l:
          self.id = id
        self.max_l = max(self.max_l,(len(desc.split(" "))+len(q.split(" "))))
        if id not in self.lookup:
          self.lookup[id] = [[],[]]
          self.lookup[id][0].append(q)
          self.lookup[id][1].append(desc)
        else:
          self.lookup[id][0].append(q)
          self.lookup[id][1].append(desc)
    lex = set()
    for key in self.lookup:
      [lex.update(d) for d in self.lookup[key]]
    return self.max_l,self.lookup,lex,self.id

class Data_Gamb():
  def __init__(self):
    self.st = "sos"
    self.sp = "eos"
  def preprocess(self,data1,w_c_thresh = 0):
    wc = {}
    nsents = 0

    for l in data1.values():
      for t in l:
        for sent in t:
            nsents += 1
            for w in sent.split(' '):
                wc[w] = wc.get(w, 0) + 1
    vocab = [w for w in wc if wc[w] >= w_c_thresh]
    vocab.append(self.st);vocab.append(self.sp); 
    itos = {}
    stoi = {}
    ix = 1
    for w in vocab:
        stoi[w] = ix
        itos[ix] = w
        ix += 1
    vocab_size = len(itos) + 1 
    # print('preprocessed words %d ==> %d' % (len(wc), len(vocab)))
    return vocab,vocab_size,stoi,itos
  def datagen(self,descriptions, photos, wordtoidx, \
                    max_length, num_photos_per_batch,vocab_size,stoi):
    x1, x2, y = [], [], []
    n=0
    while True:
      for key, desc_list in descriptions.items():
        n+=1
        photo = photos[key]
        q_l,desc_l = desc_list[0],desc_list[1]
        for q,d in zip(q_l,desc_l):
          u = q.split(" ")
          qseq = [stoi[word] for word in u \
                if word in stoi]
          dseq = [stoi[word] for word in d.split(' ') \
                if word in stoi]
          dseq.append(stoi[self.sp])
          for i in range(0, len(dseq)):
            cp_qseq = qseq.copy()
            cp_qseq.insert(0,stoi[self.st])
            cp_qseq.extend(dseq[:i])
            
            in_seq, out_seq = cp_qseq, dseq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            x1.append(photo)
            x2.append(in_seq)
            y.append(out_seq)
        if n==num_photos_per_batch:
          yield ([np.array(x1), np.array(x2)], np.array(y))
          x1, x2, y = [], [], []
          n=0
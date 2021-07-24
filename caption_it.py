
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from tensorflow.keras.preprocessing import image

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import pickle


model=ResNet50(weights='imagenet',input_shape=(224,224,3))
                                                                   # resent for image feature extraction

model_Resnet=Model(model.input,model.layers[-2].output)
model_Resnet.make_predict_function()

def img_preprocess(img_path):
  img=image.load_img(img_path,target_size=(224,224,3))
  img=image.img_to_array(img)                                 
  img=img.reshape(-1,224,224,3)
                                                     # resnet 50 preprocessing function
  img=preprocess_input(img)
  return img

def encode_img(img):
  img=img_preprocess(img)
  features=model_Resnet.predict(img)

  feature_vec=features.reshape((-1,))
  return feature_vec

with open('static/idx_word.pkl','rb') as f:
  idx_word=pickle.load(f)

with open('static/word_idx.pkl','rb') as f:
  word_idx=pickle.load(f)

from tensorflow.keras.models import load_model
model_x=load_model('static/latest.h5')
model_x.make_predict_function()                                    #importing vocab dict



def predict_caption(photo):                                          #caption from endcoded img
  photo=photo.reshape(1,-1)
  
  ans="<s>"
  max_len=16

  for i in range(max_len):
    seq=[word_idx[x] for x in ans.split() if x in word_idx]
    seq=pad_sequences([seq],maxlen=max_len,padding='post')

    y=model_x.predict([photo,seq])
    y_final=y.argmax()
    word=idx_word[y_final]

    ans+=(" "+word)
    if(word=="<e>"):
      break
  
  ans=ans.split()
  caption=ans[1:-1]
  caption=" ".join(caption)
  
  return caption

def caption_this(img):                        # final prediction function
  
  img=encode_img(img)
  caption=predict_caption(img)
  print(caption)
  return caption






def readTextfile(path):
  with open(path) as file:
    captions=file.read()
  return captions

captions=readTextfile('captions.txt')

caption_list=captions.split('\n')
caption_list=caption_list[:-1]
caption_list=caption_list[1:]

descriptions={}

for x in caption_list:
  first,second= x.split('.jpg,')
  img_name=first

  if descriptions.get(img_name) is None:
    descriptions[img_name]=[]
  
  descriptions[img_name].append(second)

import re

def clean_text(sentence):                           #not doing stemming lemmanization and stopword removal for human like output...
  sentence=sentence.lower()
  sentence=re.sub("[^a-z]"," ",sentence)
  sentence=sentence.split()

  sentence=" ".join(sentence)

  return sentence

for key,cap in descriptions.items():
  for i in range(len(cap)):
    cap[i]=clean_text(cap[i])

with open("drive/MyDrive/descriptions.txt",'w') as f:                         #saving in descriptions.txt
  f.write(str(descriptions))


import json

descriptions=None

with open('/content/drive/MyDrive/descriptions.txt','r') as f:
  descriptions=f.read()

json_acceptable_string=descriptions.replace("'","\"")
descriptions=json.loads(json_acceptable_string)

vocab=set()                                                      #vocab created

for cap in descriptions.values():
  for sentence in cap:
    vocab.update(sentence.split())

import collections

total_words=[]
for key in descriptions.keys():
  for s in descriptions[key]:
    for word in s.split():
      total_words.append(word)


counter=collections.Counter(total_words)
freq_count=dict(counter)

sorted_freq=sorted(freq_count.items(),reverse=True,key=lambda x:x[1])

threshold=10                # min frequency for word's consideration

freq=[x for x in sorted_freq if x[1]>threshold ]

freq_vocab=[x[0] for x in freq]

import pickle

f=open('/content/drive/MyDrive/encoded_train.pkl','rb')
train=pickle.load(f)
f.close()

train_descriptions={}
test=[]
                                             #for reloading the train/test set....not to be done multiple times

for key in descriptions.keys():

  if train.get(key) is not None:
    train_descriptions[key]=[]
    for sentence in descriptions[key]:
       train_descriptions[key].append("<s> "+sentence+" <e>")
  
  else:
    test.append(key)

"""###  Train/test
  



"""

#run ONLY ONCE

train_descriptions={}

test=[]
                                                                  # random train/test split (85/15)
import random

for key in descriptions.keys():
  choice=random.random()
  if choice >=0.40:
    train_descriptions[key]=[]

    for sentence in descriptions[key]:
      train_descriptions[key].append("<s> "+sentence+" <e>")
  
  else:
    test.append(key)

test=test[:1000] 
                                  # 1000 testing images

"""*image preprocessing *




"""

from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import Model

model=ResNet50(weights='imagenet',input_shape=(224,224,3))

model.summary()

model_new=Model(model.input,model.layers[-2].output)

from keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
                                                                                     
def img_preprocess(img_path):
  img=image.load_img(img_path,target_size=(224,224,3))
  img=image.img_to_array(img)                                 
  img=img.reshape(-1,224,224,3)
# resnet 50 preprocessing function
  img=preprocess_input(img)
  return img

def encode_img(img):
  img=img_preprocess(img)
  features=model_new.predict(img)

  feature_vec=features.reshape((-1,))
  return feature_vec

train_encoded={}
IMG_PATH="Images"
c=0
for key in train_descriptions.keys():
  img_data=encode_img(IMG_PATH+"/"+key+".jpg")

  train_encoded[key]=img_data
  if c<100:
    c+=1
  else:
    print("one batch done")
    c=0

import pickle

with open("drive/MyDrive/encoded_train.pkl",'wb') as f:
  pickle.dump(train_encoded,f)



test_encoded={}
IMG_PATH="Images"
c=0
for t in test:
  img_data=encode_img(IMG_PATH+"/"+t+".jpg")
  test_encoded[t]=img_data

  if c%100==0:
    print(c/100)
  c+=1

with open("drive/MyDrive/encoded_test.pkl",'wb') as f:
  pickle.dump(test_encoded,f)



"""# text preprocessing

"""

print(len(freq_vocab))

vocab=[]
vocab=freq_vocab

word_idx={}
idx_word={}

for ix,word in enumerate(vocab):
  word_idx[vocab[ix]]=ix+1
  idx_word[ix+1]=vocab[ix]

print(word_idx['cannon'])

idx_word[1851]="<s>"
word_idx["<s>"]=1851
                                   #adding start seq and end seq to vocabulary
idx_word[1852]="<e>"
word_idx["<e>"]=1852

vocab_size=len(word_idx)+1

"""# data generator

"""

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np


def data_generator(train_descriptions,encoded_train,word_idx,maxlen,batch_size):
  x1,x2,y= [],[],[]

  n=0
  while True:
    
    for key,data in train_descriptions.items():
      n+=1

      photo=encoded_train[key]
      for desc in data:
        seq=[word_idx[x] for x in desc.split() if x in word_idx]
        for i in range(1,len(seq)):
          xi=seq[0:i]                   #generating and padding
          yi=seq[i]
          
          xi=pad_sequences([xi],maxlen=maxlen,value=0,padding="post")[0]

          yi=to_categorical([yi],num_classes=len(word_idx)+1)[0]
      
          x1.append(photo)
          x2.append(xi)
          y.append(yi)
          

        if n==batch_size:
        
          yield([np.array(x1),np.array(x2)],np.array(y))

          n=0
          x1,x2,y= [],[],[]

"""# Glove embedding

"""

import numpy as np


  f=open('/content/drive/MyDrive/glove.6B.50d.txt','r',encoding='utf-8')

  word_embedding={}

  lines=f.readlines()
  f.close()

  for line in lines:
    line=line.split()
    word=line[0]
    word_embedding[word]=np.array(line[1:],dtype='float')

def form_embeddding_matrix(word_idx):
  embedd_dim=50
  matrix=np.zeros((vocab_size,embedd_dim))

  for w,i in word_idx.items():
    embedding_values=word_embedding.get(w)
    if embedding_values is not None:
      matrix[i]=embedding_values

  return matrix

embedding_matrix=form_embeddding_matrix(word_idx)

"""Image caption model"""

import tensorflow.keras

max_len=35

from tensorflow.keras.layers import *
from tensorflow.keras import Input

# img features input
input_img_features=Input(shape=(2048,))
inp_img1=Dropout(0.3)(input_img_features)
inp_img2=Dense(256,activation='relu')(inp_img1)


# caption input
input_captions=Input(shape=(max_len,))
inp_cap1=Embedding(input_dim=vocab_size,output_dim=50,mask_zero=True)(input_captions)
inp_cap2=Dropout(0.3)(inp_cap1)
inp_cap3=LSTM(256)(inp_cap2)

print(inp_img1.shape)
print(input_img_features.shape)

#decoder

decoder1=add([inp_img2,inp_cap3])
decoder2=Dense(256,activation='relu')(decoder1)
outputs=Dense(vocab_size,activation='softmax')(decoder2)

from tensorflow.keras.models import Model

model_x=Model(inputs=[input_img_features,input_captions],outputs=outputs)
model_x.summary()

#setting embedding weights

model_x.layers[2].set_weights([embedding_matrix])
model_x.layers[2].trainable=False

model_x.compile(optimizer='adam',loss='categorical_crossentropy')

"""Train>>>...

"""

def train_model(epochs,batch_size):
  
  steps=len(train)//batch_size
  for i in range(epochs):
    generator=data_generator(train_descriptions,train,word_idx,35,3)
    model_x.fit_generator(generator,epochs=1,verbose=1,steps_per_epoch=steps)
    print('epoch complete')
    if i%5==0:
      model_x.save("drive/MyDrive/imgcap_"+str(i)+".h5")

train_model(11,3)

from keras.models import load_model
model_x=load_model('/content/drive/MyDrive/imgcap_5.h5')

def predict_caption(photo):
  ans="<s>"

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

#testing a bit
photo_list=list(train.keys())
import matplotlib.pyplot as plt


for i in range(6):
  num=np.random.randint(0,1000)
  img_num=photo_list[num]

  img_encoding=train[img_num].reshape(1,2048)
  cap=predict_caption(img_encoding)

  image=plt.imread("/content/Images/"+img_num+".jpg")
  plt.imshow(image)
  plt.axis("off")
  plt.show()
  print(cap)
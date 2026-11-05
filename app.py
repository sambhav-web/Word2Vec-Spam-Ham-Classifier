import streamlit as st
import numpy as np
import joblib
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

def remove_unwanted(col):
  col=col.lower()
  col=re.sub('<.*?>',' ',col)
  col=re.sub('http?://\S+|www\.\S+',' ',col)
  return re.sub('[^a-zA-Z]',' ',col)

def remove_stopwords(col):
  L=[]
  words=stopwords.words('english')
  negations=['not','never','isnt','wouldnt','wasnt','havent','arent']
  tokens=word_tokenize(col)
  for token in tokens:
    if token not in words or token in negations:
      L.append(token)
  return " ".join(L)

def lemmatize(col):
  lemmatizer=WordNetLemmatizer()
  tokens=word_tokenize(col)
  L=[lemmatizer.lemmatize(token) for token in tokens]
  return L

with open('preprocessor.pkl','rb') as f:
  processor=joblib.load(f)
with open('model.pkl','rb') as f:
  model=joblib.load(f)
def AvgW2V(col):
  return np.mean([processor.wv[token] for token in col if token in processor.wv.index_to_key],axis=0)

def preprocess(msg:str):
  message=remove_unwanted(msg)
  message=remove_stopwords(message)
  message=lemmatize(message)
  message=AvgW2V(message)
  df=pd.DataFrame([message])
  return model.predict(df)

st.title('SPAM HAM CLASSIFIER')
message=st.text_input('Enter your message')
if st.button('Detect'):
  container=st.container(border=True)
  if preprocess(message)[0]==0:
    container.header('SPAM')
  else:
    container.header('HAM')
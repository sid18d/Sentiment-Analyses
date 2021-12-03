#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torch torchvision torchaudio')
# Installing Pytorch


# In[2]:


get_ipython().system('pip install transformers requests beautifulsoup4 pandas numpy')
# Installing transformers


# In[3]:


from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
# Import required packages


# In[4]:


tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
# Setting up the Model to calculating Sentiment


# In[5]:


tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')
# Passing a token, a string in our case


# In[6]:


result = model(tokens)


# In[7]:


result.logits
# Loading Sentiment Score


# In[8]:


int(torch.argmax(result.logits))+1


# In[14]:


r = requests.get('https://www.yelp.com/biz/social-brew-cafe-pyrmont')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class':regex})     
reviews = [result.text for result in results]
# Extracting text from Reviews from the website using BeautifulSoup


# In[17]:


reviews
# Loading the Reviews


# In[18]:


import numpy as np
import pandas as pd


# In[19]:


df = pd.DataFrame(np.array(reviews), columns=['review'])
#


# In[20]:


df['review'].iloc[0]


# In[21]:


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1


# In[22]:


sentiment_score(df['review'].iloc[1])


# In[23]:


df['sentiment'] = df['review'].apply(lambda x: sentiment_score(x[:512]))


# In[24]:


df


# In[25]:


df['review'].iloc[3]


# In[ ]:





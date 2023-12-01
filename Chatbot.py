#!/usr/bin/env python
# coding: utf-8

# In[19]:


import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# In[20]:


import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True)


# In[21]:


f=open("C:/Users/praku/OneDrive/Desktop/chatbot.txt",'r',errors = 'ignore')
raw=f.read()
raw = raw.lower()


# In[22]:


sent_tokens = nltk.sent_tokenize(raw) 
word_tokens = nltk.word_tokenize(raw)


# In[23]:


lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[24]:


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[25]:


def response(user_response):
    Praky_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        Praky_response=Praky_response+"I am sorry! I don't understand you"
        return Praky_response
    else:
        Praky_response = Praky_response+sent_tokens[idx]
        return Praky_response


# In[26]:


flag=True
print("Praky: My name is Praky. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Praky: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("Praky: "+greeting(user_response))
            else:
                print("Praky: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Praky: Bye! take care..")

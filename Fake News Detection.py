#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
text = ['Hello Rupali here, I love machine learning','Welcome to the Machine learning hub' ]


# In[3]:


vect = TfidfVectorizer()


# In[4]:


vect.fit(text)


# In[5]:


print(vect.idf_)


# In[6]:


print(vect.vocabulary_)


# In[7]:


example = text[0]
example


# In[8]:


example = vect.transform([example])
print(example.toarray())


# ## PassiveAggressiveClassifier

# In[9]:


import os
import pandas as pd


# In[10]:


dataframe = pd.read_csv('news.csv')
dataframe.head()


# In[11]:


x = dataframe['text']
y = dataframe['label']


# In[12]:


x


# In[13]:


y


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[15]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
y_train


# In[16]:


y_train


# In[17]:


tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)


# In[18]:


classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train,y_train)


# In[19]:


y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[20]:


cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(cf)


# In[21]:


def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)


# In[23]:


fake_news_det('U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.')


# In[25]:


fake_news_det("""Go to Article 
President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  """)


# In[26]:


import pickle
pickle.dump(classifier,open('model.pkl', 'wb'))


# In[27]:


# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))


# In[28]:


def fake_news_det1(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    print(prediction)


# In[29]:


fake_news_det1("""U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.""")


# In[30]:


fake_news_det1("""Go to Article 
President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  """)


# In[ ]:





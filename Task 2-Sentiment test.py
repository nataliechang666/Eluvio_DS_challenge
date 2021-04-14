#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import os
import seaborn as sb
import seaborn as sns
import matplotlib.pyplot as plt

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
import string


import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder,PowerTransformer

import transformers
import tensorflow as tf
import torch
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures


# # 1.Data Processing and Analysis

# In[3]:


data = pd.read_csv("/Users/nataliechang/Desktop/Eluvio_DS_Challenge.csv")


# In[4]:


data = data.drop(columns=['down_votes','category'])

class_mapping = {label: idx for idx, label in enumerate(np.unique(data['over_18']))}
data['over_18'] = data['over_18'].map(class_mapping)
class_mapping2 = {label: idx for idx, label in enumerate(np.unique(data['author']))}
data['author'] = data['author'].map(class_mapping2)

data['date_created'] = data['date_created'].map(lambda x: x.replace('-',''))
data.head()


# #  Top 10 title Tokenized

# In[16]:


stopwords = set(stopwords.words('english'))
top_titles = data.sort_values(by='up_votes', ascending=False)['title'].values[:10]
top_words = set(np.concatenate([word_tokenize(t) for t in top_titles])) - stopwords
top_words = {word.lower() for word in top_words}
top_words = top_words - set(string.punctuation) - set(string.digits)


# # Add Tokenized data

# In[17]:


words_tokenized = [[w.lower() for w in word_tokenize(t)] for t in data['title']]
data['tokenized_title'] = words_tokenized
all_words = pd.Series(np.concatenate(words_tokenized)).value_counts()


# In[18]:


all_words = all_words[[word not in stopwords for word in all_words.index]]
all_words = all_words[[word not in string.punctuation for word in all_words.index]]
all_words = all_words[[word not in string.digits for word in all_words.index]]


# In[19]:


stemmer = PorterStemmer()
all_words.index = [stemmer.stem(w) for w in all_words.index]


# In[20]:


all_words[:1500].plot()


# In[21]:


all_stems = all_words.groupby(by=all_words.index).sum().sort_values(ascending=False)
data['stemmed_title'] = data['tokenized_title'].map(lambda wl: [stemmer.stem(w) for w in wl])


# In[22]:


pos_tag(data['tokenized_title'][0])
data['pos_title'] = data['tokenized_title'].map(lambda t: [t[1] for t in pos_tag(t)])
data['pos_title'].head()


# In[23]:


def noun(df):
    tags = df['pos_title']
    stems = df['stemmed_title']
    return [s for (t, s) in zip(tags, stems) if t in ['NN', 'NNS']]


# In[24]:


data['main_title'] = data.apply(noun, axis='columns')


# In[25]:


title_counts = pd.Series(np.concatenate(data['main_title'].values)).value_counts()
title_counts[1:].head(10)


# In[26]:


title_counts[1:].head(20).plot(kind='bar')


# In[27]:


title_list = data['title']
sa = SentimentIntensityAnalyzer()
sentiment = [sa.polarity_scores(t)['compound'] for t in title_list]


# In[28]:


sentiment = np.asarray(sentiment)

for i in range(sentiment.size):
    if (sentiment[i] >= 0):
        sentiment[i] = 1 
    else:
        sentiment[i] = 0 

data['sentiment'] = sentiment


# In[29]:


data.head()


# In[30]:


from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(data['tokenized_title'])
sequences = tokenizer_obj.texts_to_sequences(data['tokenized_title'])
word_index = tokenizer_obj.word_index

#pad sequences
max_length = max([len(t.split()) for t in data['title']])
title_pad = pad_sequences(sequences, maxlen = max_length)

print("Shape of title tensor",title_pad.shape)
print("Shape of sentiment tensor",data['sentiment'].shape)


# # Bert PreTrain Model

# In[13]:


model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# In[15]:


model.summary()


# In[44]:


data2 = data.drop(columns=['time_created','date_created','up_votes','over_18','author','tokenized_title','stemmed_title','pos_title','main_title'])


# In[62]:


data2


# In[47]:


def convert_data_to_examples(train, DATA_COLUMN, LABEL_COLUMN): 
    train_InputExamples = train.apply(lambda x: InputExample(guid=None,
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
    return train_InputExamples

train_InputExamples = convert_data_to_examples(data2, 'title', 'sentiment')


# In[58]:


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )
        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


# In[59]:


train_InputExamples= convert_data_to_examples(data2, 'title', 'sentiment')

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)


# In[60]:


train_data


# In[ ]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data,epochs=2)


# # Predict sentiment

# In[19]:


X_train = title_pad
indices = np.arange(title_pad.shape[0])
y_train = data['sentiment'][indices]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# # LogisticRegression

# In[20]:


LR = LogisticRegression(C=1.0, tol=0.01)
LR.fit(X_train, y_train)


# In[21]:


y_predict = LR.predict(X_val)
LR.score(X_val, y_val)


# In[22]:


print(classification_report(y_val, y_predict))


# # GBDT

# In[23]:


gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)


# In[24]:


y_predict = gbdt.predict(X_val)
gbdt.score(X_val, y_val)


# In[25]:


print(classification_report(y_val, y_predict))


# # Multi-layer Neural network

# In[26]:


mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
mlp.fit(X_train, y_train)


# In[27]:


y_predict = mlp.predict(X_val)
mlp.score(X_val, y_val)


# In[28]:


print(classification_report(y_val, y_predict))


# # LightGBM

# In[29]:


from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
from lightgbm import LGBMClassifier
lgb_classifier = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=500, learning_rate=0.001, 
                                    max_depth=800, feature_fraction=0.8, subsample=0.2,
                                    is_unbalance=True)
lgb_classifier.fit(X_train,y_train)
y_head = lgb_classifier.predict(X_val)


# In[31]:


y_head = y_head.tolist()


# In[32]:


print(classification_report(y_head, y_val))


# In[ ]:





# In[ ]:





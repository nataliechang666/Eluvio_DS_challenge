#!/usr/bin/env python
# coding: utf-8

# In[69]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, KFold,GridSearchCV
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler,RobustScaler,LabelEncoder,PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_fscore_support


# # 1.Data Processing and Analysis

# In[89]:


data = pd.read_csv("/Users/nataliechang/Desktop/Eluvio_DS_Challenge.csv")


# In[90]:


data = data.drop(columns=['down_votes','category'])
data.head()


# In[91]:


data['date_created'] = data['date_created'].map(lambda x: x.replace('-',''))


# In[92]:


set(data['over_18'])


# In[93]:


data['over_18'].value_counts()


# In[94]:


plt.bar(data['over_18'].unique(),
        data['over_18'].value_counts(), 
        width=0.5, 
        bottom=None, 
        align='center', 
        color=['lightsteelblue',  
               'olive'])
plt.xticks(rotation='vertical')
plt.show()


# In[95]:


count_Class=pd.value_counts(data['over_18'], sort= True)
count_Class.plot(kind = 'pie',autopct='%1.4f%%')
plt.title('Pie chart')
plt.ylabel('')
plt.show()


# In[96]:


from collections import Counter
word_counts = Counter(list(data['author']))

top_three = word_counts.most_common(50)
print(top_three)


# Label encoding - author

# In[97]:


from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# creating instance of labelencoder
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column
data['author'] = labelencoder.fit_transform(data['author'])
data['over_18'] = labelencoder.fit_transform(data['over_18'])
data


# In[98]:


from collections import Counter
word_counts = Counter(list(data['author']))

top_three = word_counts.most_common(50)
print(top_three)


# In[99]:


data[data['author']==44158]


# In[100]:


data[data['over_18']==1]


# In[101]:


plt.scatter(data['up_votes'],data['over_18'])


# In[102]:


data_upvote = data.groupby(['up_votes'],as_index=False)['up_votes'].agg({'cnt':'count'})


# In[103]:


plt.hist(data_upvote['up_votes'],bins=10)


# In[85]:


def binned(data,col,n):
    data[col] = pd.qcut(data[col], n, labels=[0,1,2,3,4], duplicates='drop')
    return data


# In[86]:


binned(data,'up_votes',5)


# In[87]:


data[data['up_votes']==3]


# # 2.Text processing - Word2Vec

# In[104]:


from gensim.models import Word2Vec
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


# In[105]:


def cleanText(data):
    stemmer = SnowballStemmer("english")
    text = data['title']
    text2 = []
    for i in range(len(data)):
        tokens = word_tokenize(text[i])
        # stemming of words
        porter = PorterStemmer()
        stemmed = [porter.stem(word) for word in tokens]
        text2.append(stemmed)
    return text2


# In[106]:


def build_sentence_vector(sentence,size,w2v_model):
    sen_vec = np.random.uniform(0,1/size,size).reshape((1,size))
    count = 0
    for word in sentence:
        try:
            sen_vec += w2v_model[word].reshape((1,size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        sen_vec /= count
    return sen_vec


# In[107]:


def new_dataset(data):
    dataset = cleanText(data)
    n_dim = 50
    w2v_model = Word2Vec(dataset, sg=1, size=n_dim, min_count=10, hs=0)
    w2v_model.save('w2v_model')

    data_list = []
    for i in range(len(data)):
        data_list.append(str(dataset[i]))
    docvec_list = np.concatenate([build_sentence_vector(sen,50,w2v_model) for sen in data_list])
    
    #class_mapping = {label: idx for idx, label in enumerate(np.unique(data['issue']))}
    #data['issue'] = data['issue'].map(class_mapping)
    #class_mapping2 = {label: idx for idx, label in enumerate(np.unique(train_data['author']))}
    #data['author'] = data['author'].map(class_mapping2)
    
    docvec_list = pd.DataFrame(docvec_list)
    #train_docvec_list
    new_data = data.iloc[:,[0,1,2,4,5]]
    #new_train_data
    new_data2 = pd.concat([new_data,docvec_list], axis=1)
    return new_data2


# In[108]:


data2 = new_dataset(data)


# In[109]:


data2


# In[26]:


data2.to_csv("word2vec_data.csv",index=False)


# # Model1 - predict up_vote 5 classes

# In[27]:


import torch.nn as nn
import torch


# In[28]:


X_train = np.array(data2.drop(columns=['up_votes']))
y = np.array(data2['up_votes'])

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(-1, X_train.shape[1]).astype('float32')
X_val = X_val.reshape(-1, X_val.shape[1]).astype('float32')

X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val)


# In[29]:


from torch.utils.data import Dataset, DataLoader
class Data(Dataset):
    def __init__(self):
        self.x=torch.from_numpy(X_train)
        self.y=torch.from_numpy(y_train)
        self.len=self.x.shape[0]
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len


# In[30]:


data_set = Data()
trainloader = DataLoader(dataset=data_set,batch_size=64)
data_set.x[1:2]


# In[31]:


data_set.y[1:10]


# In[32]:


data_set.x.shape, data_set.y.shape


# In[33]:


class Net(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(Net,self).__init__()
        self.linear1=nn.Linear(D_in,H)
        self.linear2=nn.Linear(H,D_out)

        
    def forward(self,x):
        x=torch.sigmoid(self.linear1(x))  
        x=self.linear2(x)
        return x


# In[34]:


input_dim = 54     # number of Variables
hidden_dim = 25    # hidden layers
output_dim = 5     # number of classes
input_dim


# In[35]:


model = Net(input_dim,hidden_dim,output_dim)


# In[36]:


print('W:',list(model.parameters())[0].size())
print('b',list(model.parameters())[1].size())


# In[37]:


criterion = nn.CrossEntropyLoss()


# In[38]:


learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[39]:


n_epochs = 1000
loss_list = []

for epoch in range(n_epochs):
    for x, y in trainloader:
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z,y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.data) 
        #print('epoch {}, loss {}'.format(epoch, loss.item()))


# In[41]:


z = model(X_val)


# In[42]:


yhat = torch.argmax(z.data,1)
yhat.tolist()
y_val.tolist()


# In[43]:


print(f1_score(yhat,y_val,average='weighted'))
print(precision_score(yhat,y_val,average='weighted'))
print(recall_score(yhat,y_val,average='weighted'))


# # Model2 - predict up_vote 2 classes

# In[111]:


X_train = np.array(data2.drop(columns=['up_votes']))

thre = 100 #np.quantile(data2['up_votes'], 0.8)
y = [1 if i > thre else 0 for i in data['up_votes']]

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=42)


# # LogisticRegression

# In[112]:


LR = LogisticRegression(C=1.0, tol=0.01)
LR.fit(X_train, y_train)


# In[113]:


y_predict = LR.predict(X_val)
LR.score(X_val, y_val)


# In[114]:


print(classification_report(y_val, y_predict))


# # GBDT

# In[115]:


gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)


# In[116]:


y_predict = gbdt.predict(X_val)
gbdt.score(X_val, y_val)


# In[117]:


print(classification_report(y_val, y_predict))


# # Multi-layer Neural network

# In[118]:


mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=1)
mlp.fit(X_train, y_train)


# In[119]:


y_predict = mlp.predict(X_val)
mlp.score(X_val, y_val)


# In[120]:


print(classification_report(y_val, y_predict))


# # LightGBM

# In[121]:


from sklearn.kernel_ridge import KernelRidge
import lightgbm as lgb
from lightgbm import LGBMClassifier
lgb_classifier = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=500, learning_rate=0.001, 
                                    max_depth=800, feature_fraction=0.8, subsample=0.2,
                                    is_unbalance=True)
lgb_classifier.fit(X_train,y_train)
y_head = lgb_classifier.predict(X_val)


# In[122]:


y_head = y_head.tolist()


# In[123]:


print(classification_report(y_head, y_val))


# In[ ]:





# In[ ]:





# In[ ]:




